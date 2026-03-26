"""
position_monitor.py — Enforces hold period rules and intraday exit discipline.

Runs on every scheduler cycle to audit all open positions against their
time-based exit constraints. Two exit mechanisms are managed here:

1. Hold Period Expiry — any position that has been held longer than its
   max_hold_days budget (set at entry time by position_sizer.py) is closed
   regardless of P&L. This prevents trades from drifting outside their
   intended risk window.

2. Intraday Forced Close — all positions classified as 'intraday' are closed
   before market close (3:45 PM cycle) to eliminate overnight gap risk.

Note: Stop-loss and take-profit exits are handled by Alpaca bracket orders
placed at entry time via trade_executor.py — not by this module.

Usage:
    from position_monitor import PositionMonitor
    monitor = PositionMonitor(trade_executor=executor)
    monitor.check_all_positions()
"""

from database import Database
from config import config, HoldPeriod
from datetime import datetime, timedelta
from logger import log_error
from typing import Optional
import yfinance as yf


class PositionMonitor:
    """
    Audits open positions on every cycle and triggers time-based exits.
    Requires a live TradeExecutor instance to place closing orders.
    """

    def __init__(self, trade_executor):
        self.db = Database()
        # Executor is injected rather than instantiated here to share the
        # same authenticated Alpaca client used by the rest of the crew
        self.executor = trade_executor

    # ── Public Interface ──────────────────────────────────────────────────────

    def check_all_positions(self):
        """
        Retrieve all open trades from the database and evaluate each one
        against its hold period constraint. Called at the start of every
        scheduler cycle before new signals are analysed.
        """
        print('[check_all_positions] Starting position audit...')
        open_trades = self.db.get_open_trades()
        print(f'[check_all_positions] {len(open_trades)} open trade(s) found in DB')

        for trade in open_trades:
            print(f'[check_all_positions] Checking hold expiry for {trade["ticker"]} (trade_id={trade["trade_id"]}, entry={trade.get("entry_time")})')
            self._check_hold_expiry(trade)

        print('[check_all_positions] Hold expiry checks complete — running Alpaca sync...')
        self.sync_closed_positions()
        print('[check_all_positions] Position audit complete.')

    # ── Alpaca Sync ───────────────────────────────────────────────────────────

    def sync_closed_positions(self):
        """
        Detect positions that were closed in Alpaca (via bracket stop-loss or
        take-profit orders) and mark the corresponding DB trades as closed.

        position_monitor only handles time-based exits directly. Bracket order
        fills happen asynchronously on Alpaca's side — this method reconciles
        the gap by comparing open DB trades against live Alpaca positions.
        """
        print('[sync_closed_positions] Starting check for bracket-closed positions...')
        try:
            open_trades = self.db.get_open_trades()
            print(f'[sync_closed_positions] Found {len(open_trades)} open trade(s) in DB')

            if not open_trades:
                print('[sync_closed_positions] No open trades to check — skipping Alpaca call')
                return

            alpaca_positions = self.executor.get_open_positions()
            alpaca_tickers = {p['ticker'] for p in alpaca_positions}
            print(f'[sync_closed_positions] Alpaca currently holds {len(alpaca_positions)} open position(s): {sorted(alpaca_tickers) or "none"}')

            for trade in open_trades:
                ticker = trade['ticker']
                trade_id = trade['trade_id']
                print(f'[sync_closed_positions] Checking DB trade {trade_id} — {ticker} ({trade.get("trade_type")}, entry {trade.get("entry_price")})')

                if ticker in alpaca_tickers:
                    print(f'[sync_closed_positions]   {ticker} — still open in Alpaca, no action needed')
                else:
                    print(f'[sync_closed_positions]   {ticker} — NOT found in Alpaca positions (likely closed by stop-loss or take-profit)')

                    # Fetch the actual fill price from Alpaca order history
                    exit_price = self.executor.get_filled_exit_price(ticker)
                    if exit_price is None:
                        # Fall back to live market price if no filled order found
                        print(f'[sync_closed_positions]   {ticker} — no fill price from Alpaca, falling back to market price')
                        exit_price = self._get_current_price(ticker)
                    print(f'[sync_closed_positions]   {ticker} — exit price: {exit_price}')

                    # Calculate P&L using entry price and share count from the trade record
                    pnl, pnl_pct = self._calculate_pnl(trade, exit_price)
                    print(f'[sync_closed_positions]   {ticker} — P&L: {pnl} ({pnl_pct}%)')

                    try:
                        self.db.update_trade_status(
                            trade_id,
                            status='closed',
                            exit_reason='bracket_order_fill',
                            exit_price=exit_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                        print(f'[sync_closed_positions]   {ticker} — DB trade {trade_id} updated to CLOSED (exit_price={exit_price}, pnl={pnl}, exit_reason=bracket_order_fill)')
                    except Exception as update_err:
                        print(f'[sync_closed_positions]   ERROR updating trade {trade_id} ({ticker}) to closed: {update_err}')

            print('[sync_closed_positions] Done.')

        except Exception as e:
            print(f'[sync_closed_positions] ERROR during sync: {e}')
            log_error('sync_closed_positions', 'all', str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _calculate_pnl(self, trade: dict, exit_price: Optional[float]):
        """
        Compute realised P&L for a closed trade.

        Direction:
            long  (buy)   — profit when price rises: (exit - entry) * shares
            short (short) — profit when price falls: (entry - exit) * shares

        Returns:
            (pnl, pnl_pct) — both None if exit_price is unavailable or
            entry_price / shares are missing from the trade record.
        """
        if exit_price is None:
            return None, None

        entry_price = trade.get('entry_price')
        shares = trade.get('shares')
        trade_type = trade.get('trade_type', 'buy')

        if not entry_price or not shares:
            return None, None

        if trade_type in ('short', 'sell'):
            pnl = (entry_price - exit_price) * shares
        else:
            pnl = (exit_price - entry_price) * shares

        cost_basis = entry_price * shares
        pnl_pct = round((pnl / cost_basis) * 100, 4) if cost_basis else None
        pnl = round(pnl, 4)

        return pnl, pnl_pct

    def _get_current_price(self, ticker: str) -> Optional[float]:
        """
        Fetch the latest market price for a ticker using yfinance.
        Returns None if the price cannot be retrieved — callers store None
        as exit_price rather than blocking the status update.
        """
        try:
            price = yf.Ticker(ticker).fast_info.last_price
            return float(price) if price else None
        except Exception as e:
            print(f'[_get_current_price] Could not fetch price for {ticker}: {e}')
            return None

    # ── Hold Period Enforcement ───────────────────────────────────────────────

    def _check_hold_expiry(self, trade: dict):
        """
        Close a position if it has exceeded its maximum allowed hold duration.

        max_hold_days is stored on the trade record at entry time (sourced from
        config: 1 for intraday, 5 for swing, 20 for position trades). Using the
        per-record value rather than re-reading config means the rule applied
        at entry is always the rule enforced at exit, even if config changes.

        Args:
            trade: A trade record dict as returned by Database.get_open_trades().
        """
        entry_time = datetime.fromisoformat(trade['entry_time'])
        days_held = (datetime.now() - entry_time).days

        # Fall back to 5 days (swing default) if the field is missing from
        # legacy records written before max_hold_days was added to the schema
        max_days = trade.get('max_hold_days', 5)

        if days_held >= max_days:
            ticker = trade['ticker']
            print(f"Position {ticker} exceeded max hold period ({days_held} days). Closing.")
            try:
                # Place a market close order via Alpaca
                self.executor.close_position(ticker, trade['trade_type'])

                # Try Alpaca order history first; fall back to live market price
                exit_price = self.executor.get_filled_exit_price(ticker)
                if exit_price is None:
                    print(f'[_check_hold_expiry] {ticker} — no Alpaca fill price yet, using market price')
                    exit_price = self._get_current_price(ticker)
                print(f'[_check_hold_expiry] {ticker} — exit price: {exit_price}')

                pnl, pnl_pct = self._calculate_pnl(trade, exit_price)
                print(f'[_check_hold_expiry] {ticker} — P&L: {pnl} ({pnl_pct}%)')

                self.db.update_trade_status(
                    trade['trade_id'],
                    status='closed',
                    exit_reason='hold_period_expired',
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            except Exception as e:
                # Log and continue — a failure on one position should not
                # prevent the monitor from checking the remaining positions
                log_error('position_monitor', trade['ticker'], str(e))

    # ── Intraday Exit Logic ───────────────────────────────────────────────────

    def is_intraday_close_time(self) -> bool:
        """
        Returns True after 3:30 PM local time — the window during which all
        intraday positions must be closed before the 4:00 PM market close.

        The 3:45 PM scheduler cycle checks this flag before calling
        close_all_intraday(), giving a 15-minute execution buffer.
        """
        now = datetime.now()
        return now.hour >= 15 and now.minute >= 30

    def close_all_intraday(self):
        """
        Force-close every open intraday position before market close.

        Called by the scheduler on the 3:45 PM cycle when is_intraday_close_time()
        returns True. Intraday positions must never be held overnight — the
        wider gap risk is outside the risk budget defined for this hold tier.

        Guard: if config.allow_intraday is False, no intraday positions should
        exist (get_hold_period_safe() upgrades them all to swing at entry).
        The early return here is a safety net for that case.

        Failures are logged individually; remaining intraday positions continue
        to be processed so a single bad close does not leave others open.
        """
        # No intraday positions can exist when PDT protection is active
        if not config.allow_intraday:
            print('⚠️  Intraday disabled — no intraday positions to close')
            return

        open_trades = self.db.get_open_trades()

        # Filter to intraday tier only — swing and position trades are unaffected
        intraday = [t for t in open_trades if t.get('hold_period') == 'intraday']

        for trade in intraday:
            ticker = trade['ticker']
            try:
                self.executor.close_position(ticker, trade['trade_type'])

                # Try Alpaca order history first; fall back to live market price
                exit_price = self.executor.get_filled_exit_price(ticker)
                if exit_price is None:
                    print(f'[close_all_intraday] {ticker} — no Alpaca fill price yet, using market price')
                    exit_price = self._get_current_price(ticker)
                print(f'[close_all_intraday] {ticker} — exit price: {exit_price}')

                pnl, pnl_pct = self._calculate_pnl(trade, exit_price)
                print(f'[close_all_intraday] {ticker} — P&L: {pnl} ({pnl_pct}%)')

                self.db.update_trade_status(
                    trade['trade_id'],
                    status='closed',
                    exit_reason='intraday_forced_close',
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            except Exception as e:
                log_error('intraday_close', ticker, str(e))
