"""
news_monitor.py — Real-time breaking news scanner for trade catalysts.

Runs as a background thread in scheduler.py, scanning market-wide news every
15 minutes and triggering immediate single-stock analysis when high-impact
headlines are detected.

Design principles:
    - One Finnhub API call per check (general_news) to stay within free tier limits
    - Local filtering against the S&P 500 universe to identify affected tickers
    - Deduplication via seen_ids set to prevent the same headline triggering twice
    - Watchlist stocks get full position size; universe-only stocks get half size

Flow:
    scheduler.py (background thread)
        → get_breaking_news()         — fetches and filters headlines
        → run_single_ticker()         — triggers full crew analysis per ticker
        → trade_executor.execute()    — places order if crew approves

Usage:
    from news_monitor import NewsMonitor
    monitor = NewsMonitor()
    breaking = monitor.get_breaking_news()
"""

import finnhub
import time
from datetime import datetime, timedelta
from config import config
from logger import log_error


class NewsMonitor:
    """
    Stateful news scanner that tracks seen headline IDs to avoid
    duplicate triggers across the 60-second polling loop in scheduler.py.
    """

    def __init__(self):
        self.client     = finnhub.Client(api_key=config.finnhub_api_key)
        self.seen_ids   = set()   # Prevents same headline triggering multiple analyses
        self.last_check = None    # Timestamp of last API call — enforces 15-min rate limit

    # ── S&P 500 Universe ──────────────────────────────────────────────────────
    # Broader than the watchlist — used for news scanning only. Tickers found
    # here but not in config.watchlist receive half position size to reflect
    # lower conviction (no scheduled technical analysis for these symbols).
    SP500_UNIVERSE = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'BRK.B', 'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA',
        'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'KO', 'BAC', 'PFE',
        'TMO', 'COST', 'DIS', 'CSCO', 'WMT', 'ABT', 'MCD',
        'ACN', 'NEE', 'LIN', 'DHR', 'BMY', 'TXN', 'AVGO', 'QCOM',
        'UPS', 'RTX', 'HON', 'AMGN', 'SBUX', 'GS', 'MS', 'BLK',
        'CAT', 'DE', 'GE', 'F', 'GM', 'UBER', 'NFLX', 'SPOT',
        'COIN', 'AMD', 'INTC', 'MU', 'AMAT', 'CRM', 'NOW', 'SNOW',
    ]

    # ── Rate Limiting ─────────────────────────────────────────────────────────

    def should_check(self) -> bool:
        """
        Enforce a 15-minute minimum interval between API calls.

        Finnhub free tier allows limited requests — fetching general_news more
        frequently than this would exhaust the quota. The 60-second sleep in
        the scheduler loop calls this method on each iteration; it only returns
        True when 15 minutes have elapsed since the last actual API call.

        Returns:
            True if enough time has passed to make a new API call.
        """
        if self.last_check is None:
            return True  # First call — always proceed
        minutes_since = (datetime.now() - self.last_check).seconds / 60
        return minutes_since >= 15

    # ── News Fetching ─────────────────────────────────────────────────────────

    def get_breaking_news(self) -> list:
        """
        Fetch market-wide news and filter for high-impact, ticker-relevant headlines.

        Uses a single general_news API call rather than per-ticker calls to stay
        within the Finnhub free tier. Filtering happens locally:
            1. _is_high_impact() — keyword match on headline text
            2. SP500_UNIVERSE scan — check if any known ticker is mentioned
            3. Deduplication — skip any headline ID already in seen_ids

        Position size multiplier:
            1.0 — ticker is in config.watchlist (we have regular technical analysis)
            0.5 — ticker is universe-only (news-only signal, less context available)

        Returns:
            List of dicts with keys: ticker, headline, url, time,
            is_watchlist_stock, position_size_multiplier.
            Returns empty list if rate limit hasn't elapsed or on API failure.
        """
        if not self.should_check():
            return []

        breaking = []
        try:
            # Single API call covers all market news — no per-ticker calls needed
            news = self.client.general_news('general', min_id=0)
            self.last_check = datetime.now()
            print(f'📰 News check at {datetime.now().strftime("%H:%M")} — {len(news)} headlines scanned')

            for item in news:
                # Skip already-processed headlines
                if item.get('id') in self.seen_ids:
                    continue
                self.seen_ids.add(item.get('id'))

                headline = item.get('headline', '')
                summary  = item.get('summary', '')

                # Filter out low-impact headlines before ticker matching
                if not self._is_high_impact(headline):
                    continue

                # Scan headline and summary for S&P 500 ticker mentions
                for ticker in self.SP500_UNIVERSE:
                    if (ticker.lower() in headline.lower() or
                            ticker.lower() in summary.lower()):

                        is_watchlist = ticker in config.watchlist

                        breaking.append({
                            'ticker':                  ticker,
                            'headline':                headline,
                            'url':                     item.get('url', ''),
                            'time':                    item.get('datetime', ''),
                            'is_watchlist_stock':      is_watchlist,
                            # Half size for universe stocks — less analytical context
                            'position_size_multiplier': 1.0 if is_watchlist else 0.5,
                        })

                        print(
                            f'  🚨 {ticker} — '
                            f'{"WATCHLIST" if is_watchlist else "UNIVERSE"} — '
                            f'{headline[:60]}'
                        )
                        # Break after first ticker match per headline to avoid
                        # one article triggering multiple simultaneous analyses
                        break

        except Exception as e:
            log_error('news_monitor', 'general', str(e))

        return breaking

    # ── Impact Filter ─────────────────────────────────────────────────────────

    def _is_high_impact(self, headline: str) -> bool:
        """
        Return True if the headline contains keywords associated with
        price-moving corporate events.

        Keyword categories covered:
            Earnings:      beat, miss, revenue, profit, guidance, outlook
            M&A:           merger, acquisition, buyout, takeover
            Regulatory:    fda, approval, recall, lawsuit, sec, settlement
            Leadership:    ceo, resign, fired, appointed, steps down
            Capital:       dividend, buyback, split, ipo, bankruptcy
            Business:      partnership, contract, deal, layoffs, upgrade, downgrade

        Args:
            headline: Raw headline string from Finnhub.

        Returns:
            True if any keyword matches; False for general market commentary.
        """
        keywords = [
            # Earnings & financials
            'earnings', 'beat', 'miss', 'revenue', 'profit',
            'guidance', 'outlook', 'forecast',
            # M&A
            'merger', 'acquisition', 'buyout', 'takeover',
            # Regulatory & legal
            'fda', 'approval', 'approved', 'rejected', 'recall',
            'lawsuit', 'settlement', 'fine', 'investigation', 'sec',
            # Leadership changes
            'ceo', 'resign', 'fired', 'appointed', 'steps down',
            # Analyst actions
            'upgrade', 'downgrade',
            # Capital events
            'dividend', 'buyback', 'split', 'ipo',
            # Distress signals
            'bankruptcy', 'default', 'debt', 'layoffs',
            # Business events
            'partnership', 'contract', 'deal', 'agreement',
        ]
        return any(kw in headline.lower() for kw in keywords)
