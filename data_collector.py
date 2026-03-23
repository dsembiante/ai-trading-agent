"""
data_collector.py — Aggregates market data from all external sources.

Each source is wrapped in an independent try/except block so a single API
outage or rate-limit does not abort the collection cycle. Failures are
recorded in DataSourceStatus and logged via logger.py; downstream agents
receive the best available data and adjust their confidence accordingly.

Sources:
    1. Alpaca      — OHLCV bars + locally-computed technical indicators
    2. Finnhub     — Recent news headlines + company sentiment score
    3. Alpha Vantage — Company fundamentals (cached daily to stay within 25 call/day limit)
    4. FRED        — Macro series: Fed Funds Rate + trailing CPI inflation

Usage:
    from data_collector import DataCollector
    data = DataCollector().collect('AAPL')
"""

import finnhub
import pandas as pd
import pandas_ta as ta
import requests, json, os
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from fredapi import Fred
from models import MarketData, DataSourceStatus
from config import config
from logger import log_error


class DataCollector:
    """
    Stateful collector that holds authenticated API clients for the lifetime
    of a scheduler run. Instantiate once per cycle and call collect() per ticker.
    """

    def __init__(self):
        # Alpaca historical client — used for OHLCV bar data only (not order placement)
        self.alpaca = StockHistoricalDataClient(
            config.alpaca_api_key, config.alpaca_secret_key)

        # Finnhub client — news and sentiment
        self.finnhub = finnhub.Client(api_key=config.finnhub_api_key)

        # FRED client — macro economic series
        self.fred = Fred(api_key=config.fred_api_key)

        # Ensure cache directory exists before any source tries to write to it
        os.makedirs(config.cache_dir, exist_ok=True)

    def collect(self, ticker: str) -> MarketData:
        """
        Fetch and aggregate all available signals for a single ticker.

        Returns a MarketData object populated with whatever data was reachable.
        Fields are None when their source was unavailable — agents must handle
        this via reduced-signal analysis rather than raising exceptions.
        """
        # Track which sources succeeded; passed to agents for confidence weighting
        status = DataSourceStatus()

        # Initialise all fields to None so we can safely return partials on failure
        price = volume = rsi = macd = ma50 = ma200 = None
        news_sentiment = None
        headlines = []
        macro_context = None

        # ── 1. Alpaca — Price & Technical Indicators ──────────────────────────
        # Pull 250 days of daily bars to have enough history for SMA-200.
        # Technical indicators are computed locally with pandas-ta to avoid
        # additional API calls and rate-limit pressure.
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=300)
            ))
            df = bars.df.reset_index()

            # Latest close and volume for position sizing and display
            price = float(df['close'].iloc[-1])
            volume = int(df['volume'].iloc[-1])

            # Compute indicators in-place — pandas-ta appends columns to df
            df.ta.rsi(append=True)
            df.ta.macd(append=True)
            df['SMA_50'] = df['close'].rolling(50).mean()
            df['SMA_200'] = df['close'].rolling(200).mean()

            # Guard with column existence check — indicator may be NaN if
            # fewer bars than the lookback period were returned
            rsi = float(df['RSI_14'].iloc[-1]) if 'RSI_14' in df else None
            macd = float(df['MACD_12_26_9'].iloc[-1]) if 'MACD_12_26_9' in df else None
            ma50 = float(df['SMA_50'].iloc[-1])
            ma200 = float(df['SMA_200'].iloc[-1])

        except Exception as e:
            status.alpaca = False
            log_error('alpaca', ticker, str(e))

        # ── 2. Finnhub — News Headlines & Sentiment ───────────────────────────
        # Split into two separate try blocks so a 403 on news_sentiment
        # (paid-tier endpoint) does not abort the headlines fetch (free tier).
        # Headlines are available on the Finnhub free plan; sentiment is not.

        # 2a. Headlines — free tier, should succeed with any valid API key
        try:
            news = self.finnhub.company_news(
                ticker,
                _from=(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d')
            )
            if news:
                headlines = [n['headline'] for n in news[:5]]
        except Exception as e:
            status.finnhub = False
            log_error('finnhub_news', ticker, str(e))

        # 2b. Sentiment score — requires Finnhub paid plan; degrades gracefully
        # to None if unavailable. news_sentiment stays None on free tier.
        try:
            sd = self.finnhub.news_sentiment(ticker)
            # companyNewsScore: -1.0 (very bearish) to 1.0 (very bullish)
            news_sentiment = sd.get('companyNewsScore', None)
        except Exception as e:
            # 403 on free tier is expected — log at debug level only
            log_error('finnhub_sentiment', ticker, str(e))

        # ── 3. Alpha Vantage — Company Fundamentals ───────────────────────────
        # Alpha Vantage free tier is capped at 25 requests/day across all tickers.
        # We mitigate this with a date-keyed per-ticker cache so each symbol is
        # fetched at most once per calendar day regardless of how many cycles run.
        try:
            cache_file = (
                f"{config.cache_dir}/{ticker}_av_{datetime.now().strftime('%Y%m%d')}.json"
            )
            if os.path.exists(cache_file):
                # Serve from disk cache — no API call consumed
                with open(cache_file) as f:
                    av_data = json.load(f)
            else:
                url = (
                    f'https://www.alphavantage.co/query'
                    f'?function=OVERVIEW&symbol={ticker}&apikey={config.alpha_vantage_key}'
                )
                av_data = requests.get(url, timeout=10).json()
                # Persist to cache immediately after a successful fetch
                with open(cache_file, 'w') as f:
                    json.dump(av_data, f)

        except Exception as e:
            status.alpha_vantage = False
            log_error('alpha_vantage', ticker, str(e))

        # ── 4. FRED — Macro Economic Context ─────────────────────────────────
        # Macro data is shared across all tickers in a cycle so it is cached
        # in a single date-keyed file rather than per-ticker. FEDFUNDS and
        # CPIAUCSL are the two primary macro signals used in agent prompts.
        try:
            macro_cache = f"{config.cache_dir}/macro_{datetime.now().strftime('%Y%m%d')}.json"
            if os.path.exists(macro_cache):
                with open(macro_cache) as f:
                    macro = json.load(f)
            else:
                fed_rate = self.fred.get_series('FEDFUNDS').iloc[-1]
                # Year-over-year % change in CPI as a proxy for trailing inflation
                inflation = self.fred.get_series('CPIAUCSL').pct_change(12, fill_method=None).iloc[-1] * 100
                macro = {'fed_rate': float(fed_rate), 'inflation': float(inflation)}
                with open(macro_cache, 'w') as f:
                    json.dump(macro, f)

            # Format as a single string — injected verbatim into agent prompts
            macro_context = (
                f"Fed Rate: {macro['fed_rate']:.2f}%, "
                f"Inflation: {macro['inflation']:.2f}%"
            )

        except Exception as e:
            status.fred = False
            log_error('fred', ticker, str(e))

        # ── Assemble & Return ─────────────────────────────────────────────────
        # price/volume fall back to 0.0/0 so the model is always constructable;
        # agents must check DataSourceStatus.alpaca before trusting these values.
        return MarketData(
            ticker=ticker,
            current_price=price or 0.0,
            volume=volume or 0,
            rsi=rsi,
            macd=macd,
            moving_avg_50=ma50,
            moving_avg_200=ma200,
            news_sentiment=news_sentiment,
            news_headlines=headlines,
            macro_context=macro_context,
            data_sources_used=status,
        )

    def get_market_regime(self) -> str:
        """
        Determine the current broad market regime using SPY price vs. moving averages.

        Uses the classic golden cross / death cross framework:
            Bull:     Price > SMA-50 > SMA-200 — uptrend confirmed on both timeframes
            Bear:     Price < SMA-50 < SMA-200 — downtrend confirmed on both timeframes
            Sideways: Neither condition met — mixed or transitioning market

        Called once per cycle in crew.py before the per-ticker loop so all agents
        operate with the same regime context. Defaults to 'sideways' on failure
        to err on the side of caution rather than assuming a bull market.

        Returns:
            'bull', 'bear', or 'sideways'
        """
        try:
            bars = self.alpaca.get_stock_bars(StockBarsRequest(
                symbol_or_symbols='SPY',
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=300),
            ))
            df = bars.df.reset_index()
            df['SMA_50']  = df['close'].rolling(50).mean()
            df['SMA_200'] = df['close'].rolling(200).mean()

            current_price = float(df['close'].iloc[-1])
            sma_50        = float(df['SMA_50'].iloc[-1])
            sma_200       = float(df['SMA_200'].iloc[-1])

            # Golden cross = bull market; death cross = bear market
            if current_price > sma_50 and sma_50 > sma_200:
                return 'bull'
            elif current_price < sma_50 and sma_50 < sma_200:
                return 'bear'
            else:
                return 'sideways'

        except Exception as e:
            log_error('market_regime', 'SPY', str(e))
            return 'sideways'  # Cautious default if detection fails
