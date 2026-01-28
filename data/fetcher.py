"""
Data fetching module using yfinance.
Retrieves historical and real-time stock data from Yahoo Finance.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


class DataFetcher:
    """
    Fetches stock market data using Yahoo Finance API.
    
    Attributes:
        cache: Dictionary to cache fetched data
    """
    
    def __init__(self):
        """Initialize the DataFetcher with empty cache."""
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def get_stock_data(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        cache_key = f"{ticker}_{period}_{interval}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                print(f"⚠️  No data found for {ticker}")
                return None
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return None
    
    def get_realtime_quote(self, ticker: str) -> Optional[Dict]:
        """
        Get real-time quote for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with quote data or None if fetch fails
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'symbol': ticker,
                'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'open': info.get('regularMarketOpen'),
                'high': info.get('dayHigh'),
                'low': info.get('dayLow'),
                'volume': info.get('regularMarketVolume'),
                'prev_close': info.get('regularMarketPreviousClose'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'name': info.get('shortName', ticker)
            }
            
        except Exception as e:
            print(f"❌ Error getting quote for {ticker}: {e}")
            return None
    
    def get_multiple_stocks(
        self,
        tickers: List[str],
        period: str = "3mo",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            data = self.get_stock_data(ticker, period, interval)
            if data is not None:
                results[ticker] = data
        
        return results
    
    def get_intraday_data(
        self,
        ticker: str,
        interval: str = "5m"
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday data for day trading analysis.
        
        Args:
            ticker: Stock symbol
            interval: Intraday interval (1m, 2m, 5m, 15m, 30m)
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        return self.get_stock_data(ticker, period="1d", interval=interval)
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
