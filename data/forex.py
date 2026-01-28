"""
Forex Data Module.
Fetches forex pair data using yfinance.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# Major forex pairs with their descriptions
FOREX_PAIRS = {
    'EURUSD=X': {'name': 'EUR/USD', 'description': 'Euro / US Dollar'},
    'GBPUSD=X': {'name': 'GBP/USD', 'description': 'British Pound / US Dollar'},
    'USDJPY=X': {'name': 'USD/JPY', 'description': 'US Dollar / Japanese Yen'},
    'USDCAD=X': {'name': 'USD/CAD', 'description': 'US Dollar / Canadian Dollar'},
    'AUDUSD=X': {'name': 'AUD/USD', 'description': 'Australian Dollar / US Dollar'},
    'USDCHF=X': {'name': 'USD/CHF', 'description': 'US Dollar / Swiss Franc'},
    'NZDUSD=X': {'name': 'NZD/USD', 'description': 'New Zealand Dollar / US Dollar'},
    'EURGBP=X': {'name': 'EUR/GBP', 'description': 'Euro / British Pound'},
    'EURJPY=X': {'name': 'EUR/JPY', 'description': 'Euro / Japanese Yen'},
    'GBPJPY=X': {'name': 'GBP/JPY', 'description': 'British Pound / Japanese Yen'},
}


class ForexFetcher:
    """
    Fetches forex market data using Yahoo Finance API.
    
    Supports major forex pairs with caching for performance.
    """
    
    def __init__(self):
        """Initialize the ForexFetcher with empty cache."""
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def get_forex_data(
        self,
        pair: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical forex data for a currency pair.
        
        Args:
            pair: Forex pair symbol (e.g., 'EURUSD=X')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y)
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        cache_key = f"{pair}_{period}_{interval}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"⚠️  No forex data found for {pair}")
                return None
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"❌ Error fetching forex {pair}: {e}")
            return None
    
    def get_realtime_quote(self, pair: str) -> Optional[Dict]:
        """
        Get real-time quote for a forex pair.
        
        Args:
            pair: Forex pair symbol
            
        Returns:
            Dictionary with quote data or None if fetch fails
        """
        try:
            ticker = yf.Ticker(pair)
            info = ticker.info
            
            # Get the pair info from our dictionary
            pair_info = FOREX_PAIRS.get(pair, {'name': pair, 'description': pair})
            
            return {
                'symbol': pair,
                'name': pair_info['name'],
                'description': pair_info['description'],
                'price': info.get('regularMarketPrice') or info.get('ask'),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'prev_close': info.get('regularMarketPreviousClose'),
                'open': info.get('regularMarketOpen'),
                'change': info.get('regularMarketChange'),
                'change_pct': info.get('regularMarketChangePercent'),
            }
            
        except Exception as e:
            print(f"❌ Error getting forex quote for {pair}: {e}")
            return None
    
    def get_multiple_pairs(
        self,
        pairs: List[str],
        period: str = "3mo",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple forex pairs.
        
        Args:
            pairs: List of forex pair symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping pair to DataFrame
        """
        results = {}
        
        for pair in pairs:
            data = self.get_forex_data(pair, period, interval)
            if data is not None:
                results[pair] = data
        
        return results
    
    def get_pip_value(self, pair: str, lot_size: float = 100000) -> float:
        """
        Calculate pip value for a forex pair.
        
        Args:
            pair: Forex pair symbol
            lot_size: Standard lot size (default 100,000)
            
        Returns:
            Value of one pip in base currency
        """
        # For pairs ending in USD
        if pair.endswith('USD=X') and not pair.startswith('USD'):
            return lot_size * 0.0001  # Standard pip value
        elif pair.startswith('USD'):
            # Need to get current rate for USD-first pairs
            quote = self.get_realtime_quote(pair)
            if quote and quote['price']:
                return (lot_size * 0.0001) / quote['price']
        
        return lot_size * 0.0001  # Default pip calculation
    
    def get_spread(self, pair: str) -> Optional[float]:
        """
        Get current spread for a forex pair.
        
        Args:
            pair: Forex pair symbol
            
        Returns:
            Spread in pips or None
        """
        quote = self.get_realtime_quote(pair)
        if quote and quote['bid'] and quote['ask']:
            # Calculate spread in pips
            if 'JPY' in pair:
                return (quote['ask'] - quote['bid']) * 100  # JPY pairs
            else:
                return (quote['ask'] - quote['bid']) * 10000  # Standard pairs
        return None
    
    def is_forex_session_active(self) -> Dict[str, bool]:
        """
        Check which forex sessions are currently active.
        
        Returns:
            Dictionary with session status
        """
        now = datetime.utcnow()
        hour = now.hour
        
        # Forex sessions (approximate UTC times)
        return {
            'sydney': 21 <= hour or hour < 6,    # 21:00 - 06:00 UTC
            'tokyo': 0 <= hour < 9,              # 00:00 - 09:00 UTC
            'london': 7 <= hour < 16,            # 07:00 - 16:00 UTC
            'new_york': 12 <= hour < 21,         # 12:00 - 21:00 UTC
            'overlap_london_ny': 12 <= hour < 16,  # Best volatility
        }
    
    def get_pair_name(self, symbol: str) -> str:
        """Get human-readable name for a forex pair."""
        return FOREX_PAIRS.get(symbol, {}).get('name', symbol.replace('=X', ''))
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
