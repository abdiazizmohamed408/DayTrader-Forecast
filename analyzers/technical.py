"""
Technical Analysis Module.
Calculates various technical indicators for stock analysis.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class TechnicalAnalyzer:
    """
    Performs technical analysis on stock price data.
    
    Calculates indicators like RSI, MACD, Moving Averages, 
    Bollinger Bands, and identifies support/resistance levels.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration settings.
        
        Args:
            config: Dictionary with analysis parameters
        """
        self.config = config.get('analysis', {})
        
        # RSI settings
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        
        # MACD settings
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        
        # Moving average settings
        self.sma_short = self.config.get('sma_short', 20)
        self.sma_long = self.config.get('sma_long', 50)
        self.ema_period = self.config.get('ema_period', 12)
        
        # Bollinger Bands settings
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        
        # Volume settings
        self.volume_ma_period = self.config.get('volume_ma_period', 20)
    
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures momentum by comparing recent gains to recent losses.
        Values above 70 indicate overbought, below 30 indicate oversold.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Series with RSI values
        """
        delta = data['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD shows relationship between two moving averages.
        Crossovers signal potential buy/sell opportunities.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = data['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: DataFrame with 'close' column
            period: Number of periods for averaging
            
        Returns:
            Series with SMA values
        """
        return data['close'].rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        EMA gives more weight to recent prices than SMA.
        
        Args:
            data: DataFrame with 'close' column
            period: Number of periods
            
        Returns:
            Series with EMA values
        """
        return data['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_bollinger_bands(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Bands expand/contract based on volatility.
        Price near upper band may indicate overbought,
        near lower band may indicate oversold.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle = data['close'].rolling(window=self.bb_period).mean()
        std = data['close'].rolling(window=self.bb_period).std()
        
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        return upper, middle, lower
    
    def calculate_volume_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Analyze volume patterns.
        
        Compares current volume to average and identifies unusual activity.
        
        Args:
            data: DataFrame with 'volume' column
            
        Returns:
            Dictionary with volume analysis results
        """
        volume_ma = data['volume'].rolling(window=self.volume_ma_period).mean()
        current_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        return {
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'is_high_volume': volume_ratio > 1.5,
            'is_low_volume': volume_ratio < 0.5
        }
    
    def find_support_resistance(
        self, data: pd.DataFrame, lookback: int = 20
    ) -> Dict:
        """
        Identify support and resistance levels.
        
        Uses recent highs/lows and pivot points.
        
        Args:
            data: DataFrame with OHLC data
            lookback: Number of periods to analyze
            
        Returns:
            Dictionary with support/resistance levels
        """
        recent_data = data.tail(lookback)
        
        # Simple pivot point calculation
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        close = data['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        
        # Support and resistance levels
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'resistance_1': r1,
            'resistance_2': r2,
            'support_1': s1,
            'support_2': s2,
            'recent_high': high,
            'recent_low': low
        }
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        Perform complete technical analysis on stock data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all technical indicators
        """
        if data is None or len(data) < self.sma_long:
            return None
        
        # Calculate all indicators
        rsi = self.calculate_rsi(data)
        macd_line, signal_line, histogram = self.calculate_macd(data)
        sma_short = self.calculate_sma(data, self.sma_short)
        sma_long = self.calculate_sma(data, self.sma_long)
        ema = self.calculate_ema(data, self.ema_period)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
        volume_analysis = self.calculate_volume_analysis(data)
        support_resistance = self.find_support_resistance(data)
        
        # Get current values
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
        
        return {
            'current_price': current_price,
            'prev_price': prev_price,
            'price_change': current_price - prev_price,
            'price_change_pct': ((current_price - prev_price) / prev_price) * 100,
            
            # RSI
            'rsi': rsi.iloc[-1],
            'rsi_prev': rsi.iloc[-2] if len(rsi) > 1 else rsi.iloc[-1],
            'rsi_overbought': rsi.iloc[-1] > self.rsi_overbought,
            'rsi_oversold': rsi.iloc[-1] < self.rsi_oversold,
            
            # MACD
            'macd': macd_line.iloc[-1],
            'macd_signal': signal_line.iloc[-1],
            'macd_histogram': histogram.iloc[-1],
            'macd_crossover': (macd_line.iloc[-1] > signal_line.iloc[-1] and 
                              macd_line.iloc[-2] <= signal_line.iloc[-2]),
            'macd_crossunder': (macd_line.iloc[-1] < signal_line.iloc[-1] and 
                               macd_line.iloc[-2] >= signal_line.iloc[-2]),
            
            # Moving Averages
            'sma_short': sma_short.iloc[-1],
            'sma_long': sma_long.iloc[-1],
            'ema': ema.iloc[-1],
            'above_sma_short': current_price > sma_short.iloc[-1],
            'above_sma_long': current_price > sma_long.iloc[-1],
            'golden_cross': (sma_short.iloc[-1] > sma_long.iloc[-1] and 
                            sma_short.iloc[-2] <= sma_long.iloc[-2]),
            'death_cross': (sma_short.iloc[-1] < sma_long.iloc[-1] and 
                           sma_short.iloc[-2] >= sma_long.iloc[-2]),
            
            # Bollinger Bands
            'bb_upper': bb_upper.iloc[-1],
            'bb_middle': bb_middle.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'bb_position': self._get_bb_position(current_price, bb_upper.iloc[-1], 
                                                  bb_middle.iloc[-1], bb_lower.iloc[-1]),
            
            # Volume
            'volume': volume_analysis,
            
            # Support/Resistance
            'levels': support_resistance
        }
    
    def _get_bb_position(
        self, price: float, upper: float, middle: float, lower: float
    ) -> str:
        """
        Determine price position relative to Bollinger Bands.
        
        Args:
            price: Current price
            upper: Upper band
            middle: Middle band
            lower: Lower band
            
        Returns:
            Position description string
        """
        if price >= upper:
            return "ABOVE_UPPER"
        elif price <= lower:
            return "BELOW_LOWER"
        elif price > middle:
            return "UPPER_HALF"
        else:
            return "LOWER_HALF"
