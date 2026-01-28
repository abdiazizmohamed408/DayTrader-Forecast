"""
Data Validation Module.
Validates data quality and handles errors gracefully.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataValidator:
    """
    Validates market data quality before analysis.
    
    Checks for:
    - Missing values
    - Outliers
    - Stale data
    - Data consistency
    """
    
    def __init__(self, config: Dict):
        """
        Initialize validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('validation', {})
        self.max_missing_pct = self.config.get('max_missing_pct', 5)
        self.max_stale_days = self.config.get('max_stale_days', 5)
        self.outlier_std = self.config.get('outlier_std', 5)
        
        # Set up logging
        self.logger = logging.getLogger('DataValidator')
    
    def validate_ohlcv(self, data: pd.DataFrame, ticker: str = "") -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data quality.
        
        Args:
            data: DataFrame with OHLCV data
            ticker: Stock symbol for logging
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if data is None:
            return False, ["No data received"]
        
        if data.empty:
            return False, ["Empty DataFrame"]
        
        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required if c not in data.columns]
        if missing_cols:
            return False, [f"Missing columns: {missing_cols}"]
        
        # Check for NaN values
        nan_pct = (data[required].isna().sum() / len(data) * 100).to_dict()
        for col, pct in nan_pct.items():
            if pct > self.max_missing_pct:
                issues.append(f"{col} has {pct:.1f}% missing values")
        
        # Check for stale data
        if hasattr(data.index, 'max'):
            last_date = data.index.max()
            if hasattr(last_date, 'tz_localize'):
                last_date = last_date.tz_localize(None)
            
            now = datetime.now()
            if hasattr(last_date, 'to_pydatetime'):
                last_date = last_date.to_pydatetime()
            
            days_old = (now - last_date).days
            
            # Account for weekends
            if now.weekday() == 0:  # Monday
                days_old -= 2
            elif now.weekday() == 6:  # Sunday
                days_old -= 1
            
            if days_old > self.max_stale_days:
                issues.append(f"Data is {days_old} days old")
        
        # Check OHLC consistency
        invalid_rows = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        ).sum()
        
        if invalid_rows > 0:
            issues.append(f"{invalid_rows} rows with invalid OHLC relationships")
        
        # Check for outliers in returns
        returns = data['close'].pct_change().dropna()
        if len(returns) > 0:
            mean_ret = returns.mean()
            std_ret = returns.std()
            
            outliers = (np.abs(returns - mean_ret) > self.outlier_std * std_ret).sum()
            if outliers > len(returns) * 0.05:  # More than 5% outliers
                issues.append(f"{outliers} potential price outliers detected")
        
        # Check for zero/negative prices
        invalid_prices = (
            (data['close'] <= 0) |
            (data['open'] <= 0) |
            (data['high'] <= 0) |
            (data['low'] <= 0)
        ).sum()
        
        if invalid_prices > 0:
            issues.append(f"{invalid_prices} rows with zero/negative prices")
        
        # Check for sufficient data
        if len(data) < 50:
            issues.append(f"Insufficient data ({len(data)} rows, need 50+)")
        
        is_valid = len(issues) == 0
        
        if issues:
            self.logger.warning(f"{ticker}: {', '.join(issues)}")
        
        return is_valid, issues
    
    def validate_signal(
        self,
        signal_type: str,
        entry_price: float,
        stop_loss: float,
        target_price: float
    ) -> Tuple[bool, List[str]]:
        """
        Validate a trading signal.
        
        Args:
            signal_type: BUY or SELL
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if entry_price <= 0:
            issues.append("Entry price must be positive")
        
        if stop_loss <= 0:
            issues.append("Stop loss must be positive")
        
        if target_price <= 0:
            issues.append("Target price must be positive")
        
        if signal_type == 'BUY':
            if stop_loss >= entry_price:
                issues.append("BUY signal: stop loss must be below entry")
            if target_price <= entry_price:
                issues.append("BUY signal: target must be above entry")
        elif signal_type == 'SELL':
            if stop_loss <= entry_price:
                issues.append("SELL signal: stop loss must be above entry")
            if target_price >= entry_price:
                issues.append("SELL signal: target must be below entry")
        
        return len(issues) == 0, issues
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and fix common data issues.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Cleaned DataFrame
        """
        if data is None or data.empty:
            return data
        
        data = data.copy()
        
        # Fill missing values
        data = data.ffill()  # Forward fill
        data = data.bfill()  # Back fill remaining
        
        # Fix OHLC consistency
        # High should be max of OHLC
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        # Low should be min of OHLC
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Remove rows with zero/negative prices
        mask = (
            (data['close'] > 0) &
            (data['open'] > 0) &
            (data['high'] > 0) &
            (data['low'] > 0)
        )
        data = data[mask]
        
        return data
    
    def validate_quote(self, quote: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a real-time quote.
        
        Args:
            quote: Quote dictionary
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if quote is None:
            return False, ["No quote data"]
        
        price = quote.get('price')
        if price is None or price <= 0:
            issues.append("Invalid or missing price")
        
        volume = quote.get('volume')
        if volume is not None and volume < 0:
            issues.append("Negative volume")
        
        return len(issues) == 0, issues


class ErrorHandler:
    """
    Centralized error handling and logging.
    """
    
    def __init__(self, log_file: str = "./data/errors.log"):
        """
        Initialize error handler.
        
        Args:
            log_file: Path to error log file
        """
        self.log_file = log_file
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DayTrader')
    
    def log_error(self, component: str, error: Exception, context: str = ""):
        """
        Log an error with context.
        
        Args:
            component: Component where error occurred
            error: The exception
            context: Additional context
        """
        self.logger.error(f"[{component}] {str(error)} - {context}")
    
    def log_warning(self, component: str, message: str):
        """Log a warning."""
        self.logger.warning(f"[{component}] {message}")
    
    def log_info(self, component: str, message: str):
        """Log info message."""
        self.logger.info(f"[{component}] {message}")
