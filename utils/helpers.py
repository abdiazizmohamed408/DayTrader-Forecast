"""
Helper utilities for DayTrader-Forecast.
Contains configuration loading, logging setup, and formatting functions.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    load_dotenv()
    
    return config


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('DayTrader')


def format_currency(value: float, symbol: str = "$") -> str:
    """
    Format a number as currency.
    
    Args:
        value: Numeric value to format
        symbol: Currency symbol (default: $)
        
    Returns:
        Formatted currency string
    """
    if value >= 0:
        return f"{symbol}{value:,.2f}"
    return f"-{symbol}{abs(value):,.2f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Numeric value (0.05 = 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_date_str() -> str:
    """Get current date string."""
    return datetime.now().strftime("%Y-%m-%d")


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def classify_signal_strength(probability: float) -> str:
    """
    Classify signal strength based on probability score.
    
    Args:
        probability: Probability score (0-100)
        
    Returns:
        Signal strength classification
    """
    if probability >= 80:
        return "STRONG"
    elif probability >= 60:
        return "MODERATE"
    elif probability >= 40:
        return "WEAK"
    else:
        return "VERY WEAK"
