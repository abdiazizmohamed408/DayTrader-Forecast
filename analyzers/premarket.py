"""
Pre-Market Scanner Module.
Scans for gaps, pre-market movers, and unusual activity.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional

import yfinance as yf
import pandas as pd


@dataclass
class PremarketData:
    """Pre-market data for a stock."""
    ticker: str
    premarket_price: Optional[float]
    prev_close: float
    gap_percent: float
    gap_direction: str  # UP, DOWN, FLAT
    premarket_volume: Optional[int]
    avg_volume: int
    volume_ratio: float
    is_gapper: bool  # Gap > 2%
    is_unusual_volume: bool
    action: str  # WATCH, POTENTIAL_LONG, POTENTIAL_SHORT, SKIP


@dataclass
class PremarketScan:
    """Pre-market scan results."""
    scan_time: datetime
    market_status: str  # PRE_MARKET, MARKET_OPEN, AFTER_HOURS, CLOSED
    gap_ups: List[PremarketData]
    gap_downs: List[PremarketData]
    unusual_volume: List[PremarketData]
    top_movers: List[PremarketData]


class PremarketScanner:
    """
    Scans for pre-market opportunities.
    
    Identifies:
    - Gap ups/downs > 2%
    - Unusual pre-market volume
    - Potential momentum plays
    """
    
    def __init__(self, config: Dict):
        """
        Initialize pre-market scanner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('premarket', {})
        self.min_gap_percent = self.config.get('min_gap_percent', 2.0)
        self.min_volume_ratio = self.config.get('min_volume_ratio', 1.5)
    
    def get_market_status(self) -> str:
        """Determine current market status."""
        now = datetime.now()
        current_time = now.time()
        
        # Check if weekend
        if now.weekday() >= 5:
            return "CLOSED"
        
        # Market hours (EST approximation - adjust for your timezone)
        premarket_start = time(4, 0)
        market_open = time(9, 30)
        market_close = time(16, 0)
        after_hours_end = time(20, 0)
        
        if premarket_start <= current_time < market_open:
            return "PRE_MARKET"
        elif market_open <= current_time < market_close:
            return "MARKET_OPEN"
        elif market_close <= current_time < after_hours_end:
            return "AFTER_HOURS"
        else:
            return "CLOSED"
    
    def scan_stock(self, ticker: str) -> Optional[PremarketData]:
        """
        Scan a single stock for pre-market data.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            PremarketData or None if unavailable
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get previous close
            prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
            if not prev_close:
                return None
            
            # Get pre-market price if available
            premarket_price = info.get('preMarketPrice')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            # Use pre-market price if available, otherwise current
            compare_price = premarket_price or current_price
            if not compare_price:
                return None
            
            # Calculate gap
            gap_percent = ((compare_price - prev_close) / prev_close) * 100
            
            if gap_percent > 0.5:
                gap_direction = "UP"
            elif gap_percent < -0.5:
                gap_direction = "DOWN"
            else:
                gap_direction = "FLAT"
            
            # Volume analysis
            premarket_volume = info.get('preMarketVolume')
            avg_volume = info.get('averageVolume', 1)
            current_volume = info.get('regularMarketVolume', 0)
            
            # Volume ratio (compare to average daily)
            if premarket_volume and avg_volume:
                # Pre-market is typically ~10% of daily volume, so adjust
                volume_ratio = (premarket_volume / (avg_volume * 0.1))
            elif current_volume and avg_volume:
                volume_ratio = current_volume / avg_volume
            else:
                volume_ratio = 1.0
            
            is_gapper = abs(gap_percent) >= self.min_gap_percent
            is_unusual_volume = volume_ratio >= self.min_volume_ratio
            
            # Determine action
            if is_gapper and is_unusual_volume:
                if gap_percent > 0:
                    action = "POTENTIAL_LONG"
                else:
                    action = "POTENTIAL_SHORT"
            elif is_gapper or is_unusual_volume:
                action = "WATCH"
            else:
                action = "SKIP"
            
            return PremarketData(
                ticker=ticker,
                premarket_price=premarket_price,
                prev_close=prev_close,
                gap_percent=gap_percent,
                gap_direction=gap_direction,
                premarket_volume=premarket_volume,
                avg_volume=avg_volume,
                volume_ratio=volume_ratio,
                is_gapper=is_gapper,
                is_unusual_volume=is_unusual_volume,
                action=action
            )
            
        except Exception as e:
            return None
    
    def scan_watchlist(self, tickers: List[str]) -> PremarketScan:
        """
        Scan entire watchlist for pre-market opportunities.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            PremarketScan with categorized results
        """
        market_status = self.get_market_status()
        results = []
        
        for ticker in tickers:
            data = self.scan_stock(ticker)
            if data:
                results.append(data)
        
        # Categorize results
        gap_ups = sorted(
            [r for r in results if r.gap_direction == "UP" and r.is_gapper],
            key=lambda x: x.gap_percent,
            reverse=True
        )
        
        gap_downs = sorted(
            [r for r in results if r.gap_direction == "DOWN" and r.is_gapper],
            key=lambda x: x.gap_percent
        )
        
        unusual_volume = sorted(
            [r for r in results if r.is_unusual_volume],
            key=lambda x: x.volume_ratio,
            reverse=True
        )
        
        # Top movers by absolute gap
        top_movers = sorted(
            results,
            key=lambda x: abs(x.gap_percent),
            reverse=True
        )[:10]
        
        return PremarketScan(
            scan_time=datetime.now(),
            market_status=market_status,
            gap_ups=gap_ups,
            gap_downs=gap_downs,
            unusual_volume=unusual_volume,
            top_movers=top_movers
        )
    
    def get_gapper_signals(
        self, scan: PremarketScan, min_gap: float = 3.0
    ) -> List[Dict]:
        """
        Get trading signals from gap analysis.
        
        Args:
            scan: Pre-market scan results
            min_gap: Minimum gap percentage for signal
            
        Returns:
            List of potential trade setups
        """
        signals = []
        
        # Gap and go setups (trade with the gap)
        for gapper in scan.gap_ups:
            if gapper.gap_percent >= min_gap and gapper.is_unusual_volume:
                signals.append({
                    'ticker': gapper.ticker,
                    'strategy': 'GAP_AND_GO_LONG',
                    'gap_percent': gapper.gap_percent,
                    'volume_ratio': gapper.volume_ratio,
                    'entry': 'Break above pre-market high',
                    'stop': 'Below VWAP or gap fill level'
                })
        
        for gapper in scan.gap_downs:
            if gapper.gap_percent <= -min_gap and gapper.is_unusual_volume:
                signals.append({
                    'ticker': gapper.ticker,
                    'strategy': 'GAP_AND_GO_SHORT',
                    'gap_percent': gapper.gap_percent,
                    'volume_ratio': gapper.volume_ratio,
                    'entry': 'Break below pre-market low',
                    'stop': 'Above VWAP or gap fill level'
                })
        
        # Gap fill setups (trade against the gap)
        for gapper in scan.gap_ups:
            if 2.0 <= gapper.gap_percent < min_gap and not gapper.is_unusual_volume:
                signals.append({
                    'ticker': gapper.ticker,
                    'strategy': 'GAP_FILL_SHORT',
                    'gap_percent': gapper.gap_percent,
                    'volume_ratio': gapper.volume_ratio,
                    'entry': 'Weakness at open, break of support',
                    'target': 'Previous close (gap fill)'
                })
        
        return signals
