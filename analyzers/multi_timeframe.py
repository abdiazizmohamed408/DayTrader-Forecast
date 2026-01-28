"""
Multi-Timeframe Analysis Module.
Analyzes multiple timeframes to find confluence and higher-probability setups.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .technical import TechnicalAnalyzer


class TimeframeTrend(Enum):
    """Trend direction for a timeframe."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe."""
    timeframe: str
    trend: TimeframeTrend
    strength: float  # 0-100
    rsi: float
    macd_bullish: bool
    above_sma20: bool
    above_sma50: bool
    volume_confirmed: bool


@dataclass
class MultiTimeframeResult:
    """Combined multi-timeframe analysis result."""
    ticker: str
    timeframes: Dict[str, TimeframeAnalysis]
    alignment_score: float  # 0-100, how aligned timeframes are
    dominant_trend: TimeframeTrend
    confluence_count: int  # How many timeframes agree
    is_valid_signal: bool
    reasons: List[str]


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes to identify high-probability setups.
    
    Only generates signals when multiple timeframes align, reducing
    false signals and increasing win probability.
    """
    
    # Timeframe configurations: (period, interval, weight)
    TIMEFRAMES = {
        '5min': ('5d', '5m', 0.10),
        '15min': ('5d', '15m', 0.15),
        '1hour': ('1mo', '1h', 0.20),
        '4hour': ('3mo', '1h', 0.25),  # Approximate with 1h
        'daily': ('6mo', '1d', 0.30),
    }
    
    def __init__(self, config: Dict):
        """
        Initialize multi-timeframe analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tech_analyzer = TechnicalAnalyzer(config)
        self.min_alignment = config.get('multi_tf', {}).get('min_alignment', 0.6)
        self.min_confluence = config.get('multi_tf', {}).get('min_confluence', 3)
    
    def analyze_timeframe(
        self,
        data: pd.DataFrame,
        timeframe: str
    ) -> Optional[TimeframeAnalysis]:
        """
        Analyze a single timeframe.
        
        Args:
            data: OHLCV DataFrame
            timeframe: Timeframe identifier
            
        Returns:
            TimeframeAnalysis or None if insufficient data
        """
        if data is None or len(data) < 50:
            return None
        
        analysis = self.tech_analyzer.analyze(data)
        if analysis is None:
            return None
        
        # Determine trend
        rsi = analysis.get('rsi', 50)
        macd_bullish = analysis.get('macd_histogram', 0) > 0
        above_sma20 = analysis.get('above_sma_short', False)
        above_sma50 = analysis.get('above_sma_long', False)
        
        # Volume confirmation
        volume = analysis.get('volume', {})
        volume_ratio = volume.get('volume_ratio', 1.0)
        volume_confirmed = volume_ratio >= 0.8  # At least 80% of average
        
        # Calculate trend strength
        bullish_signals = sum([
            1 if rsi > 50 else 0,
            1 if macd_bullish else 0,
            1 if above_sma20 else 0,
            1 if above_sma50 else 0,
        ])
        
        if bullish_signals >= 3:
            trend = TimeframeTrend.BULLISH
            strength = min(100, 50 + (bullish_signals * 12.5) + (rsi - 50))
        elif bullish_signals <= 1:
            trend = TimeframeTrend.BEARISH
            strength = min(100, 50 + ((4 - bullish_signals) * 12.5) + (50 - rsi))
        else:
            trend = TimeframeTrend.NEUTRAL
            strength = 50
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend,
            strength=strength,
            rsi=rsi,
            macd_bullish=macd_bullish,
            above_sma20=above_sma20,
            above_sma50=above_sma50,
            volume_confirmed=volume_confirmed
        )
    
    def analyze_all_timeframes(
        self,
        ticker: str,
        fetcher
    ) -> Optional[MultiTimeframeResult]:
        """
        Analyze all timeframes for a ticker.
        
        Args:
            ticker: Stock symbol
            fetcher: DataFetcher instance
            
        Returns:
            MultiTimeframeResult or None if analysis fails
        """
        timeframe_results = {}
        reasons = []
        
        for tf_name, (period, interval, weight) in self.TIMEFRAMES.items():
            try:
                data = fetcher.get_stock_data(ticker, period=period, interval=interval)
                analysis = self.analyze_timeframe(data, tf_name)
                
                if analysis:
                    timeframe_results[tf_name] = analysis
            except Exception as e:
                reasons.append(f"Failed to analyze {tf_name}: {str(e)}")
        
        if len(timeframe_results) < 3:
            return None
        
        # Calculate alignment
        bullish_count = sum(1 for tf in timeframe_results.values() 
                          if tf.trend == TimeframeTrend.BULLISH)
        bearish_count = sum(1 for tf in timeframe_results.values() 
                          if tf.trend == TimeframeTrend.BEARISH)
        total = len(timeframe_results)
        
        # Determine dominant trend
        if bullish_count > bearish_count:
            dominant = TimeframeTrend.BULLISH
            confluence = bullish_count
            alignment = bullish_count / total
        elif bearish_count > bullish_count:
            dominant = TimeframeTrend.BEARISH
            confluence = bearish_count
            alignment = bearish_count / total
        else:
            dominant = TimeframeTrend.NEUTRAL
            confluence = max(bullish_count, bearish_count)
            alignment = 0.5
        
        # Weight by timeframe importance (higher timeframes = more weight)
        weighted_score = 0
        total_weight = 0
        
        for tf_name, analysis in timeframe_results.items():
            weight = self.TIMEFRAMES[tf_name][2]
            total_weight += weight
            
            if analysis.trend == dominant:
                weighted_score += weight * (analysis.strength / 100)
        
        alignment_score = (weighted_score / total_weight * 100) if total_weight > 0 else 50
        
        # Check volume confirmation across timeframes
        volume_confirmed = sum(1 for tf in timeframe_results.values() 
                              if tf.volume_confirmed) >= (total // 2)
        
        # Determine if valid signal
        is_valid = (
            alignment >= self.min_alignment and
            confluence >= self.min_confluence and
            dominant != TimeframeTrend.NEUTRAL and
            volume_confirmed
        )
        
        # Build reasons
        if is_valid:
            reasons.append(f"{confluence}/{total} timeframes aligned {dominant.value}")
            if volume_confirmed:
                reasons.append("Volume confirmed across timeframes")
        else:
            if alignment < self.min_alignment:
                reasons.append(f"Insufficient alignment ({alignment:.0%} < {self.min_alignment:.0%})")
            if confluence < self.min_confluence:
                reasons.append(f"Low confluence ({confluence} < {self.min_confluence} timeframes)")
            if not volume_confirmed:
                reasons.append("Weak volume - possible false signal")
        
        return MultiTimeframeResult(
            ticker=ticker,
            timeframes=timeframe_results,
            alignment_score=alignment_score,
            dominant_trend=dominant,
            confluence_count=confluence,
            is_valid_signal=is_valid,
            reasons=reasons
        )
    
    def get_confluence_boost(self, result: MultiTimeframeResult) -> float:
        """
        Calculate probability boost from multi-timeframe confluence.
        
        Args:
            result: Multi-timeframe analysis result
            
        Returns:
            Probability boost (0-20%)
        """
        if not result.is_valid_signal:
            return -10  # Penalty for misalignment
        
        # Boost based on alignment
        base_boost = (result.alignment_score - 50) * 0.3
        
        # Extra boost for high confluence
        confluence_boost = min(10, (result.confluence_count - 2) * 3)
        
        return min(20, base_boost + confluence_boost)
