"""
Market Context Analyzer.
Analyzes overall market conditions (SPY, QQQ) to adjust signal confidence.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class MarketTrend(Enum):
    """Overall market trend."""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class MarketContext:
    """Market context information."""
    spy_trend: MarketTrend
    qqq_trend: MarketTrend
    overall_trend: MarketTrend
    spy_change_pct: float
    qqq_change_pct: float
    spy_above_20sma: bool
    spy_above_50sma: bool
    qqq_above_20sma: bool
    qqq_above_50sma: bool
    vix_level: Optional[float] = None
    confidence_adjustment: float = 0.0
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'spy_trend': self.spy_trend.value,
            'qqq_trend': self.qqq_trend.value,
            'overall_trend': self.overall_trend.value,
            'spy_change_pct': float(self.spy_change_pct),
            'qqq_change_pct': float(self.qqq_change_pct),
            'spy_above_20sma': bool(self.spy_above_20sma),
            'spy_above_50sma': bool(self.spy_above_50sma),
            'qqq_above_20sma': bool(self.qqq_above_20sma),
            'qqq_above_50sma': bool(self.qqq_above_50sma),
            'vix_level': float(self.vix_level) if self.vix_level else None,
            'confidence_adjustment': float(self.confidence_adjustment),
            'description': str(self.description)
        }


class MarketAnalyzer:
    """
    Analyzes overall market conditions.
    
    Checks SPY and QQQ trends to provide context for
    individual stock signals.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize market analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def analyze_market(self, fetcher) -> Optional[MarketContext]:
        """
        Analyze current market conditions.
        
        Args:
            fetcher: DataFetcher instance
            
        Returns:
            MarketContext or None if data unavailable
        """
        from analyzers.technical import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer(self.config)
        
        # Fetch SPY and QQQ data
        spy_data = fetcher.get_stock_data('SPY', period='3mo', interval='1d')
        qqq_data = fetcher.get_stock_data('QQQ', period='3mo', interval='1d')
        
        if spy_data is None or qqq_data is None:
            return None
        
        # Analyze both
        spy_analysis = analyzer.analyze(spy_data)
        qqq_analysis = analyzer.analyze(qqq_data)
        
        if spy_analysis is None or qqq_analysis is None:
            return None
        
        # Determine trends
        spy_trend = self._determine_trend(spy_analysis)
        qqq_trend = self._determine_trend(qqq_analysis)
        
        # Overall trend (weighted average favoring SPY)
        overall_trend = self._calculate_overall_trend(spy_trend, qqq_trend)
        
        # Calculate confidence adjustment
        conf_adj = self._calculate_confidence_adjustment(overall_trend)
        
        # Build description
        description = self._build_description(spy_trend, qqq_trend, overall_trend)
        
        # Try to get VIX
        vix_level = self._get_vix_level(fetcher)
        
        return MarketContext(
            spy_trend=spy_trend,
            qqq_trend=qqq_trend,
            overall_trend=overall_trend,
            spy_change_pct=spy_analysis.get('price_change_pct', 0),
            qqq_change_pct=qqq_analysis.get('price_change_pct', 0),
            spy_above_20sma=spy_analysis.get('above_sma_short', False),
            spy_above_50sma=spy_analysis.get('above_sma_long', False),
            qqq_above_20sma=qqq_analysis.get('above_sma_short', False),
            qqq_above_50sma=qqq_analysis.get('above_sma_long', False),
            vix_level=vix_level,
            confidence_adjustment=conf_adj,
            description=description
        )
    
    def _determine_trend(self, analysis: Dict) -> MarketTrend:
        """
        Determine trend from technical analysis.
        
        Args:
            analysis: Technical analysis results
            
        Returns:
            MarketTrend enum value
        """
        score = 0
        
        # RSI contribution
        rsi = analysis.get('rsi', 50)
        if rsi > 70:
            score += 2
        elif rsi > 60:
            score += 1
        elif rsi < 30:
            score -= 2
        elif rsi < 40:
            score -= 1
        
        # Moving average position
        if analysis.get('above_sma_short'):
            score += 1
        else:
            score -= 1
            
        if analysis.get('above_sma_long'):
            score += 2
        else:
            score -= 2
        
        # MACD
        if analysis.get('macd_histogram', 0) > 0:
            score += 1
        else:
            score -= 1
        
        if analysis.get('macd_crossover'):
            score += 2
        elif analysis.get('macd_crossunder'):
            score -= 2
        
        # Golden/Death cross
        if analysis.get('golden_cross'):
            score += 3
        elif analysis.get('death_cross'):
            score -= 3
        
        # Price change
        change_pct = analysis.get('price_change_pct', 0)
        if change_pct > 2:
            score += 2
        elif change_pct > 0.5:
            score += 1
        elif change_pct < -2:
            score -= 2
        elif change_pct < -0.5:
            score -= 1
        
        # Map score to trend
        if score >= 6:
            return MarketTrend.STRONG_BULLISH
        elif score >= 2:
            return MarketTrend.BULLISH
        elif score <= -6:
            return MarketTrend.STRONG_BEARISH
        elif score <= -2:
            return MarketTrend.BEARISH
        else:
            return MarketTrend.NEUTRAL
    
    def _calculate_overall_trend(
        self,
        spy_trend: MarketTrend,
        qqq_trend: MarketTrend
    ) -> MarketTrend:
        """
        Calculate overall market trend from SPY and QQQ.
        
        SPY is weighted slightly higher as the broader market indicator.
        """
        trend_scores = {
            MarketTrend.STRONG_BULLISH: 2,
            MarketTrend.BULLISH: 1,
            MarketTrend.NEUTRAL: 0,
            MarketTrend.BEARISH: -1,
            MarketTrend.STRONG_BEARISH: -2
        }
        
        # SPY weighted 60%, QQQ 40%
        score = (trend_scores[spy_trend] * 0.6 + 
                 trend_scores[qqq_trend] * 0.4)
        
        if score >= 1.5:
            return MarketTrend.STRONG_BULLISH
        elif score >= 0.5:
            return MarketTrend.BULLISH
        elif score <= -1.5:
            return MarketTrend.STRONG_BEARISH
        elif score <= -0.5:
            return MarketTrend.BEARISH
        else:
            return MarketTrend.NEUTRAL
    
    def _calculate_confidence_adjustment(self, trend: MarketTrend) -> float:
        """
        Calculate confidence adjustment based on market trend.
        
        Returns a value to add/subtract from signal probability.
        """
        adjustments = {
            MarketTrend.STRONG_BULLISH: 10,  # Boost longs, reduce shorts
            MarketTrend.BULLISH: 5,
            MarketTrend.NEUTRAL: 0,
            MarketTrend.BEARISH: -5,
            MarketTrend.STRONG_BEARISH: -10
        }
        return adjustments.get(trend, 0)
    
    def _build_description(
        self,
        spy_trend: MarketTrend,
        qqq_trend: MarketTrend,
        overall: MarketTrend
    ) -> str:
        """Build human-readable market description."""
        trend_names = {
            MarketTrend.STRONG_BULLISH: "strongly bullish 📈📈",
            MarketTrend.BULLISH: "bullish 📈",
            MarketTrend.NEUTRAL: "neutral ➡️",
            MarketTrend.BEARISH: "bearish 📉",
            MarketTrend.STRONG_BEARISH: "strongly bearish 📉📉"
        }
        
        return (
            f"Market is {trend_names[overall]}. "
            f"SPY: {trend_names[spy_trend]}, "
            f"QQQ: {trend_names[qqq_trend]}"
        )
    
    def _get_vix_level(self, fetcher) -> Optional[float]:
        """
        Get current VIX level.
        
        Args:
            fetcher: DataFetcher instance
            
        Returns:
            VIX close price or None
        """
        try:
            vix_data = fetcher.get_stock_data('^VIX', period='5d', interval='1d')
            if vix_data is not None and len(vix_data) > 0:
                return vix_data['close'].iloc[-1]
        except Exception:
            pass
        return None
    
    def adjust_signal_confidence(
        self,
        signal_type: str,
        probability: float,
        market_context: MarketContext
    ) -> float:
        """
        Adjust signal probability based on market context.
        
        Args:
            signal_type: BUY or SELL
            probability: Original probability
            market_context: Current market context
            
        Returns:
            Adjusted probability (capped at 0-100)
        """
        adjustment = market_context.confidence_adjustment
        
        if signal_type == 'BUY':
            # Bullish market boosts long signals
            new_prob = probability + adjustment
        elif signal_type == 'SELL':
            # Bullish market reduces short signals (opposite adjustment)
            new_prob = probability - adjustment
        else:
            new_prob = probability
        
        # VIX adjustment
        if market_context.vix_level:
            if market_context.vix_level > 30:
                # High fear - reduce all confidence
                new_prob -= 5
            elif market_context.vix_level < 15:
                # Low fear - slight boost
                new_prob += 2
        
        # Cap between 0 and 100
        return max(0, min(100, new_prob))
    
    def format_context(self, context: MarketContext) -> str:
        """
        Format market context for display.
        
        Args:
            context: MarketContext object
            
        Returns:
            Formatted string
        """
        lines = []
        lines.append("🌍 MARKET CONTEXT")
        lines.append("─" * 40)
        lines.append(context.description)
        lines.append("")
        lines.append(f"SPY: {context.spy_change_pct:+.2f}% | "
                    f"Above 20 SMA: {'✅' if context.spy_above_20sma else '❌'} | "
                    f"Above 50 SMA: {'✅' if context.spy_above_50sma else '❌'}")
        lines.append(f"QQQ: {context.qqq_change_pct:+.2f}% | "
                    f"Above 20 SMA: {'✅' if context.qqq_above_20sma else '❌'} | "
                    f"Above 50 SMA: {'✅' if context.qqq_above_50sma else '❌'}")
        
        if context.vix_level:
            vix_status = "🟢 Low" if context.vix_level < 20 else \
                        "🟡 Normal" if context.vix_level < 30 else "🔴 High"
            lines.append(f"VIX: {context.vix_level:.2f} ({vix_status})")
        
        lines.append("")
        return "\n".join(lines)
