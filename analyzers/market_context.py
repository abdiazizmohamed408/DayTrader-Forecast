"""
Market Context Analyzer.
Analyzes overall market conditions to filter signals appropriately.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from .technical import TechnicalAnalyzer


class MarketRegime(Enum):
    """Overall market condition."""
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CRASH = "CRASH"


@dataclass
class MarketContext:
    """Market context analysis result."""
    regime: MarketRegime
    spy_trend: str  # BULLISH, BEARISH, NEUTRAL
    qqq_trend: str
    market_strength: float  # 0-100
    volatility_level: str  # LOW, NORMAL, HIGH, EXTREME
    breadth_positive: bool  # Are most stocks up?
    safe_to_long: bool
    safe_to_short: bool
    reasons: List[str]
    sentiment_adjustment: float  # -20 to +20 adjustment


class MarketContextAnalyzer:
    """
    Analyzes overall market conditions.
    
    Checks SPY and QQQ to determine if it's appropriate to
    go long or short based on broader market context.
    """
    
    MARKET_INDICES = ['SPY', 'QQQ']
    
    def __init__(self, config: Dict):
        """
        Initialize market context analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tech_analyzer = TechnicalAnalyzer(config)
    
    def analyze_market(self, fetcher) -> MarketContext:
        """
        Analyze overall market conditions.
        
        Args:
            fetcher: DataFetcher instance
            
        Returns:
            MarketContext with current market analysis
        """
        reasons = []
        trends = {}
        
        # Analyze SPY and QQQ
        for symbol in self.MARKET_INDICES:
            try:
                data = fetcher.get_stock_data(symbol, period="3mo", interval="1d")
                if data is not None and len(data) >= 50:
                    analysis = self.tech_analyzer.analyze(data)
                    if analysis:
                        trends[symbol] = self._determine_trend(analysis)
            except Exception as e:
                reasons.append(f"Failed to analyze {symbol}: {str(e)}")
        
        spy_trend = trends.get('SPY', {}).get('trend', 'NEUTRAL')
        qqq_trend = trends.get('QQQ', {}).get('trend', 'NEUTRAL')
        
        spy_strength = trends.get('SPY', {}).get('strength', 50)
        qqq_strength = trends.get('QQQ', {}).get('strength', 50)
        
        # Calculate overall market strength
        market_strength = (spy_strength + qqq_strength) / 2
        
        # Determine regime
        if spy_trend == 'BULLISH' and qqq_trend == 'BULLISH':
            regime = MarketRegime.BULL_MARKET
            reasons.append("Both SPY and QQQ in uptrend - Bull Market")
        elif spy_trend == 'BEARISH' and qqq_trend == 'BEARISH':
            regime = MarketRegime.BEAR_MARKET
            reasons.append("Both SPY and QQQ in downtrend - Bear Market")
        elif market_strength < 20:
            regime = MarketRegime.CRASH
            reasons.append("Extreme weakness detected - Possible crash")
        else:
            regime = MarketRegime.SIDEWAYS
            reasons.append("Mixed signals - Sideways/Choppy market")
        
        # Check volatility
        spy_data = fetcher.get_stock_data('SPY', period="1mo", interval="1d")
        volatility = self._calculate_volatility(spy_data) if spy_data is not None else 'NORMAL'
        
        if volatility == 'EXTREME':
            regime = MarketRegime.HIGH_VOLATILITY
            reasons.append("Extreme volatility - Exercise caution")
        
        # Determine safe trading directions
        safe_to_long = regime in [MarketRegime.BULL_MARKET, MarketRegime.SIDEWAYS]
        safe_to_short = regime in [MarketRegime.BEAR_MARKET, MarketRegime.SIDEWAYS, 
                                   MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]
        
        # Calculate sentiment adjustment
        if regime == MarketRegime.BULL_MARKET:
            sentiment_adjustment = 10  # Boost longs
        elif regime == MarketRegime.BEAR_MARKET:
            sentiment_adjustment = -10  # Reduce long confidence
        elif regime == MarketRegime.CRASH:
            sentiment_adjustment = -20  # Strongly reduce long confidence
        else:
            sentiment_adjustment = 0
        
        # Breadth check (simplified - based on SPY/QQQ alignment)
        breadth_positive = spy_trend == 'BULLISH' or qqq_trend == 'BULLISH'
        
        return MarketContext(
            regime=regime,
            spy_trend=spy_trend,
            qqq_trend=qqq_trend,
            market_strength=market_strength,
            volatility_level=volatility,
            breadth_positive=breadth_positive,
            safe_to_long=safe_to_long,
            safe_to_short=safe_to_short,
            reasons=reasons,
            sentiment_adjustment=sentiment_adjustment
        )
    
    def _determine_trend(self, analysis: Dict) -> Dict:
        """Determine trend from technical analysis."""
        rsi = analysis.get('rsi', 50)
        above_sma20 = analysis.get('above_sma_short', False)
        above_sma50 = analysis.get('above_sma_long', False)
        macd_bullish = analysis.get('macd_histogram', 0) > 0
        price_change = analysis.get('price_change_pct', 0)
        
        bullish_signals = sum([
            1 if above_sma20 else 0,
            1 if above_sma50 else 0,
            1 if macd_bullish else 0,
            1 if rsi > 50 else 0,
            1 if price_change > 0 else 0
        ])
        
        if bullish_signals >= 4:
            trend = 'BULLISH'
            strength = min(100, 60 + bullish_signals * 8)
        elif bullish_signals <= 1:
            trend = 'BEARISH'
            strength = max(0, 40 - (4 - bullish_signals) * 8)
        else:
            trend = 'NEUTRAL'
            strength = 50
        
        return {'trend': trend, 'strength': strength}
    
    def _calculate_volatility(self, data: pd.DataFrame) -> str:
        """Calculate volatility level."""
        if data is None or len(data) < 20:
            return 'NORMAL'
        
        # Calculate daily returns volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * 100  # Convert to percentage
        
        if volatility > 3:
            return 'EXTREME'
        elif volatility > 2:
            return 'HIGH'
        elif volatility > 1:
            return 'NORMAL'
        else:
            return 'LOW'
    
    def filter_signal(
        self,
        signal_type: str,
        market_context: MarketContext
    ) -> tuple:
        """
        Check if a signal is valid given market context.
        
        Args:
            signal_type: BUY or SELL
            market_context: Current market analysis
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if signal_type == 'BUY':
            if not market_context.safe_to_long:
                return False, f"Long signals blocked - {market_context.regime.value}"
            return True, "Market supports long positions"
        
        elif signal_type == 'SELL':
            if not market_context.safe_to_short:
                return False, f"Short signals blocked - {market_context.regime.value}"
            return True, "Market supports short positions"
        
        return True, "HOLD signal not affected by market context"
