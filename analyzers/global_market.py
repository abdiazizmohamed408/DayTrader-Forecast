"""
Global Market Analyzer.
Tracks global market indicators and their correlation with trading signals.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


class MarketSentiment(Enum):
    """Overall market sentiment."""
    RISK_ON = "RISK_ON"       # Bullish, risk appetite high
    RISK_OFF = "RISK_OFF"     # Bearish, flight to safety
    NEUTRAL = "NEUTRAL"       # Mixed signals
    UNCERTAIN = "UNCERTAIN"   # High volatility, unclear direction


@dataclass
class GlobalIndicator:
    """Single global market indicator."""
    symbol: str
    name: str
    current_price: float
    change_pct: float
    signal: str  # BULLISH, BEARISH, NEUTRAL
    weight: float = 1.0


@dataclass
class GlobalMarketContext:
    """Complete global market context."""
    sentiment: MarketSentiment
    indicators: List[GlobalIndicator] = field(default_factory=list)
    vix_level: float = 0
    dxy_trend: str = "NEUTRAL"
    treasury_10y: float = 0
    risk_score: float = 50  # 0-100, higher = more risk-off
    confidence_adjustment: float = 0
    summary: str = ""


class GlobalMarketAnalyzer:
    """
    Analyzes global market indicators for trading context.
    
    Tracks:
    - VIX (Volatility Index)
    - DXY (US Dollar Index)
    - 10-Year Treasury Yield
    - Gold (GC=F)
    - Oil (CL=F)
    - Major indices (SPY, QQQ, IWM)
    """
    
    # Global market indicators to track
    INDICATORS = {
        # Volatility
        '^VIX': {'name': 'VIX', 'type': 'volatility', 'weight': 1.5},
        
        # Dollar strength
        'DX-Y.NYB': {'name': 'DXY', 'type': 'currency', 'weight': 1.0},
        
        # Treasury
        '^TNX': {'name': '10Y Treasury', 'type': 'bond', 'weight': 1.0},
        
        # Commodities
        'GC=F': {'name': 'Gold', 'type': 'commodity', 'weight': 0.8},
        'CL=F': {'name': 'Crude Oil', 'type': 'commodity', 'weight': 0.7},
        
        # Major indices
        'SPY': {'name': 'S&P 500', 'type': 'index', 'weight': 1.2},
        'QQQ': {'name': 'Nasdaq 100', 'type': 'index', 'weight': 1.0},
        'IWM': {'name': 'Russell 2000', 'type': 'index', 'weight': 0.8},
    }
    
    # VIX thresholds
    VIX_LOW = 15      # Low volatility, complacent
    VIX_NORMAL = 20   # Normal volatility
    VIX_HIGH = 25     # Elevated volatility
    VIX_EXTREME = 35  # Extreme fear
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.cache: Dict = {}
    
    def analyze_global_context(self) -> GlobalMarketContext:
        """
        Analyze all global market indicators.
        
        Returns:
            GlobalMarketContext with complete analysis
        """
        indicators = []
        risk_signals = []
        bullish_count = 0
        bearish_count = 0
        total_weight = 0
        
        vix_level = 20  # Default
        dxy_trend = "NEUTRAL"
        treasury_10y = 4.0  # Default
        
        for symbol, info in self.INDICATORS.items():
            try:
                data = self._get_indicator_data(symbol)
                if data is None:
                    continue
                
                current = data['close'].iloc[-1]
                prev = data['close'].iloc[-5] if len(data) > 5 else data['close'].iloc[0]
                change_pct = ((current - prev) / prev) * 100
                
                # Determine signal based on indicator type
                signal, risk_contribution = self._interpret_indicator(
                    symbol, info, current, change_pct, data
                )
                
                indicator = GlobalIndicator(
                    symbol=symbol,
                    name=info['name'],
                    current_price=current,
                    change_pct=change_pct,
                    signal=signal,
                    weight=info['weight']
                )
                indicators.append(indicator)
                
                # Count signals
                weight = info['weight']
                total_weight += weight
                if signal == "BULLISH":
                    bullish_count += weight
                elif signal == "BEARISH":
                    bearish_count += weight
                
                risk_signals.append(risk_contribution)
                
                # Capture specific values
                if symbol == '^VIX':
                    vix_level = current
                elif symbol == 'DX-Y.NYB':
                    dxy_trend = "STRONG" if change_pct > 0.5 else ("WEAK" if change_pct < -0.5 else "NEUTRAL")
                elif symbol == '^TNX':
                    treasury_10y = current
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        # Calculate overall sentiment
        sentiment = self._determine_sentiment(
            bullish_count, bearish_count, total_weight, vix_level
        )
        
        # Calculate risk score (0-100, higher = more risk-off)
        risk_score = 50 + sum(risk_signals) / max(len(risk_signals), 1)
        risk_score = max(0, min(100, risk_score))
        
        # Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(
            sentiment, vix_level, risk_score
        )
        
        # Generate summary
        summary = self._generate_summary(
            sentiment, vix_level, dxy_trend, indicators
        )
        
        return GlobalMarketContext(
            sentiment=sentiment,
            indicators=indicators,
            vix_level=vix_level,
            dxy_trend=dxy_trend,
            treasury_10y=treasury_10y,
            risk_score=risk_score,
            confidence_adjustment=confidence_adjustment,
            summary=summary
        )
    
    def _get_indicator_data(
        self, symbol: str, period: str = "1mo"
    ) -> Optional[pd.DataFrame]:
        """Get data for an indicator with caching."""
        cache_key = f"{symbol}_{period}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            self.cache[cache_key] = data
            return data
            
        except Exception:
            return None
    
    def _interpret_indicator(
        self,
        symbol: str,
        info: Dict,
        current: float,
        change_pct: float,
        data: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Interpret an indicator's signal and risk contribution.
        
        Returns:
            Tuple of (signal, risk_contribution)
        """
        signal = "NEUTRAL"
        risk_contribution = 0
        
        if symbol == '^VIX':
            # VIX interpretation (inverted - high VIX = bearish)
            if current > self.VIX_EXTREME:
                signal = "BEARISH"
                risk_contribution = 30
            elif current > self.VIX_HIGH:
                signal = "BEARISH"
                risk_contribution = 15
            elif current < self.VIX_LOW:
                signal = "BULLISH"
                risk_contribution = -10
            else:
                signal = "NEUTRAL"
                
        elif info['type'] == 'index':
            # Index interpretation
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            if current > sma_20 and change_pct > 0:
                signal = "BULLISH"
                risk_contribution = -5
            elif current < sma_20 and change_pct < 0:
                signal = "BEARISH"
                risk_contribution = 10
                
        elif symbol == 'GC=F':
            # Gold - risk-off indicator
            if change_pct > 1:
                signal = "BEARISH"  # Gold up = risk-off = bearish for stocks
                risk_contribution = 10
            elif change_pct < -1:
                signal = "BULLISH"
                risk_contribution = -5
                
        elif symbol == 'DX-Y.NYB':
            # Dollar strength
            if change_pct > 0.5:
                signal = "NEUTRAL"  # Strong dollar can be mixed
                risk_contribution = 5
            elif change_pct < -0.5:
                signal = "BULLISH"  # Weak dollar often bullish for stocks
                risk_contribution = -5
                
        elif symbol == '^TNX':
            # 10Y Treasury - rising yields can pressure stocks
            if change_pct > 5:  # Rapid yield rise
                signal = "BEARISH"
                risk_contribution = 15
            elif change_pct < -5:
                signal = "BULLISH"
                risk_contribution = -10
        
        return signal, risk_contribution
    
    def _determine_sentiment(
        self,
        bullish_count: float,
        bearish_count: float,
        total_weight: float,
        vix_level: float
    ) -> MarketSentiment:
        """Determine overall market sentiment."""
        if total_weight == 0:
            return MarketSentiment.UNCERTAIN
        
        bullish_ratio = bullish_count / total_weight
        bearish_ratio = bearish_count / total_weight
        
        # High VIX overrides other signals
        if vix_level > self.VIX_EXTREME:
            return MarketSentiment.RISK_OFF
        
        if vix_level > self.VIX_HIGH:
            return MarketSentiment.UNCERTAIN
        
        if bullish_ratio > 0.6:
            return MarketSentiment.RISK_ON
        elif bearish_ratio > 0.6:
            return MarketSentiment.RISK_OFF
        else:
            return MarketSentiment.NEUTRAL
    
    def _calculate_confidence_adjustment(
        self,
        sentiment: MarketSentiment,
        vix_level: float,
        risk_score: float
    ) -> float:
        """Calculate confidence adjustment for signals."""
        adjustment = 0
        
        # VIX-based adjustment
        if vix_level > self.VIX_EXTREME:
            adjustment -= 15
        elif vix_level > self.VIX_HIGH:
            adjustment -= 10
        elif vix_level < self.VIX_LOW:
            adjustment += 5
        
        # Sentiment adjustment
        if sentiment == MarketSentiment.RISK_ON:
            adjustment += 5  # Bullish for buy signals
        elif sentiment == MarketSentiment.RISK_OFF:
            adjustment -= 5
        elif sentiment == MarketSentiment.UNCERTAIN:
            adjustment -= 10
        
        return adjustment
    
    def _generate_summary(
        self,
        sentiment: MarketSentiment,
        vix_level: float,
        dxy_trend: str,
        indicators: List[GlobalIndicator]
    ) -> str:
        """Generate human-readable summary."""
        parts = []
        
        # Sentiment
        sentiment_text = {
            MarketSentiment.RISK_ON: "🟢 Risk-On (Bullish)",
            MarketSentiment.RISK_OFF: "🔴 Risk-Off (Bearish)",
            MarketSentiment.NEUTRAL: "🟡 Neutral",
            MarketSentiment.UNCERTAIN: "⚠️ Uncertain (High Volatility)"
        }
        parts.append(f"Market: {sentiment_text.get(sentiment, 'Unknown')}")
        
        # VIX
        if vix_level > self.VIX_EXTREME:
            parts.append(f"VIX: {vix_level:.1f} 🚨 EXTREME FEAR")
        elif vix_level > self.VIX_HIGH:
            parts.append(f"VIX: {vix_level:.1f} ⚠️ Elevated")
        elif vix_level < self.VIX_LOW:
            parts.append(f"VIX: {vix_level:.1f} 😴 Complacent")
        else:
            parts.append(f"VIX: {vix_level:.1f}")
        
        # Dollar
        parts.append(f"USD: {dxy_trend}")
        
        return " | ".join(parts)
    
    def get_vix_signal(self) -> Dict:
        """Get detailed VIX analysis."""
        data = self._get_indicator_data('^VIX', '3mo')
        if data is None:
            return {'level': 20, 'trend': 'UNKNOWN', 'signal': 'NEUTRAL'}
        
        current = data['close'].iloc[-1]
        prev_week = data['close'].iloc[-5] if len(data) > 5 else current
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        
        trend = "RISING" if current > prev_week else "FALLING"
        
        if current > self.VIX_EXTREME:
            signal = "EXTREME_FEAR"
        elif current > self.VIX_HIGH:
            signal = "FEAR"
        elif current > self.VIX_NORMAL:
            signal = "CAUTION"
        elif current < self.VIX_LOW:
            signal = "COMPLACENT"
        else:
            signal = "NORMAL"
        
        return {
            'level': current,
            'trend': trend,
            'signal': signal,
            'sma_20': sma_20,
            'above_sma': current > sma_20
        }
    
    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """Check if global conditions suggest reducing exposure."""
        context = self.analyze_global_context()
        
        if context.sentiment == MarketSentiment.RISK_OFF:
            return True, "Risk-off sentiment across global markets"
        
        if context.vix_level > self.VIX_HIGH:
            return True, f"VIX elevated at {context.vix_level:.1f}"
        
        if context.risk_score > 70:
            return True, f"High risk score: {context.risk_score:.0f}"
        
        return False, "Global conditions acceptable"
    
    def format_global_context(self, context: GlobalMarketContext) -> str:
        """Format global context for display."""
        lines = []
        lines.append("GLOBAL MARKET INDICATORS")
        lines.append("═" * 50)
        lines.append(context.summary)
        lines.append("")
        
        lines.append(f"{'INDICATOR':<15} {'PRICE':>10} {'CHANGE':>8} {'SIGNAL':>10}")
        lines.append("─" * 50)
        
        for ind in context.indicators:
            change_str = f"{ind.change_pct:+.1f}%"
            signal_emoji = "🟢" if ind.signal == "BULLISH" else ("🔴" if ind.signal == "BEARISH" else "⚪")
            lines.append(f"{ind.name:<15} {ind.current_price:>10.2f} {change_str:>8} {signal_emoji} {ind.signal}")
        
        return "\n".join(lines)
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
