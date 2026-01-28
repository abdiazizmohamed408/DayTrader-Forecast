"""
Signal Generation Module.
Combines technical indicators to generate trading signals with probability scores.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Sentiment(Enum):
    """Market sentiment classification."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class TradingSignal:
    """
    Represents a trading signal with all relevant information.
    
    Attributes:
        ticker: Stock symbol
        signal_type: BUY, SELL, or HOLD
        probability: Confidence score (0-100)
        sentiment: BULLISH, BEARISH, or NEUTRAL
        entry_price: Suggested entry price
        stop_loss: Suggested stop loss price
        target_price: Suggested target price
        risk_reward_ratio: Calculated R:R ratio
        reasons: List of reasons supporting the signal
        volume_confirmed: Whether volume > 1.5x average
        timeframe_alignment: Multi-timeframe alignment score
        market_context: Overall market condition
    """
    ticker: str
    signal_type: SignalType
    probability: float
    sentiment: Sentiment
    entry_price: float
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    reasons: List[str] = None
    volume_confirmed: bool = False
    timeframe_alignment: Optional[float] = None
    market_context: Optional[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'signal_type': self.signal_type.value,
            'probability': self.probability,
            'sentiment': self.sentiment.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'risk_reward_ratio': self.risk_reward_ratio,
            'reasons': self.reasons,
            'volume_confirmed': self.volume_confirmed,
            'timeframe_alignment': self.timeframe_alignment,
            'market_context': self.market_context
        }


class SignalGenerator:
    """
    Generates trading signals based on technical analysis.
    
    Combines multiple indicators using weighted scoring to produce
    probability-based trading signals.
    """
    
    # Volume confirmation threshold (1.5x average)
    VOLUME_THRESHOLD = 1.5
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration settings.
        
        Args:
            config: Dictionary with weights and risk settings
        """
        self.weights = config.get('weights', {
            'rsi': 0.20,
            'macd': 0.20,
            'moving_averages': 0.15,
            'bollinger_bands': 0.15,
            'volume': 0.15,
            'support_resistance': 0.15
        })
        
        self.risk_config = config.get('risk', {
            'stop_loss_percent': 2.0,
            'take_profit_percent': 4.0,
            'risk_reward_ratio': 2.0
        })
        
        # Optional: require volume confirmation for signals
        self.require_volume = config.get('require_volume_confirmation', False)
        
        # Store last calculated scores for tracking
        self.last_scores: Optional[Dict] = None
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update indicator weights (for adaptive learning).
        
        Args:
            new_weights: Dictionary of indicator name to weight
        """
        self.weights.update(new_weights)
    
    def get_last_scores(self) -> Optional[Dict]:
        """Get the indicator scores from the last signal generation."""
        return self.last_scores
    
    def generate_signal(
        self, ticker: str, analysis: Dict
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from technical analysis.
        
        Args:
            ticker: Stock symbol
            analysis: Dictionary from TechnicalAnalyzer.analyze()
            
        Returns:
            TradingSignal object or None if analysis is invalid
        """
        if analysis is None:
            return None
        
        # Calculate individual indicator scores
        scores = self._calculate_scores(analysis)
        self.last_scores = scores  # Store for tracking
        
        # Calculate weighted probability
        bullish_score = sum(
            scores[ind]['bullish'] * self.weights.get(ind, 0)
            for ind in scores
        )
        bearish_score = sum(
            scores[ind]['bearish'] * self.weights.get(ind, 0)
            for ind in scores
        )
        
        # Normalize to percentage
        total_weight = sum(self.weights.values())
        bullish_prob = (bullish_score / total_weight) * 100
        bearish_prob = (bearish_score / total_weight) * 100
        
        # Determine signal type and sentiment
        signal_type, sentiment, probability = self._determine_signal(
            bullish_prob, bearish_prob
        )
        
        # Collect reasons
        reasons = self._collect_reasons(analysis, scores)
        
        # Calculate price targets
        entry_price = analysis['current_price']
        stop_loss, target_price = self._calculate_targets(
            entry_price, signal_type, analysis
        )
        
        # Calculate risk-reward ratio
        risk_reward = None
        if stop_loss and target_price and signal_type == SignalType.BUY:
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
        elif stop_loss and target_price and signal_type == SignalType.SELL:
            risk = stop_loss - entry_price
            reward = entry_price - target_price
            risk_reward = reward / risk if risk > 0 else 0
        
        return TradingSignal(
            ticker=ticker,
            signal_type=signal_type,
            probability=probability,
            sentiment=sentiment,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_reward_ratio=risk_reward,
            reasons=reasons
        )
    
    def _calculate_scores(self, analysis: Dict) -> Dict:
        """
        Calculate bullish/bearish scores for each indicator.
        
        Args:
            analysis: Technical analysis results
            
        Returns:
            Dictionary of indicator scores
        """
        scores = {}
        
        # RSI Score
        rsi = analysis.get('rsi', 50)
        rsi_bullish = 0
        rsi_bearish = 0
        
        if rsi < 30:  # Oversold - bullish
            rsi_bullish = min(1.0, (30 - rsi) / 30 + 0.5)
        elif rsi > 70:  # Overbought - bearish
            rsi_bearish = min(1.0, (rsi - 70) / 30 + 0.5)
        elif rsi < 50:
            rsi_bullish = 0.3
        else:
            rsi_bearish = 0.3
            
        scores['rsi'] = {'bullish': rsi_bullish, 'bearish': rsi_bearish}
        
        # MACD Score
        macd_bullish = 0
        macd_bearish = 0
        
        if analysis.get('macd_crossover'):
            macd_bullish = 1.0
        elif analysis.get('macd_crossunder'):
            macd_bearish = 1.0
        elif analysis.get('macd_histogram', 0) > 0:
            macd_bullish = 0.6
        else:
            macd_bearish = 0.6
            
        scores['macd'] = {'bullish': macd_bullish, 'bearish': macd_bearish}
        
        # Moving Averages Score
        ma_bullish = 0
        ma_bearish = 0
        
        if analysis.get('golden_cross'):
            ma_bullish = 1.0
        elif analysis.get('death_cross'):
            ma_bearish = 1.0
        else:
            if analysis.get('above_sma_short'):
                ma_bullish += 0.3
            else:
                ma_bearish += 0.3
            if analysis.get('above_sma_long'):
                ma_bullish += 0.3
            else:
                ma_bearish += 0.3
                
        scores['moving_averages'] = {'bullish': ma_bullish, 'bearish': ma_bearish}
        
        # Bollinger Bands Score
        bb_bullish = 0
        bb_bearish = 0
        bb_position = analysis.get('bb_position', 'MIDDLE')
        
        if bb_position == 'BELOW_LOWER':
            bb_bullish = 0.8  # Potential bounce
        elif bb_position == 'ABOVE_UPPER':
            bb_bearish = 0.8  # Potential pullback
        elif bb_position == 'LOWER_HALF':
            bb_bullish = 0.4
        elif bb_position == 'UPPER_HALF':
            bb_bearish = 0.4
            
        scores['bollinger_bands'] = {'bullish': bb_bullish, 'bearish': bb_bearish}
        
        # Volume Score
        vol_bullish = 0
        vol_bearish = 0
        volume = analysis.get('volume', {})
        price_change = analysis.get('price_change', 0)
        
        if volume.get('is_high_volume'):
            if price_change > 0:
                vol_bullish = 0.8
            else:
                vol_bearish = 0.8
        elif volume.get('is_low_volume'):
            vol_bullish = 0.2
            vol_bearish = 0.2
        else:
            vol_bullish = 0.4 if price_change > 0 else 0.2
            vol_bearish = 0.4 if price_change < 0 else 0.2
            
        scores['volume'] = {'bullish': vol_bullish, 'bearish': vol_bearish}
        
        # Support/Resistance Score
        sr_bullish = 0
        sr_bearish = 0
        levels = analysis.get('levels', {})
        price = analysis.get('current_price', 0)
        
        if levels:
            s1 = levels.get('support_1', 0)
            r1 = levels.get('resistance_1', 0)
            
            # Near support is bullish
            if price and s1 and abs(price - s1) / price < 0.02:
                sr_bullish = 0.7
            # Near resistance is bearish
            elif price and r1 and abs(price - r1) / price < 0.02:
                sr_bearish = 0.7
            # Above pivot is slightly bullish
            elif price > levels.get('pivot', price):
                sr_bullish = 0.4
            else:
                sr_bearish = 0.4
                
        scores['support_resistance'] = {'bullish': sr_bullish, 'bearish': sr_bearish}
        
        return scores
    
    def _determine_signal(
        self, bullish_prob: float, bearish_prob: float
    ) -> tuple:
        """
        Determine signal type, sentiment, and probability.
        
        Args:
            bullish_prob: Bullish probability percentage
            bearish_prob: Bearish probability percentage
            
        Returns:
            Tuple of (signal_type, sentiment, probability)
        """
        diff = bullish_prob - bearish_prob
        
        if diff > 15:
            signal_type = SignalType.BUY
            sentiment = Sentiment.BULLISH
            probability = min(95, bullish_prob)
        elif diff < -15:
            signal_type = SignalType.SELL
            sentiment = Sentiment.BEARISH
            probability = min(95, bearish_prob)
        else:
            signal_type = SignalType.HOLD
            sentiment = Sentiment.NEUTRAL
            probability = max(bullish_prob, bearish_prob)
        
        return signal_type, sentiment, probability
    
    def _calculate_targets(
        self, entry: float, signal_type: SignalType, analysis: Dict
    ) -> tuple:
        """
        Calculate stop loss and target prices.
        
        Args:
            entry: Entry price
            signal_type: BUY, SELL, or HOLD
            analysis: Technical analysis results
            
        Returns:
            Tuple of (stop_loss, target_price)
        """
        stop_pct = self.risk_config.get('stop_loss_percent', 2.0) / 100
        target_pct = self.risk_config.get('take_profit_percent', 4.0) / 100
        
        levels = analysis.get('levels', {})
        
        if signal_type == SignalType.BUY:
            # Use support as stop loss if available
            s1 = levels.get('support_1')
            if s1 and s1 < entry:
                stop_loss = s1 * 0.99  # Slightly below support
            else:
                stop_loss = entry * (1 - stop_pct)
            
            # Use resistance as target if available
            r1 = levels.get('resistance_1')
            if r1 and r1 > entry:
                target = r1 * 0.99  # Slightly below resistance
            else:
                target = entry * (1 + target_pct)
                
        elif signal_type == SignalType.SELL:
            # Use resistance as stop loss for shorts
            r1 = levels.get('resistance_1')
            if r1 and r1 > entry:
                stop_loss = r1 * 1.01
            else:
                stop_loss = entry * (1 + stop_pct)
            
            # Use support as target
            s1 = levels.get('support_1')
            if s1 and s1 < entry:
                target = s1 * 1.01
            else:
                target = entry * (1 - target_pct)
        else:
            # HOLD signal
            stop_loss = entry * (1 - stop_pct)
            target = entry * (1 + target_pct)
        
        return stop_loss, target
    
    def _collect_reasons(self, analysis: Dict, scores: Dict) -> List[str]:
        """
        Collect human-readable reasons for the signal.
        
        Args:
            analysis: Technical analysis results
            scores: Indicator scores
            
        Returns:
            List of reason strings
        """
        reasons = []
        
        # RSI reasons
        rsi = analysis.get('rsi', 50)
        if rsi < 30:
            reasons.append(f"RSI ({rsi:.1f}) indicates oversold conditions")
        elif rsi > 70:
            reasons.append(f"RSI ({rsi:.1f}) indicates overbought conditions")
        
        # MACD reasons
        if analysis.get('macd_crossover'):
            reasons.append("MACD bullish crossover detected")
        elif analysis.get('macd_crossunder'):
            reasons.append("MACD bearish crossunder detected")
        
        # Moving average reasons
        if analysis.get('golden_cross'):
            reasons.append("Golden cross: short-term MA crossed above long-term MA")
        elif analysis.get('death_cross'):
            reasons.append("Death cross: short-term MA crossed below long-term MA")
        elif analysis.get('above_sma_long'):
            reasons.append("Price trading above long-term moving average (bullish)")
        elif not analysis.get('above_sma_long'):
            reasons.append("Price trading below long-term moving average (bearish)")
        
        # Bollinger Band reasons
        bb_pos = analysis.get('bb_position')
        if bb_pos == 'BELOW_LOWER':
            reasons.append("Price below lower Bollinger Band (potential bounce)")
        elif bb_pos == 'ABOVE_UPPER':
            reasons.append("Price above upper Bollinger Band (potential pullback)")
        
        # Volume reasons
        volume = analysis.get('volume', {})
        if volume.get('is_high_volume'):
            reasons.append(f"High volume ({volume.get('volume_ratio', 1):.1f}x average)")
        
        return reasons
