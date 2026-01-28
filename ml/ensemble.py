"""
ML Ensemble Module.

Combines technical analysis with AI-powered predictions and sentiment
to generate enhanced trading signals.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .price_predictor import PricePredictor, PricePrediction
from .sentiment import SentimentAnalyzer, TickerSentiment

logger = logging.getLogger(__name__)


@dataclass
class MLAnalysis:
    """
    Complete ML analysis for a ticker.
    
    Attributes:
        ticker: Stock symbol
        prediction: AI price prediction result
        sentiment: Sentiment analysis result
        combined_adjustment: Total probability adjustment
        ml_confidence: How confident we are in ML signals (0-1)
        reasons: List of AI-related reasons for signal
        ai_agrees_with_technical: Whether AI agrees with technical signal
    """
    ticker: str
    prediction: Optional[PricePrediction]
    sentiment: Optional[TickerSentiment]
    combined_adjustment: float
    ml_confidence: float
    reasons: List[str]
    ai_agrees_with_technical: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'prediction': self.prediction.to_dict() if self.prediction else None,
            'sentiment': self.sentiment.to_dict() if self.sentiment else None,
            'combined_adjustment': self.combined_adjustment,
            'ml_confidence': self.ml_confidence,
            'reasons': self.reasons,
            'ai_agrees_with_technical': self.ai_agrees_with_technical
        }
    
    def get_direction_str(self) -> str:
        """Get formatted direction string."""
        if self.prediction:
            change = self.prediction.predicted_change_pct
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            return f"{arrow} {change:+.1f}%"
        return "N/A"
    
    def get_sentiment_str(self) -> str:
        """Get formatted sentiment string."""
        if self.sentiment:
            emoji = self.sentiment.get_emoji()
            return f"{emoji} {self.sentiment.overall_score:.2f}"
        return "N/A"


class MLEnsemble:
    """
    Ensemble model combining technical analysis with ML predictions.
    
    Weight distribution:
    - Technical Analysis: 50%
    - AI Price Prediction: 30%  
    - Sentiment Analysis: 20%
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ML ensemble.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get('ml', {})
        self.enabled = self.ml_config.get('enabled', True)
        
        # Get component weights (should sum to 0.5 for ML, leaving 0.5 for technical)
        pred_config = self.ml_config.get('price_prediction', {})
        sent_config = self.ml_config.get('sentiment', {})
        
        self.prediction_weight = pred_config.get('weight', 0.30)
        self.sentiment_weight = sent_config.get('weight', 0.20)
        
        # Initialize components
        self.predictor = None
        self.sentiment_analyzer = None
        
        if self.enabled:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ML components with graceful fallback."""
        try:
            self.predictor = PricePredictor(self.config)
            logger.info("Price predictor initialized")
        except Exception as e:
            logger.warning(f"Could not initialize price predictor: {e}")
            self.predictor = None
        
        try:
            self.sentiment_analyzer = SentimentAnalyzer(self.config)
            logger.info("Sentiment analyzer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def is_available(self) -> bool:
        """Check if any ML component is available."""
        predictor_ok = self.predictor and self.predictor.is_available()
        sentiment_ok = self.sentiment_analyzer and self.sentiment_analyzer.is_available()
        return self.enabled and (predictor_ok or sentiment_ok)
    
    def analyze(
        self,
        ticker: str,
        price_data,
        technical_signal_type: Optional[str] = None
    ) -> MLAnalysis:
        """
        Perform complete ML analysis for a ticker.
        
        Args:
            ticker: Stock symbol
            price_data: DataFrame with OHLCV data
            technical_signal_type: 'BUY', 'SELL', or 'HOLD' from technical analysis
            
        Returns:
            MLAnalysis object with all ML insights
        """
        prediction = None
        sentiment = None
        reasons = []
        total_adjustment = 0.0
        confidence_factors = []
        
        # 1. Price Prediction
        if self.predictor:
            try:
                prediction = self.predictor.predict(ticker, price_data)
                if prediction:
                    adj, reason = self.predictor.get_signal_adjustment(prediction)
                    
                    # Scale adjustment by weight
                    scaled_adj = adj * (self.prediction_weight / 0.30)
                    total_adjustment += scaled_adj
                    
                    if reason:
                        reasons.append(reason)
                    
                    # Track confidence
                    if prediction.model_available:
                        confidence_factors.append(0.9)  # High confidence with real model
                    else:
                        confidence_factors.append(0.5)  # Lower confidence with fallback
                        
            except Exception as e:
                logger.error(f"Price prediction error for {ticker}: {e}")
        
        # 2. Sentiment Analysis
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer.analyze_ticker(ticker)
                if sentiment and sentiment.headline_count >= self.sentiment_analyzer.min_headlines:
                    adj, reason = self.sentiment_analyzer.get_signal_adjustment(sentiment)
                    
                    # Scale adjustment by weight
                    scaled_adj = adj * (self.sentiment_weight / 0.20)
                    total_adjustment += scaled_adj
                    
                    if reason:
                        reasons.append(reason)
                    
                    # Track confidence based on headline count
                    conf = min(1.0, sentiment.headline_count / 10)
                    if sentiment.model_available:
                        confidence_factors.append(conf)
                    else:
                        confidence_factors.append(conf * 0.6)
                        
            except Exception as e:
                logger.error(f"Sentiment analysis error for {ticker}: {e}")
        
        # Calculate overall ML confidence
        ml_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
        
        # Check if AI agrees with technical signal
        ai_agrees = None
        if technical_signal_type and (prediction or sentiment):
            ai_bullish = (
                (prediction and prediction.direction == "UP") or
                (sentiment and sentiment.sentiment_label == "BULLISH")
            )
            ai_bearish = (
                (prediction and prediction.direction == "DOWN") or
                (sentiment and sentiment.sentiment_label == "BEARISH")
            )
            
            if technical_signal_type == "BUY":
                ai_agrees = ai_bullish
            elif technical_signal_type == "SELL":
                ai_agrees = ai_bearish
            else:
                ai_agrees = not (ai_bullish or ai_bearish)
            
            # Add agreement note to reasons
            if ai_agrees:
                reasons.append("✅ AI confirms technical signal")
            elif ai_agrees is False:
                reasons.append("⚠️ AI diverges from technical signal")
        
        # Cap adjustment based on confidence
        if ml_confidence < 0.5:
            total_adjustment *= ml_confidence / 0.5  # Reduce impact if low confidence
        
        return MLAnalysis(
            ticker=ticker,
            prediction=prediction,
            sentiment=sentiment,
            combined_adjustment=total_adjustment,
            ml_confidence=ml_confidence,
            reasons=reasons,
            ai_agrees_with_technical=ai_agrees
        )
    
    def enhance_signal_probability(
        self,
        base_probability: float,
        ml_analysis: MLAnalysis,
        signal_type: str
    ) -> Tuple[float, List[str]]:
        """
        Enhance signal probability with ML analysis.
        
        Args:
            base_probability: Original probability from technical analysis
            ml_analysis: MLAnalysis object
            signal_type: 'BUY', 'SELL', or 'HOLD'
            
        Returns:
            Tuple of (enhanced_probability, list of reasons)
        """
        if not self.enabled or signal_type == "HOLD":
            return base_probability, []
        
        enhanced_prob = base_probability
        reasons = ml_analysis.reasons.copy()
        
        # Apply adjustment based on signal direction
        adjustment = ml_analysis.combined_adjustment
        
        if signal_type == "BUY":
            # Positive adjustment helps BUY, negative hurts
            enhanced_prob += adjustment
        elif signal_type == "SELL":
            # Negative adjustment helps SELL (inverse)
            enhanced_prob -= adjustment
        
        # Cap probability
        enhanced_prob = max(0, min(95, enhanced_prob))
        
        # Add warning if AI strongly disagrees
        if ml_analysis.ai_agrees_with_technical is False:
            if abs(adjustment) > 5:
                reasons.append(f"🚨 Strong AI divergence detected")
        
        return enhanced_prob, reasons
    
    def get_status(self) -> Dict:
        """Get status of all ML components."""
        return {
            'enabled': self.enabled,
            'predictor': self.predictor.get_status() if self.predictor else None,
            'sentiment': self.sentiment_analyzer.get_status() if self.sentiment_analyzer else None,
            'weights': {
                'prediction': self.prediction_weight,
                'sentiment': self.sentiment_weight
            }
        }
    
    def get_scan_row_data(self, ml_analysis: MLAnalysis) -> Dict:
        """
        Get formatted data for scan output table.
        
        Args:
            ml_analysis: MLAnalysis object
            
        Returns:
            Dictionary with formatted strings for display
        """
        return {
            'ai_prediction': ml_analysis.get_direction_str(),
            'sentiment': ml_analysis.get_sentiment_str(),
            'ai_agrees': "✅" if ml_analysis.ai_agrees_with_technical else "⚠️" if ml_analysis.ai_agrees_with_technical is False else "–"
        }
