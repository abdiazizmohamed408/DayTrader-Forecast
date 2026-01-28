"""
ML Ensemble Module.

Combines technical analysis with sklearn price prediction, VADER sentiment,
and chart pattern recognition.

Weight distribution:
- Technical Analysis: 40%
- sklearn Price Prediction: 30%
- Sentiment (VADER): 15%
- Pattern Recognition: 15%
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .price_predictor import PricePredictor, PricePrediction
from .sentiment import SentimentAnalyzer, TickerSentiment
from .patterns import PatternRecognizer, PatternAnalysis

logger = logging.getLogger(__name__)


@dataclass
class MLAnalysis:
    """
    Complete ML analysis for a ticker.
    """
    ticker: str
    prediction: Optional[PricePrediction]
    sentiment: Optional[TickerSentiment]
    patterns: Optional[PatternAnalysis]
    combined_adjustment: float
    ml_confidence: float
    reasons: List[str] = field(default_factory=list)
    ai_agrees_with_technical: Optional[bool] = None

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'prediction': self.prediction.to_dict() if self.prediction else None,
            'sentiment': self.sentiment.to_dict() if self.sentiment else None,
            'patterns': self.patterns.to_dict() if self.patterns else None,
            'combined_adjustment': self.combined_adjustment,
            'ml_confidence': self.ml_confidence,
            'reasons': self.reasons,
            'ai_agrees_with_technical': self.ai_agrees_with_technical,
        }

    def get_direction_str(self) -> str:
        if self.prediction:
            change = self.prediction.predicted_change_pct
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            return f"{arrow} {change:+.1f}%"
        return "N/A"

    def get_sentiment_str(self) -> str:
        if self.sentiment:
            emoji = self.sentiment.get_emoji()
            return f"{emoji} {self.sentiment.overall_score:.2f}"
        return "N/A"


class MLEnsemble:
    """
    Ensemble model combining technical analysis with lightweight ML.

    Weight distribution (total ML = 60%, technical = 40%):
    - Technical Analysis: 40% (handled outside this module)
    - sklearn Price Prediction: 30%
    - Sentiment (VADER): 15%
    - Pattern Recognition: 15%
    """

    def __init__(self, config: Dict):
        self.config = config
        self.ml_config = config.get('ml', {})
        self.enabled = self.ml_config.get('enabled', True)

        pred_config = self.ml_config.get('price_prediction', {})
        sent_config = self.ml_config.get('sentiment', {})
        pat_config = self.ml_config.get('patterns', {})

        self.prediction_weight = pred_config.get('weight', 0.30)
        self.sentiment_weight = sent_config.get('weight', 0.15)
        self.pattern_weight = pat_config.get('weight', 0.15)

        self.predictor: Optional[PricePredictor] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.pattern_recognizer: Optional[PatternRecognizer] = None

        if self.enabled:
            self._initialize_components()

    def _initialize_components(self):
        try:
            self.predictor = PricePredictor(self.config)
            logger.info("Price predictor initialized (sklearn)")
        except Exception as e:
            logger.warning(f"Could not initialize price predictor: {e}")

        try:
            self.sentiment_analyzer = SentimentAnalyzer(self.config)
            logger.info("Sentiment analyzer initialized (VADER)")
        except Exception as e:
            logger.warning(f"Could not initialize sentiment analyzer: {e}")

        try:
            self.pattern_recognizer = PatternRecognizer(self.config)
            logger.info("Pattern recognizer initialized (sklearn)")
        except Exception as e:
            logger.warning(f"Could not initialize pattern recognizer: {e}")

    def is_available(self) -> bool:
        predictor_ok = self.predictor is not None
        sentiment_ok = self.sentiment_analyzer and self.sentiment_analyzer.is_available()
        pattern_ok = self.pattern_recognizer is not None
        return self.enabled and (predictor_ok or sentiment_ok or pattern_ok)

    def train_all(self, tickers: List[str], fetcher, force: bool = False) -> Dict:
        """Train all ML models for the given tickers."""
        results = {'price': {}, 'patterns': {}}

        if self.predictor:
            logger.info("Training price prediction models...")
            results['price'] = self.predictor.train_all(tickers, fetcher, force=force)

        if self.pattern_recognizer:
            logger.info("Training pattern recognition models...")
            results['patterns'] = self.pattern_recognizer.train_all(tickers, fetcher, force=force)

        return results

    def analyze(
        self,
        ticker: str,
        price_data,
        technical_signal_type: Optional[str] = None,
    ) -> MLAnalysis:
        """
        Perform complete ML analysis for a ticker.
        """
        prediction = None
        sentiment = None
        patterns = None
        reasons: List[str] = []
        total_adjustment = 0.0
        confidence_factors: List[float] = []

        # 1. Price Prediction (30% weight)
        if self.predictor:
            try:
                prediction = self.predictor.predict(ticker, price_data)
                if prediction:
                    adj, reason = self.predictor.get_signal_adjustment(prediction)
                    scaled_adj = adj * (self.prediction_weight / 0.30)
                    total_adjustment += scaled_adj
                    if reason:
                        reasons.append(reason)

                    if prediction.model_available:
                        confidence_factors.append(
                            min(1.0, prediction.confidence_score / 100)
                        )
                    else:
                        confidence_factors.append(0.3)
            except Exception as e:
                logger.error(f"Price prediction error for {ticker}: {e}")

        # 2. Sentiment Analysis (15% weight)
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer.analyze_ticker(ticker)
                if sentiment and sentiment.headline_count >= self.sentiment_analyzer.min_headlines:
                    adj, reason = self.sentiment_analyzer.get_signal_adjustment(sentiment)
                    scaled_adj = adj * (self.sentiment_weight / 0.15)
                    total_adjustment += scaled_adj
                    if reason:
                        reasons.append(reason)
                    conf = min(1.0, sentiment.headline_count / 10)
                    confidence_factors.append(conf)
            except Exception as e:
                logger.error(f"Sentiment analysis error for {ticker}: {e}")

        # 3. Pattern Recognition (15% weight)
        if self.pattern_recognizer:
            try:
                patterns = self.pattern_recognizer.analyze(ticker, price_data)
                if patterns:
                    adj = patterns.probability_adjustment * (self.pattern_weight / 0.15)
                    total_adjustment += adj
                    reasons.extend(patterns.reasons)

                    if patterns.detected_patterns:
                        confidence_factors.append(0.7)
                    elif patterns.cluster_match and patterns.cluster_match.sample_count >= 5:
                        confidence_factors.append(0.5)
            except Exception as e:
                logger.error(f"Pattern analysis error for {ticker}: {e}")

        # Overall ML confidence
        ml_confidence = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors else 0.0
        )

        # Check if AI agrees with technical signal
        ai_agrees = None
        if technical_signal_type and (prediction or sentiment or patterns):
            ai_bullish = (
                (prediction and prediction.direction == "UP")
                or (sentiment and sentiment.sentiment_label == "BULLISH")
                or (patterns and patterns.combined_score > 0.3)
            )
            ai_bearish = (
                (prediction and prediction.direction == "DOWN")
                or (sentiment and sentiment.sentiment_label == "BEARISH")
                or (patterns and patterns.combined_score < -0.3)
            )

            if technical_signal_type == "BUY":
                ai_agrees = ai_bullish
            elif technical_signal_type == "SELL":
                ai_agrees = ai_bearish
            else:
                ai_agrees = not (ai_bullish or ai_bearish)

            if ai_agrees:
                reasons.append("✅ ML confirms technical signal")
            elif ai_agrees is False:
                reasons.append("⚠️ ML diverges from technical signal")

        # Scale adjustment by confidence
        if ml_confidence < 0.5:
            total_adjustment *= ml_confidence / 0.5

        return MLAnalysis(
            ticker=ticker,
            prediction=prediction,
            sentiment=sentiment,
            patterns=patterns,
            combined_adjustment=total_adjustment,
            ml_confidence=ml_confidence,
            reasons=reasons,
            ai_agrees_with_technical=ai_agrees,
        )

    def enhance_signal_probability(
        self,
        base_probability: float,
        ml_analysis: MLAnalysis,
        signal_type: str,
    ) -> Tuple[float, List[str]]:
        """Enhance signal probability with ML analysis."""
        if not self.enabled or signal_type == "HOLD":
            return base_probability, []

        enhanced_prob = base_probability
        reasons = ml_analysis.reasons.copy()
        adjustment = ml_analysis.combined_adjustment

        if signal_type == "BUY":
            enhanced_prob += adjustment
        elif signal_type == "SELL":
            enhanced_prob -= adjustment

        enhanced_prob = max(0, min(95, enhanced_prob))

        if ml_analysis.ai_agrees_with_technical is False and abs(adjustment) > 5:
            reasons.append("🚨 Strong ML divergence detected")

        return enhanced_prob, reasons

    def get_status(self) -> Dict:
        return {
            'enabled': self.enabled,
            'backend': 'sklearn',
            'predictor': self.predictor.get_status() if self.predictor else None,
            'sentiment': self.sentiment_analyzer.get_status() if self.sentiment_analyzer else None,
            'patterns': self.pattern_recognizer.get_status() if self.pattern_recognizer else None,
            'weights': {
                'prediction': self.prediction_weight,
                'sentiment': self.sentiment_weight,
                'patterns': self.pattern_weight,
            },
        }

    def get_scan_row_data(self, ml_analysis: MLAnalysis) -> Dict:
        return {
            'ai_prediction': ml_analysis.get_direction_str(),
            'sentiment': ml_analysis.get_sentiment_str(),
            'ai_agrees': (
                "✅" if ml_analysis.ai_agrees_with_technical
                else "⚠️" if ml_analysis.ai_agrees_with_technical is False
                else "–"
            ),
        }
