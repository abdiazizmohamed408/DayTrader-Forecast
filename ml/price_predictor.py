"""
Price Prediction Module using Amazon Chronos.

Uses the Chronos-T5 time series foundation model for stock price prediction.
Model: amazon/chronos-t5-small (46M params, runs on CPU)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PricePrediction:
    """
    Container for price prediction results.
    
    Attributes:
        ticker: Stock symbol
        current_price: Current stock price
        predicted_prices: List of predicted prices for next N days
        predicted_change_pct: Predicted percentage change
        confidence_low: Lower bound of confidence interval
        confidence_high: Upper bound of confidence interval
        direction: 'UP', 'DOWN', or 'NEUTRAL'
        model_available: Whether the model loaded successfully
    """
    ticker: str
    current_price: float
    predicted_prices: List[float]
    predicted_change_pct: float
    confidence_low: float
    confidence_high: float
    direction: str
    model_available: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'current_price': self.current_price,
            'predicted_prices': self.predicted_prices,
            'predicted_change_pct': self.predicted_change_pct,
            'confidence_low': self.confidence_low,
            'confidence_high': self.confidence_high,
            'direction': self.direction,
            'model_available': self.model_available
        }


class PricePredictor:
    """
    AI-powered price prediction using Amazon Chronos.
    
    Chronos is a family of pretrained time series forecasting models
    that can generate probabilistic forecasts for unseen time series.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the price predictor.
        
        Args:
            config: Configuration dictionary with ML settings
        """
        self.config = config
        self.ml_config = config.get('ml', {})
        self.enabled = self.ml_config.get('enabled', True)
        
        pred_config = self.ml_config.get('price_prediction', {})
        self.model_name = pred_config.get('model', 'amazon/chronos-t5-small')
        self.weight = pred_config.get('weight', 0.30)
        self.prediction_days = pred_config.get('prediction_days', 5)
        self.lookback_days = pred_config.get('lookback_days', 60)
        
        self.pipeline = None
        self.model_loaded = False
        self._load_error = None
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load the Chronos model with graceful fallback."""
        if not self.enabled:
            return
            
        try:
            import torch
            from chronos import ChronosPipeline
            
            logger.info(f"Loading Chronos model: {self.model_name}")
            
            # Use CPU by default, GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=device,
                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
            )
            
            self.model_loaded = True
            logger.info(f"✅ Chronos model loaded successfully on {device}")
            
        except ImportError as e:
            self._load_error = f"Missing dependencies: {e}"
            logger.warning(f"⚠️ Could not load Chronos: {self._load_error}")
            logger.warning("Install with: pip install chronos-forecasting torch")
            
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"⚠️ Could not load Chronos model: {e}")
    
    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model_loaded and self.pipeline is not None
    
    def predict(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        prediction_days: Optional[int] = None
    ) -> Optional[PricePrediction]:
        """
        Predict future prices using Chronos.
        
        Args:
            ticker: Stock symbol
            price_data: DataFrame with OHLCV data (needs 'close' column)
            prediction_days: Number of days to predict (default from config)
            
        Returns:
            PricePrediction object or None if prediction fails
        """
        if not self.is_available():
            return self._fallback_prediction(ticker, price_data)
        
        if price_data is None or len(price_data) < self.lookback_days:
            logger.warning(f"Insufficient data for {ticker}: need {self.lookback_days} days")
            return None
        
        try:
            import torch
            
            days = prediction_days or self.prediction_days
            
            # Extract close prices
            if 'close' in price_data.columns:
                prices = price_data['close'].values
            elif 'Close' in price_data.columns:
                prices = price_data['Close'].values
            else:
                logger.error(f"No close price column found for {ticker}")
                return None
            
            # Use last N days for prediction
            context = prices[-self.lookback_days:]
            context_tensor = torch.tensor(context, dtype=torch.float32)
            
            # Generate forecast with multiple samples for confidence intervals
            forecast = self.pipeline.predict(
                context_tensor,
                prediction_length=days,
                num_samples=20  # Generate 20 samples for confidence intervals
            )
            
            # Convert to numpy
            forecast_np = forecast.numpy()
            
            # Calculate statistics
            median_forecast = np.median(forecast_np, axis=0)
            low_forecast = np.percentile(forecast_np, 10, axis=0)
            high_forecast = np.percentile(forecast_np, 90, axis=0)
            
            current_price = float(prices[-1])
            predicted_price = float(median_forecast[-1])  # Last day prediction
            
            # Calculate change percentage
            change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Determine direction
            if change_pct > 2:
                direction = "UP"
            elif change_pct < -2:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
            
            # Confidence intervals
            conf_low_pct = ((float(low_forecast[-1]) - current_price) / current_price) * 100
            conf_high_pct = ((float(high_forecast[-1]) - current_price) / current_price) * 100
            
            return PricePrediction(
                ticker=ticker,
                current_price=current_price,
                predicted_prices=[float(p) for p in median_forecast],
                predicted_change_pct=change_pct,
                confidence_low=conf_low_pct,
                confidence_high=conf_high_pct,
                direction=direction,
                model_available=True
            )
            
        except Exception as e:
            logger.error(f"Prediction error for {ticker}: {e}")
            return self._fallback_prediction(ticker, price_data)
    
    def _fallback_prediction(
        self,
        ticker: str,
        price_data: pd.DataFrame
    ) -> Optional[PricePrediction]:
        """
        Simple momentum-based fallback when model unavailable.
        
        Uses recent price momentum to estimate direction.
        """
        if price_data is None or len(price_data) < 20:
            return None
        
        try:
            if 'close' in price_data.columns:
                prices = price_data['close'].values
            elif 'Close' in price_data.columns:
                prices = price_data['Close'].values
            else:
                return None
            
            current_price = float(prices[-1])
            
            # Calculate simple momentum (10-day return)
            momentum_10d = ((prices[-1] - prices[-10]) / prices[-10]) * 100
            
            # Calculate volatility for confidence
            returns = np.diff(prices[-20:]) / prices[-21:-1]
            volatility = np.std(returns) * 100 * np.sqrt(5)  # 5-day volatility
            
            # Simple linear extrapolation
            daily_return = momentum_10d / 10
            predicted_5d = current_price * (1 + daily_return * 5 / 100)
            
            change_pct = ((predicted_5d - current_price) / current_price) * 100
            
            if change_pct > 2:
                direction = "UP"
            elif change_pct < -2:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
            
            return PricePrediction(
                ticker=ticker,
                current_price=current_price,
                predicted_prices=[predicted_5d],
                predicted_change_pct=change_pct,
                confidence_low=change_pct - volatility,
                confidence_high=change_pct + volatility,
                direction=direction,
                model_available=False  # Indicates fallback was used
            )
            
        except Exception as e:
            logger.error(f"Fallback prediction error for {ticker}: {e}")
            return None
    
    def get_signal_adjustment(self, prediction: PricePrediction) -> Tuple[float, str]:
        """
        Get probability adjustment based on prediction.
        
        Args:
            prediction: PricePrediction object
            
        Returns:
            Tuple of (adjustment amount, reason string)
        """
        if prediction is None:
            return 0.0, ""
        
        change = prediction.predicted_change_pct
        
        # Strong positive prediction
        if change >= 5:
            return 10.0, f"AI predicts strong upside (+{change:.1f}%)"
        elif change >= 2:
            return 5.0, f"AI predicts upside (+{change:.1f}%)"
        # Strong negative prediction  
        elif change <= -5:
            return -10.0, f"AI predicts strong downside ({change:.1f}%)"
        elif change <= -2:
            return -5.0, f"AI predicts downside ({change:.1f}%)"
        else:
            return 0.0, f"AI predicts sideways ({change:+.1f}%)"
    
    def get_status(self) -> Dict:
        """Get model status information."""
        return {
            'enabled': self.enabled,
            'model': self.model_name,
            'loaded': self.model_loaded,
            'error': self._load_error,
            'weight': self.weight
        }
