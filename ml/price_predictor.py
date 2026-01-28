"""
Price Prediction Module using sklearn.

Uses GradientBoostingRegressor with engineered features for lightweight
stock price prediction. Runs fast on 2GB RAM.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')


@dataclass
class PricePrediction:
    """
    Container for price prediction results.
    """
    ticker: str
    current_price: float
    predicted_prices: List[float]
    predicted_change_pct: float
    confidence_low: float
    confidence_high: float
    confidence_score: float  # 0-100%
    direction: str
    model_available: bool = True
    features_used: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'current_price': self.current_price,
            'predicted_prices': self.predicted_prices,
            'predicted_change_pct': self.predicted_change_pct,
            'confidence_low': self.confidence_low,
            'confidence_high': self.confidence_high,
            'confidence_score': self.confidence_score,
            'direction': self.direction,
            'model_available': self.model_available,
        }


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ML features from OHLCV data.

    Returns a DataFrame of features aligned to the input index.
    All NaN rows at the head are dropped.
    """
    close = df['close'] if 'close' in df.columns else df['Close']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    volume = df['volume'] if 'volume' in df.columns else df['Volume']

    feat = pd.DataFrame(index=df.index)

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    feat['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat['macd'] = macd_line
    feat['macd_signal'] = signal_line
    feat['macd_hist'] = macd_line - signal_line

    # Bollinger position (0 = lower band, 1 = upper band)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    feat['bb_position'] = (close - lower) / (upper - lower + 1e-10)

    # Volume ratio
    vol_ma20 = volume.rolling(20).mean()
    feat['volume_ratio'] = volume / (vol_ma20 + 1e-10)

    # SMA slopes (normalized)
    sma20_slope = sma20.diff(5) / (close + 1e-10) * 100
    sma50 = close.rolling(50).mean()
    sma50_slope = sma50.diff(5) / (close + 1e-10) * 100
    feat['sma20_slope'] = sma20_slope
    feat['sma50_slope'] = sma50_slope

    # Price relative to SMA
    feat['price_sma20_ratio'] = close / (sma20 + 1e-10)
    feat['price_sma50_ratio'] = close / (sma50 + 1e-10)

    # Momentum (rate of change)
    feat['momentum_5'] = close.pct_change(5) * 100
    feat['momentum_10'] = close.pct_change(10) * 100
    feat['momentum_20'] = close.pct_change(20) * 100

    # Recent returns
    feat['return_1d'] = close.pct_change(1) * 100
    feat['return_3d'] = close.pct_change(3) * 100
    feat['return_5d'] = close.pct_change(5) * 100

    # Volatility
    feat['volatility_10'] = close.pct_change().rolling(10).std() * 100
    feat['volatility_20'] = close.pct_change().rolling(20).std() * 100

    # Average True Range (normalized)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean() / (close + 1e-10) * 100

    # Day of week (0-4)
    if hasattr(df.index, 'dayofweek'):
        feat['day_of_week'] = df.index.dayofweek

    feat.dropna(inplace=True)
    return feat


class PricePredictor:
    """
    Price prediction using sklearn GradientBoostingRegressor.

    Trains on last 6 months of data per stock, predicts next-day returns,
    and provides a confidence score.
    """

    FEATURE_COLS = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'volume_ratio',
        'sma20_slope', 'sma50_slope',
        'price_sma20_ratio', 'price_sma50_ratio',
        'momentum_5', 'momentum_10', 'momentum_20',
        'return_1d', 'return_3d', 'return_5d',
        'volatility_10', 'volatility_20', 'atr_14',
    ]

    def __init__(self, config: Dict):
        self.config = config
        self.ml_config = config.get('ml', {})
        self.enabled = self.ml_config.get('enabled', True)

        pred_config = self.ml_config.get('price_prediction', {})
        self.model_type = pred_config.get('model', 'gradient_boosting')
        self.weight = pred_config.get('weight', 0.30)
        self.lookback_days = pred_config.get('lookback_days', 180)

        self._models: Dict[str, object] = {}  # ticker -> trained model
        self._model_scores: Dict[str, float] = {}
        self.model_loaded = False
        self._load_error = None

        os.makedirs(MODEL_DIR, exist_ok=True)

    def _get_model_path(self, ticker: str) -> str:
        return os.path.join(MODEL_DIR, f'price_{ticker.lower()}.joblib')

    def _build_model(self):
        """Build a fresh sklearn model."""
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )

    def train(self, ticker: str, price_data: pd.DataFrame, force: bool = False) -> bool:
        """
        Train (or load cached) model for a ticker.

        Returns True on success.
        """
        if not self.enabled:
            return False

        import joblib

        model_path = self._get_model_path(ticker)

        # Try loading cached model
        if not force and os.path.exists(model_path):
            try:
                cached = joblib.load(model_path)
                self._models[ticker] = cached['model']
                self._model_scores[ticker] = cached.get('score', 0)
                self.model_loaded = True
                logger.info(f"Loaded cached model for {ticker} (R²={cached.get('score', 0):.3f})")
                return True
            except Exception as e:
                logger.warning(f"Could not load cached model for {ticker}: {e}")

        # Train new model
        try:
            features = _compute_features(price_data)
            close = price_data['close'] if 'close' in price_data.columns else price_data['Close']

            # Target: next-day return (%)
            target = close.pct_change().shift(-1) * 100
            target = target.reindex(features.index).dropna()
            features = features.loc[target.index]

            if len(features) < 60:
                logger.warning(f"Not enough data to train {ticker}: {len(features)} rows")
                return False

            # Use available feature columns
            available = [c for c in self.FEATURE_COLS if c in features.columns]
            X = features[available].values
            y = target.values

            # Train/test split (last 20% for validation)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = self._build_model()
            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)
            self._models[ticker] = model
            self._model_scores[ticker] = score
            self.model_loaded = True

            # Cache
            joblib.dump({'model': model, 'score': score, 'features': available}, model_path)
            logger.info(f"Trained model for {ticker}: R²={score:.3f}")
            return True

        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Training error for {ticker}: {e}")
            return False

    def is_available(self) -> bool:
        return self.enabled and self.model_loaded

    def predict(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        prediction_days: Optional[int] = None
    ) -> Optional[PricePrediction]:
        """
        Predict next-day price direction and magnitude.
        """
        if price_data is None or len(price_data) < 60:
            return None

        # Auto-train if needed
        if ticker not in self._models:
            if not self.train(ticker, price_data):
                return self._fallback_prediction(ticker, price_data)

        try:
            features = _compute_features(price_data)
            if len(features) == 0:
                return self._fallback_prediction(ticker, price_data)

            available = [c for c in self.FEATURE_COLS if c in features.columns]
            X_latest = features[available].iloc[-1:].values

            model = self._models[ticker]
            predicted_return = model.predict(X_latest)[0]

            close = price_data['close'] if 'close' in price_data.columns else price_data['Close']
            current_price = float(close.iloc[-1])

            predicted_price = current_price * (1 + predicted_return / 100)

            # Confidence from model R² and prediction magnitude
            r2 = max(0, self._model_scores.get(ticker, 0))
            magnitude = abs(predicted_return)

            # Confidence: higher R² + stronger signal = more confident
            raw_conf = (r2 * 60) + min(magnitude * 10, 40)
            confidence = max(10, min(95, raw_conf))

            # Volatility for confidence interval
            returns = close.pct_change().dropna()
            daily_vol = returns.tail(20).std() * 100
            conf_low = predicted_return - 1.5 * daily_vol
            conf_high = predicted_return + 1.5 * daily_vol

            if predicted_return > 0.5:
                direction = "UP"
            elif predicted_return < -0.5:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            # Multi-day prediction (simple extrapolation with decay)
            days = prediction_days or 5
            predicted_prices = []
            p = current_price
            for d in range(1, days + 1):
                decay = 1.0 / (1 + 0.2 * (d - 1))
                p = p * (1 + (predicted_return * decay) / 100)
                predicted_prices.append(round(p, 2))

            return PricePrediction(
                ticker=ticker,
                current_price=current_price,
                predicted_prices=predicted_prices,
                predicted_change_pct=predicted_return,
                confidence_low=conf_low,
                confidence_high=conf_high,
                confidence_score=confidence,
                direction=direction,
                model_available=True,
                features_used=available,
            )

        except Exception as e:
            logger.error(f"Prediction error for {ticker}: {e}")
            return self._fallback_prediction(ticker, price_data)

    def _fallback_prediction(self, ticker: str, price_data: pd.DataFrame) -> Optional[PricePrediction]:
        """Momentum-based fallback when model unavailable."""
        if price_data is None or len(price_data) < 20:
            return None

        try:
            close = price_data['close'] if 'close' in price_data.columns else price_data['Close']
            prices = close.values
            current_price = float(prices[-1])

            momentum_10d = ((prices[-1] - prices[-10]) / prices[-10]) * 100
            returns = np.diff(prices[-21:]) / prices[-21:-1]
            volatility = np.std(returns) * 100 * np.sqrt(5)

            daily_return = momentum_10d / 10
            predicted_5d = current_price * (1 + daily_return * 5 / 100)
            change_pct = ((predicted_5d - current_price) / current_price) * 100

            if change_pct > 0.5:
                direction = "UP"
            elif change_pct < -0.5:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            return PricePrediction(
                ticker=ticker,
                current_price=current_price,
                predicted_prices=[round(predicted_5d, 2)],
                predicted_change_pct=change_pct,
                confidence_low=change_pct - volatility,
                confidence_high=change_pct + volatility,
                confidence_score=25.0,  # low confidence for fallback
                direction=direction,
                model_available=False,
            )
        except Exception as e:
            logger.error(f"Fallback prediction error for {ticker}: {e}")
            return None

    def get_signal_adjustment(self, prediction: PricePrediction) -> Tuple[float, str]:
        """Get probability adjustment based on prediction."""
        if prediction is None:
            return 0.0, ""

        change = prediction.predicted_change_pct
        conf_mult = prediction.confidence_score / 100.0

        if change >= 3:
            adj = 10.0 * conf_mult
            reason = f"ML predicts strong upside (+{change:.1f}%, conf {prediction.confidence_score:.0f}%)"
        elif change >= 1:
            adj = 5.0 * conf_mult
            reason = f"ML predicts upside (+{change:.1f}%, conf {prediction.confidence_score:.0f}%)"
        elif change <= -3:
            adj = -10.0 * conf_mult
            reason = f"ML predicts strong downside ({change:.1f}%, conf {prediction.confidence_score:.0f}%)"
        elif change <= -1:
            adj = -5.0 * conf_mult
            reason = f"ML predicts downside ({change:.1f}%, conf {prediction.confidence_score:.0f}%)"
        else:
            adj = 0.0
            reason = f"ML predicts sideways ({change:+.1f}%)"

        return adj, reason

    def train_all(self, tickers: List[str], fetcher, force: bool = False) -> Dict[str, bool]:
        """Train models for all tickers. Returns dict of ticker -> success."""
        results = {}
        for ticker in tickers:
            try:
                data = fetcher.get_stock_data(ticker, period="1y")
                if data is not None and len(data) >= 60:
                    results[ticker] = self.train(ticker, data, force=force)
                else:
                    results[ticker] = False
                    logger.warning(f"Insufficient data for {ticker}")
            except Exception as e:
                results[ticker] = False
                logger.error(f"Error training {ticker}: {e}")
        return results

    def get_status(self) -> Dict:
        return {
            'enabled': self.enabled,
            'model': self.model_type,
            'backend': 'sklearn',
            'loaded': self.model_loaded,
            'trained_tickers': list(self._models.keys()),
            'error': self._load_error,
            'weight': self.weight,
        }
