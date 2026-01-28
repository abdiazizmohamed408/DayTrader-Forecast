"""
Chart Pattern Recognition Module using sklearn.

Detects classic chart patterns (double top/bottom, head-and-shoulders,
breakouts) and uses KMeans clustering to find similar historical price
patterns and their outcomes.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')


@dataclass
class PatternMatch:
    """A detected chart pattern."""
    name: str
    pattern_type: str       # 'bullish', 'bearish', 'neutral'
    confidence: float       # 0-1
    description: str


@dataclass
class ClusterMatch:
    """Result of historical pattern clustering."""
    cluster_id: int
    similarity: float           # 0-1
    avg_forward_return: float   # % return after similar patterns
    win_rate: float             # % of times price went up after pattern
    sample_count: int


@dataclass
class PatternAnalysis:
    """Complete pattern analysis for a ticker."""
    ticker: str
    detected_patterns: List[PatternMatch]
    cluster_match: Optional[ClusterMatch]
    combined_score: float       # -1 (very bearish) to +1 (very bullish)
    probability_adjustment: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'patterns': [
                {'name': p.name, 'type': p.pattern_type, 'confidence': p.confidence}
                for p in self.detected_patterns
            ],
            'cluster': {
                'avg_return': self.cluster_match.avg_forward_return,
                'win_rate': self.cluster_match.win_rate,
                'samples': self.cluster_match.sample_count,
            } if self.cluster_match else None,
            'combined_score': self.combined_score,
        }


def _normalize_window(prices: np.ndarray) -> np.ndarray:
    """Normalize a price window to 0-1 range."""
    mn, mx = prices.min(), prices.max()
    if mx - mn < 1e-10:
        return np.zeros_like(prices)
    return (prices - mn) / (mx - mn)


class PatternRecognizer:
    """
    Chart pattern detection and historical pattern matching.

    Combines rule-based pattern detection with KMeans clustering
    of normalized price windows for probability estimation.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.ml_config = config.get('ml', {})
        self.enabled = self.ml_config.get('enabled', True)
        self.weight = self.ml_config.get('patterns', {}).get('weight', 0.15)

        self._cluster_models: Dict[str, object] = {}
        self._cluster_data: Dict[str, Dict] = {}

        os.makedirs(MODEL_DIR, exist_ok=True)

    # ───────────────────────────────────────────
    # Rule-based pattern detection
    # ───────────────────────────────────────────

    def _detect_double_top(self, prices: np.ndarray) -> Optional[PatternMatch]:
        """Detect double top (bearish reversal)."""
        if len(prices) < 30:
            return None

        window = prices[-30:]
        # Find two peaks in the last 30 days
        peaks = []
        for i in range(2, len(window) - 2):
            if window[i] > window[i-1] and window[i] > window[i+1]:
                if window[i] > window[i-2] and window[i] > min(window[i+1], window[i+2] if i+2 < len(window) else window[i+1]):
                    peaks.append((i, window[i]))

        if len(peaks) < 2:
            return None

        # Check if last two peaks are at similar levels
        p1_idx, p1_val = peaks[-2]
        p2_idx, p2_val = peaks[-1]

        if abs(p2_idx - p1_idx) < 5:  # Too close
            return None

        similarity = 1 - abs(p1_val - p2_val) / max(p1_val, p2_val)
        if similarity > 0.97:
            # Check that there's a dip between them
            valley = min(window[p1_idx:p2_idx+1])
            dip_pct = (max(p1_val, p2_val) - valley) / max(p1_val, p2_val)
            if dip_pct > 0.02:
                return PatternMatch(
                    name='Double Top',
                    pattern_type='bearish',
                    confidence=similarity * min(1.0, dip_pct * 10),
                    description=f'Two peaks at similar levels ({similarity:.0%} match), {dip_pct:.1%} dip between',
                )
        return None

    def _detect_double_bottom(self, prices: np.ndarray) -> Optional[PatternMatch]:
        """Detect double bottom (bullish reversal)."""
        if len(prices) < 30:
            return None

        window = prices[-30:]
        valleys = []
        for i in range(2, len(window) - 2):
            if window[i] < window[i-1] and window[i] < window[i+1]:
                if window[i] < window[i-2]:
                    valleys.append((i, window[i]))

        if len(valleys) < 2:
            return None

        v1_idx, v1_val = valleys[-2]
        v2_idx, v2_val = valleys[-1]

        if abs(v2_idx - v1_idx) < 5:
            return None

        similarity = 1 - abs(v1_val - v2_val) / max(v1_val, v2_val)
        if similarity > 0.97:
            peak = max(window[v1_idx:v2_idx+1])
            bounce_pct = (peak - min(v1_val, v2_val)) / max(v1_val, v2_val)
            if bounce_pct > 0.02:
                return PatternMatch(
                    name='Double Bottom',
                    pattern_type='bullish',
                    confidence=similarity * min(1.0, bounce_pct * 10),
                    description=f'Two troughs at similar levels ({similarity:.0%} match), {bounce_pct:.1%} bounce between',
                )
        return None

    def _detect_head_and_shoulders(self, prices: np.ndarray) -> Optional[PatternMatch]:
        """Detect head-and-shoulders (bearish) pattern."""
        if len(prices) < 40:
            return None

        window = prices[-40:]
        peaks = []
        for i in range(2, len(window) - 2):
            if window[i] > window[i-1] and window[i] > window[i+1]:
                peaks.append((i, window[i]))

        if len(peaks) < 3:
            return None

        # Check last three peaks: middle should be highest
        for j in range(len(peaks) - 2):
            left_i, left_v = peaks[j]
            head_i, head_v = peaks[j+1]
            right_i, right_v = peaks[j+2]

            if head_v > left_v and head_v > right_v:
                # Shoulders at similar levels
                shoulder_sim = 1 - abs(left_v - right_v) / max(left_v, right_v)
                head_prominence = (head_v - max(left_v, right_v)) / head_v

                if shoulder_sim > 0.95 and head_prominence > 0.01:
                    return PatternMatch(
                        name='Head & Shoulders',
                        pattern_type='bearish',
                        confidence=shoulder_sim * min(1.0, head_prominence * 20),
                        description=f'H&S pattern: shoulders {shoulder_sim:.0%} matched, head {head_prominence:.1%} above',
                    )
        return None

    def _detect_breakout(self, prices: np.ndarray, volume: Optional[np.ndarray] = None) -> Optional[PatternMatch]:
        """Detect breakout from consolidation range."""
        if len(prices) < 25:
            return None

        # Check if price was in a tight range for 10-20 days then broke out
        consolidation = prices[-25:-5]
        recent = prices[-5:]

        cons_range = (consolidation.max() - consolidation.min()) / consolidation.mean()
        if cons_range > 0.05:  # Range too wide for consolidation
            return None

        upper = consolidation.max()
        lower = consolidation.min()
        current = recent[-1]

        if current > upper * 1.01:  # Breakout above
            magnitude = (current - upper) / upper
            conf = min(1.0, magnitude * 20)

            # Volume confirmation
            if volume is not None and len(volume) >= 5:
                vol_ratio = volume[-1] / (np.mean(volume[-25:-5]) + 1e-10)
                if vol_ratio > 1.5:
                    conf = min(1.0, conf * 1.3)

            return PatternMatch(
                name='Bullish Breakout',
                pattern_type='bullish',
                confidence=conf,
                description=f'Broke above {cons_range:.1%} consolidation range by {magnitude:.1%}',
            )

        elif current < lower * 0.99:  # Breakdown below
            magnitude = (lower - current) / lower
            conf = min(1.0, magnitude * 20)

            return PatternMatch(
                name='Bearish Breakdown',
                pattern_type='bearish',
                confidence=conf,
                description=f'Broke below {cons_range:.1%} consolidation range by {magnitude:.1%}',
            )

        return None

    def detect_patterns(self, price_data: pd.DataFrame) -> List[PatternMatch]:
        """Run all pattern detectors on price data."""
        close = (price_data['close'] if 'close' in price_data.columns
                 else price_data['Close']).values
        volume = None
        for col in ('volume', 'Volume'):
            if col in price_data.columns:
                volume = price_data[col].values
                break

        patterns = []
        for detector in [
            lambda: self._detect_double_top(close),
            lambda: self._detect_double_bottom(close),
            lambda: self._detect_head_and_shoulders(close),
            lambda: self._detect_breakout(close, volume),
        ]:
            try:
                result = detector()
                if result:
                    patterns.append(result)
            except Exception as e:
                logger.debug(f"Pattern detector error: {e}")

        return patterns

    # ───────────────────────────────────────────
    # KMeans historical pattern matching
    # ───────────────────────────────────────────

    def train_clusters(self, ticker: str, price_data: pd.DataFrame, force: bool = False) -> bool:
        """
        Build KMeans clusters from historical price windows.

        Each window is 20 days of normalized prices.
        We record what happened in the 5 days after each window.
        """
        import joblib
        model_path = os.path.join(MODEL_DIR, f'pattern_{ticker.lower()}.joblib')

        if not force and os.path.exists(model_path):
            try:
                cached = joblib.load(model_path)
                self._cluster_models[ticker] = cached['model']
                self._cluster_data[ticker] = cached['data']
                logger.info(f"Loaded cached pattern clusters for {ticker}")
                return True
            except Exception:
                pass

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            close = (price_data['close'] if 'close' in price_data.columns
                     else price_data['Close']).values

            if len(close) < 80:
                return False

            window_size = 20
            forward_days = 5

            windows = []
            forward_returns = []

            for i in range(window_size, len(close) - forward_days):
                w = close[i - window_size:i]
                norm = _normalize_window(w)
                windows.append(norm)

                future_price = close[i + forward_days - 1]
                current_price = close[i - 1]
                ret = ((future_price - current_price) / current_price) * 100
                forward_returns.append(ret)

            X = np.array(windows)
            returns = np.array(forward_returns)

            n_clusters = min(8, max(3, len(X) // 30))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # Compute per-cluster statistics
            cluster_stats = {}
            for c in range(n_clusters):
                mask = labels == c
                c_returns = returns[mask]
                cluster_stats[c] = {
                    'avg_return': float(np.mean(c_returns)),
                    'win_rate': float(np.sum(c_returns > 0) / len(c_returns) * 100),
                    'count': int(np.sum(mask)),
                    'std': float(np.std(c_returns)),
                }

            self._cluster_models[ticker] = {'kmeans': kmeans, 'scaler': scaler}
            self._cluster_data[ticker] = cluster_stats

            joblib.dump({
                'model': self._cluster_models[ticker],
                'data': cluster_stats,
            }, model_path)

            logger.info(f"Trained {n_clusters} pattern clusters for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Cluster training error for {ticker}: {e}")
            return False

    def match_current_pattern(self, ticker: str, price_data: pd.DataFrame) -> Optional[ClusterMatch]:
        """
        Match the current price window to historical clusters.
        """
        if ticker not in self._cluster_models:
            if not self.train_clusters(ticker, price_data):
                return None

        try:
            close = (price_data['close'] if 'close' in price_data.columns
                     else price_data['Close']).values

            if len(close) < 20:
                return None

            window = _normalize_window(close[-20:])
            model_info = self._cluster_models[ticker]
            kmeans = model_info['kmeans']
            scaler = model_info['scaler']

            X = scaler.transform(window.reshape(1, -1))
            cluster_id = kmeans.predict(X)[0]

            # Distance to centroid (lower = more similar)
            centroid = kmeans.cluster_centers_[cluster_id]
            distance = np.linalg.norm(X[0] - centroid)
            max_dist = np.max([np.linalg.norm(c) for c in kmeans.cluster_centers_]) + 1
            similarity = max(0, 1 - distance / max_dist)

            stats = self._cluster_data[ticker].get(cluster_id, {})

            return ClusterMatch(
                cluster_id=cluster_id,
                similarity=similarity,
                avg_forward_return=stats.get('avg_return', 0),
                win_rate=stats.get('win_rate', 50),
                sample_count=stats.get('count', 0),
            )

        except Exception as e:
            logger.error(f"Cluster matching error for {ticker}: {e}")
            return None

    # ───────────────────────────────────────────
    # Combined analysis
    # ───────────────────────────────────────────

    def analyze(self, ticker: str, price_data: pd.DataFrame) -> PatternAnalysis:
        """
        Full pattern analysis: rule-based patterns + cluster matching.
        """
        detected = self.detect_patterns(price_data) if self.enabled else []
        cluster = self.match_current_pattern(ticker, price_data) if self.enabled else None

        # Combine scores
        pattern_score = 0.0
        reasons = []

        for p in detected:
            if p.pattern_type == 'bullish':
                pattern_score += p.confidence * 0.5
                reasons.append(f"📊 {p.name} detected ({p.confidence:.0%} conf)")
            elif p.pattern_type == 'bearish':
                pattern_score -= p.confidence * 0.5
                reasons.append(f"📊 {p.name} detected ({p.confidence:.0%} conf)")

        if cluster and cluster.sample_count >= 5:
            # Normalize cluster contribution
            cluster_signal = (cluster.win_rate - 50) / 50  # -1 to +1
            cluster_score = cluster_signal * cluster.similarity * 0.5
            pattern_score += cluster_score

            if cluster.win_rate >= 60:
                reasons.append(
                    f"📈 Similar patterns → {cluster.win_rate:.0f}% win rate "
                    f"(avg {cluster.avg_forward_return:+.1f}%, n={cluster.sample_count})"
                )
            elif cluster.win_rate <= 40:
                reasons.append(
                    f"📉 Similar patterns → {cluster.win_rate:.0f}% win rate "
                    f"(avg {cluster.avg_forward_return:+.1f}%, n={cluster.sample_count})"
                )

        combined = max(-1, min(1, pattern_score))

        # Convert to probability adjustment
        if combined > 0.3:
            prob_adj = combined * 8
        elif combined < -0.3:
            prob_adj = combined * 8
        else:
            prob_adj = 0

        return PatternAnalysis(
            ticker=ticker,
            detected_patterns=detected,
            cluster_match=cluster,
            combined_score=combined,
            probability_adjustment=prob_adj,
            reasons=reasons,
        )

    def train_all(self, tickers: List[str], fetcher, force: bool = False) -> Dict[str, bool]:
        """Train cluster models for all tickers."""
        results = {}
        for ticker in tickers:
            try:
                data = fetcher.get_stock_data(ticker, period="1y")
                if data is not None and len(data) >= 80:
                    results[ticker] = self.train_clusters(ticker, data, force=force)
                else:
                    results[ticker] = False
            except Exception as e:
                results[ticker] = False
                logger.error(f"Pattern training error for {ticker}: {e}")
        return results

    def get_status(self) -> Dict:
        return {
            'enabled': self.enabled,
            'weight': self.weight,
            'trained_tickers': list(self._cluster_models.keys()),
        }
