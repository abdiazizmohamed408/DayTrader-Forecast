"""
Financial Sentiment Analysis Module using VADER.

Lightweight sentiment analysis using VADER (Valence Aware Dictionary and
sEntiment Reasoner) – no GPU, no large models, runs in <10MB RAM.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HeadlineSentiment:
    """Sentiment analysis result for a single headline."""
    headline: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    score: float    # Confidence score 0-1
    source: str
    date: Optional[str] = None


@dataclass
class TickerSentiment:
    """
    Aggregated sentiment analysis for a ticker.
    """
    ticker: str
    overall_score: float        # -1 to +1
    sentiment_label: str        # 'BULLISH', 'BEARISH', 'NEUTRAL'
    headline_count: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    headlines: List[HeadlineSentiment]
    model_available: bool = True

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'overall_score': self.overall_score,
            'sentiment_label': self.sentiment_label,
            'headline_count': self.headline_count,
            'positive_pct': self.positive_pct,
            'negative_pct': self.negative_pct,
            'neutral_pct': self.neutral_pct,
            'model_available': self.model_available,
        }

    def get_emoji(self) -> str:
        if self.overall_score >= 0.3:
            return "🟢"
        elif self.overall_score <= -0.3:
            return "🔴"
        else:
            return "🟡"


class SentimentAnalyzer:
    """
    Financial news sentiment analyzer using VADER.

    VADER is specifically attuned to sentiments expressed in social media
    and works well on financial headlines. Tiny footprint, no GPU needed.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.ml_config = config.get('ml', {})
        self.enabled = self.ml_config.get('enabled', True)

        sent_config = self.ml_config.get('sentiment', {})
        self.engine = sent_config.get('engine', 'vader')
        self.weight = sent_config.get('weight', 0.15)
        self.min_headlines = sent_config.get('min_headlines', 3)

        self.vader = None
        self.model_loaded = False
        self._load_error = None

        if self.enabled:
            self._load_model()

    def _load_model(self):
        """Load VADER sentiment analyzer."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()

            # Extend VADER lexicon with financial terms
            financial_lexicon = {
                'bull': 2.0, 'bullish': 2.5, 'buy': 1.5, 'upgrade': 2.0,
                'outperform': 2.0, 'breakout': 1.8, 'rally': 2.0, 'surge': 2.5,
                'soar': 2.5, 'beat': 1.5, 'record': 1.5, 'strong': 1.5,
                'growth': 1.5, 'profit': 1.5, 'boom': 2.0, 'rocket': 2.5,
                'moon': 2.0, 'undervalued': 1.5,
                'bear': -2.0, 'bearish': -2.5, 'sell': -1.5, 'downgrade': -2.0,
                'underperform': -2.0, 'crash': -3.0, 'plunge': -2.5,
                'tumble': -2.0, 'slump': -2.0, 'miss': -1.5, 'loss': -1.5,
                'weak': -1.5, 'fear': -2.0, 'recession': -2.5, 'layoff': -2.0,
                'layoffs': -2.0, 'bankrupt': -3.0, 'bankruptcy': -3.0,
                'overvalued': -1.5, 'bubble': -2.0, 'default': -2.5,
                'investigation': -1.5, 'lawsuit': -1.5, 'fraud': -3.0,
                'sec': -1.0, 'fine': -1.5, 'penalty': -1.5,
                'dividend': 1.5, 'buyback': 1.5, 'acquisition': 1.0,
                'merger': 0.5, 'ipo': 1.0, 'earnings': 0.5,
                'revenue': 0.5, 'guidance': 0.5,
            }
            self.vader.lexicon.update(financial_lexicon)

            self.model_loaded = True
            logger.info("✅ VADER sentiment analyzer loaded with financial lexicon")

        except ImportError as e:
            self._load_error = f"Missing vaderSentiment: {e}"
            logger.warning(f"⚠️ {self._load_error}")
            logger.warning("Install with: pip install vaderSentiment")

        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"⚠️ Could not load VADER: {e}")

    def is_available(self) -> bool:
        return self.model_loaded and self.vader is not None

    def fetch_news(self, ticker: str, days: int = 3) -> List[Dict]:
        """Fetch recent news headlines for a ticker using yfinance."""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            news = stock.news

            if not news:
                return []

            cutoff = datetime.now() - timedelta(days=days)
            headlines = []

            for item in news[:20]:
                content = item.get('content', item)
                title = content.get('title', '')
                if not title:
                    continue

                pub_time = None
                pub_date_str = content.get('pubDate') or item.get('providerPublishTime')

                if pub_date_str:
                    try:
                        if isinstance(pub_date_str, (int, float)):
                            pub_time = datetime.fromtimestamp(pub_date_str)
                        else:
                            pub_time = datetime.fromisoformat(
                                pub_date_str.replace('Z', '+00:00')
                            )
                    except Exception:
                        pass

                provider = content.get('provider', {})
                if isinstance(provider, dict):
                    publisher = provider.get('displayName', 'Unknown')
                else:
                    publisher = item.get('publisher', 'Unknown')

                headlines.append({
                    'title': title,
                    'publisher': publisher,
                    'date': pub_time.strftime('%Y-%m-%d') if pub_time else None,
                    'link': (
                        content.get('canonicalUrl', {}).get('url', '')
                        if isinstance(content.get('canonicalUrl'), dict)
                        else item.get('link', '')
                    ),
                })

            return headlines

        except Exception as e:
            logger.warning(f"Could not fetch news for {ticker}: {e}")
            return []

    def analyze_headline(self, headline: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a single headline using VADER.

        Returns (sentiment_label, confidence_score).
        """
        if not self.is_available():
            return self._fallback_sentiment(headline)

        try:
            clean = self._clean_text(headline)
            if not clean:
                return 'neutral', 0.5

            scores = self.vader.polarity_scores(clean)
            compound = scores['compound']

            # VADER compound: -1 (most neg) to +1 (most pos)
            if compound >= 0.15:
                sentiment = 'positive'
                confidence = min(1.0, 0.5 + compound * 0.5)
            elif compound <= -0.15:
                sentiment = 'negative'
                confidence = min(1.0, 0.5 + abs(compound) * 0.5)
            else:
                sentiment = 'neutral'
                confidence = 0.5

            return sentiment, confidence

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._fallback_sentiment(headline)

    def _fallback_sentiment(self, headline: str) -> Tuple[str, float]:
        """Keyword-based fallback."""
        headline_lower = headline.lower()

        positive_words = [
            'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'beat', 'exceed',
            'strong', 'growth', 'profit', 'upgrade', 'bullish', 'record', 'high',
            'success', 'positive', 'outperform', 'buy', 'breakout', 'boom',
        ]
        negative_words = [
            'fall', 'drop', 'plunge', 'crash', 'decline', 'loss', 'miss', 'below',
            'weak', 'downgrade', 'bearish', 'low', 'fail', 'negative', 'sell',
            'warning', 'concern', 'risk', 'fear', 'slump', 'tumble', 'cut',
        ]

        pos_count = sum(1 for w in positive_words if w in headline_lower)
        neg_count = sum(1 for w in negative_words if w in headline_lower)

        if pos_count > neg_count:
            return 'positive', 0.6
        elif neg_count > pos_count:
            return 'negative', 0.6
        return 'neutral', 0.5

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return ' '.join(text.split()).strip()

    def analyze_ticker(
        self,
        ticker: str,
        headlines: Optional[List[Dict]] = None,
    ) -> Optional[TickerSentiment]:
        """Analyze overall sentiment for a ticker."""
        if headlines is None:
            headlines = self.fetch_news(ticker)

        if not headlines:
            logger.warning(f"No news headlines found for {ticker}")
            return None

        sentiments: List[HeadlineSentiment] = []
        for item in headlines:
            title = item.get('title', '')
            if not title:
                continue
            sentiment, score = self.analyze_headline(title)
            sentiments.append(HeadlineSentiment(
                headline=title[:100],
                sentiment=sentiment,
                score=score,
                source=item.get('publisher', 'Unknown'),
                date=item.get('date'),
            ))

        if len(sentiments) < self.min_headlines:
            logger.warning(
                f"Insufficient headlines for {ticker}: {len(sentiments)} < {self.min_headlines}"
            )
            return TickerSentiment(
                ticker=ticker,
                overall_score=0.0,
                sentiment_label='NEUTRAL',
                headline_count=len(sentiments),
                positive_pct=0, negative_pct=0, neutral_pct=100,
                headlines=sentiments,
                model_available=self.is_available(),
            )

        positive = sum(1 for s in sentiments if s.sentiment == 'positive')
        negative = sum(1 for s in sentiments if s.sentiment == 'negative')
        total = len(sentiments)

        # Weighted score (more recent = higher weight)
        weighted_score = 0.0
        total_weight = 0.0
        for i, s in enumerate(sentiments):
            weight = 1.0 + (i / total) * 0.5
            if s.sentiment == 'positive':
                weighted_score += s.score * weight
            elif s.sentiment == 'negative':
                weighted_score -= s.score * weight
            total_weight += weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0

        if overall_score >= 0.3:
            label = 'BULLISH'
        elif overall_score <= -0.3:
            label = 'BEARISH'
        else:
            label = 'NEUTRAL'

        return TickerSentiment(
            ticker=ticker,
            overall_score=overall_score,
            sentiment_label=label,
            headline_count=total,
            positive_pct=(positive / total) * 100 if total else 0,
            negative_pct=(negative / total) * 100 if total else 0,
            neutral_pct=((total - positive - negative) / total) * 100 if total else 100,
            headlines=sentiments,
            model_available=self.is_available(),
        )

    def get_signal_adjustment(self, sentiment: TickerSentiment) -> Tuple[float, str]:
        if sentiment is None:
            return 0.0, ""

        score = sentiment.overall_score

        if score >= 0.5:
            return 8.0, f"Strong bullish sentiment ({sentiment.headline_count} headlines)"
        elif score >= 0.3:
            return 4.0, f"Positive sentiment ({sentiment.headline_count} headlines)"
        elif score <= -0.5:
            return -8.0, f"Strong bearish sentiment ({sentiment.headline_count} headlines)"
        elif score <= -0.3:
            return -4.0, f"Negative sentiment ({sentiment.headline_count} headlines)"
        return 0.0, f"Neutral sentiment ({sentiment.headline_count} headlines)"

    def get_status(self) -> Dict:
        return {
            'enabled': self.enabled,
            'engine': self.engine,
            'backend': 'vader',
            'loaded': self.model_loaded,
            'error': self._load_error,
            'weight': self.weight,
        }
