"""
Financial Sentiment Analysis Module.

Uses DistilRoBERTa fine-tuned on financial news for sentiment analysis.
Model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re

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
    
    Attributes:
        ticker: Stock symbol
        overall_score: Aggregated sentiment score (-1 to +1)
        sentiment_label: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        headline_count: Number of headlines analyzed
        positive_pct: Percentage of positive headlines
        negative_pct: Percentage of negative headlines
        neutral_pct: Percentage of neutral headlines
        headlines: List of individual headline sentiments
        model_available: Whether the model loaded successfully
    """
    ticker: str
    overall_score: float
    sentiment_label: str
    headline_count: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    headlines: List[HeadlineSentiment]
    model_available: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'overall_score': self.overall_score,
            'sentiment_label': self.sentiment_label,
            'headline_count': self.headline_count,
            'positive_pct': self.positive_pct,
            'negative_pct': self.negative_pct,
            'neutral_pct': self.neutral_pct,
            'model_available': self.model_available
        }
    
    def get_emoji(self) -> str:
        """Get emoji representation of sentiment."""
        if self.overall_score >= 0.3:
            return "🟢"
        elif self.overall_score <= -0.3:
            return "🔴"
        else:
            return "🟡"


class SentimentAnalyzer:
    """
    Financial news sentiment analyzer using Hugging Face transformers.
    
    Uses DistilRoBERTa fine-tuned specifically on financial news
    for accurate sentiment classification.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary with ML settings
        """
        self.config = config
        self.ml_config = config.get('ml', {})
        self.enabled = self.ml_config.get('enabled', True)
        
        sent_config = self.ml_config.get('sentiment', {})
        self.model_name = sent_config.get(
            'model', 
            'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
        )
        self.weight = sent_config.get('weight', 0.20)
        self.min_headlines = sent_config.get('min_headlines', 3)
        
        self.pipeline = None
        self.model_loaded = False
        self._load_error = None
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load the sentiment model with graceful fallback."""
        if not self.enabled:
            return
            
        try:
            from transformers import pipeline
            
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None  # Get all sentiment scores
            )
            
            self.model_loaded = True
            logger.info("✅ Sentiment model loaded successfully")
            
        except ImportError as e:
            self._load_error = f"Missing dependencies: {e}"
            logger.warning(f"⚠️ Could not load sentiment model: {self._load_error}")
            logger.warning("Install with: pip install transformers torch")
            
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"⚠️ Could not load sentiment model: {e}")
    
    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model_loaded and self.pipeline is not None
    
    def fetch_news(self, ticker: str, days: int = 3) -> List[Dict]:
        """
        Fetch recent news headlines for a ticker.
        
        Uses yfinance for free news data.
        
        Args:
            ticker: Stock symbol
            days: Number of days of news to fetch
            
        Returns:
            List of news items with 'title' and 'publisher' keys
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return []
            
            # Filter recent news and extract relevant fields
            cutoff = datetime.now() - timedelta(days=days)
            headlines = []
            
            for item in news[:20]:  # Limit to 20 most recent
                # Handle both old and new yfinance news format
                content = item.get('content', item)  # New format nests under 'content'
                
                # Get title
                title = content.get('title', '')
                if not title:
                    continue
                
                # Get publish date (try different field names)
                pub_time = None
                pub_date_str = content.get('pubDate') or item.get('providerPublishTime')
                
                if pub_date_str:
                    try:
                        if isinstance(pub_date_str, (int, float)):
                            pub_time = datetime.fromtimestamp(pub_date_str)
                        else:
                            # Parse ISO format date string
                            pub_time = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    except:
                        pass
                
                # Get publisher
                provider = content.get('provider', {})
                if isinstance(provider, dict):
                    publisher = provider.get('displayName', 'Unknown')
                else:
                    publisher = item.get('publisher', 'Unknown')
                
                headlines.append({
                    'title': title,
                    'publisher': publisher,
                    'date': pub_time.strftime('%Y-%m-%d') if pub_time else None,
                    'link': content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else item.get('link', '')
                })
            
            return headlines
            
        except Exception as e:
            logger.warning(f"Could not fetch news for {ticker}: {e}")
            return []
    
    def analyze_headline(self, headline: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a single headline.
        
        Args:
            headline: News headline text
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if not self.is_available():
            return self._fallback_sentiment(headline)
        
        try:
            # Clean the headline
            clean_headline = self._clean_text(headline)
            
            if not clean_headline:
                return 'neutral', 0.5
            
            # Get sentiment prediction
            result = self.pipeline(clean_headline[:512])  # Truncate to model max
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # top_k=None returns list of lists
                    scores = {r['label'].lower(): r['score'] for r in result[0]}
                else:
                    scores = {r['label'].lower(): r['score'] for r in result}
                
                # Find highest scoring sentiment
                max_sentiment = max(scores, key=scores.get)
                max_score = scores[max_sentiment]
                
                return max_sentiment, max_score
            
            return 'neutral', 0.5
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._fallback_sentiment(headline)
    
    def _fallback_sentiment(self, headline: str) -> Tuple[str, float]:
        """
        Simple keyword-based fallback sentiment analysis.
        """
        headline_lower = headline.lower()
        
        positive_words = [
            'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'beat', 'exceed',
            'strong', 'growth', 'profit', 'upgrade', 'bullish', 'record', 'high',
            'success', 'positive', 'outperform', 'buy', 'breakout', 'boom'
        ]
        
        negative_words = [
            'fall', 'drop', 'plunge', 'crash', 'decline', 'loss', 'miss', 'below',
            'weak', 'downgrade', 'bearish', 'low', 'fail', 'negative', 'sell',
            'warning', 'concern', 'risk', 'fear', 'slump', 'tumble', 'cut'
        ]
        
        pos_count = sum(1 for word in positive_words if word in headline_lower)
        neg_count = sum(1 for word in negative_words if word in headline_lower)
        
        if pos_count > neg_count:
            return 'positive', 0.6
        elif neg_count > pos_count:
            return 'negative', 0.6
        else:
            return 'neutral', 0.5
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def analyze_ticker(
        self,
        ticker: str,
        headlines: Optional[List[Dict]] = None
    ) -> Optional[TickerSentiment]:
        """
        Analyze overall sentiment for a ticker.
        
        Args:
            ticker: Stock symbol
            headlines: Optional list of headlines (will fetch if not provided)
            
        Returns:
            TickerSentiment object or None if insufficient data
        """
        # Fetch headlines if not provided
        if headlines is None:
            headlines = self.fetch_news(ticker)
        
        if not headlines:
            logger.warning(f"No news headlines found for {ticker}")
            return None
        
        # Analyze each headline
        sentiments = []
        for item in headlines:
            title = item.get('title', '')
            if not title:
                continue
                
            sentiment, score = self.analyze_headline(title)
            
            sentiments.append(HeadlineSentiment(
                headline=title[:100],  # Truncate for display
                sentiment=sentiment,
                score=score,
                source=item.get('publisher', 'Unknown'),
                date=item.get('date')
            ))
        
        if len(sentiments) < self.min_headlines:
            logger.warning(f"Insufficient headlines for {ticker}: {len(sentiments)} < {self.min_headlines}")
            return TickerSentiment(
                ticker=ticker,
                overall_score=0.0,
                sentiment_label='NEUTRAL',
                headline_count=len(sentiments),
                positive_pct=0,
                negative_pct=0,
                neutral_pct=100,
                headlines=sentiments,
                model_available=self.is_available()
            )
        
        # Calculate aggregated sentiment
        positive = sum(1 for s in sentiments if s.sentiment == 'positive')
        negative = sum(1 for s in sentiments if s.sentiment == 'negative')
        neutral = sum(1 for s in sentiments if s.sentiment == 'neutral')
        total = len(sentiments)
        
        # Weight more recent headlines higher
        weighted_score = 0
        total_weight = 0
        for i, s in enumerate(sentiments):
            weight = 1.0 + (i / total) * 0.5  # More recent = higher weight
            if s.sentiment == 'positive':
                weighted_score += s.score * weight
            elif s.sentiment == 'negative':
                weighted_score -= s.score * weight
            total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine label
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
            positive_pct=(positive / total) * 100 if total > 0 else 0,
            negative_pct=(negative / total) * 100 if total > 0 else 0,
            neutral_pct=(neutral / total) * 100 if total > 0 else 0,
            headlines=sentiments,
            model_available=self.is_available()
        )
    
    def get_signal_adjustment(self, sentiment: TickerSentiment) -> Tuple[float, str]:
        """
        Get probability adjustment based on sentiment.
        
        Args:
            sentiment: TickerSentiment object
            
        Returns:
            Tuple of (adjustment amount, reason string)
        """
        if sentiment is None:
            return 0.0, ""
        
        score = sentiment.overall_score
        
        # Strong positive sentiment
        if score >= 0.5:
            return 8.0, f"Strong bullish sentiment ({sentiment.headline_count} headlines)"
        elif score >= 0.3:
            return 4.0, f"Positive sentiment ({sentiment.headline_count} headlines)"
        # Strong negative sentiment
        elif score <= -0.5:
            return -8.0, f"Strong bearish sentiment ({sentiment.headline_count} headlines)"
        elif score <= -0.3:
            return -4.0, f"Negative sentiment ({sentiment.headline_count} headlines)"
        else:
            return 0.0, f"Neutral sentiment ({sentiment.headline_count} headlines)"
    
    def get_status(self) -> Dict:
        """Get model status information."""
        return {
            'enabled': self.enabled,
            'model': self.model_name,
            'loaded': self.model_loaded,
            'error': self._load_error,
            'weight': self.weight
        }
