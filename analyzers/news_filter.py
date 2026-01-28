"""
News and Earnings Filter Module.
Checks for upcoming earnings and major news events that increase risk.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

import yfinance as yf


@dataclass
class EarningsInfo:
    """Earnings information for a stock."""
    ticker: str
    has_upcoming_earnings: bool
    earnings_date: Optional[datetime]
    days_until_earnings: Optional[int]
    is_high_risk: bool  # Within 3 days of earnings
    recommendation: str


@dataclass
class NewsInfo:
    """News analysis for a stock."""
    ticker: str
    recent_news_count: int
    has_major_news: bool
    sentiment: str  # POSITIVE, NEGATIVE, NEUTRAL
    headlines: List[str]
    is_high_risk: bool
    recommendation: str


@dataclass
class EventFilter:
    """Combined event filter result."""
    ticker: str
    earnings: Optional[EarningsInfo]
    news: Optional[NewsInfo]
    should_avoid: bool
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    reasons: List[str]


class NewsAndEarningsFilter:
    """
    Filters stocks based on upcoming events and news.
    
    Flags high-risk situations:
    - Earnings within 3 days
    - Major news events
    - Unusual volatility
    """
    
    def __init__(self, config: Dict):
        """
        Initialize news filter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('news_filter', {})
        self.earnings_buffer_days = self.config.get('earnings_buffer_days', 3)
        self.avoid_earnings = self.config.get('avoid_earnings', True)
    
    def check_earnings(self, ticker: str) -> EarningsInfo:
        """
        Check for upcoming earnings.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            EarningsInfo with earnings analysis
        """
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is None or calendar.empty:
                return EarningsInfo(
                    ticker=ticker,
                    has_upcoming_earnings=False,
                    earnings_date=None,
                    days_until_earnings=None,
                    is_high_risk=False,
                    recommendation="No earnings data available"
                )
            
            # Get earnings date
            earnings_date = None
            if 'Earnings Date' in calendar.index:
                earnings_dates = calendar.loc['Earnings Date']
                if isinstance(earnings_dates, (list, tuple)) and len(earnings_dates) > 0:
                    earnings_date = earnings_dates[0]
                elif hasattr(earnings_dates, 'iloc'):
                    earnings_date = earnings_dates.iloc[0] if len(earnings_dates) > 0 else None
                else:
                    earnings_date = earnings_dates
            
            if earnings_date is None:
                return EarningsInfo(
                    ticker=ticker,
                    has_upcoming_earnings=False,
                    earnings_date=None,
                    days_until_earnings=None,
                    is_high_risk=False,
                    recommendation="No upcoming earnings found"
                )
            
            # Convert to datetime if needed
            if hasattr(earnings_date, 'to_pydatetime'):
                earnings_date = earnings_date.to_pydatetime()
            elif isinstance(earnings_date, str):
                earnings_date = datetime.fromisoformat(earnings_date.replace('Z', '+00:00'))
            
            # Remove timezone if present for comparison
            if hasattr(earnings_date, 'replace') and hasattr(earnings_date, 'tzinfo'):
                earnings_date = earnings_date.replace(tzinfo=None)
            
            now = datetime.now()
            days_until = (earnings_date - now).days
            
            is_high_risk = 0 <= days_until <= self.earnings_buffer_days
            
            if is_high_risk:
                recommendation = f"⚠️ AVOID - Earnings in {days_until} days"
            elif days_until < 0:
                recommendation = f"Earnings passed {abs(days_until)} days ago"
            else:
                recommendation = f"Earnings in {days_until} days - OK to trade"
            
            return EarningsInfo(
                ticker=ticker,
                has_upcoming_earnings=days_until >= 0,
                earnings_date=earnings_date,
                days_until_earnings=days_until if days_until >= 0 else None,
                is_high_risk=is_high_risk,
                recommendation=recommendation
            )
            
        except Exception as e:
            return EarningsInfo(
                ticker=ticker,
                has_upcoming_earnings=False,
                earnings_date=None,
                days_until_earnings=None,
                is_high_risk=False,
                recommendation=f"Could not check earnings: {str(e)}"
            )
    
    def check_news(self, ticker: str) -> NewsInfo:
        """
        Check for recent news.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            NewsInfo with news analysis
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return NewsInfo(
                    ticker=ticker,
                    recent_news_count=0,
                    has_major_news=False,
                    sentiment='NEUTRAL',
                    headlines=[],
                    is_high_risk=False,
                    recommendation="No recent news"
                )
            
            # Analyze recent news (last 24 hours)
            recent_news = []
            headlines = []
            
            for item in news[:10]:  # Check last 10 news items
                title = item.get('title', '')
                headlines.append(title)
                
                # Check for major news keywords
                major_keywords = [
                    'fda', 'sec', 'lawsuit', 'merger', 'acquisition',
                    'bankruptcy', 'layoff', 'recall', 'investigation',
                    'downgrade', 'upgrade', 'scandal', 'ceo', 'resign'
                ]
                
                title_lower = title.lower()
                if any(keyword in title_lower for keyword in major_keywords):
                    recent_news.append(item)
            
            has_major_news = len(recent_news) > 0
            is_high_risk = len(recent_news) >= 2  # Multiple major news = high risk
            
            # Simple sentiment analysis
            positive_words = ['upgrade', 'beat', 'surge', 'soar', 'rally', 'profit']
            negative_words = ['downgrade', 'miss', 'plunge', 'crash', 'loss', 'cut']
            
            positive_count = sum(
                1 for h in headlines
                for word in positive_words
                if word in h.lower()
            )
            negative_count = sum(
                1 for h in headlines
                for word in negative_words
                if word in h.lower()
            )
            
            if positive_count > negative_count:
                sentiment = 'POSITIVE'
            elif negative_count > positive_count:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            if is_high_risk:
                recommendation = "⚠️ Multiple major news events - exercise caution"
            elif has_major_news:
                recommendation = "Major news detected - increased volatility likely"
            else:
                recommendation = "Normal news flow"
            
            return NewsInfo(
                ticker=ticker,
                recent_news_count=len(news),
                has_major_news=has_major_news,
                sentiment=sentiment,
                headlines=headlines[:5],
                is_high_risk=is_high_risk,
                recommendation=recommendation
            )
            
        except Exception as e:
            return NewsInfo(
                ticker=ticker,
                recent_news_count=0,
                has_major_news=False,
                sentiment='NEUTRAL',
                headlines=[],
                is_high_risk=False,
                recommendation=f"Could not check news: {str(e)}"
            )
    
    def filter_stock(self, ticker: str) -> EventFilter:
        """
        Complete event filter for a stock.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            EventFilter with combined analysis
        """
        earnings = self.check_earnings(ticker)
        news = self.check_news(ticker)
        
        reasons = []
        risk_points = 0
        
        # Earnings risk
        if earnings.is_high_risk:
            risk_points += 3
            reasons.append(f"Earnings in {earnings.days_until_earnings} days")
        
        # News risk
        if news.is_high_risk:
            risk_points += 2
            reasons.append("Multiple major news events")
        elif news.has_major_news:
            risk_points += 1
            reasons.append("Major news detected")
        
        # Negative sentiment
        if news.sentiment == 'NEGATIVE':
            risk_points += 1
            reasons.append("Negative news sentiment")
        
        # Determine risk level
        if risk_points >= 4:
            risk_level = 'EXTREME'
            should_avoid = True
        elif risk_points >= 3:
            risk_level = 'HIGH'
            should_avoid = self.avoid_earnings and earnings.is_high_risk
        elif risk_points >= 1:
            risk_level = 'MEDIUM'
            should_avoid = False
        else:
            risk_level = 'LOW'
            should_avoid = False
        
        if not reasons:
            reasons.append("No significant events detected")
        
        return EventFilter(
            ticker=ticker,
            earnings=earnings,
            news=news,
            should_avoid=should_avoid,
            risk_level=risk_level,
            reasons=reasons
        )
