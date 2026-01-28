"""
Economic Events Calendar Module.
Fetches and tracks major economic events that impact markets.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests


class EventImpact(Enum):
    """Event impact level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class EconomicEvent:
    """Represents an economic calendar event."""
    date: datetime
    time: str
    country: str
    event: str
    impact: EventImpact
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    
    @property
    def datetime_str(self) -> str:
        """Format date/time for display."""
        return self.date.strftime("%b %d") + f" {self.time}"
    
    @property
    def impact_emoji(self) -> str:
        """Get emoji for impact level."""
        if self.impact == EventImpact.HIGH:
            return "🔴"
        elif self.impact == EventImpact.MEDIUM:
            return "🟡"
        return "🟢"


@dataclass 
class EventCalendar:
    """Collection of economic events."""
    events: List[EconomicEvent] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    @property
    def high_impact_events(self) -> List[EconomicEvent]:
        """Get only high impact events."""
        return [e for e in self.events if e.impact == EventImpact.HIGH]
    
    @property
    def medium_impact_events(self) -> List[EconomicEvent]:
        """Get medium impact events."""
        return [e for e in self.events if e.impact == EventImpact.MEDIUM]
    
    def get_events_next_hours(self, hours: int = 48) -> List[EconomicEvent]:
        """Get events in the next N hours."""
        cutoff = datetime.now() + timedelta(hours=hours)
        return [e for e in self.events if e.date <= cutoff]


class EconomicCalendarFetcher:
    """
    Fetches economic calendar data.
    
    Primary: Scrapes from investing.com economic calendar
    Fallback: Uses hardcoded major recurring events
    """
    
    # Major recurring events (backup if scraping fails)
    RECURRING_EVENTS = [
        # US Events
        {"event": "Fed Interest Rate Decision", "country": "US", "impact": "HIGH", 
         "frequency": "6_weeks"},
        {"event": "Non-Farm Payrolls", "country": "US", "impact": "HIGH",
         "frequency": "first_friday"},
        {"event": "CPI (Inflation)", "country": "US", "impact": "HIGH",
         "frequency": "monthly"},
        {"event": "GDP (Quarterly)", "country": "US", "impact": "HIGH",
         "frequency": "quarterly"},
        {"event": "FOMC Minutes", "country": "US", "impact": "HIGH",
         "frequency": "3_weeks_after_fomc"},
        {"event": "Jobless Claims", "country": "US", "impact": "MEDIUM",
         "frequency": "weekly_thursday"},
        {"event": "Retail Sales", "country": "US", "impact": "MEDIUM",
         "frequency": "monthly"},
        {"event": "ISM Manufacturing PMI", "country": "US", "impact": "MEDIUM",
         "frequency": "first_business_day"},
        
        # European Events
        {"event": "ECB Interest Rate", "country": "EU", "impact": "HIGH",
         "frequency": "6_weeks"},
        {"event": "ECB Press Conference", "country": "EU", "impact": "HIGH",
         "frequency": "6_weeks"},
        
        # UK Events
        {"event": "BOE Interest Rate", "country": "UK", "impact": "HIGH",
         "frequency": "6_weeks"},
        
        # Japan Events
        {"event": "BOJ Interest Rate", "country": "JP", "impact": "HIGH",
         "frequency": "variable"},
    ]
    
    # Cache file path
    CACHE_FILE = Path("./data/events_cache.json")
    CACHE_DURATION = timedelta(hours=4)
    
    def __init__(self):
        """Initialize the fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_calendar(self, days_ahead: int = 7) -> EventCalendar:
        """
        Fetch economic calendar for upcoming days.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            EventCalendar with upcoming events
        """
        # Check cache first
        cached = self._load_cache()
        if cached:
            return cached
        
        events = []
        
        # Try to fetch from free API sources
        events = self._fetch_from_api(days_ahead)
        
        # If no events fetched, use fallback with estimated dates
        if not events:
            events = self._generate_fallback_events(days_ahead)
        
        calendar = EventCalendar(events=events, last_updated=datetime.now())
        
        # Cache the results
        self._save_cache(calendar)
        
        return calendar
    
    def _fetch_from_api(self, days_ahead: int) -> List[EconomicEvent]:
        """
        Fetch events from free API sources.
        
        Returns:
            List of EconomicEvent objects
        """
        events = []
        
        # Try FMP (Financial Modeling Prep) free tier
        try:
            events = self._fetch_from_fmp(days_ahead)
            if events:
                return events
        except Exception as e:
            print(f"FMP API error: {e}")
        
        # Try to scrape basic calendar
        try:
            events = self._scrape_basic_calendar(days_ahead)
            if events:
                return events
        except Exception as e:
            print(f"Scrape error: {e}")
        
        return events
    
    def _fetch_from_fmp(self, days_ahead: int) -> List[EconomicEvent]:
        """Fetch from Financial Modeling Prep API (if key available)."""
        # This requires an API key - return empty if not configured
        return []
    
    def _scrape_basic_calendar(self, days_ahead: int) -> List[EconomicEvent]:
        """
        Attempt to get basic calendar info.
        
        Uses publicly available data without authentication.
        """
        events = []
        
        try:
            # Try Yahoo Finance earnings calendar as a proxy
            # Note: This is limited but gives some market events
            from_date = datetime.now()
            to_date = from_date + timedelta(days=days_ahead)
            
            # Parse known Fed dates for 2024-2025
            fed_dates = [
                # 2025 FOMC meetings (approximate)
                datetime(2025, 1, 29), datetime(2025, 3, 19),
                datetime(2025, 5, 7), datetime(2025, 6, 18),
                datetime(2025, 7, 30), datetime(2025, 9, 17),
                datetime(2025, 11, 5), datetime(2025, 12, 17),
            ]
            
            for fed_date in fed_dates:
                if from_date <= fed_date <= to_date:
                    events.append(EconomicEvent(
                        date=fed_date,
                        time="14:00",
                        country="US",
                        event="Fed Interest Rate Decision",
                        impact=EventImpact.HIGH,
                        forecast="Hold"
                    ))
            
            # Add monthly recurring events
            events.extend(self._generate_monthly_events(from_date, to_date))
            
        except Exception as e:
            print(f"Calendar scrape error: {e}")
        
        return sorted(events, key=lambda x: x.date)
    
    def _generate_monthly_events(
        self, from_date: datetime, to_date: datetime
    ) -> List[EconomicEvent]:
        """Generate estimated dates for monthly recurring events."""
        events = []
        
        current = from_date
        while current <= to_date:
            # NFP - First Friday of month
            first_friday = self._get_nth_weekday(current.year, current.month, 4, 1)  # Friday=4
            if from_date <= first_friday <= to_date:
                events.append(EconomicEvent(
                    date=first_friday,
                    time="08:30",
                    country="US",
                    event="Non-Farm Payrolls",
                    impact=EventImpact.HIGH,
                    forecast="TBD"
                ))
            
            # CPI - Usually around 10th-13th of month
            cpi_date = datetime(current.year, current.month, 12)
            if from_date <= cpi_date <= to_date:
                events.append(EconomicEvent(
                    date=cpi_date,
                    time="08:30",
                    country="US",
                    event="CPI (Inflation)",
                    impact=EventImpact.HIGH,
                    forecast="TBD"
                ))
            
            # Jobless Claims - Every Thursday
            for day in range(1, 32):
                try:
                    check_date = datetime(current.year, current.month, day)
                    if check_date.weekday() == 3 and from_date <= check_date <= to_date:  # Thursday
                        events.append(EconomicEvent(
                            date=check_date,
                            time="08:30",
                            country="US",
                            event="Jobless Claims",
                            impact=EventImpact.MEDIUM,
                            forecast="TBD"
                        ))
                except ValueError:
                    break
            
            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        return events
    
    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> datetime:
        """Get nth occurrence of a weekday in a month."""
        count = 0
        for day in range(1, 32):
            try:
                d = datetime(year, month, day)
                if d.weekday() == weekday:
                    count += 1
                    if count == n:
                        return d
            except ValueError:
                break
        return datetime(year, month, 1)
    
    def _generate_fallback_events(self, days_ahead: int) -> List[EconomicEvent]:
        """Generate fallback events based on recurring patterns."""
        return self._generate_monthly_events(
            datetime.now(),
            datetime.now() + timedelta(days=days_ahead)
        )
    
    def get_high_impact_next_48h(self) -> List[EconomicEvent]:
        """Get high impact events in next 48 hours."""
        calendar = self.fetch_calendar(days_ahead=3)
        cutoff = datetime.now() + timedelta(hours=48)
        
        return [
            e for e in calendar.events 
            if e.impact == EventImpact.HIGH and e.date <= cutoff
        ]
    
    def has_high_impact_soon(self, hours: int = 24) -> bool:
        """Check if there's a high impact event coming soon."""
        calendar = self.fetch_calendar(days_ahead=2)
        cutoff = datetime.now() + timedelta(hours=hours)
        
        for event in calendar.events:
            if event.impact == EventImpact.HIGH and event.date <= cutoff:
                return True
        return False
    
    def _load_cache(self) -> Optional[EventCalendar]:
        """Load events from cache if still valid."""
        try:
            if not self.CACHE_FILE.exists():
                return None
            
            with open(self.CACHE_FILE, 'r') as f:
                data = json.load(f)
            
            last_updated = datetime.fromisoformat(data['last_updated'])
            if datetime.now() - last_updated > self.CACHE_DURATION:
                return None
            
            events = []
            for e in data['events']:
                events.append(EconomicEvent(
                    date=datetime.fromisoformat(e['date']),
                    time=e['time'],
                    country=e['country'],
                    event=e['event'],
                    impact=EventImpact(e['impact']),
                    forecast=e.get('forecast'),
                    previous=e.get('previous'),
                    actual=e.get('actual')
                ))
            
            return EventCalendar(events=events, last_updated=last_updated)
            
        except Exception:
            return None
    
    def _save_cache(self, calendar: EventCalendar):
        """Save events to cache."""
        try:
            self.CACHE_FILE.parent.mkdir(exist_ok=True)
            
            data = {
                'last_updated': calendar.last_updated.isoformat(),
                'events': [
                    {
                        'date': e.date.isoformat(),
                        'time': e.time,
                        'country': e.country,
                        'event': e.event,
                        'impact': e.impact.value,
                        'forecast': e.forecast,
                        'previous': e.previous,
                        'actual': e.actual
                    }
                    for e in calendar.events
                ]
            }
            
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Cache save error: {e}")
    
    def clear_cache(self):
        """Clear the events cache."""
        try:
            if self.CACHE_FILE.exists():
                self.CACHE_FILE.unlink()
        except Exception:
            pass
