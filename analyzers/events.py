"""
Event Risk Analyzer.
Analyzes upcoming economic events and their potential market impact.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from data.events import EconomicCalendarFetcher, EconomicEvent, EventImpact, EventCalendar


class EventRiskLevel(Enum):
    """Overall event risk level."""
    EXTREME = "EXTREME"  # Major event imminent (Fed decision, NFP)
    HIGH = "HIGH"        # High impact event within 24h
    MODERATE = "MODERATE"  # Medium impact event or high impact 24-48h away
    LOW = "LOW"          # No significant events soon
    CLEAR = "CLEAR"      # Trading conditions favorable


@dataclass
class EventRiskAssessment:
    """Assessment of event risk for trading."""
    risk_level: EventRiskLevel
    confidence_adjustment: float  # Adjustment to signal confidence (-20 to +5)
    should_reduce_position: bool
    should_avoid_trading: bool
    upcoming_events: List[EconomicEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def risk_emoji(self) -> str:
        """Get emoji for risk level."""
        emojis = {
            EventRiskLevel.EXTREME: "🚨",
            EventRiskLevel.HIGH: "⚠️",
            EventRiskLevel.MODERATE: "⚡",
            EventRiskLevel.LOW: "✅",
            EventRiskLevel.CLEAR: "🟢"
        }
        return emojis.get(self.risk_level, "❓")


class EventRiskAnalyzer:
    """
    Analyzes economic events and their impact on trading signals.
    
    Provides:
    - Event risk assessment
    - Signal confidence adjustments
    - Trading warnings
    """
    
    # Events that have maximum market impact
    EXTREME_IMPACT_EVENTS = [
        "Fed Interest Rate Decision",
        "FOMC",
        "Federal Reserve",
        "Non-Farm Payrolls",
        "NFP",
    ]
    
    # Events with significant but not extreme impact
    HIGH_IMPACT_KEYWORDS = [
        "CPI",
        "Inflation",
        "GDP",
        "ECB",
        "BOE",
        "BOJ",
        "Central Bank",
        "Employment",
        "Unemployment",
    ]
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.fetcher = EconomicCalendarFetcher()
        
        # Risk thresholds
        self.extreme_hours = 4     # Hours before extreme event
        self.high_hours = 24       # Hours before high event
        self.moderate_hours = 48   # Hours before any event to note
    
    def assess_event_risk(self) -> EventRiskAssessment:
        """
        Assess current event risk for trading.
        
        Returns:
            EventRiskAssessment with risk level and recommendations
        """
        calendar = self.fetcher.fetch_calendar(days_ahead=3)
        now = datetime.now()
        
        upcoming_events = []
        warnings = []
        risk_level = EventRiskLevel.CLEAR
        confidence_adjustment = 0
        should_reduce_position = False
        should_avoid_trading = False
        
        # Analyze each event
        for event in calendar.events:
            hours_until = (event.date - now).total_seconds() / 3600
            
            if hours_until < 0:
                continue  # Skip past events
            
            if hours_until <= self.moderate_hours:
                upcoming_events.append(event)
            
            # Check for extreme impact events
            is_extreme = any(
                keyword.lower() in event.event.lower() 
                for keyword in self.EXTREME_IMPACT_EVENTS
            )
            
            is_high = event.impact == EventImpact.HIGH or any(
                keyword.lower() in event.event.lower()
                for keyword in self.HIGH_IMPACT_KEYWORDS
            )
            
            # Determine risk level based on timing and impact
            if is_extreme and hours_until <= self.extreme_hours:
                risk_level = EventRiskLevel.EXTREME
                should_avoid_trading = True
                should_reduce_position = True
                confidence_adjustment = -20
                warnings.append(f"🚨 {event.event} in {hours_until:.0f}h - AVOID TRADING")
                
            elif is_extreme and hours_until <= self.high_hours:
                if risk_level != EventRiskLevel.EXTREME:
                    risk_level = EventRiskLevel.HIGH
                should_reduce_position = True
                confidence_adjustment = min(confidence_adjustment, -10)
                warnings.append(f"⚠️ {event.event} in {hours_until:.0f}h - Reduce positions")
                
            elif is_high and hours_until <= self.high_hours:
                if risk_level not in [EventRiskLevel.EXTREME, EventRiskLevel.HIGH]:
                    risk_level = EventRiskLevel.HIGH
                confidence_adjustment = min(confidence_adjustment, -10)
                warnings.append(f"⚠️ {event.event} in {hours_until:.0f}h")
                
            elif is_high and hours_until <= self.moderate_hours:
                if risk_level == EventRiskLevel.CLEAR:
                    risk_level = EventRiskLevel.MODERATE
                confidence_adjustment = min(confidence_adjustment, -5)
                
            elif event.impact == EventImpact.MEDIUM and hours_until <= self.high_hours:
                if risk_level == EventRiskLevel.CLEAR:
                    risk_level = EventRiskLevel.LOW
        
        return EventRiskAssessment(
            risk_level=risk_level,
            confidence_adjustment=confidence_adjustment,
            should_reduce_position=should_reduce_position,
            should_avoid_trading=should_avoid_trading,
            upcoming_events=upcoming_events,
            warnings=warnings
        )
    
    def adjust_signal_for_events(
        self, 
        base_probability: float,
        signal_type: str
    ) -> Tuple[float, List[str]]:
        """
        Adjust signal probability based on upcoming events.
        
        Args:
            base_probability: Original signal probability
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Tuple of (adjusted_probability, list of reasons)
        """
        assessment = self.assess_event_risk()
        reasons = []
        
        adjusted = base_probability + assessment.confidence_adjustment
        
        if assessment.confidence_adjustment < 0:
            reasons.append(f"Event risk: {abs(assessment.confidence_adjustment):.0f}% penalty")
        
        if assessment.should_avoid_trading:
            adjusted = min(adjusted, 40)  # Cap probability when should avoid
            reasons.append("Major event imminent - signal degraded")
        
        # Clamp probability
        adjusted = max(0, min(95, adjusted))
        
        return adjusted, reasons
    
    def get_event_summary(self, hours: int = 48) -> Dict:
        """
        Get summary of upcoming events for display.
        
        Args:
            hours: Hours to look ahead
            
        Returns:
            Dictionary with event summary
        """
        calendar = self.fetcher.fetch_calendar(days_ahead=3)
        cutoff = datetime.now() + timedelta(hours=hours)
        
        events_in_window = [e for e in calendar.events if e.date <= cutoff]
        high_impact = [e for e in events_in_window if e.impact == EventImpact.HIGH]
        medium_impact = [e for e in events_in_window if e.impact == EventImpact.MEDIUM]
        
        assessment = self.assess_event_risk()
        
        return {
            'total_events': len(events_in_window),
            'high_impact_count': len(high_impact),
            'medium_impact_count': len(medium_impact),
            'events': events_in_window,
            'high_impact': high_impact,
            'medium_impact': medium_impact,
            'risk_level': assessment.risk_level.value,
            'risk_emoji': assessment.risk_emoji,
            'warnings': assessment.warnings,
            'should_avoid': assessment.should_avoid_trading,
            'confidence_adjustment': assessment.confidence_adjustment
        }
    
    def format_events_table(self, events: List[EconomicEvent]) -> str:
        """
        Format events as a table string.
        
        Args:
            events: List of events to format
            
        Returns:
            Formatted table string
        """
        if not events:
            return "No upcoming events"
        
        lines = []
        lines.append("TIME (ET)      EVENT                         IMPACT   FORECAST")
        lines.append("─" * 65)
        
        for event in events[:10]:  # Show max 10 events
            time_str = event.datetime_str
            event_name = event.event[:28].ljust(28)
            impact_str = f"{event.impact_emoji} {event.impact.value:6s}"
            forecast = (event.forecast or "TBD")[:10]
            
            lines.append(f"{time_str:14s} {event_name} {impact_str} {forecast}")
        
        return "\n".join(lines)
    
    def is_safe_to_trade(self) -> Tuple[bool, str]:
        """
        Quick check if it's safe to trade now.
        
        Returns:
            Tuple of (is_safe, reason)
        """
        assessment = self.assess_event_risk()
        
        if assessment.should_avoid_trading:
            return False, f"Major event within {self.extreme_hours}h"
        
        if assessment.risk_level == EventRiskLevel.EXTREME:
            return False, "Extreme event risk - wait for resolution"
        
        if assessment.risk_level == EventRiskLevel.HIGH:
            return True, "Caution: High impact event upcoming"
        
        return True, "Event risk acceptable"
