"""
Report Generation Module.
Creates formatted markdown reports for trading signals and analysis.
"""

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

from analyzers.signals import SignalType, TradingSignal
from utils.helpers import ensure_dir, format_currency, format_percent, get_date_str


DISCLAIMER = """
---

## ⚠️ DISCLAIMER

**This tool is for EDUCATIONAL PURPOSES ONLY.**

- This is NOT financial advice
- Past performance does NOT guarantee future results
- Day trading involves SIGNIFICANT RISK of loss
- Never trade with money you cannot afford to lose
- Always do your own research before making any trading decisions
- Consider consulting a licensed financial advisor

**The creators of this tool are not responsible for any financial losses.**

---
"""


class ReportGenerator:
    """
    Generates markdown reports from trading signals and analysis.
    
    Can save reports to files and optionally send via email.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration settings.
        
        Args:
            config: Dictionary with report settings
        """
        self.config = config.get('reports', {})
        self.output_dir = self.config.get('output_dir', './output')
        self.top_picks_count = self.config.get('top_picks_count', 5)
        
        # Ensure output directory exists
        ensure_dir(self.output_dir)
    
    def generate_scan_report(
        self, signals: List[TradingSignal], analyses: Dict
    ) -> str:
        """
        Generate a market scan report.
        
        Args:
            signals: List of trading signals
            analyses: Dictionary of technical analyses by ticker
            
        Returns:
            Markdown formatted report string
        """
        date_str = datetime.now().strftime("%B %d, %Y %H:%M")
        
        report = f"""# 📊 Day Trading Scan Report

**Generated:** {date_str}

{DISCLAIMER}

## 📈 Market Overview

Scanned **{len(signals)}** stocks from the watchlist.

"""
        # Sort signals by probability
        buy_signals = sorted(
            [s for s in signals if s.signal_type == SignalType.BUY],
            key=lambda x: x.probability,
            reverse=True
        )
        sell_signals = sorted(
            [s for s in signals if s.signal_type == SignalType.SELL],
            key=lambda x: x.probability,
            reverse=True
        )
        hold_signals = [s for s in signals if s.signal_type == SignalType.HOLD]
        
        # Summary
        report += f"""### Signal Summary

- 🟢 **BUY Signals:** {len(buy_signals)}
- 🔴 **SELL Signals:** {len(sell_signals)}
- 🟡 **HOLD Signals:** {len(hold_signals)}

"""
        
        # Top BUY picks
        if buy_signals:
            report += "## 🟢 Top BUY Signals\n\n"
            for signal in buy_signals[:self.top_picks_count]:
                report += self._format_signal_card(signal, analyses.get(signal.ticker))
        
        # Top SELL picks
        if sell_signals:
            report += "## 🔴 Top SELL Signals\n\n"
            for signal in sell_signals[:self.top_picks_count]:
                report += self._format_signal_card(signal, analyses.get(signal.ticker))
        
        # HOLD signals summary
        if hold_signals:
            report += "## 🟡 HOLD Signals\n\n"
            for signal in hold_signals:
                report += f"- **{signal.ticker}** - {signal.probability:.1f}% confidence\n"
            report += "\n"
        
        # Risk assessment
        report += self._generate_risk_assessment(signals)
        
        return report
    
    def generate_single_analysis(
        self, signal: TradingSignal, analysis: Dict
    ) -> str:
        """
        Generate detailed analysis report for a single stock.
        
        Args:
            signal: Trading signal
            analysis: Technical analysis dictionary
            
        Returns:
            Markdown formatted report string
        """
        date_str = datetime.now().strftime("%B %d, %Y %H:%M")
        
        emoji = "🟢" if signal.signal_type == SignalType.BUY else (
            "🔴" if signal.signal_type == SignalType.SELL else "🟡"
        )
        
        report = f"""# {emoji} {signal.ticker} Analysis Report

**Generated:** {date_str}

{DISCLAIMER}

## Overview

- **Current Price:** {format_currency(signal.entry_price)}
- **Signal:** {signal.signal_type.value}
- **Probability:** {signal.probability:.1f}%
- **Sentiment:** {signal.sentiment.value}

"""
        
        # Price targets
        if signal.stop_loss and signal.target_price:
            report += f"""## Price Targets

- **Entry Price:** {format_currency(signal.entry_price)}
- **Stop Loss:** {format_currency(signal.stop_loss)} ({((signal.stop_loss - signal.entry_price) / signal.entry_price * 100):+.2f}%)
- **Target Price:** {format_currency(signal.target_price)} ({((signal.target_price - signal.entry_price) / signal.entry_price * 100):+.2f}%)
"""
            if signal.risk_reward_ratio:
                report += f"- **Risk/Reward Ratio:** {signal.risk_reward_ratio:.2f}\n"
            report += "\n"
        
        # Technical indicators
        if analysis:
            report += """## Technical Indicators

### Momentum
"""
            report += f"- **RSI (14):** {analysis.get('rsi', 0):.2f}"
            if analysis.get('rsi_overbought'):
                report += " ⚠️ Overbought"
            elif analysis.get('rsi_oversold'):
                report += " ⚠️ Oversold"
            report += "\n"
            
            report += f"""- **MACD:** {analysis.get('macd', 0):.4f}
- **MACD Signal:** {analysis.get('macd_signal', 0):.4f}
- **MACD Histogram:** {analysis.get('macd_histogram', 0):.4f}

### Moving Averages
- **SMA (20):** {format_currency(analysis.get('sma_short', 0))}
- **SMA (50):** {format_currency(analysis.get('sma_long', 0))}
- **EMA (12):** {format_currency(analysis.get('ema', 0))}
- **Price vs SMA(20):** {'Above ✅' if analysis.get('above_sma_short') else 'Below ❌'}
- **Price vs SMA(50):** {'Above ✅' if analysis.get('above_sma_long') else 'Below ❌'}

### Bollinger Bands
- **Upper Band:** {format_currency(analysis.get('bb_upper', 0))}
- **Middle Band:** {format_currency(analysis.get('bb_middle', 0))}
- **Lower Band:** {format_currency(analysis.get('bb_lower', 0))}
- **Position:** {analysis.get('bb_position', 'N/A')}

### Volume
"""
            vol = analysis.get('volume', {})
            report += f"""- **Current Volume:** {vol.get('current_volume', 0):,.0f}
- **Average Volume:** {vol.get('average_volume', 0):,.0f}
- **Volume Ratio:** {vol.get('volume_ratio', 1):.2f}x average
"""
            
            # Support/Resistance
            levels = analysis.get('levels', {})
            if levels:
                report += f"""
### Support & Resistance Levels
- **Resistance 2:** {format_currency(levels.get('resistance_2', 0))}
- **Resistance 1:** {format_currency(levels.get('resistance_1', 0))}
- **Pivot:** {format_currency(levels.get('pivot', 0))}
- **Support 1:** {format_currency(levels.get('support_1', 0))}
- **Support 2:** {format_currency(levels.get('support_2', 0))}
"""
        
        # Signal reasons
        if signal.reasons:
            report += "\n## Signal Reasoning\n\n"
            for reason in signal.reasons:
                report += f"- {reason}\n"
        
        return report
    
    def _format_signal_card(
        self, signal: TradingSignal, analysis: Optional[Dict]
    ) -> str:
        """Format a signal as a card for reports."""
        emoji = "🟢" if signal.signal_type == SignalType.BUY else "🔴"
        
        card = f"""### {emoji} {signal.ticker}

- **Price:** {format_currency(signal.entry_price)}
- **Signal:** {signal.signal_type.value} ({signal.probability:.1f}% confidence)
- **Sentiment:** {signal.sentiment.value}
"""
        
        if signal.stop_loss and signal.target_price:
            card += f"""- **Stop Loss:** {format_currency(signal.stop_loss)}
- **Target:** {format_currency(signal.target_price)}
"""
            if signal.risk_reward_ratio:
                card += f"- **R:R Ratio:** {signal.risk_reward_ratio:.2f}\n"
        
        if signal.reasons:
            card += "\n**Key Factors:**\n"
            for reason in signal.reasons[:3]:  # Top 3 reasons
                card += f"  - {reason}\n"
        
        card += "\n"
        return card
    
    def _generate_risk_assessment(self, signals: List[TradingSignal]) -> str:
        """Generate risk assessment section."""
        high_prob = [s for s in signals if s.probability >= 70]
        
        report = """## ⚠️ Risk Assessment

### Signal Quality

"""
        if high_prob:
            report += f"- **High Confidence Signals (≥70%):** {len(high_prob)}\n"
        else:
            report += "- **Note:** No high-confidence signals detected today.\n"
        
        report += """
### Trading Guidelines

1. **Position Sizing:** Never risk more than 1-2% of your portfolio on a single trade
2. **Stop Losses:** Always set stop losses before entering a trade
3. **Take Profits:** Consider scaling out at target prices
4. **Market Conditions:** Check overall market direction before trading
5. **News:** Be aware of upcoming earnings or news events

"""
        return report
    
    def save_report(self, content: str, filename: Optional[str] = None) -> str:
        """
        Save report to file.
        
        Args:
            content: Report content
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"report_{get_date_str()}.md"
        
        filepath = Path(self.output_dir) / filename
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return str(filepath)
    
    def send_email(
        self, content: str, subject: Optional[str] = None
    ) -> bool:
        """
        Send report via email.
        
        Args:
            content: Report content
            subject: Email subject
            
        Returns:
            True if sent successfully, False otherwise
        """
        smtp_host = os.getenv('SMTP_HOST')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_user = os.getenv('SMTP_USER')
        smtp_pass = os.getenv('SMTP_PASSWORD')
        email_to = os.getenv('EMAIL_TO')
        
        if not all([smtp_host, smtp_user, smtp_pass, email_to]):
            print("❌ Email not configured. Set SMTP_* environment variables.")
            return False
        
        if subject is None:
            subject = f"DayTrader Forecast Report - {get_date_str()}"
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = smtp_user
            msg['To'] = email_to
            
            # Add plain text version
            msg.attach(MIMEText(content, 'plain'))
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            
            print(f"✅ Email sent to {email_to}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False
