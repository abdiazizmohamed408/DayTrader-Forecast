"""
Alert System Module.
Sends email alerts for high-probability trading signals.
"""

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional
import json
from pathlib import Path


class AlertSystem:
    """
    Handles email alerts for trading signals.
    
    Sends notifications when high-probability signals are detected.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize alert system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('alerts', {})
        
        # Email settings
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.email_to = os.getenv('EMAIL_TO', self.config.get('email', ''))
        
        # Alert thresholds
        self.min_probability = self.config.get('min_probability', 65)
        self.alert_on_sell = self.config.get('alert_on_sell', True)
        
        # Rate limiting
        self.max_alerts_per_hour = self.config.get('max_alerts_per_hour', 10)
        self.alert_log_file = Path("./data/alert_log.json")
        self.alert_log_file.parent.mkdir(exist_ok=True)
    
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.smtp_user and self.smtp_password and self.email_to)
    
    def _load_alert_log(self) -> List[Dict]:
        """Load alert log for rate limiting."""
        if self.alert_log_file.exists():
            try:
                with open(self.alert_log_file) as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_alert_log(self, log: List[Dict]):
        """Save alert log."""
        with open(self.alert_log_file, 'w') as f:
            json.dump(log, f)
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        log = self._load_alert_log()
        now = datetime.now()
        
        # Filter to last hour
        one_hour_ago = (now.timestamp() - 3600)
        recent = [a for a in log if a.get('timestamp', 0) > one_hour_ago]
        
        return len(recent) < self.max_alerts_per_hour
    
    def _record_alert(self, ticker: str, signal_type: str):
        """Record that an alert was sent."""
        log = self._load_alert_log()
        log.append({
            'ticker': ticker,
            'signal_type': signal_type,
            'timestamp': datetime.now().timestamp()
        })
        
        # Keep only last 100 entries
        self._save_alert_log(log[-100:])
    
    def should_alert(
        self,
        signal_type: str,
        probability: float
    ) -> bool:
        """
        Determine if an alert should be sent.
        
        Args:
            signal_type: BUY or SELL
            probability: Signal probability
            
        Returns:
            True if alert should be sent
        """
        if probability < self.min_probability:
            return False
        
        if signal_type == 'SELL' and not self.alert_on_sell:
            return False
        
        if signal_type == 'HOLD':
            return False
        
        if not self._check_rate_limit():
            return False
        
        return True
    
    def send_signal_alert(
        self,
        ticker: str,
        signal_type: str,
        probability: float,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        reasons: List[str],
        additional_info: Optional[Dict] = None
    ) -> bool:
        """
        Send email alert for a trading signal.
        
        Args:
            ticker: Stock symbol
            signal_type: BUY or SELL
            probability: Confidence score
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price
            reasons: List of signal reasons
            additional_info: Extra information to include
            
        Returns:
            True if sent successfully
        """
        if not self.is_configured():
            print("⚠️ Email alerts not configured")
            return False
        
        if not self.should_alert(signal_type, probability):
            return False
        
        emoji = "🟢" if signal_type == 'BUY' else "🔴"
        
        subject = f"{emoji} {signal_type} Signal: {ticker} ({probability:.0f}%)"
        
        # Calculate risk/reward
        if signal_type == 'BUY':
            risk = entry_price - stop_loss
            reward = target_price - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - target_price
        
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Build email body
        body = f"""
{emoji} HIGH-PROBABILITY {signal_type} SIGNAL DETECTED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 TICKER: {ticker}
🎯 SIGNAL: {signal_type}
📊 PROBABILITY: {probability:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 TRADE SETUP:

   Entry Price:  ${entry_price:.2f}
   Stop Loss:    ${stop_loss:.2f}
   Target:       ${target_price:.2f}
   Risk/Reward:  {rr_ratio:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 SIGNAL FACTORS:

"""
        for reason in reasons:
            body += f"   • {reason}\n"
        
        if additional_info:
            body += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            body += "📊 ADDITIONAL INFO:\n\n"
            
            if 'market_context' in additional_info:
                mc = additional_info['market_context']
                body += f"   Market: {mc}\n"
            
            if 'multi_tf' in additional_info:
                mtf = additional_info['multi_tf']
                body += f"   Timeframe Alignment: {mtf}\n"
            
            if 'earnings_warning' in additional_info:
                body += f"   ⚠️ {additional_info['earnings_warning']}\n"
        
        body += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ Alert sent: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

⚠️ DISCLAIMER: This is NOT financial advice. 
   Always do your own research before trading.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DayTrader-Forecast
"""
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = self.email_to
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            self._record_alert(ticker, signal_type)
            return True
            
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
            return False
    
    def send_daily_summary(
        self,
        signals: List[Dict],
        performance: Optional[Dict] = None
    ) -> bool:
        """
        Send daily summary email.
        
        Args:
            signals: List of today's signals
            performance: Performance statistics
            
        Returns:
            True if sent successfully
        """
        if not self.is_configured():
            return False
        
        subject = f"📊 DayTrader Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        buy_signals = [s for s in signals if s.get('type') == 'BUY']
        sell_signals = [s for s in signals if s.get('type') == 'SELL']
        
        body = f"""
📊 DAYTRADER DAILY SUMMARY
{datetime.now().strftime("%B %d, %Y")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 TODAY'S SIGNALS:

   🟢 BUY Signals:  {len(buy_signals)}
   🔴 SELL Signals: {len(sell_signals)}

"""
        
        if buy_signals:
            body += "━━ TOP BUY SIGNALS ━━\n\n"
            for s in sorted(buy_signals, key=lambda x: x.get('probability', 0), reverse=True)[:5]:
                body += f"   {s.get('ticker')}: {s.get('probability', 0):.0f}% @ ${s.get('price', 0):.2f}\n"
            body += "\n"
        
        if sell_signals:
            body += "━━ TOP SELL SIGNALS ━━\n\n"
            for s in sorted(sell_signals, key=lambda x: x.get('probability', 0), reverse=True)[:5]:
                body += f"   {s.get('ticker')}: {s.get('probability', 0):.0f}% @ ${s.get('price', 0):.2f}\n"
            body += "\n"
        
        if performance:
            body += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MODEL PERFORMANCE:

   Win Rate: {performance.get('win_rate', 0):.1f}%
   Total Trades: {performance.get('total', 0)}
   Profit Factor: {performance.get('profit_factor', 0):.2f}

"""
        
        body += """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ This is NOT financial advice.

DayTrader-Forecast
"""
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = self.email_to
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to send summary: {e}")
            return False
