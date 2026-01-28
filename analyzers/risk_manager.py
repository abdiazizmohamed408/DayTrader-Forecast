"""
Risk Management Module.
Calculates position sizes, enforces risk limits, and tracks daily P&L.
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class PositionSize:
    """Position sizing recommendation."""
    ticker: str
    shares: int
    position_value: float
    risk_amount: float
    risk_percent: float
    stop_loss: float
    entry_price: float
    target_price: float
    risk_reward_ratio: float
    is_valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class RiskStatus:
    """Current risk status for the day."""
    daily_pnl: float
    daily_pnl_pct: float
    trades_today: int
    wins_today: int
    losses_today: int
    at_daily_limit: bool
    remaining_risk: float
    open_positions: int
    total_exposure: float
    warnings: List[str]


class RiskManager:
    """
    Manages trading risk and position sizing.
    
    Enforces:
    - Maximum 1-2% risk per trade
    - Minimum 1:2 risk/reward ratio
    - Daily loss limits
    - Position size limits
    """
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager.
        
        Args:
            config: Configuration with risk settings
        """
        risk_config = config.get('risk', {})
        
        # Account settings
        self.account_size = risk_config.get('account_size', 10000)
        
        # Per-trade limits
        self.max_risk_per_trade_pct = risk_config.get('max_risk_per_trade', 1.0) / 100
        self.min_risk_reward = risk_config.get('min_risk_reward', 2.0)
        self.max_position_pct = risk_config.get('max_position_pct', 10) / 100
        
        # Daily limits
        self.daily_loss_limit_pct = risk_config.get('daily_loss_limit', 3.0) / 100
        self.max_trades_per_day = risk_config.get('max_trades_per_day', 10)
        self.max_open_positions = risk_config.get('max_open_positions', 5)
        
        # State file
        self.state_file = Path("./data/risk_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        
        # Load or initialize state
        self._load_state()
    
    def _load_state(self):
        """Load or initialize daily state."""
        today = date.today().isoformat()
        
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    
                # Reset if new day
                if state.get('date') != today:
                    state = self._new_day_state(today)
            except:
                state = self._new_day_state(today)
        else:
            state = self._new_day_state(today)
        
        self.state = state
        self._save_state()
    
    def _new_day_state(self, today: str) -> Dict:
        """Create new day state."""
        return {
            'date': today,
            'daily_pnl': 0.0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'open_positions': [],
            'closed_trades': []
        }
    
    def _save_state(self):
        """Save state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def calculate_position_size(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        target_price: float
    ) -> PositionSize:
        """
        Calculate proper position size based on risk parameters.
        
        Args:
            ticker: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price
            
        Returns:
            PositionSize with recommendation
        """
        # Calculate risk per share
        if entry_price <= 0 or stop_loss <= 0:
            return PositionSize(
                ticker=ticker, shares=0, position_value=0,
                risk_amount=0, risk_percent=0, stop_loss=stop_loss,
                entry_price=entry_price, target_price=target_price,
                risk_reward_ratio=0, is_valid=False,
                rejection_reason="Invalid prices"
            )
        
        risk_per_share = abs(entry_price - stop_loss)
        reward_per_share = abs(target_price - entry_price)
        
        if risk_per_share == 0:
            return PositionSize(
                ticker=ticker, shares=0, position_value=0,
                risk_amount=0, risk_percent=0, stop_loss=stop_loss,
                entry_price=entry_price, target_price=target_price,
                risk_reward_ratio=0, is_valid=False,
                rejection_reason="Stop loss equals entry price"
            )
        
        # Risk/reward ratio
        rr_ratio = reward_per_share / risk_per_share
        
        if rr_ratio < self.min_risk_reward:
            return PositionSize(
                ticker=ticker, shares=0, position_value=0,
                risk_amount=0, risk_percent=0, stop_loss=stop_loss,
                entry_price=entry_price, target_price=target_price,
                risk_reward_ratio=rr_ratio, is_valid=False,
                rejection_reason=f"R:R ratio {rr_ratio:.2f} < minimum {self.min_risk_reward}"
            )
        
        # Maximum risk amount
        max_risk_amount = self.account_size * self.max_risk_per_trade_pct
        
        # Calculate shares based on risk
        shares = int(max_risk_amount / risk_per_share)
        
        # Check position size limit
        position_value = shares * entry_price
        max_position_value = self.account_size * self.max_position_pct
        
        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        if shares <= 0:
            return PositionSize(
                ticker=ticker, shares=0, position_value=0,
                risk_amount=0, risk_percent=0, stop_loss=stop_loss,
                entry_price=entry_price, target_price=target_price,
                risk_reward_ratio=rr_ratio, is_valid=False,
                rejection_reason="Position too small for account size"
            )
        
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / self.account_size
        
        return PositionSize(
            ticker=ticker,
            shares=shares,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percent=actual_risk_pct * 100,
            stop_loss=stop_loss,
            entry_price=entry_price,
            target_price=target_price,
            risk_reward_ratio=rr_ratio,
            is_valid=True
        )
    
    def check_can_trade(self) -> tuple:
        """
        Check if trading is allowed based on daily limits.
        
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check daily loss limit
        daily_pnl_pct = self.state['daily_pnl'] / self.account_size
        if daily_pnl_pct <= -self.daily_loss_limit_pct:
            return False, f"Daily loss limit reached ({daily_pnl_pct*100:.1f}%)"
        
        # Check trade count
        if self.state['trades'] >= self.max_trades_per_day:
            return False, f"Maximum trades per day ({self.max_trades_per_day}) reached"
        
        # Check open positions
        if len(self.state['open_positions']) >= self.max_open_positions:
            return False, f"Maximum open positions ({self.max_open_positions}) reached"
        
        return True, "Trading allowed"
    
    def record_trade_open(
        self,
        ticker: str,
        shares: int,
        entry_price: float,
        stop_loss: float,
        target_price: float
    ):
        """Record opening a new position."""
        self.state['open_positions'].append({
            'ticker': ticker,
            'shares': shares,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'opened_at': datetime.now().isoformat()
        })
        self.state['trades'] += 1
        self._save_state()
    
    def record_trade_close(
        self,
        ticker: str,
        exit_price: float,
        pnl: float
    ):
        """Record closing a position."""
        # Remove from open positions
        self.state['open_positions'] = [
            p for p in self.state['open_positions']
            if p['ticker'] != ticker
        ]
        
        # Update P&L
        self.state['daily_pnl'] += pnl
        
        if pnl >= 0:
            self.state['wins'] += 1
        else:
            self.state['losses'] += 1
        
        # Record closed trade
        self.state['closed_trades'].append({
            'ticker': ticker,
            'pnl': pnl,
            'closed_at': datetime.now().isoformat()
        })
        
        self._save_state()
    
    def get_risk_status(self) -> RiskStatus:
        """Get current risk status."""
        daily_pnl_pct = (self.state['daily_pnl'] / self.account_size) * 100
        remaining_risk = (self.daily_loss_limit_pct * self.account_size) + self.state['daily_pnl']
        
        total_exposure = sum(
            p['shares'] * p['entry_price']
            for p in self.state['open_positions']
        )
        
        warnings = []
        
        if daily_pnl_pct <= -2:
            warnings.append("⚠️ Significant daily loss - consider stopping")
        
        if len(self.state['open_positions']) >= self.max_open_positions - 1:
            warnings.append("⚠️ Near maximum open positions")
        
        if self.state['losses'] > self.state['wins'] and self.state['trades'] > 3:
            warnings.append("⚠️ More losses than wins today")
        
        return RiskStatus(
            daily_pnl=self.state['daily_pnl'],
            daily_pnl_pct=daily_pnl_pct,
            trades_today=self.state['trades'],
            wins_today=self.state['wins'],
            losses_today=self.state['losses'],
            at_daily_limit=remaining_risk <= 0,
            remaining_risk=max(0, remaining_risk),
            open_positions=len(self.state['open_positions']),
            total_exposure=total_exposure,
            warnings=warnings
        )
    
    def update_account_size(self, new_size: float):
        """Update account size."""
        self.account_size = new_size
    
    def reset_day(self):
        """Manually reset daily state."""
        self.state = self._new_day_state(date.today().isoformat())
        self._save_state()
