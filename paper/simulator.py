"""
Paper Trading Simulator.
Virtual trading session with configurable starting balance.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
import time


@dataclass
class Position:
    """Represents an open position."""
    ticker: str
    signal_type: str  # BUY or SELL
    entry_price: float
    quantity: float
    target_price: float
    stop_loss: float
    entry_time: str
    probability: float
    position_value: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ClosedPosition:
    """Represents a closed position."""
    ticker: str
    signal_type: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: str
    exit_time: str
    profit_loss: float
    profit_pct: float
    outcome: str  # WIN, LOSS, MANUAL


@dataclass
class PaperTradeSession:
    """Paper trading session state."""
    session_id: str
    start_time: str
    initial_balance: float
    current_balance: float
    positions: List[Position] = field(default_factory=list)
    closed_trades: List[ClosedPosition] = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    
    @property
    def portfolio_value(self) -> float:
        """Total value including open positions."""
        position_value = sum(p.position_value for p in self.positions)
        return self.current_balance + position_value
    
    @property
    def total_pnl(self) -> float:
        """Total profit/loss since session start."""
        return self.portfolio_value - self.initial_balance
    
    @property
    def total_pnl_pct(self) -> float:
        """Total P&L as percentage."""
        return (self.total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
    
    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.total_trades == 0:
            return 0
        return (self.wins / self.total_trades) * 100
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'portfolio_value': self.portfolio_value,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'positions': [p.to_dict() for p in self.positions],
            'closed_trades': [asdict(t) for t in self.closed_trades],
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate
        }


class PaperTrader:
    """
    Paper trading simulator for virtual trading sessions.
    
    Allows opening positions based on signals and tracking
    portfolio value over time.
    """
    
    SESSION_FILE = "data/paper_session.json"
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        position_size_pct: float = 10.0,
        max_positions: int = 5
    ):
        """
        Initialize paper trader.
        
        Args:
            initial_balance: Starting virtual balance
            position_size_pct: Percentage of balance per trade
            max_positions: Maximum concurrent positions
        """
        self.initial_balance = initial_balance
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.session: Optional[PaperTradeSession] = None
    
    def start_session(self, balance: Optional[float] = None) -> PaperTradeSession:
        """
        Start a new paper trading session.
        
        Args:
            balance: Optional custom starting balance
            
        Returns:
            New PaperTradeSession
        """
        if balance is None:
            balance = self.initial_balance
        
        self.session = PaperTradeSession(
            session_id=f"paper_{int(time.time())}",
            start_time=datetime.now().isoformat(),
            initial_balance=balance,
            current_balance=balance
        )
        
        self._save_session()
        return self.session
    
    def load_session(self) -> Optional[PaperTradeSession]:
        """
        Load existing session from file.
        
        Returns:
            Loaded session or None if not found
        """
        if not os.path.exists(self.SESSION_FILE):
            return None
        
        try:
            with open(self.SESSION_FILE, 'r') as f:
                data = json.load(f)
            
            positions = [Position(**p) for p in data.get('positions', [])]
            closed = [ClosedPosition(**t) for t in data.get('closed_trades', [])]
            
            self.session = PaperTradeSession(
                session_id=data['session_id'],
                start_time=data['start_time'],
                initial_balance=data['initial_balance'],
                current_balance=data['current_balance'],
                positions=positions,
                closed_trades=closed,
                total_trades=data.get('total_trades', 0),
                wins=data.get('wins', 0),
                losses=data.get('losses', 0)
            )
            
            return self.session
            
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def _save_session(self):
        """Save current session to file."""
        if self.session is None:
            return
        
        os.makedirs(os.path.dirname(self.SESSION_FILE), exist_ok=True)
        
        with open(self.SESSION_FILE, 'w') as f:
            json.dump(self.session.to_dict(), f, indent=2)
    
    def open_position(
        self,
        ticker: str,
        signal_type: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        probability: float
    ) -> Optional[Position]:
        """
        Open a new paper trading position.
        
        Args:
            ticker: Stock symbol
            signal_type: BUY or SELL
            entry_price: Entry price
            target_price: Target price
            stop_loss: Stop loss price
            probability: Signal probability
            
        Returns:
            New Position or None if cannot open
        """
        if self.session is None:
            print("No active session. Call start_session() first.")
            return None
        
        if len(self.session.positions) >= self.max_positions:
            print(f"Maximum positions ({self.max_positions}) reached.")
            return None
        
        # Check if already in this ticker
        for pos in self.session.positions:
            if pos.ticker == ticker:
                print(f"Already have a position in {ticker}")
                return None
        
        # Calculate position size
        position_value = self.session.current_balance * (self.position_size_pct / 100)
        
        if position_value > self.session.current_balance:
            print("Insufficient balance")
            return None
        
        quantity = position_value / entry_price
        
        position = Position(
            ticker=ticker.upper(),
            signal_type=signal_type,
            entry_price=entry_price,
            quantity=quantity,
            target_price=target_price,
            stop_loss=stop_loss,
            entry_time=datetime.now().isoformat(),
            probability=probability,
            position_value=position_value
        )
        
        # Deduct from balance
        self.session.current_balance -= position_value
        self.session.positions.append(position)
        
        self._save_session()
        return position
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        outcome: str = 'MANUAL'
    ) -> Optional[ClosedPosition]:
        """
        Close an open position.
        
        Args:
            ticker: Stock symbol
            exit_price: Exit price
            outcome: WIN, LOSS, or MANUAL
            
        Returns:
            ClosedPosition or None if not found
        """
        if self.session is None:
            return None
        
        # Find position
        position = None
        for p in self.session.positions:
            if p.ticker == ticker.upper():
                position = p
                break
        
        if position is None:
            print(f"No open position for {ticker}")
            return None
        
        # Calculate P&L
        if position.signal_type == 'BUY':
            profit_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        else:
            profit_pct = ((position.entry_price - exit_price) / position.entry_price) * 100
        
        exit_value = position.quantity * exit_price
        profit_loss = exit_value - position.position_value
        
        closed = ClosedPosition(
            ticker=position.ticker,
            signal_type=position.signal_type,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=datetime.now().isoformat(),
            profit_loss=profit_loss,
            profit_pct=profit_pct,
            outcome=outcome
        )
        
        # Update session
        self.session.positions.remove(position)
        self.session.closed_trades.append(closed)
        self.session.current_balance += exit_value
        self.session.total_trades += 1
        
        if profit_loss > 0:
            self.session.wins += 1
        else:
            self.session.losses += 1
        
        self._save_session()
        return closed
    
    def update_positions(self, fetcher) -> List[ClosedPosition]:
        """
        Check all positions against current prices.
        Auto-close if target or stop hit.
        
        Args:
            fetcher: DataFetcher instance
            
        Returns:
            List of auto-closed positions
        """
        if self.session is None:
            return []
        
        closed = []
        
        for position in list(self.session.positions):
            try:
                quote = fetcher.get_realtime_quote(position.ticker)
                if quote is None:
                    continue
                
                current_price = quote.get('price')
                high = quote.get('high')
                low = quote.get('low')
                
                if current_price is None:
                    continue
                
                # Update position value
                position.position_value = position.quantity * current_price
                
                # Check exit conditions
                if position.signal_type == 'BUY':
                    if high and high >= position.target_price:
                        result = self.close_position(
                            position.ticker,
                            position.target_price,
                            'WIN'
                        )
                        if result:
                            closed.append(result)
                    elif low and low <= position.stop_loss:
                        result = self.close_position(
                            position.ticker,
                            position.stop_loss,
                            'LOSS'
                        )
                        if result:
                            closed.append(result)
                
                elif position.signal_type == 'SELL':
                    if low and low <= position.target_price:
                        result = self.close_position(
                            position.ticker,
                            position.target_price,
                            'WIN'
                        )
                        if result:
                            closed.append(result)
                    elif high and high >= position.stop_loss:
                        result = self.close_position(
                            position.ticker,
                            position.stop_loss,
                            'LOSS'
                        )
                        if result:
                            closed.append(result)
                            
            except Exception as e:
                print(f"Error updating {position.ticker}: {e}")
        
        self._save_session()
        return closed
    
    def get_status(self) -> str:
        """
        Get formatted status of current session.
        
        Returns:
            Formatted string for display
        """
        if self.session is None:
            return "No active paper trading session."
        
        lines = []
        lines.append("")
        lines.append("📄 PAPER TRADING SESSION")
        lines.append("═" * 50)
        lines.append(f"Session ID: {self.session.session_id}")
        lines.append(f"Started: {self.session.start_time}")
        lines.append("")
        lines.append("💰 ACCOUNT")
        lines.append("─" * 30)
        lines.append(f"Initial Balance:  ${self.session.initial_balance:,.2f}")
        lines.append(f"Cash Balance:     ${self.session.current_balance:,.2f}")
        lines.append(f"Portfolio Value:  ${self.session.portfolio_value:,.2f}")
        
        pnl_color = "+" if self.session.total_pnl >= 0 else ""
        lines.append(f"Total P&L:        {pnl_color}${self.session.total_pnl:,.2f} "
                    f"({self.session.total_pnl_pct:+.2f}%)")
        lines.append("")
        
        # Open positions
        if self.session.positions:
            lines.append("📈 OPEN POSITIONS")
            lines.append("─" * 50)
            for pos in self.session.positions:
                lines.append(
                    f"  {pos.ticker:6s} {pos.signal_type:4s} "
                    f"Entry: ${pos.entry_price:.2f} "
                    f"Target: ${pos.target_price:.2f} "
                    f"Stop: ${pos.stop_loss:.2f}"
                )
            lines.append("")
        else:
            lines.append("📈 No open positions")
            lines.append("")
        
        # Stats
        lines.append("📊 STATISTICS")
        lines.append("─" * 30)
        lines.append(f"Total Trades:     {self.session.total_trades}")
        lines.append(f"Wins:             {self.session.wins}")
        lines.append(f"Losses:           {self.session.losses}")
        lines.append(f"Win Rate:         {self.session.win_rate:.1f}%")
        lines.append("")
        
        # Recent trades
        if self.session.closed_trades:
            lines.append("📋 RECENT TRADES")
            lines.append("─" * 50)
            for trade in self.session.closed_trades[-5:]:
                emoji = "✅" if trade.profit_loss > 0 else "❌"
                lines.append(
                    f"{emoji} {trade.ticker:6s} "
                    f"${trade.entry_price:.2f} → ${trade.exit_price:.2f} "
                    f"({trade.profit_pct:+.2f}%)"
                )
            lines.append("")
        
        return "\n".join(lines)
    
    def reset_session(self) -> PaperTradeSession:
        """
        Reset and start a new session.
        
        Returns:
            New PaperTradeSession
        """
        if os.path.exists(self.SESSION_FILE):
            os.remove(self.SESSION_FILE)
        
        return self.start_session()
