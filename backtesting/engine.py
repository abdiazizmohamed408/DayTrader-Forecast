"""
Backtesting Engine.
Simulates trading signals on historical data to evaluate strategy performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class Trade:
    """Represents a simulated trade."""
    ticker: str
    signal_type: str  # BUY or SELL
    entry_date: str
    entry_price: float
    target_price: float
    stop_loss: float
    probability: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # WIN, LOSS, TIMEOUT
    profit_pct: float = 0.0
    profit_usd: float = 0.0
    holding_days: int = 0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: str
    end_date: str
    days_tested: int
    initial_balance: float
    final_balance: float
    total_return_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_profit_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display."""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'days_tested': self.days_tested,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'total_return_pct': self.total_return_pct,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'avg_profit_pct': self.avg_profit_pct,
            'avg_loss_pct': self.avg_loss_pct,
            'profit_factor': self.profit_factor,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio
        }


class BacktestEngine:
    """
    Backtesting engine for trading signals.
    
    Simulates what signals WOULD have been generated on historical data
    and tracks simulated P&L.
    """
    
    def __init__(
        self,
        config: Dict,
        initial_balance: float = 10000.0,
        position_size_pct: float = 10.0,
        max_holding_days: int = 5
    ):
        """
        Initialize backtesting engine.
        
        Args:
            config: Configuration dictionary
            initial_balance: Starting virtual balance
            position_size_pct: Percentage of balance per trade
            max_holding_days: Maximum days to hold a position
        """
        self.config = config
        self.initial_balance = initial_balance
        self.position_size_pct = position_size_pct
        self.max_holding_days = max_holding_days
    
    def run_backtest(
        self,
        tickers: List[str],
        days: int = 30,
        fetcher=None,
        analyzer=None,
        signal_gen=None,
        min_probability: float = 50.0
    ) -> BacktestResult:
        """
        Run a backtest simulation.
        
        Args:
            tickers: List of stock symbols to test
            days: Number of historical days to test
            fetcher: DataFetcher instance
            analyzer: TechnicalAnalyzer instance
            signal_gen: SignalGenerator instance
            min_probability: Minimum probability threshold
            
        Returns:
            BacktestResult with simulation results
        """
        from data.fetcher import DataFetcher
        from analyzers.technical import TechnicalAnalyzer
        from analyzers.signals import SignalGenerator, SignalType
        
        if fetcher is None:
            fetcher = DataFetcher()
        if analyzer is None:
            analyzer = TechnicalAnalyzer(self.config)
        if signal_gen is None:
            signal_gen = SignalGenerator(self.config)
        
        # Calculate period needed (add buffer for indicator calculations)
        period = f"{days + 60}d"
        
        all_trades: List[Trade] = []
        balance = self.initial_balance
        equity_curve = [balance]
        
        for ticker in tickers:
            # Fetch historical data
            data = fetcher.get_stock_data(ticker, period=period, interval='1d')
            if data is None or len(data) < 60:
                continue
            
            # Get data for the test period
            test_start_idx = len(data) - days
            if test_start_idx < 50:
                test_start_idx = 50
            
            # Simulate day by day
            for i in range(test_start_idx, len(data) - 1):
                # Get data up to this point (simulating we only know past data)
                historical_data = data.iloc[:i+1]
                
                # Analyze
                analysis = analyzer.analyze(historical_data)
                if analysis is None:
                    continue
                
                # Generate signal
                signal = signal_gen.generate_signal(ticker, analysis)
                if signal is None:
                    continue
                
                # Check if signal meets criteria
                if signal.signal_type == SignalType.HOLD:
                    continue
                if signal.probability < min_probability:
                    continue
                
                # Simulate the trade
                trade = self._simulate_trade(
                    ticker=ticker,
                    signal_type=signal.signal_type.value,
                    entry_date=str(data.index[i].date()),
                    entry_price=signal.entry_price,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    probability=signal.probability,
                    future_data=data.iloc[i+1:],
                    balance=balance
                )
                
                if trade:
                    all_trades.append(trade)
                    balance += trade.profit_usd
                    equity_curve.append(balance)
        
        # Calculate results
        return self._calculate_results(
            trades=all_trades,
            equity_curve=equity_curve,
            days=days
        )
    
    def _simulate_trade(
        self,
        ticker: str,
        signal_type: str,
        entry_date: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        probability: float,
        future_data: pd.DataFrame,
        balance: float
    ) -> Optional[Trade]:
        """
        Simulate a single trade on future data.
        
        Args:
            ticker: Stock symbol
            signal_type: BUY or SELL
            entry_date: Trade entry date
            entry_price: Entry price
            target_price: Target price
            stop_loss: Stop loss price
            probability: Signal probability
            future_data: DataFrame with future price data
            balance: Current account balance
            
        Returns:
            Trade object with outcome, or None if no data
        """
        if len(future_data) == 0:
            return None
        
        # Calculate position size
        position_value = balance * (self.position_size_pct / 100)
        shares = position_value / entry_price
        
        trade = Trade(
            ticker=ticker,
            signal_type=signal_type,
            entry_date=entry_date,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            probability=probability
        )
        
        # Check each day for outcome
        for day_idx, (date, row) in enumerate(future_data.iterrows()):
            if day_idx >= self.max_holding_days:
                # Timeout - exit at close
                trade.exit_date = str(date.date())
                trade.exit_price = row['close']
                trade.outcome = 'TIMEOUT'
                trade.holding_days = day_idx + 1
                break
            
            high = row['high']
            low = row['low']
            close = row['close']
            
            if signal_type == 'BUY':
                # Check if target hit
                if high >= target_price:
                    trade.exit_date = str(date.date())
                    trade.exit_price = target_price
                    trade.outcome = 'WIN'
                    trade.holding_days = day_idx + 1
                    break
                # Check if stop hit
                elif low <= stop_loss:
                    trade.exit_date = str(date.date())
                    trade.exit_price = stop_loss
                    trade.outcome = 'LOSS'
                    trade.holding_days = day_idx + 1
                    break
                    
            elif signal_type == 'SELL':
                # Check if target hit (price goes down)
                if low <= target_price:
                    trade.exit_date = str(date.date())
                    trade.exit_price = target_price
                    trade.outcome = 'WIN'
                    trade.holding_days = day_idx + 1
                    break
                # Check if stop hit (price goes up)
                elif high >= stop_loss:
                    trade.exit_date = str(date.date())
                    trade.exit_price = stop_loss
                    trade.outcome = 'LOSS'
                    trade.holding_days = day_idx + 1
                    break
        
        # If no outcome yet, exit at last close
        if trade.outcome is None:
            last_date = future_data.index[-1]
            trade.exit_date = str(last_date.date())
            trade.exit_price = future_data['close'].iloc[-1]
            trade.outcome = 'TIMEOUT'
            trade.holding_days = len(future_data)
        
        # Calculate profit
        if signal_type == 'BUY':
            trade.profit_pct = ((trade.exit_price - entry_price) / entry_price) * 100
        else:
            trade.profit_pct = ((entry_price - trade.exit_price) / entry_price) * 100
        
        trade.profit_usd = position_value * (trade.profit_pct / 100)
        
        return trade
    
    def _calculate_results(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        days: int
    ) -> BacktestResult:
        """
        Calculate backtest statistics.
        
        Args:
            trades: List of completed trades
            equity_curve: List of balance values over time
            days: Number of days tested
            
        Returns:
            BacktestResult with all statistics
        """
        if not trades:
            return BacktestResult(
                start_date='',
                end_date='',
                days_tested=days,
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                total_return_pct=0,
                total_trades=0,
                wins=0,
                losses=0,
                win_rate=0,
                avg_profit_pct=0,
                avg_loss_pct=0,
                profit_factor=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                trades=[],
                equity_curve=[self.initial_balance]
            )
        
        # Sort trades by date
        trades.sort(key=lambda t: t.entry_date)
        
        # Basic stats
        start_date = trades[0].entry_date
        end_date = trades[-1].exit_date or trades[-1].entry_date
        final_balance = equity_curve[-1] if equity_curve else self.initial_balance
        
        wins = [t for t in trades if t.outcome == 'WIN']
        losses = [t for t in trades if t.outcome == 'LOSS']
        
        win_count = len(wins)
        loss_count = len(losses)
        total_trades = len(trades)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Average profit/loss
        avg_profit = np.mean([t.profit_pct for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.profit_pct for t in losses])) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.profit_pct for t in wins)
        gross_loss = abs(sum(t.profit_pct for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Total return
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Sharpe ratio (simplified)
        returns = [t.profit_pct for t in trades]
        sharpe = self._calculate_sharpe(returns)
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            days_tested=days,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return_pct=total_return,
            total_trades=total_trades,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            avg_profit_pct=avg_profit,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if len(equity_curve) < 2:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate simplified Sharpe ratio."""
        if len(returns) < 2:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualized (assuming daily returns)
        sharpe = (avg_return - (risk_free_rate / 252)) / std_return * np.sqrt(252)
        return sharpe
    
    def print_results(self, result: BacktestResult) -> str:
        """
        Format backtest results for display.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Formatted string for console output
        """
        lines = []
        lines.append("")
        lines.append("📊 BACKTEST RESULTS")
        lines.append("═" * 50)
        lines.append("")
        lines.append(f"📅 Period: {result.start_date} to {result.end_date}")
        lines.append(f"📆 Days Tested: {result.days_tested}")
        lines.append("")
        lines.append("💰 PERFORMANCE")
        lines.append("─" * 30)
        lines.append(f"Initial Balance:  ${result.initial_balance:,.2f}")
        lines.append(f"Final Balance:    ${result.final_balance:,.2f}")
        lines.append(f"Total Return:     {result.total_return_pct:+.2f}%")
        lines.append(f"Max Drawdown:     {result.max_drawdown_pct:.2f}%")
        lines.append("")
        lines.append("📈 TRADE STATISTICS")
        lines.append("─" * 30)
        lines.append(f"Total Trades:     {result.total_trades}")
        lines.append(f"Wins:             {result.wins}")
        lines.append(f"Losses:           {result.losses}")
        lines.append(f"Win Rate:         {result.win_rate:.1f}%")
        lines.append("")
        lines.append(f"Avg Profit:       {result.avg_profit_pct:+.2f}%")
        lines.append(f"Avg Loss:         {result.avg_loss_pct:-.2f}%")
        lines.append(f"Profit Factor:    {result.profit_factor:.2f}")
        lines.append(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        lines.append("")
        
        # Recent trades
        if result.trades:
            lines.append("📋 RECENT TRADES")
            lines.append("─" * 50)
            for trade in result.trades[-10:]:
                emoji = "✅" if trade.outcome == 'WIN' else "❌" if trade.outcome == 'LOSS' else "⏱️"
                lines.append(
                    f"{emoji} {trade.ticker:6s} {trade.signal_type:4s} "
                    f"${trade.entry_price:.2f} → ${trade.exit_price:.2f} "
                    f"({trade.profit_pct:+.2f}%)"
                )
        
        lines.append("")
        return "\n".join(lines)
