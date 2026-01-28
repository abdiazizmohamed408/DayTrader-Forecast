"""
Backtesting Engine.
Tests trading strategy on historical data to validate performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

import pandas as pd
import numpy as np

from analyzers.technical import TechnicalAnalyzer
from analyzers.signals import SignalGenerator, SignalType


@dataclass
class BacktestTrade:
    """A simulated historical trade."""
    ticker: str
    signal_type: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime]
    exit_price: Optional[float]
    stop_loss: float
    target_price: float
    probability: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None  # TARGET, STOP_LOSS, TIME_EXIT
    days_held: Optional[int] = None


@dataclass
class BacktestResult:
    """Complete backtest results."""
    ticker: str
    period_start: datetime
    period_end: datetime
    total_days: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L statistics
    total_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    
    # Trade details
    avg_days_held: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Individual trades
    trades: List[BacktestTrade]
    
    # Equity curve
    equity_curve: List[Tuple[datetime, float]]


class BacktestEngine:
    """
    Backtesting engine for validating trading strategies.
    
    Tests the signal generator on historical data to measure
    what the system WOULD have done in the past.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtest engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tech_analyzer = TechnicalAnalyzer(config)
        self.signal_gen = SignalGenerator(config)
        
        # Backtest settings
        bt_config = config.get('backtest', {})
        self.max_hold_days = bt_config.get('max_hold_days', 10)
        self.min_probability = bt_config.get('min_probability', 0)
    
    def run_backtest(
        self,
        ticker: str,
        data: pd.DataFrame,
        days: int = 30,
        starting_capital: float = 10000
    ) -> Optional[BacktestResult]:
        """
        Run backtest on historical data.
        
        Args:
            ticker: Stock symbol
            data: Historical OHLCV data
            days: Number of days to backtest
            starting_capital: Starting capital for equity curve
            
        Returns:
            BacktestResult or None if insufficient data
        """
        if data is None or len(data) < 60:
            return None
        
        # Use last N days
        if len(data) > days + 50:
            backtest_data = data.tail(days + 50)
        else:
            backtest_data = data
            days = len(data) - 50
        
        trades: List[BacktestTrade] = []
        equity = starting_capital
        equity_curve = []
        
        active_trade: Optional[BacktestTrade] = None
        
        # Iterate through each day
        for i in range(50, len(backtest_data)):
            current_date = backtest_data.index[i]
            current_row = backtest_data.iloc[i]
            current_price = current_row['close']
            current_high = current_row['high']
            current_low = current_row['low']
            
            # Get historical data up to this point
            historical_data = backtest_data.iloc[:i+1]
            
            # Check if we have an active trade
            if active_trade:
                days_held = (current_date - active_trade.entry_date).days
                
                # Check exit conditions
                should_exit = False
                exit_reason = None
                exit_price = None
                
                if active_trade.signal_type == 'BUY':
                    if current_low <= active_trade.stop_loss:
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                        exit_price = active_trade.stop_loss
                    elif current_high >= active_trade.target_price:
                        should_exit = True
                        exit_reason = 'TARGET'
                        exit_price = active_trade.target_price
                else:  # SELL
                    if current_high >= active_trade.stop_loss:
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                        exit_price = active_trade.stop_loss
                    elif current_low <= active_trade.target_price:
                        should_exit = True
                        exit_reason = 'TARGET'
                        exit_price = active_trade.target_price
                
                # Time-based exit
                if days_held >= self.max_hold_days:
                    should_exit = True
                    exit_reason = 'TIME_EXIT'
                    exit_price = current_price
                
                if should_exit:
                    # Calculate P&L
                    if active_trade.signal_type == 'BUY':
                        pnl_pct = ((exit_price - active_trade.entry_price) / 
                                  active_trade.entry_price) * 100
                    else:
                        pnl_pct = ((active_trade.entry_price - exit_price) / 
                                  active_trade.entry_price) * 100
                    
                    pnl = (pnl_pct / 100) * (equity * 0.1)  # Assuming 10% position size
                    
                    active_trade.exit_date = current_date
                    active_trade.exit_price = exit_price
                    active_trade.pnl = pnl
                    active_trade.pnl_pct = pnl_pct
                    active_trade.exit_reason = exit_reason
                    active_trade.days_held = days_held
                    
                    trades.append(active_trade)
                    equity += pnl
                    active_trade = None
            
            # Look for new signals if no active trade
            if not active_trade:
                analysis = self.tech_analyzer.analyze(historical_data)
                
                if analysis:
                    signal = self.signal_gen.generate_signal(ticker, analysis)
                    
                    if signal and signal.signal_type != SignalType.HOLD:
                        if signal.probability >= self.min_probability:
                            active_trade = BacktestTrade(
                                ticker=ticker,
                                signal_type=signal.signal_type.value,
                                entry_date=current_date,
                                entry_price=current_price,
                                exit_date=None,
                                exit_price=None,
                                stop_loss=signal.stop_loss,
                                target_price=signal.target_price,
                                probability=signal.probability
                            )
            
            equity_curve.append((current_date, equity))
        
        # Close any remaining active trade
        if active_trade:
            current_price = backtest_data['close'].iloc[-1]
            current_date = backtest_data.index[-1]
            
            if active_trade.signal_type == 'BUY':
                pnl_pct = ((current_price - active_trade.entry_price) / 
                          active_trade.entry_price) * 100
            else:
                pnl_pct = ((active_trade.entry_price - current_price) / 
                          active_trade.entry_price) * 100
            
            pnl = (pnl_pct / 100) * (equity * 0.1)
            
            active_trade.exit_date = current_date
            active_trade.exit_price = current_price
            active_trade.pnl = pnl
            active_trade.pnl_pct = pnl_pct
            active_trade.exit_reason = 'END_OF_BACKTEST'
            active_trade.days_held = (current_date - active_trade.entry_date).days
            
            trades.append(active_trade)
            equity += pnl
        
        # Calculate statistics
        if not trades:
            return BacktestResult(
                ticker=ticker,
                period_start=backtest_data.index[50],
                period_end=backtest_data.index[-1],
                total_days=days,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl_pct=0,
                avg_win_pct=0,
                avg_loss_pct=0,
                profit_factor=0,
                max_drawdown_pct=0,
                avg_days_held=0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                trades=[],
                equity_curve=equity_curve
            )
        
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        
        win_rate = (len(wins) / len(trades)) * 100
        
        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
        
        gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        total_pnl_pct = ((equity - starting_capital) / starting_capital) * 100
        
        # Calculate max drawdown
        peak = starting_capital
        max_drawdown = 0
        for date, eq in equity_curve:
            if eq > peak:
                peak = eq
            drawdown = (peak - eq) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        avg_days_held = np.mean([t.days_held for t in trades if t.days_held])
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_streak = 0
        last_was_win = None
        
        for trade in trades:
            is_win = trade.pnl_pct > 0
            if is_win == last_was_win:
                current_streak += 1
            else:
                current_streak = 1
            
            if is_win:
                max_consec_wins = max(max_consec_wins, current_streak)
            else:
                max_consec_losses = max(max_consec_losses, current_streak)
            
            last_was_win = is_win
        
        return BacktestResult(
            ticker=ticker,
            period_start=backtest_data.index[50] if hasattr(backtest_data.index[50], 'to_pydatetime') 
                        else backtest_data.index[50],
            period_end=backtest_data.index[-1] if hasattr(backtest_data.index[-1], 'to_pydatetime')
                      else backtest_data.index[-1],
            total_days=days,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl_pct=total_pnl_pct,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown,
            avg_days_held=avg_days_held,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def run_portfolio_backtest(
        self,
        tickers: List[str],
        fetcher,
        days: int = 30
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest on multiple tickers.
        
        Args:
            tickers: List of stock symbols
            fetcher: DataFetcher instance
            days: Number of days to backtest
            
        Returns:
            Dictionary of ticker to BacktestResult
        """
        results = {}
        
        for ticker in tickers:
            data = fetcher.get_stock_data(ticker, period="6mo", interval="1d")
            if data is not None:
                result = self.run_backtest(ticker, data, days)
                if result:
                    results[ticker] = result
        
        return results
    
    def generate_report(self, results: Dict[str, BacktestResult]) -> str:
        """
        Generate backtest report.
        
        Args:
            results: Dictionary of backtest results
            
        Returns:
            Markdown formatted report
        """
        report = f"""# 📊 Backtest Report

**Generated:** {datetime.now().strftime("%B %d, %Y %H:%M")}

---

## Summary

| Ticker | Trades | Win Rate | Total P&L | Profit Factor | Max DD |
|--------|--------|----------|-----------|---------------|--------|
"""
        
        total_trades = 0
        total_wins = 0
        
        for ticker, result in sorted(results.items(), key=lambda x: x[1].win_rate, reverse=True):
            report += (
                f"| {ticker} | {result.total_trades} | {result.win_rate:.1f}% | "
                f"{result.total_pnl_pct:+.1f}% | {result.profit_factor:.2f} | "
                f"{result.max_drawdown_pct:.1f}% |\n"
            )
            total_trades += result.total_trades
            total_wins += result.winning_trades
        
        # Overall stats
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = np.mean([r.total_pnl_pct for r in results.values()]) if results else 0
        
        report += f"""
## Overall Performance

- **Total Trades:** {total_trades}
- **Overall Win Rate:** {overall_win_rate:.1f}%
- **Average P&L per Ticker:** {avg_pnl:+.1f}%

---

## Trade Details

"""
        
        for ticker, result in results.items():
            if result.trades:
                report += f"### {ticker}\n\n"
                report += "| Date | Type | Entry | Exit | P&L | Reason |\n"
                report += "|------|------|-------|------|-----|--------|\n"
                
                for trade in result.trades[-10:]:  # Last 10 trades
                    entry_date = trade.entry_date.strftime("%m/%d") if hasattr(trade.entry_date, 'strftime') else str(trade.entry_date)[:5]
                    report += (
                        f"| {entry_date} | {trade.signal_type} | "
                        f"${trade.entry_price:.2f} | ${trade.exit_price:.2f} | "
                        f"{trade.pnl_pct:+.1f}% | {trade.exit_reason} |\n"
                    )
                
                report += "\n"
        
        report += """
---

*Past performance does not guarantee future results.*
"""
        
        return report
