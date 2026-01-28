#!/usr/bin/env python3
"""
DayTrader-Forecast - Professional Day Trading Analysis Tool

A bulletproof technical analysis system with:
- Multi-timeframe analysis
- Market context awareness
- Risk management
- News/earnings filtering
- Paper trading simulation
- Backtesting engine
- Email alerts

⚠️  DISCLAIMER: This is for educational purposes only. Not financial advice.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from colorama import Fore, Style, init as colorama_init

from analyzers.technical import TechnicalAnalyzer
from analyzers.signals import SignalGenerator, SignalType, TradingSignal
from analyzers.multi_timeframe import MultiTimeframeAnalyzer
from analyzers.market_context import MarketContextAnalyzer, MarketRegime
from analyzers.risk_manager import RiskManager
from analyzers.news_filter import NewsAndEarningsFilter
from analyzers.premarket import PremarketScanner
from backtesting.engine import BacktestEngine
from data.fetcher import DataFetcher
from paper.simulator import PaperTrader
from reports.generator import ReportGenerator
from tracking.tracker import PredictionTracker
from utils.helpers import load_config, format_currency, ensure_dir
from utils.alerts import AlertSystem
from utils.validator import DataValidator, ErrorHandler


colorama_init()

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   {Fore.YELLOW}📈 DayTrader-Forecast PRO{Fore.CYAN}                                      ║
║   {Fore.WHITE}Professional Day Trading Analysis System{Fore.CYAN}                       ║
║                                                                  ║
║   {Fore.GREEN}✓ Multi-Timeframe  ✓ Risk Management  ✓ Backtesting{Fore.CYAN}            ║
║   {Fore.GREEN}✓ Market Context   ✓ News Filter      ✓ Paper Trading{Fore.CYAN}          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

DISCLAIMER = f"""
{Fore.RED}⚠️  DISCLAIMER:{Style.RESET_ALL}
This tool is for EDUCATIONAL PURPOSES ONLY.
• This is NOT financial advice
• Day trading involves SIGNIFICANT RISK
• Past performance does NOT guarantee future results
• Never trade with money you cannot afford to lose
"""


def get_db_path() -> str:
    db_dir = Path("./data")
    db_dir.mkdir(exist_ok=True)
    return str(db_dir / "predictions.db")


def print_banner():
    print(BANNER)
    print(DISCLAIMER)
    print()


def format_signal_line(signal: TradingSignal, confidence: str = "") -> str:
    if signal.signal_type == SignalType.BUY:
        color = Fore.GREEN
        emoji = "🟢"
    elif signal.signal_type == SignalType.SELL:
        color = Fore.RED
        emoji = "🔴"
    else:
        color = Fore.YELLOW
        emoji = "🟡"
    
    conf_str = f" {Fore.CYAN}[{confidence}]{Style.RESET_ALL}" if confidence else ""
    
    return (
        f"{emoji} {color}{signal.ticker:6s}{Style.RESET_ALL} │ "
        f"{signal.signal_type.value:4s} │ "
        f"{signal.probability:5.1f}% │ "
        f"{format_currency(signal.entry_price):>10s}{conf_str}"
    )


def cmd_scan(config: Dict, args: argparse.Namespace) -> int:
    """Enhanced scan with all filters."""
    print(f"\n{Fore.CYAN}📊 Professional Market Scan{Style.RESET_ALL}\n")
    
    watchlist = config.get('watchlist', [])
    filters = config.get('filters', {})
    min_prob = filters.get('high_confidence_threshold', 65)
    
    # Initialize all analyzers
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer(config)
    signal_gen = SignalGenerator(config)
    market_analyzer = MarketContextAnalyzer(config)
    mtf_analyzer = MultiTimeframeAnalyzer(config)
    news_filter = NewsAndEarningsFilter(config)
    risk_manager = RiskManager(config)
    tracker = PredictionTracker(get_db_path())
    validator = DataValidator(config)
    alert_system = AlertSystem(config)
    
    # Check market context first
    print(f"  {Fore.CYAN}Analyzing market context...{Style.RESET_ALL}")
    market_context = market_analyzer.analyze_market(fetcher)
    
    print(f"  Market: {market_context.regime.value}")
    print(f"  SPY: {market_context.spy_trend} | QQQ: {market_context.qqq_trend}")
    print(f"  Volatility: {market_context.volatility_level}")
    print()
    
    # Check if trading is allowed
    can_trade, trade_reason = risk_manager.check_can_trade()
    if not can_trade:
        print(f"{Fore.RED}⚠️ {trade_reason}{Style.RESET_ALL}\n")
    
    signals: List[TradingSignal] = []
    analyses: Dict = {}
    high_confidence: List[TradingSignal] = []
    filtered_out = 0
    
    print(f"  {Fore.CYAN}Scanning {len(watchlist)} stocks...{Style.RESET_ALL}\n")
    
    for ticker in watchlist:
        print(f"  {ticker}...", end=" ", flush=True)
        
        # Fetch and validate data
        data = fetcher.get_stock_data(ticker)
        is_valid, issues = validator.validate_ohlcv(data, ticker)
        
        if not is_valid:
            print(f"{Fore.RED}Invalid data{Style.RESET_ALL}")
            continue
        
        if issues:
            data = validator.clean_data(data)
        
        # Technical analysis
        analysis = analyzer.analyze(data)
        if analysis is None:
            print(f"{Fore.YELLOW}Skip{Style.RESET_ALL}")
            continue
        
        # Generate base signal
        signal = signal_gen.generate_signal(ticker, analysis)
        if signal is None:
            print(f"{Fore.YELLOW}No signal{Style.RESET_ALL}")
            continue
        
        # Apply filters
        confidence_notes = []
        adjusted_probability = signal.probability
        
        # 1. Market context filter
        if filters.get('respect_market_context', True):
            is_valid_context, context_reason = market_analyzer.filter_signal(
                signal.signal_type.value, market_context
            )
            if not is_valid_context:
                print(f"{Fore.YELLOW}Filtered: {context_reason}{Style.RESET_ALL}")
                filtered_out += 1
                continue
            adjusted_probability += market_context.sentiment_adjustment
        
        # 2. Volume confirmation
        if filters.get('require_volume_confirmation', True):
            volume = analysis.get('volume', {})
            vol_ratio = volume.get('volume_ratio', 1.0)
            if vol_ratio < filters.get('min_volume_ratio', 0.8):
                print(f"{Fore.YELLOW}Low volume{Style.RESET_ALL}")
                filtered_out += 1
                continue
            if vol_ratio > 1.5:
                confidence_notes.append("High Vol")
                adjusted_probability += 5
        
        # 3. News/Earnings filter
        event_filter = news_filter.filter_stock(ticker)
        if event_filter.should_avoid:
            print(f"{Fore.YELLOW}Event risk: {', '.join(event_filter.reasons)}{Style.RESET_ALL}")
            filtered_out += 1
            continue
        if event_filter.risk_level in ['HIGH', 'MEDIUM']:
            confidence_notes.append(event_filter.risk_level)
        
        # 4. Multi-timeframe analysis (if enabled)
        if config.get('multi_tf', {}).get('enabled', True) and signal.signal_type != SignalType.HOLD:
            mtf_result = mtf_analyzer.analyze_all_timeframes(ticker, fetcher)
            if mtf_result:
                if mtf_result.is_valid_signal:
                    boost = mtf_analyzer.get_confluence_boost(mtf_result)
                    adjusted_probability += boost
                    confidence_notes.append(f"{mtf_result.confluence_count}TF")
                else:
                    adjusted_probability -= 10
        
        # 5. Indicator agreement check
        scores = signal_gen.get_last_scores()
        if scores:
            bullish_count = sum(1 for s in scores.values() if s.get('bullish', 0) > 0.5)
            bearish_count = sum(1 for s in scores.values() if s.get('bearish', 0) > 0.5)
            agreement = max(bullish_count, bearish_count)
            
            min_agreement = filters.get('min_indicator_agreement', 3)
            if agreement < min_agreement:
                adjusted_probability -= 10
        
        # Cap probability
        adjusted_probability = max(0, min(95, adjusted_probability))
        
        # Create adjusted signal
        adjusted_signal = TradingSignal(
            ticker=signal.ticker,
            signal_type=signal.signal_type,
            probability=adjusted_probability,
            sentiment=signal.sentiment,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            risk_reward_ratio=signal.risk_reward_ratio,
            reasons=signal.reasons
        )
        
        signals.append(adjusted_signal)
        analyses[ticker] = analysis
        
        # Track high confidence signals
        if adjusted_probability >= min_prob and adjusted_signal.signal_type != SignalType.HOLD:
            high_confidence.append(adjusted_signal)
            confidence_notes.append("⭐")
        
        # Log prediction
        if adjusted_signal.signal_type != SignalType.HOLD and not args.no_track:
            tracker.log_signal(
                ticker=adjusted_signal.ticker,
                signal_type=adjusted_signal.signal_type.value,
                entry_price=adjusted_signal.entry_price,
                target_price=adjusted_signal.target_price,
                stop_loss=adjusted_signal.stop_loss,
                probability=adjusted_probability,
                reasons=adjusted_signal.reasons
            )
        
        conf_str = " ".join(confidence_notes)
        print(f"{Fore.GREEN}Done{Style.RESET_ALL} {conf_str}")
    
    # Print results
    print(f"\n{Fore.CYAN}{'═' * 65}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📈 SCAN RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 65}{Style.RESET_ALL}\n")
    
    if filtered_out > 0:
        print(f"  {Fore.YELLOW}ℹ️ {filtered_out} stocks filtered out by safety checks{Style.RESET_ALL}\n")
    
    signals.sort(key=lambda x: x.probability, reverse=True)
    
    print(f"{'TICKER':<8} │ {'SIG':4s} │ {'PROB':>5s} │ {'PRICE':>10s}")
    print(f"{'─' * 8}─┼─{'─' * 4}─┼─{'─' * 5}─┼─{'─' * 10}")
    
    for signal in signals:
        conf = "⭐ HIGH" if signal.probability >= min_prob and signal.signal_type != SignalType.HOLD else ""
        print(format_signal_line(signal, conf))
    
    print()
    
    # Summary
    buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
    sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
    hold_count = sum(1 for s in signals if s.signal_type == SignalType.HOLD)
    
    print(f"{Fore.GREEN}🟢 BUY:{Style.RESET_ALL} {buy_count}  │  "
          f"{Fore.RED}🔴 SELL:{Style.RESET_ALL} {sell_count}  │  "
          f"{Fore.YELLOW}🟡 HOLD:{Style.RESET_ALL} {hold_count}")
    
    # High confidence alerts
    if high_confidence:
        print(f"\n{Fore.CYAN}⭐ HIGH CONFIDENCE SIGNALS ({len(high_confidence)}):{Style.RESET_ALL}")
        for sig in high_confidence:
            emoji = "🟢" if sig.signal_type == SignalType.BUY else "🔴"
            print(f"   {emoji} {sig.ticker}: {sig.signal_type.value} @ {format_currency(sig.entry_price)} ({sig.probability:.0f}%)")
        
        # Send alerts
        if args.alert and alert_system.is_configured():
            for sig in high_confidence:
                alert_system.send_signal_alert(
                    ticker=sig.ticker,
                    signal_type=sig.signal_type.value,
                    probability=sig.probability,
                    entry_price=sig.entry_price,
                    stop_loss=sig.stop_loss,
                    target_price=sig.target_price,
                    reasons=sig.reasons,
                    additional_info={'market_context': market_context.regime.value}
                )
            print(f"\n{Fore.GREEN}📧 Email alerts sent!{Style.RESET_ALL}")
    
    print()
    return 0


def cmd_premarket(config: Dict, args: argparse.Namespace) -> int:
    """Pre-market scanner."""
    print(f"\n{Fore.CYAN}🌅 Pre-Market Scanner{Style.RESET_ALL}\n")
    
    fetcher = DataFetcher()
    scanner = PremarketScanner(config)
    watchlist = config.get('watchlist', [])
    
    scan = scanner.scan_watchlist(watchlist)
    
    print(f"  Market Status: {scan.market_status}")
    print(f"  Scan Time: {scan.scan_time.strftime('%H:%M:%S')}")
    print()
    
    if scan.gap_ups:
        print(f"{Fore.GREEN}📈 GAP UPS:{Style.RESET_ALL}")
        for g in scan.gap_ups[:5]:
            vol = f"{g.volume_ratio:.1f}x vol" if g.is_unusual_volume else ""
            print(f"   {g.ticker}: +{g.gap_percent:.1f}% {vol}")
        print()
    
    if scan.gap_downs:
        print(f"{Fore.RED}📉 GAP DOWNS:{Style.RESET_ALL}")
        for g in scan.gap_downs[:5]:
            vol = f"{g.volume_ratio:.1f}x vol" if g.is_unusual_volume else ""
            print(f"   {g.ticker}: {g.gap_percent:.1f}% {vol}")
        print()
    
    if scan.unusual_volume:
        print(f"{Fore.CYAN}📊 UNUSUAL VOLUME:{Style.RESET_ALL}")
        for u in scan.unusual_volume[:5]:
            print(f"   {u.ticker}: {u.volume_ratio:.1f}x average volume")
        print()
    
    # Trading signals
    signals = scanner.get_gapper_signals(scan)
    if signals:
        print(f"{Fore.YELLOW}💡 POTENTIAL SETUPS:{Style.RESET_ALL}")
        for s in signals[:5]:
            print(f"   {s['ticker']}: {s['strategy']} ({s['gap_percent']:+.1f}%)")
        print()
    
    return 0


def cmd_paper(config: Dict, args: argparse.Namespace) -> int:
    """Paper trading commands."""
    paper = PaperTrader(
        starting_balance=config.get('paper_trading', {}).get('starting_balance', 10000)
    )
    
    if args.paper_action == 'status':
        account = paper.get_account_status()
        
        print(f"\n{Fore.CYAN}📄 Paper Trading Account{Style.RESET_ALL}\n")
        print(f"  Starting Balance: {format_currency(account.starting_balance)}")
        print(f"  Current Balance:  {format_currency(account.current_balance)}")
        
        pnl_color = Fore.GREEN if account.total_pnl >= 0 else Fore.RED
        print(f"  Total P&L:        {pnl_color}{format_currency(account.total_pnl)} ({account.total_pnl_pct:+.1f}%){Style.RESET_ALL}")
        print()
        
        print(f"  Total Trades:     {account.total_trades}")
        print(f"  Win Rate:         {account.win_rate:.1f}%")
        print(f"  Profit Factor:    {account.profit_factor:.2f}")
        print(f"  Max Drawdown:     {account.max_drawdown:.1f}%")
        print()
        
        if account.open_positions:
            print(f"{Fore.CYAN}Open Positions:{Style.RESET_ALL}")
            for pos in account.open_positions:
                print(f"   {pos.ticker}: {pos.shares} shares @ {format_currency(pos.entry_price)}")
        print()
        
    elif args.paper_action == 'buy':
        # Buy based on latest signal
        fetcher = DataFetcher()
        analyzer = TechnicalAnalyzer(config)
        signal_gen = SignalGenerator(config)
        risk_mgr = RiskManager(config)
        
        ticker = args.ticker.upper()
        data = fetcher.get_stock_data(ticker)
        if data is None:
            print(f"{Fore.RED}❌ Could not fetch data for {ticker}{Style.RESET_ALL}")
            return 1
        
        analysis = analyzer.analyze(data)
        signal = signal_gen.generate_signal(ticker, analysis)
        
        if signal and signal.signal_type == SignalType.BUY:
            pos_size = risk_mgr.calculate_position_size(
                ticker, signal.entry_price, signal.stop_loss, signal.target_price
            )
            
            if pos_size.is_valid:
                success = paper.open_position(
                    ticker=ticker,
                    shares=pos_size.shares,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    target_price=signal.target_price,
                    signal_type='BUY'
                )
                
                if success:
                    print(f"\n{Fore.GREEN}✅ Paper BUY: {pos_size.shares} shares of {ticker} @ {format_currency(signal.entry_price)}{Style.RESET_ALL}")
                    print(f"   Stop: {format_currency(signal.stop_loss)} | Target: {format_currency(signal.target_price)}")
                else:
                    print(f"{Fore.RED}❌ Could not open position{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ {pos_size.rejection_reason}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ No BUY signal for {ticker}{Style.RESET_ALL}")
    
    elif args.paper_action == 'close':
        ticker = args.ticker.upper()
        fetcher = DataFetcher()
        quote = fetcher.get_realtime_quote(ticker)
        
        if quote and quote.get('price'):
            trade = paper.close_position(ticker, quote['price'], 'MANUAL')
            if trade:
                pnl_color = Fore.GREEN if trade['pnl'] >= 0 else Fore.RED
                print(f"\n{Fore.CYAN}Closed {ticker}:{Style.RESET_ALL}")
                print(f"   P&L: {pnl_color}{format_currency(trade['pnl'])} ({trade['pnl_pct']:+.1f}%){Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}No open position in {ticker}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Could not get price for {ticker}{Style.RESET_ALL}")
    
    elif args.paper_action == 'reset':
        paper.reset_account()
        print(f"{Fore.GREEN}✅ Paper account reset{Style.RESET_ALL}")
    
    return 0


def cmd_backtest(config: Dict, args: argparse.Namespace) -> int:
    """Run backtest on historical data."""
    print(f"\n{Fore.CYAN}📊 Running Backtest ({args.days} days){Style.RESET_ALL}\n")
    
    fetcher = DataFetcher()
    engine = BacktestEngine(config)
    watchlist = config.get('watchlist', [])
    
    results = engine.run_portfolio_backtest(watchlist, fetcher, args.days)
    
    if not results:
        print(f"{Fore.RED}❌ No backtest results{Style.RESET_ALL}")
        return 1
    
    # Print summary
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    print(f"{'TICKER':<8} {'TRADES':>7} {'WIN%':>7} {'P&L':>10} {'PF':>8} {'MAX DD':>8}")
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    
    total_trades = 0
    total_wins = 0
    total_pnl = 0
    
    for ticker, result in sorted(results.items(), key=lambda x: x[1].total_pnl_pct, reverse=True):
        wr_color = Fore.GREEN if result.win_rate >= 50 else Fore.RED
        pnl_color = Fore.GREEN if result.total_pnl_pct >= 0 else Fore.RED
        
        print(f"{ticker:<8} {result.total_trades:>7} {wr_color}{result.win_rate:>6.1f}%{Style.RESET_ALL} "
              f"{pnl_color}{result.total_pnl_pct:>+9.1f}%{Style.RESET_ALL} "
              f"{result.profit_factor:>7.2f} {result.max_drawdown_pct:>7.1f}%")
        
        total_trades += result.total_trades
        total_wins += result.winning_trades
        total_pnl += result.total_pnl_pct
    
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = total_pnl / len(results) if results else 0
    
    print(f"\n{Fore.WHITE}Summary:{Style.RESET_ALL}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Overall Win Rate: {overall_wr:.1f}%")
    print(f"  Average P&L: {avg_pnl:+.1f}%")
    
    # Save report
    if args.save:
        report = engine.generate_report(results)
        ensure_dir("./output")
        filepath = Path("./output") / f"backtest_{args.days}d.md"
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"\n{Fore.GREEN}✅ Report saved to {filepath}{Style.RESET_ALL}")
    
    print()
    return 0


def cmd_risk(config: Dict, args: argparse.Namespace) -> int:
    """Display risk status."""
    risk_mgr = RiskManager(config)
    status = risk_mgr.get_risk_status()
    
    print(f"\n{Fore.CYAN}⚠️ Risk Status{Style.RESET_ALL}\n")
    
    pnl_color = Fore.GREEN if status.daily_pnl >= 0 else Fore.RED
    print(f"  Daily P&L: {pnl_color}{format_currency(status.daily_pnl)} ({status.daily_pnl_pct:+.1f}%){Style.RESET_ALL}")
    print(f"  Trades Today: {status.trades_today} ({status.wins_today}W / {status.losses_today}L)")
    print(f"  Open Positions: {status.open_positions}")
    print(f"  Remaining Risk: {format_currency(status.remaining_risk)}")
    
    if status.at_daily_limit:
        print(f"\n{Fore.RED}🛑 DAILY LOSS LIMIT REACHED - STOP TRADING{Style.RESET_ALL}")
    
    if status.warnings:
        print(f"\n{Fore.YELLOW}Warnings:{Style.RESET_ALL}")
        for w in status.warnings:
            print(f"  {w}")
    
    print()
    return 0


def cmd_analyze(config: Dict, args: argparse.Namespace) -> int:
    """Detailed single stock analysis."""
    ticker = args.ticker.upper()
    print(f"\n{Fore.CYAN}📊 Detailed Analysis: {ticker}{Style.RESET_ALL}\n")
    
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer(config)
    signal_gen = SignalGenerator(config)
    news_filter = NewsAndEarningsFilter(config)
    risk_mgr = RiskManager(config)
    mtf_analyzer = MultiTimeframeAnalyzer(config)
    
    # Basic analysis
    data = fetcher.get_stock_data(ticker)
    if data is None:
        print(f"{Fore.RED}❌ Could not fetch data{Style.RESET_ALL}")
        return 1
    
    analysis = analyzer.analyze(data)
    signal = signal_gen.generate_signal(ticker, analysis)
    quote = fetcher.get_realtime_quote(ticker)
    
    if signal is None:
        print(f"{Fore.RED}❌ Could not generate signal{Style.RESET_ALL}")
        return 1
    
    # Signal
    if signal.signal_type == SignalType.BUY:
        color = Fore.GREEN
        emoji = "🟢"
    elif signal.signal_type == SignalType.SELL:
        color = Fore.RED
        emoji = "🔴"
    else:
        color = Fore.YELLOW
        emoji = "🟡"
    
    name = quote.get('name', ticker) if quote else ticker
    print(f"  {emoji} {color}{name}{Style.RESET_ALL}")
    print(f"  Price: {format_currency(signal.entry_price)}")
    print(f"  Signal: {color}{signal.signal_type.value}{Style.RESET_ALL} ({signal.probability:.0f}%)")
    print()
    
    # Position sizing
    if signal.signal_type != SignalType.HOLD:
        pos_size = risk_mgr.calculate_position_size(
            ticker, signal.entry_price, signal.stop_loss, signal.target_price
        )
        
        print(f"{Fore.CYAN}Position Sizing:{Style.RESET_ALL}")
        if pos_size.is_valid:
            print(f"  Shares: {pos_size.shares}")
            print(f"  Position Value: {format_currency(pos_size.position_value)}")
            print(f"  Risk: {format_currency(pos_size.risk_amount)} ({pos_size.risk_percent:.1f}%)")
            print(f"  R:R Ratio: {pos_size.risk_reward_ratio:.2f}")
        else:
            print(f"  {Fore.RED}❌ {pos_size.rejection_reason}{Style.RESET_ALL}")
        print()
    
    # News/Earnings
    events = news_filter.filter_stock(ticker)
    print(f"{Fore.CYAN}Events:{Style.RESET_ALL}")
    print(f"  Risk Level: {events.risk_level}")
    if events.earnings and events.earnings.has_upcoming_earnings:
        print(f"  Earnings: {events.earnings.days_until_earnings} days")
    for reason in events.reasons:
        print(f"  • {reason}")
    print()
    
    # Multi-timeframe
    print(f"{Fore.CYAN}Multi-Timeframe:{Style.RESET_ALL}")
    mtf_result = mtf_analyzer.analyze_all_timeframes(ticker, fetcher)
    if mtf_result:
        print(f"  Alignment: {mtf_result.alignment_score:.0f}%")
        print(f"  Confluence: {mtf_result.confluence_count} timeframes")
        print(f"  Valid Signal: {'✅' if mtf_result.is_valid_signal else '❌'}")
    print()
    
    # Technical indicators
    print(f"{Fore.CYAN}Technical Indicators:{Style.RESET_ALL}")
    print(f"  RSI(14): {analysis.get('rsi', 0):.1f}")
    print(f"  MACD: {analysis.get('macd', 0):.4f}")
    print(f"  SMA(20): {format_currency(analysis.get('sma_short', 0))}")
    print(f"  SMA(50): {format_currency(analysis.get('sma_long', 0))}")
    
    vol = analysis.get('volume', {})
    print(f"  Volume: {vol.get('volume_ratio', 1):.1f}x average")
    print()
    
    if signal.reasons:
        print(f"{Fore.CYAN}Signal Factors:{Style.RESET_ALL}")
        for reason in signal.reasons:
            print(f"  • {reason}")
        print()
    
    return 0


def cmd_performance(config: Dict, args: argparse.Namespace) -> int:
    """View prediction performance."""
    print(f"\n{Fore.CYAN}📊 Model Performance{Style.RESET_ALL}\n")
    
    tracker = PredictionTracker(get_db_path())
    stats = tracker.get_performance_stats()
    
    if stats['total_predictions'] == 0:
        print(f"{Fore.YELLOW}No completed predictions yet.{Style.RESET_ALL}")
        print(f"Run 'python main.py scan' to log predictions.")
        print(f"Run 'python main.py verify' to check outcomes.")
        return 0
    
    completed = stats['wins'] + stats['losses']
    
    if completed > 0:
        win_rate = stats['win_rate']
        bar_len = 30
        filled = int(win_rate / 100 * bar_len)
        bar = f"{Fore.GREEN}{'█' * filled}{Style.RESET_ALL}{Fore.RED}{'░' * (bar_len - filled)}{Style.RESET_ALL}"
        
        print(f"  Model Accuracy: [{bar}] {Fore.CYAN}{win_rate:.1f}%{Style.RESET_ALL}")
        print(f"  {stats['wins']} wins / {stats['losses']} losses")
        print()
    
    print(f"  Total Completed: {stats['total_predictions']}")
    
    if completed > 0:
        print(f"  Avg Win: {Fore.GREEN}+{stats['avg_profit']:.2f}%{Style.RESET_ALL}")
        print(f"  Avg Loss: {Fore.RED}-{stats['avg_loss']:.2f}%{Style.RESET_ALL}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Total Return: {stats['total_return']:+.1f}%")
    
    # Show by ticker
    ticker_stats = tracker.get_ticker_performance()
    if ticker_stats:
        print(f"\n{Fore.CYAN}By Ticker:{Style.RESET_ALL}")
        for ts in sorted(ticker_stats, key=lambda x: x['win_rate'], reverse=True)[:5]:
            wr_color = Fore.GREEN if ts['win_rate'] >= 50 else Fore.RED
            print(f"  {ts['ticker']}: {wr_color}{ts['win_rate']:.0f}%{Style.RESET_ALL} ({ts['wins']}W/{ts['losses']}L)")
    
    print()
    return 0


def cmd_verify(config: Dict, args: argparse.Namespace) -> int:
    """Verify pending predictions."""
    print(f"\n{Fore.CYAN}🔍 Verifying predictions...{Style.RESET_ALL}\n")
    
    tracker = PredictionTracker(get_db_path())
    fetcher = DataFetcher()
    
    results = tracker.update_pending_outcomes(fetcher)
    
    print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Wins: {results['wins']}{Style.RESET_ALL}")
    print(f"  {Fore.RED}Losses: {results['losses']}{Style.RESET_ALL}")
    print(f"  Pending: {results['still_pending']}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="DayTrader-Forecast PRO - Professional Day Trading Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Scan
    scan_p = subparsers.add_parser('scan', help='Professional market scan')
    scan_p.add_argument('--no-track', action='store_true', help="Don't log predictions")
    scan_p.add_argument('--alert', '-a', action='store_true', help='Send email alerts')
    
    # Analyze
    analyze_p = subparsers.add_parser('analyze', help='Analyze a stock')
    analyze_p.add_argument('ticker', help='Stock symbol')
    
    # Pre-market
    subparsers.add_parser('premarket', help='Pre-market scanner')
    
    # Paper trading
    paper_p = subparsers.add_parser('paper', help='Paper trading')
    paper_p.add_argument('paper_action', choices=['status', 'buy', 'close', 'reset'])
    paper_p.add_argument('ticker', nargs='?', help='Stock symbol')
    
    # Backtest
    bt_p = subparsers.add_parser('backtest', help='Backtest strategy')
    bt_p.add_argument('--days', '-d', type=int, default=30, help='Days to backtest')
    bt_p.add_argument('--save', '-s', action='store_true', help='Save report')
    
    # Risk
    subparsers.add_parser('risk', help='View risk status')
    
    # Performance
    perf_p = subparsers.add_parser('performance', help='View model performance')
    perf_p.add_argument('--save', '-s', action='store_true', help='Save report')
    
    # Verify
    verify_p = subparsers.add_parser('verify', help='Verify predictions')
    verify_p.add_argument('--days', '-d', type=int, default=10, help='Max days to check')
    
    # Report
    report_p = subparsers.add_parser('report', help='Generate report')
    report_p.add_argument('--email', '-e', action='store_true', help='Email report')
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return 0
    
    print_banner()
    
    try:
        config = load_config()
    except FileNotFoundError:
        print(f"{Fore.RED}❌ config.yaml not found{Style.RESET_ALL}")
        return 1
    
    commands = {
        'scan': cmd_scan,
        'analyze': cmd_analyze,
        'premarket': cmd_premarket,
        'paper': cmd_paper,
        'backtest': cmd_backtest,
        'risk': cmd_risk,
        'performance': cmd_performance,
        'verify': cmd_verify,
    }
    
    if args.command in commands:
        return commands[args.command](config, args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
