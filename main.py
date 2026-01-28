#!/usr/bin/env python3
"""
DayTrader-Forecast - Day Trading Probability Scanner

A technical analysis tool that scans stocks and generates trading signals
with probability scores based on multiple indicators.

Usage:
    python main.py scan                 # Scan all watchlist stocks
    python main.py analyze AAPL         # Analyze specific stock
    python main.py report               # Generate daily report
    python main.py report --email       # Generate and email report
    python main.py performance          # View prediction accuracy
    python main.py verify               # Verify pending predictions

⚠️  DISCLAIMER: This is for educational purposes only. Not financial advice.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from colorama import Fore, Style, init as colorama_init

from analyzers.technical import TechnicalAnalyzer
from analyzers.signals import SignalGenerator, SignalType, TradingSignal
from data.fetcher import DataFetcher
from reports.generator import ReportGenerator
from tracking.tracker import PredictionTracker
from tracking.performance import PerformanceAnalyzer
from utils.helpers import load_config, setup_logging, format_currency, ensure_dir


# Initialize colorama for cross-platform colored output
colorama_init()


BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   {Fore.YELLOW}📈 DayTrader-Forecast{Fore.CYAN}                                     ║
║   {Fore.WHITE}Technical Analysis & Probability Scanner{Fore.CYAN}                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

DISCLAIMER = f"""
{Fore.RED}⚠️  DISCLAIMER:{Style.RESET_ALL}
This tool is for EDUCATIONAL PURPOSES ONLY.
• This is NOT financial advice
• Day trading involves SIGNIFICANT RISK
• Past performance does NOT guarantee future results
• Never trade with money you cannot afford to lose
"""


def print_banner():
    """Print the application banner."""
    print(BANNER)
    print(DISCLAIMER)
    print()


def format_signal_line(signal: TradingSignal) -> str:
    """Format a signal for console output."""
    if signal.signal_type == SignalType.BUY:
        color = Fore.GREEN
        emoji = "🟢"
    elif signal.signal_type == SignalType.SELL:
        color = Fore.RED
        emoji = "🔴"
    else:
        color = Fore.YELLOW
        emoji = "🟡"
    
    return (
        f"{emoji} {color}{signal.ticker:6s}{Style.RESET_ALL} │ "
        f"{signal.signal_type.value:4s} │ "
        f"{signal.probability:5.1f}% │ "
        f"{format_currency(signal.entry_price):>10s} │ "
        f"{signal.sentiment.value}"
    )


def get_db_path() -> str:
    """Get the database path, ensuring the directory exists."""
    db_dir = Path("./data")
    db_dir.mkdir(exist_ok=True)
    return str(db_dir / "predictions.db")


def cmd_scan(config: Dict, args: argparse.Namespace) -> int:
    """
    Scan all watchlist stocks and log predictions.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print(f"\n{Fore.CYAN}📊 Scanning watchlist stocks...{Style.RESET_ALL}\n")
    
    watchlist = config.get('watchlist', [])
    if not watchlist:
        print(f"{Fore.RED}❌ No stocks in watchlist. Check config.yaml{Style.RESET_ALL}")
        return 1
    
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer(config)
    signal_gen = SignalGenerator(config)
    tracker = PredictionTracker(get_db_path())
    
    # Check for adaptive weights
    if args.adaptive:
        perf = PerformanceAnalyzer(get_db_path())
        new_weights, reasons = perf.calculate_optimal_weights(signal_gen.weights)
        if reasons and "Not enough data" not in reasons[0]:
            print(f"{Fore.CYAN}🧠 Applying adaptive weights:{Style.RESET_ALL}")
            for reason in reasons:
                print(f"   {reason}")
            print()
            signal_gen.update_weights(new_weights)
        perf.close()
    
    signals: List[TradingSignal] = []
    analyses: Dict = {}
    
    for ticker in watchlist:
        print(f"  Analyzing {ticker}...", end=" ", flush=True)
        
        # Fetch data
        data = fetcher.get_stock_data(ticker)
        if data is None:
            print(f"{Fore.RED}Failed{Style.RESET_ALL}")
            continue
        
        # Analyze
        analysis = analyzer.analyze(data)
        if analysis is None:
            print(f"{Fore.YELLOW}Insufficient data{Style.RESET_ALL}")
            continue
        
        # Generate signal
        signal = signal_gen.generate_signal(ticker, analysis)
        if signal:
            signals.append(signal)
            analyses[ticker] = analysis
            
            # Log prediction to database (only BUY/SELL)
            if signal.signal_type != SignalType.HOLD and not args.no_track:
                tracker.log_prediction(
                    ticker=signal.ticker,
                    signal_type=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    probability=signal.probability,
                    sentiment=signal.sentiment.value,
                    indicator_scores=signal_gen.get_last_scores(),
                    indicator_values={
                        'rsi': analysis.get('rsi'),
                        'macd': analysis.get('macd')
                    },
                    reasons=signal.reasons
                )
            
            print(f"{Fore.GREEN}Done{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No signal{Style.RESET_ALL}")
    
    tracker.close()
    
    # Print results
    print(f"\n{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📈 SCAN RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}\n")
    
    print(f"{'TICKER':<8} │ {'SIG':4s} │ {'PROB':>5s} │ {'PRICE':>10s} │ SENTIMENT")
    print(f"{'─' * 8}─┼─{'─' * 4}─┼─{'─' * 5}─┼─{'─' * 10}─┼─{'─' * 10}")
    
    # Sort by probability
    signals.sort(key=lambda x: x.probability, reverse=True)
    
    for signal in signals:
        print(format_signal_line(signal))
    
    print()
    
    # Summary
    buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
    sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
    hold_count = sum(1 for s in signals if s.signal_type == SignalType.HOLD)
    
    print(f"{Fore.GREEN}🟢 BUY:{Style.RESET_ALL} {buy_count}  │  "
          f"{Fore.RED}🔴 SELL:{Style.RESET_ALL} {sell_count}  │  "
          f"{Fore.YELLOW}🟡 HOLD:{Style.RESET_ALL} {hold_count}")
    
    if not args.no_track:
        tracked = buy_count + sell_count
        print(f"\n{Fore.CYAN}📝 Logged {tracked} predictions for tracking{Style.RESET_ALL}")
    
    print()
    
    return 0


def cmd_analyze(config: Dict, args: argparse.Namespace) -> int:
    """
    Analyze a specific stock.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments with ticker
        
    Returns:
        Exit code
    """
    ticker = args.ticker.upper()
    print(f"\n{Fore.CYAN}📊 Analyzing {ticker}...{Style.RESET_ALL}\n")
    
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer(config)
    signal_gen = SignalGenerator(config)
    report_gen = ReportGenerator(config)
    tracker = PredictionTracker(get_db_path())
    
    # Fetch data
    data = fetcher.get_stock_data(ticker)
    if data is None:
        print(f"{Fore.RED}❌ Failed to fetch data for {ticker}{Style.RESET_ALL}")
        return 1
    
    # Get quote
    quote = fetcher.get_realtime_quote(ticker)
    
    # Analyze
    analysis = analyzer.analyze(data)
    if analysis is None:
        print(f"{Fore.RED}❌ Insufficient data for analysis{Style.RESET_ALL}")
        return 1
    
    # Generate signal
    signal = signal_gen.generate_signal(ticker, analysis)
    if signal is None:
        print(f"{Fore.RED}❌ Failed to generate signal{Style.RESET_ALL}")
        return 1
    
    # Log prediction
    if signal.signal_type != SignalType.HOLD and not args.no_track:
        pred_id = tracker.log_prediction(
            ticker=signal.ticker,
            signal_type=signal.signal_type.value,
            entry_price=signal.entry_price,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            probability=signal.probability,
            sentiment=signal.sentiment.value,
            indicator_scores=signal_gen.get_last_scores(),
            indicator_values={
                'rsi': analysis.get('rsi'),
                'macd': analysis.get('macd')
            },
            reasons=signal.reasons
        )
        print(f"{Fore.CYAN}📝 Prediction logged (ID: {pred_id}){Style.RESET_ALL}\n")
    
    tracker.close()
    
    # Print results
    if signal.signal_type == SignalType.BUY:
        color = Fore.GREEN
        emoji = "🟢"
    elif signal.signal_type == SignalType.SELL:
        color = Fore.RED
        emoji = "🔴"
    else:
        color = Fore.YELLOW
        emoji = "🟡"
    
    print(f"{'═' * 50}")
    name = quote.get('name', ticker) if quote else ticker
    print(f"{emoji} {color}{name} ({ticker}){Style.RESET_ALL}")
    print(f"{'═' * 50}\n")
    
    # Price info
    print(f"  {Fore.WHITE}Current Price:{Style.RESET_ALL} {format_currency(signal.entry_price)}")
    if analysis.get('price_change_pct'):
        pct = analysis['price_change_pct']
        pct_color = Fore.GREEN if pct >= 0 else Fore.RED
        print(f"  {Fore.WHITE}Change:{Style.RESET_ALL} {pct_color}{pct:+.2f}%{Style.RESET_ALL}")
    print()
    
    # Signal
    print(f"  {Fore.WHITE}Signal:{Style.RESET_ALL} {color}{signal.signal_type.value}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Probability:{Style.RESET_ALL} {signal.probability:.1f}%")
    print(f"  {Fore.WHITE}Sentiment:{Style.RESET_ALL} {signal.sentiment.value}")
    print()
    
    # Targets
    if signal.stop_loss and signal.target_price:
        print(f"  {Fore.WHITE}Stop Loss:{Style.RESET_ALL} {format_currency(signal.stop_loss)}")
        print(f"  {Fore.WHITE}Target:{Style.RESET_ALL} {format_currency(signal.target_price)}")
        if signal.risk_reward_ratio:
            print(f"  {Fore.WHITE}Risk/Reward:{Style.RESET_ALL} {signal.risk_reward_ratio:.2f}")
    print()
    
    # Key indicators
    print(f"  {Fore.CYAN}Technical Indicators:{Style.RESET_ALL}")
    print(f"    RSI(14): {analysis.get('rsi', 0):.1f}")
    print(f"    MACD: {analysis.get('macd', 0):.4f}")
    print(f"    SMA(20): {format_currency(analysis.get('sma_short', 0))}")
    print(f"    SMA(50): {format_currency(analysis.get('sma_long', 0))}")
    print()
    
    # Reasons
    if signal.reasons:
        print(f"  {Fore.CYAN}Signal Factors:{Style.RESET_ALL}")
        for reason in signal.reasons:
            print(f"    • {reason}")
    print()
    
    # Historical accuracy for this ticker
    perf = PerformanceAnalyzer(get_db_path())
    ticker_stats = perf.db.get_stats_by_ticker()
    perf.close()
    
    for ts in ticker_stats:
        if ts['ticker'] == ticker and ts['total'] >= 3:
            print(f"  {Fore.CYAN}Historical Accuracy for {ticker}:{Style.RESET_ALL}")
            print(f"    {ts['wins']}/{ts['total']} predictions correct ({ts['win_rate']:.1f}%)")
            print()
            break
    
    # Save option
    if args.save:
        report = report_gen.generate_single_analysis(signal, analysis)
        filepath = report_gen.save_report(report, f"{ticker}_analysis.md")
        print(f"  {Fore.GREEN}✅ Report saved to {filepath}{Style.RESET_ALL}\n")
    
    return 0


def cmd_report(config: Dict, args: argparse.Namespace) -> int:
    """
    Generate daily market report.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code
    """
    print(f"\n{Fore.CYAN}📊 Generating daily report...{Style.RESET_ALL}\n")
    
    watchlist = config.get('watchlist', [])
    if not watchlist:
        print(f"{Fore.RED}❌ No stocks in watchlist{Style.RESET_ALL}")
        return 1
    
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer(config)
    signal_gen = SignalGenerator(config)
    report_gen = ReportGenerator(config)
    
    signals: List[TradingSignal] = []
    analyses: Dict = {}
    
    for ticker in watchlist:
        print(f"  Scanning {ticker}...", end=" ", flush=True)
        
        data = fetcher.get_stock_data(ticker)
        if data is None:
            print(f"{Fore.RED}Failed{Style.RESET_ALL}")
            continue
        
        analysis = analyzer.analyze(data)
        if analysis is None:
            print(f"{Fore.YELLOW}Skip{Style.RESET_ALL}")
            continue
        
        signal = signal_gen.generate_signal(ticker, analysis)
        if signal:
            signals.append(signal)
            analyses[ticker] = analysis
            print(f"{Fore.GREEN}Done{Style.RESET_ALL}")
    
    # Generate report
    report = report_gen.generate_scan_report(signals, analyses)
    
    # Save report
    filepath = report_gen.save_report(report)
    print(f"\n{Fore.GREEN}✅ Report saved to {filepath}{Style.RESET_ALL}")
    
    # Email if requested
    if args.email:
        report_gen.send_email(report)
    
    return 0


def cmd_verify(config: Dict, args: argparse.Namespace) -> int:
    """
    Verify pending predictions and update outcomes.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code
    """
    print(f"\n{Fore.CYAN}🔍 Verifying pending predictions...{Style.RESET_ALL}\n")
    
    tracker = PredictionTracker(get_db_path(), max_days=args.days)
    
    results = tracker.verify_outcomes(verbose=True)
    
    print(f"\n{Fore.CYAN}{'═' * 40}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 VERIFICATION SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 40}{Style.RESET_ALL}\n")
    
    print(f"  Predictions checked: {results['checked']}")
    print(f"  {Fore.GREEN}✅ Wins:{Style.RESET_ALL} {results['wins']}")
    print(f"  {Fore.RED}❌ Losses:{Style.RESET_ALL} {results['losses']}")
    print(f"  {Fore.YELLOW}⏳ Still pending:{Style.RESET_ALL} {results['still_pending']}")
    if results['errors']:
        print(f"  {Fore.RED}⚠️  Errors:{Style.RESET_ALL} {results['errors']}")
    
    print()
    tracker.close()
    
    return 0


def cmd_performance(config: Dict, args: argparse.Namespace) -> int:
    """
    Display prediction performance and accuracy.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code
    """
    print(f"\n{Fore.CYAN}📊 Loading performance data...{Style.RESET_ALL}\n")
    
    perf = PerformanceAnalyzer(get_db_path())
    stats = perf.db.get_stats()
    
    # Check if we have data
    if stats['total_predictions'] == 0:
        print(f"{Fore.YELLOW}⚠️  No predictions logged yet.{Style.RESET_ALL}")
        print(f"   Run '{Fore.WHITE}python main.py scan{Style.RESET_ALL}' to start tracking predictions.")
        print(f"   Then run '{Fore.WHITE}python main.py verify{Style.RESET_ALL}' after a few days to check outcomes.")
        perf.close()
        return 0
    
    # Print overview
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📈 PREDICTION PERFORMANCE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}\n")
    
    # Win rate with visual bar
    completed = stats['wins'] + stats['losses']
    if completed > 0:
        win_rate = stats['win_rate']
        bar_len = 30
        filled = int(win_rate / 100 * bar_len)
        bar = f"{Fore.GREEN}{'█' * filled}{Style.RESET_ALL}{Fore.RED}{'░' * (bar_len - filled)}{Style.RESET_ALL}"
        
        print(f"  {Fore.WHITE}Model Accuracy:{Style.RESET_ALL}")
        print(f"  [{bar}] {Fore.CYAN}{win_rate:.1f}%{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}{stats['wins']} wins{Style.RESET_ALL} / {Fore.RED}{stats['losses']} losses{Style.RESET_ALL}")
        print()
    
    # Key metrics
    print(f"  {Fore.WHITE}Key Metrics:{Style.RESET_ALL}")
    print(f"    Total Predictions: {stats['total_predictions']}")
    print(f"    Pending: {stats['pending']}")
    
    if completed > 0:
        print(f"    Avg Win: {Fore.GREEN}+{stats['avg_win_pct']:.2f}%{Style.RESET_ALL}")
        print(f"    Avg Loss: {Fore.RED}{stats['avg_loss_pct']:.2f}%{Style.RESET_ALL}")
        
        pf = stats['profit_factor']
        pf_color = Fore.GREEN if pf >= 1.5 else (Fore.YELLOW if pf >= 1.0 else Fore.RED)
        print(f"    Profit Factor: {pf_color}{pf:.2f}{Style.RESET_ALL}")
    print()
    
    # Performance by ticker
    ticker_stats = perf.db.get_stats_by_ticker()
    if ticker_stats:
        print(f"  {Fore.WHITE}Performance by Ticker:{Style.RESET_ALL}")
        print(f"  {'TICKER':<8} {'TRADES':>7} {'WINS':>6} {'WIN%':>7}")
        print(f"  {'─' * 32}")
        
        for ts in sorted(ticker_stats, key=lambda x: x['win_rate'], reverse=True)[:10]:
            wr = ts['win_rate']
            wr_color = Fore.GREEN if wr >= 55 else (Fore.YELLOW if wr >= 45 else Fore.RED)
            print(f"  {ts['ticker']:<8} {ts['total']:>7} {ts['wins']:>6} {wr_color}{wr:>6.1f}%{Style.RESET_ALL}")
        print()
    
    # Indicator effectiveness
    indicator_stats = perf.db.get_stats_by_indicator()
    if any(s['high_score_trades'] + s['low_score_trades'] > 0 for s in indicator_stats.values()):
        print(f"  {Fore.WHITE}Indicator Effectiveness:{Style.RESET_ALL}")
        print(f"  {'INDICATOR':<20} {'EFFECT':>10}")
        print(f"  {'─' * 32}")
        
        sorted_ind = sorted(
            indicator_stats.items(),
            key=lambda x: x[1]['effectiveness'],
            reverse=True
        )
        
        for name, ind_stats in sorted_ind:
            eff = ind_stats['effectiveness']
            if ind_stats['high_score_trades'] + ind_stats['low_score_trades'] > 0:
                eff_color = Fore.GREEN if eff > 5 else (Fore.RED if eff < -5 else Fore.YELLOW)
                emoji = "📈" if eff > 5 else ("📉" if eff < -5 else "➖")
                print(f"  {emoji} {name:<17} {eff_color}{eff:>+9.1f}%{Style.RESET_ALL}")
        print()
    
    # Save full report if requested
    if args.save:
        report = perf.generate_report()
        ensure_dir("./output")
        filepath = Path("./output") / "performance_report.md"
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"{Fore.GREEN}✅ Full report saved to {filepath}{Style.RESET_ALL}\n")
    
    # Suggest running verify if there are pending predictions
    if stats['pending'] > 0:
        print(f"{Fore.YELLOW}💡 Tip: Run 'python main.py verify' to check {stats['pending']} pending predictions{Style.RESET_ALL}\n")
    
    perf.close()
    return 0


def cmd_optimize(config: Dict, args: argparse.Namespace) -> int:
    """
    Optimize indicator weights based on historical performance.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code
    """
    print(f"\n{Fore.CYAN}🧠 Optimizing indicator weights...{Style.RESET_ALL}\n")
    
    perf = PerformanceAnalyzer(get_db_path())
    signal_gen = SignalGenerator(config)
    
    current_weights = signal_gen.weights
    
    print(f"  {Fore.WHITE}Current Weights:{Style.RESET_ALL}")
    for ind, weight in current_weights.items():
        print(f"    {ind}: {weight:.2f}")
    print()
    
    new_weights, reasons = perf.calculate_optimal_weights(
        current_weights,
        min_trades=args.min_trades,
        learning_rate=args.learning_rate
    )
    
    print(f"  {Fore.WHITE}Optimization Results:{Style.RESET_ALL}")
    for reason in reasons:
        print(f"    {reason}")
    print()
    
    if args.apply:
        print(f"  {Fore.WHITE}New Weights:{Style.RESET_ALL}")
        for ind, weight in new_weights.items():
            old = current_weights.get(ind, 0)
            if abs(weight - old) > 0.005:
                print(f"    {ind}: {old:.2f} → {Fore.CYAN}{weight:.2f}{Style.RESET_ALL}")
            else:
                print(f"    {ind}: {weight:.2f}")
        print()
        
        # Save to config suggestion
        print(f"{Fore.YELLOW}💡 To apply these weights permanently, update config.yaml:{Style.RESET_ALL}")
        print()
        print("weights:")
        for ind, weight in new_weights.items():
            print(f"  {ind}: {weight:.3f}")
        print()
    
    perf.close()
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DayTrader-Forecast - Technical Analysis Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan                 Scan all watchlist stocks
  python main.py scan --adaptive      Scan with adaptive weights
  python main.py analyze AAPL         Analyze Apple stock
  python main.py analyze TSLA --save  Analyze Tesla and save report
  python main.py report               Generate daily report
  python main.py verify               Check pending predictions
  python main.py performance          View model accuracy
  python main.py performance --save   Save performance report
  python main.py optimize             Calculate optimal weights
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan all watchlist stocks')
    scan_parser.add_argument('--no-track', action='store_true',
                             help="Don't log predictions to database")
    scan_parser.add_argument('--adaptive', '-a', action='store_true',
                             help='Use adaptive weights based on performance')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a specific stock')
    analyze_parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL)')
    analyze_parser.add_argument('--save', '-s', action='store_true',
                                help='Save analysis to file')
    analyze_parser.add_argument('--no-track', action='store_true',
                                help="Don't log prediction to database")
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate daily report')
    report_parser.add_argument('--email', '-e', action='store_true',
                               help='Send report via email')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify pending predictions')
    verify_parser.add_argument('--days', '-d', type=int, default=10,
                               help='Maximum days to track predictions (default: 10)')
    
    # Performance command
    perf_parser = subparsers.add_parser('performance', help='View prediction accuracy')
    perf_parser.add_argument('--save', '-s', action='store_true',
                             help='Save full performance report')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Calculate optimal weights')
    opt_parser.add_argument('--apply', '-a', action='store_true',
                            help='Show how to apply optimized weights')
    opt_parser.add_argument('--min-trades', type=int, default=10,
                            help='Minimum trades for weight adjustment (default: 10)')
    opt_parser.add_argument('--learning-rate', type=float, default=0.1,
                            help='Learning rate for adjustments (default: 0.1)')
    
    args = parser.parse_args()
    
    if args.command is None:
        print_banner()
        parser.print_help()
        return 0
    
    print_banner()
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        print(f"{Fore.RED}❌ config.yaml not found. Run from project directory.{Style.RESET_ALL}")
        return 1
    
    # Execute command
    if args.command == 'scan':
        return cmd_scan(config, args)
    elif args.command == 'analyze':
        return cmd_analyze(config, args)
    elif args.command == 'report':
        return cmd_report(config, args)
    elif args.command == 'verify':
        return cmd_verify(config, args)
    elif args.command == 'performance':
        return cmd_performance(config, args)
    elif args.command == 'optimize':
        return cmd_optimize(config, args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
