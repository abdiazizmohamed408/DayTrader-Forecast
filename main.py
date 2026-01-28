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

⚠️  DISCLAIMER: This is for educational purposes only. Not financial advice.
"""

import argparse
import sys
from typing import Dict, List, Optional

from colorama import Fore, Style, init as colorama_init

from analyzers.technical import TechnicalAnalyzer
from analyzers.signals import SignalGenerator, SignalType, TradingSignal
from data.fetcher import DataFetcher
from reports.generator import ReportGenerator
from utils.helpers import load_config, setup_logging, format_currency


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


def cmd_scan(config: Dict, args: argparse.Namespace) -> int:
    """
    Scan all watchlist stocks.
    
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
            print(f"{Fore.GREEN}Done{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No signal{Style.RESET_ALL}")
    
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DayTrader-Forecast - Technical Analysis Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan                 Scan all watchlist stocks
  python main.py analyze AAPL         Analyze Apple stock
  python main.py analyze TSLA --save  Analyze Tesla and save report
  python main.py report               Generate daily report
  python main.py report --email       Generate and email report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan all watchlist stocks')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a specific stock')
    analyze_parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL)')
    analyze_parser.add_argument('--save', '-s', action='store_true',
                                help='Save analysis to file')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate daily report')
    report_parser.add_argument('--email', '-e', action='store_true',
                               help='Send report via email')
    
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
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
