#!/usr/bin/env python3
"""
DayTrader-Forecast - Day Trading Probability Scanner

A technical analysis tool that scans stocks and generates trading signals
with probability scores based on multiple indicators.

Usage:
    python main.py scan                     # Scan all watchlist stocks
    python main.py scan --min-prob 70       # Only show 70%+ probability signals
    python main.py analyze AAPL             # Analyze specific stock
    python main.py report                   # Generate daily report
    python main.py report --email           # Generate and email report
    python main.py backtest --days 30       # Backtest on historical data
    python main.py paper                    # Start paper trading session
    python main.py performance              # Show historical accuracy

⚠️  DISCLAIMER: This is for educational purposes only. Not financial advice.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

from colorama import Fore, Style, init as colorama_init

from analyzers.technical import TechnicalAnalyzer, MultiTimeframeAnalyzer
from analyzers.signals import SignalGenerator, SignalType, TradingSignal
from analyzers.market import MarketAnalyzer
from data.fetcher import DataFetcher
from reports.generator import ReportGenerator
from tracking.tracker import PredictionTracker
from backtesting.engine import BacktestEngine
from paper.simulator import PaperTrader
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
    
    vol_icon = "📊" if signal.volume_confirmed else ""
    
    return (
        f"{emoji} {color}{signal.ticker:6s}{Style.RESET_ALL} │ "
        f"{signal.signal_type.value:4s} │ "
        f"{signal.probability:5.1f}% │ "
        f"{format_currency(signal.entry_price):>10s} │ "
        f"{signal.sentiment.value:8s} │ {vol_icon}"
    )


def send_high_confidence_email(signal: TradingSignal, config: Dict):
    """Send email alert for high-confidence signals."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from dotenv import load_dotenv
        
        load_dotenv()
        
        smtp_host = os.getenv('SMTP_HOST')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_user = os.getenv('SMTP_USER')
        smtp_pass = os.getenv('SMTP_PASSWORD')
        email_to = os.getenv('EMAIL_TO', 'Abdiazizmohamed408@gmail.com')
        
        if not all([smtp_host, smtp_user, smtp_pass]):
            return  # SMTP not configured
        
        subject = f"🚨 HIGH CONFIDENCE SIGNAL: {signal.ticker} {signal.signal_type.value} ({signal.probability:.1f}%)"
        
        body = f"""
        High Confidence Trading Signal Alert
        =====================================
        
        Ticker: {signal.ticker}
        Signal: {signal.signal_type.value}
        Probability: {signal.probability:.1f}%
        Entry Price: ${signal.entry_price:.2f}
        Target: ${signal.target_price:.2f}
        Stop Loss: ${signal.stop_loss:.2f}
        Risk/Reward: {signal.risk_reward_ratio:.2f}
        
        Volume Confirmed: {'Yes' if signal.volume_confirmed else 'No'}
        Timeframe Alignment: {signal.timeframe_alignment:.0f}% if signal.timeframe_alignment else 'N/A'
        
        Reasons:
        {chr(10).join('• ' + r for r in signal.reasons)}
        
        ---
        ⚠️ DISCLAIMER: This is not financial advice. Trade at your own risk.
        """
        
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = email_to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        
        print(f"  {Fore.GREEN}📧 Email alert sent for {signal.ticker}{Style.RESET_ALL}")
        
    except Exception as e:
        # Silently fail - email is optional
        pass


def cmd_scan(config: Dict, args: argparse.Namespace) -> int:
    """
    Scan all watchlist stocks.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    min_prob = getattr(args, 'min_prob', 0) or 0
    
    print(f"\n{Fore.CYAN}📊 Scanning watchlist stocks...{Style.RESET_ALL}")
    if min_prob > 0:
        print(f"   (Filtering for {min_prob}%+ probability signals)")
    print()
    
    watchlist = config.get('watchlist', [])
    if not watchlist:
        print(f"{Fore.RED}❌ No stocks in watchlist. Check config.yaml{Style.RESET_ALL}")
        return 1
    
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer(config)
    signal_gen = SignalGenerator(config)
    market_analyzer = MarketAnalyzer(config)
    tracker = PredictionTracker()
    
    # Get market context first
    print(f"  Checking market conditions...", end=" ", flush=True)
    market_context = market_analyzer.analyze_market(fetcher)
    if market_context:
        print(f"{Fore.GREEN}Done{Style.RESET_ALL}")
        print()
        print(market_analyzer.format_context(market_context))
    else:
        print(f"{Fore.YELLOW}Unavailable{Style.RESET_ALL}")
        market_context = None
    
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
        
        # Get market context string
        mkt_ctx_str = market_context.description if market_context else None
        
        # Generate signal with market context
        signal = signal_gen.generate_signal(
            ticker, 
            analysis,
            market_context=mkt_ctx_str
        )
        
        if signal:
            # Adjust probability based on market context
            if market_context:
                adjusted_prob = market_analyzer.adjust_signal_confidence(
                    signal.signal_type.value,
                    signal.probability,
                    market_context
                )
                # Update the signal probability
                signal = TradingSignal(
                    ticker=signal.ticker,
                    signal_type=signal.signal_type,
                    probability=adjusted_prob,
                    sentiment=signal.sentiment,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    target_price=signal.target_price,
                    risk_reward_ratio=signal.risk_reward_ratio,
                    reasons=signal.reasons,
                    volume_confirmed=signal.volume_confirmed,
                    timeframe_alignment=signal.timeframe_alignment,
                    market_context=mkt_ctx_str
                )
            
            # Apply minimum probability filter
            if signal.probability >= min_prob:
                signals.append(signal)
                analyses[ticker] = analysis
                
                # Log to tracker
                if signal.signal_type != SignalType.HOLD:
                    tracker.log_signal(
                        ticker=signal.ticker,
                        signal_type=signal.signal_type.value,
                        entry_price=signal.entry_price,
                        target_price=signal.target_price,
                        stop_loss=signal.stop_loss,
                        probability=signal.probability,
                        market_context=market_context.to_dict() if market_context else None,
                        volume_confirmed=signal.volume_confirmed,
                        timeframe_alignment=signal.timeframe_alignment,
                        reasons=signal.reasons
                    )
                    
                    # Send email for high-confidence signals
                    if signal.probability >= 75:
                        send_high_confidence_email(signal, config)
                
                print(f"{Fore.GREEN}Done{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Below threshold ({signal.probability:.1f}%){Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No signal{Style.RESET_ALL}")
    
    # Print results
    print(f"\n{Fore.CYAN}{'═' * 65}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📈 SCAN RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 65}{Style.RESET_ALL}\n")
    
    print(f"{'TICKER':<8} │ {'SIG':4s} │ {'PROB':>5s} │ {'PRICE':>10s} │ {'SENTIMENT':8s} │ VOL")
    print(f"{'─' * 8}─┼─{'─' * 4}─┼─{'─' * 5}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 4}")
    
    # Sort by probability
    signals.sort(key=lambda x: x.probability, reverse=True)
    
    for signal in signals:
        print(format_signal_line(signal))
    
    print()
    
    # Summary
    buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
    sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
    hold_count = sum(1 for s in signals if s.signal_type == SignalType.HOLD)
    high_conf = sum(1 for s in signals if s.probability >= 70)
    
    print(f"{Fore.GREEN}🟢 BUY:{Style.RESET_ALL} {buy_count}  │  "
          f"{Fore.RED}🔴 SELL:{Style.RESET_ALL} {sell_count}  │  "
          f"{Fore.YELLOW}🟡 HOLD:{Style.RESET_ALL} {hold_count}  │  "
          f"{Fore.CYAN}⭐ HIGH CONF:{Style.RESET_ALL} {high_conf}")
    print()
    
    return 0


def cmd_analyze(config: Dict, args: argparse.Namespace) -> int:
    """
    Analyze a specific stock with multi-timeframe analysis.
    
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
    mtf_analyzer = MultiTimeframeAnalyzer(config)
    signal_gen = SignalGenerator(config)
    market_analyzer = MarketAnalyzer(config)
    report_gen = ReportGenerator(config)
    
    # Fetch data
    data = fetcher.get_stock_data(ticker)
    if data is None:
        print(f"{Fore.RED}❌ Failed to fetch data for {ticker}{Style.RESET_ALL}")
        return 1
    
    # Get quote
    quote = fetcher.get_realtime_quote(ticker)
    
    # Get market context
    market_context = market_analyzer.analyze_market(fetcher)
    mkt_ctx_str = market_context.description if market_context else None
    
    # Multi-timeframe analysis
    print(f"  Running multi-timeframe analysis...")
    mtf_result = mtf_analyzer.analyze_multi_timeframe(ticker, fetcher)
    timeframe_alignment = mtf_result.get('alignment_score') if mtf_result else None
    
    # Standard analysis
    analysis = analyzer.analyze(data)
    if analysis is None:
        print(f"{Fore.RED}❌ Insufficient data for analysis{Style.RESET_ALL}")
        return 1
    
    # Generate signal with all context
    signal = signal_gen.generate_signal(
        ticker, 
        analysis,
        timeframe_alignment=timeframe_alignment,
        market_context=mkt_ctx_str
    )
    
    if signal is None:
        print(f"{Fore.RED}❌ Failed to generate signal{Style.RESET_ALL}")
        return 1
    
    # Adjust for market context
    if market_context:
        adjusted_prob = market_analyzer.adjust_signal_confidence(
            signal.signal_type.value,
            signal.probability,
            market_context
        )
        signal = TradingSignal(
            ticker=signal.ticker,
            signal_type=signal.signal_type,
            probability=adjusted_prob,
            sentiment=signal.sentiment,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            risk_reward_ratio=signal.risk_reward_ratio,
            reasons=signal.reasons,
            volume_confirmed=signal.volume_confirmed,
            timeframe_alignment=timeframe_alignment,
            market_context=mkt_ctx_str
        )
    
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
    
    print(f"{'═' * 55}")
    name = quote.get('name', ticker) if quote else ticker
    print(f"{emoji} {color}{name} ({ticker}){Style.RESET_ALL}")
    print(f"{'═' * 55}\n")
    
    # Market context
    if market_context:
        print(market_analyzer.format_context(market_context))
    
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
    print(f"  {Fore.WHITE}Volume Confirmed:{Style.RESET_ALL} {'✅ Yes' if signal.volume_confirmed else '❌ No'}")
    
    if timeframe_alignment is not None:
        tf_color = Fore.GREEN if timeframe_alignment >= 70 else Fore.YELLOW if timeframe_alignment >= 50 else Fore.RED
        print(f"  {Fore.WHITE}Timeframe Alignment:{Style.RESET_ALL} {tf_color}{timeframe_alignment:.0f}%{Style.RESET_ALL}")
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
    vol = analysis.get('volume', {})
    print(f"    Volume Ratio: {vol.get('volume_ratio', 1.0):.2f}x")
    print()
    
    # Multi-timeframe breakdown
    if mtf_result:
        print(f"  {Fore.CYAN}Multi-Timeframe Analysis:{Style.RESET_ALL}")
        for tf, data in mtf_result.get('timeframes', {}).items():
            bias_color = Fore.GREEN if data['bias'] == 'BULLISH' else Fore.RED if data['bias'] == 'BEARISH' else Fore.YELLOW
            print(f"    {tf:4s}: {bias_color}{data['bias']}{Style.RESET_ALL}")
        print(f"    Overall: {mtf_result.get('dominant_bias', 'N/A')}")
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


def cmd_backtest(config: Dict, args: argparse.Namespace) -> int:
    """
    Run backtesting on historical data.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code
    """
    days = args.days
    min_prob = getattr(args, 'min_prob', 50) or 50
    
    print(f"\n{Fore.CYAN}📊 Running backtest ({days} days)...{Style.RESET_ALL}\n")
    
    watchlist = config.get('watchlist', [])
    if not watchlist:
        print(f"{Fore.RED}❌ No stocks in watchlist{Style.RESET_ALL}")
        return 1
    
    engine = BacktestEngine(config)
    
    print(f"  Testing on: {', '.join(watchlist)}")
    print(f"  Minimum probability: {min_prob}%")
    print(f"  Position size: {engine.position_size_pct}%")
    print()
    
    result = engine.run_backtest(
        tickers=watchlist,
        days=days,
        min_probability=min_prob
    )
    
    print(engine.print_results(result))
    
    return 0


def cmd_paper(config: Dict, args: argparse.Namespace) -> int:
    """
    Paper trading mode.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code
    """
    print(f"\n{Fore.CYAN}📄 Paper Trading Mode{Style.RESET_ALL}\n")
    
    trader = PaperTrader()
    fetcher = DataFetcher()
    
    # Try to load existing session
    session = trader.load_session()
    
    if args.reset:
        session = trader.reset_session()
        print(f"{Fore.GREEN}✅ Session reset{Style.RESET_ALL}")
    elif session is None:
        balance = getattr(args, 'balance', 10000) or 10000
        session = trader.start_session(balance)
        print(f"{Fore.GREEN}✅ New session started with ${balance:,.2f}{Style.RESET_ALL}")
    
    # Update positions with current prices
    if session.positions:
        print(f"\nUpdating position prices...")
        closed = trader.update_positions(fetcher)
        for c in closed:
            emoji = "✅" if c.profit_loss > 0 else "❌"
            print(f"  {emoji} {c.ticker} closed: ${c.entry_price:.2f} → ${c.exit_price:.2f} ({c.profit_pct:+.2f}%)")
    
    # Show status
    print(trader.get_status())
    
    # If --auto flag, automatically execute signals
    if getattr(args, 'auto', False):
        print(f"\n{Fore.CYAN}Auto-executing signals...{Style.RESET_ALL}\n")
        
        analyzer = TechnicalAnalyzer(config)
        signal_gen = SignalGenerator(config)
        
        for ticker in config.get('watchlist', []):
            if len(session.positions) >= trader.max_positions:
                print(f"  Maximum positions reached")
                break
            
            data = fetcher.get_stock_data(ticker)
            if data is None:
                continue
            
            analysis = analyzer.analyze(data)
            if analysis is None:
                continue
            
            signal = signal_gen.generate_signal(ticker, analysis)
            if signal and signal.signal_type != SignalType.HOLD and signal.probability >= 65:
                pos = trader.open_position(
                    ticker=signal.ticker,
                    signal_type=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    probability=signal.probability
                )
                if pos:
                    print(f"  {Fore.GREEN}✅ Opened {signal.signal_type.value} on {ticker} at ${signal.entry_price:.2f}{Style.RESET_ALL}")
        
        print(trader.get_status())
    
    return 0


def cmd_performance(config: Dict, args: argparse.Namespace) -> int:
    """
    Show historical performance statistics.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Exit code
    """
    tracker = PredictionTracker()
    fetcher = DataFetcher()
    
    # Update pending predictions first
    print(f"\n{Fore.CYAN}Checking pending predictions...{Style.RESET_ALL}")
    updates = tracker.update_pending_outcomes(fetcher)
    
    if updates['wins'] or updates['losses']:
        print(f"  New outcomes: {updates['wins']} wins, {updates['losses']} losses")
    
    # Get overall stats
    days = getattr(args, 'days', None)
    stats = tracker.get_performance_stats(days=days)
    
    print()
    print(f"{Fore.CYAN}📊 PERFORMANCE SUMMARY{Style.RESET_ALL}")
    print(f"{'═' * 45}")
    print()
    
    if stats['total_predictions'] == 0:
        print(f"{Fore.YELLOW}No completed predictions yet.{Style.RESET_ALL}")
        print(f"Run 'python main.py scan' to generate signals,")
        print(f"then check back later to see outcomes.")
        print()
        return 0
    
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Wins: {stats['wins']} | Losses: {stats['losses']}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print()
    print(f"Avg Profit: {Fore.GREEN}+{stats['avg_profit']:.2f}%{Style.RESET_ALL}")
    print(f"Avg Loss: {Fore.RED}-{stats['avg_loss']:.2f}%{Style.RESET_ALL}")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Total Return: {'+' if stats['total_return'] >= 0 else ''}{stats['total_return']:.2f}%")
    print()
    
    # Per-ticker breakdown
    ticker_stats = tracker.get_ticker_performance()
    
    if ticker_stats:
        print(f"{Fore.CYAN}📈 PERFORMANCE BY TICKER{Style.RESET_ALL}")
        print(f"{'─' * 45}")
        print(f"{'TICKER':<8} │ {'TRADES':>6} │ {'WIN RATE':>8} │ {'RETURN':>8}")
        print(f"{'─' * 8}─┼─{'─' * 6}─┼─{'─' * 8}─┼─{'─' * 8}")
        
        for ts in ticker_stats[:10]:
            ret_color = Fore.GREEN if ts['total_return'] >= 0 else Fore.RED
            print(
                f"{ts['ticker']:<8} │ {ts['total_predictions']:>6} │ "
                f"{ts['win_rate']:>7.1f}% │ "
                f"{ret_color}{ts['total_return']:>+7.1f}%{Style.RESET_ALL}"
            )
        
        print()
        
        # Best and worst
        if len(ticker_stats) >= 2:
            best = ticker_stats[0]
            worst = ticker_stats[-1]
            print(f"🏆 Best Performer:  {best['ticker']} ({best['win_rate']:.0f}% win rate)")
            print(f"📉 Worst Performer: {worst['ticker']} ({worst['win_rate']:.0f}% win rate)")
    
    print()
    
    # Pending predictions
    pending = tracker.get_pending_predictions()
    if pending:
        print(f"{Fore.YELLOW}⏳ {len(pending)} predictions still pending{Style.RESET_ALL}")
    
    print()
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DayTrader-Forecast - Technical Analysis Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan                    Scan all watchlist stocks
  python main.py scan --min-prob 70      Only show 70%+ probability signals
  python main.py analyze AAPL            Analyze Apple stock
  python main.py analyze TSLA --save     Analyze Tesla and save report
  python main.py report                  Generate daily report
  python main.py report --email          Generate and email report
  python main.py backtest --days 30      Backtest strategy on 30 days of data
  python main.py paper                   Show paper trading status
  python main.py paper --reset           Start new paper trading session
  python main.py paper --auto            Auto-execute signals in paper mode
  python main.py performance             Show historical accuracy
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan all watchlist stocks')
    scan_parser.add_argument('--min-prob', type=float, default=0,
                             help='Minimum probability threshold (0-100)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a specific stock')
    analyze_parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL)')
    analyze_parser.add_argument('--save', '-s', action='store_true',
                                help='Save analysis to file')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate daily report')
    report_parser.add_argument('--email', '-e', action='store_true',
                               help='Send report via email')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting on historical data')
    backtest_parser.add_argument('--days', '-d', type=int, default=30,
                                 help='Number of days to backtest (default: 30)')
    backtest_parser.add_argument('--min-prob', type=float, default=50,
                                 help='Minimum probability for trades (default: 50)')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Paper trading mode')
    paper_parser.add_argument('--reset', action='store_true',
                              help='Reset and start new session')
    paper_parser.add_argument('--balance', type=float, default=10000,
                              help='Starting balance (default: $10,000)')
    paper_parser.add_argument('--auto', action='store_true',
                              help='Auto-execute signals')
    
    # Performance command
    perf_parser = subparsers.add_parser('performance', help='Show historical performance')
    perf_parser.add_argument('--days', type=int,
                             help='Filter to last N days')
    
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
    elif args.command == 'backtest':
        return cmd_backtest(config, args)
    elif args.command == 'paper':
        return cmd_paper(config, args)
    elif args.command == 'performance':
        return cmd_performance(config, args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
