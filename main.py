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
from analyzers.events import EventRiskAnalyzer
from analyzers.global_market import GlobalMarketAnalyzer
from backtesting.engine import BacktestEngine
from data.fetcher import DataFetcher
from data.forex import ForexFetcher, FOREX_PAIRS
from data.events import EconomicCalendarFetcher
from paper.simulator import PaperTrader
from reports.generator import ReportGenerator
from tracking.tracker import PredictionTracker
from utils.helpers import load_config, format_currency, ensure_dir
from utils.alerts import AlertSystem
from utils.validator import DataValidator, ErrorHandler

# ML imports (optional - graceful fallback)
try:
    from ml.ensemble import MLEnsemble
    from ml.price_predictor import PricePredictor
    from ml.sentiment import SentimentAnalyzer
    from ml.patterns import PatternRecognizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


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
    global_analyzer = GlobalMarketAnalyzer(config)
    event_analyzer = EventRiskAnalyzer(config)
    
    # Initialize ML ensemble (optional)
    ml_ensemble = None
    ml_enabled = config.get('ml', {}).get('enabled', True) and ML_AVAILABLE
    ml_analyses = {}  # Store ML analysis per ticker
    
    if ml_enabled:
        try:
            ml_ensemble = MLEnsemble(config)
            if ml_ensemble.is_available():
                print(f"  {Fore.GREEN}🤖 AI features enabled{Style.RESET_ALL}\n")
            else:
                print(f"  {Fore.YELLOW}🤖 AI features: fallback mode{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"  {Fore.YELLOW}⚠️ AI features unavailable: {e}{Style.RESET_ALL}\n")
            ml_ensemble = None
    
    # Check global market context first
    global_config = config.get('global_indicators', {})
    events_config = config.get('events', {})
    
    if global_config.get('show_in_scan', True):
        print(f"  {Fore.CYAN}🌍 Global Context:{Style.RESET_ALL}")
        try:
            global_context = global_analyzer.analyze_global_context()
            vix_data = global_analyzer.get_vix_signal()
            
            # Sentiment
            sentiment_emoji = {'RISK_ON': '🟢', 'RISK_OFF': '🔴', 'NEUTRAL': '🟡', 'UNCERTAIN': '⚠️'}
            print(f"     Sentiment: {sentiment_emoji.get(global_context.sentiment.value, '❓')} {global_context.sentiment.value}")
            
            # VIX
            vix_level = vix_data['level']
            if vix_level > 25:
                print(f"     VIX: {Fore.RED}{vix_level:.1f}{Style.RESET_ALL} ⚠️ Elevated")
            elif vix_level < 15:
                print(f"     VIX: {Fore.GREEN}{vix_level:.1f}{Style.RESET_ALL} 😴 Low")
            else:
                print(f"     VIX: {vix_level:.1f}")
            
            print(f"     USD: {global_context.dxy_trend}")
        except Exception as e:
            print(f"     {Fore.YELLOW}Could not fetch global data{Style.RESET_ALL}")
        print()
    
    # Check economic events
    if events_config.get('show_in_scan', True):
        print(f"  {Fore.CYAN}📅 Economic Events:{Style.RESET_ALL}")
        try:
            event_risk = event_analyzer.assess_event_risk()
            
            if event_risk.should_avoid_trading:
                print(f"     {Fore.RED}🚨 MAJOR EVENT IMMINENT - Consider waiting{Style.RESET_ALL}")
            elif event_risk.risk_level.value in ['EXTREME', 'HIGH']:
                print(f"     {Fore.YELLOW}⚠️ High-impact event upcoming{Style.RESET_ALL}")
            else:
                print(f"     {Fore.GREEN}✓ No major events soon{Style.RESET_ALL}")
            
            # Show warnings
            for warning in event_risk.warnings[:2]:
                print(f"     {warning}")
            
            if event_risk.confidence_adjustment != 0:
                print(f"     Signal adjustment: {event_risk.confidence_adjustment:+.0f}%")
        except Exception as e:
            print(f"     {Fore.YELLOW}Could not fetch event data{Style.RESET_ALL}")
        print()
    
    # Check market context
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
        
        # 6. ML/AI Analysis (if enabled)
        ml_analysis = None
        if ml_ensemble and signal.signal_type != SignalType.HOLD:
            try:
                ml_analysis = ml_ensemble.analyze(
                    ticker, data, signal.signal_type.value
                )
                ml_analyses[ticker] = ml_analysis
                
                # Enhance probability with ML
                adjusted_probability, ml_reasons = ml_ensemble.enhance_signal_probability(
                    adjusted_probability, ml_analysis, signal.signal_type.value
                )
                
                # Add ML reasons
                signal.reasons.extend(ml_reasons)
                
                # Note if AI agrees
                if ml_analysis.ai_agrees_with_technical:
                    confidence_notes.append("🤖✅")
                elif ml_analysis.ai_agrees_with_technical is False:
                    confidence_notes.append("🤖⚠️")
                    
            except Exception as e:
                pass  # Silently continue without ML
        
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
    
    # Determine if we have ML data to show
    has_ml_data = bool(ml_analyses)
    
    if has_ml_data:
        # Extended format with AI columns
        print(f"{'TICKER':<8} │ {'SIG':4s} │ {'PROB':>5s} │ {'AI PRED':>9s} │ {'SENTIMENT':>10s} │ {'PRICE':>10s}")
        print(f"{'─' * 8}─┼─{'─' * 4}─┼─{'─' * 5}─┼─{'─' * 9}─┼─{'─' * 10}─┼─{'─' * 10}")
        
        for signal in signals:
            # Signal color
            if signal.signal_type == SignalType.BUY:
                color = Fore.GREEN
                emoji = "🟢"
            elif signal.signal_type == SignalType.SELL:
                color = Fore.RED
                emoji = "🔴"
            else:
                color = Fore.YELLOW
                emoji = "🟡"
            
            # ML data for this ticker
            ml = ml_analyses.get(signal.ticker)
            if ml:
                ai_pred = ml.get_direction_str()
                sentiment = ml.get_sentiment_str()
            else:
                ai_pred = "–"
                sentiment = "–"
            
            # Confidence star
            conf = " ⭐" if signal.probability >= min_prob and signal.signal_type != SignalType.HOLD else ""
            
            print(
                f"{emoji} {color}{signal.ticker:6s}{Style.RESET_ALL} │ "
                f"{signal.signal_type.value:4s} │ "
                f"{signal.probability:5.1f}%│ "
                f"{ai_pred:>9s} │ "
                f"{sentiment:>10s} │ "
                f"{format_currency(signal.entry_price):>10s}{conf}"
            )
    else:
        # Original format (no ML)
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


def cmd_forex(config: Dict, args: argparse.Namespace) -> int:
    """Forex pair analysis."""
    print(f"\n{Fore.CYAN}💱 Forex Pair Analysis{Style.RESET_ALL}\n")
    
    forex_config = config.get('forex', {})
    if not forex_config.get('enabled', True):
        print(f"{Fore.YELLOW}Forex analysis is disabled in config{Style.RESET_ALL}")
        return 0
    
    pairs = forex_config.get('pairs', list(FOREX_PAIRS.keys())[:6])
    
    # Initialize analyzers
    forex_fetcher = ForexFetcher()
    analyzer = TechnicalAnalyzer(config)
    signal_gen = SignalGenerator(config)
    global_analyzer = GlobalMarketAnalyzer(config)
    event_analyzer = EventRiskAnalyzer(config)
    
    # Show forex session status
    sessions = forex_fetcher.is_forex_session_active()
    active_sessions = [s for s, active in sessions.items() if active and s != 'overlap_london_ny']
    print(f"  {Fore.CYAN}Active Sessions:{Style.RESET_ALL} {', '.join(active_sessions) if active_sessions else 'Weekend'}")
    
    if sessions.get('overlap_london_ny'):
        print(f"  {Fore.GREEN}🔥 London/NY Overlap - Peak Volatility!{Style.RESET_ALL}")
    print()
    
    # Check global context
    global_context = global_analyzer.analyze_global_context()
    print(f"  {Fore.CYAN}USD Trend:{Style.RESET_ALL} {global_context.dxy_trend}")
    print(f"  {Fore.CYAN}Risk Sentiment:{Style.RESET_ALL} {global_context.sentiment.value}")
    print()
    
    # Check event risk
    event_risk = event_analyzer.assess_event_risk()
    if event_risk.warnings:
        for warning in event_risk.warnings[:3]:
            print(f"  {warning}")
        print()
    
    # Analyze forex pairs
    print(f"{'PAIR':<12} │ {'PRICE':>10} │ {'CHANGE':>8} │ {'SIGNAL':>6} │ {'PROB':>5} │ NOTES")
    print(f"{'─' * 12}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 6}─┼─{'─' * 5}─┼─{'─' * 20}")
    
    for pair in pairs:
        pair_name = forex_fetcher.get_pair_name(pair)
        
        # Get data
        data = forex_fetcher.get_forex_data(pair)
        if data is None or len(data) < 50:
            print(f"{pair_name:<12} │ {'N/A':>10} │ {'---':>8} │ {'---':>6} │ {'---':>5} │ No data")
            continue
        
        # Get quote
        quote = forex_fetcher.get_realtime_quote(pair)
        current_price = quote.get('price', data['close'].iloc[-1]) if quote else data['close'].iloc[-1]
        change = quote.get('change_pct', 0) if quote else 0
        
        # Technical analysis
        analysis = analyzer.analyze(data)
        if analysis is None:
            print(f"{pair_name:<12} │ {current_price:>10.4f} │ {change:>+7.2f}% │ {'---':>6} │ {'---':>5} │ Insufficient data")
            continue
        
        # Generate signal
        signal = signal_gen.generate_signal(pair, analysis)
        if signal is None:
            signal_str = "HOLD"
            prob = 50
            color = Fore.YELLOW
        else:
            signal_str = signal.signal_type.value
            prob = signal.probability
            
            # Adjust for event risk
            prob += event_risk.confidence_adjustment
            prob = max(0, min(95, prob))
            
            if signal.signal_type == SignalType.BUY:
                color = Fore.GREEN
            elif signal.signal_type == SignalType.SELL:
                color = Fore.RED
            else:
                color = Fore.YELLOW
        
        # Notes
        notes = []
        if analysis.get('rsi_overbought'):
            notes.append("Overbought")
        elif analysis.get('rsi_oversold'):
            notes.append("Oversold")
        if analysis.get('macd_crossover'):
            notes.append("MACD↑")
        elif analysis.get('macd_crossunder'):
            notes.append("MACD↓")
        
        notes_str = ", ".join(notes) if notes else "-"
        
        change_color = Fore.GREEN if change > 0 else (Fore.RED if change < 0 else Fore.WHITE)
        
        print(f"{pair_name:<12} │ {current_price:>10.4f} │ {change_color}{change:>+7.2f}%{Style.RESET_ALL} │ "
              f"{color}{signal_str:>6}{Style.RESET_ALL} │ {prob:>4.0f}% │ {notes_str}")
    
    print()
    
    # Session recommendations
    print(f"{Fore.CYAN}📋 Session Notes:{Style.RESET_ALL}")
    if sessions.get('overlap_london_ny'):
        print(f"  • Best time for EUR/USD, GBP/USD volatility")
    if sessions.get('tokyo') and not sessions.get('london'):
        print(f"  • Focus on JPY pairs during Tokyo session")
    if not any(sessions.values()):
        print(f"  • Forex market closed (weekend)")
    
    print()
    return 0


def cmd_events(config: Dict, args: argparse.Namespace) -> int:
    """Show upcoming economic events."""
    hours = args.hours if hasattr(args, 'hours') and args.hours else 48
    
    print(f"\n{Fore.CYAN}📅 UPCOMING ECONOMIC EVENTS (Next {hours} Hours){Style.RESET_ALL}")
    print(f"{'═' * 65}\n")
    
    event_analyzer = EventRiskAnalyzer(config)
    summary = event_analyzer.get_event_summary(hours=hours)
    
    # Risk assessment
    risk_level = summary['risk_level']
    risk_emoji = summary['risk_emoji']
    
    if risk_level == 'EXTREME':
        risk_color = Fore.RED
    elif risk_level == 'HIGH':
        risk_color = Fore.YELLOW
    else:
        risk_color = Fore.GREEN
    
    print(f"  Current Risk: {risk_emoji} {risk_color}{risk_level}{Style.RESET_ALL}")
    
    if summary['confidence_adjustment'] != 0:
        print(f"  Signal Adjustment: {summary['confidence_adjustment']:+.0f}% probability")
    
    if summary['should_avoid']:
        print(f"  {Fore.RED}⚠️ AVOID TRADING - Major event imminent{Style.RESET_ALL}")
    
    print()
    
    # Warnings
    if summary['warnings']:
        print(f"{Fore.YELLOW}Warnings:{Style.RESET_ALL}")
        for warning in summary['warnings']:
            print(f"  {warning}")
        print()
    
    # Event table
    events = summary['events']
    if not events:
        print(f"  {Fore.GREEN}✓ No major events in the next {hours} hours{Style.RESET_ALL}")
    else:
        print(f"TIME (ET)      EVENT                         IMPACT   FORECAST")
        print(f"{'─' * 65}")
        
        for event in events[:15]:  # Show max 15 events
            time_str = event.datetime_str
            event_name = event.event[:28].ljust(28)
            
            if event.impact.value == 'HIGH':
                impact_color = Fore.RED
            elif event.impact.value == 'MEDIUM':
                impact_color = Fore.YELLOW
            else:
                impact_color = Fore.GREEN
            
            impact_str = f"{event.impact_emoji} {impact_color}{event.impact.value:6s}{Style.RESET_ALL}"
            forecast = (event.forecast or "TBD")[:10]
            
            print(f"{time_str:14s} {event_name} {impact_str} {forecast}")
    
    print()
    
    # Trading recommendations
    print(f"{Fore.CYAN}📋 Recommendations:{Style.RESET_ALL}")
    
    if summary['high_impact_count'] > 0:
        print(f"  • {summary['high_impact_count']} high-impact events upcoming")
        print(f"  • Consider reducing position sizes")
        print(f"  • Set tighter stops before events")
    else:
        print(f"  • Low event risk - normal trading conditions")
    
    print()
    return 0


def cmd_predict(config: Dict, args: argparse.Namespace) -> int:
    """AI price prediction for a ticker."""
    ticker = args.ticker.upper()
    print(f"\n{Fore.CYAN}🤖 ML Price Prediction: {ticker}{Style.RESET_ALL}\n")
    
    if not ML_AVAILABLE:
        print(f"{Fore.YELLOW}⚠️ ML features not available. Install with:{Style.RESET_ALL}")
        print(f"   pip install scikit-learn vaderSentiment joblib")
        return 1
    
    if not config.get('ml', {}).get('enabled', True):
        print(f"{Fore.YELLOW}⚠️ ML features disabled in config.yaml{Style.RESET_ALL}")
        return 1
    
    fetcher = DataFetcher()
    predictor = PricePredictor(config)
    
    print(f"  Backend: {Fore.GREEN}sklearn (GradientBoosting){Style.RESET_ALL}")
    print()
    
    # Fetch data
    print(f"  Fetching {ticker} data...", end=" ", flush=True)
    data = fetcher.get_stock_data(ticker, period="1y")
    if data is None:
        print(f"{Fore.RED}Failed{Style.RESET_ALL}")
        return 1
    print(f"{Fore.GREEN}Done ({len(data)} days){Style.RESET_ALL}")
    
    # Train and predict
    print(f"  Training model...", end=" ", flush=True)
    predictor.train(ticker, data)
    print(f"{Fore.GREEN}Done{Style.RESET_ALL}\n")
    
    prediction = predictor.predict(ticker, data)
    
    if prediction is None:
        print(f"{Fore.RED}❌ Could not generate prediction{Style.RESET_ALL}")
        return 1
    
    # Display results
    print(f"{Fore.CYAN}{'═' * 50}{Style.RESET_ALL}")
    print(f"  Current Price: {format_currency(prediction.current_price)}")
    print()
    
    if prediction.direction == "UP":
        dir_color = Fore.GREEN
        dir_emoji = "📈"
    elif prediction.direction == "DOWN":
        dir_color = Fore.RED
        dir_emoji = "📉"
    else:
        dir_color = Fore.YELLOW
        dir_emoji = "➡️"
    
    print(f"  {dir_emoji} Predicted Direction: {dir_color}{prediction.direction}{Style.RESET_ALL}")
    print(f"  Expected Change: {dir_color}{prediction.predicted_change_pct:+.2f}%{Style.RESET_ALL}")
    print(f"  Confidence: {prediction.confidence_score:.0f}%")
    print()
    
    print(f"  Confidence Interval:")
    print(f"    Low:  {prediction.confidence_low:+.2f}%")
    print(f"    High: {prediction.confidence_high:+.2f}%")
    print()
    
    if len(prediction.predicted_prices) > 1:
        print(f"  {Fore.CYAN}Daily Predictions:{Style.RESET_ALL}")
        for i, price in enumerate(prediction.predicted_prices, 1):
            change = ((price - prediction.current_price) / prediction.current_price) * 100
            c = Fore.GREEN if change > 0 else (Fore.RED if change < 0 else Fore.WHITE)
            print(f"    Day {i}: {format_currency(price)} ({c}{change:+.2f}%{Style.RESET_ALL})")
        print()
    
    if not prediction.model_available:
        print(f"  {Fore.YELLOW}ℹ️ Using fallback (momentum-based) prediction{Style.RESET_ALL}")
        print(f"     Run 'python main.py train' to train ML models")
    
    print(f"{Fore.CYAN}{'═' * 50}{Style.RESET_ALL}\n")
    return 0


def cmd_sentiment(config: Dict, args: argparse.Namespace) -> int:
    """Sentiment analysis for a ticker."""
    ticker = args.ticker.upper()
    print(f"\n{Fore.CYAN}📰 Sentiment Analysis: {ticker}{Style.RESET_ALL}\n")
    
    if not ML_AVAILABLE:
        print(f"{Fore.YELLOW}⚠️ ML features not available. Install with:{Style.RESET_ALL}")
        print(f"   pip install vaderSentiment")
        return 1
    
    if not config.get('ml', {}).get('enabled', True):
        print(f"{Fore.YELLOW}⚠️ ML features disabled in config.yaml{Style.RESET_ALL}")
        return 1
    
    analyzer = SentimentAnalyzer(config)
    
    status = analyzer.get_status()
    if not status['loaded']:
        print(f"{Fore.YELLOW}⚠️ VADER not loaded: {status.get('error', 'Unknown error')}{Style.RESET_ALL}")
        print(f"   Using keyword-based fallback analysis\n")
    else:
        print(f"  Engine: {Fore.GREEN}VADER (with financial lexicon){Style.RESET_ALL}")
        print()
    
    print(f"  Fetching news for {ticker}...", end=" ", flush=True)
    sentiment = analyzer.analyze_ticker(ticker)
    
    if sentiment is None:
        print(f"{Fore.RED}No news found{Style.RESET_ALL}")
        return 1
    print(f"{Fore.GREEN}Found {sentiment.headline_count} headlines{Style.RESET_ALL}\n")
    
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}")
    
    emoji = sentiment.get_emoji()
    if sentiment.sentiment_label == "BULLISH":
        label_color = Fore.GREEN
    elif sentiment.sentiment_label == "BEARISH":
        label_color = Fore.RED
    else:
        label_color = Fore.YELLOW
    
    print(f"  {emoji} Overall Sentiment: {label_color}{sentiment.sentiment_label}{Style.RESET_ALL}")
    print(f"  Score: {sentiment.overall_score:+.2f} (range: -1 to +1)")
    print()
    
    print(f"  {Fore.GREEN}Positive:{Style.RESET_ALL} {sentiment.positive_pct:.0f}%")
    print(f"  {Fore.RED}Negative:{Style.RESET_ALL} {sentiment.negative_pct:.0f}%")
    print(f"  {Fore.YELLOW}Neutral:{Style.RESET_ALL} {sentiment.neutral_pct:.0f}%")
    print()
    
    if sentiment.headlines:
        print(f"{Fore.CYAN}Recent Headlines:{Style.RESET_ALL}")
        print(f"{'─' * 60}")
        for h in sentiment.headlines[:8]:
            if h.sentiment == 'positive':
                s_emoji = "🟢"
                s_color = Fore.GREEN
            elif h.sentiment == 'negative':
                s_emoji = "🔴"
                s_color = Fore.RED
            else:
                s_emoji = "⚪"
                s_color = Fore.WHITE
            
            headline_short = h.headline[:55] + "..." if len(h.headline) > 55 else h.headline
            print(f"  {s_emoji} {s_color}{headline_short}{Style.RESET_ALL}")
            print(f"     {Fore.CYAN}{h.source}{Style.RESET_ALL} • {h.date or 'Recent'}")
        print()
    
    if not sentiment.model_available:
        print(f"  {Fore.YELLOW}ℹ️ Using fallback (keyword-based) analysis{Style.RESET_ALL}")
        print(f"     Install vaderSentiment for enhanced analysis")
    
    print(f"{Fore.CYAN}{'═' * 60}{Style.RESET_ALL}\n")
    return 0


def cmd_global(config: Dict, args: argparse.Namespace) -> int:
    """Show global market indicators."""
    print(f"\n{Fore.CYAN}🌍 GLOBAL MARKET INDICATORS{Style.RESET_ALL}")
    print(f"{'═' * 55}\n")
    
    global_analyzer = GlobalMarketAnalyzer(config)
    context = global_analyzer.analyze_global_context()
    
    # Overall sentiment
    sentiment_colors = {
        'RISK_ON': Fore.GREEN,
        'RISK_OFF': Fore.RED,
        'NEUTRAL': Fore.YELLOW,
        'UNCERTAIN': Fore.MAGENTA
    }
    sentiment_emojis = {
        'RISK_ON': '🟢',
        'RISK_OFF': '🔴',
        'NEUTRAL': '🟡',
        'UNCERTAIN': '⚠️'
    }
    
    s_color = sentiment_colors.get(context.sentiment.value, Fore.WHITE)
    s_emoji = sentiment_emojis.get(context.sentiment.value, '❓')
    
    print(f"  Market Sentiment: {s_emoji} {s_color}{context.sentiment.value}{Style.RESET_ALL}")
    print(f"  Risk Score: {context.risk_score:.0f}/100 (higher = more risk-off)")
    print(f"  Signal Adjustment: {context.confidence_adjustment:+.0f}%")
    print()
    
    # Key indicators
    print(f"{Fore.CYAN}Key Indicators:{Style.RESET_ALL}")
    print(f"{'─' * 55}")
    
    vix_data = global_analyzer.get_vix_signal()
    vix_level = vix_data['level']
    
    if vix_level > 35:
        vix_color = Fore.RED
        vix_status = "🚨 EXTREME FEAR"
    elif vix_level > 25:
        vix_color = Fore.YELLOW
        vix_status = "⚠️ Elevated"
    elif vix_level < 15:
        vix_color = Fore.GREEN
        vix_status = "😴 Complacent"
    else:
        vix_color = Fore.WHITE
        vix_status = "Normal"
    
    print(f"  VIX: {vix_color}{vix_level:.1f}{Style.RESET_ALL} - {vix_status}")
    print(f"  USD (DXY): {context.dxy_trend}")
    print(f"  10Y Treasury: {context.treasury_10y:.2f}%")
    print()
    
    # Indicator table
    print(f"{'INDICATOR':<15} {'PRICE':>12} {'CHANGE':>10} {'SIGNAL':>10}")
    print(f"{'─' * 15}─{'─' * 12}─{'─' * 10}─{'─' * 10}")
    
    for ind in context.indicators:
        change_str = f"{ind.change_pct:+.2f}%"
        
        if ind.signal == "BULLISH":
            sig_color = Fore.GREEN
            sig_emoji = "🟢"
        elif ind.signal == "BEARISH":
            sig_color = Fore.RED
            sig_emoji = "🔴"
        else:
            sig_color = Fore.YELLOW
            sig_emoji = "⚪"
        
        change_color = Fore.GREEN if ind.change_pct > 0 else (Fore.RED if ind.change_pct < 0 else Fore.WHITE)
        
        print(f"{ind.name:<15} {ind.current_price:>12.2f} {change_color}{change_str:>10}{Style.RESET_ALL} {sig_emoji} {sig_color}{ind.signal}{Style.RESET_ALL}")
    
    print()
    
    # Recommendations
    should_reduce, reason = global_analyzer.should_reduce_exposure()
    
    print(f"{Fore.CYAN}📋 Recommendations:{Style.RESET_ALL}")
    if should_reduce:
        print(f"  {Fore.YELLOW}⚠️ Consider reducing exposure: {reason}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.GREEN}✓ Global conditions support normal trading{Style.RESET_ALL}")
    
    print()
    return 0


def cmd_train(config: Dict, args: argparse.Namespace) -> int:
    """Train all ML models."""
    print(f"\n{Fore.CYAN}🤖 Training ML Models{Style.RESET_ALL}\n")
    
    if not ML_AVAILABLE:
        print(f"{Fore.YELLOW}⚠️ ML features not available. Install with:{Style.RESET_ALL}")
        print(f"   pip install scikit-learn vaderSentiment joblib")
        return 1
    
    if not config.get('ml', {}).get('enabled', True):
        print(f"{Fore.YELLOW}⚠️ ML features disabled in config.yaml{Style.RESET_ALL}")
        return 1
    
    watchlist = config.get('watchlist', [])
    force = getattr(args, 'force', False)
    
    print(f"  Backend: {Fore.GREEN}sklearn (lightweight){Style.RESET_ALL}")
    print(f"  Stocks: {len(watchlist)}")
    print(f"  Force retrain: {'Yes' if force else 'No'}")
    print()
    
    fetcher = DataFetcher()
    ensemble = MLEnsemble(config)
    
    import time
    start = time.time()
    
    results = ensemble.train_all(watchlist, fetcher, force=force)
    
    elapsed = time.time() - start
    
    # Price prediction results
    print(f"\n{Fore.CYAN}📈 Price Prediction Models:{Style.RESET_ALL}")
    price_results = results.get('price', {})
    success = sum(1 for v in price_results.values() if v)
    failed = sum(1 for v in price_results.values() if not v)
    print(f"  {Fore.GREEN}✅ Trained: {success}{Style.RESET_ALL}")
    if failed:
        print(f"  {Fore.RED}❌ Failed: {failed}{Style.RESET_ALL}")
        for ticker, ok in price_results.items():
            if not ok:
                print(f"     - {ticker}")
    
    # Pattern results
    print(f"\n{Fore.CYAN}📊 Pattern Recognition Models:{Style.RESET_ALL}")
    pat_results = results.get('patterns', {})
    success = sum(1 for v in pat_results.values() if v)
    failed = sum(1 for v in pat_results.values() if not v)
    print(f"  {Fore.GREEN}✅ Trained: {success}{Style.RESET_ALL}")
    if failed:
        print(f"  {Fore.RED}❌ Failed: {failed}{Style.RESET_ALL}")
    
    # Sentiment status
    print(f"\n{Fore.CYAN}📰 Sentiment Engine:{Style.RESET_ALL}")
    if ensemble.sentiment_analyzer and ensemble.sentiment_analyzer.is_available():
        print(f"  {Fore.GREEN}✅ VADER ready (no training needed){Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}⚠️ VADER not available{Style.RESET_ALL}")
    
    print(f"\n  ⏱️ Total time: {elapsed:.1f}s")
    print(f"  💾 Models cached in data/models/")
    print()
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
    
    # Forex
    forex_p = subparsers.add_parser('forex', help='Forex pair analysis')
    
    # Events
    events_p = subparsers.add_parser('events', help='Economic calendar and events')
    events_p.add_argument('--hours', type=int, default=48, help='Hours to look ahead')
    
    # Global
    global_p = subparsers.add_parser('global', help='Global market indicators')
    
    # AI Predict
    predict_p = subparsers.add_parser('predict', help='AI price prediction')
    predict_p.add_argument('ticker', help='Stock symbol')
    
    # Sentiment
    sentiment_p = subparsers.add_parser('sentiment', help='News sentiment analysis')
    sentiment_p.add_argument('ticker', help='Stock symbol')
    
    # Train ML models
    train_p = subparsers.add_parser('train', help='Train ML models')
    train_p.add_argument('--force', '-f', action='store_true', help='Force retrain all models')
    
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
        'forex': cmd_forex,
        'events': cmd_events,
        'global': cmd_global,
        'predict': cmd_predict,
        'sentiment': cmd_sentiment,
        'train': cmd_train,
    }
    
    if args.command in commands:
        return commands[args.command](config, args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
