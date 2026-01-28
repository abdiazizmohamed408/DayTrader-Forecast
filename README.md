# 📈 DayTrader-Forecast

A Python-based technical analysis tool for day trading that scans stocks and generates probability-based trading signals with **performance tracking**, **backtesting**, and **paper trading** capabilities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ⚠️ DISCLAIMER

**This tool is for EDUCATIONAL PURPOSES ONLY.**

- ❌ This is **NOT** financial advice
- ❌ Past performance does **NOT** guarantee future results
- ❌ Day trading involves **SIGNIFICANT RISK** of loss
- ❌ Never trade with money you cannot afford to lose
- ✅ Always do your own research before making any trading decisions
- ✅ Consider consulting a licensed financial advisor

**The creators of this tool are not responsible for any financial losses.**

---

## 🎯 Features

### Technical Analysis
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **SMA/EMA** (Simple/Exponential Moving Averages)
- **Bollinger Bands**
- **Volume Analysis** with confirmation (1.5x average threshold)
- **Support/Resistance Levels**

### Multi-Timeframe Analysis 🆕
- Analyzes 15min, 1hr, 4hr, and Daily timeframes
- Calculates alignment score across timeframes
- Higher confidence when multiple timeframes agree

### Market Context 🆕
- Checks SPY and QQQ trends before generating signals
- Adjusts signal confidence based on overall market direction
- VIX level monitoring for volatility context

### Probability Scoring
- Weighted scoring system (0-100%)
- Multiple indicator agreement
- Bullish/Bearish/Neutral classification
- Volume confirmation bonus

### Signal Generation
- BUY/SELL/HOLD recommendations
- Entry price, target, and stop-loss levels
- Risk/Reward ratio calculations
- Volume confirmation status

### Performance Tracking 🆕
- SQLite database for prediction logging
- Tracks every signal with entry, target, and stop-loss
- Automatic outcome checking (WIN/LOSS)
- Win rate, profit factor, and per-ticker statistics

### Backtesting Engine 🆕
- Test strategies on historical data
- Simulated P&L tracking
- Detailed trade-by-trade results
- Performance metrics (Sharpe ratio, max drawdown)

### Paper Trading 🆕
- Virtual trading with configurable balance
- Automatic position management
- Real-time price updates
- Portfolio tracking over time

### Email Alerts 🆕
- Automatic alerts for high-confidence signals (>75%)
- Configurable SMTP settings

### Reports
- Daily market scan reports
- Individual stock analysis
- Markdown export
- Email delivery (optional)

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/abdiazizmohamed408/DayTrader-Forecast.git
cd DayTrader-Forecast
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure (optional)

Edit `config.yaml` to customize:
- Watchlist stocks
- Technical analysis parameters
- Signal weights
- Risk settings

## 🚀 Usage

### Scan All Watchlist Stocks

```bash
python main.py scan
```

With minimum probability filter:

```bash
python main.py scan --min-prob 70   # Only show 70%+ signals
```

### Analyze a Specific Stock

```bash
python main.py analyze AAPL
```

This now includes:
- Multi-timeframe analysis breakdown
- Market context (SPY/QQQ trend)
- Volume confirmation status
- Timeframe alignment score

Options:
- `--save` or `-s`: Save analysis to a markdown file

```bash
python main.py analyze TSLA --save
```

### Generate Daily Report

```bash
python main.py report
```

Options:
- `--email` or `-e`: Send report via email (requires SMTP configuration)

```bash
python main.py report --email
```

### Backtesting 🆕

Test your strategy on historical data:

```bash
python main.py backtest --days 30
```

Options:
- `--days` or `-d`: Number of days to test (default: 30)
- `--min-prob`: Minimum probability for trades (default: 50)

Example output:
```
📊 BACKTEST RESULTS
══════════════════════════════════════════════════

📅 Period: 2024-01-01 to 2024-01-30
📆 Days Tested: 30

💰 PERFORMANCE
──────────────────────────────
Initial Balance:  $10,000.00
Final Balance:    $10,856.32
Total Return:     +8.56%
Max Drawdown:     3.21%

📈 TRADE STATISTICS
──────────────────────────────
Total Trades:     45
Wins:             28
Losses:           17
Win Rate:         62.2%

Avg Profit:       +2.85%
Avg Loss:         -1.67%
Profit Factor:    1.72
Sharpe Ratio:     1.45
```

### Paper Trading 🆕

Start a virtual trading session:

```bash
python main.py paper
```

Options:
- `--reset`: Start a new session
- `--balance`: Set starting balance (default: $10,000)
- `--auto`: Automatically execute signals

```bash
# Start with custom balance
python main.py paper --balance 25000

# Auto-execute signals based on current scan
python main.py paper --auto

# Reset and start fresh
python main.py paper --reset
```

### Performance Statistics 🆕

View your historical accuracy:

```bash
python main.py performance
```

Example output:
```
📊 PERFORMANCE SUMMARY
═════════════════════════════════════════════

Total Predictions: 156
Wins: 98 | Losses: 58
Win Rate: 62.8%

Avg Profit: +3.2%
Avg Loss: -1.8%
Profit Factor: 1.78
Total Return: +127.3%

📈 PERFORMANCE BY TICKER
─────────────────────────────────────────────
TICKER   │ TRADES │ WIN RATE │   RETURN
─────────┼────────┼──────────┼──────────
NVDA     │     23 │    75.0% │   +42.3%
AAPL     │     18 │    66.7% │   +28.1%
MSFT     │     15 │    60.0% │   +18.5%
TSLA     │     20 │    45.0% │    -5.2%

🏆 Best Performer:  NVDA (75% win rate)
📉 Worst Performer: TSLA (45% win rate)
```

Options:
- `--days`: Filter to last N days

## ⚙️ Configuration

### config.yaml

```yaml
# Your watchlist
watchlist:
  - AAPL
  - MSFT
  - GOOGL
  - TSLA
  - NVDA
  - SPY
  - QQQ

# Technical analysis settings
analysis:
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  sma_short: 20
  sma_long: 50

# Signal weights (must sum to 1.0)
weights:
  rsi: 0.20
  macd: 0.20
  moving_averages: 0.15
  bollinger_bands: 0.15
  volume: 0.15
  support_resistance: 0.15

# Risk management
risk:
  stop_loss_percent: 2.0
  take_profit_percent: 4.0

# Volume confirmation (optional)
require_volume_confirmation: false
```

### Email Configuration

To enable email alerts for high-confidence signals, create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your SMTP settings:

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_TO=recipient@example.com
```

High-confidence signals (>75%) will automatically trigger email alerts.

## 📊 Technical Indicators Explained

### RSI (Relative Strength Index)
- **Range:** 0-100
- **Overbought:** > 70 (potential sell signal)
- **Oversold:** < 30 (potential buy signal)

### MACD
- **Bullish crossover:** MACD line crosses above signal line
- **Bearish crossunder:** MACD line crosses below signal line

### Moving Averages
- **Golden Cross:** Short-term MA crosses above long-term MA (bullish)
- **Death Cross:** Short-term MA crosses below long-term MA (bearish)

### Bollinger Bands
- **Price near upper band:** Potentially overbought
- **Price near lower band:** Potentially oversold

### Volume Confirmation 🆕
- Signal is **confirmed** when volume > 1.5x average
- Adds +5% to probability when confirmed
- Reduces -3% when volume is below average

### Multi-Timeframe Alignment 🆕
- Analyzes 15min, 1hr, 4hr, Daily timeframes
- Strong alignment (80%+): +10% probability bonus
- Good alignment (70%+): +5% probability bonus
- Conflicting signals (<40%): -5% probability penalty

## 📁 Project Structure

```
DayTrader-Forecast/
├── main.py              # CLI entry point (all commands)
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .env.example         # Environment template
├── .gitignore          # Git ignore rules
├── analyzers/
│   ├── __init__.py
│   ├── technical.py     # Technical indicators + Multi-TF
│   ├── signals.py       # Signal generation
│   └── market.py        # Market context (SPY/QQQ) 🆕
├── data/
│   ├── __init__.py
│   ├── fetcher.py       # Data fetching (yfinance)
│   └── predictions.db   # SQLite database 🆕
├── tracking/            # 🆕
│   ├── __init__.py
│   └── tracker.py       # Performance tracking
├── backtesting/         # 🆕
│   ├── __init__.py
│   └── engine.py        # Backtesting engine
├── paper/               # 🆕
│   ├── __init__.py
│   └── simulator.py     # Paper trading
├── reports/
│   ├── __init__.py
│   └── generator.py     # Report generation
├── utils/
│   ├── __init__.py
│   └── helpers.py       # Utility functions
└── output/              # Generated reports
```

## 🔧 Dependencies

- **yfinance** - Yahoo Finance data API
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **pyyaml** - Configuration parsing
- **python-dotenv** - Environment variables
- **tabulate** - Table formatting
- **colorama** - Colored terminal output

## 📝 Example Output

### Scan Output
```
📈 DayTrader-Forecast
═══════════════════════════════════════════════════════

🌍 MARKET CONTEXT
────────────────────────────────────────
Market is bullish 📈. SPY: bullish 📈, QQQ: bullish 📈

SPY: +0.85% | Above 20 SMA: ✅ | Above 50 SMA: ✅
QQQ: +1.12% | Above 20 SMA: ✅ | Above 50 SMA: ✅
VIX: 14.32 (🟢 Low)

📊 SCAN RESULTS
═════════════════════════════════════════════════════════════════

TICKER   │ SIG  │  PROB │      PRICE │ SENTIMENT │ VOL
─────────┼──────┼───────┼────────────┼───────────┼─────
🟢 NVDA  │ BUY  │  82.5% │    $875.32 │ BULLISH   │ 📊
🟢 AAPL  │ BUY  │  68.2% │    $178.90 │ BULLISH   │ 📊
🟡 MSFT  │ HOLD │  52.1% │    $415.67 │ NEUTRAL   │
🔴 TSLA  │ SELL │  61.8% │    $185.42 │ BEARISH   │

🟢 BUY: 2  │  🔴 SELL: 1  │  🟡 HOLD: 1  │  ⭐ HIGH CONF: 1
```

### Analyze Output
```
═══════════════════════════════════════════════════════
🟢 Apple Inc. (AAPL)
═══════════════════════════════════════════════════════

🌍 MARKET CONTEXT
────────────────────────────────────────
Market is bullish 📈. SPY: bullish 📈, QQQ: bullish 📈

  Current Price: $178.90
  Change: +1.25%

  Signal: BUY
  Probability: 68.2%
  Sentiment: BULLISH
  Volume Confirmed: ✅ Yes
  Timeframe Alignment: 75%

  Stop Loss: $174.20
  Target: $186.50
  Risk/Reward: 1.62

  Technical Indicators:
    RSI(14): 58.3
    MACD: 0.8542
    SMA(20): $176.45
    SMA(50): $172.30
    Volume Ratio: 1.82x

  Multi-Timeframe Analysis:
    15m : BULLISH
    1h  : BULLISH
    4h  : NEUTRAL
    1d  : BULLISH
    Overall: BULLISH

  Signal Factors:
    • Price trading above long-term moving average (bullish)
    • MACD bullish crossover detected
    • ✅ Volume confirmed (1.8x average)
    • ✅ Multi-timeframe alignment: 75%
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for the excellent Yahoo Finance API wrapper
- The trading community for technical analysis knowledge

---

**Remember:** Trading involves risk. Use this tool responsibly and always do your own research.

## 📧 Contact

For questions or feedback: Abdiazizmohamed408@gmail.com
