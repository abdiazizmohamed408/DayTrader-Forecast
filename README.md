# 📈 DayTrader-Forecast

A Python-based technical analysis tool for day trading that scans stocks and generates probability-based trading signals.

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

- **Technical Analysis**
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - SMA/EMA (Simple/Exponential Moving Averages)
  - Bollinger Bands
  - Volume Analysis
  - Support/Resistance Levels

- **Probability Scoring**
  - Weighted scoring system (0-100%)
  - Multiple indicator agreement
  - Bullish/Bearish/Neutral classification

- **Signal Generation**
  - BUY/SELL/HOLD recommendations
  - Entry price, target, and stop-loss levels
  - Risk/Reward ratio calculations

- **📊 Prediction Tracking & Backtesting**
  - Automatic logging of all predictions to SQLite database
  - Outcome verification (WIN/LOSS based on target/stop-loss)
  - Win rate and profit factor calculation
  - Performance breakdown by ticker and indicator
  - Adaptive weight optimization based on historical accuracy

- **Reports**
  - Daily market scan reports
  - Individual stock analysis
  - Performance accuracy reports
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

This scans all stocks in your watchlist and displays:
- Current signals (BUY/SELL/HOLD)
- Probability scores
- Current prices
- Market sentiment

### Analyze a Specific Stock

```bash
python main.py analyze AAPL
```

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

### Verify Predictions

Check if pending predictions hit their targets or stop-losses:

```bash
python main.py verify
```

Options:
- `--days` or `-d`: Maximum days to track predictions (default: 10)

```bash
python main.py verify --days 5
```

### View Performance

Display model accuracy and prediction statistics:

```bash
python main.py performance
```

Options:
- `--save` or `-s`: Save full performance report to markdown file

```bash
python main.py performance --save
```

Example output:
```
📈 PREDICTION PERFORMANCE
══════════════════════════════════════════════════════════════

  Model Accuracy:
  [████████████████████░░░░░░░░░░] 67.5%
  27 wins / 13 losses

  Key Metrics:
    Total Predictions: 45
    Avg Win: +3.42%
    Avg Loss: -1.85%
    Profit Factor: 2.45
```

### Optimize Weights

Calculate optimal indicator weights based on historical performance:

```bash
python main.py optimize
```

Options:
- `--apply` or `-a`: Show how to apply optimized weights
- `--min-trades`: Minimum trades for weight adjustment (default: 10)
- `--learning-rate`: Learning rate for adjustments (default: 0.1)

```bash
python main.py optimize --apply
```

### Adaptive Scanning

Use performance-optimized weights when scanning:

```bash
python main.py scan --adaptive
```

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
```

### Email Configuration

To enable email reports, create a `.env` file:

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

## 📊 Prediction Tracking System

The tool automatically tracks all predictions to measure accuracy over time.

### How It Works

1. **Logging**: When you run `scan` or `analyze`, BUY/SELL signals are logged to a SQLite database with:
   - Entry price, target, stop-loss
   - Probability score
   - Individual indicator scores (RSI, MACD, etc.)

2. **Verification**: Run `verify` to check if predictions hit their targets:
   - **WIN**: Price reached target before stop-loss
   - **LOSS**: Price hit stop-loss before target
   - **EXPIRED**: Neither hit within the tracking period

3. **Performance Analysis**: Run `performance` to see:
   - Overall win rate
   - Average profit/loss per trade
   - Profit factor (gross profit / gross loss)
   - Performance by ticker
   - Indicator effectiveness

4. **Adaptive Learning**: Run `optimize` to calculate which indicators are most predictive and adjust weights accordingly.

### Database Location

Predictions are stored in `./data/predictions.db` (SQLite).

## 📁 Project Structure

```
DayTrader-Forecast/
├── main.py              # CLI entry point
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .env.example         # Environment template
├── .gitignore          # Git ignore rules
├── analyzers/
│   ├── __init__.py
│   ├── technical.py     # Technical indicators
│   └── signals.py       # Signal generation
├── data/
│   ├── __init__.py
│   ├── fetcher.py       # Data fetching (yfinance)
│   └── predictions.db   # SQLite database (auto-created)
├── tracking/
│   ├── __init__.py
│   ├── database.py      # SQLite database handler
│   ├── tracker.py       # Prediction logging & verification
│   └── performance.py   # Performance analysis
├── reports/
│   ├── __init__.py
│   └── generator.py     # Report generation
└── utils/
    ├── __init__.py
    └── helpers.py       # Utility functions
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

```
📈 DayTrader-Forecast
═══════════════════════════════════════════════════════

📊 SCAN RESULTS
═══════════════════════════════════════════════════════

TICKER   │ SIG  │  PROB │      PRICE │ SENTIMENT
─────────┼──────┼───────┼────────────┼───────────
🟢 NVDA  │ BUY  │  78.5% │    $875.32 │ BULLISH
🟢 AAPL  │ BUY  │  65.2% │    $178.90 │ BULLISH
🟡 MSFT  │ HOLD │  52.1% │    $415.67 │ NEUTRAL
🔴 TSLA  │ SELL │  61.8% │    $185.42 │ BEARISH

🟢 BUY: 2  │  🔴 SELL: 1  │  🟡 HOLD: 1
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
