# LVMH-Kering Pairs Trading Strategy

A statistical arbitrage strategy that exploits price divergences between luxury giants LVMH (MC.PA) and Kering (KER.PA) using mean reversion principles.

## Strategy Overview

This pairs trading algorithm monitors the historical price relationship between LVMH and Kering, two highly correlated luxury goods companies. When their price spread deviates significantly from the historical mean, the strategy takes market-neutral positions betting on convergence.

### Key Features
- **Market Neutral**: Simultaneous long/short positions minimize market risk
- **Statistical Foundation**: Uses z-score normalization and cointegration testing
- **Risk Managed**: Clear entry/exit signals based on statistical thresholds
- **Automated Execution**: Full backtesting and signal generation capabilities

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/lvmh-kering-pairs-trading.git
cd lvmh-kering-pairs-trading

# Install dependencies
pip install -r requirements.txt

# Run the strategy
python PairsTrading.LVMH.py
```

## Requirements

```txt
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
statsmodels>=0.14.0
matplotlib>=3.7.0
```

## How It Works

### 1. **Signal Generation**
- Calculates the log price spread between LVMH and Kering
- Normalizes spread using 30-day rolling z-score
- Generates signals when z-score exceeds ±2 standard deviations

### 2. **Position Management**
| Z-Score | Action | Position |
|---------|--------|----------|
| < -2.0 | Enter Long Spread | Buy LVMH, Sell Kering |
| > +2.0 | Enter Short Spread | Sell LVMH, Buy Kering |
| -0.5 to +0.5 | Exit | Close all positions |

### 3. **Risk Controls**
- Cointegration testing validates pair suitability
- Position sizing maintains market neutrality
- Rolling statistics adapt to changing market conditions

## Performance Metrics

The strategy tracks and reports:
- **Total Return**: Cumulative strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Annualized Return**: Year-over-year performance

## Strategy Parameters

```python
# Configurable parameters
LOOKBACK_WINDOW = 30      # Days for rolling statistics
ENTRY_THRESHOLD = 2.0     # Z-score for trade entry
EXIT_THRESHOLD = 0.5      # Z-score for trade exit
START_DATE = "2020-01-01" # Backtest start date
```

## Example Output

The strategy generates comprehensive visualizations including:
1. **Price Chart**: Both stocks with dual-axis plotting
2. **Spread Analysis**: Log spread with mean and standard deviation bands
3. **Z-Score Signals**: Entry/exit thresholds and current score
4. **Position Tracking**: Long/short spread positions over time
5. **Performance Chart**: Cumulative returns with drawdown periods

## Risk Disclaimer

**Important Considerations:**
- Past performance does not guarantee future results
- Requires ability to short sell (may incur borrowing costs)
- Vulnerable to structural breaks in pair relationship
- Transaction costs not included in basic backtest
- Best suited for high-liquidity trading environments

## Customization

### Modifying the Pair
```python
# Change to different stock pairs
tickers = ["AAPL", "MSFT"]  # Tech pair example
tickers = ["KO", "PEP"]      # Consumer goods example
```

### Adjusting Sensitivity
```python
# More conservative (fewer trades)
ENTRY_THRESHOLD = 2.5
EXIT_THRESHOLD = 1.0

# More aggressive (more trades)
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 0.0
```

## Understanding the Mathematics

### Z-Score Calculation
```
z_score = (current_spread - mean_spread) / std_spread
```

### Cointegration Test
Uses Augmented Dickey-Fuller test to verify statistical relationship:
- p-value < 0.05 indicates cointegration
- Lower p-values suggest stronger mean reversion tendency

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Improvement
- Dynamic threshold adjustment
- Multiple timeframe analysis
- Machine learning signal enhancement
- Additional risk metrics
- Real-time execution capabilities

## Trading Disclaimer
This software is for educational purposes only. Do not risk money you cannot afford to lose. The developers assume no responsibility for your trading results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

**Note**: This strategy is designed for educational and research purposes. Live trading requires additional considerations including execution slippage, transaction costs, and real-time data feeds.
