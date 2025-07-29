import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Step 1: Fetch historical data
def fetch_data(tickers, start_date, end_date):
    """
    Fetch adjusted close prices for multiple tickers.
    Returns a DataFrame with tickers as columns and dates as index.
    """
    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Initialize empty DataFrame
    data = pd.DataFrame()
    
    # Download each ticker separately for maximum reliability
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}...")
            ticker_data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True
            )
            
            if not ticker_data.empty:
                # Use Adj Close if available, otherwise use Close
                if "Adj Close" in ticker_data.columns:
                    data[ticker] = ticker_data["Adj Close"]
                else:
                    data[ticker] = ticker_data["Close"]
                print(f"✓ Successfully downloaded {ticker}")
            else:
                print(f"✗ No data retrieved for {ticker}")
                
        except Exception as e:
            print(f"✗ Error downloading {ticker}: {e}")
    
    # Clean the data
    data = data.dropna()
    
    if data.empty:
        raise ValueError("No data retrieved. Please check your ticker symbols and date range.")
    
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    return data

# Step 2: Test for cointegration
def test_cointegration(series1, series2):
    """Test if two series are cointegrated using the Engle-Granger test."""
    coint_test = ts.coint(series1, series2)
    p_value = coint_test[1]
    return p_value

# Step 3: Calculate spread and z-score
def calculate_spread_zscore(series1, series2, window=30):
    """Calculate the log spread and its z-score."""
    # Calculate log spread
    spread = np.log(series1) - np.log(series2)
    
    # Calculate rolling statistics
    mean_spread = spread.rolling(window=window).mean()
    std_spread = spread.rolling(window=window).std()
    
    # Calculate z-score
    z_score = (spread - mean_spread) / std_spread
    
    return spread, z_score

# Step 4: Generate trading signals
def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """Generate trading signals based on z-score thresholds."""
    signals = pd.DataFrame(index=z_score.index)
    signals["Z_Score"] = z_score
    
    # Initialize position column
    signals["Position"] = 0
    
    # Generate entry signals
    signals.loc[z_score < -entry_threshold, "Position"] = 1  # Long spread
    signals.loc[z_score > entry_threshold, "Position"] = -1  # Short spread
    
    # Forward fill positions
    signals["Position"] = signals["Position"].replace(0, np.nan).ffill().fillna(0)
    
    # Generate exit signals
    exit_mask = (abs(z_score) < exit_threshold) & (signals["Position"] != 0)
    signals.loc[exit_mask, "Position"] = 0
    
    # Forward fill again to maintain positions
    signals["Position"] = signals["Position"].replace(0, np.nan).ffill().fillna(0)
    
    # Create individual stock signals
    signals["Long_Stock1"] = signals["Position"] == 1
    signals["Short_Stock1"] = signals["Position"] == -1
    signals["Exit"] = signals["Position"] == 0
    
    return signals

# Step 5: Backtest the strategy
def backtest_strategy(signals, returns, stock1_name, stock2_name):
    """Backtest the pairs trading strategy."""
    # Create positions DataFrame
    positions = pd.DataFrame(index=signals.index, columns=[stock1_name, stock2_name])
    positions = positions.fillna(0.0)
    
    # Set positions based on signals
    # When Position = 1 (Long spread): Long Stock1, Short Stock2
    # When Position = -1 (Short spread): Short Stock1, Long Stock2
    positions.loc[signals["Position"] == 1, stock1_name] = 1.0
    positions.loc[signals["Position"] == 1, stock2_name] = -1.0
    positions.loc[signals["Position"] == -1, stock1_name] = -1.0
    positions.loc[signals["Position"] == -1, stock2_name] = 1.0
    
    # Calculate portfolio returns
    portfolio_returns = (positions.shift(1) * returns).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, cumulative_returns, positions

# Step 6: Calculate performance metrics
def calculate_performance_metrics(portfolio_returns, cumulative_returns):
    """Calculate key performance metrics."""
    # Total return
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    
    # Annualized return
    days = len(cumulative_returns)
    annualized_return = ((cumulative_returns.iloc[-1] ** (252 / days)) - 1) * 100
    
    # Sharpe ratio
    if portfolio_returns.std() != 0:
        sharpe_ratio = (portfolio_returns.mean() * np.sqrt(252)) / portfolio_returns.std()
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    winning_days = (portfolio_returns > 0).sum()
    total_days = (portfolio_returns != 0).sum()
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Win Rate": win_rate
    }

# Step 7: Plot results
def plot_results(data, spread, z_score, signals, cumulative_returns, stock1_name, stock2_name):
    """Plot comprehensive results of the pairs trading strategy."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 16))
    
    # 1. Stock prices
    ax1 = axes[0]
    ax1.plot(data.index, data[stock1_name], label=stock1_name, color='blue')
    ax1.set_ylabel(f'{stock1_name} Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(data.index, data[stock2_name], label=stock2_name, color='orange')
    ax1_twin.set_ylabel(f'{stock2_name} Price', color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('Stock Prices')
    ax1.grid(True, alpha=0.3)
    
    # 2. Price spread
    ax2 = axes[1]
    ax2.plot(spread, label='Log Spread', color='purple')
    ax2.axhline(spread.mean(), color='black', linestyle='--', alpha=0.5, label='Mean')
    ax2.fill_between(spread.index, 
                     spread.mean() - spread.std(), 
                     spread.mean() + spread.std(), 
                     alpha=0.2, color='gray', label='±1 Std Dev')
    ax2.set_title('Log Price Spread')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Z-Score with entry/exit signals
    ax3 = axes[2]
    ax3.plot(z_score, label='Z-Score', color='blue', linewidth=1)
    ax3.axhline(2.0, color='red', linestyle='--', label='Entry Threshold', alpha=0.7)
    ax3.axhline(-2.0, color='red', linestyle='--', alpha=0.7)
    ax3.axhline(0.5, color='green', linestyle='--', label='Exit Threshold', alpha=0.5)
    ax3.axhline(-0.5, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(0.0, color='black', linestyle='-', alpha=0.3)
    ax3.fill_between(z_score.index, -2, 2, alpha=0.1, color='gray')
    ax3.set_title('Z-Score with Trading Signals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Positions
    ax4 = axes[3]
    ax4.plot(signals["Position"], label='Position', color='darkgreen', linewidth=2)
    ax4.fill_between(signals.index, 0, signals["Position"], 
                     where=signals["Position"] > 0, alpha=0.3, color='green', label='Long Spread')
    ax4.fill_between(signals.index, 0, signals["Position"], 
                     where=signals["Position"] < 0, alpha=0.3, color='red', label='Short Spread')
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_title('Trading Positions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative returns
    ax5 = axes[4]
    ax5.plot(cumulative_returns, label='Strategy Returns', color='green', linewidth=2)
    ax5.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax5.fill_between(cumulative_returns.index, 1, cumulative_returns, 
                     where=cumulative_returns > 1, alpha=0.3, color='green')
    ax5.fill_between(cumulative_returns.index, 1, cumulative_returns, 
                     where=cumulative_returns < 1, alpha=0.3, color='red')
    ax5.set_title('Cumulative Returns')
    ax5.set_ylabel('Cumulative Return')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    # Parameters
    tickers = ["MC.PA", "KER.PA"]  # LVMH and Kering
    stock1_name = "MC.PA"  # LVMH
    stock2_name = "KER.PA"  # Kering
    
    # Use current date as end date
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Strategy parameters
    window = 30
    entry_threshold = 2.0
    exit_threshold = 0.5
    
    print("=" * 60)
    print("PAIRS TRADING STRATEGY: LVMH vs KERING")
    print("=" * 60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Z-Score Window: {window} days")
    print(f"Entry Threshold: ±{entry_threshold}")
    print(f"Exit Threshold: ±{exit_threshold}")
    print("=" * 60)

    # Fetch data
    try:
        data = fetch_data(tickers, start_date, end_date)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Extract individual series
    stock1 = data[stock1_name]
    stock2 = data[stock2_name]

    # Test cointegration
    print("\nTesting for cointegration...")
    coint_pvalue = test_cointegration(stock1, stock2)
    print(f"Cointegration p-value: {coint_pvalue:.4f}")
    
    if coint_pvalue < 0.05:
        print("✓ The pair is cointegrated (p < 0.05), suitable for pairs trading.")
    else:
        print("⚠ Warning: The pair is not strongly cointegrated (p >= 0.05).")
        print("  Consider using a shorter time period or different pair.")

    # Calculate spread and z-score
    spread, z_score = calculate_spread_zscore(stock1, stock2, window)

    # Generate trading signals
    signals = generate_signals(z_score, entry_threshold, exit_threshold)

    # Calculate returns
    returns = data.pct_change().dropna()

    # Backtest
    portfolio_returns, cumulative_returns, positions = backtest_strategy(
        signals, returns, stock1_name, stock2_name
    )

    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_returns, cumulative_returns)
    
    # Print metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    for metric, value in metrics.items():
        if metric == "Sharpe Ratio":
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value:.2f}%")
    
    # Print current status
    current_z = z_score.iloc[-1] if not z_score.empty and not np.isnan(z_score.iloc[-1]) else None
    if current_z is not None:
        print("\n" + "=" * 60)
        print("CURRENT STATUS")
        print("=" * 60)
        print(f"Current Z-Score: {current_z:.3f}")
        
        if current_z > entry_threshold:
            print(f"Signal: SHORT {stock1_name}, LONG {stock2_name}")
            print("(Spread is too high, expect mean reversion)")
        elif current_z < -entry_threshold:
            print(f"Signal: LONG {stock1_name}, SHORT {stock2_name}")
            print("(Spread is too low, expect mean reversion)")
        elif abs(current_z) < exit_threshold:
            print("Signal: EXIT/NO POSITION")
            print("(Spread is near mean)")
        else:
            print("Signal: HOLD CURRENT POSITION")
            print("(Waiting for entry or exit signal)")

    # Plot results
    plot_results(data, spread, z_score, signals, cumulative_returns, stock1_name, stock2_name)

if __name__ == "__main__":
    main()