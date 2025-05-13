#!/usr/bin/env python
"""
Example script demonstrating how to use the Historical Data Management service for backtesting.

This script shows how to retrieve historical data with point-in-time accuracy for backtesting.
"""

import asyncio
import datetime
import logging
from typing import Dict, List, Any

import httpx
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API URL
API_URL = "http://localhost:8000"


async def get_historical_data(
    symbol: str,
    timeframe: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    point_in_time: datetime.datetime = None
) -> pd.DataFrame:
    """
    Get historical OHLCV data with point-in-time accuracy.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date
        end_date: End date
        point_in_time: Point-in-time for historical accuracy (None for latest)
        
    Returns:
        DataFrame with OHLCV data
    """
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start_timestamp": start_date.isoformat(),
        "end_timestamp": end_date.isoformat(),
        "format": "json"
    }
    
    if point_in_time:
        params["point_in_time"] = point_in_time.isoformat()
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/historical/ohlcv", params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        return df


def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        df: DataFrame with OHLCV data
        period: SMA period
        
    Returns:
        Series with SMA values
    """
    return df["close"].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        df: DataFrame with OHLCV data
        period: EMA period
        
    Returns:
        Series with EMA values
    """
    return df["close"].ewm(span=period, adjust=False).mean()


def backtest_sma_crossover(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 20
) -> Dict[str, Any]:
    """
    Backtest a simple SMA crossover strategy.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast SMA period
        slow_period: Slow SMA period
        
    Returns:
        Dictionary with backtest results
    """
    # Calculate indicators
    df["sma_fast"] = calculate_sma(df, fast_period)
    df["sma_slow"] = calculate_sma(df, slow_period)
    
    # Generate signals
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1
    
    # Calculate returns
    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["signal"].shift(1) * df["returns"]
    
    # Calculate cumulative returns
    df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1
    df["strategy_cumulative_returns"] = (1 + df["strategy_returns"]).cumprod() - 1
    
    # Calculate statistics
    total_return = df["strategy_cumulative_returns"].iloc[-1]
    annual_return = (1 + total_return) ** (252 / len(df)) - 1
    sharpe_ratio = df["strategy_returns"].mean() / df["strategy_returns"].std() * (252 ** 0.5)
    max_drawdown = (df["strategy_cumulative_returns"] - df["strategy_cumulative_returns"].cummax()).min()
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "df": df
    }


def plot_backtest_results(results: Dict[str, Any], symbol: str, timeframe: str) -> None:
    """
    Plot backtest results.
    
    Args:
        results: Dictionary with backtest results
        symbol: Trading symbol
        timeframe: Timeframe
    """
    df = results["df"]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price and indicators
    ax1.plot(df.index, df["close"], label="Close")
    ax1.plot(df.index, df["sma_fast"], label=f"SMA ({df['sma_fast'].name.split('_')[1]})")
    ax1.plot(df.index, df["sma_slow"], label=f"SMA ({df['sma_slow'].name.split('_')[1]})")
    
    # Plot buy/sell signals
    buy_signals = df[df["signal"] == 1].index
    sell_signals = df[df["signal"] == -1].index
    
    ax1.scatter(buy_signals, df.loc[buy_signals, "close"], marker="^", color="green", label="Buy")
    ax1.scatter(sell_signals, df.loc[sell_signals, "close"], marker="v", color="red", label="Sell")
    
    ax1.set_title(f"SMA Crossover Strategy - {symbol} {timeframe}")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative returns
    ax2.plot(df.index, df["cumulative_returns"], label="Buy & Hold")
    ax2.plot(df.index, df["strategy_cumulative_returns"], label="Strategy")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Returns")
    ax2.legend()
    ax2.grid(True)
    
    # Add statistics as text
    stats_text = (
        f"Total Return: {results['total_return']:.2%}\n"
        f"Annual Return: {results['annual_return']:.2%}\n"
        f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {results['max_drawdown']:.2%}"
    )
    
    ax2.text(
        0.02, 0.95, stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(f"backtest_{symbol}_{timeframe}.png")
    plt.show()


async def main() -> None:
    """Main entry point."""
    logger.info("Starting backtest example")
    
    # Set parameters
    symbol = "EURUSD"
    timeframe = "1h"
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=30)
    
    # Get historical data
    logger.info(f"Getting historical data for {symbol} {timeframe}")
    df = await get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    if df.empty:
        logger.error(f"No data found for {symbol} {timeframe}")
        return
    
    logger.info(f"Retrieved {len(df)} records")
    
    # Run backtest
    logger.info("Running backtest")
    results = backtest_sma_crossover(df)
    
    # Print results
    logger.info("Backtest results:")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Annual Return: {results['annual_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    
    # Plot results
    logger.info("Plotting results")
    plot_backtest_results(results, symbol, timeframe)
    
    logger.info("Backtest example complete")


if __name__ == "__main__":
    asyncio.run(main())
