"""

Backtesting engine for Yuanta/P‑shares Taiwan Dividend Plus ETF (ticker 0056).

This module implements a simple position‑based backtester tailored to the rules
specified by the user.  It reads historical price data from a pandas
``DataFrame`` and computes technical indicators (stochastic K%D, RSI, MACD and
simple moving average) used to determine buy and sell signals.  A generic
``backtest`` function evaluates a single parameter set, returning the win
rate, total profit and a detailed list of all completed trades.  A helper
function ``enumerate_parameter_grid`` iterates over all combinations of the
specified parameter ranges and stores the results in a list of dictionaries.

The backtester makes the following assumptions:

* One ``unit`` represents a single lot of 1000 shares.  When buying the
  security, capital is reduced by ``price * unit_size * units_to_buy`` and
  when selling it is increased accordingly.
* Only one long position is allowed at a time; additional buys while a
  position is open scale in and adjust the average cost.  The maximum
  number of units held simultaneously is controlled by ``max_units``.
* A moving average (MA) filter is optional.  If enabled, new buys are only
  executed when the closing price is above the chosen MA period.
* When ``sell_price_above_avg`` is true, the backtester will immediately
  liquidate the position when the closing price exceeds the average cost by
  3 %.  This check has higher priority than indicator‑based exits.
* If ``allow_loss_sell`` is false, the backtester will ignore sell
  conditions when the position would result in a loss.  The position will
  remain open until either a profitable sell signal occurs or the data ends.

The indicator implementations are straightforward and avoid external
dependencies beyond ``pandas`` and ``numpy``.  The stochastic oscillator uses
high/low ranges over ``kd_period`` days to compute %K and a simple moving
average over ``kd_smooth`` days to compute %D.  RSI is calculated using
exponentially smoothed gains and losses with a default 14‑day period.  MACD
calculates the difference between two exponential moving averages and a
signal line; the difference between the MACD line and the signal line
determines whether it is positive or negative.

Example usage::

    import pandas as pd
    from backtester.backtest import load_data, backtest, enumerate_parameter_grid

    df = load_data('0056.xlsx')
    # define your parameter ranges here
    param_ranges = {
        'buy_kd_upper': [70, 75, 80],
        'buy_rsi_upper': [65, 70, 75],
        'sell_kd_lower': [10, 20, 30],
        'sell_rsi_lower': [10, 20, 30],
        'sell_price_above_avg': [True, False],
        'enable_ma_filter': [True, False],
        'kd_period': [9, 12],
        'kd_smooth': [3, 6],
        'allow_loss_sell': [True, False],
        'buy_units_each_time': [1, 2],
    }
    results = enumerate_parameter_grid(df, param_ranges)
    # results is a list of dictionaries with win rate and total profit

"""

import itertools
from dataclasses import dataclass
import dataclasses
from typing import Dict, Iterable, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load historical price data from an Excel file.

    The Excel file must contain at least the columns ``Date``, ``Open``,
    ``High``, ``Low``, ``Close`` and ``Volume``.  Dates are parsed into
    ``datetime64`` dtype and sorted in ascending order.

    Parameters
    ----------
    path : str
        Path to the Excel file.

    Returns
    -------
    pandas.DataFrame
        A dataframe with parsed dates and numerical columns.
    """
    df = pd.read_excel(path)
    # Ensure correct column names
    expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[list(expected_cols)].copy()
    # Convert date to datetime and sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_indicators(
    df: pd.DataFrame,
    kd_period: int,
    kd_smooth: int,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    ma_period: int = 20,
) -> pd.DataFrame:
    """Compute technical indicators used in the backtester.

    Parameters
    ----------
    df : pandas.DataFrame
        Price data with columns ``High``, ``Low`` and ``Close``.
    kd_period : int
        Look‑back window for stochastic %K.
    kd_smooth : int
        Smoothing window for stochastic %D.
    rsi_period : int, optional
        Period for RSI calculation, by default 14.
    macd_fast : int, optional
        Fast EMA period for MACD, by default 12.
    macd_slow : int, optional
        Slow EMA period for MACD, by default 26.
    macd_signal : int, optional
        Signal line EMA period for MACD, by default 9.
    ma_period : int, optional
        Simple moving average period for the optional MA filter, by default 20.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns: ``K``, ``D``, ``RSI``, ``MACD`` (MACD
        minus signal) and ``MA``.
    """
    data = df.copy()
    # Stochastic oscillator %K
    low_min = data["Low"].rolling(window=kd_period, min_periods=kd_period).min()
    high_max = data["High"].rolling(window=kd_period, min_periods=kd_period).max()
    k = 100 * (data["Close"] - low_min) / (high_max - low_min)
    # Avoid division by zero
    k = k.replace([np.inf, -np.inf], np.nan).fillna(0)
    data["K"] = k
    # %D: simple moving average of K
    data["D"] = data["K"].rolling(window=kd_smooth, min_periods=kd_smooth).mean()

    # RSI (exponential smoothing)
    close_diff = data["Close"].diff()
    gain = close_diff.where(close_diff > 0, 0.0)
    loss = -close_diff.where(close_diff < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = data["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    data["MACD"] = macd_line - signal_line  # positive when MACD above signal

    # Simple moving average for filter
    data["MA"] = data["Close"].rolling(window=ma_period, min_periods=ma_period).mean()
    return data


@dataclass
class BacktestParameters:
    buy_kd_upper: float
    buy_rsi_upper: float
    sell_kd_lower: float
    sell_rsi_lower: float
    sell_price_above_avg: bool
    enable_ma_filter: bool
    kd_period: int
    kd_smooth: int
    allow_loss_sell: bool
    buy_units_each_time: int
    max_units: int = 10
    initial_capital: float = 400000.0


def backtest(
    data: pd.DataFrame,
    params: BacktestParameters,
    ma_period: int = 20,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Execute a backtest with the given parameters on price/indicator data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing price data and precomputed indicators columns
        (``K``, ``D``, ``RSI``, ``MACD`` and ``MA``).  The indices must be
        sorted by date ascending.
    params : BacktestParameters
        Set of strategy parameters.
    ma_period : int, optional
        Period of the moving average used for the filter, by default 20.

    Returns
    -------
    win_rate : float
        Fraction of completed trades with positive profit (0–1).
    total_profit : float
        Net profit over all trades in currency units.
    trades : list of dict
        List of trade dictionaries with keys: ``buy_date``, ``sell_date``,
        ``buy_price``, ``sell_price``, ``units``, ``cost``, ``revenue``,
        ``profit``.
    """
    # Initialize capital and position state
    cash = params.initial_capital
    units_held = 0
    avg_cost = 0.0  # average purchase price per share (per unit lot)
    trades: List[Dict[str, Any]] = []
    current_trade_start: Optional[pd.Timestamp] = None
    current_trade_cost: float = 0.0

    for idx, row in data.iterrows():
        k_val = row["K"]
        rsi_val = row["RSI"]
        macd_val = row["MACD"]
        ma_val = row["MA"]
        price = row["Close"]
        date = row["Date"]

        # Buy condition: K >= buy_kd_upper, RSI >= buy_rsi_upper, MACD positive
        buy_signal = (
            (k_val >= params.buy_kd_upper)
            and (rsi_val >= params.buy_rsi_upper)
            and (macd_val > 0)
        )
        if params.enable_ma_filter:
            buy_signal = buy_signal and (not np.isnan(ma_val) and price > ma_val)

        # Execute buy if conditions met and capacity available
        if (
            buy_signal
            and units_held < params.max_units
            and params.buy_units_each_time > 0
        ):
            units_to_buy = min(
                params.buy_units_each_time, params.max_units - units_held
            )
            cost_per_unit = price * 1000  # each unit is 1000 shares
            total_cost = cost_per_unit * units_to_buy
            if cash >= total_cost:
                cash -= total_cost
                new_total_units = units_held + units_to_buy
                if units_held == 0:
                    avg_cost = price
                    current_trade_start = date
                    current_trade_cost = total_cost
                else:
                    avg_cost = (
                        avg_cost * units_held + price * units_to_buy
                    ) / new_total_units
                    current_trade_cost += total_cost
                units_held = new_total_units

        # Sell logic
        sell_reason_profit = False
        if units_held > 0:
            # Condition (1): price above average cost by 3%
            if params.sell_price_above_avg and price >= avg_cost * 1.03:
                sell_reason_profit = True
            # Condition (2): all sell indicators triggered
            sell_signal = (
                (k_val <= params.sell_kd_lower)
                and (rsi_val <= params.sell_rsi_lower)
                and (macd_val < 0)
            )
            if sell_reason_profit or sell_signal:
                potential_profit = (price - avg_cost) * units_held * 1000
                if (potential_profit >= 0) or params.allow_loss_sell or sell_reason_profit:
                    revenue = price * units_held * 1000
                    cash += revenue
                    trade_profit = revenue - current_trade_cost
                    trades.append(
                        {
                            "buy_date": current_trade_start,
                            "sell_date": date,
                            "buy_price": avg_cost,
                            "sell_price": price,
                            "units": units_held,
                            "cost": current_trade_cost,
                            "revenue": revenue,
                            "profit": trade_profit,
                        }
                    )
                    units_held = 0
                    avg_cost = 0.0
                    current_trade_start = None
                    current_trade_cost = 0.0

    # Close any open position at the last bar
    if units_held > 0 and current_trade_start is not None:
        price = data.iloc[-1]["Close"]
        revenue = price * units_held * 1000
        cash += revenue
        trade_profit = revenue - current_trade_cost
        trades.append(
            {
                "buy_date": current_trade_start,
                "sell_date": data.iloc[-1]["Date"],
                "buy_price": avg_cost,
                "sell_price": price,
                "units": units_held,
                "cost": current_trade_cost,
                "revenue": revenue,
                "profit": trade_profit,
            }
        )
        units_held = 0

    if trades:
        wins = sum(1 for t in trades if t["profit"] > 0)
        win_rate = wins / len(trades)
        total_profit = sum(t["profit"] for t in trades)
    else:
        win_rate = 0.0
        total_profit = 0.0
    return win_rate, total_profit, trades


def enumerate_parameter_grid(
    df: pd.DataFrame,
    param_ranges: Dict[str, Iterable[Any]],
    ma_period: int = 20,
) -> List[Dict[str, Any]]:
    """Enumerate all combinations of parameter ranges and evaluate each.

    This helper computes indicators only once per unique (kd_period, kd_smooth)
    pair to improve efficiency.  For each parameter combination the
    ``backtest`` function is invoked and win rate and total profit are
    recorded.

    Parameters
    ----------
    df : pandas.DataFrame
        Price data loaded via :func:`load_data`.
    param_ranges : dict
        Dictionary mapping parameter names (matching ``BacktestParameters``
        fields except ``max_units`` and ``initial_capital``) to iterables of
        candidate values.
    ma_period : int, optional
        Moving average period for the MA filter, by default 20.

    Returns
    -------
    list of dict
        Each dict contains keys ``win_rate``, ``total_profit`` and all
        parameter values.  Results are appended in the same order as the
        parameter grid.
    """
    unique_pairs = set(
        (p, s)
        for p in param_ranges.get("kd_period", [])
        for s in param_ranges.get("kd_smooth", [])
    )
    indicator_cache: Dict[Tuple[int, int], pd.DataFrame] = {}
    results: List[Dict[str, Any]] = []

    for (kd_period, kd_smooth) in unique_pairs:
        indicator_cache[(kd_period, kd_smooth)] = compute_indicators(
            df, kd_period, kd_smooth, ma_period=ma_period
        )

    keys = [
        "buy_kd_upper",
        "buy_rsi_upper",
        "sell_kd_lower",
        "sell_rsi_lower",
        "sell_price_above_avg",
        "enable_ma_filter",
        "kd_period",
        "kd_smooth",
        "allow_loss_sell",
        "buy_units_each_time",
    ]
    # Build a list of candidate values for each key.  If a key is not
    # provided in ``param_ranges``, fall back to the default specified on
    # the dataclass.  Dataclass fields are not accessible as attributes
    # when no default is provided, so we consult ``__dataclass_fields__``
    # instead.  ``Field.default`` will be ``MISSING`` if no default was
    # given.
    value_lists = []
    for k in keys:
        if k in param_ranges:
            value_lists.append(list(param_ranges[k]))
        else:
            field = BacktestParameters.__dataclass_fields__[k]
            if field.default is not dataclasses.MISSING:
                value_lists.append([field.default])
            else:
                raise ValueError(f"No values provided for parameter '{k}' and no default available")
    for values in itertools.product(*value_lists):
        params_dict = dict(zip(keys, values))
        kd_pair = (params_dict["kd_period"], params_dict["kd_smooth"])
        data_ind = indicator_cache[kd_pair]
        params = BacktestParameters(**params_dict)
        win_rate, total_profit, _ = backtest(data_ind, params, ma_period=ma_period)
        result_row = {
            "win_rate": win_rate,
            "total_profit": total_profit,
        }
        result_row.update(params_dict)
        results.append(result_row)
    return results


def export_results_to_excel(
    results: List[Dict[str, Any]],
    top_n: int = 10,
    details_data: Optional[pd.DataFrame] = None,
    best_params: Optional[BacktestParameters] = None,
    output_dir: str = ".",
    prefix: str = "results",
) -> Tuple[str, str, str]:
    """Export results and reports to Excel files.

    Parameters
    ----------
    results : list of dict
        Full list of result dictionaries as returned by
        :func:`enumerate_parameter_grid`.
    top_n : int, optional
        Number of top entries (by win rate) to include in the second file.
    details_data : pandas.DataFrame, optional
        DataFrame of trade details for the best parameter set.
    best_params : BacktestParameters, optional
        Parameter set used to generate the trade details sheet.
    output_dir : str, optional
        Directory in which to place the output files.
    prefix : str, optional
        Prefix for output filenames.

    Returns
    -------
    tuple of str
        Filenames for all results, top results and trade details (if
        applicable).
    """
    import os
    all_results_df = pd.DataFrame(results)
    all_results_path = os.path.join(output_dir, f"{prefix}_all.xlsx")
    all_results_df.to_excel(all_results_path, index=False)

    # Sort by win rate and total profit
    top_df = all_results_df.sort_values(
        by=["win_rate", "total_profit"], ascending=[False, False]
    ).head(top_n)
    top_results_path = os.path.join(output_dir, f"{prefix}_top{top_n}.xlsx")
    top_df.to_excel(top_results_path, index=False)

    details_path = ""
    if best_params is not None and details_data is not None:
        details_path = os.path.join(output_dir, f"{prefix}_best_details.xlsx")
        details_data.to_excel(details_path, index=False)
    return all_results_path, top_results_path, details_path
https://github.com/abow0403X/0056-backtest/edit/main/web/script.js
https://github.com/abow0403X/0056-backtest/edit/main/web/script.js
