# 0056 Backtesting Engine

This repository contains a complete backtesting engine and simple web
interface for analysing trading strategies on the Yuanta/P‑shares Taiwan
Dividend Plus ETF (stock code **0056**).  The project was built in
response to a user request to test a range of technical indicator
parameters and identify the most profitable combinations.

## Contents

- **0056.xlsx**: Five years of daily price data for the 0056 ETF.  This
  dataset is loaded by the backtester and serves as the basis for all
  simulations.
- **backtester/backtest.py**: Core backtesting engine.  Provides
  functions to load the data, compute technical indicators (Stochastic
  K%D, RSI, MACD and moving average), execute a single backtest and
  enumerate a grid of parameter combinations.
- **backtester/run_grid.py**: Script that performs a full grid search
  over the specified parameter ranges.  It writes three Excel files:
  all results, the top 10 parameter sets and a detailed log of trades
  for the single best set.
- **server.py**: A lightweight HTTP server that exposes a web form for
  running a backtest with arbitrary parameters.  Start it with
  `python server.py` and then navigate to
  [http://localhost:8000](http://localhost:8000) to interactively test
  strategies.

## Running the Grid Search

To evaluate every combination of parameters defined in
`backtester/run_grid.py`, run:

```bash
python backtester/run_grid.py
```

The script will iterate over all parameter combinations (over
300 000 in total), compute win rates and total profits and write the
results to Excel files in the current directory:

- `0056_backtest_all.xlsx` – complete results for every tested
  combination.
- `0056_backtest_top10.xlsx` – the top 10 combinations sorted by win
  rate (ties broken by total profit).
- `0056_backtest_best_details.xlsx` – a detailed trade log for the
  single best parameter set.

Running the grid search may take several minutes.  Progress messages
are printed to the console.

## Starting the Web Server

To explore individual parameter sets interactively, launch the
built‑in HTTP server:

```bash
python server.py
```

The server listens on port 8000.  Open your browser and navigate to
`http://localhost:8000`.  You will see a form with fields for each
strategy parameter.  Adjust the values as desired and click “Run
Backtest” to see the win rate, total profit and a table of all trades.
No data leaves your machine; the computation is performed locally
against the bundled 0056 dataset.

## Parameter Definitions

The strategy uses a combination of technical indicators and rules:

- **Buy KD Upper Limit** – trigger a buy when the Stochastic %K is
  greater than or equal to this value.
- **Buy RSI Upper Limit** – trigger a buy when the RSI is greater than
  or equal to this value.
- **Sell KD Lower Limit** – trigger a sell when the Stochastic %K is
  less than or equal to this value.
- **Sell RSI Lower Limit** – trigger a sell when the RSI is less than or
  equal to this value.
- **Sell when price ≥ 3% above average** – if enabled, close the
  position whenever the closing price exceeds the average cost by 3 %,
  regardless of other sell signals.
- **Enable MA Filter** – if enabled, only allow new buys when the
  closing price is above its 20‑day moving average.
- **KD Period Days** – lookback window for computing the Stochastic %K.
- **KD D‑Value Smoothing Days** – smoothing period for Stochastic %D
  (simple moving average of %K).
- **Allow loss sell** – if disabled, the backtester will not sell a
  position at a loss unless the 3 % profit trigger fires.
- **Units to buy per trade** – number of 1 000‑share lots to purchase
  each time the buy conditions are met.  The maximum position size is
  10 units.

## License

This project is provided for educational purposes and carries no
warranty.  Use at your own risk.