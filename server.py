"""Simple web interface for backtesting the 0056 strategy.

This module exposes a minimal HTTP server that allows a user to specify
strategy parameters via a web form and run a single backtest on demand.
The server leverages the backtesting engine defined in
``backtester/backtest.py`` and uses the historical price data from
``0056.xlsx``.  When a parameter set is submitted, the server returns
the win rate, total profit and a table of all trades.

Usage::

    python server.py

Once running, navigate to ``http://localhost:8000`` in your browser to
adjust parameters and execute backtests.  The server runs in the
foreground; to stop it press ``Ctrl+C``.

This server intentionally avoids external dependencies such as Flask.
Instead it uses the standard library's ``http.server`` module to serve
requests.  It is not intended for production use but suffices for
demonstrations and exploration.
"""

import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from backtester.backtest import (
    BacktestParameters,
    compute_indicators,
    load_data,
    backtest,
)


# Load the historical data once at module import time.  This prevents
# reloading the Excel file on every request and improves performance.
DATA_PATH = "0056.xlsx"
try:
    HIST_DF = load_data(DATA_PATH)
except FileNotFoundError:
    # If the data file is missing, provide a clear error.  The server will
    # still start, but any request that attempts to run a backtest will
    # display this message.
    HIST_DF = None


def render_form(values: dict) -> str:
    """Render the HTML form with the given default values.

    Parameters
    ----------
    values : dict
        Dictionary of parameter names to their current values.  Missing
        entries will fallback to sensible defaults.

    Returns
    -------
    str
        HTML string representing the form.
    """
    # Provide default values for each parameter
    defaults = {
        "buy_kd_upper": "80",
        "buy_rsi_upper": "75",
        "sell_kd_lower": "15",
        "sell_rsi_lower": "25",
        "sell_price_above_avg": "on",  # checkbox on means enabled
        "enable_ma_filter": "on",
        "kd_period": "12",
        "kd_smooth": "3",
        "allow_loss_sell": "on",
        "buy_units_each_time": "1",
    }
    # Merge provided values with defaults
    merged = {**defaults, **values}
    # Helper to determine whether a checkbox should be checked
    def checked(name: str) -> str:
        return "checked" if merged.get(name) in ("on", "true", "True", True) else ""

    form_html = f"""
        <form method="get" action="/run">
            <h2>0056 Backtest Parameters</h2>
            <label>Buy KD Upper Limit:
                <input type="number" name="buy_kd_upper" value="{html.escape(str(merged['buy_kd_upper']))}" step="1" min="0" max="100">
            </label><br>
            <label>Buy RSI Upper Limit:
                <input type="number" name="buy_rsi_upper" value="{html.escape(str(merged['buy_rsi_upper']))}" step="1" min="0" max="100">
            </label><br>
            <label>Sell KD Lower Limit:
                <input type="number" name="sell_kd_lower" value="{html.escape(str(merged['sell_kd_lower']))}" step="1" min="0" max="100">
            </label><br>
            <label>Sell RSI Lower Limit:
                <input type="number" name="sell_rsi_lower" value="{html.escape(str(merged['sell_rsi_lower']))}" step="1" min="0" max="100">
            </label><br>
            <label>Sell when price ≥ 3% above average:
                <input type="checkbox" name="sell_price_above_avg" {checked('sell_price_above_avg')}>
            </label><br>
            <label>Enable MA Filter:
                <input type="checkbox" name="enable_ma_filter" {checked('enable_ma_filter')}>
            </label><br>
            <label>KD Period Days:
                <input type="number" name="kd_period" value="{html.escape(str(merged['kd_period']))}" step="1" min="1">
            </label><br>
            <label>KD D-Value Smoothing Days:
                <input type="number" name="kd_smooth" value="{html.escape(str(merged['kd_smooth']))}" step="1" min="1">
            </label><br>
            <label>Allow loss sell:
                <input type="checkbox" name="allow_loss_sell" {checked('allow_loss_sell')}>
            </label><br>
            <label>Units to buy per trade:
                <input type="number" name="buy_units_each_time" value="{html.escape(str(merged['buy_units_each_time']))}" step="1" min="1" max="10">
            </label><br>
            <input type="submit" value="Run Backtest">
        </form>
    """
    return form_html


class BacktestRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler that serves the backtesting form and results."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        if parsed.path == "/":
            # Render the parameter form
            html_body = render_form({})
            self._send_html(200, html_body)
        elif parsed.path == "/run":
            # Ensure the historical data was loaded
            if HIST_DF is None:
                self._send_html(500, "<h1>Error</h1><p>Historical data file not found.</p>")
                return
            # Parse query parameters
            params = parse_qs(parsed.query)
            # Extract and coerce parameters
            try:
                def get_num(name: str, default: float) -> float:
                    return float(params.get(name, [default])[0])
                def get_int(name: str, default: int) -> int:
                    return int(params.get(name, [default])[0])
                def get_bool(name: str, default: bool) -> bool:
                    val = params.get(name, [None])[0]
                    if val is None:
                        return default
                    return val.lower() in ("on", "true", "1", "yes") if isinstance(val, str) else bool(val)

                p = BacktestParameters(
                    buy_kd_upper=get_num("buy_kd_upper", 80),
                    buy_rsi_upper=get_num("buy_rsi_upper", 75),
                    sell_kd_lower=get_num("sell_kd_lower", 15),
                    sell_rsi_lower=get_num("sell_rsi_lower", 25),
                    sell_price_above_avg=get_bool("sell_price_above_avg", True),
                    enable_ma_filter=get_bool("enable_ma_filter", True),
                    kd_period=get_int("kd_period", 12),
                    kd_smooth=get_int("kd_smooth", 3),
                    allow_loss_sell=get_bool("allow_loss_sell", True),
                    buy_units_each_time=get_int("buy_units_each_time", 1),
                )
            except ValueError:
                self._send_html(400, "<h1>Bad Request</h1><p>Invalid parameter value.</p>")
                return
            # Compute indicators and run backtest
            ind_df = compute_indicators(HIST_DF, p.kd_period, p.kd_smooth)
            win_rate, total_profit, trades = backtest(ind_df, p)
            # Generate HTML result page
            result_html = f"<h2>Backtest Result</h2>"
            result_html += f"<p><strong>Win rate:</strong> {win_rate:.2%}</p>"
            result_html += f"<p><strong>Total profit:</strong> {total_profit:,.2f}</p>"
            # Include trades table if there are any trades
            if trades:
                trades_df = pd.DataFrame(trades)
                # Build an HTML table from the DataFrame
                table_html = trades_df.to_html(index=False, border=1, classes="trade-table")
                result_html += "<h3>Trade Details</h3>" + table_html
            else:
                result_html += "<p>No trades were executed with the given parameters.</p>"
            # Append a link back to the form, preserving the entered values
            back_link_params = {k: v[0] for k, v in params.items()}
            back_link_qs = "&".join(f"{k}={html.escape(str(v))}" for k, v in back_link_params.items())
            result_html += f"<p><a href='/?{back_link_qs}'>Back to form</a></p>"
            self._send_html(200, result_html)
        else:
            self._send_html(404, "<h1>Not Found</h1><p>The requested page was not found.</p>")

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging to keep the console clean."""
        return

    def _send_html(self, code: int, body: str) -> None:
        """Send an HTML response with the given status code and body."""
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))


def run(server_class=HTTPServer, handler_class=BacktestRequestHandler, port: int = 8000) -> None:
    """Start the HTTP server and serve requests forever."""
    server_address = ("0.0.0.0", port)
    httpd = server_class(server_address, handler_class)
    print(f"Serving on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run()
