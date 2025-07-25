// 0056 Backtest JavaScript
//
// This script loads five years of historical price data for the 0056 ETF
// from a JSON file and implements technical indicator computations and
// backtesting logic entirely in the browser.  Users can adjust strategy
// parameters via a simple form and immediately see win rates, total
// profits and trade details.

(() => {
  const form = document.getElementById('backtest-form');
  const resultDiv = document.getElementById('result');
  let priceData = [];

  // Load the price data from the JSON file once the page is loaded.
  fetch('0056.json')
    .then((resp) => resp.json())
    .then((records) => {
      priceData = records.map((row) => {
        return {
          date: new Date(row.Date),
          open: parseFloat(row.Open),
          high: parseFloat(row.High),
          low: parseFloat(row.Low),
          close: parseFloat(row.Close),
          volume: parseFloat(row.Volume),
        };
      });
    })
    .catch((err) => {
      resultDiv.innerHTML = `<p style="color:red">Failed to load data: ${err}</p>`;
    });

  // Event listener for form submission
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (priceData.length === 0) {
      resultDiv.innerHTML = '<p style="color:red">Data not loaded yet.</p>';
      return;
    }
    // Extract parameters from the form
    const formData = new FormData(form);
    const params = {
      buy_kd_upper: parseFloat(formData.get('buy_kd_upper')),
      buy_rsi_upper: parseFloat(formData.get('buy_rsi_upper')),
      sell_kd_lower: parseFloat(formData.get('sell_kd_lower')),
      sell_rsi_lower: parseFloat(formData.get('sell_rsi_lower')),
      sell_price_above_avg: formData.get('sell_price_above_avg') !== null,
      enable_ma_filter: formData.get('enable_ma_filter') !== null,
      kd_period: parseInt(formData.get('kd_period'), 10),
      kd_smooth: parseInt(formData.get('kd_smooth'), 10),
      allow_loss_sell: formData.get('allow_loss_sell') !== null,
      buy_units_each_time: parseInt(formData.get('buy_units_each_time'), 10),
      max_units: 10,
      initial_capital: 400000,
    };
    // Compute indicators and run backtest
    const dataWithIndicators = computeIndicators(priceData, params.kd_period, params.kd_smooth);
    const { winRate, totalProfit, trades } = backtest(dataWithIndicators, params);
    renderResult(winRate, totalProfit, trades);
  });

  // Compute indicators: K, D, RSI, MACD and MA
  function computeIndicators(data, kdPeriod, kdSmooth, rsiPeriod = 14, macdFast = 12, macdSlow = 26, macdSignal = 9, maPeriod = 20) {
    const n = data.length;
    // Initialize arrays
    const K = new Array(n).fill(0);
    const D = new Array(n).fill(0);
    const RSI = new Array(n).fill(0);
    const MACD = new Array(n).fill(0);
    const MA = new Array(n).fill(NaN);
    // Stochastic %K
    for (let i = 0; i < n; i++) {
      if (i + 1 >= kdPeriod) {
        let lowMin = Infinity;
        let highMax = -Infinity;
        for (let j = i - kdPeriod + 1; j <= i; j++) {
          const row = data[j];
          if (row.low < lowMin) lowMin = row.low;
          if (row.high > highMax) highMax = row.high;
        }
        const denom = highMax - lowMin;
        K[i] = denom !== 0 ? ((data[i].close - lowMin) / denom) * 100 : 0;
      } else {
        K[i] = 0;
      }
    }
    // %D: simple moving average of K
    for (let i = 0; i < n; i++) {
      if (i + 1 >= kdSmooth) {
        let sum = 0;
        for (let j = i - kdSmooth + 1; j <= i; j++) {
          sum += K[j];
        }
        D[i] = sum / kdSmooth;
      } else {
        D[i] = 0;
      }
    }
    // RSI: exponential smoothing
    let avgGain = 0;
    let avgLoss = 0;
    for (let i = 1; i < n; i++) {
      const diff = data[i].close - data[i - 1].close;
      const gain = diff > 0 ? diff : 0;
      const loss = diff < 0 ? -diff : 0;
      if (i === 1) {
        avgGain = gain;
        avgLoss = loss;
      } else {
        avgGain = (avgGain * (rsiPeriod - 1) + gain) / rsiPeriod;
        avgLoss = (avgLoss * (rsiPeriod - 1) + loss) / rsiPeriod;
      }
      if (avgLoss === 0) {
        RSI[i] = 100;
      } else {
        const rs = avgGain / avgLoss;
        RSI[i] = 100 - 100 / (1 + rs);
      }
    }
    // MACD: EMA fast/slow and signal
    const emaFast = new Array(n).fill(0);
    const emaSlow = new Array(n).fill(0);
    const signal = new Array(n).fill(0);
    const kFast = 2 / (macdFast + 1);
    const kSlow = 2 / (macdSlow + 1);
    const kSignal = 2 / (macdSignal + 1);
    for (let i = 0; i < n; i++) {
      const close = data[i].close;
      if (i === 0) {
        emaFast[i] = close;
        emaSlow[i] = close;
        signal[i] = 0;
      } else {
        emaFast[i] = emaFast[i - 1] + kFast * (close - emaFast[i - 1]);
        emaSlow[i] = emaSlow[i - 1] + kSlow * (close - emaSlow[i - 1]);
        const macdLine = emaFast[i] - emaSlow[i];
        signal[i] = signal[i - 1] + kSignal * (macdLine - signal[i - 1]);
        MACD[i] = macdLine - signal[i];
      }
    }
    // Simple moving average (MA)
    let maSum = 0;
    for (let i = 0; i < n; i++) {
      maSum += data[i].close;
      if (i >= maPeriod) {
        maSum -= data[i - maPeriod].close;
      }
      if (i + 1 >= maPeriod) {
        MA[i] = maSum / maPeriod;
      } else {
        MA[i] = NaN;
      }
    }
    // Return a new array of objects including computed indicators
    return data.map((row, idx) => {
      return {
        ...row,
        K: K[idx],
        D: D[idx],
        RSI: RSI[idx],
        MACD: MACD[idx] || 0,
        MA: MA[idx],
      };
    });
  }

  // Backtesting algorithm replicating the Python logic
  function backtest(data, params) {
    let cash = params.initial_capital;
    let unitsHeld = 0;
    let avgCost = 0;
    let currentTradeStart = null;
    let currentTradeCost = 0;
    const trades = [];
    for (let i = 0; i < data.length; i++) {
      const row = data[i];
      const kVal = row.K;
      const rsiVal = row.RSI;
      const macdVal = row.MACD;
      const maVal = row.MA;
      const price = row.close;
      const date = row.date;
      // Buy signal
      let buySignal = (kVal >= params.buy_kd_upper) && (rsiVal >= params.buy_rsi_upper) && (macdVal > 0);
      if (params.enable_ma_filter) {
        buySignal = buySignal && !isNaN(maVal) && price > maVal;
      }
      if (buySignal && unitsHeld < params.max_units && params.buy_units_each_time > 0) {
        const unitsToBuy = Math.min(params.buy_units_each_time, params.max_units - unitsHeld);
        const costPerUnit = price * 1000;
        const totalCost = costPerUnit * unitsToBuy;
        if (cash >= totalCost) {
          cash -= totalCost;
          const newTotalUnits = unitsHeld + unitsToBuy;
          if (unitsHeld === 0) {
            avgCost = price;
            currentTradeStart = date;
            currentTradeCost = totalCost;
          } else {
            avgCost = (avgCost * unitsHeld + price * unitsToBuy) / newTotalUnits;
            currentTradeCost += totalCost;
          }
          unitsHeld = newTotalUnits;
        }
      }
      // Sell logic
      let sellReasonProfit = false;
      if (unitsHeld > 0) {
        // Condition (1): price above average cost by 3%
        if (params.sell_price_above_avg && price >= avgCost * 1.03) {
          sellReasonProfit = true;
        }
        // Condition (2): indicators
        const sellSignal = (kVal <= params.sell_kd_lower) && (rsiVal <= params.sell_rsi_lower) && (macdVal < 0);
        if (sellReasonProfit || sellSignal) {
          const potentialProfit = (price - avgCost) * unitsHeld * 1000;
          if (potentialProfit >= 0 || params.allow_loss_sell || sellReasonProfit) {
            const revenue = price * unitsHeld * 1000;
            cash += revenue;
            const tradeProfit = revenue - currentTradeCost;
            trades.push({
              buy_date: currentTradeStart,
              sell_date: date,
              buy_price: avgCost,
              sell_price: price,
              units: unitsHeld,
              cost: currentTradeCost,
              revenue: revenue,
              profit: tradeProfit,
            });
            unitsHeld = 0;
            avgCost = 0;
            currentTradeStart = null;
            currentTradeCost = 0;
          }
        }
      }
    }
    // Close any open position at the last bar
    if (unitsHeld > 0 && currentTradeStart !== null) {
      const last = data[data.length - 1];
      const price = last.close;
      const revenue = price * unitsHeld * 1000;
      cash += revenue;
      const tradeProfit = revenue - currentTradeCost;
      trades.push({
        buy_date: currentTradeStart,
        sell_date: last.date,
        buy_price: avgCost,
        sell_price: price,
        units: unitsHeld,
        cost: currentTradeCost,
        revenue: revenue,
        profit: tradeProfit,
      });
    }
    // Compute win rate and total profit
    let winCount = 0;
    let totalProfit = 0;
    trades.forEach((t) => {
      if (t.profit > 0) winCount += 1;
      totalProfit += t.profit;
    });
    const winRate = trades.length > 0 ? winCount / trades.length : 0;
    return { winRate, totalProfit, trades };
  }

  // Render the result into the resultDiv
  function renderResult(winRate, totalProfit, trades) {
    let html = '';
    html += `<p><strong>Win rate:</strong> ${(winRate * 100).toFixed(2)}%</p>`;
    html += `<p><strong>Total profit:</strong> ${totalProfit.toFixed(2)}</p>`;
    if (trades.length > 0) {
      html += '<h3>Trades</h3>';
      html += '<table><thead><tr>';
      const headers = ['Buy Date','Sell Date','Buy Price','Sell Price','Units','Cost','Revenue','Profit'];
      headers.forEach((h) => { html += `<th>${h}</th>`; });
      html += '</tr></thead><tbody>';
      trades.forEach((tr) => {
        html += '<tr>';
        html += `<td>${tr.buy_date.toISOString().slice(0,10)}</td>`;
        html += `<td>${tr.sell_date.toISOString().slice(0,10)}</td>`;
        html += `<td>${tr.buy_price.toFixed(2)}</td>`;
        html += `<td>${tr.sell_price.toFixed(2)}</td>`;
        html += `<td>${tr.units}</td>`;
        html += `<td>${tr.cost.toFixed(2)}</td>`;
        html += `<td>${tr.revenue.toFixed(2)}</td>`;
        html += `<td>${tr.profit.toFixed(2)}</td>`;
        html += '</tr>';
      });
      html += '</tbody></table>';
    } else {
      html += '<p>No trades executed with the given parameters.</p>';
    }
    resultDiv.innerHTML = html;
  }
})();
