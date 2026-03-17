## Dataset

- **93,290 price observations** across **19 events** (11 closed, 8 active), **406 brackets**
- Date range: Feb 13 – Mar 13, 2026 (about 1 month of data)
- Total market volume: **$92.9M**
- Limitation: The CLOB API only retains recent price history, so the June 2024+ historical data you mentioned is unavailable — older events return empty histories

## Top Predictive Variables

The strongest correlations with max future return (for cheap brackets under 5%):


| Feature              | Correlation | Interpretation                                               |
| -------------------- | ----------- | ------------------------------------------------------------ |
| `event_volume`       | +0.18       | Higher-volume events produce bigger swings                   |
| `pct_elapsed`        | -0.14       | **Earlier in the event = more upside potential**             |
| `price_max_lifetime` | +0.13       | Brackets that already spiked once tend to spike again        |
| `hours_elapsed`      | -0.11       | Same as pct_elapsed — buy early                              |
| `price`              | -0.08       | Cheaper brackets have more room to multiply                  |
| `price_vol_24h`      | +0.06       | Volatile brackets offer more opportunity                     |
| `entropy`            | +0.05       | When outcome is uncertain (spread-out odds), more volatility |


These are weak individual correlations (all under 0.2), which is typical for financial markets — no single variable is a magic bullet.

## ML Model Performance

**Buy Model** (will this cheap bracket hit 2x+ return?):

- Accuracy: **53%** — barely above coin-flip (50%)
- The model struggles because cheap bracket outcomes are inherently noisy
- Return regression MAE: 5.39x — meaning predicted returns are off by ~5x on average

**Sell Model** (should I sell, price won't rise >10% in 6h?):

- Accuracy: **58%** — slightly better but still weak
- It tends to be conservative (leans toward "sell")

**Verdict: The ML models are not reliable enough to use as standalone trading signals.** 53% accuracy on buy decisions with high variance is essentially noise. The models capture some signal but not enough to trust blindly.

## Your Manual Strategy — Backtest Results

Your strategy (buy <2%, 2-3 brackets away, sell at 4-5x or when time runs out) was backtested on 11 closed events:


| Metric                | Value              |
| --------------------- | ------------------ |
| Total trades          | 78                 |
| Win rate              | 28.2%              |
| Avg return (winners)  | **4.63x**          |
| Avg return (losers)   | 0.15x              |
| Net P&L (if $1/trade) | **+$32.30 (+41%)** |
| Best trade            | 5.6x               |


By sell reason:

- **Target hit (4x+)**: 22 trades, avg **4.63x** — these are your profit engine
- **Expired (market ended)**: 37 trades, avg **0.11x** — total losses
- **Timeout**: 19 trades, avg **0.24x** — mostly losses

**Your manual strategy is profitable** — the 22 winning trades at 4.6x average more than compensate for the 56 losing trades. The key insight is that you only need ~28% of trades to win big to be profitable overall.

## Can You Use This for Real Betting?

**Your manual strategy: Yes, keep using it.** The backtest confirms your intuition works — buying cheap brackets far from the leader and selling on spikes is a positive-expectation strategy with +41% return over this period.

**The ML models alone: No.** At 53% buy accuracy, they don't add enough edge. However, they're useful as **filters to combine with your manual judgment**:

- The model's buy probability score can help **rank** which of several cheap brackets to prioritize
- The sell model's probability can act as a **second opinion** when you're unsure whether to hold or exit

**Key improvements to consider:**

1. **More data** — only 1 month of CLOB price history is available; if Polymarket improves data retention, retraining on 6+ months would significantly help
2. **Integrate xtracker pace data** — the current tweet pace during an event is probably the strongest signal for bracket selection, and it's not yet feeding into the Polymarket models
3. **Use the monitor** (`python monitor.py`) to automate the tedious part: scanning all active brackets every 5 minutes and alerting you when something matches your criteria

