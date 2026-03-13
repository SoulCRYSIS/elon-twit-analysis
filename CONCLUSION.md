# Elon Musk Tweet Prediction — Analysis Conclusion

## Dataset

- **3,175 hourly observations** spanning Nov 1, 2025 to Mar 13, 2026 (~4.5 months)
- **6,404 total tweets** tracked, averaging **48.2 tweets/day**
- 35 completed weekly periods, 28 completed 2-day periods from Polymarket

---

## Which Variables Are Useful?

### Hourly-Level Correlations with Tweet Count

| Feature | Pearson r | Spearman r | Interpretation |
|---|---|---|---|
| `consecutive_active_hours` | **+0.457** | **+0.918** | Strongest signal. Once Elon starts tweeting, he tends to continue for a while. |
| `hours_since_last_tweet` | **-0.402** | -0.852 | Longer silence = lower probability of a tweet this hour. |
| `consecutive_zero_hours` | **-0.402** | -0.852 | Same as above, different framing. |
| `lag_1h` | +0.242 | +0.275 | Recent activity (last hour) is moderately predictive. |
| `rolling_mean_2h` | +0.160 | +0.193 | Very recent trend matters. |
| `rolling_mean_48h` | +0.146 | +0.103 | 2-day baseline rate has mild predictive power. |

Calendar features (`hour`, `day_of_week`, `is_weekend`) showed weak correlations at the hourly level (<0.05), meaning Elon's tweeting schedule is irregular — he doesn't follow a strict daily pattern.

### Period-Level Feature Importance (GradientBoosting)

**Weekly model** — top drivers:
1. `period_duration_hours` (0.31) — distinguishing period lengths
2. `pre_weekend_ratio` (0.21) — weekend vs weekday split in pre-period window
3. `period_start_hour` (0.21) — what time the period starts matters
4. `prev_period_total` (0.10) — mean reversion from last period

**2-Day model** — top drivers:
1. `pre_rolling_std_24h_last` (0.46) — volatility in the 24h before the period is the single strongest predictor
2. `pre_max_hourly` (0.15) — peak activity level before the period
3. `pre_total_48h` (0.11) — total tweet volume in the 48h window before
4. `prev_period_total` (0.08) — what the previous 2-day period counted

Key takeaway: **recent volatility and momentum** are the strongest predictors for 2-day windows, while **structural features** (period timing, weekend mix) dominate the weekly model.

---

## Model Performance

### Weekly Period Prediction (7-day windows)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression | 113.4 | 128.2 | 44.8% |
| Random Forest | 72.3 | 79.7 | 29.1% |
| **Gradient Boosting** | **74.0** | **80.7** | **27.1%** |

Best model (GradientBoosting) is off by ~74 tweets on average out of typical weekly totals of 200-360. Test set predictions ranged from very close (25 off) to significantly wrong (128 off).

### 2-Day Period Prediction

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression | 42.4 | 50.0 | 66.1% |
| **Random Forest** | **28.3** | **37.7** | **54.4%** |
| Gradient Boosting | 32.8 | 39.3 | 57.3% |

Best model (RandomForest) is off by ~28 tweets on average. Some predictions were near-perfect (5 off), while others missed badly (63 off).

### Trend-Change Classifier (Hourly)

- **Accuracy: 62.5%** (vs 54% random baseline given class imbalance)
- Predicts whether the next 6 hours will be a "burst" (above trailing 48h average) or "calm"
- Modest edge, driven primarily by `rolling_std_24h`, `rolling_std_12h`, and `rolling_mean_48h`

---

## Can You Use This for Real Betting?

### The honest answer: not yet reliably.

**What works:**
- The weekly model (27% MAPE) gives a rough directional estimate — useful for gut-checking whether "this week feels high or low."
- The feature importance analysis reveals *what to watch*: recent volatility, momentum from the previous period, and whether the pre-period window was weekend-heavy.
- The trend classifier (62.5%) is slightly better than a coin flip at predicting activity bursts, which could help time intra-period bets if the market offers that.

**What doesn't work yet:**
- **MAPE of 27-54% is too high for confident wagering.** A weekly prediction of 280 could easily be 200 or 360 in reality. Polymarket betting brackets are typically tighter than that error margin.
- **Small training set** (28-35 completed periods) means the models haven't seen enough variety — a single unusual week heavily impacts results.
- **No exogenous signals.** Elon's tweet volume is driven by external events (news, controversies, product launches, political events) that this model can't see. A major news event can easily 2x his hourly output.
- **The 2-day model is particularly fragile** (54% MAPE) — with only 48h to predict, the variance is much higher relative to the mean.

### Practical recommendations:

1. **Use the model as a baseline estimate, not a trading signal.** Combine it with your own judgment about current events.
2. **Watch the real-time volatility features** (`rolling_std_24h`, `short_long_ratio`). When pre-period volatility is high, expect wider outcomes — bet conservatively or avoid.
3. **The previous period total is genuinely predictive** (mean reversion). If Elon tweeted way above average last period, expect a pullback. If unusually quiet, expect a bounce.
4. **More data will help.** With 6+ months of history, the models would be significantly more reliable. Continue collecting data.
5. **Consider adding external features** in the future: news sentiment, market hours (US market open/close), known scheduled events (earnings calls, rocket launches, political hearings).

### Bottom line

The model identifies real patterns (momentum, mean reversion, volatility clustering) and the right features to monitor, but **the prediction error is currently too large for high-confidence bets**. Use it as one input among several, not as a standalone oracle.
