"""Unified trade model for Polymarket bracket trading (v3).

Key design: buy and sell are evaluated TOGETHER as a complete trade cycle.
- The model predicts expected trade return at entry time
- Evaluation uses weighted continuous metrics, not binary accuracy
- Backtest simulates realistic full trades with ML-guided exits
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error

from pm_features import RETURN_CAP, build_hold_states

TRADE_FEATURES = [
    "price",
    "hours_remaining",
    "pct_elapsed",
    "price_change_1h",
    "price_change_3h",
    "price_change_6h",
    "price_change_12h",
    "price_change_24h",
    "price_vol_6h",
    "price_vol_12h",
    "price_vol_24h",
    "price_pct_range",
    "distance_from_leading",
    "bracket_rank",
    "hhi",
    "entropy",
    "bracket_count",
    "brackets_away",
    "in_sweet_spot",
    "volume_rank",
    "leading_bracket_price",
    "recency_weight",
]

# Exit model adds hold-period features (computed during the trade)
EXIT_HOLD_FEATURES = [
    "hold_hours",
    "current_return",
    "max_return_since_buy",
    "drawdown_from_peak",
    "days_held",
]

MAX_TRAIN = 50000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe_cols(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def _prepare(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    cols = _dedupe_cols(feature_cols)
    valid = df[cols + [target_col]].dropna()
    return valid[cols].copy(), valid[target_col].copy(), valid.index


def _time_split(df: pd.DataFrame, test_frac: float = 0.2):
    events_sorted = df.sort_values("event_start")["event_slug"].drop_duplicates()
    n = len(events_sorted)
    split = int(n * (1 - test_frac))
    train_events = set(events_sorted.iloc[:split])
    test_events = set(events_sorted.iloc[split:])
    return df[df["event_slug"].isin(train_events)], df[df["event_slug"].isin(test_events)]


def _subsample(X, y, max_n=MAX_TRAIN):
    if len(X) > max_n:
        idx = X.sample(max_n, random_state=42).index
        return X.loc[idx], y.loc[idx]
    return X, y


# ---------------------------------------------------------------------------
# Weighted evaluation metrics
# ---------------------------------------------------------------------------

def weighted_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate predictions with return-weighted metrics.

    Instead of binary "did it hit 2x?", we measure:
    - How well the model RANKS opportunities (do higher predictions = higher returns?)
    - Weighted accuracy where higher-return trades count more
    - Profit if you followed the model's top-N recommendations
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    # 1. Rank correlation (Spearman): does the model rank better opportunities higher?
    from scipy.stats import spearmanr
    rank_corr, rank_p = spearmanr(y_pred, y_true)

    # 2. Weighted MAE: errors on high-return trades penalized more
    weights = np.clip(y_true, 0.1, RETURN_CAP)
    wmae = np.average(np.abs(y_true - y_pred), weights=weights)

    # 3. Top-N profit: if you buy the top 20% by prediction, what's your avg return?
    top_n = max(1, n // 5)
    top_idx = np.argsort(y_pred)[-top_n:]
    top_avg_return = y_true[top_idx].mean()
    random_avg_return = y_true.mean()
    top_lift = top_avg_return / random_avg_return if random_avg_return > 0 else 0

    # 4. Profitable trade rate in top quintile
    top_win_rate = (y_true[top_idx] > 1.0).mean()

    # 5. Calibration at thresholds: for each prediction bucket, avg actual return
    buckets = {}
    for lo, hi, label in [(0, 1, "<1x"), (1, 2, "1-2x"), (2, 4, "2-4x"), (4, RETURN_CAP, "4x+")]:
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() > 0:
            buckets[label] = {"pred_avg": y_pred[mask].mean(), "actual_avg": y_true[mask].mean(), "n": int(mask.sum())}

    return {
        "rank_correlation": round(rank_corr, 4),
        "rank_p_value": round(rank_p, 6),
        "weighted_mae": round(wmae, 3),
        "plain_mae": round(np.mean(np.abs(y_true - y_pred)), 3),
        "top20_avg_return": round(top_avg_return, 3),
        "random_avg_return": round(random_avg_return, 3),
        "top20_lift": round(top_lift, 2),
        "top20_win_rate": round(top_win_rate, 3),
        "calibration": buckets,
    }


# ---------------------------------------------------------------------------
# Unified trade model
# ---------------------------------------------------------------------------

def train_trade_model(
    df: pd.DataFrame,
    max_price: float = 0.05,
    test_frac: float = 0.2,
    verbose: bool = True,
) -> dict:
    """Train a unified model that predicts full trade return at entry time.

    The target is max_return_all (capped) — the best possible exit price
    divided by current price. This combines buy quality and sell potential
    into one continuous metric.
    """
    pool = df[(df["price"] < max_price) & (df["price"] > 0.001) & (df["event_closed"] == True)].copy()
    if "max_return_all" not in pool.columns:
        raise ValueError("Run add_return_labels() first")
    pool = pool.dropna(subset=["max_return_all"])

    train_df, test_df = _time_split(pool, test_frac)
    features = _dedupe_cols(TRADE_FEATURES)

    # Regressor: predict continuous return (the core model)
    X_train, y_train, _ = _prepare(train_df, features, "max_return_all")
    X_test, y_test, test_idx = _prepare(test_df, features, "max_return_all")
    X_train_s, y_train_s = _subsample(X_train, y_train)

    if verbose:
        print(f"=== Unified Trade Model ===")
        print(f"  Train: {len(X_train_s)} obs from {train_df['event_slug'].nunique()} events")
        print(f"  Test:  {len(X_test)} obs from {test_df['event_slug'].nunique()} events")
        print(f"  Avg return (train): {y_train_s.mean():.2f}x, (test): {y_test.mean():.2f}x")

    reg = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.08, random_state=42)
    reg.fit(X_train_s, y_train_s)
    preds = reg.predict(X_test)

    scores = weighted_score(y_test.values, preds)

    if verbose:
        print(f"\n  --- Weighted Evaluation ---")
        print(f"  Rank correlation (Spearman): {scores['rank_correlation']:.4f} (p={scores['rank_p_value']:.6f})")
        print(f"  Weighted MAE: {scores['weighted_mae']:.3f}")
        print(f"  Plain MAE:    {scores['plain_mae']:.3f}")
        print(f"  Top 20% avg return:  {scores['top20_avg_return']:.2f}x  (random: {scores['random_avg_return']:.2f}x)")
        print(f"  Top 20% lift:        {scores['top20_lift']:.2f}x")
        print(f"  Top 20% win rate:    {scores['top20_win_rate']:.1%}")
        print(f"\n  Calibration (does prediction bucket match actual return?):")
        for label, info in scores["calibration"].items():
            print(f"    {label:6s}: predicted {info['pred_avg']:.2f}x → actual {info['actual_avg']:.2f}x  (n={info['n']})")

    # Exit model: train on hold states (includes hold-period features)
    # Target: max_return_6h — how much upside is left in the next 6 hours
    hold_states = build_hold_states(df, buy_threshold=max_price, min_price=0.001)
    exit_features = _dedupe_cols(features + EXIT_HOLD_FEATURES)

    use_hold_features = (
        not hold_states.empty
        and "max_return_6h" in hold_states.columns
        and all(c in hold_states.columns for c in EXIT_HOLD_FEATURES)
    )
    if use_hold_features:
        pool_sell = hold_states[(hold_states["price"] > 0.005)].copy()
        pool_sell = pool_sell.dropna(subset=["max_return_6h"] + exit_features)
    else:
        pool_sell = pd.DataFrame()

    if not pool_sell.empty:
        train_sell, test_sell = _time_split(pool_sell, test_frac)
        X_tr_s, y_tr_s, _ = _prepare(train_sell, exit_features, "max_return_6h")
        X_te_s, y_te_s, _ = _prepare(test_sell, exit_features, "max_return_6h")
        X_tr_s, y_tr_s = _subsample(X_tr_s, y_tr_s)

        sell_reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        sell_reg.fit(X_tr_s, y_tr_s)
        sell_mae = mean_absolute_error(y_te_s, sell_reg.predict(X_te_s))
        if verbose:
            print(f"\n  --- Exit Model (with hold-period features) ---")
            print(f"  Train: {len(X_tr_s)} hold states, MAE: {sell_mae:.3f}x")
    else:
        # Fallback: train on raw rows without hold features
        pool_sell = df[(df["price"] > 0.005) & (df["event_closed"] == True)].copy()
        pool_sell = pool_sell.dropna(subset=["max_return_6h"])
        train_sell, test_sell = _time_split(pool_sell, test_frac)
        X_tr_s, y_tr_s, _ = _prepare(train_sell, features, "max_return_6h")
        X_te_s, y_te_s, _ = _prepare(test_sell, features, "max_return_6h")
        X_tr_s, y_tr_s = _subsample(X_tr_s, y_tr_s)

        sell_reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        sell_reg.fit(X_tr_s, y_tr_s)
        exit_features = features
        if verbose:
            print(f"\n  --- Exit Model (fallback, no hold features) ---")

    return {
        "entry_model": reg,
        "exit_model": sell_reg,
        "scores": scores,
        "feature_names": features,
        "exit_feature_names": exit_features,
        "X_test": X_test,
        "y_test": y_test,
        "test_preds": preds,
        "test_df": test_df.loc[test_idx],
    }


# ---------------------------------------------------------------------------
# Backtest v3: unified model, ML exit, detailed labels
# ---------------------------------------------------------------------------

def backtest_strategy(
    df: pd.DataFrame,
    trade_model: dict | None = None,
    strategy: str = "ml",
    buy_threshold: float = 0.02,
    brackets_away_min: float = 1.5,
    brackets_away_max: float | None = None,
    max_hold_hours: float = 144,
    sell_target_x: float = 4.0,
    sell_1day_left_hours: float | None = None,
    recent_days: int = 90,
    verbose: bool = True,
) -> pd.DataFrame:
    """Backtest strategies.

    strategy:
      - "ml": ML entry score + ML exit (requires trade_model)
      - "fixed_4x": previous strategy — sell at 4x or timeout 144h
      - "manual": your strategy — buy 2-3 brackets away, sell at 4.5x OR 1 day left
      - "hybrid": ML entry score + your manual sell rules (4.5x or 1 day left)
    """
    closed = df[df["event_closed"] == True].copy()
    if closed.empty:
        return pd.DataFrame()

    cutoff = closed["event_start"].max() - pd.Timedelta(days=recent_days)
    closed = closed[closed["event_start"] >= cutoff]

    if verbose:
        n_events = closed["event_slug"].nunique()
        print(f"Backtest window: {cutoff.date()} to {closed['event_end'].max().date()} ({n_events} events)")

    # Buy candidates
    mask = (
        (closed["price"] < buy_threshold) &
        (closed["price"] > 0.001) &
        (closed["brackets_away"] >= brackets_away_min) &
        (closed["hours_remaining"] > 24)
    )
    if brackets_away_max is not None:
        mask = mask & (closed["brackets_away"] <= brackets_away_max)
    candidates = closed[mask].copy()

    if candidates.empty:
        return pd.DataFrame()

    use_ml = strategy in ("ml", "hybrid") and trade_model is not None
    features = _dedupe_cols(trade_model["feature_names"]) if trade_model else None
    exit_features = _dedupe_cols(trade_model.get("exit_feature_names", features or [])) if trade_model else None
    entry_model = trade_model["entry_model"] if trade_model else None
    exit_model = trade_model["exit_model"] if trade_model else None

    # Score candidates
    if use_ml and entry_model and features:
        feat_vals = candidates[features].dropna()
        if not feat_vals.empty:
            candidates.loc[feat_vals.index, "entry_score"] = entry_model.predict(feat_vals)
        else:
            candidates["entry_score"] = 1.0
    else:
        candidates["entry_score"] = 1.0

    # First buy per (event, bracket)
    if use_ml:
        candidates = candidates.sort_values(["event_slug", "condition_id", "entry_score"], ascending=[True, True, False])
    else:
        candidates = candidates.sort_values(["event_slug", "condition_id", "timestamp"])
    candidates = candidates.drop_duplicates(subset=["event_slug", "condition_id"], keep="first")

    trades: list[dict] = []
    for _, row in candidates.iterrows():
        buy_price = row["price"]
        cond_id = row["condition_id"]
        buy_time = row["timestamp"]
        entry_score = row.get("entry_score", 1.0)

        future = closed[
            (closed["condition_id"] == cond_id) & (closed["timestamp"] > buy_time)
        ].sort_values("timestamp")

        if future.empty:
            continue

        sell_price = future.iloc[-1]["price"]
        sell_reason = "expired"
        sell_time = future.iloc[-1]["timestamp"]
        sell_hours_remaining = future.iloc[-1].get("hours_remaining", 0)
        hold_hours = (sell_time - buy_time).total_seconds() / 3600

        target_price = buy_price * sell_target_x
        one_day_hours = sell_1day_left_hours if sell_1day_left_hours is not None else 24
        max_ret_so_far = 0.0

        # Walk forward
        for _, frow in future.iterrows():
            h = (frow["timestamp"] - buy_time).total_seconds() / 3600
            current_ret = frow["price"] / buy_price if buy_price > 0 else 0
            max_ret_so_far = max(max_ret_so_far, current_ret)
            drawdown = 1 - current_ret / max_ret_so_far if max_ret_so_far > 0 else 0
            hrs_left = frow.get("hours_remaining", 999)

            # --- Fixed 4x / Manual / Hybrid: target hit ---
            if strategy in ("fixed_4x", "manual", "hybrid") and frow["price"] >= target_price:
                sell_price = frow["price"]
                sell_reason = "target_hit"
                sell_time = frow["timestamp"]
                sell_hours_remaining = hrs_left
                hold_hours = h
                break

            # --- Manual / Hybrid: 1 day left ---
            if strategy in ("manual", "hybrid") and hrs_left < one_day_hours:
                sell_price = frow["price"]
                sell_reason = "1day_left"
                sell_time = frow["timestamp"]
                sell_hours_remaining = hrs_left
                hold_hours = h
                break

            # --- Timeout ---
            if h >= max_hold_hours:
                sell_price = frow["price"]
                sell_reason = "timeout"
                sell_time = frow["timestamp"]
                sell_hours_remaining = hrs_left
                hold_hours = h
                break

            # --- Event about to end ---
            if hrs_left < 6:
                sell_price = frow["price"]
                sell_reason = "expiry_close"
                sell_time = frow["timestamp"]
                sell_hours_remaining = hrs_left
                hold_hours = h
                break

            # --- ML exit (only for strategy="ml", hybrid uses manual sell) ---
            if strategy == "ml" and exit_model and exit_features:
                # Build feature row: base + hold-period features
                feat_vals = frow.reindex(exit_features)
                if EXIT_HOLD_FEATURES[0] in exit_features:
                    feat_vals["hold_hours"] = h
                    feat_vals["current_return"] = current_ret
                    feat_vals["max_return_since_buy"] = max_ret_so_far
                    feat_vals["drawdown_from_peak"] = min(drawdown, 1.0)
                    feat_vals["days_held"] = h / 24
                if feat_vals.notna().all():
                    pred_6h_return = exit_model.predict(feat_vals.values.reshape(1, -1))[0]
                    if pred_6h_return < 0.95:
                        sell_price = frow["price"]
                        sell_reason = "ml_exit_drop" if current_ret <= 1.0 else "ml_exit_profit"
                        sell_time = frow["timestamp"]
                        sell_hours_remaining = hrs_left
                        hold_hours = h
                        break
                    if current_ret >= 2.0 and pred_6h_return < 1.05:
                        sell_price = frow["price"]
                        sell_reason = "ml_take_profit"
                        sell_time = frow["timestamp"]
                        sell_hours_remaining = hrs_left
                        hold_hours = h
                        break

            sell_price = frow["price"]
            sell_time = frow["timestamp"]
            sell_hours_remaining = hrs_left
            hold_hours = h

        ret = min(sell_price / buy_price, RETURN_CAP) if buy_price > 0 else 0

        trades.append({
            "event_slug": row["event_slug"],
            "bracket_label": row["bracket_label"],
            "bracket_low": row.get("bracket_low"),
            "bracket_high": row.get("bracket_high"),
            "buy_time": buy_time,
            "buy_price": round(buy_price, 4),
            "buy_hours_remaining": round(row.get("hours_remaining", 0), 1),
            "buy_pct_elapsed": round(row.get("pct_elapsed", 0), 3),
            "buy_brackets_away": round(row.get("brackets_away", 0), 1),
            "entry_score": round(entry_score, 2) if pd.notna(entry_score) else None,
            "sell_time": sell_time,
            "sell_price": round(sell_price, 4),
            "sell_hours_remaining": round(sell_hours_remaining, 1),
            "sell_reason": sell_reason,
            "hold_hours": round(hold_hours, 1),
            "return": round(ret, 2),
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    if verbose:
        _print_backtest_report(trades_df, recent_days)

    return trades_df


def run_strategy_comparison(
    df: pd.DataFrame,
    trade_model: dict | None = None,
    recent_days: int = 90,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run backtests for ML, fixed_4x, manual, and hybrid strategies. Returns {name: trades_df}."""
    results = {}

    # 1. Unified ML (entry + ML exit)
    if trade_model is not None:
        t = backtest_strategy(df, trade_model=trade_model, strategy="ml", recent_days=recent_days, verbose=verbose)
        results["ml"] = t

    # 2. Previous: fixed 4x target, timeout 144h
    t = backtest_strategy(df, trade_model=None, strategy="fixed_4x", sell_target_x=4.0, recent_days=recent_days, verbose=verbose)
    results["fixed_4x"] = t

    # 3. Your manual: buy <2%, 2-3 brackets away, sell at 4.5x OR 1 day left
    t = backtest_strategy(
        df, trade_model=None, strategy="manual",
        brackets_away_min=2.0, brackets_away_max=3.5,
        sell_target_x=4.5, sell_1day_left_hours=24,
        recent_days=recent_days, verbose=verbose,
    )
    results["manual"] = t

    # 4. Hybrid: ML entry scoring + your manual sell rules (4.5x or 1 day left)
    if trade_model is not None:
        t = backtest_strategy(
            df, trade_model=trade_model, strategy="hybrid",
            sell_target_x=4.5, sell_1day_left_hours=24,
            recent_days=recent_days, verbose=verbose,
        )
        results["hybrid"] = t

        # 5. Hybrid restricted: same but only 2-3 brackets (ML picks best within sweet spot)
        t = backtest_strategy(
            df, trade_model=trade_model, strategy="hybrid",
            brackets_away_min=2.0, brackets_away_max=3.5,
            sell_target_x=4.5, sell_1day_left_hours=24,
            recent_days=recent_days, verbose=verbose,
        )
        results["hybrid_23"] = t

    return results


def _print_backtest_report(trades_df: pd.DataFrame, window_days: int):
    total = len(trades_df)
    returned = trades_df["return"].sum()
    winners = trades_df[trades_df["return"] > 1]

    print(f"=== Backtest Results ({window_days}-day window) ===")
    print(f"  Trades: {total}")
    print(f"  Win rate (>1x): {len(winners)/total:.1%}")
    print(f"  Avg return: {trades_df['return'].mean():.2f}x  (median: {trades_df['return'].median():.2f}x)")
    print(f"  Best: {trades_df['return'].max():.1f}x, Worst: {trades_df['return'].min():.2f}x")
    print(f"  P&L ($1/trade): ${returned - total:.1f} on ${total} ({(returned/total-1)*100:+.0f}%)")

    # Weighted P&L: weight trades by entry_score
    if "entry_score" in trades_df.columns and trades_df["entry_score"].notna().any():
        scored = trades_df.dropna(subset=["entry_score"])
        top_half = scored.nlargest(len(scored) // 2, "entry_score")
        bottom_half = scored.nsmallest(len(scored) // 2, "entry_score")
        print(f"\n  --- Entry Score Effectiveness ---")
        print(f"  Top-scored half:    avg {top_half['return'].mean():.2f}x, win rate {(top_half['return']>1).mean():.1%}")
        print(f"  Bottom-scored half: avg {bottom_half['return'].mean():.2f}x, win rate {(bottom_half['return']>1).mean():.1%}")

    print(f"\n  By sell reason:")
    by_reason = trades_df.groupby("sell_reason").agg(
        count=("return", "count"),
        avg_ret=("return", "mean"),
        win_rate=("return", lambda x: (x > 1).mean()),
    ).round(3)
    print(f"    {by_reason.to_string()}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def feature_importance_chart(info: dict, title: str = "Feature Importance") -> go.Figure:
    model = info.get("entry_model") or info.get("classifier")
    names = info["feature_names"]
    imp = model.feature_importances_
    order = np.argsort(imp)
    fig = go.Figure(go.Bar(
        x=imp[order], y=[names[i] for i in order],
        orientation="h", marker_color="#636EFA",
    ))
    fig.update_layout(
        title=title, xaxis_title="Importance",
        template="plotly_white", height=max(400, len(names) * 25), margin=dict(l=180),
    )
    return fig


def prediction_vs_actual_chart(info: dict) -> go.Figure:
    y_test = info["y_test"].values
    preds = info["test_preds"]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=preds, y=y_test, mode="markers",
        marker=dict(size=3, opacity=0.3, color="#636EFA"),
        name="Observations",
    ))
    max_val = min(max(y_test.max(), preds.max()), RETURN_CAP)
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color="red"), name="Perfect",
    ))
    fig.update_layout(
        title="Predicted vs Actual Return",
        xaxis_title="Predicted Return (x)", yaxis_title="Actual Return (x)",
        template="plotly_white", height=500,
    )
    return fig


def backtest_returns_chart(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trades_df["return"].clip(upper=RETURN_CAP), nbinsx=40,
        marker_color="#636EFA", name="Trade returns",
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="breakeven")
    fig.update_layout(
        title="Backtest: Distribution of Trade Returns",
        xaxis_title="Return (x)", yaxis_title="Count",
        template="plotly_white", height=400,
    )
    return fig
