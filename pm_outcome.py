"""Trade outcome model (v2): train on ACTUAL backtest results, strict temporal split.

Optimized for your strategy: low win rate (20-30%) but jackpots at 8-10x.
- Target: JACKPOT POTENTIAL = max return during hold (not sell price)
- Weighted training: high-return trades count more (find the 8-10x winners)
- Filter: only take trades with predicted jackpot potential > 6x or 8x
- Metric: return per trade (scale by bet size), not trade count
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

from pm_features import RETURN_CAP
from pm_model import backtest_strategy, TRADE_FEATURES, _dedupe_cols

# Your strategy: low win rate, jackpots at 8-10x
JACKPOT_THRESHOLD = 8.0  # 8x+ = jackpot

# Train on events before this date; test only on events on/after
DEFAULT_TEST_CUTOFF_DAYS = 120  # test on last 120 days of data


def _get_train_test_events(df: pd.DataFrame, test_days: int = DEFAULT_TEST_CUTOFF_DAYS):
    """Split events by time: train = older, test = most recent test_days."""
    closed = df[df["event_closed"] == True]
    if closed.empty:
        return set(), set()

    events = closed.sort_values("event_start")["event_slug"].unique()
    cutoff = closed["event_start"].max() - pd.Timedelta(days=test_days)
    train_events = set(e for e in events if closed[closed["event_slug"] == e]["event_start"].iloc[0] < cutoff)
    test_events = set(e for e in events if closed[closed["event_slug"] == e]["event_start"].iloc[0] >= cutoff)
    return train_events, test_events


def generate_trade_outcomes(
    df: pd.DataFrame,
    events: set[str] | None = None,
    strategy: str = "manual",
    brackets_away_min: float = 2.0,
    brackets_away_max: float = 3.5,
    sell_target_x: float = 4.5,
    sell_1day_left_hours: float = 24,
) -> pd.DataFrame:
    """Run backtest and return trades WITH entry features for model training.

    Returns DataFrame: one row per trade, with TRADE_FEATURES + actual_return.
    """
    closed = df[df["event_closed"] == True].copy()
    if events:
        closed = closed[closed["event_slug"].isin(events)]
    if closed.empty:
        return pd.DataFrame()

    features = _dedupe_cols(TRADE_FEATURES)
    mask = (
        (closed["price"] < 0.02) & (closed["price"] > 0.001) &
        (closed["brackets_away"] >= brackets_away_min) &
        (closed["brackets_away"] <= brackets_away_max) &
        (closed["hours_remaining"] > 24)
    )
    candidates = closed[mask].sort_values(["event_slug", "condition_id", "timestamp"])
    candidates = candidates.drop_duplicates(subset=["event_slug", "condition_id"], keep="first")

    trades = []
    for _, row in candidates.iterrows():
        buy_price = row["price"]
        cond_id = row["condition_id"]
        buy_time = row["timestamp"]

        future = closed[
            (closed["condition_id"] == cond_id) & (closed["timestamp"] > buy_time)
        ].sort_values("timestamp")
        if future.empty:
            continue

        target_price = buy_price * sell_target_x
        sell_price = future.iloc[-1]["price"]
        sell_reason = "expired"
        hold_hours = 0
        max_price_during_hold = future["price"].max()

        for _, frow in future.iterrows():
            h = (frow["timestamp"] - buy_time).total_seconds() / 3600
            hrs_left = frow.get("hours_remaining", 999)

            if frow["price"] >= target_price:
                sell_price = frow["price"]
                sell_reason = "target_hit"
                hold_hours = h
                break
            if hrs_left < sell_1day_left_hours:
                sell_price = frow["price"]
                sell_reason = "1day_left"
                hold_hours = h
                break
            if h >= 144:
                sell_price = frow["price"]
                sell_reason = "timeout"
                hold_hours = h
                break
            if hrs_left < 6:
                sell_price = frow["price"]
                sell_reason = "expiry_close"
                hold_hours = h
                break

        ret = min(sell_price / buy_price, RETURN_CAP) if buy_price > 0 else 0
        # Jackpot potential: best return we could have gotten during hold
        jackpot_potential = min(max_price_during_hold / buy_price, RETURN_CAP) if buy_price > 0 else 0

        feat_row = row.reindex(features)
        if feat_row.notna().all():
            out = feat_row.to_dict()
            out["actual_return"] = ret
            out["jackpot_potential"] = jackpot_potential
            out["event_slug"] = row["event_slug"]
            out["condition_id"] = cond_id
            out["sell_reason"] = sell_reason
            trades.append(out)

    return pd.DataFrame(trades)


def train_outcome_model(
    df: pd.DataFrame,
    test_days: int = DEFAULT_TEST_CUTOFF_DAYS,
    min_pred_threshold: float = 6.0,
    jackpot_threshold: float = JACKPOT_THRESHOLD,
    brackets_away_min: float = 2.0,
    brackets_away_max: float = 3.5,
    verbose: bool = True,
) -> dict:
    """Train model to predict JACKPOT POTENTIAL (max return during hold).

    Optimized for: low win rate, but 8-10x jackpots. Weight high-return trades more.
    """
    train_events, test_events = _get_train_test_events(df, test_days)
    if not train_events:
        raise ValueError("No train events")

    if verbose:
        print(f"=== Jackpot Model (target: {jackpot_threshold}x+ potential) ===")
        print(f"  Train events: {len(train_events)}, Test: {len(test_events)} (last {test_days}d)")
        print(f"  Bracket range: {brackets_away_min}-{brackets_away_max}")

    train_trades = generate_trade_outcomes(
        df, events=train_events, strategy="manual",
        brackets_away_min=brackets_away_min, brackets_away_max=brackets_away_max,
    )
    if train_trades.empty or len(train_trades) < 50:
        raise ValueError(f"Too few train trades: {len(train_trades)}")

    features = _dedupe_cols(TRADE_FEATURES)
    X = train_trades[features]
    y_jackpot = train_trades["jackpot_potential"]
    y_actual = train_trades["actual_return"]

    # Regression: predict jackpot potential, WEIGHT by potential (find the 8x+ trades)
    sample_weight = np.clip(y_jackpot, 0.5, 15.0)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.08, random_state=42)
    model.fit(X, y_jackpot, sample_weight=sample_weight)
    train_preds = model.predict(X)

    # Classifier: P(jackpot >= 8x)
    is_jackpot = (y_jackpot >= jackpot_threshold).astype(int)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    clf.fit(X, is_jackpot, sample_weight=sample_weight)
    jackpot_proba = clf.predict_proba(X)[:, 1]

    if verbose:
        n_jackpot = is_jackpot.sum()
        print(f"  Train: {len(train_trades)} trades, {n_jackpot} jackpots ({n_jackpot/len(train_trades):.1%} hit {jackpot_threshold}x+)")
        print(f"  Avg actual return: {y_actual.mean():.2f}x, Avg jackpot potential: {y_jackpot.mean():.2f}x")
        from scipy.stats import spearmanr
        r, _ = spearmanr(train_preds, y_jackpot)
        print(f"  Rank corr (pred vs jackpot potential): {r:.4f}")
        # Top 20% by prediction: do they have higher jackpot rate?
        top_n = max(1, len(train_trades) // 5)
        top_idx = np.argsort(train_preds)[-top_n:]
        top_jackpot_rate = is_jackpot.values[top_idx].mean()
        print(f"  Top 20% by pred: jackpot rate {top_jackpot_rate:.1%} (vs {n_jackpot/len(train_trades):.1%} overall)")

    return {
        "model": model,
        "jackpot_clf": clf,
        "feature_names": features,
        "test_events": test_events,
        "train_events": train_events,
        "min_pred_threshold": min_pred_threshold,
        "jackpot_threshold": jackpot_threshold,
        "train_trades": train_trades,
        "brackets_away_min": brackets_away_min,
        "brackets_away_max": brackets_away_max,
    }


def backtest_outcome_filtered(
    df: pd.DataFrame,
    outcome_info: dict,
    pred_threshold: float | None = None,
    jackpot_proba_threshold: float | None = None,
    require_both: bool = False,
    brackets_away_min: float | None = None,
    brackets_away_max: float | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Backtest on TEST events only. Filter by pred, P(jackpot), or BOTH (high confidence)."""
    test_events = outcome_info["test_events"]
    if not test_events:
        return pd.DataFrame()

    model = outcome_info["model"]
    clf = outcome_info.get("jackpot_clf")
    features = outcome_info["feature_names"]
    threshold = pred_threshold if pred_threshold is not None else outcome_info.get("min_pred_threshold", 6.0)
    b_min = brackets_away_min if brackets_away_min is not None else outcome_info.get("brackets_away_min", 2.0)
    b_max = brackets_away_max if brackets_away_max is not None else outcome_info.get("brackets_away_max", 3.5)

    closed = df[df["event_closed"] == True].copy()
    closed = closed[closed["event_slug"].isin(test_events)]

    mask = (
        (closed["price"] < 0.02) & (closed["price"] > 0.001) &
        (closed["brackets_away"] >= b_min) & (closed["brackets_away"] <= b_max) &
        (closed["hours_remaining"] > 24)
    )
    candidates = closed[mask].sort_values(["event_slug", "condition_id", "timestamp"])
    candidates = candidates.drop_duplicates(subset=["event_slug", "condition_id"], keep="first")

    # Filter: predicted jackpot potential, P(jackpot), or BOTH (high confidence)
    valid = candidates[features].dropna()
    if valid.empty:
        return pd.DataFrame()

    preds = model.predict(valid)
    proba = clf.predict_proba(valid[features])[:, 1] if clf is not None else np.zeros(len(valid))
    candidates = candidates.loc[valid.index].copy()
    candidates["_pred"] = preds
    candidates["_jackpot_proba"] = proba

    if require_both and clf is not None and jackpot_proba_threshold is not None:
        # High confidence: BOTH pred >= threshold AND P(jackpot) >= proba_threshold
        candidates = candidates[
            (candidates["_pred"] >= threshold) &
            (candidates["_jackpot_proba"] >= jackpot_proba_threshold)
        ]
    elif jackpot_proba_threshold is not None and clf is not None:
        candidates = candidates[candidates["_jackpot_proba"] >= jackpot_proba_threshold]
    else:
        candidates = candidates[candidates["_pred"] >= threshold]

    # Run manual sell on filtered candidates
    trades = []
    for _, row in candidates.iterrows():
        buy_price = row["price"]
        cond_id = row["condition_id"]
        buy_time = row["timestamp"]
        pred_ret = row["_pred"]

        future = closed[
            (closed["condition_id"] == cond_id) & (closed["timestamp"] > buy_time)
        ].sort_values("timestamp")
        if future.empty:
            continue

        target_price = buy_price * 4.5
        sell_price = future.iloc[-1]["price"]
        sell_reason = "expired"
        hold_hours = 0

        for _, frow in future.iterrows():
            h = (frow["timestamp"] - buy_time).total_seconds() / 3600
            hrs_left = frow.get("hours_remaining", 999)
            if frow["price"] >= target_price:
                sell_price, sell_reason, hold_hours = frow["price"], "target_hit", h
                break
            if hrs_left < 24:
                sell_price, sell_reason, hold_hours = frow["price"], "1day_left", h
                break
            if h >= 144:
                sell_price, sell_reason, hold_hours = frow["price"], "timeout", h
                break
            if hrs_left < 6:
                sell_price, sell_reason, hold_hours = frow["price"], "expiry_close", h
                break

        ret = min(sell_price / buy_price, RETURN_CAP) if buy_price > 0 else 0
        trades.append({
            "event_slug": row["event_slug"],
            "bracket_label": row["bracket_label"],
            "buy_price": round(buy_price, 4),
            "predicted_return": round(pred_ret, 2),
            "sell_price": round(sell_price, 4),
            "sell_reason": sell_reason,
            "hold_hours": round(hold_hours, 1),
            "return": round(ret, 2),
        })

    trades_df = pd.DataFrame(trades)
    if verbose and not trades_df.empty:
        n = len(trades_df)
        avg_ret = trades_df["return"].mean()
        win_rate = (trades_df["return"] > 1).mean() * 100
        big_wins = (trades_df["return"] >= 4).sum()
        print(f"  Filtered: {n} trades, avg {avg_ret:.2f}x/trade, win rate {win_rate:.0f}%, {big_wins} hit 4x+")
    return trades_df


def run_outcome_comparison(
    df: pd.DataFrame,
    test_days: int = DEFAULT_TEST_CUTOFF_DAYS,
    thresholds: list[float] = (4.0, 6.0, 8.0, 10.0),
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compare manual vs outcome-filtered on TEST events. Includes expanded (wider brackets) variant."""
    # --- Manual baseline (2-3 brackets) ---
    manual_bt = backtest_strategy(
        df, trade_model=None, strategy="manual",
        brackets_away_min=2.0, brackets_away_max=3.5,
        sell_target_x=4.5, sell_1day_left_hours=24,
        recent_days=test_days, verbose=False,
    )
    if verbose and not manual_bt.empty:
        n = len(manual_bt)
        pnl = manual_bt["return"].sum() - n
        print(f"\n=== Manual (2-3 brackets, all trades) ===")
        print(f"  {n} trades, P&L ${pnl:+.1f} ({(manual_bt['return'].sum()/n-1)*100:+.0f}%)")

    # --- Outcome model: train on 2-3 brackets ---
    outcome_info = train_outcome_model(
        df, test_days=test_days,
        brackets_away_min=2.0, brackets_away_max=3.5,
        verbose=verbose,
    )

    results = {"manual": manual_bt}
    for thresh in thresholds:
        filtered = backtest_outcome_filtered(df, outcome_info, pred_threshold=thresh, verbose=verbose)
        results[f"filtered_{thresh}x"] = filtered

    # --- Expanded: train on 1.5-6 brackets, more opportunities ---
    if verbose:
        print(f"\n--- Expanded (1.5-6 brackets, model filters) ---")
    outcome_expanded = train_outcome_model(
        df, test_days=test_days,
        brackets_away_min=1.5, brackets_away_max=6.0,
        verbose=verbose,
    )
    # Manual expanded (all 1.5-6 bracket trades, no filter)
    manual_expanded = backtest_strategy(
        df, trade_model=None, strategy="manual",
        brackets_away_min=1.5, brackets_away_max=6.0,
        sell_target_x=4.5, sell_1day_left_hours=24,
        recent_days=test_days, verbose=False,
    )
    if verbose and not manual_expanded.empty:
        n = len(manual_expanded)
        pnl = manual_expanded["return"].sum() - n
        print(f"  Manual expanded (all): {n} trades, P&L ${pnl:+.1f} ({(manual_expanded['return'].sum()/n-1)*100:+.0f}%)")

    results["manual_expanded"] = manual_expanded
    for thresh in [6.0, 8.0, 10.0]:
        f = backtest_outcome_filtered(
            df, outcome_expanded, pred_threshold=thresh,
            brackets_away_min=1.5, brackets_away_max=6.0,
            verbose=verbose,
        )
        results[f"expanded_filtered_{thresh}x"] = f

    # High confidence: BOTH pred >= 6 AND P(jackpot) >= proba
    if verbose:
        print(f"\n--- High confidence (pred>=6 AND P(jackpot)>=X) ---")
    for proba in [0.2, 0.3, 0.4, 0.5, 0.6]:
        f = backtest_outcome_filtered(
            df, outcome_expanded,
            pred_threshold=6.0, jackpot_proba_threshold=proba, require_both=True,
            brackets_away_min=1.5, brackets_away_max=6.0,
            verbose=verbose,
        )
        results[f"conf_p{int(proba*100)}"] = f

    return results
