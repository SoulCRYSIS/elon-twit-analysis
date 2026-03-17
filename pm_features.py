"""Feature engineering for Polymarket bracket trading signals.

Operates on the bracket_prices DataFrame produced by polymarket_data.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_CAP = 30.0


# ---------------------------------------------------------------------------
# Per-bracket time-series features
# ---------------------------------------------------------------------------

def build_bracket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trading features to each bracket price observation."""
    f = df.copy()

    f = f.sort_values(["event_slug", "condition_id", "timestamp"]).reset_index(drop=True)

    # -- Time context --
    f["hours_remaining"] = (f["event_end"] - f["timestamp"]).dt.total_seconds() / 3600
    f["hours_elapsed"] = (f["timestamp"] - f["event_start"]).dt.total_seconds() / 3600
    event_dur = (f["event_end"] - f["event_start"]).dt.total_seconds() / 3600
    f["pct_elapsed"] = np.clip(f["hours_elapsed"] / event_dur.replace(0, np.nan), 0, 1)

    # -- Temporal era features (suppress 2024 noise) --
    f["event_year"] = f["event_start"].dt.year
    f["event_month"] = f["event_start"].dt.month
    # days since the earliest event in the dataset — captures market maturity
    earliest = f["event_start"].min()
    f["days_since_start"] = (f["event_start"] - earliest).dt.total_seconds() / 86400
    # recency weight: exponential decay, half-life ~120 days
    f["recency_weight"] = np.exp(-0.00578 * (f["days_since_start"].max() - f["days_since_start"]))

    # -- Price momentum --
    grp = f.groupby("condition_id")["price"]
    for lag_h in [1, 3, 6, 12, 24]:
        f[f"price_change_{lag_h}h"] = f["price"] - grp.shift(lag_h)

    # -- Price volatility --
    shifted = grp.shift(1)
    for win in [6, 12, 24]:
        f[f"price_vol_{win}h"] = shifted.groupby(f["condition_id"]).rolling(win, min_periods=2).std().droplevel(0)

    # -- Price relative to its own range --
    f["price_min_lifetime"] = grp.cummin()
    f["price_max_lifetime"] = grp.cummax()
    rng = (f["price_max_lifetime"] - f["price_min_lifetime"]).replace(0, np.nan)
    f["price_pct_range"] = (f["price"] - f["price_min_lifetime"]) / rng

    # -- Event-level context --
    ts_grp = f.groupby(["event_slug", "timestamp"])

    f["leading_bracket_price"] = ts_grp["price"].transform("max")
    f["distance_from_leading"] = f["leading_bracket_price"] - f["price"]
    f["is_leading"] = (f["price"] == f["leading_bracket_price"]).astype(int)
    f["bracket_rank"] = ts_grp["price"].transform("rank", ascending=False, method="min")

    # HHI (vectorized)
    price_sum = ts_grp["price"].transform("sum")
    f["price_share"] = f["price"] / price_sum.replace(0, np.nan)
    sq_share = f["price_share"] ** 2
    f["hhi"] = sq_share.groupby([f["event_slug"], f["timestamp"]]).transform("sum")

    # Shannon entropy (vectorized)
    log_share = np.where(f["price_share"] > 0, f["price_share"] * np.log2(f["price_share"]), 0)
    f["_neg_plogp"] = -log_share
    f["entropy"] = f.groupby(["event_slug", "timestamp"])["_neg_plogp"].transform("sum")
    f.drop(columns=["_neg_plogp"], inplace=True)

    f["bracket_count"] = ts_grp["price"].transform("count")

    # -- Bracket distance from leading bracket --
    f["bracket_mid"] = (f["bracket_low"].fillna(0) + f["bracket_high"].fillna(0)) / 2
    f["_is_max"] = f["price"] == f["leading_bracket_price"]
    leading = f[f["_is_max"]].drop_duplicates(subset=["event_slug", "timestamp"], keep="first")[
        ["event_slug", "timestamp", "bracket_mid"]
    ].rename(columns={"bracket_mid": "leading_mid"})
    f = f.merge(leading, on=["event_slug", "timestamp"], how="left")
    f["brackets_away"] = np.abs(f["bracket_mid"] - f["leading_mid"]) / 25
    # Sweet spot: 2-3 brackets from leading (your manual rule — higher-probability setups)
    f["in_sweet_spot"] = ((f["brackets_away"] >= 2.0) & (f["brackets_away"] <= 3.5)).astype(float)
    f.drop(columns=["_is_max", "price_share", "leading_mid"], inplace=True, errors="ignore")

    # -- Volume rank --
    f["volume_rank"] = f.groupby("event_slug")["market_volume"].transform(
        "rank", ascending=False, method="min"
    )

    return f


# ---------------------------------------------------------------------------
# Buy/sell label construction (retrospective on completed events)
# ---------------------------------------------------------------------------

def add_return_labels(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Add max future return labels, capped at RETURN_CAP to remove outliers."""
    if horizons is None:
        horizons = [6, 12, 24, 48, 9999]

    f = df.sort_values(["condition_id", "timestamp"]).copy()
    safe_price = f["price"].replace(0, np.nan)

    for h in horizons:
        col = "max_return_all" if h == 9999 else f"max_return_{h}h"

        if h == 9999:
            rev_cummax = f.groupby("condition_id")["price"].transform(
                lambda s: s.iloc[::-1].cummax().iloc[::-1]
            )
            f[col] = np.clip(rev_cummax / safe_price, None, RETURN_CAP)
        else:
            def _rolling_future_max(s):
                rev = s.iloc[::-1]
                return rev.rolling(h, min_periods=1).max().iloc[::-1]

            future_max = f.groupby("condition_id")["price"].transform(_rolling_future_max)
            f[col] = np.clip(future_max / safe_price, None, RETURN_CAP)

    return f


def build_hold_states(
    df: pd.DataFrame,
    buy_threshold: float = 0.05,
    min_price: float = 0.001,
) -> pd.DataFrame:
    """Build hold-state rows: for each simulated buy, add hold-period features to future rows.

    Hold-period features (missing from raw rows):
    - hold_hours: how long we've been holding
    - current_return: price / buy_price
    - max_return_since_buy: best return seen during this hold
    - drawdown_from_peak: 1 - current_return/max_return_since_buy (0 at peak)
    - days_held: hold_hours / 24
    """
    closed = df[df["event_closed"] == True].copy()
    if closed.empty:
        return pd.DataFrame()

    closed = closed.sort_values(["event_slug", "condition_id", "timestamp"]).reset_index(drop=True)

    # First valid buy per (event_slug, condition_id)
    buy_mask = (closed["price"] < buy_threshold) & (closed["price"] > min_price)
    first_buys = closed[buy_mask].drop_duplicates(subset=["event_slug", "condition_id"], keep="first")[
        ["event_slug", "condition_id", "timestamp", "price"]
    ].rename(columns={"timestamp": "buy_time", "price": "buy_price"})

    # Merge to get buy context for all future rows
    merged = closed.merge(first_buys, on=["event_slug", "condition_id"], how="inner")
    merged = merged[merged["timestamp"] > merged["buy_time"]].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["hold_hours"] = (merged["timestamp"] - merged["buy_time"]).dt.total_seconds() / 3600
    merged["current_return"] = merged["price"] / merged["buy_price"].replace(0, np.nan)

    # Cumulative max of price from buy onward, per (event, condition_id, buy_time)
    merged = merged.sort_values(["event_slug", "condition_id", "buy_time", "timestamp"])
    merged["_price_cummax"] = merged.groupby(["event_slug", "condition_id", "buy_time"])["price"].cummax()
    merged["max_return_since_buy"] = merged["_price_cummax"] / merged["buy_price"]
    merged["drawdown_from_peak"] = 1 - merged["current_return"] / merged["max_return_since_buy"].replace(0, np.nan)
    merged["drawdown_from_peak"] = np.clip(merged["drawdown_from_peak"], 0, 1)
    merged["days_held"] = merged["hold_hours"] / 24
    merged.drop(columns=["_price_cummax", "buy_time", "buy_price"], inplace=True)

    return merged


def compute_bracket_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of features vs max_return_all for buy candidates (price < 0.05)."""
    buy_pool = df[df["price"] < 0.05].copy()
    if buy_pool.empty or "max_return_all" not in buy_pool.columns:
        return pd.DataFrame()

    target = buy_pool["max_return_all"]
    numeric = buy_pool.select_dtypes(include="number").drop(
        columns=["max_return_all", "max_return_6h", "max_return_12h",
                 "max_return_24h", "max_return_48h"],
        errors="ignore",
    )

    results = []
    for col in numeric.columns:
        valid = numeric[col].notna() & target.notna()
        if valid.sum() < 30:
            continue
        r = numeric.loc[valid, col].corr(target[valid])
        results.append({"feature": col, "pearson": round(r, 4), "abs_pearson": round(abs(r), 4)})

    return pd.DataFrame(results).sort_values("abs_pearson", ascending=False).reset_index(drop=True)
