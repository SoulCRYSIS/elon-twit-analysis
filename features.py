"""Feature engineering for Elon Musk tweet count prediction.

Produces two levels of features:
- Hourly: one row per hour, for correlation analysis and trend detection.
- Period-level: one row per Polymarket betting period, for predicting totals.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fetch_data import fetch_user_trackings, _parse_dt


# ---------------------------------------------------------------------------
# Hourly feature engineering
# ---------------------------------------------------------------------------

def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to an hourly DataFrame with columns (date, count)."""
    f = df[["date", "count"]].copy()
    f = f.sort_values("date").reset_index(drop=True)
    c = f["count"]

    # -- Time / calendar ---------------------------------------------------
    f["hour"] = f["date"].dt.hour
    f["day_of_week"] = f["date"].dt.dayofweek  # 0=Mon .. 6=Sun
    f["day_of_month"] = f["date"].dt.day
    f["is_weekend"] = f["day_of_week"].isin([5, 6]).astype(int)
    f["part_of_day"] = pd.cut(
        f["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3],  # night, morning, afternoon, evening
    ).astype(int)

    # -- Lag features (mean-reversion signals) -----------------------------
    for lag in [1, 2, 3, 4, 6, 12, 24]:
        f[f"lag_{lag}h"] = c.shift(lag)

    # -- Rolling window averages -------------------------------------------
    for win in [2, 4, 6, 12, 24, 48, 168]:
        f[f"rolling_mean_{win}h"] = c.shift(1).rolling(win, min_periods=1).mean()

    # -- Rolling standard deviations (volatility) --------------------------
    for win in [6, 12, 24]:
        f[f"rolling_std_{win}h"] = c.shift(1).rolling(win, min_periods=1).std()

    # -- Same-time comparisons ---------------------------------------------
    f["same_hour_yesterday"] = c.shift(24)
    f["same_hour_last_week"] = c.shift(168)

    # -- Activity patterns -------------------------------------------------
    f["hours_since_last_tweet"] = _hours_since_last(c)
    f["day_total_so_far"] = _day_total_so_far(f)
    f["consecutive_zero_hours"] = _streak(c, zero=True)
    f["consecutive_active_hours"] = _streak(c, zero=False)

    # -- EWMA --------------------------------------------------------------
    f["ewma_12h"] = c.shift(1).ewm(span=12, min_periods=1).mean()
    f["ewma_24h"] = c.shift(1).ewm(span=24, min_periods=1).mean()

    # -- Trend / burst detection -------------------------------------------
    rm6 = f["rolling_mean_6h"]
    rm48 = f["rolling_mean_48h"]
    f["short_long_ratio"] = np.where(rm48 > 0, rm6 / rm48, 0.0)
    f["acceleration"] = rm6 - rm6.shift(6)

    return f


def _hours_since_last(counts: pd.Series) -> pd.Series:
    """Number of hours since the last hour with count > 0."""
    active = counts > 0
    groups = active.cumsum()
    result = pd.Series(np.nan, index=counts.index)
    for g, sub in counts.groupby(groups):
        result.iloc[sub.index] = range(len(sub))
    return result


def _day_total_so_far(f: pd.DataFrame) -> pd.Series:
    """Cumulative tweet count within each calendar day, up to the current hour."""
    day_key = f["date"].dt.date
    return f.groupby(day_key)["count"].cumsum() - f["count"]


def _streak(counts: pd.Series, zero: bool) -> pd.Series:
    """Current streak length of consecutive zero (or non-zero) hours."""
    mask = (counts == 0) if zero else (counts > 0)
    groups = (~mask).cumsum()
    return mask.groupby(groups).cumsum()


# ---------------------------------------------------------------------------
# Period-level feature engineering
# ---------------------------------------------------------------------------

def _classify_period(dur_hours: float) -> str:
    """Label a period as '2day' or 'weekly' based on its duration."""
    if dur_hours <= 72:
        return "2day"
    return "weekly"


def build_period_features(
    df_hourly: pd.DataFrame,
    trackings: list[dict] | None = None,
) -> pd.DataFrame:
    """Build one row per Polymarket betting period with pre-period features and target.

    Parameters
    ----------
    df_hourly : DataFrame with columns (date, count, ...) — the hourly feature table.
    trackings : raw tracking list from the API. If None, fetches from cache.
    """
    if trackings is None:
        trackings = fetch_user_trackings()

    hf = build_hourly_features(df_hourly)
    hf = hf.set_index("date").sort_index()

    rows: list[dict] = []

    for t in trackings:
        start = pd.Timestamp(_parse_dt(t["startDate"])).tz_convert("UTC")
        end = pd.Timestamp(_parse_dt(t["endDate"])).tz_convert("UTC")
        dur_hours = (end - start).total_seconds() / 3600
        ptype = _classify_period(dur_hours)

        # actual tweets in the period (target)
        period_mask = (hf.index >= start) & (hf.index <= end)
        period_slice = hf.loc[period_mask]
        if len(period_slice) == 0:
            continue
        period_total = int(period_slice["count"].sum())

        # pre-period windows
        pre_48h = hf.loc[(hf.index >= start - timedelta(hours=48)) & (hf.index < start)]
        pre_168h = hf.loc[(hf.index >= start - timedelta(hours=168)) & (hf.index < start)]

        if len(pre_48h) < 12:
            continue  # not enough history

        row = {
            "period_id": t["id"],
            "title": t.get("title", ""),
            "start": start,
            "end": end,
            "period_type": ptype,
            "period_duration_hours": dur_hours,
            "period_start_hour": start.hour,
            "period_start_dow": start.dayofweek,
            "is_active": t.get("isActive", True),

            "pre_total_48h": int(pre_48h["count"].sum()),
            "pre_total_168h": int(pre_168h["count"].sum()),
            "pre_mean_hourly": pre_48h["count"].mean(),
            "pre_std_hourly": pre_48h["count"].std(),
            "pre_max_hourly": int(pre_48h["count"].max()),
            "pre_median_hourly": pre_48h["count"].median(),
            "pre_weekend_ratio": pre_48h["is_weekend"].mean() if "is_weekend" in pre_48h.columns else 0,

            "pre_ewma_12h_last": pre_48h["ewma_12h"].iloc[-1] if "ewma_12h" in pre_48h.columns else np.nan,
            "pre_ewma_24h_last": pre_48h["ewma_24h"].iloc[-1] if "ewma_24h" in pre_48h.columns else np.nan,
            "pre_rolling_std_24h_last": pre_48h["rolling_std_24h"].iloc[-1] if "rolling_std_24h" in pre_48h.columns else np.nan,

            "pre_burst_ratio": (
                (pre_48h["short_long_ratio"] > 1.5).mean()
                if "short_long_ratio" in pre_48h.columns else 0
            ),

            "period_total": period_total,
        }
        rows.append(row)

    pdf = pd.DataFrame(rows)
    if pdf.empty:
        return pdf

    pdf = pdf.sort_values("start").reset_index(drop=True)

    # prev_period_total: total of the most recent completed same-type period
    for ptype in ["2day", "weekly"]:
        mask = pdf["period_type"] == ptype
        sub = pdf.loc[mask, "period_total"].shift(1)
        pdf.loc[mask, "prev_period_total"] = sub

    return pdf


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

SKIP_COLS = {"date", "count", "cumulative"}

def compute_correlations(df_features: pd.DataFrame) -> pd.DataFrame:
    """Pearson and Spearman correlation of every numeric feature vs 'count'."""
    numeric = df_features.select_dtypes(include="number").drop(columns=["count"], errors="ignore")
    target = df_features["count"]

    results = []
    for col in numeric.columns:
        valid = numeric[col].notna() & target.notna()
        if valid.sum() < 30:
            continue
        pearson = numeric.loc[valid, col].corr(target[valid])
        spearman = numeric.loc[valid, col].corr(target[valid], method="spearman")
        results.append({
            "feature": col,
            "pearson": round(pearson, 4),
            "spearman": round(spearman, 4),
            "abs_pearson": round(abs(pearson), 4),
        })

    cdf = pd.DataFrame(results).sort_values("abs_pearson", ascending=False).reset_index(drop=True)
    return cdf
