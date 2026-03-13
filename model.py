"""ML models for predicting Elon Musk tweet counts per Polymarket period,
plus an hourly trend-change classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)

from features import build_hourly_features


# ---------------------------------------------------------------------------
# Feature columns used by period-level models
# ---------------------------------------------------------------------------

PERIOD_FEATURE_COLS = [
    "period_duration_hours",
    "period_start_hour",
    "period_start_dow",
    "pre_total_48h",
    "pre_total_168h",
    "pre_mean_hourly",
    "pre_std_hourly",
    "pre_max_hourly",
    "pre_median_hourly",
    "pre_weekend_ratio",
    "pre_ewma_12h_last",
    "pre_ewma_24h_last",
    "pre_rolling_std_24h_last",
    "pre_burst_ratio",
    "prev_period_total",
]

TARGET_COL = "period_total"


# ---------------------------------------------------------------------------
# Period prediction
# ---------------------------------------------------------------------------

def _clean_period_df(pdf: pd.DataFrame, period_type: str | None = None) -> pd.DataFrame:
    """Filter to completed periods, deduplicate by (start, end), optionally filter type."""
    d = pdf[~pdf["is_active"]].copy()
    d = d.drop_duplicates(subset=["start", "end"], keep="first")
    if period_type:
        d = d[d["period_type"] == period_type]
    return d.sort_values("start").reset_index(drop=True)


def train_test_split_periods(pdf: pd.DataFrame, test_frac: float = 0.2):
    """Time-based split: last `test_frac` of rows become test set."""
    n = len(pdf)
    split = int(n * (1 - test_frac))
    split = max(split, 1)
    return pdf.iloc[:split].copy(), pdf.iloc[split:].copy()


def _prepare_Xy(df: pd.DataFrame):
    cols = [c for c in PERIOD_FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    y = df[TARGET_COL].copy()
    X = X.fillna(X.median())
    return X, y


MODELS = {
    "LinearRegression": LinearRegression,
    "RandomForest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42,
    ),
}


def train_and_evaluate(
    pdf: pd.DataFrame,
    period_type: str | None = None,
    test_frac: float = 0.2,
    verbose: bool = True,
) -> dict:
    """Train all models on period data, return evaluation dict.

    Returns
    -------
    dict with keys: results (list of dicts), models (dict name->fitted model),
    feature_names, X_train, X_test, y_train, y_test, train_df, test_df.
    """
    clean = _clean_period_df(pdf, period_type)
    if len(clean) < 5:
        raise ValueError(f"Too few completed periods for type={period_type!r}: {len(clean)}")

    train_df, test_df = train_test_split_periods(clean, test_frac)
    X_train, y_train = _prepare_Xy(train_df)
    X_test, y_test = _prepare_Xy(test_df)

    if verbose:
        label = period_type or "all"
        print(f"=== {label.upper()} periods: {len(train_df)} train / {len(test_df)} test ===")

    results = []
    fitted = {}

    for name, factory in MODELS.items():
        model = factory() if callable(factory) and not isinstance(factory, type) else factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test.values - preds) / np.clip(y_test.values, 1, None))) * 100

        results.append({"model": name, "MAE": round(mae, 1), "RMSE": round(rmse, 1), "MAPE%": round(mape, 1)})
        fitted[name] = model

        if verbose:
            print(f"  {name:25s}  MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%")

    return {
        "results": results,
        "models": fitted,
        "feature_names": list(X_train.columns),
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "train_df": train_df, "test_df": test_df,
    }


# ---------------------------------------------------------------------------
# Feature importance chart
# ---------------------------------------------------------------------------

def feature_importance_chart(info: dict, model_name: str = "GradientBoosting") -> go.Figure:
    """Bar chart of feature importances from a tree-based model."""
    model = info["models"][model_name]
    importances = model.feature_importances_
    names = info["feature_names"]

    order = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[order],
        y=[names[i] for i in order],
        orientation="h",
        marker_color="#636EFA",
    ))
    fig.update_layout(
        title=f"Feature Importance ({model_name})",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(350, len(names) * 25),
        margin=dict(l=180),
    )
    return fig


def prediction_vs_actual_chart(info: dict, model_name: str = "GradientBoosting") -> go.Figure:
    """Scatter plot of predicted vs actual period totals on the test set."""
    model = info["models"][model_name]
    preds = model.predict(info["X_test"])
    actual = info["y_test"].values
    titles = info["test_df"]["title"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=preds, mode="markers+text",
        text=[t[:30] for t in titles],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(size=10, color="#636EFA"),
        name="Periods",
    ))
    lo = min(actual.min(), preds.min()) * 0.8
    hi = max(actual.max(), preds.max()) * 1.2
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(dash="dash", color="gray"), name="Perfect",
    ))
    fig.update_layout(
        title=f"Predicted vs Actual ({model_name})",
        xaxis_title="Actual Period Total",
        yaxis_title="Predicted Period Total",
        template="plotly_white",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Trend-change classifier (hourly)
# ---------------------------------------------------------------------------

def build_trend_labels(df_hourly: pd.DataFrame, horizon: int = 6, threshold: float = 1.0) -> pd.DataFrame:
    """Create binary trend-change labels on hourly data.

    Label = 1 if the mean count over the next `horizon` hours exceeds
    `threshold` * the trailing 48h average (i.e., a 'burst' is coming).
    """
    hf = build_hourly_features(df_hourly)
    c = hf["count"]

    future_mean = c.shift(-horizon).rolling(horizon, min_periods=horizon).mean()
    # shift(-horizon) gives us the value `horizon` steps ahead;
    # rolling(horizon) computes the mean of the next `horizon` hours starting from the *current* row.
    # Simpler approach: use a forward-looking rolling mean.
    future_mean = c.iloc[::-1].rolling(horizon, min_periods=horizon).mean().iloc[::-1]

    trailing_48h = c.shift(1).rolling(48, min_periods=24).mean()

    hf["trend_label"] = (future_mean > threshold * trailing_48h).astype(int)
    hf["future_mean_6h"] = future_mean
    hf["trailing_48h_mean"] = trailing_48h
    return hf


TREND_FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "part_of_day",
    "lag_1h", "lag_2h", "lag_3h", "lag_4h", "lag_6h", "lag_12h", "lag_24h",
    "rolling_mean_6h", "rolling_mean_12h", "rolling_mean_24h", "rolling_mean_48h",
    "rolling_std_6h", "rolling_std_12h", "rolling_std_24h",
    "same_hour_yesterday", "same_hour_last_week",
    "hours_since_last_tweet", "day_total_so_far",
    "consecutive_zero_hours", "consecutive_active_hours",
    "ewma_12h", "ewma_24h",
    "short_long_ratio", "acceleration",
]


def train_trend_classifier(
    df_hourly: pd.DataFrame,
    test_frac: float = 0.2,
    verbose: bool = True,
) -> dict:
    """Train a GradientBoosting classifier to predict upcoming trend changes."""
    hf = build_trend_labels(df_hourly)
    hf = hf.dropna(subset=["trend_label"] + TREND_FEATURE_COLS).copy()

    n = len(hf)
    split = int(n * (1 - test_frac))

    train = hf.iloc[:split]
    test = hf.iloc[split:]

    X_train = train[TREND_FEATURE_COLS].fillna(0)
    y_train = train["trend_label"]
    X_test = test[TREND_FEATURE_COLS].fillna(0)
    y_test = test["trend_label"]

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds.astype(float)
    acc = accuracy_score(y_test, preds)

    if verbose:
        print(f"\n=== Trend-Change Classifier (6h horizon) ===")
        print(f"  Train: {len(train)}, Test: {len(test)}")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Label balance (test): {y_test.value_counts().to_dict()}")
        print(classification_report(y_test, preds, target_names=["Calm", "Burst"], zero_division=0))

    importances = clf.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": TREND_FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "classifier": clf,
        "accuracy": acc,
        "feature_importance": feat_imp,
        "test_dates": test["date"].values,
        "test_proba": proba,
        "test_actual": y_test.values,
        "X_test": X_test,
        "y_test": y_test,
    }


def trend_importance_chart(trend_info: dict) -> go.Figure:
    """Bar chart of trend classifier feature importances."""
    fi = trend_info["feature_importance"]
    fig = go.Figure(go.Bar(
        x=fi["importance"].values[::-1],
        y=fi["feature"].values[::-1],
        orientation="h",
        marker_color="#AB63FA",
    ))
    fig.update_layout(
        title="Trend-Change Classifier Feature Importance",
        xaxis_title="Importance",
        template="plotly_white",
        height=max(400, len(fi) * 22),
        margin=dict(l=200),
    )
    return fig


def trend_probability_chart(trend_info: dict) -> go.Figure:
    """Time series of burst probability vs actual label on the test set."""
    dates = trend_info["test_dates"]
    proba = trend_info["test_proba"]
    actual = trend_info["test_actual"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=proba, mode="lines",
        name="Burst probability", line=dict(color="#EF553B"),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=actual, mode="markers",
        name="Actual burst", marker=dict(size=3, color="#636EFA", opacity=0.4),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Trend-Change Probability (Test Set)",
        xaxis_title="Date",
        yaxis_title="Burst Probability",
        template="plotly_white",
        hovermode="x unified",
        height=400,
    )
    return fig
