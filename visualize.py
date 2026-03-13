import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Resample hourly data to the requested resolution."""
    ts = df.set_index("date")[["count"]].copy()
    if mode == "hourly":
        return ts.reset_index()
    elif mode == "daily":
        agg = ts.resample("1D").sum()
    elif mode == "2day":
        agg = ts.resample("2D").sum()
    elif mode == "weekly":
        agg = ts.resample("7D").sum()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return agg.reset_index()


MODES = ["hourly", "daily", "2day", "weekly"]
MODE_LABELS = {"hourly": "Hourly", "daily": "Daily", "2day": "2-Day", "weekly": "Weekly"}


# ---------------------------------------------------------------------------
# Chart 1: Tweet count with aggregation dropdown
# ---------------------------------------------------------------------------

def tweet_count_chart(df: pd.DataFrame) -> go.Figure:
    """Bar + line chart with dropdown buttons to switch aggregation."""
    fig = go.Figure()

    agg_data = {m: _aggregate(df, m) for m in MODES}

    for i, mode in enumerate(MODES):
        ad = agg_data[mode]
        visible = mode == "daily"  # default view
        bar_mode = "lines+markers" if mode == "hourly" else "markers+text" if False else "lines+markers"
        fig.add_trace(go.Bar(
            x=ad["date"],
            y=ad["count"],
            name=MODE_LABELS[mode],
            visible=visible,
            marker_color="#636EFA",
            opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=ad["date"],
            y=ad["count"],
            mode="lines",
            name=f"{MODE_LABELS[mode]} trend",
            visible=visible,
            line=dict(color="#EF553B", width=2),
        ))

    buttons = []
    for i, mode in enumerate(MODES):
        vis = [False] * (len(MODES) * 2)
        vis[i * 2] = True      # bar trace
        vis[i * 2 + 1] = True  # line trace
        buttons.append(dict(
            label=MODE_LABELS[mode],
            method="update",
            args=[{"visible": vis}],
        ))

    fig.update_layout(
        title="Elon Musk Tweet Count",
        xaxis_title="Date",
        yaxis_title="Tweet Count",
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
            buttons=buttons,
            bgcolor="#E2E2E2",
            font=dict(size=12),
        )],
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Hour-of-day heatmap
# ---------------------------------------------------------------------------

def hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: date (x) vs hour-of-day (y), colored by tweet count."""
    hm = df.copy()
    hm["day"] = hm["date"].dt.date
    hm["hour"] = hm["date"].dt.hour

    pivot = hm.pivot_table(index="hour", columns="day", values="count", aggfunc="sum", fill_value=0)

    x_labels = [str(d) for d in pivot.columns]
    # show every Nth label to avoid clutter
    n = max(1, len(x_labels) // 20)
    tick_vals = list(range(0, len(x_labels), n))
    tick_text = [x_labels[i] for i in tick_vals]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(range(len(x_labels))),
        y=pivot.index,
        colorscale="YlOrRd",
        colorbar_title="Tweets",
        hovertemplate="Date: %{customdata}<br>Hour: %{y}:00 UTC<br>Tweets: %{z}<extra></extra>",
        customdata=[[x_labels[c] for c in range(len(x_labels))] for _ in pivot.index],
    ))
    fig.update_layout(
        title="Tweet Activity Heatmap (Hour of Day vs Date)",
        xaxis=dict(title="Date", tickvals=tick_vals, ticktext=tick_text, tickangle=-45),
        yaxis=dict(title="Hour of Day (UTC)", dtick=1),
        template="plotly_white",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 3: Cumulative tweet chart with period boundaries
# ---------------------------------------------------------------------------

def cumulative_chart(df: pd.DataFrame, periods: list[dict] | None = None) -> go.Figure:
    """Line chart of cumulative tweets, with optional period boundary markers."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["cumulative"],
        mode="lines",
        name="Cumulative Tweets",
        line=dict(color="#636EFA", width=2),
        fill="tozeroy",
        fillcolor="rgba(99,110,250,0.1)",
    ))

    if periods:
        for p in periods:
            start = pd.Timestamp(p["startDate"]).tz_convert("UTC") if isinstance(p.get("startDate"), str) else p.get("_start")
            if start is None:
                continue
            if isinstance(start, str):
                start = pd.Timestamp(start)
            fig.add_vline(
                x=start,
                line_dash="dot",
                line_color="gray",
                opacity=0.4,
            )

    fig.update_layout(
        title="Cumulative Tweet Count",
        xaxis_title="Date",
        yaxis_title="Total Tweets",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Convenience: show all charts
# ---------------------------------------------------------------------------

def show_all(df: pd.DataFrame, periods: list[dict] | None = None):
    """Render all three charts (for use in a notebook)."""
    tweet_count_chart(df).show()
    hourly_heatmap(df).show()
    cumulative_chart(df, periods).show()
