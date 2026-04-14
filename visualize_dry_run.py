#!/usr/bin/env python3
"""Visualize dry-run JSONL: 2-day vs 7-day markets, buy/sell/profit table, balance timeline."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _classify_period_hours(dur_hours: float) -> str:
    """Match features._classify_period: short windows vs ~weekly."""
    if dur_hours <= 72:
        return "2day"
    return "7day"


def parse_event_slug_window(event_slug: str, year: int = 2026) -> tuple[float, str] | None:
    """Return (duration_hours, label) from slug like elon-musk-of-tweets-march-13-march-20."""
    lower = event_slug.lower()
    if "elon-musk-of-tweets-" not in lower:
        return None
    rest = lower.split("elon-musk-of-tweets-", 1)[-1]
    m = re.match(r"([a-z]+)-(\d+)-([a-z]+)-(\d+)", rest)
    if not m:
        return None
    ma, da, mb, db = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        d0 = datetime.strptime(f"{ma.capitalize()} {int(da)} {year}", "%B %d %Y")
        d1 = datetime.strptime(f"{mb.capitalize()} {int(db)} {year}", "%B %d %Y")
    except ValueError:
        return None
    delta = d1 - d0
    dur_hours = max(delta.total_seconds() / 3600.0, 1e-6)
    return dur_hours, _classify_period_hours(dur_hours)


def load_transactions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    types: list[str] = []
    for slug in df["event_slug"].astype(str):
        parsed = parse_event_slug_window(slug)
        types.append(parsed[1] if parsed else "unknown")
    df["period_type"] = types

    return df.sort_values("time").reset_index(drop=True)


def summarize_by_type(df: pd.DataFrame) -> pd.DataFrame:
    buys = df[df["action"] == "buy"].copy()
    sells = df[df["action"] == "sell"].copy()

    if buys.empty:
        b = pd.DataFrame(columns=["buys_n", "buy_usd"]).set_index(
            pd.Index([], name="period_type")
        )
    else:
        b = buys.groupby("period_type", dropna=False).agg(
            buys_n=("usd", "count"), buy_usd=("usd", "sum")
        )
    sell_agg: dict = {
        "sells_n": ("event_slug", "size"),
        "sell_proceeds": ("proceeds", "sum"),
        "realized_profit_usd": ("profit_usd", "sum"),
    }
    if sells.empty:
        s = pd.DataFrame(
            columns=["sells_n", "sell_proceeds", "realized_profit_usd"]
        ).set_index(pd.Index([], name="period_type"))
    else:
        s = sells.groupby("period_type", dropna=False).agg(**sell_agg)

    out = b.join(s, how="outer").fillna(0)
    for c in ["buys_n", "sells_n"]:
        out[c] = out[c].astype(int)
    out["roi_on_buys_pct"] = (
        out["realized_profit_usd"] / out["buy_usd"].replace(0, float("nan")) * 100
    ).round(2)
    return out.reset_index().sort_values("period_type")


def fig_comparison(summary: pd.DataFrame) -> go.Figure:
    sub = summary[summary["period_type"].isin(["2day", "7day"])].copy()
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(title="No 2day/7day rows to compare", template="plotly_white")
        return fig

    fig = go.Figure(
        data=[
            go.Bar(
                name="Realized profit (USD)",
                x=sub["period_type"],
                y=sub["realized_profit_usd"],
                marker_color=["#636EFA", "#EF553B"][: len(sub)],
                text=[f"${v:.2f}" for v in sub["realized_profit_usd"]],
                textposition="outside",
            ),
            go.Bar(
                name="Total buys (USD)",
                x=sub["period_type"],
                y=sub["buy_usd"],
                marker_color=["#AB63FA", "#FFA15A"][: len(sub)],
                text=[f"${v:.2f}" for v in sub["buy_usd"]],
                textposition="outside",
            ),
        ]
    )
    winner = sub.loc[sub["realized_profit_usd"].idxmax(), "period_type"]
    fig.update_layout(
        title=f"2-day vs 7-day: realized profit & capital deployed (higher profit: {winner})",
        barmode="group",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_balance_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df["time"],
            y=df["balance"],
            mode="lines+markers",
            name="Cash balance",
            line=dict(color="#00CC96", width=2),
            marker=dict(size=6),
        )
    )
    if not df.empty:
        fig.add_hline(
            y=df["balance"].iloc[0],
            line_dash="dot",
            line_color="gray",
            annotation_text="Start",
        )
    fig.update_layout(
        title="Portfolio cash (balance after each transaction)",
        xaxis_title="Time (UTC)",
        yaxis_title="USD",
        template="plotly_white",
        height=480,
        hovermode="x unified",
    )
    return fig


def fig_summary_table(summary: pd.DataFrame) -> go.Figure:
    disp = summary.copy()
    disp["buy_usd"] = disp["buy_usd"].map(lambda x: f"{x:.2f}")
    disp["sell_proceeds"] = disp["sell_proceeds"].map(lambda x: f"{x:.2f}")
    disp["realized_profit_usd"] = disp["realized_profit_usd"].map(lambda x: f"{x:.2f}")
    disp["roi_on_buys_pct"] = disp["roi_on_buys_pct"].map(
        lambda x: "" if pd.isna(x) else f"{x:.1f}%"
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(disp.columns),
                    fill_color="#E2E2E2",
                    align="left",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[disp[c] for c in disp.columns],
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )
    fig.update_layout(title="Buy / sell / profit by period type", height=280)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dry-run transaction log.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dry_run_transactions.jsonl"),
        help="Path to dry_run_transactions.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/dry_run_report.html"),
        help="Write combined HTML report here",
    )
    args = parser.parse_args()

    df = load_transactions(args.input)
    summary = summarize_by_type(df)

    print("=== Buy / sell / profit by period type (2-day = ≤72h window in slug) ===\n")
    print(summary.to_string(index=False))
    print()

    sub = summary[summary["period_type"].isin(["2day", "7day"])]
    if len(sub) >= 2:
        d2 = sub.loc[sub["period_type"] == "2day", "realized_profit_usd"].sum()
        d7 = sub.loc[sub["period_type"] == "7day", "realized_profit_usd"].sum()
        print(f"Realized profit — 2-day markets: ${d2:.2f}  |  7-day markets: ${d7:.2f}")
        print(f"More profit: {'2-day' if d2 > d7 else '7-day' if d7 > d2 else 'tie'}\n")
    elif len(sub) == 1:
        only = sub.iloc[0]["period_type"]
        print(f"Only one category present in data: {only}\n")

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.38, 0.35, 0.27],
        specs=[[{"type": "bar"}], [{"type": "scatter"}], [{"type": "table"}]],
        subplot_titles=(
            "2-day vs 7-day: profit & buys",
            "Cash balance over time",
            "Summary table",
        ),
        vertical_spacing=0.10,
    )

    comp = fig_comparison(summary)
    for tr in comp.data:
        fig.add_trace(tr, row=1, col=1)

    bal = fig_balance_timeline(df)
    for tr in bal.data:
        fig.add_trace(tr, row=2, col=1)

    tbl = fig_summary_table(summary)
    for tr in tbl.data:
        fig.add_trace(tr, row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        height=1100,
        showlegend=True,
        title_text=f"Dry run report ({args.input.name})",
    )
    fig.update_xaxes(title_text="Period type", row=1, col=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
