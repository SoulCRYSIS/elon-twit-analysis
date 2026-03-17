"""Background monitor for Polymarket Elon tweet bracket trading.

Periodically checks active events, scores buy/sell opportunities,
and sends macOS desktop notifications when signals fire.

Usage:
    python monitor.py                          # start monitoring loop
    python monitor.py --holdings               # show current holdings
    python monitor.py --buy <slug> <bracket> <price>  # record a buy
    python monitor.py --sell <slug> <bracket>  # remove a holding
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HOLDINGS_PATH = Path(__file__).parent / "data" / "holdings.json"
MODEL_DIR = Path(__file__).parent / "data" / "polymarket"
GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

POLL_INTERVAL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Holdings management
# ---------------------------------------------------------------------------

def load_holdings() -> list[dict]:
    if HOLDINGS_PATH.exists():
        with open(HOLDINGS_PATH) as f:
            return json.load(f)
    return []


def save_holdings(holdings: list[dict]):
    HOLDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HOLDINGS_PATH, "w") as f:
        json.dump(holdings, f, indent=2)


def add_holding(slug: str, bracket: str, price: float):
    holdings = load_holdings()
    holdings.append({
        "event_slug": slug,
        "bracket_label": bracket,
        "buy_price": price,
        "buy_time": datetime.now(timezone.utc).isoformat(),
    })
    save_holdings(holdings)
    print(f"Added: {slug} / {bracket} @ {price}")


def remove_holding(slug: str, bracket: str):
    holdings = load_holdings()
    before = len(holdings)
    holdings = [h for h in holdings if not (h["event_slug"] == slug and h["bracket_label"] == bracket)]
    save_holdings(holdings)
    print(f"Removed {before - len(holdings)} holding(s) for {slug} / {bracket}")


def show_holdings():
    holdings = load_holdings()
    if not holdings:
        print("No holdings.")
        return
    for h in holdings:
        print(f"  {h['event_slug']:50s} {h['bracket_label']:10s} buy={h['buy_price']:.4f}  @ {h['buy_time']}")


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------

def notify(title: str, message: str):
    """Send a macOS desktop notification."""
    try:
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "{title}"'
        ], check=False, capture_output=True)
    except FileNotFoundError:
        pass
    print(f"[ALERT] {title}: {message}")


# ---------------------------------------------------------------------------
# Live data fetching
# ---------------------------------------------------------------------------

def fetch_active_events() -> list[dict]:
    """Get currently active Elon tweet bracket events."""
    resp = requests.get(f"{GAMMA_URL}/public-search", params={
        "q": "elon musk tweets",
        "limit_per_type": 50,
        "events_status": "active",
    })
    resp.raise_for_status()
    stubs = resp.json().get("events", [])

    events = []
    for stub in stubs:
        slug = stub.get("slug", "")
        if not slug:
            continue
        resp2 = requests.get(f"{GAMMA_URL}/events", params={"slug": slug})
        resp2.raise_for_status()
        arr = resp2.json()
        if arr and isinstance(arr, list) and len(arr[0].get("markets", [])) >= 5:
            events.append(arr[0])
        time.sleep(0.3)
    return events


def fetch_current_prices(events: list[dict]) -> pd.DataFrame:
    """Get latest price for every bracket in the given events."""
    rows = []
    for e in events:
        slug = e.get("slug", "")
        event_end = e.get("endDate")
        for m in e.get("markets", []):
            tokens = m.get("clobTokenIds", [])
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            if not tokens:
                continue

            outcomes = m.get("outcomes", [])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            prices = m.get("outcomePrices", [])
            if isinstance(prices, str):
                prices = json.loads(prices)

            yes_price = float(prices[0]) if prices else 0

            from polymarket_data import parse_bracket
            bracket_low, bracket_high, bracket_label = parse_bracket(m.get("question", ""))

            rows.append({
                "event_slug": slug,
                "event_end": event_end,
                "bracket_label": bracket_label,
                "bracket_low": bracket_low,
                "bracket_high": bracket_high,
                "price": yes_price,
                "condition_id": m.get("conditionId", ""),
                "volume": float(m.get("volume", 0)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["event_end"] = pd.to_datetime(df["event_end"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["hours_remaining"] = (df["event_end"] - now).dt.total_seconds() / 3600

    # Compute leading bracket and distance
    grp = df.groupby("event_slug")
    df["leading_bracket_price"] = grp["price"].transform("max")
    df["distance_from_leading"] = df["leading_bracket_price"] - df["price"]
    df["bracket_mid"] = (df["bracket_low"].fillna(0) + df["bracket_high"].fillna(0)) / 2
    leading_mid = grp.apply(
        lambda g: g.loc[g["price"].idxmax(), "bracket_mid"]
    ).rename("leading_mid")
    df = df.merge(leading_mid.reset_index(), on="event_slug", how="left")
    df["brackets_away"] = np.abs(df["bracket_mid"] - df["leading_mid"]) / 25

    return df


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def score_buy_opportunities(df: pd.DataFrame, max_price: float = 0.03) -> pd.DataFrame:
    """Score brackets as buy candidates using simple heuristics.

    Uses the user's strategy rules when no trained model is available.
    """
    candidates = df[
        (df["price"] > 0.001) &
        (df["price"] < max_price) &
        (df["brackets_away"] >= 1.5) &
        (df["hours_remaining"] > 24)
    ].copy()

    if candidates.empty:
        return candidates

    # Score: lower price + more brackets away + more time remaining = better
    candidates["score"] = (
        (1 - candidates["price"] / max_price) * 0.4 +
        np.clip(candidates["brackets_away"] / 5, 0, 1) * 0.3 +
        np.clip(candidates["hours_remaining"] / 168, 0, 1) * 0.3
    )
    return candidates.sort_values("score", ascending=False)


def check_sell_signals(df: pd.DataFrame, holdings: list[dict]) -> list[dict]:
    """Check if any holdings should be sold."""
    signals = []
    for h in holdings:
        match = df[
            (df["event_slug"] == h["event_slug"]) &
            (df["bracket_label"] == h["bracket_label"])
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        buy_price = h["buy_price"]
        current_price = row["price"]
        ret = current_price / buy_price if buy_price > 0 else 0
        hours_left = row["hours_remaining"]

        reason = None
        if ret >= 4.0:
            reason = f"TARGET HIT: {ret:.1f}x return"
        elif hours_left < 24:
            reason = f"TIME: only {hours_left:.0f}h left, current {ret:.1f}x"
        elif ret >= 2.0 and hours_left < 48:
            reason = f"GOOD EXIT: {ret:.1f}x return, {hours_left:.0f}h left"

        if reason:
            signals.append({**h, "current_price": current_price, "return": ret, "reason": reason})

    return signals


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_monitor(poll_interval: int = POLL_INTERVAL):
    """Main monitoring loop."""
    print(f"Starting Polymarket monitor (polling every {poll_interval}s)...")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"[{now}] Checking markets...")

            events = fetch_active_events()
            if not events:
                print("  No active bracket events found.")
                time.sleep(poll_interval)
                continue

            df = fetch_current_prices(events)
            if df.empty:
                print("  No price data.")
                time.sleep(poll_interval)
                continue

            # Buy opportunities
            buys = score_buy_opportunities(df)
            if not buys.empty:
                top = buys.head(5)
                print(f"  Top {len(top)} buy candidates:")
                for _, r in top.iterrows():
                    print(f"    {r['event_slug'][:45]:45s} {r['bracket_label']:10s} "
                          f"price={r['price']:.4f} away={r['brackets_away']:.1f} "
                          f"hrs={r['hours_remaining']:.0f} score={r['score']:.3f}")

                best = buys.iloc[0]
                if best["score"] > 0.7:
                    notify(
                        "Buy Signal",
                        f"{best['bracket_label']} @ {best['price']:.3f} "
                        f"({best['event_slug'][:30]})"
                    )

            # Sell signals
            holdings = load_holdings()
            if holdings:
                sell_signals = check_sell_signals(df, holdings)
                for s in sell_signals:
                    notify(
                        "Sell Signal",
                        f"{s['bracket_label']} {s['reason']} "
                        f"({s['event_slug'][:30]})"
                    )
            else:
                print("  No holdings to check for sell signals.")

            print()

        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as exc:
            print(f"  Error: {exc}")

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymarket bracket trading monitor")
    parser.add_argument("--holdings", action="store_true", help="Show current holdings")
    parser.add_argument("--buy", nargs=3, metavar=("SLUG", "BRACKET", "PRICE"), help="Record a buy")
    parser.add_argument("--sell", nargs=2, metavar=("SLUG", "BRACKET"), help="Remove a holding")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="Poll interval in seconds")
    args = parser.parse_args()

    if args.holdings:
        show_holdings()
    elif args.buy:
        add_holding(args.buy[0], args.buy[1], float(args.buy[2]))
    elif args.sell:
        remove_holding(args.sell[0], args.sell[1])
    else:
        run_monitor(args.interval)


if __name__ == "__main__":
    main()
