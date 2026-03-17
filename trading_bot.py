"""Auto-trading bot for Polymarket Elon tweet brackets.

Runs 24/7, uses jackpot model (conf_p30) to buy, manual rules to sell.
- Configurable bet amount (default $1)
- Can buy same bracket multiple times if price changes and model approves
- Max 20 open positions (hard limit)

Data (persists across restarts):
  data/holdings.json       — open positions (loaded on startup)
  data/transactions.jsonl — buy/sell log (append-only)
  data/model_outcome.pkl  — trained model

.env variables:
  PK or PRIVATE_KEY       - Wallet private key (required)
  BET_USD                 - USD per trade (default 1)
  MAX_POSITIONS           - Max open positions (default 20)
  BUY_COOLDOWN_HOURS      - Hours before adding to same bracket again (default 24)
  BUY_AGAIN_PRICE_RATIO   - Only add when price < last/this (default 3 = must be 3x lower)
  DRY_RUN                 - Set to "true" to disable real orders
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Paths (all under data/ for persistence across restarts)
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
POLYMARKET_DIR = DATA_DIR / "polymarket"
HOLDINGS_PATH = DATA_DIR / "holdings.json"  # Positions — loaded on bot restart (real mode only)
DRY_RUN_HOLDINGS_PATH = DATA_DIR / "dry_run_holdings.json"  # Simulated positions when --dry-run
TRANSACTIONS_PATH = DATA_DIR / "transactions.jsonl"  # Buy/sell log
MODEL_PATH = DATA_DIR / "model_outcome.pkl"

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
CHAIN_ID = 137

POLL_INTERVAL = 300  # 5 min
MAX_OPEN_POSITIONS = 20
DEFAULT_BET_USD = 1.0
PRICE_HISTORY_HOURS = 48  # Keep 48h in memory for momentum/volatility features
PRED_THRESHOLD = 6.0  # Min predicted jackpot potential; set PRED_MIN in .env
JACKPOT_PROBA_THRESHOLD = 0.01  # Min P(jackpot); classifier is conservative on live (no history)
SELL_TARGET_X = 4.5
SELL_1DAY_HOURS = 24
BUY_COOLDOWN_HOURS = 12  # Don't add to same bracket within this many hours
BUY_AGAIN_PRICE_RATIO = 3.0  # Only add when price < last_buy_price / this (averaging down)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    load_dotenv(ROOT / ".env")
    pk = os.getenv("PK") or os.getenv("PRIVATE_KEY")
    if not pk:
        raise ValueError("Missing PK or PRIVATE_KEY in .env")
    return {
        "private_key": pk.strip(),
        "bet_usd": float(os.getenv("BET_USD", DEFAULT_BET_USD)),
        "max_positions": int(os.getenv("MAX_POSITIONS", MAX_OPEN_POSITIONS)),
        "dry_run": os.getenv("DRY_RUN", "false").lower() == "true",
        "pred_min": float(os.getenv("PRED_MIN", PRED_THRESHOLD)),
        "proba_min": float(os.getenv("JACKPOT_PROBA_MIN", JACKPOT_PROBA_THRESHOLD)),
        "buy_cooldown_hours": float(os.getenv("BUY_COOLDOWN_HOURS", BUY_COOLDOWN_HOURS)),
        "buy_again_price_ratio": float(os.getenv("BUY_AGAIN_PRICE_RATIO", BUY_AGAIN_PRICE_RATIO)),
    }


# ---------------------------------------------------------------------------
# Polymarket client
# ---------------------------------------------------------------------------

def get_clob_client() -> "ClobClient":
    from py_clob_client.client import ClobClient

    config = load_config()
    client = ClobClient(
        CLOB_URL,
        key=config["private_key"],
        chain_id=CHAIN_ID,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


# ---------------------------------------------------------------------------
# Holdings & transactions
# ---------------------------------------------------------------------------

def load_holdings(dry_run: bool = False) -> list[dict]:
    path = DRY_RUN_HOLDINGS_PATH if dry_run else HOLDINGS_PATH
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_holdings(holdings: list[dict]):
    HOLDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HOLDINGS_PATH, "w") as f:
        json.dump(holdings, f, indent=2)


def add_dry_holding(slug: str, bracket: str, price: float, shares: float, token_id: str):
    """Save simulated buy to dry_run_holdings.json for testing."""
    holdings = []
    if DRY_RUN_HOLDINGS_PATH.exists():
        with open(DRY_RUN_HOLDINGS_PATH) as f:
            holdings = json.load(f)
    existing = next((h for h in holdings if h["event_slug"] == slug and h["bracket_label"] == bracket), None)
    now_iso = datetime.now(timezone.utc).isoformat()
    if existing:
        total_shares = existing["shares"] + shares
        avg_price = (existing["buy_price"] * existing["shares"] + price * shares) / total_shares
        existing["shares"] = total_shares
        existing["buy_price"] = avg_price
        existing["last_buy_time"] = now_iso
    else:
        holdings.append({
            "event_slug": slug,
            "bracket_label": bracket,
            "buy_price": price,
            "shares": shares,
            "token_id": token_id,
            "buy_time": now_iso,
        })
    DRY_RUN_HOLDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRY_RUN_HOLDINGS_PATH, "w") as f:
        json.dump(holdings, f, indent=2)


def add_holding(slug: str, bracket: str, price: float, shares: float, token_id: str, order_id: str):
    holdings = load_holdings()
    existing = next((h for h in holdings if h["event_slug"] == slug and h["bracket_label"] == bracket), None)
    now_iso = datetime.now(timezone.utc).isoformat()
    if existing:
        total_shares = existing["shares"] + shares
        avg_price = (existing["buy_price"] * existing["shares"] + price * shares) / total_shares
        existing["shares"] = total_shares
        existing["buy_price"] = avg_price
        existing["order_id"] = order_id
        existing["last_buy_time"] = now_iso
    else:
        holdings.append({
            "event_slug": slug,
            "bracket_label": bracket,
            "buy_price": price,
            "shares": shares,
            "token_id": token_id,
            "order_id": order_id,
            "buy_time": now_iso,
        })
    save_holdings(holdings)


def remove_holding(slug: str, bracket: str):
    holdings = load_holdings()
    before = len(holdings)
    holdings = [h for h in holdings if not (h["event_slug"] == slug and h["bracket_label"] == bracket)]
    save_holdings(holdings)
    return before - len(holdings)


def _fmt_event(slug: str) -> str:
    """Shorten event slug: elon-musk-of-tweets-march-10-march-17 -> March 10 March 17"""
    for prefix in ("elon-musk-of-tweets-", "elon-musk-tweets-"):
        if slug.lower().startswith(prefix):
            rest = slug[len(prefix):].replace("-", " ").title()
            return rest[:35] + ("…" if len(rest) > 35 else "")
    return slug[:40] + ("…" if len(slug) > 40 else "")


def _print_buy(tag: str, event_slug: str, bracket: str, price: float, pred: float = None, usd: float = None):
    ev = _fmt_event(event_slug)
    parts = [f"  [{tag}]", ev, f"bracket {bracket}", f"@ {price:.4f}"]
    if pred is not None:
        parts.append(f"pred={pred:.1f}x")
    if usd is not None:
        parts.append(f"${usd}")
    print("  ".join(parts))


def _print_sell(event_slug: str, bracket: str, price: float, reason: str):
    ev = _fmt_event(event_slug)
    print(f"  [SOLD] {ev}  bracket {bracket}  @ {price:.3f}  ({reason})")


def log_transaction(action: str, **kwargs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "action": action,
        **kwargs,
    }
    with open(TRANSACTIONS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# In-memory price history (runtime only, ~50 brackets × 48h = trivial)
# ---------------------------------------------------------------------------

_price_history: dict[str, list[tuple[int, float]]] = {}  # condition_id -> [(ts, price), ...]


def _append_prices_to_history(df: pd.DataFrame):
    """Append current prices to in-memory history. Prune to PRICE_HISTORY_HOURS."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    cutoff = now_ts - PRICE_HISTORY_HOURS * 3600
    for _, row in df.iterrows():
        cid = row.get("condition_id", "")
        if not cid:
            continue
        price = float(row.get("price", 0))
        if cid not in _price_history:
            _price_history[cid] = []
        hist = _price_history[cid]
        hist.append((now_ts, price))
        while hist and hist[0][0] < cutoff:
            hist.pop(0)


def _fetch_clob_history(token_id: str, condition_id: str) -> list[tuple[int, float]]:
    """Fetch last 24h from CLOB prices-history for backfill. Returns [(ts, price), ...]."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = now_ts - 24 * 3600
    try:
        resp = requests.get(
            f"{CLOB_URL}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": now_ts, "fidelity": 60},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        pts = resp.json().get("history", [])
        return [(p["t"], p["p"]) for p in pts if "t" in p and "p" in p]
    except Exception:
        return []


def _compute_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price_change_*, price_vol_*, price_pct_range from in-memory history."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    for col in ["price_change_1h", "price_change_3h", "price_change_6h", "price_change_12h", "price_change_24h",
                "price_vol_6h", "price_vol_12h", "price_vol_24h", "price_pct_range", "hhi", "entropy"]:
        if col not in df.columns:
            df[col] = 0.0

    for i, row in df.iterrows():
        cid = row.get("condition_id", "")
        token_id = row.get("token_id", "")
        price = float(row.get("price", 0))

        hist = _price_history.get(cid)
        if not hist and token_id:
            backfill = _fetch_clob_history(token_id, cid)
            if backfill:
                _price_history[cid] = sorted(backfill, key=lambda x: x[0])
                hist = _price_history[cid]
            time.sleep(0.05)
        if not hist:
            continue

        prices = [p for _, p in hist]
        ts_list = [t for t, _ in hist]

        for lag_h in [1, 3, 6, 12, 24]:
            target_ts = now_ts - lag_h * 3600
            closest = min(ts_list, key=lambda t: abs(t - target_ts))
            if abs(closest - target_ts) <= 2 * 3600:
                idx = ts_list.index(closest)
                df.at[i, f"price_change_{lag_h}h"] = price - prices[idx]
            else:
                df.at[i, f"price_change_{lag_h}h"] = 0.0

        for win in [6, 12, 24]:
            recent = [(t, p) for t, p in hist if t >= now_ts - win * 3600]
            if len(recent) >= 2:
                vals = [p for _, p in recent]
                df.at[i, f"price_vol_{win}h"] = float(np.std(vals))
            else:
                df.at[i, f"price_vol_{win}h"] = 0.0

        pmin, pmax = min(prices), max(prices)
        if pmax > pmin:
            df.at[i, "price_pct_range"] = (price - pmin) / (pmax - pmin)
        else:
            df.at[i, "price_pct_range"] = 0.0

    ts_grp = df.groupby("event_slug")
    price_sum = ts_grp["price"].transform("sum")
    df["price_share"] = df["price"] / price_sum.replace(0, np.nan)
    df["hhi"] = ts_grp["price_share"].transform(lambda s: (s ** 2).sum())
    df["_neg_plogp"] = np.where(
        df["price_share"] > 0,
        -df["price_share"] * np.log2(df["price_share"].clip(lower=1e-10)),
        0,
    )
    df["entropy"] = ts_grp["_neg_plogp"].transform("sum")
    df.drop(columns=["price_share", "_neg_plogp"], inplace=True, errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Live data
# ---------------------------------------------------------------------------

def fetch_active_events() -> list[dict]:
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
        time.sleep(0.2)
    return events


def fetch_current_prices(events: list[dict]) -> pd.DataFrame:
    from polymarket_data import parse_bracket

    rows = []
    for e in events:
        slug = e.get("slug", "")
        event_start = e.get("startDate")
        event_end = e.get("endDate")
        event_vol = float(e.get("volume", 0))

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
            bracket_low, bracket_high, bracket_label = parse_bracket(m.get("question", ""))

            rows.append({
                "event_slug": slug,
                "event_start": event_start,
                "event_end": event_end,
                "event_volume": event_vol,
                "bracket_label": bracket_label,
                "bracket_low": bracket_low,
                "bracket_high": bracket_high,
                "price": yes_price,
                "condition_id": m.get("conditionId", ""),
                "token_id": tokens[0] if tokens else "",
                "market_volume": float(m.get("volume", 0)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["event_start"] = pd.to_datetime(df["event_start"], utc=True)
    df["event_end"] = pd.to_datetime(df["event_end"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["hours_remaining"] = (df["event_end"] - now).dt.total_seconds() / 3600
    df["event_closed"] = False

    grp = df.groupby("event_slug")
    df["leading_bracket_price"] = grp["price"].transform("max")
    df["distance_from_leading"] = df["leading_bracket_price"] - df["price"]
    df["bracket_mid"] = (df["bracket_low"].fillna(0) + df["bracket_high"].fillna(0)) / 2
    idx = df.groupby("event_slug")["price"].idxmax()
    leading_mid = df.loc[idx, ["event_slug", "bracket_mid"]].rename(columns={"bracket_mid": "leading_mid"})
    df = df.merge(leading_mid, on="event_slug", how="left")
    df["brackets_away"] = np.abs(df["bracket_mid"] - df["leading_mid"]) / 25
    df["in_sweet_spot"] = ((df["brackets_away"] >= 2.0) & (df["brackets_away"] <= 3.5)).astype(float)
    df["bracket_rank"] = grp["price"].transform("rank", ascending=False, method="min")
    df["bracket_count"] = grp["price"].transform("count")
    df["volume_rank"] = df.groupby("event_slug")["market_volume"].transform("rank", ascending=False, method="min")

    earliest = df["event_start"].min()
    df["days_since_start"] = (df["event_start"] - earliest).dt.total_seconds() / 86400
    df["recency_weight"] = np.exp(-0.00578 * (df["days_since_start"].max() - df["days_since_start"]))
    df["leading_bracket_price"] = grp["price"].transform("max")

    df = _compute_history_features(df)
    _append_prices_to_history(df)

    event_dur = (df["event_end"] - df["event_start"]).dt.total_seconds() / 3600
    event_dur = event_dur.replace(0, 1)
    df["hours_elapsed"] = (now - df["event_start"]).dt.total_seconds() / 3600
    df["pct_elapsed"] = np.clip(df["hours_elapsed"] / event_dur, 0, 1)
    df["distance_from_leading"] = df["leading_bracket_price"] - df["price"]

    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_or_train_model() -> dict:
    from pm_features import build_bracket_features, add_return_labels
    from pm_model import TRADE_FEATURES, _dedupe_cols

    if MODEL_PATH.exists():
        import pickle
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Train from historical data
    parquet = POLYMARKET_DIR / "bracket_prices.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Run polymarket data fetch first. Missing {parquet}")

    df = pd.read_parquet(parquet)
    featured = build_bracket_features(df)
    labeled = add_return_labels(featured)

    from pm_outcome import train_outcome_model
    info = train_outcome_model(labeled, test_days=120, brackets_away_min=1.5, brackets_away_max=6.0, verbose=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(info, f)
    return info


# ---------------------------------------------------------------------------
# Trading logic
# ---------------------------------------------------------------------------

def score_candidates(df: pd.DataFrame, model_info: dict, pred_min: float = None, proba_min: float = None) -> pd.DataFrame:
    candidates = df[
        (df["price"] < 0.02) & (df["price"] > 0.001) &
        (df["brackets_away"] >= 1.5) & (df["brackets_away"] <= 6.0) &
        (df["hours_remaining"] > 24)
    ].copy()

    if candidates.empty:
        return candidates

    features = model_info["feature_names"]
    model = model_info["model"]
    clf = model_info.get("jackpot_clf")

    for col in features:
        if col not in candidates.columns:
            candidates[col] = 0.0

    valid = candidates[features].fillna(0)
    preds = model.predict(valid)
    proba = clf.predict_proba(valid[features])[:, 1] if clf is not None else np.zeros(len(valid))

    candidates["_pred"] = preds
    candidates["_proba"] = proba
    pred_thresh = pred_min if pred_min is not None else PRED_THRESHOLD
    proba_thresh = proba_min if proba_min is not None else JACKPOT_PROBA_THRESHOLD
    passed = (candidates["_pred"] >= pred_thresh) & (candidates["_proba"] >= proba_thresh)
    return candidates[passed].sort_values("_pred", ascending=False)


def place_buy_order(client, token_id: str, price: float, usd_amount: float, tick_size: str = "0.01") -> dict | None:
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY

    size = int(usd_amount / price) if price > 0 else 0
    if size < 1:
        return None

    try:
        resp = client.create_and_post_order(
            OrderArgs(token_id=token_id, price=round(price, 2), size=size, side=BUY),
            options={"tick_size": tick_size, "neg_risk": True},
            order_type=OrderType.GTC,
        )
        return resp
    except Exception as e:
        print(f"  Order failed: {e}")
        return None


def place_sell_order(client, token_id: str, price: float, size: int, tick_size: str = "0.01") -> dict | None:
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import SELL

    if size < 1:
        return None

    try:
        resp = client.create_and_post_order(
            OrderArgs(token_id=token_id, price=round(price, 2), size=size, side=SELL),
            options={"tick_size": tick_size, "neg_risk": True},
            order_type=OrderType.GTC,
        )
        return resp
    except Exception as e:
        print(f"  Sell order failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_bot():
    config = load_config()
    dry_run = config["dry_run"]

    print(f"Loading model...")
    model_info = load_or_train_model()
    print(f"Model loaded. Features: {len(model_info['feature_names'])}")

    if not dry_run:
        client = get_clob_client()
        print("Connected to Polymarket CLOB")
        print(f"Holdings saved to: {HOLDINGS_PATH}")
    else:
        client = None
        print("DRY RUN - no real orders")
        print(f"Simulated holdings saved to: {DRY_RUN_HOLDINGS_PATH}")

    print(f"Config: bet=${config['bet_usd']}, max_positions={config['max_positions']}, poll={POLL_INTERVAL}s")
    holdings = load_holdings(dry_run=dry_run)
    holdings_path = DRY_RUN_HOLDINGS_PATH if dry_run else HOLDINGS_PATH
    if holdings:
        print(f"Loaded {len(holdings)} position(s) from {holdings_path}")
    print(f"Transactions: {TRANSACTIONS_PATH}")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            config = load_config()
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"[{now}] Checking...")

            events = fetch_active_events()
            if not events:
                print("  No active events")
                time.sleep(POLL_INTERVAL)
                continue

            df = fetch_current_prices(events)
            if df.empty:
                print("  No price data")
                time.sleep(POLL_INTERVAL)
                continue

            holdings = load_holdings(dry_run=dry_run)
            n_positions = len(set((h["event_slug"], h["bracket_label"]) for h in holdings))
            max_pos = config["max_positions"]
            bet_usd = config["bet_usd"]

            # --- SELL: check holdings for sell signals ---
            for h in holdings[:]:
                match = df[(df["event_slug"] == h["event_slug"]) & (df["bracket_label"] == h["bracket_label"])]
                if match.empty:
                    continue
                row = match.iloc[0]
                buy_price = h["buy_price"]
                current = row["price"]
                ret = current / buy_price if buy_price > 0 else 0
                hrs_left = row["hours_remaining"]

                should_sell = False
                reason = ""
                if ret >= SELL_TARGET_X:
                    should_sell = True
                    reason = f"target {ret:.1f}x"
                elif hrs_left < SELL_1DAY_HOURS:
                    should_sell = True
                    reason = f"1 day left ({ret:.1f}x)"

                if should_sell and not dry_run and client:
                    resp = place_sell_order(client, h["token_id"], current, int(h["shares"]))
                    if resp:
                        remove_holding(h["event_slug"], h["bracket_label"])
                        profit = (current - buy_price) * h["shares"]
                        log_transaction("sell", event_slug=h["event_slug"], bracket=h["bracket_label"],
                                        buy_price=buy_price, sell_price=current, shares=h["shares"],
                                        return_x=ret, profit_usd=profit)
                        _print_sell(h["event_slug"], h["bracket_label"], current, reason)

            # --- BUY: score and place orders ---
            cooldown_h = config.get("buy_cooldown_hours", BUY_COOLDOWN_HOURS)
            price_ratio = config.get("buy_again_price_ratio", BUY_AGAIN_PRICE_RATIO)
            if n_positions < max_pos:
                scored = score_candidates(df, model_info, config.get("pred_min"), config.get("proba_min"))
                if not scored.empty:
                    for _, row in scored.iterrows():
                        if n_positions >= max_pos:
                            break

                        slug, bracket, price = row["event_slug"], row["bracket_label"], row["price"]
                        existing = next((h for h in holdings if h["event_slug"] == slug and h["bracket_label"] == bracket), None)
                        if existing:
                            last_buy = existing.get("last_buy_time") or existing.get("buy_time", "")
                            if last_buy:
                                try:
                                    dt = datetime.fromisoformat(last_buy.replace("Z", "+00:00"))
                                    hrs = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                                    if hrs < cooldown_h:
                                        continue
                                    if price >= existing["buy_price"] / price_ratio:
                                        continue
                                except (ValueError, TypeError):
                                    pass

                        if dry_run:
                            shares = bet_usd / row["price"]
                            add_dry_holding(row["event_slug"], row["bracket_label"], row["price"], shares, row["token_id"])
                            _print_buy("DRY", row["event_slug"], row["bracket_label"], row["price"], pred=row["_pred"], usd=bet_usd)
                            n_positions += 1
                            continue

                        if client:
                            resp = place_buy_order(client, row["token_id"], row["price"], bet_usd)
                            if resp:
                                shares = bet_usd / row["price"]
                                add_holding(row["event_slug"], row["bracket_label"], row["price"], shares, row["token_id"], resp.get("orderID", ""))
                                log_transaction("buy", event_slug=row["event_slug"], bracket=row["bracket_label"],
                                                price=row["price"], shares=shares, usd=bet_usd, pred=row["_pred"])
                                _print_buy("BOUGHT", row["event_slug"], row["bracket_label"], row["price"], usd=bet_usd)
                                n_positions += 1
                                time.sleep(2)

            print()

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymarket auto-trading bot")
    parser.add_argument("--dry-run", action="store_true", help="No real orders")
    parser.add_argument("--bet", type=float, default=None, help="USD per trade")
    parser.add_argument("--holdings", action="store_true", help="Show holdings")
    parser.add_argument("--transactions", action="store_true", help="Show recent transactions")
    parser.add_argument("--correlations", action="store_true", help="Print feature correlations vs max_return_all")
    parser.add_argument("--train-only", action="store_true", help="Pre-train model and exit")
    args = parser.parse_args()

    if args.train_only:
        print("Training model...")
        load_or_train_model()
        print(f"Model saved to {MODEL_PATH}")
        return

    if args.holdings:
        dry = os.getenv("DRY_RUN", "false").lower() == "true" or args.dry_run
        holdings = load_holdings(dry_run=dry)
        path = DRY_RUN_HOLDINGS_PATH if dry else HOLDINGS_PATH
        print(f"Holdings ({path}):")
        for h in holdings:
            ev = _fmt_event(h["event_slug"])
            print(f"  {ev}  bracket {h['bracket_label']}  @ {h['buy_price']:.4f}  x{h['shares']:.0f}")
        return

    if args.transactions:
        if TRANSACTIONS_PATH.exists():
            for line in open(TRANSACTIONS_PATH).readlines()[-20:]:
                print(line.strip())
        return

    if args.correlations:
        parquet = POLYMARKET_DIR / "bracket_prices.parquet"
        if not parquet.exists():
            print(f"Missing {parquet}. Run polymarket data fetch first.")
            return
        from pm_features import build_bracket_features, add_return_labels, compute_bracket_correlations
        df = pd.read_parquet(parquet)
        featured = build_bracket_features(df)
        labeled = add_return_labels(featured)
        corr = compute_bracket_correlations(labeled)
        if corr.empty:
            print("No correlations (need buy candidates with max_return_all)")
            return
        print("Feature correlations vs max_return_all (buy candidates, price < 5%):")
        for _, row in corr.iterrows():
            print(f"  {row['feature']:25s}  r={row['pearson']:+.4f}")
        return

    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.bet is not None:
        os.environ["BET_USD"] = str(args.bet)

    run_bot()


if __name__ == "__main__":
    main()
