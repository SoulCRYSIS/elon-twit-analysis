"""Auto-trading bot for Polymarket Elon tweet brackets.

Runs 24/7, uses jackpot model (conf_p30) to buy, manual rules to sell.
- Configurable bet amount (default $1)
- Can buy same bracket multiple times if price changes and model approves
- Max 20 open positions (hard limit)

Data (persists across restarts):
  data/holdings.json       — open positions (loaded on startup)
  data/transactions.jsonl — buy/sell log (append-only)
  data/model_outcome.pkl  — trained model

Configuration: set environment variables (e.g. in Railway) or in a local `.env` file.
Railway injects env vars at runtime; `.env` is optional and does not override existing vars.

  PK or PRIVATE_KEY       - Wallet private key (required for live trading)
  DRY_RUN                 - "true" to disable real orders (PK optional if dry run)
  DATA_DIR                - Base data directory (default: ./data); use a volume path on Railway/Fly
  POLYMARKET_DATA_DIR     - Optional override for polymarket cache (default: DATA_DIR/polymarket)
  GAMMA_URL, CLOB_URL     - Polymarket APIs (defaults: gamma-api / clob hostnames)
  CHAIN_ID                - Polygon chain id (default 137)
  BET_USD                 - USD per trade (default 1)
  MAX_POSITIONS           - Max open positions (default 20)
  POLL_INTERVAL           - Seconds between loop iterations (default 3600)
  PRICE_HISTORY_HOURS     - In-memory price history window (default 48)
  PRED_MIN                - Min model pred for buys (default 6)
  JACKPOT_PROBA_MIN       - Min jackpot probability (default 0.01)
  BUY_PRICE_MAX, BUY_PRICE_MIN - Candidate price band (default 0.02 / 0.001)
  BRACKETS_AWAY_MIN, BRACKETS_AWAY_MAX (default 1.5 / 6.0)
  BUY_MIN_HOURS_REMAINING - Min hours left on event to buy (default 24)
  SELL_TARGET_X           - Take-profit multiple vs buy price (default 4.5)
  SELL_1DAY_HOURS         - Force sell if less than this many hours left (default 24)
  BUY_COOLDOWN_HOURS      - Hours before adding to same bracket again (default 4)
  BUY_AGAIN_PRICE_RATIO   - Only add when price < last_buy / this (default 2)
  DRY_RUN_START_BALANCE   - Simulated starting $ for dry run (default 100)
  RETRAIN_HOURS           - Hours between data fetch + retrain (default 72; 0=disabled)
  EVENT_SEARCH_QUERY      - Gamma public-search query (default: elon musk tweets)
  EVENT_SEARCH_LIMIT      - limit_per_type (default 50)
  EVENTS_STATUS           - Gamma events_status filter (default active; empty to omit)
  EVENT_FETCH_SLEEP_SEC   - Pause between event detail fetches (default 0.2)
  CLOB_HTTP_TIMEOUT       - Seconds for CLOB HTTP requests (default 10)
  BUY_MARKET_MAX_PRICE_DIFF - If CLOB BUY price vs snapshot price differs by at most this (0–1 scale), submit a
                              market (FOK) buy so it fills; else a limit order (default 0.1)
  BUY_MARKET_MIN_USD      - Minimum USD notional for market buys (default 1.01; API rejects under about $1)
  POLYMARKET_SIGNATURE_TYPE - CLOB signing: 0=EOA (MetaMask key is the trading wallet, no proxy). 1=POLY_PROXY
                              (Magic / email login — PK exported from Polymarket). 2=POLY_GNOSIS_SAFE (most browser
                              wallets: MetaMask/Rabby connected to Polymarket — use this if you did NOT sign up with
                              email alone). Wrong type → API returns invalid_signature even with the right funder.
  POLYMARKET_FUNDER       - 0x proxy wallet that holds your Polymarket balance (NOT the relayer, NOT a random
                            contract). Polymarket docs: the address shown in the site UI / profile as *your* wallet
                            is usually this proxy. Must pair with the PK Polymarket gave you for that account.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.resolve()
load_dotenv(ROOT / ".env")


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return default
    return str(v).strip()


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return default
    return int(v)


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or not str(v).strip():
        return default
    return float(v)


# Paths (under DATA_DIR; set DATA_DIR for Railway volumes)
_data_env = os.getenv("DATA_DIR", "").strip()
DATA_DIR = Path(_data_env).expanduser().resolve() if _data_env else ROOT / "data"
POLYMARKET_DIR = DATA_DIR / "polymarket"
# So polymarket_data (imported later) uses the same root as this module, not /app/data/polymarket from the image.
os.environ.setdefault("POLYMARKET_DATA_DIR", str(POLYMARKET_DIR))
HOLDINGS_PATH = DATA_DIR / "holdings.json"  # Positions — loaded on bot restart (real mode only)
DRY_RUN_HOLDINGS_PATH = DATA_DIR / "dry_run_holdings.json"  # Simulated positions when --dry-run
DRY_RUN_STATE_PATH = DATA_DIR / "dry_run_state.json"  # balance, start_balance (persists across restarts)
DRY_RUN_TRANSACTIONS_PATH = DATA_DIR / "dry_run_transactions.jsonl"  # Simulated buy/sell log
TRANSACTIONS_PATH = DATA_DIR / "transactions.jsonl"  # Buy/sell log
MODEL_PATH = DATA_DIR / "model_outcome.pkl"

GAMMA_URL = _env_str("GAMMA_URL", "https://gamma-api.polymarket.com").rstrip("/")
CLOB_URL = _env_str("CLOB_URL", "https://clob.polymarket.com").rstrip("/")
CHAIN_ID = _env_int("CHAIN_ID", 137)

MAX_OPEN_POSITIONS = 20
DEFAULT_BET_USD = 1.0
PRED_THRESHOLD = 6.0  # Min predicted jackpot potential; set PRED_MIN in env
JACKPOT_PROBA_THRESHOLD = 0.01  # Min P(jackpot); classifier is conservative on live (no history)
BUY_COOLDOWN_HOURS = 4  # Don't add to same bracket within this many hours
BUY_AGAIN_PRICE_RATIO = 2.0  # Only add when price < last_buy_price / this (averaging down)
DRY_RUN_START_BALANCE = 100.0  # Simulated starting balance for performance testing
RETRAIN_HOURS = 72.0  # Hours between data fetch + retrain; 0 = disabled

EVENT_SEARCH_QUERY = _env_str("EVENT_SEARCH_QUERY", "elon musk tweets")
EVENT_SEARCH_LIMIT = _env_int("EVENT_SEARCH_LIMIT", 50)
EVENTS_STATUS = _env_str("EVENTS_STATUS", "active")
EVENT_FETCH_SLEEP_SEC = _env_float("EVENT_FETCH_SLEEP_SEC", 0.2)
CLOB_HTTP_TIMEOUT = _env_float("CLOB_HTTP_TIMEOUT", 10.0)

RETRAIN_STATE_PATH = DATA_DIR / "retrain_state.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    dry = os.getenv("DRY_RUN", "false").lower() == "true"
    pk = (os.getenv("PK") or os.getenv("PRIVATE_KEY") or "").strip()
    if not dry and not pk:
        raise ValueError("Missing PK or PRIVATE_KEY (required unless DRY_RUN=true)")
    return {
        "private_key": pk,
        "bet_usd": _env_float("BET_USD", DEFAULT_BET_USD),
        "max_positions": _env_int("MAX_POSITIONS", MAX_OPEN_POSITIONS),
        "dry_run": dry,
        "pred_min": _env_float("PRED_MIN", PRED_THRESHOLD),
        "proba_min": _env_float("JACKPOT_PROBA_MIN", JACKPOT_PROBA_THRESHOLD),
        "buy_cooldown_hours": _env_float("BUY_COOLDOWN_HOURS", BUY_COOLDOWN_HOURS),
        "buy_again_price_ratio": _env_float("BUY_AGAIN_PRICE_RATIO", BUY_AGAIN_PRICE_RATIO),
        "dry_run_start_balance": _env_float("DRY_RUN_START_BALANCE", DRY_RUN_START_BALANCE),
        "poll_interval": _env_int("POLL_INTERVAL", 3600),
        "price_history_hours": _env_int("PRICE_HISTORY_HOURS", 48),
        "sell_target_x": _env_float("SELL_TARGET_X", 4.5),
        "sell_1day_hours": _env_float("SELL_1DAY_HOURS", 24.0),
        "buy_price_max": _env_float("BUY_PRICE_MAX", 0.02),
        "buy_price_min": _env_float("BUY_PRICE_MIN", 0.001),
        "brackets_away_min": _env_float("BRACKETS_AWAY_MIN", 1.5),
        "brackets_away_max": _env_float("BRACKETS_AWAY_MAX", 6.0),
        "buy_min_hours_remaining": _env_float("BUY_MIN_HOURS_REMAINING", 24.0),
        "buy_market_max_price_diff": _env_float("BUY_MARKET_MAX_PRICE_DIFF", 0.1),
        "buy_market_min_usd": _env_float("BUY_MARKET_MIN_USD", 1.01),
    }


# ---------------------------------------------------------------------------
# Polymarket client
# ---------------------------------------------------------------------------

def _clob_signature_and_funder() -> tuple[int | None, str | None]:
    """Parse POLYMARKET_SIGNATURE_TYPE and POLYMARKET_FUNDER for ClobClient.

    Polymarket email / Magic Link accounts usually hold trading USDC in a **proxy wallet** (POLY_PROXY).
    The bot's default is EOA-only: orders attribute collateral to the signer address, which can show $0
    on the CLOB even when the UI shows a balance under the proxy.
    """
    from py_order_utils.model import EOA, POLY_GNOSIS_SAFE, POLY_PROXY

    raw = os.getenv("POLYMARKET_SIGNATURE_TYPE", "").strip()
    funder_env = os.getenv("POLYMARKET_FUNDER", "").strip() or None

    if not raw:
        if funder_env:
            print(
                "  Warning: POLYMARKET_FUNDER is set but POLYMARKET_SIGNATURE_TYPE is not; funder is ignored for EOA. "
                "Set POLYMARKET_SIGNATURE_TYPE=1 (or 2) if your balance is in a Polymarket proxy/safe.",
                flush=True,
            )
        return None, None

    upper = raw.upper()
    if upper in ("0", "EOA"):
        sig = EOA
    elif upper in ("1", "POLY_PROXY", "PROXY"):
        sig = POLY_PROXY
    elif upper in ("2", "POLY_GNOSIS_SAFE", "GNOSIS", "SAFE"):
        sig = POLY_GNOSIS_SAFE
    else:
        try:
            sig = int(raw)
        except ValueError as e:
            raise ValueError(
                f"Invalid POLYMARKET_SIGNATURE_TYPE={raw!r}; use 0, EOA, 1, POLY_PROXY, 2, or POLY_GNOSIS_SAFE"
            ) from e
        if sig not in (EOA, POLY_PROXY, POLY_GNOSIS_SAFE):
            raise ValueError(f"Invalid POLYMARKET_SIGNATURE_TYPE={sig}; must be 0, 1, or 2")

    if sig in (POLY_PROXY, POLY_GNOSIS_SAFE):
        if not funder_env:
            raise ValueError(
                "POLYMARKET_FUNDER is required when POLYMARKET_SIGNATURE_TYPE is 1 (POLY_PROXY) or 2 "
                "(POLY_GNOSIS_SAFE). In the Polymarket UI, copy the wallet/proxy address that holds your balance "
                "(often shown under profile or deposit; it may differ from your MetaMask EOA)."
            )
        return sig, funder_env

    if funder_env:
        print(
            "  Warning: POLYMARKET_FUNDER is ignored when using EOA (type 0); collateral is attributed to the signer only.",
            flush=True,
        )
    return sig, None


def get_clob_client() -> "ClobClient":
    from py_clob_client.client import ClobClient

    config = load_config()
    sig_type, funder = _clob_signature_and_funder()
    client = ClobClient(
        CLOB_URL,
        key=config["private_key"],
        chain_id=CHAIN_ID,
        signature_type=sig_type,
        funder=funder,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    b = client.builder
    print(
        f"  CLOB: signer={client.get_address()}  signature_type={b.sig_type}  funder={b.funder}",
        flush=True,
    )
    if b.sig_type != 0:
        print(
            "  Hint: funder = Polymarket “your wallet” 0x from the site (proxy). If orders fail invalid_signature, "
            "try POLYMARKET_SIGNATURE_TYPE=2 for MetaMask/Rabby, or 1 for Magic/email export only.",
            flush=True,
        )
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


def load_dry_run_state(start_balance: float = DRY_RUN_START_BALANCE) -> dict:
    """Load or init dry run state. Returns {balance, start_balance}."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DRY_RUN_STATE_PATH.exists():
        with open(DRY_RUN_STATE_PATH) as f:
            return json.load(f)
    state = {"balance": start_balance, "start_balance": start_balance}
    with open(DRY_RUN_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    return state


def save_dry_run_state(state: dict):
    with open(DRY_RUN_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def log_dry_transaction(action: str, **kwargs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(), "action": action, **kwargs}
    with open(DRY_RUN_TRANSACTIONS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def remove_dry_holding(slug: str, bracket: str) -> dict | None:
    """Remove from dry_run_holdings. Returns the removed holding or None."""
    if not DRY_RUN_HOLDINGS_PATH.exists():
        return None
    with open(DRY_RUN_HOLDINGS_PATH) as f:
        holdings = json.load(f)
    removed = next((h for h in holdings if h["event_slug"] == slug and h["bracket_label"] == bracket), None)
    if not removed:
        return None
    holdings = [h for h in holdings if not (h["event_slug"] == slug and h["bracket_label"] == bracket)]
    with open(DRY_RUN_HOLDINGS_PATH, "w") as f:
        json.dump(holdings, f, indent=2)
    return removed


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


def _print_buy(
    tag: str,
    event_slug: str,
    bracket: str,
    price: float,
    pred: float = None,
    usd: float = None,
    suffix: str = "",
):
    ev = _fmt_event(event_slug)
    parts = [f"  [{tag}]", ev, f"bracket {bracket}", f"@ {price:.4f}"]
    if pred is not None:
        parts.append(f"pred={pred:.1f}x")
    if usd is not None:
        parts.append(f"${usd}")
    if suffix:
        parts.append(suffix.strip())
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


def _append_prices_to_history(df: pd.DataFrame, max_hours: int):
    """Append current prices to in-memory history. Prune to max_hours."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    cutoff = now_ts - max_hours * 3600
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


PRICES_CACHE_DIR = POLYMARKET_DIR / "prices"


def _load_history_from_cache_or_api(condition_id: str, token_id: str) -> tuple[list[tuple[int, float]], bool]:
    """Load price history. Returns (data, used_cache). Use cache file if exists; else fetch from CLOB API."""
    cache_path = PRICES_CACHE_DIR / f"{condition_id}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                pts = json.load(f)
            if pts:
                data = [(p["t"], p["p"]) for p in pts if isinstance(p, dict) and "t" in p and "p" in p]
                if data:
                    return (data, True)
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: fetch from API
    now_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = now_ts - 24 * 3600
    try:
        resp = requests.get(
            f"{CLOB_URL}/prices-history",
            params={"market": token_id, "startTs": start_ts, "endTs": now_ts, "fidelity": 60},
            headers={"Accept": "application/json"},
            timeout=CLOB_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        pts = resp.json().get("history", [])
        return ([(p["t"], p["p"]) for p in pts if "t" in p and "p" in p], False)
    except Exception:
        return ([], False)


def _compute_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price_change_*, price_vol_*, price_pct_range from in-memory history."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    for col in ["price_change_1h", "price_change_3h", "price_change_6h", "price_change_12h", "price_change_24h",
                "price_vol_6h", "price_vol_12h", "price_vol_24h", "price_pct_range", "hhi", "entropy"]:
        if col not in df.columns:
            df[col] = 0.0

    n_rows = len(df)
    if n_rows > 50 and len(_price_history) < 10:
        print(f"  Computing history features for {n_rows} brackets (first run, may take 1–2 min)...", flush=True)

    # Pre-load all histories from cache/API (avoids slow per-row df.at[] in loop)
    step = max(1, n_rows // 10)
    for idx, row in enumerate(df.itertuples(index=False)):
        # if n_rows > 50 and (idx + 1) % step == 0:
        #     print(f"    ... {idx + 1}/{n_rows} brackets", flush=True)
        cid = getattr(row, "condition_id", "") or ""
        token_id = getattr(row, "token_id", "") or ""
        if not cid or not token_id:
            continue
        if cid not in _price_history:
            backfill, used_cache = _load_history_from_cache_or_api(cid, token_id)
            if backfill:
                _price_history[cid] = sorted(backfill, key=lambda x: x[0])
            if not used_cache:
                time.sleep(0.05)

    # Compute features in bulk (much faster than df.at[] per cell)
    price_changes = {f"price_change_{h}h": [] for h in [1, 3, 6, 12, 24]}
    price_vols = {f"price_vol_{w}h": [] for w in [6, 12, 24]}
    price_pct_ranges = []

    for row in df.itertuples(index=False):
        cid = getattr(row, "condition_id", "") or ""
        price = float(getattr(row, "price", 0))
        hist = _price_history.get(cid) if cid else None

        if not hist:
            for k in price_changes:
                price_changes[k].append(0.0)
            for k in price_vols:
                price_vols[k].append(0.0)
            price_pct_ranges.append(0.0)
            continue

        prices = [p for _, p in hist]
        ts_list = [t for t, _ in hist]

        for lag_h in [1, 3, 6, 12, 24]:
            target_ts = now_ts - lag_h * 3600
            closest = min(ts_list, key=lambda t: abs(t - target_ts))
            if abs(closest - target_ts) <= 2 * 3600:
                idx = ts_list.index(closest)
                price_changes[f"price_change_{lag_h}h"].append(price - prices[idx])
            else:
                price_changes[f"price_change_{lag_h}h"].append(0.0)

        for win in [6, 12, 24]:
            recent = [(t, p) for t, p in hist if t >= now_ts - win * 3600]
            if len(recent) >= 2:
                vals = [p for _, p in recent]
                price_vols[f"price_vol_{win}h"].append(float(np.std(vals)))
            else:
                price_vols[f"price_vol_{win}h"].append(0.0)

        pmin, pmax = min(prices), max(prices)
        if pmax > pmin:
            price_pct_ranges.append((price - pmin) / (pmax - pmin))
        else:
            price_pct_ranges.append(0.0)

    for col, vals in price_changes.items():
        df[col] = vals
    for col, vals in price_vols.items():
        df[col] = vals
    df["price_pct_range"] = price_pct_ranges

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
    from polymarket_data import filter_weekly_events

    params: dict = {
        "q": EVENT_SEARCH_QUERY,
        "limit_per_type": EVENT_SEARCH_LIMIT,
    }
    if EVENTS_STATUS:
        params["events_status"] = EVENTS_STATUS
    resp = requests.get(f"{GAMMA_URL}/public-search", params=params)
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
        time.sleep(EVENT_FETCH_SLEEP_SEC)
    return filter_weekly_events(events)


def fetch_current_prices(events: list[dict], *, price_history_hours: int = 48) -> pd.DataFrame:
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
    _append_prices_to_history(df, price_history_hours)

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
    """
    Load existing outcome model if present; otherwise automatically fetch data and train.

    This makes the bot self-contained: on first run it will download Polymarket
    data, build features, train the model, and persist it to MODEL_PATH.
    """
    from polymarket_data import get_polymarket_data
    from pm_features import build_bracket_features, add_return_labels
    from pm_outcome import train_outcome_model

    if MODEL_PATH.exists():
        import pickle
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    parquet = POLYMARKET_DIR / "bracket_prices.parquet"

    if parquet.exists():
        df = pd.read_parquet(parquet)
        # Filter to weekly events only (exclude 2-day with limited history)
        df["event_start"] = pd.to_datetime(df["event_start"], utc=True)
        df["event_end"] = pd.to_datetime(df["event_end"], utc=True)
        dur_hours = (df["event_end"] - df["event_start"]).dt.total_seconds() / 3600
        df = df[dur_hours > 72].copy()
        if df.empty:
            print("Parquet had only 2-day events; fetching fresh data...")
            df, _ = get_polymarket_data(incremental=False, force_refresh=False, verbose=True)
            if df.empty:
                raise RuntimeError("No weekly events found; cannot train model.")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("No trained model found; fetching Polymarket data and training from scratch...")
        df, _ = get_polymarket_data(incremental=False, force_refresh=False, verbose=True)
        if df.empty:
            raise RuntimeError(
                f"Polymarket data fetch returned no rows; cannot train model. "
                f"Check network access and try again."
            )

    featured = build_bracket_features(df)
    labeled = add_return_labels(featured)

    info = train_outcome_model(
        labeled,
        test_frac=0.2,
        brackets_away_min=1.5,
        brackets_away_max=6.0,
        verbose=True,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(info, f)
    return info


def _get_last_retrain_time() -> datetime | None:
    """Return last retrain timestamp or None if never."""
    if not RETRAIN_STATE_PATH.exists():
        return None
    try:
        with open(RETRAIN_STATE_PATH) as f:
            data = json.load(f)
        s = data.get("last_retrain")
        if s:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return None


def _save_retrain_time(ts: datetime):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_STATE_PATH, "w") as f:
        json.dump({"last_retrain": ts.isoformat()}, f, indent=2)


def refresh_data_and_retrain() -> dict | None:
    """Fetch new Polymarket data (incremental) and retrain the model. Returns new model_info or None on failure."""
    from polymarket_data import get_polymarket_data
    from pm_features import build_bracket_features, add_return_labels
    from pm_outcome import train_outcome_model

    try:
        # Use incremental: only fetch new events + append new price data (fast)
        parquet = POLYMARKET_DIR / "bracket_prices.parquet"
        incremental = parquet.exists()  # We have prior data, do incremental
        print("  Fetching Polymarket data (incremental)..." if incremental else "  Fetching Polymarket data (full)...", flush=True)
        df, _ = get_polymarket_data(incremental=incremental, force_refresh=False, verbose=True)
        if df.empty:
            print("  Retrain skipped: no data")
            return None

        print("  Building features and retraining...")
        featured = build_bracket_features(df)
        labeled = add_return_labels(featured)
        info = train_outcome_model(labeled, test_frac=0.2, brackets_away_min=1.5, brackets_away_max=6.0, verbose=True)

        import pickle
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(info, f)
        _save_retrain_time(datetime.now(timezone.utc))
        print(f"  Model retrained. Features: {len(info['feature_names'])}")
        return info
    except Exception as e:
        print(f"  Retrain failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def should_retrain() -> bool:
    """True if it's time to run data fetch + retrain."""
    hours = float(os.getenv("RETRAIN_HOURS", RETRAIN_HOURS))
    if hours <= 0:
        return False
    last = _get_last_retrain_time()
    if last is None:
        return True  # Never retrained, do it on first opportunity
    elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    return elapsed >= hours


# ---------------------------------------------------------------------------
# Trading logic
# ---------------------------------------------------------------------------

def score_candidates(
    df: pd.DataFrame,
    model_info: dict,
    pred_min: float = None,
    proba_min: float = None,
    *,
    filters: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score brackets for buy; return those passing pred and proba thresholds."""
    f = {
        "buy_price_max": 0.02,
        "buy_price_min": 0.001,
        "brackets_away_min": 1.5,
        "brackets_away_max": 6.0,
        "buy_min_hours_remaining": 24.0,
    }
    if filters:
        f.update({k: filters[k] for k in f if k in filters})
    mask = (
        (df["price"] < f["buy_price_max"]) & (df["price"] > f["buy_price_min"]) &
        (df["brackets_away"] >= f["brackets_away_min"]) & (df["brackets_away"] <= f["brackets_away_max"]) &
        (df["hours_remaining"] > f["buy_min_hours_remaining"])
    )
    candidates = df[mask].copy()

    if candidates.empty:
        if verbose:
            n_total = len(df)
            n_price = ((df["price"] < f["buy_price_max"]) & (df["price"] > f["buy_price_min"])).sum()
            n_brackets = (
                (df["brackets_away"] >= f["brackets_away_min"]) & (df["brackets_away"] <= f["brackets_away_max"])
            ).sum()
            n_hrs = (df["hours_remaining"] > f["buy_min_hours_remaining"]).sum()
            print(
                f"  No pre-filter candidates: {n_total} brackets total, {n_price} in price range, "
                f"{n_brackets} in bracket range, {n_hrs} with >{f['buy_min_hours_remaining']}h left",
                flush=True,
            )
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
    result = candidates[passed].sort_values("_pred", ascending=False)

    # Fallback: classifier can be overly conservative on live (sparse history). If pred passes but proba blocks
    # everyone, use pred-only for candidates with pred >= threshold.
    if result.empty:
        pred_only = candidates[candidates["_pred"] >= pred_thresh].sort_values("_pred", ascending=False)
        if not pred_only.empty and verbose:
            proba_max = float(candidates["_proba"].max())
            print(f"  Using pred-only fallback ({len(pred_only)} candidates): proba max={proba_max:.3f} < {proba_thresh}", flush=True)
        result = pred_only

    if result.empty and verbose:
        pred_max, pred_mean = float(candidates["_pred"].max()), float(candidates["_pred"].mean())
        proba_max, proba_mean = float(candidates["_proba"].max()), float(candidates["_proba"].mean())
        n_pred_ok = (candidates["_pred"] >= pred_thresh).sum()
        n_proba_ok = (candidates["_proba"] >= proba_thresh).sum()
        print(f"  {len(candidates)} pre-filter candidates, 0 pass: pred>={pred_thresh} (max={pred_max:.1f}, "
              f"mean={pred_mean:.1f}, {n_pred_ok} pass) & proba>={proba_thresh} (max={proba_max:.3f}, "
              f"mean={proba_mean:.3f}, {n_proba_ok} pass)", flush=True)
    return result


def _order_limit_price(price: float) -> float:
    """Limit price for CLOB orders: never use round(x, 2) alone — sub-$0.01 brackets become 0.0."""
    if not math.isfinite(price) or price <= 0:
        return 0.0
    d = Decimal(str(price)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    return float(d)


def _print_polymarket_order_error(exc: BaseException, *, sell: bool = False) -> None:
    """Explain common API failures (e.g. CLOB geoblock on cloud IPs)."""
    prefix = "  Sell order failed" if sell else "  Order failed"
    print(f"{prefix}: {exc}")
    try:
        from py_clob_client.exceptions import PolyApiException

        if not isinstance(exc, PolyApiException):
            return
        blob = str(exc.error_msg).lower()
        if exc.status_code == 403 and (
            "geoblock" in blob or "restricted in your region" in blob or "trading restricted" in blob
        ):
            print(
                "  Note: Polymarket geoblocks by detected location/IP. Hosting in a US Railway region does not "
                "guarantee a US-residential or allowlisted egress IP; many cloud ranges are blocked. "
                "Options: run the bot from a home/office network in an allowed region, or use infrastructure "
                "Polymarket supports per https://docs.polymarket.com/developers/CLOB/geoblock — do not rely on "
                "VPNs to evade restrictions if that violates their terms.",
                flush=True,
            )
        if exc.status_code == 400 and (
            "not enough balance" in blob or "allowance" in blob or "balance is not enough" in blob
        ):
            print(
                "  Note: Usually not a wrong API key. The CLOB checks collateral for the configured **funder** "
                "(default: your EOA from PK). Polymarket often holds your $ balance in a **proxy or Gnosis safe** "
                "while the UI still shows one account — if PK is only the EOA, CLOB can see $0. Set "
                "POLYMARKET_SIGNATURE_TYPE=1 (POLY_PROXY) or 2 (POLY_GNOSIS_SAFE) and POLYMARKET_FUNDER=0x… "
                "to that proxy/safe address from the Polymarket UI. Also ensure USDC is deposited for trading and "
                "approvals are done. Error amounts use token decimals (often ÷1e6 for USDC).",
                flush=True,
            )
        if exc.status_code == 400 and "invalid signature" in blob:
            print(
                "  Note: invalid_signature usually means POLYMARKET_SIGNATURE_TYPE does not match how you use "
                "Polymarket: try **2** (POLY_GNOSIS_SAFE) if you connect with MetaMask/Rabby/browser wallet; try **1** "
                "(POLY_PROXY) only for Magic/email-exported keys. Funder must be your **Polymarket proxy** (the 0x "
                "shown as your wallet on the site — not a “relayer” address). PK must be the one Polymarket exports "
                "for that same account. See https://github.com/Polymarket/py-clob-client#start-trading-proxy-wallet",
                flush=True,
            )
        if exc.status_code == 400 and "invalid amount" in blob and "marketable" in blob:
            print(
                "  Note: Market buys need notional >= ~$1. Raise BET_USD or BUY_MARKET_MIN_USD (default 1.01).",
                flush=True,
            )
    except Exception:
        pass


def _clob_buy_side_price(client, token_id: str) -> float | None:
    """CLOB executable BUY price for token (for comparing to Gamma snapshot)."""
    try:
        r = client.get_price(token_id, side="BUY")
        if isinstance(r, dict):
            v = r.get("price")
            if v is None:
                return None
            return float(v)
        return float(r)
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def _merge_buy_resp(resp: dict | list | str | None, **extra) -> dict | None:
    if resp is None:
        return None
    out = dict(resp) if isinstance(resp, dict) else {"raw": resp}
    out.update(extra)
    return out


def place_buy_order(
    client,
    token_id: str,
    price: float,
    usd_amount: float,
    tick_size: str | None = None,
    *,
    market_max_price_diff: float = 0.1,
    market_min_usd: float = 1.01,
) -> dict | None:
    """Place a market (FOK) buy if CLOB BUY price is close to `price`; else limit order.

    Market path uses FOK so the order either fills immediately or fails (no resting bid).
    Polymarket enforces a minimum ~$1 notional on marketable buys; `market_min_usd` enforces a safe floor.
    """
    from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType, PartialCreateOrderOptions
    from py_clob_client.order_builder.constants import BUY

    limit_px = _order_limit_price(price)
    if limit_px <= 0:
        return None

    live_buy = _clob_buy_side_price(client, token_id)
    use_market = (
        live_buy is not None
        and abs(live_buy - limit_px) <= market_max_price_diff
    )

    opts = PartialCreateOrderOptions(tick_size=None if tick_size is None else str(tick_size), neg_risk=None)

    if use_market:
        notional = max(float(usd_amount), float(market_min_usd))
        try:
            mo = MarketOrderArgs(
                token_id=token_id,
                amount=notional,
                side=BUY,
                order_type=OrderType.FOK,
            )
            signed = client.create_market_order(mo, opts)
            resp = client.post_order(signed, OrderType.FOK)
            fill_est = live_buy if live_buy is not None else limit_px
            return _merge_buy_resp(
                resp,
                _buy_mode="market",
                _fill_price_est=float(fill_est),
                _usd=float(notional),
            )
        except Exception as e:
            _print_polymarket_order_error(e, sell=False)
            return None

    size = int(usd_amount / limit_px) if limit_px > 0 else 0
    if size < 1:
        return None
    try:
        resp = client.create_and_post_order(
            OrderArgs(token_id=token_id, price=limit_px, size=float(size), side=BUY),
            opts,
        )
        return _merge_buy_resp(
            resp,
            _buy_mode="limit",
            _fill_price_est=float(limit_px),
            _usd=float(usd_amount),
        )
    except Exception as e:
        _print_polymarket_order_error(e, sell=False)
        return None


def _clob_conditional_balance_shares(client, token_id: str) -> float | None:
    """Return spendable outcome-token balance in **shares** (human scale), or None if unknown."""
    from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

    try:
        r = client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=str(token_id),
            )
        )
    except Exception:
        return None
    if not isinstance(r, dict):
        return None
    raw_bal = r.get("balance")
    if raw_bal is None:
        return None
    try:
        micro = int(str(raw_bal))
    except (TypeError, ValueError):
        return None
    # Polymarket / CLOB: conditional token amounts use 1e6 micro-units (same scale as API errors).
    return micro / 1_000_000.0


def _sell_shares_capped(
    client, token_id: str, recorded_shares: float
) -> tuple[float, str | None]:
    """Use min(holdings, on-chain balance). Holdings often exceed wallet after partial fills or drift."""
    cap = _clob_conditional_balance_shares(client, token_id)
    if cap is None:
        return float(recorded_shares), None
    sell = min(float(recorded_shares), max(0.0, cap))
    note = None
    if cap + 1e-9 < float(recorded_shares):
        note = (
            f"Sell size capped: holdings {float(recorded_shares):.6f} shares → CLOB balance {cap:.6f}"
        )
    return sell, note


def place_sell_order(
    client, token_id: str, price: float, size: float, tick_size: str | None = None
) -> dict | None:
    from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
    from py_clob_client.order_builder.constants import SELL

    if size <= 0:
        return None

    limit_px = _order_limit_price(price)
    if limit_px <= 0:
        return None

    try:
        tick = None if tick_size is None else str(tick_size)
        opts = PartialCreateOrderOptions(tick_size=tick, neg_risk=None)
        resp = client.create_and_post_order(
            OrderArgs(token_id=token_id, price=limit_px, size=float(size), side=SELL),
            opts,
        )
        return resp
    except Exception as e:
        _print_polymarket_order_error(e, sell=True)
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
        print("DRY RUN - no real orders (runs exactly like normal: buys, sells, balance)")
        state = load_dry_run_state(config.get("dry_run_start_balance", DRY_RUN_START_BALANCE))
        print(f"Balance: ${state['balance']:.2f}  (start: ${state['start_balance']:.2f})")
        print(f"State: {DRY_RUN_STATE_PATH}")
        print(f"Holdings: {DRY_RUN_HOLDINGS_PATH}")
        print(f"Transactions: {DRY_RUN_TRANSACTIONS_PATH}")

    retrain_h = float(os.getenv("RETRAIN_HOURS", RETRAIN_HOURS))
    print(
        f"Config: bet=${config['bet_usd']}, max_positions={config['max_positions']}, "
        f"poll={config['poll_interval']}s, data_dir={DATA_DIR}, polymarket_cache={POLYMARKET_DIR}, "
        f"retrain_every={retrain_h}h"
    )
    holdings = load_holdings(dry_run=dry_run)
    holdings_path = DRY_RUN_HOLDINGS_PATH if dry_run else HOLDINGS_PATH
    print(f"Holdings ({holdings_path}):")
    for h in holdings:
        ev = _fmt_event(h["event_slug"])
        print(f"  {ev}  bracket {h['bracket_label']}  @ {h['buy_price']:.4f}  x{h['shares']:.0f}")
    if not holdings:
        print("  (none)")
    if dry_run:
        state = load_dry_run_state()
        print(f"Balance: ${state['balance']:.2f}  (start: ${state['start_balance']:.2f})")
    print(f"Transactions: {TRANSACTIONS_PATH}")
    print("Press Ctrl+C to stop.\n")

    while True:
        poll_sec = config.get("poll_interval", 3600)
        try:
            config = load_config()
            poll_sec = config["poll_interval"]
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"[{now}] Checking...")

            # Periodic data fetch + retrain
            if should_retrain():
                new_info = refresh_data_and_retrain()
                if new_info is not None:
                    model_info = new_info

            print("  Fetching active events...", flush=True)
            events = fetch_active_events()
            if not events:
                print("  No active events")
                time.sleep(poll_sec)
                continue

            print(f"  Found {len(events)} active events, fetching prices...", flush=True)
            df = fetch_current_prices(events, price_history_hours=config["price_history_hours"])
            if df.empty:
                print("  No price data")
                time.sleep(poll_sec)
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
                if ret >= config["sell_target_x"]:
                    should_sell = True
                    reason = f"target {ret:.1f}x"
                elif hrs_left < config["sell_1day_hours"]:
                    should_sell = True
                    reason = f"1 day left ({ret:.1f}x)"

                if should_sell:
                    if dry_run:
                        removed = remove_dry_holding(h["event_slug"], h["bracket_label"])
                        if removed:
                            proceeds = current * h["shares"]
                            profit = (current - buy_price) * h["shares"]
                            state = load_dry_run_state()
                            state["balance"] = state["balance"] + proceeds
                            save_dry_run_state(state)
                            log_dry_transaction("sell", event_slug=h["event_slug"], bracket=h["bracket_label"],
                                               buy_price=buy_price, sell_price=current, shares=h["shares"],
                                               return_x=ret, profit_usd=profit, proceeds=proceeds, balance=state["balance"])
                            _print_sell(h["event_slug"], h["bracket_label"], current, reason)
                            print(f"      +${proceeds:.2f}  balance=${state['balance']:.2f}")
                    elif client:
                        recorded = float(h["shares"])
                        sell_shares, cap_note = _sell_shares_capped(client, h["token_id"], recorded)
                        if cap_note:
                            print(f"  {cap_note}", flush=True)
                        if sell_shares <= 0:
                            print(
                                "  Skip sell: on-chain conditional balance is 0 (update or clear stale holdings)",
                                flush=True,
                            )
                        else:
                            resp = place_sell_order(client, h["token_id"], current, sell_shares)
                            if resp:
                                remove_holding(h["event_slug"], h["bracket_label"])
                                profit = (current - buy_price) * sell_shares
                                log_transaction(
                                    "sell",
                                    event_slug=h["event_slug"],
                                    bracket=h["bracket_label"],
                                    buy_price=buy_price,
                                    sell_price=current,
                                    shares=sell_shares,
                                    shares_recorded=recorded,
                                    return_x=ret,
                                    profit_usd=profit,
                                )
                                _print_sell(h["event_slug"], h["bracket_label"], current, reason)

            if dry_run:
                holdings = load_holdings(dry_run=True)
                n_positions = len(set((h["event_slug"], h["bracket_label"]) for h in holdings))

            # --- BUY: score and place orders ---
            cooldown_h = config.get("buy_cooldown_hours", BUY_COOLDOWN_HOURS)
            price_ratio = config.get("buy_again_price_ratio", BUY_AGAIN_PRICE_RATIO)
            if n_positions >= max_pos:
                print(f"  At max positions ({n_positions}/{max_pos}), skipping buy scan", flush=True)
            else:
                filt = {
                    "buy_price_max": config["buy_price_max"],
                    "buy_price_min": config["buy_price_min"],
                    "brackets_away_min": config["brackets_away_min"],
                    "brackets_away_max": config["brackets_away_max"],
                    "buy_min_hours_remaining": config["buy_min_hours_remaining"],
                }
                scored = score_candidates(
                    df, model_info, config.get("pred_min"), config.get("proba_min"), filters=filt
                )
                if scored.empty:
                    print("  No buy candidates (price/pred/proba filters)", flush=True)
                else:
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
                            state = load_dry_run_state()
                            if state["balance"] < bet_usd:
                                continue
                            shares = bet_usd / row["price"]
                            add_dry_holding(row["event_slug"], row["bracket_label"], row["price"], shares, row["token_id"])
                            state["balance"] -= bet_usd
                            save_dry_run_state(state)
                            log_dry_transaction("buy", event_slug=row["event_slug"], bracket=row["bracket_label"],
                                               price=row["price"], shares=shares, usd=bet_usd, pred=row["_pred"], balance=state["balance"])
                            _print_buy("DRY", row["event_slug"], row["bracket_label"], row["price"], pred=row["_pred"], usd=bet_usd)
                            print(f"      -${bet_usd:.2f}  balance=${state['balance']:.2f}")
                            n_positions += 1
                            continue

                        if client:
                            resp = place_buy_order(
                                client,
                                row["token_id"],
                                row["price"],
                                bet_usd,
                                market_max_price_diff=config["buy_market_max_price_diff"],
                                market_min_usd=config["buy_market_min_usd"],
                            )
                            if resp:
                                px = float(resp.get("_fill_price_est", row["price"]))
                                usd_spent = float(resp.get("_usd", bet_usd))
                                shares = usd_spent / max(px, 1e-12)
                                mode = resp.get("_buy_mode", "limit")
                                add_holding(
                                    row["event_slug"],
                                    row["bracket_label"],
                                    px,
                                    shares,
                                    row["token_id"],
                                    resp.get("orderID", resp.get("orderId", "")),
                                )
                                log_transaction(
                                    "buy",
                                    event_slug=row["event_slug"],
                                    bracket=row["bracket_label"],
                                    price=px,
                                    shares=shares,
                                    usd=usd_spent,
                                    pred=row["_pred"],
                                    order_mode=mode,
                                )
                                _print_buy(
                                    "BOUGHT",
                                    row["event_slug"],
                                    row["bracket_label"],
                                    px,
                                    pred=row["_pred"],
                                    usd=usd_spent,
                                    suffix=f"[{mode}]",
                                )
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

        time.sleep(poll_sec)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _load_model() -> dict:
    """Load model from disk. Fails if not found. No training."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run: python trading_bot.py --train-only"
        )
    import pickle
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _run_backtest(test_days: int = 120, include_2day: bool = False, verbose: bool = True):
    """Load model and data, run backtest. Never trains."""
    from polymarket_data import get_polymarket_data
    from pm_features import build_bracket_features, add_return_labels
    from pm_outcome import run_outcome_comparison

    model_info = _load_model()

    parquet = POLYMARKET_DIR / "bracket_prices.parquet"
    if include_2day:
        print("Fetching Polymarket data (weekly + 2-day)...")
        df, _ = get_polymarket_data(incremental=parquet.exists(), force_refresh=False, verbose=True, weekly_only=False)
    elif parquet.exists():
        df = pd.read_parquet(parquet)
        df["event_start"] = pd.to_datetime(df["event_start"], utc=True)
        df["event_end"] = pd.to_datetime(df["event_end"], utc=True)
        dur_hours = (df["event_end"] - df["event_start"]).dt.total_seconds() / 3600
        df = df[dur_hours > 72].copy()
        if df.empty:
            print("Parquet had only 2-day events; fetching fresh data...")
            df, _ = get_polymarket_data(incremental=False, force_refresh=False, verbose=True)
    else:
        print("Fetching Polymarket data...")
        df, _ = get_polymarket_data(incremental=False, force_refresh=False, verbose=True)

    if df.empty:
        print("No data; cannot run backtest.")
        return None

    period_label = "weekly + 2-day" if include_2day else "weekly only"
    if verbose:
        print(f"\n=== Backtest (model from {MODEL_PATH}, {period_label}) ===\n")
    featured = build_bracket_features(df)
    labeled = add_return_labels(featured)

    results = run_outcome_comparison(labeled, model_info=model_info, test_days=test_days, verbose=verbose)
    return results


def _run_backtest_compare(test_days: int = 120):
    """Run backtest for both weekly-only and weekly+2-day, print side-by-side comparison."""
    try:
        print("Running backtest: WEEKLY ONLY (current default)...")
        results_weekly = _run_backtest(test_days=test_days, include_2day=False, verbose=False)
    except FileNotFoundError as e:
        print(str(e))
        return
    if results_weekly is None:
        return

    try:
        print("\nRunning backtest: WEEKLY + 2-DAY (previous)...")
        results_all = _run_backtest(test_days=test_days, include_2day=True, verbose=False)
    except FileNotFoundError as e:
        print(str(e))
        return
    if results_all is None:
        return

    def _summarize(results):
        out = {}
        for name, trades in results.items():
            if trades.empty:
                out[name] = {"n": 0, "pnl": 0, "ret_pct": 0, "win": 0, "big": 0, "total_back": 0}
            else:
                n = len(trades)
                total_back = trades["return"].sum()
                pnl = total_back - n  # bet $1/trade
                ret_pct = (total_back / n - 1) * 100
                win = (trades["return"] > 1).mean() * 100
                big = (trades["return"] >= 4).sum()
                out[name] = {"n": n, "pnl": pnl, "ret_pct": ret_pct, "win": win, "big": big, "total_back": total_back}
        return out

    s_weekly = _summarize(results_weekly)
    s_all = _summarize(results_all)

    print("\n" + "=" * 90)
    print(f"COMPARISON: Weekly-only vs Weekly+2-day ({test_days}-day window)")
    print("=" * 90)
    print(f"{'Strategy':<32} {'Weekly only':<28} {'Weekly+2day':<28}")
    print("-" * 90)

    all_names = sorted(set(s_weekly) | set(s_all))
    for name in all_names:
        w = s_weekly.get(name, {})
        a = s_all.get(name, {})
        w_str = f"{w.get('n',0):3d} trades  bet ${w.get('n',0)} got ${w.get('total_back',0):.1f}  P&L ${w.get('pnl',0):+.1f} ({w.get('ret_pct',0):+.0f}% ROI)"
        a_str = f"{a.get('n',0):3d} trades  bet ${a.get('n',0)} got ${a.get('total_back',0):.1f}  P&L ${a.get('pnl',0):+.1f} ({a.get('ret_pct',0):+.0f}% ROI)"
        print(f"  {name:<30} {w_str:<28} {a_str:<28}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymarket auto-trading bot")
    parser.add_argument("--dry-run", action="store_true", help="No real orders")
    parser.add_argument("--bet", type=float, default=None, help="USD per trade")
    parser.add_argument("--holdings", action="store_true", help="Show holdings")
    parser.add_argument("--transactions", action="store_true", help="Show recent transactions")
    parser.add_argument("--reset-dry", action="store_true", help="Reset dry run state (balance + holdings) and exit")
    parser.add_argument("--correlations", action="store_true", help="Print feature correlations vs max_return_all")
    parser.add_argument("--train-only", action="store_true", help="Pre-train model and exit")
    parser.add_argument("--backtest", type=int, nargs="?", const=120, metavar="DAYS", help="Run 120-day backtest (default: 120)")
    parser.add_argument("--compare", type=int, nargs="?", const=120, metavar="DAYS", help="Compare weekly-only vs weekly+2day backtest (default: 120)")
    args = parser.parse_args()

    if args.compare is not None:
        _run_backtest_compare(test_days=args.compare)
        return

    if args.backtest is not None:
        try:
            results = _run_backtest(test_days=args.backtest, verbose=True)
        except FileNotFoundError as e:
            print(str(e))
            return
        if results:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            for name, trades in results.items():
                if trades.empty:
                    print(f"  {name:30s}  (no trades)")
                else:
                    n = len(trades)
                    total_back = trades["return"].sum()
                    pnl = total_back - n
                    ret_pct = (total_back / n - 1) * 100
                    win = (trades["return"] > 1).mean() * 100
                    big = (trades["return"] >= 4).sum()
                    print(f"  {name:30s}  {n:3d} trades  bet ${n} got ${total_back:.1f}  P&L ${pnl:+.1f} ({ret_pct:+.0f}% ROI)  {big} hit 4x+")
        return

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

    if args.reset_dry:
        removed = []
        for p in (DRY_RUN_STATE_PATH, DRY_RUN_HOLDINGS_PATH):
            if p.exists():
                p.unlink()
                removed.append(str(p))
        print("Reset dry run state:", removed or "(nothing to reset)")
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
