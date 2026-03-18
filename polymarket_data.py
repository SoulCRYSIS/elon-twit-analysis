"""Fetch and cache Polymarket bracket market data for Elon Musk tweet events.

Data sources:
- Gamma API (event/market discovery): https://gamma-api.polymarket.com
- CLOB API (price history): https://clob.polymarket.com
- Data API (trade history fallback): https://data-api.polymarket.com
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
DATA_DIR = Path(__file__).parent / "data" / "polymarket"
EVENTS_CACHE = DATA_DIR / "events.json"
PRICES_DIR = DATA_DIR / "prices"
PARQUET_PATH = DATA_DIR / "bracket_prices.parquet"

REQUEST_DELAY = 0.15
PRICE_DELAY = 0.05
CHUNK_DAYS = 14


def _ensure_dirs():
    PRICES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Event discovery
# ---------------------------------------------------------------------------

def search_elon_tweet_events(*, force_refresh: bool = False, incremental: bool = False, verbose: bool = True) -> list[dict]:
    """Discover all Elon Musk tweet bracket events from Polymarket.

    If incremental=True: fetch search slugs, only get details for slugs not in cache, merge.
    """
    cached: list[dict] = []
    if EVENTS_CACHE.exists():
        with open(EVENTS_CACHE) as f:
            cached = json.load(f)
    cached_slugs = {e.get("slug") for e in cached if e.get("slug")}

    if cached and not force_refresh and not incremental:
        if verbose:
            print(f"Loading cached events from {EVENTS_CACHE}")
        return cached

    _ensure_dirs()
    all_slugs: list[dict] = []
    page = 1
    while True:
        if verbose:
            print(f"  Searching page {page}...")
        resp = requests.get(f"{GAMMA_URL}/public-search", params={
            "q": "elon musk tweets",
            "limit_per_type": 50,
            "page": page,
        })
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events", [])
        if not events:
            break
        all_slugs.extend(events)
        if not data.get("pagination", {}).get("hasMore", False):
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    search_slugs = {s.get("slug") for s in all_slugs if s.get("slug")}
    new_slugs = search_slugs - cached_slugs if incremental and cached else search_slugs

    if incremental and cached and not new_slugs:
        if verbose:
            print("  No new events, using cache")
        return cached

    if verbose:
        print(f"Found {len(all_slugs)} search results, fetching {len(new_slugs)} new event details...")

    full_events: list[dict] = [] if not incremental else [e for e in cached if e.get("slug") in search_slugs]
    for i, stub in enumerate(all_slugs):
        slug = stub.get("slug", "")
        if not slug or (incremental and slug in cached_slugs):
            continue
        resp = requests.get(f"{GAMMA_URL}/events", params={"slug": slug})
        resp.raise_for_status()
        arr = resp.json()
        if arr and isinstance(arr, list) and arr[0].get("markets"):
            full_events.append(arr[0])
        if verbose and (i + 1) % 20 == 0:
            print(f"  Fetched {i + 1}/{len(new_slugs)} new event details")
        time.sleep(REQUEST_DELAY)

    full_events.sort(key=lambda e: (e.get("startDate") or "", e.get("slug") or ""))
    with open(EVENTS_CACHE, "w") as f:
        json.dump(full_events, f)
    if verbose:
        print(f"Cached {len(full_events)} events ({len(new_slugs)} new)")
    return full_events


def filter_bracket_events(events: list[dict], min_markets: int = 5) -> list[dict]:
    """Keep only events that have bracket-style markets (multiple outcomes)."""
    return [e for e in events if len(e.get("markets", [])) >= min_markets]


# ---------------------------------------------------------------------------
# Bracket parsing helpers
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
_LESS_RE = re.compile(r"less than\s+(\d+)", re.IGNORECASE)
_GREATER_RE = re.compile(r"(\d+)\s*\+|more than\s+(\d+)|at least\s+(\d+)|(\d+)\s*or more", re.IGNORECASE)


def parse_bracket(question: str) -> tuple[float | None, float | None, str]:
    """Extract (low, high, label) from a market question string."""
    m = _RANGE_RE.search(question)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return lo, hi, f"{lo}-{hi}"

    m = _LESS_RE.search(question)
    if m:
        hi = int(m.group(1))
        return 0, hi - 1, f"<{hi}"

    m = _GREATER_RE.search(question)
    if m:
        val = int(next(g for g in m.groups() if g is not None))
        return val, 9999, f"{val}+"

    return None, None, question[:60]


def _parse_json_field(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


# ---------------------------------------------------------------------------
# Price history fetching (chunked startTs/endTs + Data API fallback)
# ---------------------------------------------------------------------------

def _clob_chunked(token_id: str, start_ts: int, end_ts: int, fidelity: int = 60) -> list[dict]:
    """Fetch CLOB price history in time-range chunks to handle resolved markets."""
    all_points: list[dict] = []
    chunk_sec = CHUNK_DAYS * 86400
    cursor = start_ts

    while cursor < end_ts:
        chunk_end = min(cursor + chunk_sec, end_ts)
        try:
            resp = requests.get(
                f"{CLOB_URL}/prices-history",
                params={
                    "market": token_id,
                    "startTs": cursor,
                    "endTs": chunk_end,
                    "fidelity": fidelity,
                },
                headers={"Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            pts = resp.json().get("history", [])
            all_points.extend(pts)
        except Exception:
            pass
        time.sleep(PRICE_DELAY)
        cursor = chunk_end

    return all_points


def _data_api_trades(token_id: str, limit: int = 10000) -> list[dict]:
    """Fallback: build hourly OHLC from trade data via Data API."""
    trades: list[dict] = []
    offset = 0
    batch = 500

    while offset < limit:
        try:
            resp = requests.get(
                f"{DATA_API_URL}/trades",
                params={"asset": token_id, "limit": batch, "offset": offset},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            batch_data = resp.json()
            if not batch_data:
                break
            trades.extend(batch_data)
            if len(batch_data) < batch:
                break
            offset += batch
            time.sleep(PRICE_DELAY)
        except Exception:
            break

    if not trades:
        return []

    df = pd.DataFrame(trades)
    if "timestamp" not in df.columns or "price" not in df.columns:
        price_col = "outcome_price" if "outcome_price" in df.columns else None
        ts_col = "created_at" if "created_at" in df.columns else None
        if not price_col or not ts_col:
            return []
        df = df.rename(columns={price_col: "price", ts_col: "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    if df.empty:
        return []

    hourly = df.set_index("timestamp").resample("1h")["price"].last().dropna()
    return [{"t": int(ts.timestamp()), "p": round(float(p), 6)} for ts, p in hourly.items()]


def _merge_and_dedup_history(existing: list[dict], new_points: list[dict]) -> list[dict]:
    """Merge new points into existing, dedupe by timestamp, sort."""
    seen = {p["t"] for p in existing}
    for p in new_points:
        t = p.get("t")
        if t is not None and t not in seen:
            seen.add(t)
            existing.append(p)
    return sorted(existing, key=lambda x: x["t"])


def fetch_price_history(
    token_id: str,
    condition_id: str,
    event_start: str | None = None,
    event_end: str | None = None,
    *,
    force_refresh: bool = False,
    incremental: bool = False,
) -> list[dict]:
    """Fetch hourly price history using chunked CLOB queries + Data API fallback.

    If incremental=True and cache exists: fetch only from last_ts to now, merge, save.
    """
    cache = PRICES_DIR / f"{condition_id}.json"
    existing: list[dict] = []
    if cache.exists():
        with open(cache) as f:
            existing = json.load(f) or []
        if existing and not force_refresh and not incremental:
            return existing

    _ensure_dirs()

    now = datetime.now(timezone.utc)
    start_ts: int
    end_ts = int(now.timestamp())

    if incremental and existing:
        # Only fetch new data since last point
        last_ts = max(p.get("t", 0) for p in existing)
        start_ts = last_ts + 3600  # 1h after last to avoid overlap
        if start_ts >= end_ts:
            return existing  # No new data
        new_points = _clob_chunked(token_id, start_ts, end_ts)
        if not new_points:
            try:
                resp = requests.get(
                    f"{CLOB_URL}/prices-history",
                    params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
                    headers={"Accept": "application/json"},
                    timeout=15,
                )
                resp.raise_for_status()
                new_points = resp.json().get("history", [])
            except Exception:
                pass
        if new_points:
            history = _merge_and_dedup_history(existing, new_points)
            with open(cache, "w") as f:
                json.dump(history, f)
            return history
        return existing
    else:
        # Full fetch: determine time range from event dates
        if event_start:
            try:
                dt_start = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
            except ValueError:
                dt_start = now - timedelta(days=30)
            start_ts = int((dt_start - timedelta(days=3)).timestamp())
        else:
            start_ts = int((now - timedelta(days=30)).timestamp())

        if event_end:
            try:
                dt_end = datetime.fromisoformat(event_end.replace("Z", "+00:00"))
            except ValueError:
                dt_end = now
            end_ts = int((dt_end + timedelta(days=1)).timestamp())
        else:
            end_ts = int(now.timestamp())

    # Strategy 1: chunked CLOB with startTs/endTs
    history = _clob_chunked(token_id, start_ts, end_ts)

    # Strategy 2: try CLOB with interval=all as fallback (works for recent markets)
    if not history:
        try:
            resp = requests.get(
                f"{CLOB_URL}/prices-history",
                params={"market": token_id, "interval": "all", "fidelity": 60},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            history = resp.json().get("history", [])
        except Exception:
            pass
        time.sleep(PRICE_DELAY)

    # Strategy 3: Data API trade aggregation
    if not history:
        history = _data_api_trades(token_id)

    # Deduplicate by timestamp
    if history:
        seen = set()
        deduped = []
        for pt in history:
            t = pt.get("t")
            if t not in seen:
                seen.add(t)
                deduped.append(pt)
        history = sorted(deduped, key=lambda x: x["t"])

    with open(cache, "w") as f:
        json.dump(history, f)
    return history


def fetch_all_price_histories(
    events: list[dict],
    *,
    force_refresh: bool = False,
    incremental: bool = False,
    verbose: bool = True,
) -> int:
    """Fetch and cache price histories for every bracket in the given events.

    If incremental=True: only fetch new data (from last_ts to now) for existing caches;
    full fetch for markets with no cache.
    """
    _ensure_dirs()
    total = sum(len(e.get("markets", [])) for e in events)
    fetched = 0
    got_data = 0
    incremental_count = 0

    for e in events:
        event_start = e.get("startDate")
        event_end = e.get("endDate")
        for m in e.get("markets", []):
            cond_id = m.get("conditionId", "")
            tokens = _parse_json_field(m.get("clobTokenIds", []))
            if not tokens or not cond_id:
                continue
            yes_token = tokens[0]
            cache = PRICES_DIR / f"{cond_id}.json"
            use_incremental = incremental and cache.exists() and not force_refresh
            if use_incremental:
                incremental_count += 1
            history = fetch_price_history(
                yes_token, cond_id,
                event_start=event_start, event_end=event_end,
                force_refresh=force_refresh,
                incremental=use_incremental,
            )
            fetched += 1
            if history:
                got_data += 1
            if verbose and fetched % 50 == 0:
                print(f"  Price history: {fetched}/{total} ({got_data} with data, {incremental_count} incremental)")

    if verbose:
        print(f"  Price history: {fetched}/{total} done ({got_data} with data, {incremental_count} incremental)")
    return fetched


# ---------------------------------------------------------------------------
# Consolidated DataFrame
# ---------------------------------------------------------------------------

def build_bracket_dataframe(events: list[dict], *, verbose: bool = True) -> pd.DataFrame:
    """Build a flat DataFrame of (event, bracket, timestamp, price)."""
    if PARQUET_PATH.exists():
        if verbose:
            print(f"Loading cached bracket data from {PARQUET_PATH}")
        return pd.read_parquet(PARQUET_PATH)

    rows: list[dict] = []
    for e in events:
        slug = e.get("slug", "")
        title = e.get("title", "")
        event_start = e.get("startDate")
        event_end = e.get("endDate")
        event_volume = e.get("volume", 0)
        closed = e.get("closed", False)

        for m in e.get("markets", []):
            cond_id = m.get("conditionId", "")
            question = m.get("question", "")
            bracket_low, bracket_high, bracket_label = parse_bracket(question)
            market_volume = m.get("volume", 0)

            cache = PRICES_DIR / f"{cond_id}.json"
            if not cache.exists():
                continue
            with open(cache) as f:
                history = json.load(f)

            for pt in history:
                rows.append({
                    "event_slug": slug,
                    "event_title": title,
                    "event_start": event_start,
                    "event_end": event_end,
                    "event_volume": float(event_volume) if event_volume else 0,
                    "event_closed": closed,
                    "condition_id": cond_id,
                    "bracket_label": bracket_label,
                    "bracket_low": bracket_low,
                    "bracket_high": bracket_high,
                    "market_volume": float(market_volume) if market_volume else 0,
                    "timestamp": pd.Timestamp(pt["t"], unit="s", tz="UTC"),
                    "price": pt["p"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["event_start"] = pd.to_datetime(df["event_start"], utc=True)
    df["event_end"] = pd.to_datetime(df["event_end"], utc=True)
    df = df.sort_values(["event_start", "bracket_low", "timestamp"]).reset_index(drop=True)

    df.to_parquet(PARQUET_PATH, index=False)
    if verbose:
        print(f"Saved {len(df)} rows to {PARQUET_PATH}")
    return df


# ---------------------------------------------------------------------------
# Public convenience
# ---------------------------------------------------------------------------

def get_polymarket_data(
    *,
    force_refresh: bool = False,
    incremental: bool = False,
    verbose: bool = True,
    min_markets: int = 5,
) -> tuple[pd.DataFrame, list[dict]]:
    """End-to-end: discover events, fetch prices, build DataFrame.

    If incremental=True: only fetch new events and append new price data (fast).
    Use incremental for periodic retrains; use force_refresh for full rebuild.
    """
    events = search_elon_tweet_events(
        force_refresh=force_refresh and not incremental,
        incremental=incremental,
        verbose=verbose,
    )
    bracket_events = filter_bracket_events(events, min_markets=min_markets)
    if verbose:
        print(f"{len(bracket_events)} bracket events (of {len(events)} total)")

    if force_refresh or not PARQUET_PATH.exists() or incremental:
        fetch_all_price_histories(
            bracket_events,
            force_refresh=force_refresh and not incremental,
            incremental=incremental,
            verbose=verbose,
        )
        if PARQUET_PATH.exists():
            PARQUET_PATH.unlink()

    df = build_bracket_dataframe(bracket_events, verbose=verbose)
    return df, bracket_events
