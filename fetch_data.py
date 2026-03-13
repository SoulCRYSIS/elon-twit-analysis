import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://xtracker.polymarket.com/api/"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
CSV_PATH = DATA_DIR / "hourly_tweets.csv"

REQUEST_DELAY = 0.3  # polite delay between API calls


def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _get(endpoint: str, params: dict | None = None) -> dict:
    params = {k: v for k, v in (params or {}).items() if v is not None}
    resp = requests.get(BASE_URL + endpoint, params=params)
    resp.raise_for_status()
    return resp.json()


# -- period fetching & filtering -------------------------------------------

def fetch_user_trackings(handle: str = "elonmusk") -> list[dict]:
    """Return the full list of tracking periods from the user endpoint."""
    cache = RAW_DIR / f"users_{handle}.json"
    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
    else:
        _ensure_dirs()
        data = _get(f"users/{handle}", {"platform": "X"})
        with open(cache, "w") as f:
            json.dump(data, f, indent=2)
    return data["data"]["trackings"]


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def filter_periods(trackings: list[dict]) -> list[dict]:
    """Select a non-overlapping set of periods that maximises time coverage.

    Strategy:
    1. Compute duration for each period.
    2. Sort by duration descending, then startDate ascending so we prefer
       longer (weekly) periods over shorter ones.
    3. Greedily add periods whose range is not already fully covered.
    """
    for t in trackings:
        t["_start"] = _parse_dt(t["startDate"])
        t["_end"] = _parse_dt(t["endDate"])
        t["_dur"] = (t["_end"] - t["_start"]).total_seconds()

    sorted_periods = sorted(trackings, key=lambda t: (-t["_dur"], t["_start"]))

    selected: list[dict] = []
    covered: list[tuple[datetime, datetime]] = []

    def is_covered(start: datetime, end: datetime) -> bool:
        for cs, ce in covered:
            if cs <= start and ce >= end:
                return True
        return False

    for t in sorted_periods:
        if not is_covered(t["_start"], t["_end"]):
            selected.append(t)
            covered.append((t["_start"], t["_end"]))

    selected.sort(key=lambda t: t["_start"])
    return selected


# -- hourly data fetching --------------------------------------------------

def fetch_tracking_stats(tracking_id: str) -> dict:
    """Fetch detailed stats (with hourly data) for a single tracking period."""
    cache = RAW_DIR / f"tracking_{tracking_id}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    _ensure_dirs()
    data = _get(f"trackings/{tracking_id}", {"includeStats": "true"})
    with open(cache, "w") as f:
        json.dump(data, f, indent=2)
    time.sleep(REQUEST_DELAY)
    return data


def build_hourly_dataframe(
    periods: list[dict],
    *,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch hourly stats for each period and merge into a single DataFrame.

    Deduplicates by timestamp, preferring data from completed periods.
    """
    if CSV_PATH.exists() and not force_refresh:
        if verbose:
            print(f"Loading cached data from {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    _ensure_dirs()
    rows: list[dict] = []

    for i, period in enumerate(periods):
        tid = period["id"]
        if verbose:
            print(f"[{i+1}/{len(periods)}] Fetching {period['title']!r}  ({tid})")
        data = fetch_tracking_stats(tid)
        stats = data.get("data", {}).get("stats", {})
        is_complete = stats.get("isComplete", False)
        hourly = stats.get("daily", [])
        for entry in hourly:
            rows.append({
                "date": entry["date"],
                "count": entry["count"],
                "cumulative": entry["cumulative"],
                "tracking_id": tid,
                "is_complete": is_complete,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # deduplicate: prefer completed period data when timestamps overlap
    df = df.sort_values(["date", "is_complete"], ascending=[True, False])
    df = df.drop_duplicates(subset=["date"], keep="first")
    df = df.sort_values("date").reset_index(drop=True)

    df.drop(columns=["tracking_id", "is_complete"], inplace=True)

    # recompute cumulative from the merged hourly counts
    df["cumulative"] = df["count"].cumsum()

    df.to_csv(CSV_PATH, index=False)
    if verbose:
        print(f"Saved {len(df)} hourly rows to {CSV_PATH}")
    return df


# -- public convenience function -------------------------------------------

def get_hourly_data(
    force_refresh: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """End-to-end: fetch periods, filter, build hourly DataFrame.

    Returns (hourly_df, selected_periods).
    """
    trackings = fetch_user_trackings()
    periods = filter_periods(trackings)
    if verbose:
        print(f"Found {len(trackings)} total periods, selected {len(periods)} non-overlapping")
    df = build_hourly_dataframe(periods, force_refresh=force_refresh, verbose=verbose)
    return df, periods
