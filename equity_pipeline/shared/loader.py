"""
shared/loader.py
================
Load CSV, canonicalize column names, drop excluded columns,
construct fwd_return, and apply feature lagging — exactly ONCE
before the walk-forward loop.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from supabase import create_client, Client

# Canonical column aliases → rename on load
COLUMN_ALIASES = {
    "cocode":       "co_code",
    "co code":      "co_code",
    "BMsep":        "BM_sep",
    "bmsep":        "BM_sep",
    "lagret":       "lag_ret",
    "lag_return":   "lag_ret",
    "lagmv":        "lag_mv",
    "lag_market_value": "lag_mv",
}

# lag_mv is always dropped (VIF=204, near-duplicate of mktcap)
DROP_COLS = ["lag_mv"]


def load_data(
    data_path:  str | Path,
    id_col:     str = "co_code",
    date_col:   str = "Month",
    target_col: str = "monthly_gross_return",
    features:   tuple = ("BM_sep", "OpProf", "Inv", "Momentum", "lag_ret", "mktcap"),
) -> pd.DataFrame:
    """
    Load raw data (from CSV or Supabase) and return a clean panel DataFrame.
    """
    if str(data_path).startswith("supabase://"):
        table_name = str(data_path).replace("supabase://", "")
        df = load_data_from_supabase(table_name)
    else:
        path = Path(data_path)
        print(f"[Loader] Reading {path} ...")
        df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"[Loader] The loaded DataFrame is empty (from {data_path}).")

    # Convert date_col to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Canonicalize column names
    df = df.rename(columns=COLUMN_ALIASES)

    # Drop permanently excluded columns
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Sort
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    # Validate required columns
    required = list(features) + [id_col, date_col, target_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[Loader] Missing columns: {missing}")

    print(f"[Loader] {len(df):,} rows | "
          f"{df[id_col].nunique():,} firms | "
          f"{df[date_col].nunique():,} months | "
          f"{df[date_col].min().date()} → {df[date_col].max().date()}")
    return df


def load_data_from_supabase(table_name: str) -> pd.DataFrame:
    """
    Fetch all data from a Supabase table.
    Requires SUPABASE_URL and SUPABASE_KEY env variables.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")

    # Local cache check
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{table_name}.parquet"

    if cache_file.exists():
        print(f"[Loader] Loading from local cache: {cache_file} ...")
        return pd.read_parquet(cache_file)

    print(f"[Loader] Fetching from Supabase table: {table_name} ...")
    supabase: Client = create_client(url, key)
    
    # Supabase limit is 1000 rows per request
    page_size = 1000
    
    # 1. Get total count
    count_res = supabase.table(table_name).select("*", count="exact").limit(0).execute()
    total_count = count_res.count
    if total_count == 0:
        print(f"[Loader] Warning: No data found in Supabase table: {table_name}")
        return pd.DataFrame()

    print(f"[Loader] Parallel fetching {total_count:,} rows ...")

    # 2. Define page fetcher
    def fetch_page(start_offset: int):
        end_offset = start_offset + page_size - 1
        res = supabase.table(table_name).select("*").range(start_offset, end_offset).execute()
        return res.data

    # 3. Fetch in parallel
    from concurrent.futures import ThreadPoolExecutor
    offsets = list(range(0, total_count, page_size))
    all_data = []
    
    max_workers = 16  # Significant speedup
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(fetch_page, offsets), total=len(offsets), desc="  Downloading"))
        for batch in results:
            all_data.extend(batch)

    df = pd.DataFrame(all_data)
    
    # Save to cache
    print(f"[Loader] Saving to cache: {cache_file} ...")
    df.to_parquet(cache_file, index=False)

    return df


def build_target(
    df:         pd.DataFrame,
    id_col:     str = "co_code",
    target_col: str = "monthly_gross_return",
    fwd_col:    str = "fwd_return",
) -> pd.DataFrame:
    """
    Construct forward return target: fwd_return = shift(-1) per firm.
    Must be called ONCE before the walk-forward loop.
    Rows where fwd_return is NaN (last month per firm) are dropped.
    """
    df = df.copy()
    df[fwd_col] = df.groupby(id_col)[target_col].shift(-1)
    before = len(df)
    df = df.dropna(subset=[fwd_col]).reset_index(drop=True)
    print(f"[Loader] fwd_return: {before - len(df):,} rows dropped (last month per firm). "
          f"Remaining: {len(df):,}")
    return df


def lag_features(
    df:       pd.DataFrame,
    features: tuple | list,
    id_col:   str = "co_code",
) -> pd.DataFrame:
    """
    Shift all feature columns by 1 month per firm.
    Predict t+1 returns using t information.
    Must be called ONCE before the walk-forward loop.
    Rows with NaN in any feature after lagging are dropped.
    """
    df = df.copy()
    df[list(features)] = df.groupby(id_col)[list(features)].shift(1)
    before = len(df)
    df = df.dropna(subset=list(features)).reset_index(drop=True)
    print(f"[Loader] Feature lag: {before - len(df):,} rows dropped (first month per firm). "
          f"Remaining: {len(df):,}")
    return df
