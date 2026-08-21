"""Fetch and cache StatsBomb match/event data."""
from pathlib import Path
from sys import path

import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")

TOURNAMENTS = [
    {'competition_id': 43,   'season_id': 106, 'label': 'FIFA World Cup 2022'},
    {'competition_id': 55,   'season_id': 282, 'label': 'UEFA Euro 2024'},
    {'competition_id': 223,  'season_id': 282, 'label': 'Copa América 2024'},
    {'competition_id': 1267, 'season_id': 107, 'label': 'AFCON 2023'},
    {'competition_id': 72,   'season_id': 107, 'label': "Women's World Cup 2023"},
]

def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return the matches table for one competition/season.

    Reads data/raw/matches_{competition_id}_{season_id}.parquet if cached;
    otherwise fetches via statsbombpy and writes the cache before returning.
    """
    path = RAW_DIR / f"matches_{competition_id}_{season_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from statsbombpy import sb
    df = sb.matches(competition_id=competition_id, season_id=season_id)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def get_events(match_id: int) -> pd.DataFrame:
    """Return the event stream for one match.

    Reads data/raw/events_{match_id}.parquet if cached; otherwise fetches
    via statsbombpy and writes the cache before returning.
    """
    path = RAW_DIR / f"events_{match_id}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        from statsbombpy import sb
        df = sb.events(match_id=match_id)
        if "tactics" in df.columns:
            df = df.drop(columns=["tactics"])
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        
    for col in ['location', 'pass_end_location', 'carry_end_location']:
        df[col] = df[col].apply(lambda v: list(v) if isinstance(v, np.ndarray) else v)
    return df