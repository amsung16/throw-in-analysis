"""End-to-end pipeline: fetch tournaments, fit a global xT grid, build throw-in features."""

from pathlib import Path

import pandas as pd

from src.fetch import TOURNAMENTS, get_matches, get_events
from src.features import build_features
from src.xt import fit_xt_grid, xt_added

PROCESSED_PATH = Path("data/processed/throw_ins.parquet")


def run_pipeline(tournaments=TOURNAMENTS) -> pd.DataFrame:
    """Fetch all matches/events, fit a global xT grid, and build a throw-in
    level feature table (displacement + chain outcomes + xT gain) across
    every tournament. Saves the result to data/processed/throwins.parquet.
    """
    match_events = []
    
    for t in tournaments:
        try:
            matches = get_matches(t["competition_id"], t["season_id"])
        except Exception as e:
            print(f"Could not find match {t['label']}:{e}")
            continue
        
        for match_id in matches["match_id"]:
            try:
                events = get_events(match_id)
            except Exception as e:
                print(f"Could not load match {match_id}: {e}")
                continue
            match_events.append((match_id,t["label"],events))
    
    pooled_events = pd.concat([e for _, _, e in match_events], ignore_index=True)
    grid = fit_xt_grid(pooled_events)
    
    all_features = []
    for match_id, tournament_lebel, events in match_events:
        features = build_features(events, match_id, tournament_lebel)
        features['xt_added'] = features['possession'].map(lambda p: xt_added(events, p, grid))
        all_features.append(features)
    
    df = pd.concat(all_features, ignore_index=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    
    return df