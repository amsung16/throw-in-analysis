"""Per-possession outcome proxies (used alongside xT, not instead of it)."""
import pandas as pd
from src.displacement import add_displacement

def chain_xg(events: pd.DataFrame, possession_id: int) -> float:
    """Sum of StatsBomb shot xG within one possession."""
    shots = events[(events['possession'] == possession_id) & (events['type'] == 'Shot')]
    return float(shots['shot_statsbomb_xg'].fillna(0).sum())


def max_chain_x(events: pd.DataFrame, possession_id: int) -> float | None:
    """Furthest x-coordinate reached within one possession."""
    locs  = events[events['possession'] == possession_id]['location'].dropna()
    xs    = [l[0] for l in locs if isinstance(l, list)]
    return max(xs) if xs else None

def build_features(events:pd.DataFrame, match_id:int, tournament_label:str) -> pd.DataFrame:
    """Add features to the events DataFrame."""
    events = events.sort_values("index").reset_index(drop=True)
    disp = add_displacement(events)
    
    disp["tournament"] = tournament_label
    disp["match_id"]   = match_id
    disp["chain_xg"]    = disp["possession"].map(lambda p: chain_xg(events, p))
    disp["max_chain_x"] = disp["possession"].map(lambda p: max_chain_x(events, p))
    disp["throwin_x"] = disp["throw_location"].map(lambda l: l[0] if isinstance(l, list) else None)
    disp["throwin_y"] = disp["throw_location"].map(lambda l: l[1] if isinstance(l, list) else None)

    return disp[[
        "tournament","match_id","team","minute",
        "throwin_x", "throwin_y",
        "creep_m", "creep_x_m", "creep_y_m",
        "chain_xg", "max_chain_x"
    ]]
