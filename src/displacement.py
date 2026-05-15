import numpy as np
import pandas as pd

METRES_PER_X = 105 / 120  # StatsBomb: 120 units = 105 m
METRES_PER_Y = 68 / 80    # StatsBomb: 80 units = 68 m
TOUCHLINE_THRESH = 10      # units; beyond this, player location ≠ ball exit


def _best_exit_location(exit_row: pd.Series) -> list | None:
    """
    Extract ball-exit coordinates from the exiting event.
    Priority: pass_end_location > carry_end_location > player location.
    Player location only accepted when within TOUCHLINE_THRESH of the touchline.
    """
    if isinstance(exit_row.get("pass_end_location"), list):
        return exit_row["pass_end_location"]
    if isinstance(exit_row.get("carry_end_location"), list):
        return exit_row["carry_end_location"]
    loc = exit_row.get("location")
    if isinstance(loc, list) and min(loc[1], 80 - loc[1]) < TOUCHLINE_THRESH:
        return [loc[0], 0.0 if loc[1] < 40 else 80.0]
    return None


def _to_throw_frame(exit_xy: list, same_team: bool) -> list:
    """
    StatsBomb normalises so each team always attacks left→right.
    When exit and throw-in belong to different teams, rotate 180°.
    """
    if same_team:
        return exit_xy
    return [120 - exit_xy[0], 80 - exit_xy[1]]


def _get_exit_location(events: pd.DataFrame, throw_idx: int) -> list | None:
    """
    For the throw-in at positional index throw_idx, find the last out=True event
    in the preceding possession and return its exit location converted to the
    throwing team's coordinate frame.
    """
    ti = events.iloc[throw_idx]
    out_events = events[
        (events["possession"] == ti["possession"] - 1) & (events["out"] == True)
    ]
    if out_events.empty:
        return None
    exit_row = out_events.iloc[-1]
    exit_loc = _best_exit_location(exit_row)
    if exit_loc is None:
        return None
    same_team = exit_row["team"] == ti["team"]
    return _to_throw_frame(exit_loc, same_team)


def compute_displacement(throw_loc: list, exit_loc: list) -> dict:
    """
    Compute displacement metrics between ball-exit and throw-in locations.

    Returns:
        creep_m   – Euclidean distance in metres
        creep_x_m – signed forward displacement in metres (+ = toward opponent goal)
        creep_y_m – lateral displacement in metres
    """
    dx = (throw_loc[0] - exit_loc[0]) * METRES_PER_X
    dy = (throw_loc[1] - exit_loc[1]) * METRES_PER_Y
    return {
        "creep_m": round((dx**2 + dy**2) ** 0.5, 3),
        "creep_x_m": round(dx, 3),
        "creep_y_m": round(dy, 3),
    }


def add_displacement(events: pd.DataFrame) -> pd.DataFrame:
    """
    Given a sorted StatsBomb events DataFrame for one match, return a DataFrame
    of throw-in rows with displacement columns appended:
        throw_location, exit_location, creep_m, creep_x_m, creep_y_m

    Rows where the exit location cannot be determined are kept with NaN displacement.
    """
    events = events.sort_values("index").reset_index(drop=True)
    throw_mask = events["pass_type"] == "Throw-in"
    throw_indices = events.index[throw_mask].tolist()

    records = []
    for idx in throw_indices:
        pos_idx = events.index.get_loc(idx)
        ti = events.loc[idx]
        throw_loc = ti["location"]
        if not isinstance(throw_loc, list):
            continue
        exit_loc = _get_exit_location(events, pos_idx)
        disp = compute_displacement(throw_loc, exit_loc) if exit_loc else {
            "creep_m": None, "creep_x_m": None, "creep_y_m": None
        }
        records.append({
            "match_id": ti["match_id"],
            "team": ti["team"],
            "minute": ti["minute"],
            "possession": ti["possession"],
            "throw_location": throw_loc,
            "exit_location": exit_loc,
            **disp,
        })

    return pd.DataFrame(records)
