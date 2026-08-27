"""Expected Threat, Karun Singh's method: fit a 16x12 grid from a broad
actions corpus, then apply it to individual possession chains."""
import numpy as np
import pandas as pd

GRID_L = 16  # zones along pitch length
GRID_W = 12  # zones along pitch width

def zone_for_location(location: list[float], grid_shape: tuple[int, int]) -> tuple[int, int]:
    """Map a StatsBomb [x, y] location (120x80 pitch) to a (row, col)
    zone index for a grid of the given shape."""
    l, w = grid_shape
    x, y = location
    zone_l = min(int(x / 120 * l), l - 1)
    zone_w = min(int(y / 80 * w), w - 1)
    return zone_l, zone_w

def fit_xt_grid(
    actions: pd.DataFrame,
    l: int = GRID_L,
    w: int = GRID_W,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    """Fit the xT grid on a broad corpus of on-ball actions.

    Builds shot-probability, goal-probability, move-probability, and
    zone-to-zone transition matrices from `actions`, then iterates
    xT(z) = s(z)*g(z) + m(z) * sum_z'[ T(z,z') * xT(z') ] to convergence.
    Returns an (l, w) array of xT values, one per zone.
    """
    action = actions[
        (actions["type"].isin(["Pass", "Carry", "Shot"]))
        & actions["location"].apply(lambda l: isinstance(l, list))
    ].copy()
    
    grid_shape = (l, w)
    zones = action['location'].apply(zone_for_location, args=(grid_shape,))
    action['zone_l'] = zones.apply(lambda z: z[0])
    action['zone_w'] = zones.apply(lambda z: z[1])
    
    is_shot = action['type'] == 'Shot'
    is_move = (
        (action['type'] == 'Carry') | 
        ((action['type'] == 'Pass') & action['pass_outcome'].isna())
    )

    total = np.zeros((l,w))
    shots = np.zeros((l,w))
    moves = np.zeros((l,w))
    xg_sum = np.zeros((l,w))
    
    for (zl, zw), group in action.groupby(['zone_l', 'zone_w']):
        index = group.index
        total[zl,zw] = len(index)
        shots[zl,zw] = is_shot.loc[index].sum()
        moves[zl,zw] = is_move.loc[index].sum()
        xg_sum[zl,zw] = group.loc[is_shot.loc[index], 'shot_statsbomb_xg'].fillna(0).sum()
    
    shot_prob = np.where(total > 0, shots / np.where(total > 0, total, 1),0)
    goal_prob = np.where(shots > 0, xg_sum / np.where(shots > 0, shots, 1),0)
    move_prob = np.where(total > 0, moves / np.where(total > 0, total, 1),0)
    
    move_action = action[is_move].copy()
    
    def end_location(row):
        return row['pass_end_location'] if row['type'] == 'Pass' else row['carry_end_location']
    
    move_action['end_location'] = move_action.apply(end_location, axis=1)
    move_action = move_action[move_action["end_location"].apply(lambda l: isinstance(l, list))]
    end_zones = move_action['end_location'].apply(lambda l: zone_for_location(l, grid_shape))
    move_action['end_zone_l'] = end_zones.apply(lambda z: z[0])
    move_action['end_zone_w'] = end_zones.apply(lambda z: z[1])
    
    transition = np.zeros((l, w, l, w))
    counts = move_action.groupby(['zone_l', 'zone_w', 'end_zone_l', 'end_zone_w']).size()
    for (zl, zw, zl2, zw2), count in counts.items():
        transition[zl, zw, zl2, zw2] = count
    
    zone_totals = transition.sum(axis=(2,3),keepdims = True)
    transition = np.where(zone_totals > 0, transition / np.where(zone_totals > 0, zone_totals, 1),0)
    
    xt = np.zeros((l,w))
    for _ in range(max_iter):
        xt_next = shot_prob * goal_prob + move_prob * np.einsum('lwxy, xy -> lw', transition, xt)
        if np.abs(xt_next - xt).max() < tol:
            xt = xt_next
            break
        xt = xt_next
    
    return xt

def xt_value(location: list[float], grid: np.ndarray) -> float:
    """Look up the xT value at a single pitch location."""
    l, w = grid.shape
    zl, zw = zone_for_location(location, (l, w))
    return grid[zl, zw]


def xt_added(events: pd.DataFrame, possession_id: int, grid: np.ndarray) -> float:
    """xT gained across a possession: sum of xt_value(end) - xt_value(start)
    over each pass/carry action in the chain."""
    chain = events[(events['possession'] == possession_id) & (events['type'].isin(['Pass', 'Carry']))]
    xt_start = chain['location'].apply(lambda l: xt_value(l, grid) if isinstance(l, list) else 0)
    def end_location(row):
        return row['pass_end_location'] if row['type'] == 'Pass' else row['carry_end_location']
    xt_end = chain.apply(end_location, axis=1).apply(lambda l: xt_value(l, grid) if isinstance(l, list) else 0)
    return float((xt_end - xt_start).sum())