# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Statistical analysis of whether **throw-in positional displacement** (the distance a player creeps forward from where the ball actually exited play before taking the throw) generates measurable increases in Expected Threat (xT) in major international tournaments.

**Data source:** StatsBomb Open Data (free) — ~263 matches across FIFA World Cup 2022, UEFA Euro 2024, Copa América 2024, AFCON 2023, Women's World Cup 2023.

## Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install statsbombpy mplsoccer pandas numpy matplotlib scipy
```

## Development

Notebooks are the primary interface. Run Jupyter from the repo root with the venv active:

```bash
jupyter notebook
```

There is no test suite or linter configured yet. `src/` and `tests/` are empty stubs.

## Key Concepts

**StatsBomb coordinate system:** 120×80 pitch units. Each team always attacks left→right (toward x=120). Conversion factors: `105/120` m per x-unit, `68/80` m per y-unit.

**Displacement computation:** The ball exit location comes from the `end_location` of the last event with `out=True` in the preceding possession (`possession - 1`). Priority: `pass_end_location` → `carry_end_location` → `location` (only within 10 units of touchline). When exit and throw teams differ, rotate exit coordinates 180° (`[120-x, 80-y]`).

**Key columns in event data:**
- `pass_type == 'Throw-in'` — identifies throw-in events
- `out == True` — ball-exit events in preceding possession
- `possession` — groups events into possession sequences
- `shot_statsbomb_xg` — StatsBomb xG for shots
- `location`, `pass_end_location`, `carry_end_location` — coordinate fields

**Outcome variables:**
- `creep_m` — Euclidean displacement in metres
- `creep_x_m` — signed forward displacement (positive = toward opponent goal)
- `chain_xg` — sum of xG from shots in the throw-in possession
- `max_chain_x` — furthest x reached in the possession (field advancement proxy)

## Tournament IDs (StatsBomb)

| Tournament | competition_id | season_id |
|---|---|---|
| FIFA World Cup 2022 | 43 | 106 |
| UEFA Euro 2024 | 55 | 282 |
| Copa América 2024 | 223 | 282 |
| AFCON 2023 | 1267 | 107 |
| Women's World Cup 2023 | 72 | 107 |

## Project Structure

```
notebooks/   # EDA and analysis (primary workspace)
src/         # reusable Python modules (stub — not yet populated)
data/raw/    # downloaded StatsBomb event data
data/processed/  # cleaned/aggregated outputs
outputs/figures/ # saved plots
outputs/tables/  # result tables
docs/        # pre-research notes and lit review
```

## Open Research Questions

- Minimum displacement threshold to filter noise (candidate: >0.5m)
- Handling throw-ins in final seconds of each half
- Whether 360° freeze-frame data (available for WC2022, Euro2024, Copa2024, WWC2023 but NOT AFCON2023) can enrich with defensive shape context
