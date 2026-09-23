# Throw-in Positional Displacement and xT(expected threat) Analysis

## Research Question
Does throw-in positional displacement — the distance between where the ball exits play and where it is actually thrown — generate a statistically significant increase in a team’s expected threat (xT) in major international tournaments, and what is the marginal gain in xT per meter of displacement?

## Data
Source: StatsBomb Open Data (free access)
| Tournament               | Year      | Matches | 
|--------------------------|-----------|---------|
| FIFA World Cup           | 2022      | 64      | 
| UEFA Euro                | 2024      | 51      | 
| Copa América             | 2024      | 32      | 
| Africa Cup of Nations    | 2023      | 52      | 
| FIFA Women's World Cup   | 2023      | 64      | 

Total: ~263 matches, ~900k+ events

## Project Structure
data/       → raw and processed StatsBomb event data
notebooks/  → EDA and analysis
src/        → reusable Python modules
outputs/    → figures and result tables

## Setup
pip install -r requirements.txt

## Key Variables
- Independent: throw-in displacement distance (metres)
- Dependent: xT gain per throw-in chain
- Moderators: pitch zone, tournament stage, confederation

## Pipeline
`src/fetch.py` pulls and caches StatsBomb matches/events → `src/displacement.py` computes ball-exit-to-throw-in displacement per throw-in → `src/features.py` attaches chain-level outcomes (`chain_xg`, `max_chain_x`) → `src/xt.py` fits a static Expected Threat grid (Karun Singh's value-iteration method, 16×12 zones) from the pooled event corpus and computes xT gained per throw-in chain → `src/run_pipeline.py` runs all of the above across all 5 tournaments and writes `data/processed/throw_ins.parquet` (11,397 throw-ins).

## Results

**H1 (forward displacement → xT gain, all throw-ins):** No support. `creep_x_m` vs `xt_added`, n=9,638: r=-0.026, p=0.012. Statistically detectable at this sample size but practically negligible (r²≈0.0007), and the sign is opposite the hypothesized direction.

**H2 (effect strongest in the attacking third):** No support — the opposite pattern emerged.

| Zone | n | r | p |
|---|---|---|---|
| Defensive third | 2,384 | -0.053 | 0.009 |
| Middle third | 4,008 | -0.058 | <0.001 |
| Attacking third | 3,246 | -0.000 | 0.992 |

The attacking third — where H2 predicted the strongest positive relationship — shows no relationship at all. Displacement was checked for a non-linear relationship (Spearman ρ=-0.013, p=0.47; decile-binned mean xT stayed flat between 0.05–0.065 across the full displacement range) with no pattern found in the attacking third either.

**Verification:** before accepting the null, the pipeline was checked for the kind of index-alignment bugs that had shown up earlier in development (e.g. a possession-vs-row-index mixup in an early draft of `xt_added`). Checks performed: variance/degeneracy of both variables, rank correlation, decile-binned means, and a manual recomputation of `xt_added` for several real throw-ins against an independently-fit xT grid, confirming the stored values were reasonable. No pipeline bug was found — the null result held under scrutiny.

**Interpretation:** `xt_added` sums xT delta across an entire possession chain, not just the throw-in action itself. A few metres of displacement at the start of a chain may be a weak signal relative to everything that happens in the rest of the possession. A narrower test — isolating xT change on just the first pass/carry immediately following the throw-in — is a natural next step, along with testing H3 (group vs. knockout stage).