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
pip install statsbombpy mplsoccer pandas numpy

## Key Variables
- Independent: throw-in displacement distance (metres)
- Dependent: xT gain per throw-in chain
- Moderators: pitch zone, tournament stage, confederation

