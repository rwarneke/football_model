## Full refresh pipeline

Normal usage:

`python scripts/refresh_site_data.py`

This runs the full update pipeline end to end:
* pull and clean recent match results
* sync `reference_data/` into `web/public/reference_data/`
* fit the model
* run 10,000 tournament simulations
* postprocess the simulation outputs into website data

## Manual steps

After refreshing `match_results/results_clean.csv`, run the rest of the pipeline from the repo root in this order:

1. Sync reference data into the website's public copy:
   `python scripts/sync_reference_data.py`

   This copies `reference_data/` into `web/public/reference_data/` so the
   Python pipeline and the website use the same World Cup setup files.

2. Fit the model:
   `python -m src.fit_model`

   This writes:
   * `model_output/model.pkl`
   * `model_output/ratings_current.csv`
   * `model_output/ratings_history.csv`
   * `model_output/ratings_history_yearly.csv`
   * `model_output/win_probabilities.json`

   It also copies the website-facing files into `web/public/model_output/`.

3. Run tournament simulations:
   `python -m src.run_simulations`

   This writes:
   * `model_output/simulation_runs.jsonl`
   * `model_output/simulation_runs_meta.json`

4. Postprocess simulation output into site data:
   `python -m src.postprocess_simulations`

   This writes:
   * `model_output/simulation_results.csv`
   * `model_output/simulation_team_probabilities.json`

   It also copies those files, plus the ratings CSVs, into `web/public/model_output/`.

## Website data summary

The website ultimately reads from `web/public/model_output/`, mainly:

* `ratings_current.csv`
* `ratings_history_yearly.csv`
* `win_probabilities.json`
* `simulation_results.csv`
* `simulation_team_probabilities.json`

## Optional validation

This is not part of the normal website refresh. It exists to validate that the
frontend tournament simulator behaves sensibly against `win_probabilities.json`.

Run:
`cd web && npx tsx scripts/simulate-tournament.ts`

This writes:
* `model_output/web_simulation_results.csv`
* `web/public/model_output/web_simulation_results.csv`
