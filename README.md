# Conformal Calibration Under Replay

This repository reproduces the bounded-replay calibration experiments. It contains the rank correction, canonical public-data tasks, fixed experiment configurations, saved scores and splits, numerical results, and verification tools.

## Main result

For `n` source records, `N` replayed rows, maximum multiplicity `B`, and target miscoverage `alpha`, let `r = ceil((n + 1) * (1 - alpha))`. The corrected replay rank is

```text
k = min(B * (r - 1) + 1, N - n + r)
```

Use an infinite threshold when `r > n`. Every source record must occur at least once, the declared multiplicity bound must hold, and source calibration and fresh-test scores must be exchangeable conditional on the fitted model. Exact record identifiers are used by the deduplication comparator; the proposed threshold uses only the scores and count metadata.

The primary controlled stress test uses `n=128`, `N=512`, `B=8`, and nominal 90% coverage. Averaged equally across 18 dataset/model cells and 50 seeds, row-wise coverage is 59.99%, the ESS heuristic reaches 64.63%, and the corrected method and deduplication both reach 90.64%. These are interventions on public benchmarks, not observations of production incidents.

## Environment

Python 3.12.1 and the pinned packages in `requirements.txt` reproduce the recorded numerical results. The reference run used ten CPU workers on a 12-core ARM64 system with 18 GiB memory. Use fewer workers to reduce resource use. A GPU is not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verify the included results

```bash
python tests/test_rank_bounds.py
python src/verify.py
```

The first command checks the rank envelope exhaustively and tests bounded score perturbations. The second verifies archive and processed-data hashes, split separation, and independent primary-setting threshold calculations for all 900 fitted models.

## Recompute every experiment

Run these commands from the repository root:

```bash
python src/prepare_data.py
python src/run_benchmark.py --workers 10 --output-dir results/reproduced
python src/analyze.py --results-dir results/reproduced
python src/run_simulations.py --output-dir results/reproduced
python src/verify.py --reference results --candidate results/reproduced
```

`prepare_data.py` uses the included UCI archives and can retrieve them from their canonical URLs if needed. The new output directory triggers fresh fitting and calibration. Each model uses one numerical-library thread, so workers do not compete through nested parallelism. Completed per-model outputs support resuming an interrupted run in the same output directory.

The verification comparison uses a numerical tolerance of `2e-10`. It checks every saved benchmark result and derived summary, including the simulations. Execution timings are machine-dependent measurements and are reported separately.

## Repository contents

- `src/calibration.py`: clean, row-wise, ESS, count-only, and bounded-replay thresholds; deterministic replay mechanisms.
- `src/prepare_data.py`: source download verification, exact complete-row deduplication, encoding, and canonical regression arrays.
- `src/run_benchmark.py`: 900 model fits and 297,000 paired method/scenario measurements.
- `src/run_simulations.py`: 9,396 known-distribution simulation cells, perturbation and metadata stress tests, and timing measurements.
- `src/analyze.py`: seed-block uncertainty intervals, paired contrasts, and per-cell summaries.
- `src/verify.py`: data integrity, independently calculated primary thresholds, and numerical reproduction comparisons.
- `tests/test_rank_bounds.py`: exhaustive bounded-multiplicity enumeration, sharpness, minimality, and perturbation tests.
- `config/main.json`: all seeds, models, calibration budgets, coverage targets, and eleven replay scenarios.
- `data/raw/`: original public UCI archives.
- `data/processed/`: six canonical regression tasks in NPZ and CSV formats.
- `data/source_manifest.json`: canonical source URLs, dataset DOIs, and SHA-256 archive hashes.
- `data/dataset_summary.csv`: original and retained sample sizes, input dimensions, and processed CSV hashes.
- `results/benchmark.csv`: all benchmark measurements.
- `results/scores/`: source and test scores, responses, predictions, and disjoint training/calibration/test indices for each fitted model. Indices refer to canonical processed-row order.
- `results/aggregate.csv`, `primary_cells.csv`, and `key_results.json`: paired summaries and main effect estimates.
- `results/simulation.csv`, `stress.csv`, and `timings.csv`: simulation and computational measurements.
- `results/run_metadata.json`, `simulation_metadata.json`, and `verification.json`: execution settings and validation outcomes.

## Experimental interpretation

Half of each canonical dataset is used for training, one quarter for testing, and the remaining rows supply the calibration reservoir. Calibration budgets are 64 and 128; miscoverage levels are 0.05, 0.10, and 0.20. Ridge, random forest, and histogram boosting parameters are fixed before benchmark execution. All scenarios and seeds are retained.

The primary low-score replay intentionally attains the worst-case allocation. Random, balanced, high-score, and saturated replay show the corresponding robustness/width tradeoff. Width ratios use deduplication on the same source pool as the denominator. Confidence intervals average cells within each seed and then use 50 seed blocks; they describe split variability conditional on these datasets.

The source-count and multiplicity assumptions are essential. The score-perturbation experiment distinguishes exact replay from copies whose scores change. The metadata experiment demonstrates the effect of a feasible but incorrect cap. Marginal coverage is distinct from conditional coverage, simultaneous batch coverage, and operational safety.

## Data attribution and licensing

The experimental software is licensed under MIT. The source datasets and their processed derivatives retain their UCI Creative Commons Attribution 4.0 licenses. Full source attribution and transformation details are in `data/ATTRIBUTION.md`. The primary dataset links and archive hashes are in `data/source_manifest.json`.

## Citation

The authors and software version are recorded in `CITATION.cff`, which GitHub recognizes for citation export.
