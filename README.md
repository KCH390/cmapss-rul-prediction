# cmapss-rul-prediction

Remaining Useful Life (RUL) prediction for NASA's C-MAPSS turbofan engine
degradation dataset, written in Rust.

This project is a companion to `secom-fault-detection` and
`tennessee-eastman-fdd`: those two answer "is something wrong right now"
(fault detection/diagnosis). This one answers the other half of industrial
reliability engineering — "how much running time is left before this
fails" — a regression problem rather than a classification one.

## Status

- Data loading, parsing, and RUL labeling
- Rolling-window feature engineering
- Baseline models (linear regression, random forest)
- Flagship model: gradient-boosted trees, built from scratch
- Regime normalization and generalization to the multi-condition subsets (FD002/FD004)
- LSTM sequence model — implemented, not yet validated end-to-end (see [LSTM sequence model](#lstm-sequence-model-experimental))

## Dataset

NASA Ames Prognostics Center of Excellence (PCoE), *Turbofan Engine
Degradation Simulation Data Set*. Citation:

> A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation
> Data Set", NASA Prognostics Data Repository, NASA Ames Research Center,
> Moffett Field, CA.
> Official repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Four sub-datasets, each with training and test splits plus a ground-truth
RUL file for the test split:

| Subset | Operating conditions | Fault modes | Train units | Test units |
|--------|----------------------|-------------|-------------|------------|
| FD001  | 1 (sea level)         | 1 (HPC)              | 100 | 100 |
| FD002  | 6                     | 1 (HPC)              | 260 | 259 |
| FD003  | 1 (sea level)         | 2 (HPC, Fan)         | 100 | 100 |
| FD004  | 6                     | 2 (HPC, Fan)         | 249\* | 248\* |

\* NASA's own readme states FD004 has 248 training / 249 test trajectories.
The actual files contain 249 training units (IDs 1–249, no gaps) and 248
test units (IDs 1–248), and `RUL_FD004.txt` has exactly 248 lines, matching
the test set. The totals agree either way (497), so this looks like a
transcription swap in the documentation rather than missing data. Every
other subset's counts match the readme exactly. This is pinned by a test in
`core/tests/data_integrity.rs` and reflected in `dataset.rs`'s metadata.

Each row is one engine-cycle: unit number, cycle number, 3 operational
settings, 21 sensor measurements (26 whitespace-delimited columns total).

### Data provenance

The full dataset — all four subsets, train, test, and RUL files — is 43MB
and is checked directly into `data/raw/CMAPSSData/`, pulled from the
[mapr-demos/predictive-maintenance](https://github.com/mapr-demos/predictive-maintenance)
mirror of the original NASA distribution. `git clone` + `cargo run`
reproduces everything with no separate download step. The original PDF
(`Damage Propagation Modeling.pdf`) is not included — see the citation
above instead.

## RUL labeling

Training trajectories run all the way to failure, so RUL at each cycle is
"cycles remaining until the last recorded cycle." Test trajectories are
truncated before failure; `RUL_FDxxx.txt` gives the true RUL at each test
unit's last recorded cycle, and earlier cycles are reconstructed by adding
back cycles-until-that-point.

Raw "cycles until failure" isn't a sensible regression target early in an
engine's life, since there's no meaningful degradation trend yet. Following
the piecewise-linear convention introduced in Heimes (2008), "Recurrent
Neural Networks for Remaining Useful Life Estimation" (PHM08), RUL is
flattened at a cap (default 125 cycles, the most commonly cited value in
follow-on work) until an engine is actually approaching failure, then
decays linearly. The cap is a CLI flag (`--rul-cap`), not hard-coded, since
different papers use different values.

## Feature engineering

For each engine, sensor, and cycle with a full trailing window available,
three rolling statistics are computed: mean, sample standard deviation, and
least-squares slope against cycle index. Windowing is run-aware by
construction — it operates on a single engine's already-grouped
trajectory, so there's no code path that could pull a window from a
different engine's history (`windowed_features_never_mix_two_engines`
pins this structurally).

Window size (`DEFAULT_WINDOW = 10`) is constrained by the shortest test run
in the dataset rather than chosen for smoothness: FD004's shortest test run
is 19 cycles, and a larger window would leave some test engines with zero
windowed rows — including at their last recorded cycle, which is exactly
the point the scoring function evaluates. This is enforced by an
integration test against the real data rather than just assumed. Cycles
before an engine's first full window are dropped entirely (not computed
from a partial window), since a standard deviation or slope from one or
two points isn't meaningful.

## Exploratory findings

Near-constant sensors (std < 1e-6) appear only in the single-condition
subsets:

| Subset | Near-constant sensors |
|--------|-------------------------------------|
| FD001  | 1, 5, 10, 16, 18, 19 |
| FD002  | none |
| FD003  | 1, 5, 16, 18, 19 |
| FD004  | none |

This isn't a coincidence: FD002/FD004 sweep 6 operating conditions, so
sensors that read essentially flat at a single fixed condition actually
vary once the operating regime itself changes. This is why the
multi-condition subsets need per-regime normalization (see below) before
the same feature set is meaningful — otherwise these sensors look
informative for the wrong reason.

Full report: `cargo run`.

## Evaluation methodology

Training pools every windowed row of every training engine (ordinary data
augmentation — tens of thousands of rows per subset). Evaluation uses
exactly one row per test engine: the window ending at that engine's last
recorded cycle. This matches the official PHM08 competition protocol,
which scores one RUL prediction per engine, made at the point its data was
truncated — scoring every windowed row of a test trajectory would inflate
apparent performance, since consecutive windows from the same engine are
highly correlated.

Two metrics are reported throughout:

- **RMSE** — symmetric; over- and under-predicting by the same amount cost
  the same.
- **PHM08 score** — NASA's official asymmetric scoring function. For each
  prediction, `d = predicted − true`; early predictions (`d < 0`, calling
  for maintenance sooner than necessary) are penalized as `exp(-d/13) - 1`,
  late predictions (`d ≥ 0`, running an engine past its actual failure
  point) as the steeper `exp(d/10) - 1`. Lower is better; 0 is perfect.

The two metrics can disagree (see Results below), which is why both are
always reported together rather than either alone.

Feature selection excludes whichever sensors a subset's own training data
shows to be near-constant (per the table above), leaving 60–64 features
per row (raw value, rolling mean, rolling std, rolling slope × the
remaining sensors). Operational settings are excluded from the feature
vector for every subset — see `core/src/design_matrix.rs`.

## Models

### Baseline models

Linear regression (`linfa`) and random forest (`smartcore`), evaluated on
FD001 and FD003 (FD002/FD004 need regime normalization first — see below).
No hyperparameter tuning.

| Subset | Model | RMSE (cycles) | PHM08 score | Late / early |
|--------|-------|---------------|-------------|--------------|
| FD001  | Linear regression | 20.45 | 1100.1 | 62 / 38 |
| FD001  | Random forest      | **19.39** | 1501.2 | 58 / 42 |
| FD003  | Linear regression | 19.94 | **1400.2** | 64 / 36 |
| FD003  | Random forest      | 20.42 | 2031.8 | 60 / 40 |

RMSE and the PHM08 score disagree here: on FD001, random forest has the
better RMSE (19.39 vs 20.45) but the worse score (1501.2 vs 1100.1). Its
three worst test-set errors are all in the late direction (+60.5, +55.7,
+52.7), while linear regression's worst errors split between directions.
Because the score penalizes late errors exponentially more steeply than
early ones, a handful of large late outliers can dominate the sum even when
the model's average error is smaller — a known sensitivity of this scoring
function, and the reason RMSE is reported alongside it rather than instead
of it.

```bash
cargo run -p models --release           # FD001 by default
cargo run -p models --release -- fd003
```

Debug builds are noticeably slower for this kind of numeric workload
(fitting a random forest on tens of thousands of rows is real work) — use
`--release`.

Predictions are written to `data/processed/{subset}_test_predictions.csv`.

### Flagship model: gradient-boosted trees

A CART-style regression tree (`core/src/tree.rs`) and gradient boosting on
top of it (`core/src/boosting.rs`), built with no dependencies beyond
`std`. Splitting uses variance-reduction via a sort-then-scan search
(O(n log n) per feature per node): candidate splits are found by sorting
flat `(value, target, index)` tuples directly rather than sorting indices
through a comparator that reaches back into the feature matrix — the
latter is markedly slower at this scale, since it defeats cache locality
across every comparison in the sort. Because the implementation uses no
bootstrap sampling or feature subsampling, fitting is exactly
deterministic: identical input always produces identical output.

This needs no dependencies beyond `std`, so it lives directly in `core` as
a second binary (`core/src/bin/flagship.rs`, auto-discovered by Cargo)
rather than a separate crate — unlike the baseline models above, which
needed `models` specifically to keep `linfa`/`smartcore` isolated from
`core`'s dependency graph.

Same evaluation methodology and feature set as the baselines, for a direct
comparison:

| Subset | Model | RMSE (cycles) | PHM08 score |
|--------|-------|---------------|-------------|
| FD001  | Linear regression | 20.45 | 1100.1 |
| FD001  | Random forest      | 19.39 | 1501.2 |
| FD001  | **Flagship GBM (from scratch)** | **18.07** | **982.8** |
| FD003  | Linear regression | 19.94 | **1400.2** |
| FD003  | Random forest      | 20.42 | 2031.8 |
| FD003  | **Flagship GBM (from scratch)** | **19.31** | 1606.8 |

FD001 is a clean sweep: the flagship model beats both baselines on both
metrics. FD003 is more mixed — flagship has the best RMSE, but linear
regression still edges it out on the PHM08 score, the same RMSE/score
tension seen with the baselines.

Default hyperparameters (`n_trees=100, learning_rate=0.1, max_depth=3,
min_samples_leaf=20`) follow typical GBM conventions (e.g. scikit-learn's
`GradientBoostingRegressor`) and are not exhaustively tuned.

```bash
cargo build -p cmapss-rul-prediction --bin flagship --release
./target/release/flagship          # FD001 by default
./target/release/flagship fd003
./target/release/flagship fd001 --n-trees 200 --learning-rate 0.05 --max-depth 4
```

Writes `data/processed/{subset}_flagship_predictions.csv` and
`{subset}_flagship_training_curve.csv` (training RMSE after each tree is
added — see [Charting](#charting) for a plotted version).

### Regime normalization (FD002 / FD004)

FD002/FD004 sweep 6 operating conditions, so a sensor that reads flat
within one fixed condition (FD001/FD003) actually swings across
conditions here — not from degradation, but from which flight regime the
engine happens to be in at that cycle.

K-means (`core/src/kmeans.rs`, k-means++ initialization, a seeded PRNG for
reproducible fitting — zero dependencies, consistent with the rest of
`core`) clusters the 3 operational settings into `k` regimes: `k=6` for
FD002/FD004, `k=1` for FD001/FD003 (a harmless global standardization).
Each sensor is then z-scored against its own regime's training-set mean
and standard deviation (`core/src/regime.rs`); regime statistics are fit
on training data only and reused unchanged for test data, since fitting on
test data too would leak information into the transformation test
predictions are later evaluated against.

Fitting `k=6` on FD002's real training data finds six cleanly separated
regimes with clean, round centroid values and reasonably balanced sizes:

```
regime: centroid (op_setting_1, op_setting_2, op_setting_3)   count
 0.0015,  0.0005, 100.00   8044
10.0030,  0.2505, 100.00   8096
20.0030,  0.7005, 100.00   8122
25.0030,  0.6205,  60.00   8002
35.0030,  0.8405, 100.00   8037
42.0030,  0.8405, 100.00  13458
```

(the larger cluster is plausibly a more commonly-visited flight phase such
as cruise). This is pinned by an integration test against the real data,
not just assumed.

**Note on feature selection with normalization enabled**: near-constant
sensor exclusion must be computed from raw data, before normalization.
Z-scoring always produces roughly unit variance from whatever it's given,
so a sensor whose raw standard deviation sits just above the "treat as
zero" cutoff would be normalized into a full-variance column and stop
being excluded, even though it remains physically uninformative. Both
`models` and `flagship` compute the exclusion list before normalizing, for
this reason.

**Note on the transform's invariance**: with `k=1` (FD001/FD003), regime
normalization is pure global z-scoring — a per-feature affine transform
that shouldn't change predictions from either OLS or a tree model in exact
arithmetic. In practice, linear regression is exactly invariant (identical
RMSE to the displayed precision), but tree-based models — including the
flagship model, which has no randomness anywhere — show small differences
(flagship RMSE on FD001: 18.07 vs 17.94). This isn't a bug: z-scoring
introduces new floating-point rounding at every value, and split search
makes discrete branching decisions, so a candidate split that's an exact
tie in real-number arithmetic can resolve to a different (still valid,
comparably good) split once rounding breaks the tie differently. Smooth
computations like OLS's matrix solve don't have this sensitivity; discrete
ones like tree splitting do.

Results, `--normalize` vs raw (same evaluation methodology throughout):

| Subset | Model | RMSE (raw) | Score (raw) | RMSE (normalized) | Score (normalized) |
|--------|-------|-----------:|-------------:|-------------------:|---------------------:|
| FD002  | Linear regression      | 19.29 | 1764.8 | **18.64** | **1749.8** |
| FD002  | Random forest          | 20.32 | 3290.0 | **17.49** | **1836.1** |
| FD002  | Flagship GBM           | 18.68 | 2307.7 | **16.14** | **1451.3** |
| FD004  | Linear regression      | 22.08 | 2665.8 | *fit failed* | *fit failed* |
| FD004  | Random forest          | 20.29 | 3160.7 | **19.79** | 3369.2 |
| FD004  | Flagship GBM           | 20.32 | 2628.9 | **18.95** | 3212.8 |

FD002 is a clean win: normalization improves every model on every metric,
most notably random forest's score (3290.0 → 1836.1, nearly halved).

FD004 is genuinely harder and the result is mixed: RMSE improves for both
tree-based models, but the PHM08 score gets worse for both (more/larger
late-direction errors even as the average error drops — the same
RMSE/score tension seen elsewhere). Linear regression fails to fit at all
on normalized FD004: `linfa` returns a `NonInvertible` error, meaning the
design matrix is genuinely singular. FD004 combines 6 operating conditions
and 2 fault modes — the most complex of the four subsets — and per-regime
z-scoring appears to introduce near-perfect collinearity between some
features under that combination. `models` handles this gracefully: a
failed linear regression fit is reported and skipped rather than aborting
the whole run, so random forest's results aren't lost to a sibling model's
failure.

```bash
cargo run -p models --release -- fd002 --normalize
./target/release/flagship fd004 --normalize
```

### LSTM sequence model (experimental)

`core/src/sequence.rs` extracts raw, non-aggregated sliding-window
sequences — the `(window_size × features)` matrix an LSTM consumes,
rather than the aggregated rolling statistics the tree/linear models use.
This part follows the same standard as the rest of `core`: zero
dependencies, unit tested, windowing rules kept consistent with
`features.rs` (same run-aware guarantee, same last-cycle evaluation
convention).

The model itself (`sequence/src/main.rs`, using `candle`) lives in its own
workspace member — the same isolation pattern as `charts` and `models`,
keeping the deep-learning dependency stack out of `core`. It reuses the
regime-normalization machinery above for input scaling: gradient-based
training needs properly scaled inputs in a way tree models don't, and
`k=1` for FD001/FD003 provides exactly that (plain global z-scoring),
while FD002/FD004 get the regime-normalization benefit for free. The
training loop does not currently shuffle minibatches between epochs — a
simplification worth revisiting.

**This component has not yet been run end-to-end, and no results are
reported for it here.** Treat it as a first implementation rather than a
validated model.

```bash
cargo build -p sequence --release
./target/release/sequence fd001
```

## Charting

Static charts are generated by a separate `charts` crate, isolated from
`core` the same way `models` is:

```bash
cargo run -p charts
```

This writes several PNGs to `reports/figures/` for FD001: a bar chart of
each sensor's Pearson correlation with RUL; raw-vs-rolling-mean charts for
the 3 most-correlated sensors, plus one for the sensor with the highest
raw variance (kept deliberately as a contrast — high variance and high
relevance turned out to be different things); the piecewise-linear RUL
label shape; the flagship model's training-RMSE-vs-trees-added convergence
curve; and two prediction-effectiveness charts evaluated on the real test
set — predicted vs. actual RUL (colored by late/early direction, since the
scoring function treats them asymmetrically) and residuals vs. true RUL.
Charts are trained and evaluated live with the same defaults as the
corresponding binaries, so the numbers shown always match what those
binaries themselves report.

## Dependency philosophy

The core pipeline (`core/`) has zero external dependencies — including the
flagship model and the k-means/regime-normalization code — enforced
structurally rather than by convention: `charts/`, `models/`, and
`sequence/` are separate workspace members specifically so nothing
charting-, ML-library-, or deep-learning-related can end up in `core`'s
dependency tree even by accident. Parsing, RUL labeling, windowed
features, summary statistics, evaluation metrics, the regression tree and
gradient boosting implementation, k-means clustering, sequence extraction,
and CLI argument handling are all hand-rolled — every CLI in this
workspace has a small enough surface (one positional enum, a few flags)
that `std::env::args()` covers it without a dependency for plumbing.

`plotters` (in `charts/`) and `linfa`/`smartcore` (in `models/`) are the
deliberate exceptions: crates for well-understood, standard algorithms
(linear regression, random forest, rendering pixels) where a library isn't
the interesting part, hand-rolled code for the logic that is — RUL
labeling, the PHM08 scoring function, feature engineering, and the
flagship model itself.

## Project structure

A Cargo workspace with four members, split so the `plotters`, `linfa`/
`smartcore`, and `candle` dependency stacks can't leak into the core
pipeline's dependency graph:

```
cmapss-rul-prediction/
├── Cargo.toml                 # workspace root (default-members = ["core"])
├── data/
│   ├── raw/CMAPSSData/         # the 12 original NASA files + readme (checked in)
│   └── processed/              # labeled/windowed/prediction CSVs, generated by `cargo run` (gitignored)
├── reports/figures/            # charts, generated by `cargo run -p charts`
├── core/                        # the pipeline itself - zero external dependencies
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs               # CLI: argument parsing + report printing
│   │   ├── lib.rs                 # module re-exports
│   │   ├── dataset.rs             # Subset enum + metadata (incl. the FD004 discrepancy note)
│   │   ├── parser.rs              # raw line -> CycleRecord
│   │   ├── loader.rs              # file I/O + grouping into EngineRun trajectories
│   │   ├── rul.rs                  # piecewise-linear RUL labeling (train + test)
│   │   ├── features.rs             # rolling-window feature engineering
│   │   ├── design_matrix.rs        # flat feature vectors for ML libraries
│   │   ├── scoring.rs               # RMSE + PHM08 asymmetric scoring function
│   │   ├── tree.rs                  # CART regression tree, hand-built
│   │   ├── boosting.rs               # gradient boosting on top of tree.rs
│   │   ├── kmeans.rs                  # k-means clustering, hand-built
│   │   ├── regime.rs                   # operating-regime normalization for FD002/FD004
│   │   ├── sequence.rs                  # raw sliding-window sequences for the LSTM
│   │   ├── eda.rs                  # cycle-length, sensor-variance, sensor-RUL correlation
│   │   └── error.rs                 # hand-rolled error type
│   │   └── bin/flagship.rs           # flagship model CLI (auto-discovered by Cargo)
│   └── tests/data_integrity.rs   # integration tests against the real files
├── charts/                       # chart generation (plotters lives only here)
│   ├── Cargo.toml
│   └── src/main.rs
├── models/                       # baseline models (linfa + smartcore live only here)
│   ├── Cargo.toml
│   └── src/main.rs
└── sequence/                     # LSTM sequence model (candle lives only here)
    ├── Cargo.toml
    └── src/main.rs
```

## Running it

All commands below assume you're in the workspace root. Plain `cargo run`/
`cargo build`/`cargo test` default to the core pipeline (no `-p` needed) —
see `default-members` in the root `Cargo.toml`.

```bash
# Full report for all four subsets, writes labeled + windowed-feature CSVs to data/processed/
cargo run

# One subset, report only (no CSV output)
cargo run -- fd002 --report-only

# Custom RUL cap or window size
cargo run -- fd001 --rul-cap 130 --window 15

# Charts -> reports/figures/ (explicit -p: not a default member, so plain
# `cargo build`/`cargo run` never touches its dependencies)
cargo run -p charts

# Baseline models (explicit -p, same reason). Use --release.
cargo run -p models --release
cargo run -p models --release -- fd003

# Flagship model (in core, zero new dependencies). Also use --release.
cargo build -p cmapss-rul-prediction --bin flagship --release
./target/release/flagship
./target/release/flagship fd003

# LSTM sequence model (experimental, unverified)
cargo build -p sequence --release
./target/release/sequence
```

## Testing

```bash
# Defaults to the core pipeline - zero dependencies, fast
cargo test
```

63 tests (54 unit, 9 integration), all running against the real checked-in
dataset — unit-count sanity checks against the readme (including the
corrected FD004 numbers), RUL monotonicity and cap enforcement, test-set
RUL reconstruction against ground truth, full parse coverage across all 8
train/test files, window-size safety-margin and structural leakage checks,
scoring-function correctness, feature-vector construction, regression-tree
split correctness and constraint enforcement, gradient-boosting
convergence/determinism, k-means clustering correctness/determinism,
regime-normalization correctness, empirical validation that `k=6` finds
genuinely separated, balanced regimes against real FD002/FD004 data, and
sequence-extraction correctness for the LSTM.
