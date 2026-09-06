# cmapss-rul-prediction

Remaining Useful Life (RUL) prediction for NASA's C-MAPSS turbofan engine
degradation dataset, written in Rust.

This project is a companion to [`secom-fault-detection`](#) and
[`tep-fault-diagnosis`](#): those two answer "is something wrong *right
now*" (fault detection/diagnosis). This one answers the other half of
industrial reliability engineering — "how much running time is left before
this fails" — which is a regression problem, not a classification one, and
uses this repo to explore that in a systems language instead of Python.

**Status: Phase 3 of 6 — baseline models (linear regression, random forest).**
See [Roadmap](#roadmap).

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

\* **The official readme's FD004 counts are transposed relative to the real
files.** The readme states 248 train / 249 test; the actual files contain
249 train units (IDs 1–249, no gaps) and 248 test units (IDs 1–248), and
`RUL_FD004.txt` has exactly 248 lines, matching the test set. Totals agree
either way (497), so this reads as a documentation swap rather than missing
data — verified directly against the raw files in `tests/data_integrity.rs`,
not assumed from the readme. Every other subset's counts match the readme
exactly.

Each row is one engine-cycle: unit number, cycle number, 3 operational
settings, 21 sensor measurements (26 whitespace-delimited columns total,
confirmed by counting fields on real rows).

### Why the raw data is checked into this repo

Unlike `tep-fault-diagnosis` (whose dataset is ~1.3GB and requires a manual
download), the full C-MAPSS dataset — all four subsets, train, test, and RUL
files — is 43MB, small enough to commit directly. That means `git clone` +
`cargo run` reproduces everything with no separate download step.

The files here were pulled from the [mapr-demos/predictive-maintenance](https://github.com/mapr-demos/predictive-maintenance)
mirror of the original NASA distribution (verified byte-identical in
structure to the official readme's documented format). The original PDF
(`Damage Propagation Modeling.pdf`) is not included — see the citation above
instead.

## RUL labeling

Training trajectories run all the way to failure, so RUL at each cycle is
just "cycles remaining until the last recorded cycle." Test trajectories are
truncated before failure; `RUL_FDxxx.txt` gives the true RUL at each test
unit's *last* recorded cycle, and earlier cycles are reconstructed by adding
back cycles-until-that-point.

Raw "cycles until failure" isn't a sensible regression target very early in
an engine's life — there's no meaningful degradation trend yet. Following
the piecewise-linear convention introduced in:

> F.O. Heimes (2008). "Recurrent Neural Networks for Remaining Useful Life
> Estimation", PHM08.

RUL is flattened at a cap (default **125** cycles, the most commonly cited
value in follow-on work) until an engine is actually approaching failure,
then decays linearly. The cap is a CLI flag (`--rul-cap`), not hard-coded,
since different papers use different values.

## Phase 1 EDA findings

Running the full pipeline over all four subsets surfaces one genuinely
useful pattern: **near-constant sensors only show up in the single-condition
subsets.**

| Subset | Near-constant sensors (std < 1e-6) |
|--------|-------------------------------------|
| FD001  | 1, 5, 10, 16, 18, 19 |
| FD002  | none |
| FD003  | 1, 5, 16, 18, 19 |
| FD004  | none |

This isn't a coincidence: FD002/FD004 sweep 6 operating conditions, so
sensors that read essentially flat at a single fixed condition (FD001/FD003)
actually vary once the operating regime itself changes. This directly
foreshadows why **Phase 5** (generalizing to FD002/FD004) can't just reuse
Phase 1–4's feature set unchanged — it needs per-regime normalization first,
or these sensors look informative for the wrong reason.

Full report: `cargo run` (see below).

## Phase 2: rolling-window features

For each engine, each sensor, and each cycle with a full trailing window
available, three features are computed over that window: mean, sample std
dev, and least-squares slope against cycle index. This is run-aware by
construction, not by a leakage check bolted on afterward — the windowing
function operates on a single engine's already-grouped `EngineRun`, so
there's no code path that could pull a window from a different engine's
history (see `windowed_features_never_mix_two_engines` in the integration
tests, which pins this structurally rather than just trusting it).

**Window size (`DEFAULT_WINDOW = 10`) is constrained by the shortest test
run in the dataset, not chosen for smoothness.** FD004's shortest test run
is 19 cycles (see the EDA table above). If the window were too large
relative to that, some test engines would produce *zero* windowed feature
rows — including at their last recorded cycle, which is exactly the point
the official scoring function evaluates. `default_window_leaves_every_test_unit_with_at_least_one_windowed_row`
checks this directly against the real data rather than assuming a chosen
window is safe. Cycles before an engine's first full window are dropped
entirely (not computed from a partial window) since a std dev or slope
from 1-2 points isn't meaningful.

## Phase 3: baseline models

Two standard regressors — linear regression (`linfa`) and random forest
(`smartcore`) — evaluated on FD001 and FD003 (the two single-condition
subsets; FD002/FD004 wait for Phase 5's regime normalization, since running
them now would just be measuring how much the untreated multi-condition
noise hurts, not how good the model is). No hyperparameter tuning yet —
these are floor numbers, meant to be beaten by Phase 4's hand-built model.

**Evaluation methodology, which is easy to get wrong:** training pools every
windowed row of every training engine (normal data augmentation for this
task — ~20-24k rows). Evaluation uses exactly **one row per test engine**:
the window ending at that engine's last recorded cycle, since that's what
the official PHM08 protocol actually scores — one RUL prediction per
engine, made at the point its data was truncated. Scoring every windowed
row of a test trajectory would inflate apparent performance, since
consecutive windows from the same engine are highly correlated.

**Feature selection** excludes whichever sensors that subset's own EDA
flagged as near-constant back in Phase 1/2 — 6 sensors for FD001, 5 for
FD003 (see the EDA table above) — leaving 60-64 features (raw/mean/std/slope
× the remaining sensors). This is Phase 1's findings directly feeding Phase
3's modeling, not two disconnected steps. Operational settings are excluded
for every subset; see `core/src/design_matrix.rs` for why.

Results (no tuning, `--window 10`, `--rul-cap 125`):

| Subset | Model | RMSE (cycles) | PHM08 score | Late / early |
|--------|-------|---------------|-------------|--------------|
| FD001  | Linear regression | 20.45 | 1100.1 | 62 / 38 |
| FD001  | Random forest      | **19.39** | 1501.2 | 58 / 42 |
| FD003  | Linear regression | 19.94 | **1400.2** | 64 / 36 |
| FD003  | Random forest      | 20.42 | 2031.8 | 60 / 40 |

**A real, worth-explaining disagreement**: on FD001, random forest has the
*better* RMSE (19.39 vs 20.45) but the *worse* PHM08 score (1501.2 vs
1100.1). Its worst 3 test-set errors are all late (+60.5, +55.7, +52.7 —
predicting more remaining life than the engine actually had), while linear
regression's worst 3 split between directions. Since PHM08 scoring
penalizes late errors exponentially (divisor 10) more steeply than early
ones (divisor 13), a handful of large late outliers can dominate the sum
even when the model's *average* error is smaller — exactly the score's
known outlier sensitivity discussed in the literature, and why RMSE gets
reported alongside it rather than instead of it. This isn't a bug to fix;
it's the reason both metrics matter, and it's a fair preview of what
permutation importance / error analysis in a later phase should dig into
for random forest specifically.

Run it yourself:

```bash
cargo run -p models --release           # FD001 by default
cargo run -p models --release -- fd003
```

**Use `--release`.** A debug build spent 2.5+ minutes at 100% CPU on this
(random forest training on ~20k rows × 60 features is real numeric work);
release finishes in a couple of seconds. This isn't a Rust-specific gotcha,
but it's an easy one to hit coming from Python, where `import sklearn`
gives you optimized code regardless of how your own script is invoked.

Predictions are written to `data/processed/{subset}_test_predictions.csv`
(unit, true RUL, both models' predictions, both errors) for further
analysis.

## Charting

Static sanity-check charts live in a separate `charts` crate — see
[Project structure](#project-structure) for why it's isolated. Generate
them with:

```bash
cargo run -p charts
```

This writes several PNGs to `reports/figures/` for FD001: a bar chart of
each sensor's Pearson correlation with RUL, raw-vs-rolling-mean charts for
the 3 most-correlated sensors, one more for the sensor with the *highest
raw variance* (kept deliberately, even though it isn't top-3 by
correlation — high variance and high relevance turned out to be different
things, and the chart makes that visible instead of quietly picking a
better sensor), and the piecewise-linear RUL label shape.

**On notebooks:** the natural instinct for exploratory charting is a Jupyter
notebook, and there's a genuinely "rusty" way to get one — [`evcxr_jupyter`](https://github.com/evcxr/evcxr),
a real Jupyter kernel that runs actual Rust cells (not a Python wrapper
around Rust output). It would let this project have a proper `.ipynb`
walkthrough, calling directly into this crate's library and rendering
charts inline via `plotters` and `evcxr_display()`, the same way the
`secom-fault-detection` notebook narrates that pipeline in Python.

I couldn't verify it hands-on here — both `evcxr_jupyter` and `plotters`'
default text-rendering feature (`font-kit`) need a newer Rust edition than
this sandbox's toolchain has (this is exactly the same wall `clap` hit in
Phase 1). That's a sandbox limitation, not a verdict on the tool: it's
actively maintained and widely used. Worth trying locally; if Jupyter/Python
as the notebook *shell* (no Python code, just the UI) doesn't feel worth it,
or Windows setup is more friction than it's worth, the `charts` crate above
is the fallback - plain PNGs, zero notebook tooling, still 100% Rust.

## Dependency philosophy

**The core pipeline (`core/`) has zero external dependencies**, enforced
structurally rather than by convention: it's a separate workspace member
from `charts/` and `models/`, so nothing charting- or ML-library-related
can end up in its dependency tree even by accident. Parsing, RUL labeling,
windowed features, summary statistics, evaluation metrics, and CLI argument
handling are all hand-rolled. A CLI crate (`clap`) would normally be the
idiomatic, unremarkable choice for argument parsing — but every CLI in this
workspace has a small enough surface (one positional enum, a few flags)
that `std::env::args()` covers it without pulling in a dependency for
plumbing.

`plotters` (in `charts/`) and `linfa`/`smartcore` (in `models/`) are the
deliberate exceptions — the agreed hybrid approach: crates for
well-understood, standard algorithms (linear regression, random forest,
rendering pixels), hand-rolled code for the logic that's actually
differentiating for this project (RUL labeling, the PHM08 scoring function,
feature engineering, and — coming in Phase 4 — gradient-boosted trees built
from scratch).

## Project structure

This is a Cargo workspace with three members, split specifically so the
`plotters`/`font-kit` graphics stack and the `linfa`/`smartcore` ML crates
can't leak into the core pipeline's dependency graph (see
[Dependency philosophy](#dependency-philosophy)):

```
cmapss-rul-prediction/
├── Cargo.toml                 # workspace root (default-members = ["core"])
├── data/
│   ├── raw/CMAPSSData/         # the 12 original NASA files + readme (checked in)
│   └── processed/              # labeled/windowed/prediction CSVs, generated by `cargo run` (gitignored)
├── reports/figures/            # charts, generated by `cargo run -p charts` (gitignored)
├── core/                        # the actual pipeline - zero external dependencies
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs               # CLI: argument parsing + report printing
│   │   ├── lib.rs                 # module re-exports
│   │   ├── dataset.rs             # Subset enum + metadata (incl. the FD004 discrepancy note)
│   │   ├── parser.rs              # raw line -> CycleRecord
│   │   ├── loader.rs              # file I/O + grouping into EngineRun trajectories
│   │   ├── rul.rs                  # piecewise-linear RUL labeling (train + test)
│   │   ├── features.rs             # rolling-window feature engineering (Phase 2)
│   │   ├── design_matrix.rs        # flat feature vectors for ML libraries (Phase 3)
│   │   ├── scoring.rs               # RMSE + PHM08 asymmetric scoring function (Phase 3)
│   │   └── eda.rs                  # cycle-length, sensor-variance, sensor-RUL correlation
│   │   └── error.rs                 # hand-rolled error type
│   └── tests/data_integrity.rs   # integration tests against the real files
├── charts/                       # sanity-check chart generation (plotters lives only here)
│   ├── Cargo.toml
│   └── src/main.rs
└── models/                       # baseline models (linfa + smartcore live only here)
    ├── Cargo.toml
    └── src/main.rs
```

## Running it

All commands below assume you're in the workspace root (this directory).
Plain `cargo run`/`cargo build`/`cargo test` default to the core pipeline
(no `-p` needed) — see the `default-members` note in the root `Cargo.toml`.

```bash
# Full report for all four subsets, writes labeled + windowed-feature CSVs to data/processed/
cargo run

# One subset, report only (no CSV output)
cargo run -- fd002 --report-only

# Custom RUL cap or window size
cargo run -- fd001 --rul-cap 130 --window 15

# Sanity-check charts -> reports/figures/ (explicit -p: not a default member,
# so plain `cargo build`/`cargo run` never touches its plotters/font-kit deps)
cargo run -p charts

# Baseline models (explicit -p, same reason). Use --release - see Phase 3 section.
cargo run -p models --release
cargo run -p models --release -- fd003
```

## Testing

```bash
# Defaults to the core pipeline - genuinely zero dependencies, fast, no graphics/ML stack involved
cargo test
```

36 tests (28 unit, 8 integration), all running against the real checked-in
dataset (no synthetic fixtures for the integration tests) — unit-count
sanity checks against the readme (including the corrected FD004 numbers),
RUL monotonicity and cap enforcement, test-set RUL reconstruction against
ground truth, full parse coverage across all 8 train/test files, window-size
safety margin and structural leakage checks, and (new in Phase 3) scoring
function correctness and feature-vector construction.

## Roadmap

1. ~~Scaffold, data loading, RUL labeling, EDA~~ (Phase 1)
2. ~~Feature engineering: rolling-window statistics per sensor per engine, run-aware to prevent cross-engine leakage~~ (Phase 2)
3. ~~Baseline models (`linfa` linear regression, `smartcore` random forest), evaluated on RMSE and NASA's official asymmetric scoring function~~ (this phase)
4. Flagship model: gradient-boosted regression trees, hand-built from scratch
5. Generalization to FD002/FD004: operating-condition clustering + per-regime normalization, documented as an explicit extension (see EDA findings above)
6. *(stretch)* Sequence modeling (LSTM via `candle`/`burn`), the literature-standard approach for this dataset
