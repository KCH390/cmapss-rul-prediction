# cmapss-rul-prediction

Remaining Useful Life (RUL) prediction for NASA's C-MAPSS turbofan engine
degradation dataset, written in Rust.

This project is a companion to [`secom-fault-detection`](#) and
[`tep-fault-diagnosis`](#): those two answer "is something wrong *right
now*" (fault detection/diagnosis). This one answers the other half of
industrial reliability engineering — "how much running time is left before
this fails" — which is a regression problem, not a classification one, and
uses this repo to explore that in a systems language instead of Python.

**Status: Phase 2 of 6 — rolling-window feature engineering.** No modeling
yet; see [Roadmap](#roadmap).

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

## Charting

Static sanity-check charts (raw sensor trace vs. rolling mean, RUL label
shape) live in a separate `charts` crate — see [Project structure](#project-structure)
for why it's isolated. Generate them with:

```bash
cargo run -p charts
```

This writes PNGs to `reports/figures/`.

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
from `charts/`, so nothing charting-related can end up in its dependency
tree even by accident. Parsing, RUL labeling, windowed features, summary
statistics, and CLI argument handling are all hand-rolled. A CLI crate
(`clap`) would normally be the idiomatic, unremarkable choice for argument
parsing — but this CLI's surface is small enough (one positional enum, a
few flags) that `std::env::args()` covers it without pulling in a dependency
for plumbing.

`plotters` (in the separate `charts` crate) is the one deliberate exception:
rendering pixels and text is genuinely plumbing, not differentiating logic.
Later pipeline phases will pull in crates where they're actually earning
their place too (e.g. `linfa`/`smartcore` for baseline models in Phase 3) —
see the hybrid approach in the Roadmap.

## Project structure

This is a Cargo workspace with two members, split specifically so the
`plotters`/`font-kit` graphics stack can't leak into the core pipeline's
dependency graph (see [Dependency philosophy](#dependency-philosophy)):

```
cmapss-rul-prediction/
├── Cargo.toml                 # workspace root
├── data/
│   ├── raw/CMAPSSData/         # the 12 original NASA files + readme (checked in)
│   └── processed/              # labeled + windowed-feature CSVs, generated by `cargo run` (gitignored)
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
│   │   └── eda.rs                  # cycle-length and sensor-variance statistics
│   │   └── error.rs                 # hand-rolled error type
│   └── tests/data_integrity.rs   # integration tests against the real files
└── charts/                       # sanity-check chart generation (plotters lives only here)
    ├── Cargo.toml
    └── src/main.rs
```

## Running it

All commands below assume you're in the workspace root (this directory).

```bash
# Full report for all four subsets, writes labeled + windowed-feature CSVs to data/processed/
cargo run -p cmapss-rul-prediction

# One subset, report only (no CSV output)
cargo run -p cmapss-rul-prediction -- fd002 --report-only

# Custom RUL cap or window size
cargo run -p cmapss-rul-prediction -- fd001 --rul-cap 130 --window 15

# Sanity-check charts -> reports/figures/
cargo run -p charts
```

## Testing

```bash
# Core pipeline only - genuinely zero dependencies, fast, no graphics stack involved
cargo test -p cmapss-rul-prediction
```

24 tests (16 unit, 8 integration), all running against the real checked-in
dataset (no synthetic fixtures for the integration tests) — unit-count
sanity checks against the readme (including the corrected FD004 numbers),
RUL monotonicity and cap enforcement, test-set RUL reconstruction against
ground truth, full parse coverage across all 8 train/test files, and (new
in Phase 2) window-size safety margin and structural leakage checks.

## Roadmap

1. ~~Scaffold, data loading, RUL labeling, EDA~~ (Phase 1)
2. ~~Feature engineering: rolling-window statistics per sensor per engine, run-aware to prevent cross-engine leakage~~ (this phase)
3. Baseline models (crates: `linfa` linear regression, `smartcore` random forest), evaluated on RMSE and NASA's official asymmetric scoring function
4. Flagship model: gradient-boosted regression trees, hand-built from scratch
5. Generalization to FD002/FD004: operating-condition clustering + per-regime normalization, documented as an explicit extension (see EDA findings above)
6. *(stretch)* Sequence modeling (LSTM via `candle`/`burn`), the literature-standard approach for this dataset
