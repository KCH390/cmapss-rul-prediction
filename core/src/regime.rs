//! Operating-condition clustering and per-regime sensor normalization.
//!
//! This is what Phase 1's EDA foreshadowed: FD002/FD004 sweep 6 operating
//! conditions, so a sensor that reads flat within one fixed condition
//! (FD001/FD003) actually swings across conditions here — not because of
//! degradation, but because of which flight regime the engine happens to
//! be in at that cycle. Clustering the 3 operational settings into regimes
//! and normalizing each sensor within its own regime removes that
//! condition-induced variation, so what's left in the windowed features
//! (Phase 2) is (closer to) pure degradation signal.
//!
//! Regime membership and normalization statistics are fit on **training
//! data only** and then applied to both train and test — fitting them on
//! test data too would leak test-set information into the transformation
//! test predictions are later evaluated against.

use crate::kmeans::{self, KMeans};
use crate::loader::EngineRun;
use crate::parser::{CycleRecord, NUM_SENSORS};

/// Fixed seed for k-means++ initialization, so regime assignment is exactly
/// reproducible across runs (see `kmeans.rs`).
pub const KMEANS_SEED: u64 = 42;
pub const KMEANS_MAX_ITERATIONS: usize = 100;

/// Clusters every training cycle's 3 operational settings into `k` regimes.
/// `k` is normally `subset.num_conditions()` (1 for FD001/FD003, 6 for
/// FD002/FD004) - for a single-condition subset this degenerates to one
/// cluster containing everything, making normalization a harmless global
/// standardization rather than a no-op with special-cased code.
pub fn fit_regimes(train_runs: &[EngineRun], k: usize) -> KMeans {
    let points: Vec<Vec<f64>> = train_runs
        .iter()
        .flat_map(|r| r.records.iter().map(|rec| rec.op_settings.to_vec()))
        .collect();
    kmeans::fit(&points, k, KMEANS_MAX_ITERATIONS, KMEANS_SEED)
}

/// Per-regime, per-sensor mean and standard deviation, computed from
/// training data only.
pub struct RegimeStats {
    pub mean: Vec<[f64; NUM_SENSORS]>,
    pub std: Vec<[f64; NUM_SENSORS]>,
    /// Number of training cycles assigned to each regime - useful for
    /// sanity-checking that clustering found genuinely separated, non-tiny
    /// groups rather than one dominant cluster and a few stragglers.
    pub counts: Vec<usize>,
}

pub fn compute_regime_stats(train_runs: &[EngineRun], regimes: &KMeans) -> RegimeStats {
    let k = regimes.k();
    let mut sums = vec![[0.0; NUM_SENSORS]; k];
    let mut sq_sums = vec![[0.0; NUM_SENSORS]; k];
    let mut counts = vec![0usize; k];

    for run in train_runs {
        for rec in &run.records {
            let regime = regimes.predict(&rec.op_settings);
            counts[regime] += 1;
            for s in 0..NUM_SENSORS {
                sums[regime][s] += rec.sensors[s];
                sq_sums[regime][s] += rec.sensors[s].powi(2);
            }
        }
    }

    let mut mean = vec![[0.0; NUM_SENSORS]; k];
    let mut std = vec![[0.0; NUM_SENSORS]; k];
    for r in 0..k {
        let n = (counts[r].max(1)) as f64;
        for s in 0..NUM_SENSORS {
            mean[r][s] = sums[r][s] / n;
            // Population variance (not sample/n-1): consistent within this
            // module's own use (normalizing, not inferring a population
            // parameter), and counts per regime are large enough that the
            // n vs n-1 choice doesn't matter in practice.
            let variance = (sq_sums[r][s] / n) - mean[r][s].powi(2);
            std[r][s] = variance.max(0.0).sqrt();
        }
    }

    RegimeStats { mean, std, counts }
}

/// Produces a copy of `run` with every sensor z-scored against its
/// assigned regime's training-set mean/std: `(raw - regime_mean) /
/// regime_std`. Unit, cycle, and op_settings are unchanged - only the
/// sensor values are transformed, so the result can be fed straight into
/// the existing `features`/`rul` pipeline without any other code changes.
pub fn normalize_run(run: &EngineRun, regimes: &KMeans, stats: &RegimeStats) -> EngineRun {
    let records = run
        .records
        .iter()
        .map(|rec| {
            let regime = regimes.predict(&rec.op_settings);
            let mut sensors = [0.0; NUM_SENSORS];
            for s in 0..NUM_SENSORS {
                let std_dev = stats.std[regime][s];
                sensors[s] = if std_dev < 1e-9 {
                    // Sensor is constant within this regime - there's no
                    // meaningful z-score to compute, and dividing by
                    // ~zero would blow up into noise. 0.0 keeps this
                    // feature column inert rather than producing garbage.
                    0.0
                } else {
                    (rec.sensors[s] - stats.mean[regime][s]) / std_dev
                };
            }
            CycleRecord {
                unit: rec.unit,
                cycle: rec.cycle,
                op_settings: rec.op_settings,
                sensors,
            }
        })
        .collect();
    EngineRun { unit: run.unit, records }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::NUM_SENSORS as N;

    fn make_run(unit: u32, cycles: &[(f64, f64)]) -> EngineRun {
        // cycles: (op_setting_1, sensor_1_value) pairs; other fields zeroed.
        EngineRun {
            unit,
            records: cycles
                .iter()
                .enumerate()
                .map(|(i, &(op1, sensor1))| {
                    let mut sensors = [0.0; N];
                    sensors[0] = sensor1;
                    CycleRecord {
                        unit,
                        cycle: (i + 1) as u32,
                        op_settings: [op1, 0.0, 0.0],
                        sensors,
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn normalization_removes_a_pure_regime_level_shift() {
        // Two regimes (op_setting_1 = 0.0 vs 10.0), sensor 1 reads exactly
        // 100 higher in regime 2 regardless of "degradation" - a pure
        // condition-induced offset, the thing regime normalization exists
        // to remove. Add small within-regime variation so std isn't zero.
        let run = make_run(
            1,
            &[
                (0.0, 50.0), (0.0, 51.0), (0.0, 49.0), (0.0, 50.0),
                (10.0, 150.0), (10.0, 151.0), (10.0, 149.0), (10.0, 150.0),
            ],
        );
        let regimes = fit_regimes(&[run.clone()], 2);
        let stats = compute_regime_stats(&[run.clone()], &regimes);
        let normalized = normalize_run(&run, &regimes, &stats);

        // After normalization, both regimes' sensor 1 values should be
        // centered near 0, not 100 apart.
        let regime_a_normalized: Vec<f64> = normalized.records[0..4].iter().map(|r| r.sensors[0]).collect();
        let regime_b_normalized: Vec<f64> = normalized.records[4..8].iter().map(|r| r.sensors[0]).collect();
        let mean_a = regime_a_normalized.iter().sum::<f64>() / 4.0;
        let mean_b = regime_b_normalized.iter().sum::<f64>() / 4.0;
        assert!(mean_a.abs() < 1e-6, "regime A should normalize to ~0 mean, got {}", mean_a);
        assert!(mean_b.abs() < 1e-6, "regime B should normalize to ~0 mean, got {}", mean_b);
    }

    #[test]
    fn constant_sensor_within_a_regime_normalizes_to_zero_not_nan_or_infinity() {
        let run = make_run(1, &[(0.0, 42.0), (0.0, 42.0), (0.0, 42.0)]);
        let regimes = fit_regimes(&[run.clone()], 1);
        let stats = compute_regime_stats(&[run.clone()], &regimes);
        let normalized = normalize_run(&run, &regimes, &stats);
        for rec in &normalized.records {
            assert_eq!(rec.sensors[0], 0.0);
            assert!(!rec.sensors[0].is_nan());
        }
    }

    #[test]
    fn k_equals_one_normalizes_against_a_single_global_regime() {
        // With k=1 (the FD001/FD003 case), every record shares the same
        // regime, so this should just be standard global z-scoring.
        let run = make_run(1, &[(0.0, 10.0), (0.0, 20.0), (0.0, 30.0)]);
        let regimes = fit_regimes(&[run.clone()], 1);
        assert_eq!(regimes.k(), 1);
        let stats = compute_regime_stats(&[run.clone()], &regimes);
        let normalized = normalize_run(&run, &regimes, &stats);
        let mean: f64 = normalized.records.iter().map(|r| r.sensors[0]).sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-9);
    }

    #[test]
    fn stats_are_fit_from_training_data_and_reused_unchanged_for_test_data() {
        // A "test" run in a different regime distribution shouldn't change
        // the mean/std used to normalize it - those come from train only.
        let train_run = make_run(1, &[(0.0, 50.0), (0.0, 52.0), (10.0, 150.0), (10.0, 148.0)]);
        let regimes = fit_regimes(&[train_run.clone()], 2);
        let stats = compute_regime_stats(&[train_run.clone()], &regimes);

        let test_run = make_run(2, &[(0.0, 51.0)]); // a single test cycle in regime "0"
        let normalized_test = normalize_run(&test_run, &regimes, &stats);
        // Expected: (51 - train_regime_0_mean) / train_regime_0_std, using
        // ONLY train_run's regime-0 stats (mean 51, std 1 from the two
        // regime-0 training points 50 and 52).
        let expected = (51.0 - 51.0) / 1.0;
        assert!((normalized_test.records[0].sensors[0] - expected).abs() < 1e-9);
    }
}
