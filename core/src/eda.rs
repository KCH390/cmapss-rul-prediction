use crate::loader::EngineRun;
use crate::parser::NUM_SENSORS;

/// Min/mean/max over a set of run lengths (in cycles).
#[derive(Debug, Clone, Copy)]
pub struct CycleLengthStats {
    pub min: u32,
    pub max: u32,
    pub mean: f64,
}

pub fn cycle_length_stats(runs: &[EngineRun]) -> CycleLengthStats {
    let lengths: Vec<u32> = runs.iter().map(|r| r.last_cycle()).collect();
    let min = *lengths.iter().min().expect("no runs given");
    let max = *lengths.iter().max().expect("no runs given");
    let mean = lengths.iter().sum::<u32>() as f64 / lengths.len() as f64;
    CycleLengthStats { min, max, mean }
}

/// Mean and (sample) standard deviation for one sensor channel across every
/// cycle of every run passed in.
#[derive(Debug, Clone, Copy)]
pub struct SensorStats {
    pub index: usize,
    pub mean: f64,
    pub std_dev: f64,
}

/// A sensor is flagged "near-constant" if its sample std dev falls below
/// this threshold. It's a well-known property of C-MAPSS that a handful of
/// sensors carry ~no signal in some subsets — but which ones, and in which
/// subsets, is computed here from the actual data rather than asserted from
/// memory of the literature.
pub const NEAR_CONSTANT_STD_THRESHOLD: f64 = 1e-6;

/// Computes per-sensor mean/std across every cycle of every run.
pub fn sensor_stats(runs: &[EngineRun]) -> [SensorStats; NUM_SENSORS] {
    let mut sums = [0.0f64; NUM_SENSORS];
    let mut count = 0usize;

    for run in runs {
        for record in &run.records {
            for i in 0..NUM_SENSORS {
                sums[i] += record.sensors[i];
            }
            count += 1;
        }
    }
    let means: Vec<f64> = sums.iter().map(|s| s / count as f64).collect();

    let mut sq_diff_sums = [0.0f64; NUM_SENSORS];
    for run in runs {
        for record in &run.records {
            for i in 0..NUM_SENSORS {
                let diff = record.sensors[i] - means[i];
                sq_diff_sums[i] += diff * diff;
            }
        }
    }

    let mut stats = [SensorStats {
        index: 0,
        mean: 0.0,
        std_dev: 0.0,
    }; NUM_SENSORS];

    for i in 0..NUM_SENSORS {
        // Sample standard deviation (n-1 denominator). count is always >> 1
        // for this dataset so the n vs n-1 choice doesn't matter in practice;
        // n-1 is used for consistency with typical downstream stats tooling.
        let variance = sq_diff_sums[i] / (count as f64 - 1.0);
        stats[i] = SensorStats {
            index: i + 1, // 1-indexed to match the dataset's own "sensor measurement N" naming
            mean: means[i],
            std_dev: variance.sqrt(),
        };
    }
    stats
}

pub fn near_constant_sensors(stats: &[SensorStats; NUM_SENSORS]) -> Vec<usize> {
    stats
        .iter()
        .filter(|s| s.std_dev < NEAR_CONSTANT_STD_THRESHOLD)
        .map(|s| s.index)
        .collect()
}

/// Pearson correlation between each sensor's raw reading and the labeled
/// RUL, pooled across every cycle of every run passed in.
///
/// This is a cheap, honest way to rank sensors by "does this actually move
/// with degradation" rather than by raw variance — a sensor can be noisy
/// (high variance) without carrying any information about remaining life,
/// and vice versa. It's not a substitute for real feature importance from
/// a trained model (see the `models` crate), just a fast sanity check.
///
/// `runs` and `labels` must be the same length and in the same order (one
/// label vector per run, e.g. the output of `rul::label_train_rul`).
///
/// Sensors with ~zero variance (see `NEAR_CONSTANT_STD_THRESHOLD`) get a
/// correlation of exactly `0.0` rather than a numerically unstable ratio -
/// there's no meaningful correlation to compute when the denominator is
/// effectively zero.
pub fn sensor_rul_correlation(runs: &[EngineRun], labels: &[Vec<u32>]) -> [f64; NUM_SENSORS] {
    assert_eq!(runs.len(), labels.len(), "one label vector required per run");

    let mut xs: [Vec<f64>; NUM_SENSORS] = std::array::from_fn(|_| Vec::new());
    let mut ys: Vec<f64> = Vec::new();

    for (run, run_labels) in runs.iter().zip(labels.iter()) {
        for (record, &rul) in run.records.iter().zip(run_labels.iter()) {
            for s in 0..NUM_SENSORS {
                xs[s].push(record.sensors[s]);
            }
            ys.push(rul as f64);
        }
    }

    let n = ys.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / n;
    let y_var: f64 = ys.iter().map(|y| (y - y_mean).powi(2)).sum();

    let mut correlations = [0.0; NUM_SENSORS];
    for s in 0..NUM_SENSORS {
        let x_mean = xs[s].iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut x_var = 0.0;
        for (x, y) in xs[s].iter().zip(ys.iter()) {
            cov += (x - x_mean) * (y - y_mean);
            x_var += (x - x_mean).powi(2);
        }
        let denom = (x_var * y_var).sqrt();
        correlations[s] = if denom < 1e-9 { 0.0 } else { cov / denom };
    }
    correlations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::CycleRecord;

    /// Builds a run where sensor 1 (index 0) takes the given `values` across
    /// cycles, and every *other* sensor is deliberately made to vary too
    /// (via the cycle index) so the test isolates sensor 1's behavior rather
    /// than incidentally leaving 20 other sensors flat at 0.0.
    fn run_with_sensor_values(unit: u32, values: &[f64]) -> EngineRun {
        EngineRun {
            unit,
            records: values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let mut sensors = [0.0; NUM_SENSORS];
                    sensors[0] = v;
                    for (j, s) in sensors.iter_mut().enumerate().skip(1) {
                        *s = i as f64 + j as f64; // varies every cycle, never constant
                    }
                    CycleRecord {
                        unit,
                        cycle: (i + 1) as u32,
                        op_settings: [0.0; 3],
                        sensors,
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn detects_constant_sensor() {
        let run = run_with_sensor_values(1, &[5.0, 5.0, 5.0, 5.0]);
        let stats = sensor_stats(&[run]);
        assert!(stats[0].std_dev < NEAR_CONSTANT_STD_THRESHOLD);
        assert_eq!(near_constant_sensors(&stats), vec![1]);
    }

    #[test]
    fn detects_varying_sensor() {
        let run = run_with_sensor_values(1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = sensor_stats(&[run]);
        assert!(stats[0].std_dev > NEAR_CONSTANT_STD_THRESHOLD);
        assert_eq!(near_constant_sensors(&stats), Vec::<usize>::new());
    }

    #[test]
    fn perfectly_correlated_sensor_scores_near_one() {
        // sensor value == rul exactly -> correlation should be ~1.0
        let run = run_with_sensor_values(1, &[40.0, 30.0, 20.0, 10.0, 0.0]);
        let labels = vec![vec![40, 30, 20, 10, 0]];
        let corr = sensor_rul_correlation(&[run], &labels);
        assert!((corr[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn inversely_correlated_sensor_scores_near_negative_one() {
        let run = run_with_sensor_values(1, &[0.0, 10.0, 20.0, 30.0, 40.0]);
        let labels = vec![vec![40, 30, 20, 10, 0]];
        let corr = sensor_rul_correlation(&[run], &labels);
        assert!((corr[0] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn constant_sensor_correlates_at_exactly_zero_not_nan() {
        let run = run_with_sensor_values(1, &[5.0, 5.0, 5.0, 5.0]);
        let labels = vec![vec![30, 20, 10, 0]];
        let corr = sensor_rul_correlation(&[run], &labels);
        assert_eq!(corr[0], 0.0);
        assert!(!corr[0].is_nan());
    }

    #[test]
    fn cycle_length_stats_are_correct() {
        let runs = vec![
            run_with_sensor_values(1, &[0.0; 10]),
            run_with_sensor_values(2, &[0.0; 20]),
            run_with_sensor_values(3, &[0.0; 30]),
        ];
        let stats = cycle_length_stats(&runs);
        assert_eq!(stats.min, 10);
        assert_eq!(stats.max, 30);
        assert_eq!(stats.mean, 20.0);
    }
}
