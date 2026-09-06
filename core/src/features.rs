use crate::loader::EngineRun;
use crate::parser::NUM_SENSORS;

/// Default rolling window size, in cycles.
///
/// This has to stay comfortably below the shortest *test* run across all
/// four subsets, or some test engines would never produce a single windowed
/// feature row - including possibly at their last recorded cycle, which is
/// exactly the point the official scoring function evaluates. The shortest
/// test run in the dataset is FD004's, at 19 cycles (see
/// `tests::data_integrity`). 10 leaves comfortable margin under that while
/// still being long enough for a std dev / slope to mean something.
pub const DEFAULT_WINDOW: usize = 10;

/// One cycle's engineered feature vector: the raw reading plus rolling
/// statistics computed over the trailing `window` cycles (inclusive of the
/// current one), for every sensor.
#[derive(Debug, Clone)]
pub struct WindowedFeatures {
    pub unit: u32,
    pub cycle: u32,
    pub op_settings: [f64; 3],
    pub raw_sensors: [f64; NUM_SENSORS],
    pub rolling_mean: [f64; NUM_SENSORS],
    pub rolling_std: [f64; NUM_SENSORS],
    /// Least-squares slope of each sensor against cycle index, within the window.
    pub rolling_slope: [f64; NUM_SENSORS],
    pub rul: u32,
}

/// Computes windowed features for every cycle of `run` that has a full
/// trailing window available.
///
/// Run-aware by construction: this operates on a single `EngineRun`'s
/// records, which already only contain that one engine's cycles (see
/// `loader::group_by_unit`) - there is no code path here that could pull a
/// window from a different engine's history. Cycles before the first full
/// window (i.e. the first `window - 1` cycles of the run) are dropped
/// rather than computed from a partial window, since a std dev or slope
/// from 1-2 points isn't meaningful.
///
/// `labels` must be the same length as `run.records`, in the same order
/// (i.e. the output of `rul::label_train_rul` or `rul::label_test_rul` for
/// this run).
pub fn compute_windowed_features(
    run: &EngineRun,
    labels: &[u32],
    window: usize,
) -> Vec<WindowedFeatures> {
    assert_eq!(
        run.records.len(),
        labels.len(),
        "labels must line up 1:1 with run.records"
    );
    assert!(window >= 2, "window must be at least 2 to compute a std dev / slope");

    let n = run.records.len();
    if n < window {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(n - window + 1);
    for end in (window - 1)..n {
        let start = end + 1 - window;
        let slice = &run.records[start..=end];

        let mut rolling_mean = [0.0; NUM_SENSORS];
        let mut rolling_std = [0.0; NUM_SENSORS];
        let mut rolling_slope = [0.0; NUM_SENSORS];

        // x values for the slope regression: 0..window-1, same for every sensor.
        let xs: Vec<f64> = (0..window).map(|i| i as f64).collect();

        for s in 0..NUM_SENSORS {
            let ys: Vec<f64> = slice.iter().map(|r| r.sensors[s]).collect();
            let mean = ys.iter().sum::<f64>() / window as f64;
            let variance =
                ys.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);

            rolling_mean[s] = mean;
            rolling_std[s] = variance.sqrt();
            rolling_slope[s] = least_squares_slope(&xs, &ys);
        }

        out.push(WindowedFeatures {
            unit: run.unit,
            cycle: run.records[end].cycle,
            op_settings: run.records[end].op_settings,
            raw_sensors: run.records[end].sensors,
            rolling_mean,
            rolling_std,
            rolling_slope,
            rul: labels[end],
        });
    }
    out
}

/// Ordinary least-squares slope of `ys` against `xs`. Hand-rolled since it's
/// a handful of lines and is core, differentiating logic for this project
/// rather than plumbing.
fn least_squares_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = ys.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        cov += (x - x_mean) * (y - y_mean);
        var_x += (x - x_mean).powi(2);
    }

    if var_x == 0.0 {
        0.0
    } else {
        cov / var_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::CycleRecord;

    fn run_with_sensor_0(unit: u32, values: &[f64]) -> EngineRun {
        EngineRun {
            unit,
            records: values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let mut sensors = [0.0; NUM_SENSORS];
                    sensors[0] = v;
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
    fn slope_of_a_straight_line_is_exact() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![10.0, 12.0, 14.0, 16.0, 18.0]; // slope 2, intercept 10
        assert!((least_squares_slope(&xs, &ys) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn slope_of_a_flat_line_is_zero() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![5.0, 5.0, 5.0, 5.0];
        assert!(least_squares_slope(&xs, &ys).abs() < 1e-9);
    }

    #[test]
    fn drops_cycles_before_the_first_full_window() {
        let run = run_with_sensor_0(1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let labels = vec![4, 3, 2, 1, 0];
        let features = compute_windowed_features(&run, &labels, 3);
        // 5 cycles, window 3 -> valid windows end at cycle index 2,3,4 -> 3 rows
        assert_eq!(features.len(), 3);
        assert_eq!(features[0].cycle, 3); // first full window ends at cycle 3
        assert_eq!(features.last().unwrap().cycle, 5);
    }

    #[test]
    fn returns_empty_when_run_is_shorter_than_window() {
        let run = run_with_sensor_0(1, &[1.0, 2.0]);
        let labels = vec![1, 0];
        let features = compute_windowed_features(&run, &labels, 5);
        assert!(features.is_empty());
    }

    #[test]
    fn rolling_mean_and_std_match_hand_computed_values() {
        // Window [1,2,3]: mean 2.0, sample std = 1.0
        let run = run_with_sensor_0(1, &[1.0, 2.0, 3.0]);
        let labels = vec![2, 1, 0];
        let features = compute_windowed_features(&run, &labels, 3);
        assert_eq!(features.len(), 1);
        assert!((features[0].rolling_mean[0] - 2.0).abs() < 1e-9);
        assert!((features[0].rolling_std[0] - 1.0).abs() < 1e-9);
        assert!((features[0].rolling_slope[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rul_labels_carry_through_unchanged() {
        let run = run_with_sensor_0(1, &[1.0, 2.0, 3.0, 4.0]);
        let labels = vec![30, 20, 10, 0];
        let features = compute_windowed_features(&run, &labels, 2);
        // windows end at cycle indices 1,2,3 -> ruls 20,10,0
        assert_eq!(
            features.iter().map(|f| f.rul).collect::<Vec<_>>(),
            vec![20, 10, 0]
        );
    }
}
