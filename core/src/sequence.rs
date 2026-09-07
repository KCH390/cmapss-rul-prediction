//! Raw (non-aggregated) sliding-window sequence extraction for the Phase 6
//! sequence model.
//!
//! Phase 2's `features.rs` collapses each window into summary statistics
//! (mean/std/slope) because tree-based and linear models need a fixed-size
//! feature vector. An LSTM is built to consume the sequence itself — this
//! produces the raw (window_size x kept_sensor_count) matrix per sample
//! instead, flattened row-major (time-major: all features at t, then all
//! features at t+1, ...) since that's the layout `sequence`'s candle-based
//! model reshapes into a (seq_len, num_features) tensor.

use crate::loader::EngineRun;
use crate::parser::NUM_SENSORS;

#[derive(Debug, Clone)]
pub struct SequenceSample {
    pub unit: u32,
    pub cycle: u32,
    /// Row-major flattened (window_size x kept_sensor_count): index
    /// `t * kept_sensor_count + f` is feature `f` at the `t`-th cycle of
    /// this window (t=0 is the oldest cycle in the window).
    pub sequence: Vec<f64>,
    pub rul: u32,
}

/// Which raw sensor indices (0-based) survive `exclude_sensors` (1-based
/// sensor numbers, e.g. `eda::near_constant_sensors`'s output), in a fixed
/// order shared by every sample this module produces.
pub fn kept_sensor_indices(exclude_sensors: &[usize]) -> Vec<usize> {
    (0..NUM_SENSORS).filter(|&i| !exclude_sensors.contains(&(i + 1))).collect()
}

pub fn num_features(exclude_sensors: &[usize]) -> usize {
    NUM_SENSORS - exclude_sensors.len()
}

/// Extracts every full-window sequence sample from `run`, mirroring
/// `features::compute_windowed_features`'s windowing rules exactly (same
/// run-aware-by-construction guarantee, same "drop cycles before the first
/// full window rather than pad" choice, same `labels` convention) so the
/// two feature representations stay directly comparable.
pub fn extract_sequences(
    run: &EngineRun,
    labels: &[u32],
    window: usize,
    exclude_sensors: &[usize],
) -> Vec<SequenceSample> {
    assert_eq!(run.records.len(), labels.len(), "labels must line up 1:1 with run.records");
    assert!(window >= 1, "window must be at least 1");

    let keep = kept_sensor_indices(exclude_sensors);
    let n = run.records.len();
    if n < window {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(n - window + 1);
    for end in (window - 1)..n {
        let start = end + 1 - window;
        let mut sequence = Vec::with_capacity(window * keep.len());
        for t in start..=end {
            for &f in &keep {
                sequence.push(run.records[t].sensors[f]);
            }
        }
        out.push(SequenceSample {
            unit: run.unit,
            cycle: run.records[end].cycle,
            sequence,
            rul: labels[end],
        });
    }
    out
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
    fn kept_sensor_indices_excludes_the_given_sensors() {
        let kept = kept_sensor_indices(&[1, 5, 21]);
        assert_eq!(kept.len(), NUM_SENSORS - 3);
        assert!(!kept.contains(&0)); // sensor 1 -> index 0
        assert!(!kept.contains(&4)); // sensor 5 -> index 4
        assert!(!kept.contains(&20)); // sensor 21 -> index 20
    }

    #[test]
    fn returns_empty_when_run_is_shorter_than_window() {
        let run = run_with_sensor_0(1, &[1.0, 2.0]);
        let labels = vec![1, 0];
        assert!(extract_sequences(&run, &labels, 5, &[]).is_empty());
    }

    #[test]
    fn drops_cycles_before_the_first_full_window_same_as_features_rs() {
        let run = run_with_sensor_0(1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let labels = vec![4, 3, 2, 1, 0];
        let samples = extract_sequences(&run, &labels, 3, &[]);
        assert_eq!(samples.len(), 3); // 5 cycles, window 3 -> 3 valid windows
        assert_eq!(samples[0].cycle, 3);
        assert_eq!(samples.last().unwrap().cycle, 5);
    }

    #[test]
    fn sequence_values_and_order_match_hand_computed_layout() {
        // No exclusions -> NUM_SENSORS features per timestep. Only sensor
        // index 0 is non-zero here, so we can check its exact position in
        // the flattened, row-major (time-major) layout.
        let run = run_with_sensor_0(1, &[10.0, 20.0, 30.0]);
        let labels = vec![2, 1, 0];
        let samples = extract_sequences(&run, &labels, 3, &[]);
        assert_eq!(samples.len(), 1);
        let seq = &samples[0].sequence;
        assert_eq!(seq.len(), 3 * NUM_SENSORS);
        // t=0 -> value 10.0 at feature index 0
        assert_eq!(seq[0 * NUM_SENSORS], 10.0);
        // t=1 -> value 20.0
        assert_eq!(seq[1 * NUM_SENSORS], 20.0);
        // t=2 -> value 30.0 (this is the "current" cycle, oldest-to-newest order)
        assert_eq!(seq[2 * NUM_SENSORS], 30.0);
    }

    #[test]
    fn rul_labels_carry_through_unchanged() {
        let run = run_with_sensor_0(1, &[1.0, 2.0, 3.0, 4.0]);
        let labels = vec![30, 20, 10, 0];
        let samples = extract_sequences(&run, &labels, 2, &[]);
        assert_eq!(samples.iter().map(|s| s.rul).collect::<Vec<_>>(), vec![20, 10, 0]);
    }

    #[test]
    fn excluded_sensors_shrink_the_flattened_sequence_length_consistently() {
        let run = run_with_sensor_0(1, &[1.0, 2.0, 3.0]);
        let labels = vec![2, 1, 0];
        let exclude = vec![1, 5, 21];
        let samples = extract_sequences(&run, &labels, 3, &exclude);
        assert_eq!(samples[0].sequence.len(), 3 * (NUM_SENSORS - 3));
    }
}
