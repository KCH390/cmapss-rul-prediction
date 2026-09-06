//! Turns a `WindowedFeatures` struct into a flat `Vec<f64>` suitable for
//! feeding to any ML library, with a consistent, documented column order.
//!
//! This lives in `core` rather than in `models` (which is where the actual
//! ML crates live) because the column order and exclusion logic is
//! differentiating domain logic tied to this dataset, not plumbing specific
//! to any one ML library - both a linfa-based model and a smartcore-based
//! model should see the exact same feature columns in the exact same order.

use crate::features::WindowedFeatures;
use crate::parser::NUM_SENSORS;

/// Column order: raw sensors, then rolling means, then rolling stds, then
/// rolling slopes - each block skipping any sensor in `exclude_sensors`
/// (1-indexed sensor numbers, e.g. the output of `eda::near_constant_sensors`).
///
/// Excluding near-constant sensors isn't just a minor cleanup: a sensor
/// that reads ~identically across every cycle contributes a feature column
/// that's ~constant across the entire training set, which is dead weight
/// for a tree model and can cause numerical instability for a closed-form
/// linear solver on a small/collinear design matrix. This is Phase 1/2's
/// EDA work directly informing Phase 3's modeling, rather than the two
/// being disconnected.
///
/// Operational settings are deliberately excluded here, for every subset -
/// see the README's Phase 3 section for why (short version: FD001/FD003 have
/// a single condition so they carry no information; FD002/FD004 need
/// regime normalization first, which is Phase 5's job, so including raw
/// operating settings now would just be noise ahead of schedule).
pub fn to_feature_vector(f: &WindowedFeatures, exclude_sensors: &[usize]) -> Vec<f64> {
    let keep = |sensor_number: usize| !exclude_sensors.contains(&sensor_number);

    let mut v = Vec::with_capacity(4 * NUM_SENSORS);
    for i in 0..NUM_SENSORS {
        if keep(i + 1) {
            v.push(f.raw_sensors[i]);
        }
    }
    for i in 0..NUM_SENSORS {
        if keep(i + 1) {
            v.push(f.rolling_mean[i]);
        }
    }
    for i in 0..NUM_SENSORS {
        if keep(i + 1) {
            v.push(f.rolling_std[i]);
        }
    }
    for i in 0..NUM_SENSORS {
        if keep(i + 1) {
            v.push(f.rolling_slope[i]);
        }
    }
    v
}

/// Human-readable names for the columns `to_feature_vector` produces, in
/// the same order - useful for reporting which features a model leaned on.
pub fn feature_names(exclude_sensors: &[usize]) -> Vec<String> {
    let keep = |sensor_number: usize| !exclude_sensors.contains(&sensor_number);
    let mut names = Vec::with_capacity(4 * NUM_SENSORS);
    for suffix in ["raw", "mean", "std", "slope"] {
        for i in 0..NUM_SENSORS {
            if keep(i + 1) {
                names.push(format!("sensor_{}_{}", i + 1, suffix));
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_features() -> WindowedFeatures {
        WindowedFeatures {
            unit: 1,
            cycle: 10,
            op_settings: [0.1, 0.2, 0.3],
            raw_sensors: std::array::from_fn(|i| i as f64),
            rolling_mean: std::array::from_fn(|i| i as f64 + 100.0),
            rolling_std: std::array::from_fn(|i| i as f64 + 200.0),
            rolling_slope: std::array::from_fn(|i| i as f64 + 300.0),
            rul: 42,
        }
    }

    #[test]
    fn vector_length_matches_names_length_with_no_exclusions() {
        let f = dummy_features();
        let v = to_feature_vector(&f, &[]);
        let names = feature_names(&[]);
        assert_eq!(v.len(), names.len());
        assert_eq!(v.len(), 4 * NUM_SENSORS);
    }

    #[test]
    fn excluded_sensors_are_dropped_from_every_block_consistently() {
        let f = dummy_features();
        let exclude = vec![1, 5, 21]; // first, middle-ish, last sensor
        let v = to_feature_vector(&f, &exclude);
        let names = feature_names(&exclude);
        assert_eq!(v.len(), names.len());
        assert_eq!(v.len(), 4 * (NUM_SENSORS - 3));
        // None of the excluded sensor numbers should appear in any name.
        for excluded_num in &exclude {
            let needle = format!("sensor_{}_", excluded_num);
            assert!(
                names.iter().all(|n| !n.starts_with(&needle)),
                "sensor {} should have been excluded from every block",
                excluded_num
            );
        }
    }

    #[test]
    fn column_order_is_raw_then_mean_then_std_then_slope() {
        let f = dummy_features();
        let v = to_feature_vector(&f, &[]);
        // First NUM_SENSORS entries should be raw_sensors (0..21 by construction).
        assert_eq!(v[0], f.raw_sensors[0]);
        assert_eq!(v[NUM_SENSORS], f.rolling_mean[0]);
        assert_eq!(v[2 * NUM_SENSORS], f.rolling_std[0]);
        assert_eq!(v[3 * NUM_SENSORS], f.rolling_slope[0]);
    }
}
