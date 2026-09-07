//! Integration tests against the real C-MAPSS files checked into `data/raw/`.
//!
//! These are deliberately run against the actual dataset rather than
//! synthetic fixtures — the whole point of Phase 1 is confirming the parser
//! and labeling logic hold up against the real, messy files.

use std::path::PathBuf;

use cmapss_rul::dataset::Subset;
use cmapss_rul::features::{compute_windowed_features, DEFAULT_WINDOW};
use cmapss_rul::loader::{group_by_unit, load_records, load_rul};
use cmapss_rul::regime::fit_regimes;
use cmapss_rul::rul::{label_test_rul, label_train_rul, DEFAULT_RUL_CAP};

fn data_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is this crate's own directory (core/), but data/
    // lives at the workspace root one level up.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/raw/CMAPSSData")
}

#[test]
fn every_subset_has_expected_train_and_test_unit_counts() {
    for subset in Subset::all() {
        let train = group_by_unit(load_records(&subset.train_path(&data_dir())).unwrap());
        let test = group_by_unit(load_records(&subset.test_path(&data_dir())).unwrap());

        assert_eq!(
            train.len(),
            subset.expected_train_units(),
            "{}: train unit count mismatch",
            subset
        );
        assert_eq!(
            test.len(),
            subset.expected_test_units(),
            "{}: test unit count mismatch",
            subset
        );
    }
}

#[test]
fn rul_file_length_matches_test_unit_count_for_every_subset() {
    for subset in Subset::all() {
        let test = group_by_unit(load_records(&subset.test_path(&data_dir())).unwrap());
        let rul = load_rul(&subset.rul_path(&data_dir())).unwrap();
        assert_eq!(
            rul.len(),
            test.len(),
            "{}: RUL_FDxxx.txt line count doesn't match number of test units",
            subset
        );
    }
}

#[test]
fn every_row_in_every_file_parses() {
    // If any row failed to parse, load_records would have already returned
    // an Err and .unwrap() above/here would panic with the exact line number
    // and raw content. This test exists mainly as an explicit "all 4 x 2
    // (train+test) files parse cleanly end to end" checkpoint.
    for subset in Subset::all() {
        load_records(&subset.train_path(&data_dir())).unwrap();
        load_records(&subset.test_path(&data_dir())).unwrap();
    }
}

#[test]
fn train_rul_labels_are_monotonically_non_increasing_and_end_at_zero() {
    for subset in Subset::all() {
        let train = group_by_unit(load_records(&subset.train_path(&data_dir())).unwrap());
        for run in &train {
            let labels = label_train_rul(run, DEFAULT_RUL_CAP);
            assert!(
                labels.windows(2).all(|w| w[0] >= w[1]),
                "{} unit {}: RUL labels are not monotonically non-increasing",
                subset,
                run.unit
            );
            assert_eq!(
                *labels.last().unwrap(),
                0,
                "{} unit {}: last cycle of a training run should have RUL 0",
                subset,
                run.unit
            );
            assert!(
                labels.iter().all(|&r| r <= DEFAULT_RUL_CAP),
                "{} unit {}: a label exceeded the configured cap",
                subset,
                run.unit
            );
        }
    }
}

#[test]
fn test_rul_reconstruction_matches_ground_truth_at_last_cycle() {
    for subset in Subset::all() {
        let test = group_by_unit(load_records(&subset.test_path(&data_dir())).unwrap());
        let rul = load_rul(&subset.rul_path(&data_dir())).unwrap();

        for (run, &final_rul) in test.iter().zip(rul.iter()) {
            let labels = label_test_rul(run, final_rul, DEFAULT_RUL_CAP);
            let expected_last = final_rul.min(DEFAULT_RUL_CAP);
            assert_eq!(
                *labels.last().unwrap(),
                expected_last,
                "{} unit {}: reconstructed RUL at last cycle doesn't match RUL_FDxxx.txt",
                subset,
                run.unit
            );
        }
    }
}

#[test]
fn default_window_leaves_every_test_unit_with_at_least_one_windowed_row() {
    // This is the real constraint DEFAULT_WINDOW has to respect: if it's
    // too large relative to the shortest test run in any subset, that
    // engine contributes zero windowed rows - including at its last
    // recorded cycle, which is exactly what the official scoring function
    // evaluates. This test would fail loudly if DEFAULT_WINDOW were ever
    // raised without checking this first.
    for subset in Subset::all() {
        let test = group_by_unit(load_records(&subset.test_path(&data_dir())).unwrap());
        let rul = load_rul(&subset.rul_path(&data_dir())).unwrap();

        for (run, &final_rul) in test.iter().zip(rul.iter()) {
            let labels = label_test_rul(run, final_rul, DEFAULT_RUL_CAP);
            let features = compute_windowed_features(run, &labels, DEFAULT_WINDOW);
            assert!(
                !features.is_empty(),
                "{} unit {} (run length {}) produced zero windowed rows with window={}",
                subset,
                run.unit,
                run.records.len(),
                DEFAULT_WINDOW
            );
        }
    }
}

#[test]
fn windowed_features_never_mix_two_engines() {
    // compute_windowed_features takes a single EngineRun, whose records are
    // already scoped to one unit by group_by_unit - so cross-engine leakage
    // would require a bug in how callers wire units together, not in the
    // windowing math itself. This test pins that structural guarantee: every
    // windowed row's unit must match the run it came from, for every run in
    // a multi-condition subset (the case most likely to tempt a "just
    // concatenate everything" shortcut later).
    let test = group_by_unit(load_records(&Subset::FD002.test_path(&data_dir())).unwrap());
    let rul = load_rul(&Subset::FD002.rul_path(&data_dir())).unwrap();

    for (run, &final_rul) in test.iter().zip(rul.iter()) {
        let labels = label_test_rul(run, final_rul, DEFAULT_RUL_CAP);
        let features = compute_windowed_features(run, &labels, DEFAULT_WINDOW);
        assert!(
            features.iter().all(|f| f.unit == run.unit),
            "found a windowed row tagged with a different unit than its source run"
        );
    }
}

#[test]
fn regime_clustering_finds_six_well_separated_balanced_conditions_on_real_multi_condition_data() {
    // Pins the empirical validation done during Phase 5 development: fitting
    // k=6 on FD002/FD004's real operational settings should find genuinely
    // separated, reasonably balanced regimes - not, say, one dominant
    // cluster and five near-empty ones, which would mean the "6 operating
    // conditions" premise this phase relies on doesn't actually hold up
    // against the real data.
    for subset in [Subset::FD002, Subset::FD004] {
        let train_runs = group_by_unit(load_records(&subset.train_path(&data_dir())).unwrap());
        let points: Vec<Vec<f64>> = train_runs
            .iter()
            .flat_map(|r| r.records.iter().map(|rec| rec.op_settings.to_vec()))
            .collect();

        let regimes = fit_regimes(&train_runs, 6);
        let sizes = cmapss_rul::kmeans::cluster_sizes(&regimes, &points);
        let total: usize = sizes.iter().sum();

        assert_eq!(regimes.k(), 6);
        assert_eq!(total, points.len());
        for &size in &sizes {
            let fraction = size as f64 / total as f64;
            assert!(
                fraction > 0.03,
                "{}: a regime claimed only {:.1}% of cycles - suspiciously close to empty, \
                 clustering may not have found 6 genuinely separated conditions",
                subset,
                fraction * 100.0
            );
        }
    }
}

#[test]
fn fd002_and_fd004_have_more_operating_conditions_than_fd001_and_fd003() {
    // Cheap regression check that the metadata table itself is self-consistent.
    assert_eq!(Subset::FD001.num_conditions(), 1);
    assert_eq!(Subset::FD003.num_conditions(), 1);
    assert_eq!(Subset::FD002.num_conditions(), 6);
    assert_eq!(Subset::FD004.num_conditions(), 6);
}
