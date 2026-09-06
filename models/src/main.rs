//! Phase 3: baseline RUL models.
//!
//! Two standard, off-the-shelf regressors - linear regression (`linfa`) and
//! random forest (`smartcore`) - establishing a performance floor before
//! Phase 4's hand-built gradient-boosted trees. Lives in its own workspace
//! crate for the same reason `charts` does: keep real ML dependencies out
//! of the core pipeline's build graph.
//!
//! **Evaluation methodology, which is easy to get wrong**: training uses
//! every windowed row of every training engine (pooled together - lots of
//! rows per engine, which is normal data augmentation for this task). But
//! evaluation uses exactly one row per test engine - the window ending at
//! that engine's *last recorded cycle* - because that's what the official
//! PHM08 competition protocol scores: one RUL prediction per engine, made
//! at the point its data was truncated. Scoring every windowed row of a
//! test trajectory would inflate apparent performance (consecutive windows
//! from the same engine are highly correlated) and wouldn't match any
//! published numbers this could be compared against.

use std::path::PathBuf;
use std::str::FromStr;

use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;

use cmapss_rul::dataset::Subset;
use cmapss_rul::design_matrix::{feature_names, to_feature_vector};
use cmapss_rul::eda::{near_constant_sensors, sensor_stats};
use cmapss_rul::features::{compute_windowed_features, DEFAULT_WINDOW};
use cmapss_rul::loader::{group_by_unit, load_records, load_rul};
use cmapss_rul::rul::{label_test_rul, label_train_rul, DEFAULT_RUL_CAP};
use cmapss_rul::scoring::{phm08_score, rmse};

struct Cli {
    subset: Subset,
    rul_cap: u32,
    window: usize,
    data_dir: PathBuf,
    out_dir: PathBuf,
}

impl Cli {
    fn parse(args: impl Iterator<Item = String>) -> Result<Cli, String> {
        let mut cli = Cli {
            subset: Subset::FD001,
            rul_cap: DEFAULT_RUL_CAP,
            window: DEFAULT_WINDOW,
            data_dir: PathBuf::from("data/raw/CMAPSSData"),
            out_dir: PathBuf::from("data/processed"),
        };
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--rul-cap" => {
                    cli.rul_cap = args
                        .next()
                        .ok_or("--rul-cap requires a value")?
                        .parse()
                        .map_err(|_| "invalid --rul-cap value".to_string())?;
                }
                "--window" => {
                    cli.window = args
                        .next()
                        .ok_or("--window requires a value")?
                        .parse()
                        .map_err(|_| "invalid --window value".to_string())?;
                }
                "--data-dir" => cli.data_dir = PathBuf::from(args.next().ok_or("--data-dir requires a value")?),
                "--out-dir" => cli.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires a value")?),
                other if !other.starts_with('-') => cli.subset = Subset::from_str(other)?,
                other => return Err(format!("unrecognized argument {:?}", other)),
            }
        }
        Ok(cli)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse(std::env::args().skip(1)).map_err(|e| {
        eprintln!("argument error: {}", e);
        eprintln!("USAGE: models [SUBSET] [--rul-cap N] [--window N] [--data-dir PATH] [--out-dir PATH]");
        e
    })?;

    println!("=== Phase 3 baseline models: {} ===", cli.subset);

    // --- Load, label, window (same pipeline as the core CLI) ---
    let train_runs = group_by_unit(load_records(&cli.subset.train_path(&cli.data_dir))?);
    let test_runs = group_by_unit(load_records(&cli.subset.test_path(&cli.data_dir))?);
    let test_rul = load_rul(&cli.subset.rul_path(&cli.data_dir))?;

    let train_labels: Vec<Vec<u32>> = train_runs.iter().map(|r| label_train_rul(r, cli.rul_cap)).collect();
    let test_labels: Vec<Vec<u32>> = test_runs
        .iter()
        .zip(test_rul.iter())
        .map(|(r, &final_rul)| label_test_rul(r, final_rul, cli.rul_cap))
        .collect();

    // --- Feature selection: drop sensors this subset's own training data
    // showed to be near-constant (Phase 1/2 EDA feeding directly into
    // Phase 3 modeling, not a disconnected step) ---
    let stats = sensor_stats(&train_runs);
    let excluded = near_constant_sensors(&stats);
    println!(
        "excluding {} near-constant sensor(s) from the feature set: {:?}",
        excluded.len(),
        excluded
    );
    let names = feature_names(&excluded);
    println!("{} features per row: {:?}...", names.len(), &names[..names.len().min(6)]);

    // --- Training set: every windowed row of every training engine, pooled ---
    let mut x_train_rows: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    for (run, labels) in train_runs.iter().zip(train_labels.iter()) {
        for wf in compute_windowed_features(run, labels, cli.window) {
            x_train_rows.push(to_feature_vector(&wf, &excluded));
            y_train.push(wf.rul as f64);
        }
    }
    println!("training rows: {}", x_train_rows.len());

    // --- Evaluation set: exactly one row per test engine - the window
    // ending at its last recorded cycle - matching the official protocol. ---
    let mut x_test_rows: Vec<Vec<f64>> = Vec::new();
    let mut y_test: Vec<f64> = Vec::new();
    let mut test_units: Vec<u32> = Vec::new();
    for (run, labels) in test_runs.iter().zip(test_labels.iter()) {
        let windowed = compute_windowed_features(run, labels, cli.window);
        let last = windowed
            .last()
            .expect("every test unit must yield >=1 windowed row with DEFAULT_WINDOW (see core's integration test)");
        x_test_rows.push(to_feature_vector(last, &excluded));
        y_test.push(last.rul as f64);
        test_units.push(run.unit);
    }
    println!("test engines evaluated: {}", x_test_rows.len());

    let n_features = x_train_rows[0].len();

    // === Linear regression (linfa) ===
    let x_train_arr = Array2::from_shape_vec(
        (x_train_rows.len(), n_features),
        x_train_rows.iter().flatten().copied().collect(),
    )?;
    let y_train_arr = Array1::from_vec(y_train.clone());
    let dataset = Dataset::new(x_train_arr, y_train_arr);
    let linear_model = LinearRegression::default().fit(&dataset)?;

    let x_test_arr = Array2::from_shape_vec(
        (x_test_rows.len(), n_features),
        x_test_rows.iter().flatten().copied().collect(),
    )?;
    let linear_preds: Array1<f64> = linear_model.predict(&x_test_arr);
    let linear_pairs: Vec<(f64, f64)> = linear_preds.iter().copied().zip(y_test.iter().copied()).collect();

    // === Random forest (smartcore) ===
    let x_train_dm = DenseMatrix::from_2d_vec(&x_train_rows);
    let rf = RandomForestRegressor::fit(&x_train_dm, &y_train, RandomForestRegressorParameters::default())?;
    let x_test_dm = DenseMatrix::from_2d_vec(&x_test_rows);
    let rf_preds: Vec<f64> = rf.predict(&x_test_dm)?;
    let rf_pairs: Vec<(f64, f64)> = rf_preds.iter().copied().zip(y_test.iter().copied()).collect();

    // === Report ===
    report("Linear regression", &linear_pairs);
    report("Random forest", &rf_pairs);

    // --- Predictions CSV ---
    std::fs::create_dir_all(&cli.out_dir)?;
    let out_path = cli.out_dir.join(format!("{}_test_predictions.csv", cli.subset.code()));
    let mut csv = String::from("unit,true_rul,pred_linear,pred_rf,error_linear,error_rf\n");
    for i in 0..test_units.len() {
        csv.push_str(&format!(
            "{},{},{},{},{},{}\n",
            test_units[i],
            y_test[i],
            linear_preds[i],
            rf_preds[i],
            linear_preds[i] - y_test[i],
            rf_preds[i] - y_test[i],
        ));
    }
    std::fs::write(&out_path, csv)?;
    println!("wrote predictions to {}", out_path.display());

    Ok(())
}

fn report(name: &str, pairs: &[(f64, f64)]) {
    let rmse_val = rmse(pairs);
    let score_val = phm08_score(pairs);
    let late = pairs.iter().filter(|(pred, actual)| pred >= actual).count();
    let early = pairs.len() - late;

    println!("\n--- {} ---", name);
    println!("RMSE: {:.2} cycles", rmse_val);
    println!("PHM08 score: {:.1} (lower is better, 0 is perfect)", score_val);
    println!(
        "{} late predictions (predicted RUL > true, the dangerous direction), {} early",
        late, early
    );

    let mut worst: Vec<&(f64, f64)> = pairs.iter().collect();
    worst.sort_by(|a, b| (b.0 - b.1).abs().partial_cmp(&(a.0 - a.1).abs()).unwrap());
    println!("worst 3 predictions (pred, true, error):");
    for (pred, actual) in worst.iter().take(3) {
        println!("  pred={:.1}  true={:.1}  error={:+.1}", pred, actual, pred - actual);
    }
}
