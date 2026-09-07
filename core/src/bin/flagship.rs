//! Phase 4: the flagship model - gradient-boosted regression trees, hand-
//! built from scratch (`tree.rs` + `boosting.rs`), with zero dependencies
//! beyond `std`. This lives inside the `core` crate as a second binary
//! (Cargo auto-discovers anything under `src/bin/`), unlike Phase 3's
//! baselines, which needed their own `models` crate to keep `linfa`/
//! `smartcore` out of core's dependency graph - the flagship model needs no
//! such isolation because it doesn't add any dependency at all.
//!
//! Same evaluation methodology as `models`: train on every windowed row of
//! every training engine, evaluate on exactly one row per test engine (the
//! window ending at its last recorded cycle), matching the official PHM08
//! protocol - see that crate's doc comment for the full rationale.

use std::path::PathBuf;
use std::str::FromStr;

use cmapss_rul::boosting::{GbmParams, GradientBoostedTrees};
use cmapss_rul::dataset::Subset;
use cmapss_rul::design_matrix::{feature_names, to_feature_vector};
use cmapss_rul::eda::{near_constant_sensors, sensor_stats};
use cmapss_rul::features::{compute_windowed_features, DEFAULT_WINDOW};
use cmapss_rul::loader::{group_by_unit, load_records, load_rul};
use cmapss_rul::rul::{label_test_rul, label_train_rul, DEFAULT_RUL_CAP};
use cmapss_rul::scoring::{phm08_score, rmse};
use cmapss_rul::tree::TreeParams;

struct Cli {
    subset: Subset,
    rul_cap: u32,
    window: usize,
    n_trees: usize,
    learning_rate: f64,
    max_depth: usize,
    min_samples_leaf: usize,
    data_dir: PathBuf,
    out_dir: PathBuf,
}

impl Cli {
    fn parse(args: impl Iterator<Item = String>) -> Result<Cli, String> {
        let mut cli = Cli {
            subset: Subset::FD001,
            rul_cap: DEFAULT_RUL_CAP,
            window: DEFAULT_WINDOW,
            n_trees: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_leaf: 20,
            data_dir: PathBuf::from("data/raw/CMAPSSData"),
            out_dir: PathBuf::from("data/processed"),
        };
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            macro_rules! parse_flag {
                ($field:expr, $name:expr) => {
                    $field = args
                        .next()
                        .ok_or(concat!($name, " requires a value"))?
                        .parse()
                        .map_err(|_| concat!("invalid ", $name, " value").to_string())?
                };
            }
            match arg.as_str() {
                "--rul-cap" => parse_flag!(cli.rul_cap, "--rul-cap"),
                "--window" => parse_flag!(cli.window, "--window"),
                "--n-trees" => parse_flag!(cli.n_trees, "--n-trees"),
                "--learning-rate" => parse_flag!(cli.learning_rate, "--learning-rate"),
                "--max-depth" => parse_flag!(cli.max_depth, "--max-depth"),
                "--min-samples-leaf" => parse_flag!(cli.min_samples_leaf, "--min-samples-leaf"),
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
        eprintln!(
            "USAGE: flagship [SUBSET] [--n-trees N] [--learning-rate F] [--max-depth N] \
             [--min-samples-leaf N] [--rul-cap N] [--window N] [--data-dir PATH] [--out-dir PATH]"
        );
        e
    })?;

    println!("=== Phase 4 flagship model (hand-built GBM): {} ===", cli.subset);
    println!(
        "n_trees={} learning_rate={} max_depth={} min_samples_leaf={}",
        cli.n_trees, cli.learning_rate, cli.max_depth, cli.min_samples_leaf
    );

    let train_runs = group_by_unit(load_records(&cli.subset.train_path(&cli.data_dir))?);
    let test_runs = group_by_unit(load_records(&cli.subset.test_path(&cli.data_dir))?);
    let test_rul = load_rul(&cli.subset.rul_path(&cli.data_dir))?;

    let train_labels: Vec<Vec<u32>> = train_runs.iter().map(|r| label_train_rul(r, cli.rul_cap)).collect();
    let test_labels: Vec<Vec<u32>> = test_runs
        .iter()
        .zip(test_rul.iter())
        .map(|(r, &final_rul)| label_test_rul(r, final_rul, cli.rul_cap))
        .collect();

    let stats = sensor_stats(&train_runs);
    let excluded = near_constant_sensors(&stats);
    let names = feature_names(&excluded);
    println!("excluding near-constant sensors {:?}, {} features per row", excluded, names.len());

    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    for (run, labels) in train_runs.iter().zip(train_labels.iter()) {
        for wf in compute_windowed_features(run, labels, cli.window) {
            x_train.push(to_feature_vector(&wf, &excluded));
            y_train.push(wf.rul as f64);
        }
    }
    println!("training rows: {}", x_train.len());

    let mut x_test: Vec<Vec<f64>> = Vec::new();
    let mut y_test: Vec<f64> = Vec::new();
    let mut test_units: Vec<u32> = Vec::new();
    for (run, labels) in test_runs.iter().zip(test_labels.iter()) {
        let windowed = compute_windowed_features(run, labels, cli.window);
        let last = windowed.last().expect("every test unit must yield >=1 windowed row");
        x_test.push(to_feature_vector(last, &excluded));
        y_test.push(last.rul as f64);
        test_units.push(run.unit);
    }
    println!("test engines evaluated: {}", x_test.len());

    let params = GbmParams {
        n_trees: cli.n_trees,
        learning_rate: cli.learning_rate,
        tree: TreeParams { max_depth: cli.max_depth, min_samples_leaf: cli.min_samples_leaf },
    };
    let model = GradientBoostedTrees::fit(&x_train, &y_train, &params);

    println!(
        "training RMSE: {:.2} -> {:.2} over {} trees",
        model.training_rmse.first().unwrap(),
        model.training_rmse.last().unwrap(),
        model.n_trees()
    );

    let test_preds = model.predict_batch(&x_test);
    let pairs: Vec<(f64, f64)> = test_preds.iter().copied().zip(y_test.iter().copied()).collect();

    let rmse_val = rmse(&pairs);
    let score_val = phm08_score(&pairs);
    let late = pairs.iter().filter(|(pred, actual)| pred >= actual).count();
    let early = pairs.len() - late;

    println!("\n--- Gradient boosted trees (from scratch) ---");
    println!("RMSE: {:.2} cycles", rmse_val);
    println!("PHM08 score: {:.1} (lower is better, 0 is perfect)", score_val);
    println!("{} late predictions (dangerous direction), {} early", late, early);

    let mut worst: Vec<&(f64, f64)> = pairs.iter().collect();
    worst.sort_by(|a, b| (b.0 - b.1).abs().partial_cmp(&(a.0 - a.1).abs()).unwrap());
    println!("worst 3 predictions (pred, true, error):");
    for (pred, actual) in worst.iter().take(3) {
        println!("  pred={:.1}  true={:.1}  error={:+.1}", pred, actual, pred - actual);
    }

    std::fs::create_dir_all(&cli.out_dir)?;
    let out_path = cli.out_dir.join(format!("{}_flagship_predictions.csv", cli.subset.code()));
    let mut csv = String::from("unit,true_rul,pred_flagship,error_flagship\n");
    for i in 0..test_units.len() {
        csv.push_str(&format!(
            "{},{},{},{}\n",
            test_units[i],
            y_test[i],
            test_preds[i],
            test_preds[i] - y_test[i]
        ));
    }
    std::fs::write(&out_path, csv)?;

    let curve_path = cli.out_dir.join(format!("{}_flagship_training_curve.csv", cli.subset.code()));
    let mut curve_csv = String::from("tree_number,training_rmse\n");
    for (i, r) in model.training_rmse.iter().enumerate() {
        curve_csv.push_str(&format!("{},{}\n", i + 1, r));
    }
    std::fs::write(&curve_path, curve_csv)?;

    println!("wrote predictions to {} and training curve to {}", out_path.display(), curve_path.display());

    Ok(())
}
