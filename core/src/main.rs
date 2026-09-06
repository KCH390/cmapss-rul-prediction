use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;

use cmapss_rul::dataset::Subset;
use cmapss_rul::eda::{cycle_length_stats, near_constant_sensors, sensor_stats};
use cmapss_rul::features::{compute_windowed_features, WindowedFeatures, DEFAULT_WINDOW};
use cmapss_rul::loader::{group_by_unit, load_records, load_rul, EngineRun};
use cmapss_rul::rul::{label_test_rul, label_train_rul, DEFAULT_RUL_CAP};
use cmapss_rul::Result;

/// Phase 1: load, label, and summarize the NASA C-MAPSS turbofan dataset.
///
/// Usage:
///   cmapss-rul [SUBSET] [--rul-cap N] [--data-dir PATH] [--out-dir PATH] [--report-only]
///
/// SUBSET is one of FD001, FD002, FD003, FD004 (case-insensitive); omit to process all four.
struct Cli {
    subset: Option<Subset>,
    rul_cap: u32,
    window: usize,
    data_dir: PathBuf,
    out_dir: PathBuf,
    report_only: bool,
}

impl Cli {
    fn parse(args: impl Iterator<Item = String>) -> std::result::Result<Cli, String> {
        let mut cli = Cli {
            subset: None,
            rul_cap: DEFAULT_RUL_CAP,
            window: DEFAULT_WINDOW,
            data_dir: PathBuf::from("data/raw/CMAPSSData"),
            out_dir: PathBuf::from("data/processed"),
            report_only: false,
        };

        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--rul-cap" => {
                    let val = args.next().ok_or("--rul-cap requires a value")?;
                    cli.rul_cap = val
                        .parse()
                        .map_err(|_| format!("--rul-cap value {:?} is not a valid u32", val))?;
                }
                "--window" => {
                    let val = args.next().ok_or("--window requires a value")?;
                    cli.window = val
                        .parse()
                        .map_err(|_| format!("--window value {:?} is not a valid usize", val))?;
                }
                "--data-dir" => {
                    cli.data_dir = PathBuf::from(args.next().ok_or("--data-dir requires a value")?);
                }
                "--out-dir" => {
                    cli.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires a value")?);
                }
                "--report-only" => cli.report_only = true,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other if !other.starts_with('-') => {
                    cli.subset = Some(Subset::from_str(other)?);
                }
                other => return Err(format!("unrecognized argument {:?}", other)),
            }
        }

        Ok(cli)
    }
}

fn print_help() {
    println!("cmapss-rul: load, label, and summarize the NASA C-MAPSS turbofan dataset\n");
    println!("USAGE:");
    println!("  cmapss-rul [SUBSET] [--rul-cap N] [--window N] [--data-dir PATH] [--out-dir PATH] [--report-only]\n");
    println!("SUBSET: one of FD001, FD002, FD003, FD004 (case-insensitive). Omit to process all four.");
}

fn main() {
    let cli = match Cli::parse(std::env::args().skip(1)) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("argument error: {}", e);
            print_help();
            std::process::exit(2);
        }
    };

    let subsets: Vec<Subset> = match cli.subset {
        Some(s) => vec![s],
        None => Subset::all().to_vec(),
    };

    let mut had_error = false;
    for subset in subsets {
        if let Err(e) = process_subset(subset, &cli) {
            eprintln!("error processing {}: {}", subset, e);
            had_error = true;
        }
    }

    if had_error {
        std::process::exit(1);
    }
}

fn process_subset(subset: Subset, cli: &Cli) -> Result<()> {
    println!("\n=== {} ===", subset);
    println!(
        "conditions: {}   fault modes: {}",
        subset.num_conditions(),
        subset.num_fault_modes()
    );

    // --- Load ---
    let train_records = load_records(&subset.train_path(&cli.data_dir))?;
    let test_records = load_records(&subset.test_path(&cli.data_dir))?;
    let rul_values = load_rul(&subset.rul_path(&cli.data_dir))?;

    let train_runs = group_by_unit(train_records);
    let test_runs = group_by_unit(test_records);

    // --- Sanity checks against the readme's stated unit counts ---
    check_unit_count("train", subset, train_runs.len(), subset.expected_train_units());
    check_unit_count("test", subset, test_runs.len(), subset.expected_test_units());
    if rul_values.len() != test_runs.len() {
        println!(
            "  WARNING: RUL file has {} entries but test set has {} units",
            rul_values.len(),
            test_runs.len()
        );
    }

    // --- EDA ---
    let train_len_stats = cycle_length_stats(&train_runs);
    let test_len_stats = cycle_length_stats(&test_runs);
    println!(
        "train run length (cycles): min {}  mean {:.1}  max {}",
        train_len_stats.min, train_len_stats.mean, train_len_stats.max
    );
    println!(
        "test  run length (cycles): min {}  mean {:.1}  max {}",
        test_len_stats.min, test_len_stats.mean, test_len_stats.max
    );

    let stats = sensor_stats(&train_runs);
    let flat = near_constant_sensors(&stats);
    if flat.is_empty() {
        println!("near-constant sensors (std < {:.0e}): none", cmapss_rul::eda::NEAR_CONSTANT_STD_THRESHOLD);
    } else {
        println!(
            "near-constant sensors (std < {:.0e}): {:?}",
            cmapss_rul::eda::NEAR_CONSTANT_STD_THRESHOLD, flat
        );
    }

    // --- RUL labeling ---
    let train_labels: Vec<Vec<u32>> = train_runs
        .iter()
        .map(|run| label_train_rul(run, cli.rul_cap))
        .collect();
    let test_labels: Vec<Vec<u32>> = test_runs
        .iter()
        .zip(rul_values.iter())
        .map(|(run, &final_rul)| label_test_rul(run, final_rul, cli.rul_cap))
        .collect();

    let total_train_cycles: usize = train_runs.iter().map(|r| r.records.len()).sum();
    let total_test_cycles: usize = test_runs.iter().map(|r| r.records.len()).sum();
    println!(
        "loaded {} train cycles across {} units, {} test cycles across {} units",
        total_train_cycles,
        train_runs.len(),
        total_test_cycles,
        test_runs.len()
    );

    // --- Windowed features (Phase 2) ---
    let train_features: Vec<Vec<WindowedFeatures>> = train_runs
        .iter()
        .zip(train_labels.iter())
        .map(|(run, labels)| compute_windowed_features(run, labels, cli.window))
        .collect();
    let test_features: Vec<Vec<WindowedFeatures>> = test_runs
        .iter()
        .zip(test_labels.iter())
        .map(|(run, labels)| compute_windowed_features(run, labels, cli.window))
        .collect();

    let empty_test_units: Vec<u32> = test_runs
        .iter()
        .zip(test_features.iter())
        .filter(|(_, f)| f.is_empty())
        .map(|(r, _)| r.unit)
        .collect();
    if !empty_test_units.is_empty() {
        println!(
            "  WARNING: window={} is >= the run length for {} test unit(s) {:?} - they contribute \
             zero windowed rows, including at their scored final cycle",
            cli.window,
            empty_test_units.len(),
            empty_test_units
        );
    }

    let total_train_windows: usize = train_features.iter().map(|f| f.len()).sum();
    let total_test_windows: usize = test_features.iter().map(|f| f.len()).sum();
    println!(
        "window={}: {} train windowed rows, {} test windowed rows",
        cli.window, total_train_windows, total_test_windows
    );

    if !cli.report_only {
        fs::create_dir_all(&cli.out_dir).map_err(|source| cmapss_rul::CmapssError::Io {
            path: cli.out_dir.clone(),
            source,
        })?;
        write_labeled_csv(
            &cli.out_dir.join(format!("{}_train_labeled.csv", subset.code())),
            &train_runs,
            &train_labels,
        )?;
        write_labeled_csv(
            &cli.out_dir.join(format!("{}_test_labeled.csv", subset.code())),
            &test_runs,
            &test_labels,
        )?;
        write_features_csv(
            &cli.out_dir.join(format!("{}_train_features.csv", subset.code())),
            &train_features,
        )?;
        write_features_csv(
            &cli.out_dir.join(format!("{}_test_features.csv", subset.code())),
            &test_features,
        )?;
        println!("wrote labeled + windowed-feature CSVs to {}", cli.out_dir.display());
    }

    Ok(())
}

fn check_unit_count(split: &str, subset: Subset, actual: usize, expected: usize) {
    if actual != expected {
        println!(
            "  WARNING: {} {} has {} units, readme states {}",
            subset, split, actual, expected
        );
    }
}

fn write_features_csv(path: &PathBuf, per_run_features: &[Vec<WindowedFeatures>]) -> Result<()> {
    let mut out = String::new();
    out.push_str("unit,cycle,op_setting_1,op_setting_2,op_setting_3,");
    for i in 1..=21 {
        out.push_str(&format!("sensor_{}_raw,", i));
    }
    for i in 1..=21 {
        out.push_str(&format!("sensor_{}_mean,", i));
    }
    for i in 1..=21 {
        out.push_str(&format!("sensor_{}_std,", i));
    }
    for i in 1..=21 {
        out.push_str(&format!("sensor_{}_slope,", i));
    }
    out.push_str("rul\n");

    for run_features in per_run_features {
        for f in run_features {
            out.push_str(&format!("{},{},", f.unit, f.cycle));
            for s in &f.op_settings {
                out.push_str(&format!("{},", s));
            }
            for s in &f.raw_sensors {
                out.push_str(&format!("{},", s));
            }
            for s in &f.rolling_mean {
                out.push_str(&format!("{},", s));
            }
            for s in &f.rolling_std {
                out.push_str(&format!("{},", s));
            }
            for s in &f.rolling_slope {
                out.push_str(&format!("{},", s));
            }
            out.push_str(&format!("{}\n", f.rul));
        }
    }

    let mut file = fs::File::create(path).map_err(|source| cmapss_rul::CmapssError::Io {
        path: path.clone(),
        source,
    })?;
    file.write_all(out.as_bytes())
        .map_err(|source| cmapss_rul::CmapssError::Io {
            path: path.clone(),
            source,
        })?;
    Ok(())
}

fn write_labeled_csv(path: &PathBuf, runs: &[EngineRun], labels: &[Vec<u32>]) -> Result<()> {
    let mut out = String::new();
    out.push_str("unit,cycle,");
    out.push_str("op_setting_1,op_setting_2,op_setting_3,");
    for i in 1..=21 {
        out.push_str(&format!("sensor_{},", i));
    }
    out.push_str("rul\n");

    for (run, run_labels) in runs.iter().zip(labels.iter()) {
        for (record, &rul) in run.records.iter().zip(run_labels.iter()) {
            out.push_str(&format!("{},{},", record.unit, record.cycle));
            for s in &record.op_settings {
                out.push_str(&format!("{},", s));
            }
            for s in &record.sensors {
                out.push_str(&format!("{},", s));
            }
            out.push_str(&format!("{}\n", rul));
        }
    }

    let mut file = fs::File::create(path).map_err(|source| cmapss_rul::CmapssError::Io {
        path: path.clone(),
        source,
    })?;
    file.write_all(out.as_bytes())
        .map_err(|source| cmapss_rul::CmapssError::Io {
            path: path.clone(),
            source,
        })?;
    Ok(())
}
