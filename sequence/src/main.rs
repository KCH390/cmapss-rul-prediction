//! Phase 6 (stretch): LSTM sequence model.
//!
//! *** VERIFICATION STATUS: NOT compiled or run in the sandbox this project
//! was otherwise built in. *** Every other line of code in this repo was
//! built, tested, and run against real data before being shipped. This file
//! is the one exception - `candle-core`/`candle-nn` need a newer Rust
//! edition than the sandbox's apt-installed rustc 1.75 (the same wall that
//! blocked `clap`, `evcxr_jupyter`, and `plotters`' default font backend
//! earlier in this project, at increasing dependency depth each time; five
//! separate version-pin attempts across both `candle` and `burn` all hit
//! the same root cause). Written as carefully and conservatively as
//! possible against candle's documented API, but if it doesn't compile
//! cleanly on the first try on a real toolchain, that's expected - paste
//! the compiler error back and it's a normal fix, not a sign anything else
//! in this project is unreliable.
//!
//! Unlike Phase 3's baseline models, an LSTM needs properly scaled inputs
//! to train well with gradient descent (trees don't care about feature
//! scale; gradient-based optimizers do). Rather than write a new
//! standardization routine, this reuses Phase 5's regime-normalization
//! machinery (`core::regime`) for every subset, not just FD002/FD004 - for
//! FD001/FD003 that's k=1, i.e. plain global z-scoring, which is exactly
//! the input scaling an LSTM needs anyway. Same evaluation protocol as
//! every other model in this project: one prediction per test engine, at
//! the window ending on its last recorded cycle.

use std::path::PathBuf;
use std::str::FromStr;

use candle_core::{DType, Device, Tensor};
use candle_nn::rnn::{lstm, LSTMConfig, RNN};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use cmapss_rul::dataset::Subset;
use cmapss_rul::eda::{near_constant_sensors, sensor_stats};
use cmapss_rul::loader::{group_by_unit, load_records, load_rul};
use cmapss_rul::regime::{compute_regime_stats, fit_regimes, normalize_run};
use cmapss_rul::features::DEFAULT_WINDOW as SAFE_DEFAULT_WINDOW;
use cmapss_rul::rul::{label_test_rul, label_train_rul, DEFAULT_RUL_CAP};
use cmapss_rul::scoring::{phm08_score, rmse};
use cmapss_rul::sequence::{extract_sequences, num_features};

struct Cli {
    subset: Subset,
    rul_cap: u32,
    window: usize,
    hidden_size: usize,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    data_dir: PathBuf,
    out_dir: PathBuf,
}

impl Cli {
    fn parse(args: impl Iterator<Item = String>) -> Result<Cli, String> {
        let mut cli = Cli {
            subset: Subset::FD001,
            rul_cap: DEFAULT_RUL_CAP,
            window: SAFE_DEFAULT_WINDOW, // NOT 30: a larger window would panic on FD004's
            // shortest test run (19 cycles, see features.rs's DEFAULT_WINDOW doc comment) -
            // caught this exact mistake while writing this file, before it could ship as a
            // bug. Reusing the same constant the tree models use also keeps window size
            // comparable across models for a fairer before/after story.
            hidden_size: 32,
            epochs: 50,
            learning_rate: 1e-3,
            batch_size: 32,
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
                "--hidden-size" => parse_flag!(cli.hidden_size, "--hidden-size"),
                "--epochs" => parse_flag!(cli.epochs, "--epochs"),
                "--learning-rate" => parse_flag!(cli.learning_rate, "--learning-rate"),
                "--batch-size" => parse_flag!(cli.batch_size, "--batch-size"),
                "--data-dir" => cli.data_dir = PathBuf::from(args.next().ok_or("--data-dir requires a value")?),
                "--out-dir" => cli.out_dir = PathBuf::from(args.next().ok_or("--out-dir requires a value")?),
                other if !other.starts_with('-') => cli.subset = Subset::from_str(other)?,
                other => return Err(format!("unrecognized argument {:?}", other)),
            }
        }
        Ok(cli)
    }
}

/// LSTM -> final hidden state -> linear -> scalar RUL prediction.
struct RulLstm {
    lstm: candle_nn::rnn::LSTM,
    output: Linear,
}

impl RulLstm {
    fn new(input_size: usize, hidden_size: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let lstm = lstm(input_size, hidden_size, LSTMConfig::default(), vb.pp("lstm"))?;
        let output = linear(hidden_size, 1, vb.pp("output"))?;
        Ok(Self { lstm, output })
    }

    /// `x`: (batch, seq_len, input_size) -> (batch, 1)
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let states = self.lstm.seq(x)?;
        let last_state = states.last().expect("sequence length must be >= 1");
        self.output.forward(&last_state.h)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse(std::env::args().skip(1)).map_err(|e| {
        eprintln!("argument error: {}", e);
        eprintln!(
            "USAGE: sequence [SUBSET] [--window N] [--hidden-size N] [--epochs N] \
             [--learning-rate F] [--batch-size N] [--rul-cap N] [--data-dir PATH] [--out-dir PATH]"
        );
        e
    })?;

    println!("=== Phase 6 (stretch): LSTM sequence model: {} ===", cli.subset);
    println!(
        "window={} hidden_size={} epochs={} learning_rate={} batch_size={}",
        cli.window, cli.hidden_size, cli.epochs, cli.learning_rate, cli.batch_size
    );

    // --- Load ---
    let mut train_runs = group_by_unit(load_records(&cli.subset.train_path(&cli.data_dir))?);
    let mut test_runs = group_by_unit(load_records(&cli.subset.test_path(&cli.data_dir))?);
    let test_rul = load_rul(&cli.subset.rul_path(&cli.data_dir))?;

    // Near-constant sensors from RAW data, before normalization - see
    // models/src/main.rs and flagship.rs for why this ordering matters
    // (z-scoring always produces ~unit variance from whatever it's given,
    // so computing this after normalization would silently stop excluding
    // sensors that are still physically uninformative).
    let excluded = near_constant_sensors(&sensor_stats(&train_runs));
    println!("excluding near-constant sensors {:?}, {} features per timestep", excluded, num_features(&excluded));

    // --- Regime normalization, reused from Phase 5 as this model's input
    // scaling (k=1 for FD001/FD003 - plain global z-scoring; k=6 for
    // FD002/FD004 - regime-aware z-scoring, same as Phase 5) ---
    let k = cli.subset.num_conditions() as usize;
    let regimes = fit_regimes(&train_runs, k);
    let regime_stats = compute_regime_stats(&train_runs, &regimes);
    train_runs = train_runs.iter().map(|r| normalize_run(r, &regimes, &regime_stats)).collect();
    test_runs = test_runs.iter().map(|r| normalize_run(r, &regimes, &regime_stats)).collect();

    let train_labels: Vec<Vec<u32>> = train_runs.iter().map(|r| label_train_rul(r, cli.rul_cap)).collect();
    let test_labels: Vec<Vec<u32>> = test_runs
        .iter()
        .zip(test_rul.iter())
        .map(|(r, &final_rul)| label_test_rul(r, final_rul, cli.rul_cap))
        .collect();

    // --- Build sequences: every window for training (pooled), only the
    // last window per engine for evaluation - same protocol as every other
    // model in this project. ---
    let mut train_samples = Vec::new();
    for (run, labels) in train_runs.iter().zip(train_labels.iter()) {
        train_samples.extend(extract_sequences(run, labels, cli.window, &excluded));
    }
    println!("training sequences: {}", train_samples.len());

    let mut test_samples = Vec::new();
    for (run, labels) in test_runs.iter().zip(test_labels.iter()) {
        let sequences = extract_sequences(run, labels, cli.window, &excluded);
        let last = sequences
            .last()
            .expect("every test unit must yield >=1 sequence - check --window isn't larger than the shortest test run");
        test_samples.push(last.clone());
    }
    println!("test engines evaluated: {}", test_samples.len());

    // --- Tensors ---
    let device = Device::Cpu;
    let n_features = num_features(&excluded);
    let seq_len = cli.window;

    let x_train_flat: Vec<f32> = train_samples.iter().flat_map(|s| s.sequence.iter().map(|&v| v as f32)).collect();
    let y_train_flat: Vec<f32> = train_samples.iter().map(|s| s.rul as f32).collect();
    let n_train = train_samples.len();
    let x_train = Tensor::from_vec(x_train_flat, (n_train, seq_len, n_features), &device)?;
    let y_train = Tensor::from_vec(y_train_flat, (n_train, 1), &device)?;

    let x_test_flat: Vec<f32> = test_samples.iter().flat_map(|s| s.sequence.iter().map(|&v| v as f32)).collect();
    let n_test = test_samples.len();
    let x_test = Tensor::from_vec(x_test_flat, (n_test, seq_len, n_features), &device)?;

    // --- Model + optimizer ---
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = RulLstm::new(n_features, cli.hidden_size, vb)?;
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: cli.learning_rate, ..Default::default() })?;

    // --- Training loop (no shuffling - a documented simplification, see README) ---
    for epoch in 0..cli.epochs {
        let mut total_loss = 0f32;
        let mut n_batches = 0usize;
        let mut start = 0usize;
        while start < n_train {
            let end = (start + cli.batch_size).min(n_train);
            let bsz = end - start;
            let x_batch = x_train.narrow(0, start, bsz)?;
            let y_batch = y_train.narrow(0, start, bsz)?;

            let pred = model.forward(&x_batch)?;
            let loss = candle_nn::loss::mse(&pred, &y_batch)?;
            opt.backward_step(&loss)?;

            total_loss += loss.to_scalar::<f32>()?;
            n_batches += 1;
            start = end;
        }
        if epoch % 5 == 0 || epoch == cli.epochs - 1 {
            println!(
                "epoch {:>3}: train MSE = {:.2} (RMSE ~= {:.2})",
                epoch,
                total_loss / n_batches as f32,
                (total_loss / n_batches as f32).sqrt()
            );
        }
    }

    // --- Evaluate: same RMSE + PHM08 score as every other model ---
    let preds = model.forward(&x_test)?;
    let preds_vec: Vec<f32> = preds.squeeze(1)?.to_vec1()?;
    let pairs: Vec<(f64, f64)> = preds_vec
        .iter()
        .zip(test_samples.iter())
        .map(|(&pred, sample)| (pred as f64, sample.rul as f64))
        .collect();

    let rmse_val = rmse(&pairs);
    let score_val = phm08_score(&pairs);
    let late = pairs.iter().filter(|(pred, actual)| pred >= actual).count();
    let early = pairs.len() - late;

    println!("\n--- LSTM sequence model ---");
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
    let out_path = cli.out_dir.join(format!("{}_sequence_predictions.csv", cli.subset.code()));
    let mut csv = String::from("unit,true_rul,pred_lstm,error_lstm\n");
    for (i, sample) in test_samples.iter().enumerate() {
        csv.push_str(&format!(
            "{},{},{},{}\n",
            sample.unit,
            sample.rul,
            preds_vec[i],
            preds_vec[i] as f64 - sample.rul as f64
        ));
    }
    std::fs::write(&out_path, csv)?;
    println!("wrote predictions to {}", out_path.display());

    Ok(())
}
