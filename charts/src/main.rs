//! Generates sanity-check charts for the Phase 1/2 pipeline as static PNGs.
//!
//! This lives in its own workspace crate (`charts/`) rather than as a bin
//! target in the core crate, on purpose: `plotters` is a real dependency
//! (rendering pixels is plumbing, not differentiating logic - see
//! `charts/Cargo.toml`), and putting it in a separate crate means it can't
//! leak into the core pipeline's build graph. `cargo test` or `cargo build`
//! against the core crate alone never touches plotters, font-kit, or any
//! of their dependencies.

use std::path::Path;

use plotters::prelude::*;

use cmapss_rul::dataset::Subset;
use cmapss_rul::eda::sensor_rul_correlation;
use cmapss_rul::features::{compute_windowed_features, WindowedFeatures, DEFAULT_WINDOW};
use cmapss_rul::loader::{group_by_unit, load_records, EngineRun};
use cmapss_rul::parser::NUM_SENSORS;
use cmapss_rul::rul::{label_train_rul, DEFAULT_RUL_CAP};

/// Sensor kept from the original Phase 2 chart on purpose: it has the
/// *highest raw variance* in FD001's training set, which made it look like
/// an obvious pick for a "does the rolling mean smooth the signal" demo.
/// It turned out to have almost no correlation with RUL - noisy, not
/// informative. Charting it deliberately alongside the top-correlated
/// sensors makes that contrast visible instead of quietly picking a
/// better sensor and losing the lesson.
const HIGH_VARIANCE_LOW_RELEVANCE_SENSOR: usize = 9;

/// How many top-correlated sensors to chart individually.
const TOP_N_SENSORS_TO_CHART: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = Path::new("data/raw/CMAPSSData");
    let out_dir = Path::new("reports/figures");
    std::fs::create_dir_all(out_dir)?;

    let subset = Subset::FD001;
    let train_records = load_records(&subset.train_path(data_dir))?;
    let train_runs = group_by_unit(train_records);

    let labels: Vec<Vec<u32>> = train_runs
        .iter()
        .map(|run| label_train_rul(run, DEFAULT_RUL_CAP))
        .collect();

    let correlations = sensor_rul_correlation(&train_runs, &labels);

    // Rank sensors by |correlation| with RUL, descending.
    let mut ranked: Vec<(usize, f64)> = correlations
        .iter()
        .enumerate()
        .map(|(idx, &c)| (idx + 1, c)) // 1-indexed sensor number
        .collect();
    ranked.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Sensor-RUL correlation ranking (FD001 training set):");
    for (sensor_num, corr) in &ranked {
        println!("  sensor {:>2}: {:+.3}", sensor_num, corr);
    }

    draw_correlation_bar_chart(&correlations, &out_dir.join("FD001_sensor_rul_correlation.png"))?;

    // Pick the longest-running engine for the clearest visual.
    let longest_run = train_runs
        .iter()
        .max_by_key(|r| r.last_cycle())
        .expect("no runs loaded");
    let run_labels = label_train_rul(longest_run, DEFAULT_RUL_CAP);
    let windowed = compute_windowed_features(longest_run, &run_labels, DEFAULT_WINDOW);

    // Chart the top-N most correlated sensors...
    let mut charted: Vec<usize> = Vec::new();
    for &(sensor_num, corr) in ranked.iter().take(TOP_N_SENSORS_TO_CHART) {
        println!(
            "charting {} unit {}, sensor {} (corr with RUL: {:+.3}, ranked #{} by |correlation|)",
            subset,
            longest_run.unit,
            sensor_num,
            corr,
            charted.len() + 1
        );
        draw_raw_vs_rolling_mean(
            longest_run,
            sensor_num,
            &windowed,
            corr,
            &out_dir.join(format!(
                "{}_unit{}_sensor{}_rolling.png",
                subset.code(),
                longest_run.unit,
                sensor_num
            )),
        )?;
        charted.push(sensor_num);
    }

    // ...plus the high-variance-but-uninformative sensor, for contrast, if
    // it wasn't already in the top N.
    if !charted.contains(&HIGH_VARIANCE_LOW_RELEVANCE_SENSOR) {
        let corr = correlations[HIGH_VARIANCE_LOW_RELEVANCE_SENSOR - 1];
        println!(
            "charting {} unit {}, sensor {} (corr with RUL: {:+.3}) - highest raw variance in the \
             training set, kept as a contrast: high variance is not the same as high relevance",
            subset, longest_run.unit, HIGH_VARIANCE_LOW_RELEVANCE_SENSOR, corr
        );
        draw_raw_vs_rolling_mean(
            longest_run,
            HIGH_VARIANCE_LOW_RELEVANCE_SENSOR,
            &windowed,
            corr,
            &out_dir.join(format!(
                "{}_unit{}_sensor{}_rolling.png",
                subset.code(),
                longest_run.unit,
                HIGH_VARIANCE_LOW_RELEVANCE_SENSOR
            )),
        )?;
    }

    draw_rul_shape(
        longest_run,
        &run_labels,
        &out_dir.join(format!("{}_unit{}_rul_shape.png", subset.code(), longest_run.unit)),
    )?;

    println!("wrote charts to {}", out_dir.display());
    Ok(())
}

fn draw_correlation_bar_chart(
    correlations: &[f64; NUM_SENSORS],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (1000, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "FD001: Pearson correlation of each sensor with labeled RUL",
            ("sans-serif", 22),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0.5f64..(NUM_SENSORS as f64 + 0.5), -1.05f64..1.05f64)?;

    chart
        .configure_mesh()
        .x_desc("sensor number")
        .y_desc("correlation with RUL")
        .x_labels(NUM_SENSORS)
        .x_label_formatter(&|x| format!("{}", *x as i32))
        .draw()?;

    chart.draw_series(correlations.iter().enumerate().map(|(idx, &corr)| {
        let x = (idx + 1) as f64;
        let color = if corr >= 0.0 { BLUE.filled() } else { RED.filled() };
        Rectangle::new([(x - 0.35, 0.0), (x + 0.35, corr)], color)
    }))?;

    root.present()?;
    Ok(())
}

fn draw_raw_vs_rolling_mean(
    run: &EngineRun,
    sensor_number: usize, // 1-indexed
    windowed: &[WindowedFeatures],
    correlation: f64,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let sensor_idx = sensor_number - 1;
    let root = BitMapBackend::new(path, (960, 540)).into_drawing_area();
    root.fill(&WHITE)?;

    let raw: Vec<(f64, f64)> = run
        .records
        .iter()
        .map(|r| (r.cycle as f64, r.sensors[sensor_idx]))
        .collect();
    let rolling: Vec<(f64, f64)> = windowed
        .iter()
        .map(|f| (f.cycle as f64, f.rolling_mean[sensor_idx]))
        .collect();

    let y_min = raw.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = raw.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let pad = (y_max - y_min) * 0.1 + 1e-6;
    let x_max = run.last_cycle() as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Unit {} - sensor {} raw vs. {}-cycle rolling mean (corr with RUL: {:+.3})",
                run.unit, sensor_number, DEFAULT_WINDOW, correlation
            ),
            ("sans-serif", 20),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..x_max, (y_min - pad)..(y_max + pad))?;

    chart
        .configure_mesh()
        .x_desc("cycle")
        .y_desc(format!("sensor {} value", sensor_number))
        .draw()?;

    chart
        .draw_series(LineSeries::new(raw, ShapeStyle::from(&BLUE.mix(0.4)).stroke_width(1)))?
        .label("raw")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.4)));

    chart
        .draw_series(LineSeries::new(rolling, ShapeStyle::from(&RED).stroke_width(2)))?
        .label(format!("{}-cycle rolling mean", DEFAULT_WINDOW))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn draw_rul_shape(run: &EngineRun, labels: &[u32], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (960, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let points: Vec<(f64, f64)> = run
        .records
        .iter()
        .zip(labels.iter())
        .map(|(r, &rul)| (r.cycle as f64, rul as f64))
        .collect();

    let x_max = run.last_cycle() as f64;
    let y_max = DEFAULT_RUL_CAP as f64 * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Unit {} - piecewise-linear RUL label (cap={})", run.unit, DEFAULT_RUL_CAP),
            ("sans-serif", 22),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..x_max, 0f64..y_max)?;

    chart.configure_mesh().x_desc("cycle").y_desc("labeled RUL").draw()?;

    chart.draw_series(LineSeries::new(points, ShapeStyle::from(&RED).stroke_width(2)))?;

    root.present()?;
    Ok(())
}
