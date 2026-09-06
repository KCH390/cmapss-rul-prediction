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
use cmapss_rul::eda::sensor_stats;
use cmapss_rul::features::{compute_windowed_features, DEFAULT_WINDOW};
use cmapss_rul::loader::{group_by_unit, load_records, EngineRun};
use cmapss_rul::rul::{label_train_rul, DEFAULT_RUL_CAP};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = Path::new("data/raw/CMAPSSData");
    let out_dir = Path::new("reports/figures");
    std::fs::create_dir_all(out_dir)?;

    let subset = Subset::FD001;
    let train_records = load_records(&subset.train_path(data_dir))?;
    let train_runs = group_by_unit(train_records);

    // Pick the sensor with the highest variance so the chart actually shows
    // something - not hard-coded, computed the same way Phase 1's EDA report does.
    let stats = sensor_stats(&train_runs);
    let most_variable = stats
        .iter()
        .max_by(|a, b| a.std_dev.partial_cmp(&b.std_dev).unwrap())
        .unwrap();
    let sensor_idx = most_variable.index - 1; // stats.index is 1-based

    // Pick the longest-running engine for the clearest visual.
    let longest_run = train_runs
        .iter()
        .max_by_key(|r| r.last_cycle())
        .expect("no runs loaded");

    println!(
        "charting {} unit {} (run length {} cycles), sensor {} (std dev {:.3}, the most variable in the training set)",
        subset,
        longest_run.unit,
        longest_run.last_cycle(),
        most_variable.index,
        most_variable.std_dev
    );

    let labels = label_train_rul(longest_run, DEFAULT_RUL_CAP);
    let windowed = compute_windowed_features(longest_run, &labels, DEFAULT_WINDOW);

    draw_raw_vs_rolling_mean(
        longest_run,
        sensor_idx,
        &windowed,
        most_variable.index,
        &out_dir.join(format!(
            "{}_unit{}_sensor{}_rolling.png",
            subset.code(),
            longest_run.unit,
            most_variable.index
        )),
    )?;

    draw_rul_shape(
        longest_run,
        &labels,
        &out_dir.join(format!("{}_unit{}_rul_shape.png", subset.code(), longest_run.unit)),
    )?;

    println!("wrote charts to {}", out_dir.display());
    Ok(())
}

fn draw_raw_vs_rolling_mean(
    run: &EngineRun,
    sensor_idx: usize,
    windowed: &[cmapss_rul::features::WindowedFeatures],
    sensor_number: usize,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
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
                "Unit {} - sensor {} raw vs. {}-cycle rolling mean",
                run.unit, sensor_number, DEFAULT_WINDOW
            ),
            ("sans-serif", 22),
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
