use crate::loader::EngineRun;
use crate::parser::NUM_SENSORS;

/// Min/mean/max over a set of run lengths (in cycles).
#[derive(Debug, Clone, Copy)]
pub struct CycleLengthStats {
    pub min: u32,
    pub max: u32,
    pub mean: f64,
}

pub fn cycle_length_stats(runs: &[EngineRun]) -> CycleLengthStats {
    let lengths: Vec<u32> = runs.iter().map(|r| r.last_cycle()).collect();
    let min = *lengths.iter().min().expect("no runs given");
    let max = *lengths.iter().max().expect("no runs given");
    let mean = lengths.iter().sum::<u32>() as f64 / lengths.len() as f64;
    CycleLengthStats { min, max, mean }
}

/// Mean and (sample) standard deviation for one sensor channel across every
/// cycle of every run passed in.
#[derive(Debug, Clone, Copy)]
pub struct SensorStats {
    pub index: usize,
    pub mean: f64,
    pub std_dev: f64,
}

/// A sensor is flagged "near-constant" if its sample std dev falls below
/// this threshold. It's a well-known property of C-MAPSS that a handful of
/// sensors carry ~no signal in some subsets — but which ones, and in which
/// subsets, is computed here from the actual data rather than asserted from
/// memory of the literature.
pub const NEAR_CONSTANT_STD_THRESHOLD: f64 = 1e-6;

/// Computes per-sensor mean/std across every cycle of every run.
pub fn sensor_stats(runs: &[EngineRun]) -> [SensorStats; NUM_SENSORS] {
    let mut sums = [0.0f64; NUM_SENSORS];
    let mut count = 0usize;

    for run in runs {
        for record in &run.records {
            for i in 0..NUM_SENSORS {
                sums[i] += record.sensors[i];
            }
            count += 1;
        }
    }
    let means: Vec<f64> = sums.iter().map(|s| s / count as f64).collect();

    let mut sq_diff_sums = [0.0f64; NUM_SENSORS];
    for run in runs {
        for record in &run.records {
            for i in 0..NUM_SENSORS {
                let diff = record.sensors[i] - means[i];
                sq_diff_sums[i] += diff * diff;
            }
        }
    }

    let mut stats = [SensorStats {
        index: 0,
        mean: 0.0,
        std_dev: 0.0,
    }; NUM_SENSORS];

    for i in 0..NUM_SENSORS {
        // Sample standard deviation (n-1 denominator). count is always >> 1
        // for this dataset so the n vs n-1 choice doesn't matter in practice;
        // n-1 is used for consistency with typical downstream stats tooling.
        let variance = sq_diff_sums[i] / (count as f64 - 1.0);
        stats[i] = SensorStats {
            index: i + 1, // 1-indexed to match the dataset's own "sensor measurement N" naming
            mean: means[i],
            std_dev: variance.sqrt(),
        };
    }
    stats
}

pub fn near_constant_sensors(stats: &[SensorStats; NUM_SENSORS]) -> Vec<usize> {
    stats
        .iter()
        .filter(|s| s.std_dev < NEAR_CONSTANT_STD_THRESHOLD)
        .map(|s| s.index)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::CycleRecord;

    /// Builds a run where sensor 1 (index 0) takes the given `values` across
    /// cycles, and every *other* sensor is deliberately made to vary too
    /// (via the cycle index) so the test isolates sensor 1's behavior rather
    /// than incidentally leaving 20 other sensors flat at 0.0.
    fn run_with_sensor_values(unit: u32, values: &[f64]) -> EngineRun {
        EngineRun {
            unit,
            records: values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let mut sensors = [0.0; NUM_SENSORS];
                    sensors[0] = v;
                    for (j, s) in sensors.iter_mut().enumerate().skip(1) {
                        *s = i as f64 + j as f64; // varies every cycle, never constant
                    }
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
    fn detects_constant_sensor() {
        let run = run_with_sensor_values(1, &[5.0, 5.0, 5.0, 5.0]);
        let stats = sensor_stats(&[run]);
        assert!(stats[0].std_dev < NEAR_CONSTANT_STD_THRESHOLD);
        assert_eq!(near_constant_sensors(&stats), vec![1]);
    }

    #[test]
    fn detects_varying_sensor() {
        let run = run_with_sensor_values(1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = sensor_stats(&[run]);
        assert!(stats[0].std_dev > NEAR_CONSTANT_STD_THRESHOLD);
        assert_eq!(near_constant_sensors(&stats), Vec::<usize>::new());
    }

    #[test]
    fn cycle_length_stats_are_correct() {
        let runs = vec![
            run_with_sensor_values(1, &[0.0; 10]),
            run_with_sensor_values(2, &[0.0; 20]),
            run_with_sensor_values(3, &[0.0; 30]),
        ];
        let stats = cycle_length_stats(&runs);
        assert_eq!(stats.min, 10);
        assert_eq!(stats.max, 30);
        assert_eq!(stats.mean, 20.0);
    }
}
