use crate::loader::EngineRun;

/// Default cap for piecewise-linear RUL labeling.
///
/// Early in an engine's life there's no meaningful degradation trend to
/// regress against, so raw "cycles until failure" isn't a sensible target
/// near cycle 1 of a 300-cycle run. The standard fix in the C-MAPSS
/// literature (introduced by Heimes, "Recurrent Neural Networks for
/// Remaining Useful Life Estimation", PHM08) is to flatten RUL at a cap for
/// early cycles and let it decay linearly only once the engine is actually
/// approaching failure.
///
/// 125 is the most commonly cited cap in follow-on work, but it isn't a
/// universal constant — some papers use different values per subset. It's
/// exposed as a CLI flag here rather than hard-coded for that reason.
pub const DEFAULT_RUL_CAP: u32 = 125;

/// Labels every cycle of a **training** run with its piecewise-linear RUL.
///
/// Training runs go all the way to failure, so the last recorded cycle
/// defines RUL = 0 and everything before it counts down from there,
/// capped at `cap`.
pub fn label_train_rul(run: &EngineRun, cap: u32) -> Vec<u32> {
    let failure_cycle = run.last_cycle();
    run.records
        .iter()
        .map(|r| (failure_cycle - r.cycle).min(cap))
        .collect()
}

/// Labels every cycle of a **test** run with its piecewise-linear RUL.
///
/// Test runs are truncated before failure, so we don't know the failure
/// cycle directly — but `RUL_FDxxx.txt` tells us the true RUL at the last
/// recorded cycle (`final_rul`). Earlier cycles are reconstructed by adding
/// back the cycles remaining until that last recorded point, then the same
/// cap is applied for consistency with the training labels.
pub fn label_test_rul(run: &EngineRun, final_rul: u32, cap: u32) -> Vec<u32> {
    let last_cycle = run.last_cycle();
    run.records
        .iter()
        .map(|r| {
            let cycles_after_this_point = last_cycle - r.cycle;
            (final_rul + cycles_after_this_point).min(cap)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::CycleRecord;

    fn make_run(unit: u32, cycles: &[u32]) -> EngineRun {
        EngineRun {
            unit,
            records: cycles
                .iter()
                .map(|&cycle| CycleRecord {
                    unit,
                    cycle,
                    op_settings: [0.0; 3],
                    sensors: [0.0; 21],
                })
                .collect(),
        }
    }

    #[test]
    fn train_rul_counts_down_to_zero_at_failure() {
        let run = make_run(1, &[1, 2, 3, 4, 5]);
        let ruls = label_train_rul(&run, 125);
        assert_eq!(ruls, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn train_rul_is_capped_for_long_runs() {
        let cycles: Vec<u32> = (1..=200).collect();
        let run = make_run(1, &cycles);
        let ruls = label_train_rul(&run, 125);
        assert_eq!(ruls[0], 125, "first cycle of a long run should hit the cap");
        assert_eq!(*ruls.last().unwrap(), 0, "last cycle is always failure, RUL 0");
        // Monotonically non-increasing.
        assert!(ruls.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn test_rul_reconstruction_matches_provided_final_value() {
        let run = make_run(7, &[1, 2, 3]);
        let ruls = label_test_rul(&run, 50, 125);
        // Last recorded cycle's RUL must equal the ground-truth value exactly.
        assert_eq!(*ruls.last().unwrap(), 50);
        // Earlier cycles count up from there.
        assert_eq!(ruls, vec![52, 51, 50]);
    }

    #[test]
    fn test_rul_reconstruction_respects_cap() {
        let cycles: Vec<u32> = (1..=200).collect();
        let run = make_run(1, &cycles);
        let ruls = label_test_rul(&run, 10, 125);
        assert_eq!(ruls[0], 125);
        assert_eq!(*ruls.last().unwrap(), 10);
    }
}
