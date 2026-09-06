//! Evaluation metrics for RUL predictions: RMSE and the official PHM08
//! asymmetric scoring function.
//!
//! Both operate on `(predicted, true)` RUL pairs — one pair per test
//! **engine**, evaluated at that engine's last recorded cycle. That's a
//! deliberate scope, not an oversight: the official competition protocol
//! scores one prediction per engine (the one made at the point its data
//! was truncated), not one prediction per windowed row. Scoring every
//! windowed row of a test trajectory would inflate apparent performance,
//! since consecutive windows from the same engine are highly correlated -
//! see `models` crate for where these pairs actually get assembled.

/// Root mean square error over predicted/true RUL pairs.
///
/// Symmetric: over- and under-predicting by the same amount cost the same.
/// See [`phm08_score`] for the asymmetric alternative the literature
/// actually optimizes for in this domain.
pub fn rmse(pairs: &[(f64, f64)]) -> f64 {
    assert!(!pairs.is_empty(), "rmse of an empty prediction set is undefined");
    let sum_sq: f64 = pairs.iter().map(|(pred, actual)| (pred - actual).powi(2)).sum();
    (sum_sq / pairs.len() as f64).sqrt()
}

/// The scoring function from the PHM08 Prognostics Data Challenge (Saxena
/// et al., 2008), as used throughout the C-MAPSS literature for comparing
/// RUL predictors.
///
/// For each pair, `d = predicted - actual`:
/// - `d < 0` (predicted RUL is *less* than actual — an early/conservative
///   prediction, meaning the model called for maintenance sooner than
///   strictly necessary): `exp(-d/13) - 1`
/// - `d >= 0` (predicted RUL is *more* than actual — a late/dangerous
///   prediction, meaning the model would have let the engine keep running
///   past its actual failure point): `exp(d/10) - 1`
///
/// The steeper divisor (10 vs 13) on the `d >= 0` branch is what makes this
/// asymmetric: late predictions are penalized more heavily than early ones
/// of the same magnitude, reflecting that running an engine past failure is
/// worse than servicing it a bit early. Total score is the sum across all
/// pairs; 0 is a perfect predictor, and there's no upper bound (a single
/// badly-late prediction can dominate the sum, since the late branch is
/// exponential with a steep divisor - a known sensitivity of this metric
/// discussed in the literature, which is why RMSE is always reported
/// alongside it rather than instead of it).
pub fn phm08_score(pairs: &[(f64, f64)]) -> f64 {
    pairs
        .iter()
        .map(|(pred, actual)| {
            let d = pred - actual;
            if d < 0.0 {
                (-d / 13.0).exp() - 1.0
            } else {
                (d / 10.0).exp() - 1.0
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_predictions_score_zero_on_both_metrics() {
        let pairs = vec![(50.0, 50.0), (10.0, 10.0), (0.0, 0.0)];
        assert_eq!(rmse(&pairs), 0.0);
        assert!((phm08_score(&pairs)).abs() < 1e-9);
    }

    #[test]
    fn rmse_matches_hand_computation() {
        // errors: 3, -4 -> squared: 9, 16 -> mean 12.5 -> sqrt ~3.5355
        let pairs = vec![(53.0, 50.0), (6.0, 10.0)];
        assert!((rmse(&pairs) - 12.5f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn late_prediction_is_penalized_more_than_equal_magnitude_early_prediction() {
        let early = vec![(40.0, 50.0)]; // d = -10
        let late = vec![(60.0, 50.0)]; // d = +10
        assert!(
            phm08_score(&late) > phm08_score(&early),
            "a late prediction of the same magnitude should score worse than an early one"
        );
    }

    #[test]
    fn score_matches_hand_computed_value_for_a_known_case() {
        // d = +10 (late): exp(10/10) - 1 = e - 1
        let pairs = vec![(60.0, 50.0)];
        let expected = std::f64::consts::E - 1.0;
        assert!((phm08_score(&pairs) - expected).abs() < 1e-9);

        // d = -13 (early): exp(13/13) - 1 = e - 1, same magnitude of penalty
        // as a late error of exactly 10 - a concrete illustration of how
        // much more steeply the late branch grows per unit of error.
        let pairs_early = vec![(37.0, 50.0)];
        let expected_early = std::f64::consts::E - 1.0;
        assert!((phm08_score(&pairs_early) - expected_early).abs() < 1e-9);
    }

    #[test]
    fn score_is_additive_across_pairs() {
        let a = vec![(55.0, 50.0)];
        let b = vec![(40.0, 50.0)];
        let combined = vec![(55.0, 50.0), (40.0, 50.0)];
        assert!((phm08_score(&combined) - (phm08_score(&a) + phm08_score(&b))).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "empty")]
    fn rmse_panics_on_empty_input() {
        rmse(&[]);
    }
}
