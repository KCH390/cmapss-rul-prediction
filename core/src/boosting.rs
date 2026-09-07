//! Gradient-boosted regression trees, built from scratch on top of
//! `tree::fit_tree`.
//!
//! This is the Phase 4 flagship model: sequential shallow trees, each
//! fit to the residuals of the ensemble so far, combined with a learning
//! rate (shrinkage). For squared-error loss (what this project uses
//! throughout - see `scoring::rmse`), the negative gradient at each step
//! is exactly `target - current_prediction`, i.e. the plain residual -
//! that simplification is why this doesn't need a general autodiff/gradient
//! framework, just repeated residual fitting.

use crate::tree::{fit_tree, TreeNode, TreeParams};

#[derive(Debug, Clone, Copy)]
pub struct GbmParams {
    pub n_trees: usize,
    pub learning_rate: f64,
    pub tree: TreeParams,
}

pub struct GradientBoostedTrees {
    trees: Vec<TreeNode>,
    learning_rate: f64,
    init_value: f64,
    /// Training-set RMSE after each tree is added, in order (length ==
    /// `n_trees`). Useful for a training-curve chart and for sanity
    /// checking that boosting is actually converging rather than
    /// stagnating or diverging - not used by `predict` itself.
    pub training_rmse: Vec<f64>,
}

impl GradientBoostedTrees {
    /// Fits the ensemble. `rows`/`targets` follow the same convention as
    /// `tree::fit_tree` and `design_matrix::to_feature_vector` - one
    /// `Vec<f64>` feature row per sample.
    pub fn fit(rows: &[Vec<f64>], targets: &[f64], params: &GbmParams) -> Self {
        assert_eq!(rows.len(), targets.len(), "one target required per row");
        assert!(!rows.is_empty(), "cannot fit on zero rows");

        let init_value = targets.iter().sum::<f64>() / targets.len() as f64;
        let mut predictions = vec![init_value; targets.len()];
        let mut trees = Vec::with_capacity(params.n_trees);
        let mut training_rmse = Vec::with_capacity(params.n_trees);

        for _ in 0..params.n_trees {
            // For squared-error loss, the pseudo-residual is just the
            // plain residual: target - current ensemble prediction.
            let residuals: Vec<f64> = targets
                .iter()
                .zip(predictions.iter())
                .map(|(&t, &p)| t - p)
                .collect();

            let tree = fit_tree(rows, &residuals, &params.tree);

            for (i, row) in rows.iter().enumerate() {
                predictions[i] += params.learning_rate * tree.predict(row);
            }

            let sse: f64 = targets
                .iter()
                .zip(predictions.iter())
                .map(|(&t, &p)| (t - p).powi(2))
                .sum();
            training_rmse.push((sse / targets.len() as f64).sqrt());

            trees.push(tree);
        }

        GradientBoostedTrees {
            trees,
            learning_rate: params.learning_rate,
            init_value,
            training_rmse,
        }
    }

    pub fn predict(&self, row: &[f64]) -> f64 {
        self.init_value
            + self.learning_rate * self.trees.iter().map(|t| t.predict(row)).sum::<f64>()
    }

    pub fn predict_batch(&self, rows: &[Vec<f64>]) -> Vec<f64> {
        rows.iter().map(|r| self.predict(r)).collect()
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params(n_trees: usize) -> GbmParams {
        GbmParams {
            n_trees,
            learning_rate: 0.1,
            tree: TreeParams { max_depth: 2, min_samples_leaf: 1 },
        }
    }

    #[test]
    fn zero_trees_predicts_the_mean_for_every_row() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![10.0, 20.0, 30.0];
        let model = GradientBoostedTrees::fit(&rows, &targets, &default_params(0));
        assert_eq!(model.predict(&[1.0]), 20.0);
        assert_eq!(model.predict(&[999.0]), 20.0); // still just the mean - no trees to differentiate
    }

    #[test]
    fn training_rmse_length_matches_n_trees() {
        let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = (0..20).map(|i| i as f64 * 2.0).collect();
        let model = GradientBoostedTrees::fit(&rows, &targets, &default_params(15));
        assert_eq!(model.training_rmse.len(), 15);
        assert_eq!(model.n_trees(), 15);
    }

    #[test]
    fn training_rmse_generally_decreases_as_trees_are_added() {
        let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = (0..40).map(|i| (i as f64 * 2.0 - 40.0).abs()).collect(); // V-shape, needs >1 split
        let model = GradientBoostedTrees::fit(&rows, &targets, &default_params(30));

        let first = model.training_rmse[0];
        let last = *model.training_rmse.last().unwrap();
        assert!(
            last < first,
            "expected training RMSE to improve from {} to something lower, got {}",
            first,
            last
        );
        // Should also be non-increasing overall (allow tiny float noise),
        // not wildly oscillating - shrinkage should give smooth convergence.
        let regressions = model
            .training_rmse
            .windows(2)
            .filter(|w| w[1] > w[0] + 1e-9)
            .count();
        assert!(
            (regressions as f64) < model.training_rmse.len() as f64 * 0.2,
            "training RMSE regressed on more than 20% of steps: {} out of {}",
            regressions,
            model.training_rmse.len()
        );
    }

    #[test]
    fn more_trees_fit_a_nonlinear_target_better_than_a_bare_mean() {
        // Target is a step function - a single mean prediction is a poor
        // fit; boosting with enough trees should recover it closely.
        let rows: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = (0..50).map(|i| if i < 25 { 0.0 } else { 50.0 }).collect();
        let mean = targets.iter().sum::<f64>() / targets.len() as f64;

        let model = GradientBoostedTrees::fit(
            &rows,
            &targets,
            &GbmParams { n_trees: 50, learning_rate: 0.3, tree: TreeParams { max_depth: 2, min_samples_leaf: 1 } },
        );

        let mean_only_sse: f64 = targets.iter().map(|&t| (t - mean).powi(2)).sum();
        let model_sse: f64 = rows
            .iter()
            .zip(targets.iter())
            .map(|(r, &t)| (model.predict(r) - t).powi(2))
            .sum();

        assert!(
            model_sse < mean_only_sse * 0.1,
            "expected boosted model to fit the step function much better than a bare mean: model_sse={}, mean_only_sse={}",
            model_sse,
            mean_only_sse
        );
    }

    #[test]
    fn fitting_is_deterministic() {
        // No randomness anywhere in this implementation (unlike random
        // forest's bootstrap sampling / feature subsampling) - same input
        // should always produce bit-identical output.
        let rows: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let targets: Vec<f64> = (0..30).map(|i| (i as f64).sin() * 10.0).collect();
        let params = default_params(10);

        let model_a = GradientBoostedTrees::fit(&rows, &targets, &params);
        let model_b = GradientBoostedTrees::fit(&rows, &targets, &params);

        for row in &rows {
            assert_eq!(model_a.predict(row), model_b.predict(row));
        }
    }
}
