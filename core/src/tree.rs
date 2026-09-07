//! A CART-style regression tree, built from scratch.
//!
//! This is the building block for Phase 4's gradient-boosted trees. It
//! lives in `core` (not a separate crate like `models`) because it needs
//! nothing beyond `std` - the whole point of building it from scratch is
//! that it doesn't need `smartcore` or any other tree library.
//!
//! Operates on plain `&[Vec<f64>]` rows / `&[f64]` targets rather than
//! anything from an array/ndarray crate, matching what `design_matrix`
//! already produces - no conversion needed to hand data from the design
//! matrix straight to this.

/// A trained regression tree: either a leaf (a constant prediction) or a
/// split on one feature against a threshold, with `<= threshold` routed
/// left and `> threshold` routed right.
#[derive(Debug, Clone)]
pub enum TreeNode {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn predict(&self, row: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split { feature, threshold, left, right } => {
                if row[*feature] <= *threshold {
                    left.predict(row)
                } else {
                    right.predict(row)
                }
            }
        }
    }

    /// Depth of the tree (a lone leaf has depth 0), mainly for tests and
    /// diagnostics rather than anything the model needs at runtime.
    pub fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 0,
            TreeNode::Split { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TreeParams {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
}

/// Fits a regression tree to minimize sum of squared error.
///
/// Splitting is the standard CART approach: at each node, for every
/// feature, sort the node's samples by that feature's value and scan
/// candidate split points, tracking running left-side sum/sum-of-squares
/// so each candidate's SSE reduction is O(1) to evaluate rather than
/// recomputed from scratch (an O(n)-per-candidate approach would make the
/// whole search O(n^2) per feature per node, versus O(n log n) for the
/// sort-then-scan approach used here).
pub fn fit_tree(rows: &[Vec<f64>], targets: &[f64], params: &TreeParams) -> TreeNode {
    assert_eq!(rows.len(), targets.len(), "one target required per row");
    assert!(!rows.is_empty(), "cannot fit a tree on zero rows");
    let indices: Vec<usize> = (0..rows.len()).collect();
    build_node(rows, targets, &indices, 0, params)
}

fn build_node(
    rows: &[Vec<f64>],
    targets: &[f64],
    indices: &[usize],
    depth: usize,
    params: &TreeParams,
) -> TreeNode {
    let n = indices.len();
    let mean = indices.iter().map(|&i| targets[i]).sum::<f64>() / n as f64;

    if depth >= params.max_depth || n < 2 * params.min_samples_leaf {
        return TreeNode::Leaf { value: mean };
    }

    match find_best_split(rows, targets, indices, params.min_samples_leaf) {
        None => TreeNode::Leaf { value: mean },
        Some((feature, threshold, left_indices, right_indices)) => {
            let left = build_node(rows, targets, &left_indices, depth + 1, params);
            let right = build_node(rows, targets, &right_indices, depth + 1, params);
            TreeNode::Split {
                feature,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
    }
}

/// Searches every feature for the split with the largest SSE reduction.
/// Returns `None` if no split respects `min_samples_leaf` on both sides or
/// every candidate split has zero or negative reduction (e.g. all targets
/// in this node are already identical).
///
/// Performance note: this sorts a flat `Vec<(f64, f64, usize)>` of
/// (feature value, target, original index) rather than sorting `indices`
/// through a comparator that reaches back into `rows`/`targets` via double
/// indirection (`rows[a][feature]`). The earlier version was correct but
/// significantly slower in practice — sorting through indirection defeats
/// cache locality on every single comparison, and profiling against real
/// FD002 data (fitting 100 trees took ~70s for FD001 and was on track for
/// 400+s for FD002's larger training set) made clear this was a real
/// scaling problem, not just a constant-factor nicety.
#[allow(clippy::type_complexity)]
fn find_best_split(
    rows: &[Vec<f64>],
    targets: &[f64],
    indices: &[usize],
    min_samples_leaf: usize,
) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
    let n = indices.len();
    let n_features = rows[0].len();

    let total_sum: f64 = indices.iter().map(|&i| targets[i]).sum();
    let total_sq: f64 = indices.iter().map(|&i| targets[i] * targets[i]).sum();
    let sse_before = total_sq - total_sum * total_sum / n as f64;

    let mut best: Option<(f64, usize, f64)> = None; // (reduction, feature, threshold)

    // Reused across features to avoid reallocating every iteration.
    let mut sorted: Vec<(f64, f64, usize)> = Vec::with_capacity(n);

    for feature in 0..n_features {
        sorted.clear();
        sorted.extend(indices.iter().map(|&i| (rows[i][feature], targets[i], i)));
        sorted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut left_sum = 0.0;
        let mut left_sq = 0.0;

        for pos in 0..(n - 1) {
            let (value, target, _idx) = sorted[pos];
            left_sum += target;
            left_sq += target * target;
            let left_n = pos + 1;
            let right_n = n - left_n;

            // Can't split between two samples with an identical feature
            // value - they'd end up on different sides of a threshold that
            // doesn't actually separate them.
            if value == sorted[pos + 1].0 {
                continue;
            }
            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }

            let right_sum = total_sum - left_sum;
            let right_sq = total_sq - left_sq;
            let sse_left = left_sq - left_sum * left_sum / left_n as f64;
            let sse_right = right_sq - right_sum * right_sum / right_n as f64;
            let reduction = sse_before - (sse_left + sse_right);

            let better = match best {
                None => true,
                Some((best_reduction, _, _)) => reduction > best_reduction,
            };
            if better {
                let threshold = (value + sorted[pos + 1].0) / 2.0;
                best = Some((reduction, feature, threshold));
            }
        }
    }

    let (reduction, feature, threshold) = best?;
    if reduction <= 1e-12 {
        return None;
    }

    let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
        indices.iter().partition(|&&i| rows[i][feature] <= threshold);

    if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
        None
    } else {
        Some((feature, threshold, left_indices, right_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_leaf_predicts_the_mean_when_max_depth_is_zero() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![10.0, 20.0, 30.0];
        let params = TreeParams { max_depth: 0, min_samples_leaf: 1 };
        let tree = fit_tree(&rows, &targets, &params);
        assert!(matches!(tree, TreeNode::Leaf { .. }));
        assert_eq!(tree.predict(&[1.0]), 20.0); // mean of 10,20,30
    }

    #[test]
    fn finds_an_obvious_single_feature_split() {
        // y is a clean step function of x: 0 for x<5, 100 for x>=5.
        let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 100.0 }).collect();
        let params = TreeParams { max_depth: 3, min_samples_leaf: 1 };
        let tree = fit_tree(&rows, &targets, &params);

        assert!((tree.predict(&[2.0]) - 0.0).abs() < 1e-6);
        assert!((tree.predict(&[17.0]) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn respects_max_depth() {
        let rows: Vec<Vec<f64>> = (0..64).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = (0..64).map(|i| i as f64).collect(); // pure linear signal
        let params = TreeParams { max_depth: 2, min_samples_leaf: 1 };
        let tree = fit_tree(&rows, &targets, &params);
        assert!(tree.depth() <= 2);
    }

    #[test]
    fn respects_min_samples_leaf() {
        let rows: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let params = TreeParams { max_depth: 10, min_samples_leaf: 4 };
        let tree = fit_tree(&rows, &targets, &params);

        fn check_leaf_sizes(node: &TreeNode, rows: &[Vec<f64>], indices: &[usize], min_leaf: usize) {
            match node {
                TreeNode::Leaf { .. } => {
                    assert!(
                        indices.len() >= min_leaf,
                        "leaf has {} samples, expected >= {}",
                        indices.len(),
                        min_leaf
                    );
                }
                TreeNode::Split { feature, threshold, left, right } => {
                    let (l, r): (Vec<usize>, Vec<usize>) =
                        indices.iter().partition(|&&i| rows[i][*feature] <= *threshold);
                    check_leaf_sizes(left, rows, &l, min_leaf);
                    check_leaf_sizes(right, rows, &r, min_leaf);
                }
            }
        }
        let all_indices: Vec<usize> = (0..rows.len()).collect();
        check_leaf_sizes(&tree, &rows, &all_indices, 4);
    }

    #[test]
    fn duplicate_feature_values_dont_panic_or_infinite_loop() {
        let rows = vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]];
        let targets = vec![5.0, 5.0, 5.0, 5.0];
        let params = TreeParams { max_depth: 5, min_samples_leaf: 1 };
        let tree = fit_tree(&rows, &targets, &params);
        assert_eq!(tree.predict(&[1.0]), 5.0);
    }

    #[test]
    fn does_not_split_when_all_targets_are_identical() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![7.0, 7.0, 7.0, 7.0];
        let params = TreeParams { max_depth: 5, min_samples_leaf: 1 };
        let tree = fit_tree(&rows, &targets, &params);
        // No reduction possible - should collapse to a single leaf.
        assert!(matches!(tree, TreeNode::Leaf { .. }));
    }
}
