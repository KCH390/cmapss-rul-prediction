//! K-means clustering, built from scratch. Used by `regime.rs` to identify
//! the discrete operating conditions in FD002/FD004 from the 3 operational
//! settings, so sensor readings can be normalized per-regime before
//! feeding into the existing feature/model pipeline unchanged.
//!
//! Zero dependencies - including no `rand` crate. Centroid initialization
//! uses a hand-rolled SplitMix64 PRNG seeded with a fixed constant, so
//! fitting is exactly reproducible run to run, matching the determinism
//! this project already guarantees for the flagship gradient boosting
//! model (see `boosting.rs`).

/// A minimal, non-cryptographic PRNG (SplitMix64) used only to make
/// k-means++ initialization deterministic without depending on `rand`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug, Clone)]
pub struct KMeans {
    pub centroids: Vec<Vec<f64>>,
}

impl KMeans {
    /// Index of the nearest centroid to `point`.
    pub fn predict(&self, point: &[f64]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, squared_distance(c, point)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .expect("KMeans must have at least one centroid")
    }

    pub fn k(&self) -> usize {
        self.centroids.len()
    }
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Fits k-means via k-means++ initialization followed by Lloyd's algorithm.
///
/// k-means++ (rather than plain random initialization) picks the first
/// centroid uniformly at random, then each subsequent centroid with
/// probability proportional to its squared distance from the nearest
/// already-chosen centroid — this makes it much less likely to land on a
/// bad initialization (e.g. two initial centroids landing in the same true
/// cluster, starving another), which matters here because empty clusters
/// during Lloyd's algorithm are otherwise a real failure mode.
pub fn fit(data: &[Vec<f64>], k: usize, max_iterations: usize, seed: u64) -> KMeans {
    assert!(!data.is_empty(), "cannot fit k-means on zero points");
    assert!(k >= 1, "k must be at least 1");
    assert!(k <= data.len(), "k cannot exceed the number of points");

    let mut rng = SplitMix64::new(seed);
    let dims = data[0].len();

    // --- k-means++ initialization ---
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first_idx = ((rng.next_f64() * data.len() as f64) as usize).min(data.len() - 1);
    centroids.push(data[first_idx].clone());

    while centroids.len() < k {
        let distances: Vec<f64> = data
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| squared_distance(c, p))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = distances.iter().sum();

        if total <= 0.0 {
            // Every remaining point is a duplicate of an existing centroid;
            // fall back to picking any point rather than dividing by zero.
            centroids.push(data[centroids.len() % data.len()].clone());
            continue;
        }

        let mut target = rng.next_f64() * total;
        let mut chosen = data.len() - 1;
        for (i, &d) in distances.iter().enumerate() {
            if target <= d {
                chosen = i;
                break;
            }
            target -= d;
        }
        centroids.push(data[chosen].clone());
    }

    // --- Lloyd's algorithm ---
    let mut assignments = vec![usize::MAX; data.len()]; // MAX guarantees the first pass reports "changed"
    for _ in 0..max_iterations {
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let best = centroids
                .iter()
                .enumerate()
                .map(|(c_idx, c)| (c_idx, squared_distance(c, point)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            if best != assignments[i] {
                changed = true;
                assignments[i] = best;
            }
        }

        let mut sums = vec![vec![0.0; dims]; k];
        let mut counts = vec![0usize; k];
        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dims {
                sums[c][d] += point[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dims {
                    centroids[c][d] = sums[c][d] / counts[c] as f64;
                }
            }
            // An empty cluster (counts[c] == 0) keeps its previous centroid
            // rather than becoming NaN - rare with k-means++ init, but a
            // real possibility worth not crashing on.
        }

        if !changed {
            break;
        }
    }

    KMeans { centroids }
}

/// Cluster sizes for a fitted model against the data it was (or wasn't)
/// fitted on - used for reporting/validation, not by `fit` itself.
pub fn cluster_sizes(model: &KMeans, data: &[Vec<f64>]) -> Vec<usize> {
    let mut sizes = vec![0usize; model.k()];
    for point in data {
        sizes[model.predict(point)] += 1;
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn well_separated_clusters() -> Vec<Vec<f64>> {
        // Three tight, obviously-separated 2D blobs.
        let mut data = Vec::new();
        for base in [[0.0, 0.0], [50.0, 50.0], [-50.0, 50.0]] {
            for i in 0..20 {
                let jitter = (i as f64 - 10.0) * 0.05; // stays well within each blob
                data.push(vec![base[0] + jitter, base[1] + jitter]);
            }
        }
        data
    }

    #[test]
    fn finds_obviously_separated_clusters() {
        let data = well_separated_clusters();
        let model = fit(&data, 3, 100, 42);

        // Every point in a blob should share the same cluster assignment
        // as every other point in that same blob.
        let assignment_of = |idx: usize| model.predict(&data[idx]);
        assert_eq!(assignment_of(0), assignment_of(19)); // blob 1
        assert_eq!(assignment_of(20), assignment_of(39)); // blob 2
        assert_eq!(assignment_of(40), assignment_of(59)); // blob 3
        // And the three blobs should NOT all collapse into the same cluster.
        let a = assignment_of(0);
        let b = assignment_of(20);
        let c = assignment_of(40);
        assert!(a != b && b != c && a != c);
    }

    #[test]
    fn cluster_sizes_are_roughly_balanced_for_balanced_input() {
        let data = well_separated_clusters();
        let model = fit(&data, 3, 100, 42);
        let sizes = cluster_sizes(&model, &data);
        assert_eq!(sizes.iter().sum::<usize>(), data.len());
        for &size in &sizes {
            assert_eq!(size, 20, "each of the three equal-sized blobs should map to its own cluster");
        }
    }

    #[test]
    fn fitting_is_deterministic() {
        let data = well_separated_clusters();
        let model_a = fit(&data, 3, 100, 42);
        let model_b = fit(&data, 3, 100, 42);
        assert_eq!(model_a.centroids, model_b.centroids);
    }

    #[test]
    fn k_equals_one_puts_everything_in_a_single_cluster() {
        let data = well_separated_clusters();
        let model = fit(&data, 1, 100, 42);
        assert_eq!(model.k(), 1);
        for point in &data {
            assert_eq!(model.predict(point), 0);
        }
    }

    #[test]
    fn handles_duplicate_points_without_panicking() {
        let data = vec![vec![1.0, 1.0]; 10];
        let model = fit(&data, 2, 50, 7);
        // Shouldn't panic, and every point should still get a valid assignment.
        for point in &data {
            let assignment = model.predict(point);
            assert!(assignment < model.k());
        }
    }
}
