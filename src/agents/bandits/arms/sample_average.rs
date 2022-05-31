use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use super::Arm;

/// Sample average bandit arm.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SampleAverage {
    count: usize,
    mean: f64,
}

impl SampleAverage {
    /// Constructs a new sample-average bandit arm.
    pub fn new() -> Self {
        Self { count: 0, mean: 0. }
    }
}

impl Arm<f64> for SampleAverage {
    fn get_count(&self) -> usize {
        self.count
    }

    fn call(&self) -> f64 {
        self.mean
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.;
    }

    fn sample<T: rand::Rng + ?Sized>(&self, rng: &mut T) -> f64 {
        Normal::new(self.mean, 1.)
            .expect("Unable to construct sampling distribution")
            .sample(rng)
    }

    fn update(&mut self, reward: &f64) {
        // Increase counter.
        self.count += 1;
        // Update as Q(a) := Q(a) + 1 / N(a) * [R - Q(a)].
        self.mean += (1. / self.count as f64) * (reward - self.mean);
    }
}
