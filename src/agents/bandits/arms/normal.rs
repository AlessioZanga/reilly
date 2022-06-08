use rand_distr::Distribution;
use serde::{Deserialize, Serialize};

use super::Arm;

/// Sample average bandit arm.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Normal {
    count: usize,
    srewd: f64,
    mean: f64,
}

impl Normal {
    /// Constructs a new sample-average bandit arm.
    pub fn new() -> Self {
        Self {
            count: 0,
            srewd: 0.,
            mean: 0.,
        }
    }
}

impl Arm<f64> for Normal {
    fn get_count(&self) -> usize {
        self.count
    }

    fn get_sum_squared_rewards(&self) -> f64 {
        self.srewd
    }

    fn call(&self) -> f64 {
        self.mean
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.;
    }

    fn sample<T: rand::Rng + ?Sized>(&self, rng: &mut T) -> f64 {
        rand_distr::Normal::new(self.mean, 1. / (self.count as f64 + 1.))
            .expect("Unable to construct sampling distribution")
            .sample(rng)
    }

    fn update(&mut self, reward: f64) {
        // Update the counter and sum of squared rewards.
        self.count += 1;
        self.srewd += f64::powi(reward, 2);
        // Update as Q(a) := Q(a) + 1 / N(a) * [R - Q(a)].
        self.mean += (1. / self.count as f64) * (reward - self.mean);
    }
}
