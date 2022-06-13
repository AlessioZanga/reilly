use rand::prelude::*;
use rand_distr::Beta;
use serde::{Deserialize, Serialize};

use super::Arm;

/// Bernoulli bandit arm.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bernoulli {
    count: usize,
    srewd: f64,
    alpha_0: f64,
    beta_0: f64,
    alpha: f64,
    beta: f64,
}

impl Bernoulli {
    /// Constructs a Bernoulli bandit arm.
    pub fn new(alpha: f64, beta: f64) -> Self {
        // FIXME: Sanitize inputs.

        Self {
            count: 0,
            srewd: 0.,
            alpha_0: alpha,
            beta_0: beta,
            alpha,
            beta,
        }
    }
}

impl Arm<f64> for Bernoulli {
    fn count(&self) -> usize {
        self.count
    }

    fn sum_squared_rewards(&self) -> f64 {
        self.srewd
    }

    fn call(&self) -> f64 {
        // Compute the expected reward.
        self.alpha / (self.alpha + self.beta)
    }

    fn sample<T: Rng + ?Sized>(&self, rng: &mut T) -> f64 {
        // Sample from given distribution.
        Beta::new(self.alpha, self.beta)
            .expect("Unable to construct Beta distribution for given parameters")
            .sample(rng)
    }

    #[allow(unused_parens)]
    fn update(&mut self, reward: f64) {
        // Update the counter and sum of squared rewards.
        self.count += 1;
        self.srewd += f64::powi(reward, 2);
        // Update distributions parameter.
        self.alpha += reward;
        self.beta += (1. - reward);
    }
}
