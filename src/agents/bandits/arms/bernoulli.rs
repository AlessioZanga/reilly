use rand::prelude::*;
use rand_distr::Beta;
use serde::{Deserialize, Serialize};

use super::Arm;

/// Bernoulli bandit arm.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bernoulli {
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
            alpha_0: alpha,
            beta_0: beta,
            alpha,
            beta,
        }
    }
}

impl Arm<f64> for Bernoulli {
    fn call(&self) -> f64 {
        // Compute the expected reward.
        self.alpha / (self.alpha + self.beta)
    }

    fn reset(&mut self) {
        self.alpha = self.alpha_0;
        self.beta = self.beta_0;
    }

    fn sample<T: Rng + ?Sized>(&self, rng: &mut T) -> f64 {
        // Sample from given distribution.
        Beta::new(self.alpha, self.beta)
            .expect("Unable to construct Beta distribution for given parameters")
            .sample(rng)
    }

    #[allow(unused_parens)]
    fn update(&mut self, reward: &f64) {
        // Update distributions parameter.
        self.alpha += reward;
        self.beta += (1. - reward);
    }
}

impl Default for Bernoulli {
    fn default() -> Self {
        Self {
            alpha_0: 1.,
            beta_0: 1.,
            alpha: 1.,
            beta: 1.,
        }
    }
}
