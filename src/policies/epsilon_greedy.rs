use std::fmt::{Debug, Display, Formatter};

use rand::prelude::*;
use rand_distr::Uniform;
use serde::{Deserialize, Serialize};

use super::{Greedy, Policy, Random};
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Epsilon-greedy policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpsilonGreedy {
    epsilon_0: f64,
    epsilon: f64,
    greedy: Greedy,
    random: Random,
}

impl EpsilonGreedy {
    /// Constructs an epsilon-greedy policy.
    pub fn new(epsilon: f64) -> Self {
        // Check if epsilon is in [0, 1) range.
        assert!((0. ..1.).contains(&epsilon), "Epsilon must be in [0, 1) range");

        Self {
            epsilon_0: epsilon,
            epsilon,
            greedy: Greedy::new(),
            random: Random::new(),
        }
    }
}

impl Display for EpsilonGreedy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "EpsilonGreedy(ϵ = {})", self.epsilon_0)
    }
}

impl Policy for EpsilonGreedy {
    fn call<A, R, S, V, T>(&self, f: &V, state: S, rng: &mut T) -> A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
        T: Rng + ?Sized,
    {
        // Sample probability.
        let p = Uniform::new(0., 1.).sample(rng);
        // With probability (1 - epsilon) ...
        match p < (1. - self.epsilon) {
            // ... select an action greedily, otherwise ...
            false => self.greedy.call(f, state, rng),
            // ... select a random action form the action space.
            true => self.random.call(f, state, rng),
        }
    }

    fn reset(&mut self) {
        // Reset epsilon.
        self.epsilon = self.epsilon_0;
        // Reset helper policies.
        self.greedy.reset();
        self.random.reset();
    }

    fn update(&mut self, _is_done: bool) {}
}
