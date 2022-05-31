use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use super::{Greedy, Policy, Random};
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Epsilon-greedy policy with decay factor and minimum epsilon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpsilonDecayGreedy {
    epsilon_decay: f64,
    epsilon_min: f64,
    epsilon_0: f64,
    epsilon: f64,
    greedy: Greedy,
    random: Random,
}

impl EpsilonDecayGreedy {
    /// Constructs an epsilon-decay greedy policy.
    pub fn new(epsilon: f64, epsilon_decay: f64, epsilon_min: f64) -> Self {
        // Check if epsilon is in [0, 1) range.
        assert!((0. ..1.).contains(&epsilon), "Epsilon must be in [0, 1) range");
        // Check if decay factor is in [0, 1) range.
        assert!(
            (0. ..1.).contains(&epsilon_decay),
            "Decay factor must be in [0, 1) range"
        );
        // Check if minimum value of epsilon is in [0, 1) range and less then initial value.
        assert!(
            (0. ..epsilon).contains(&epsilon_min),
            "Minimum epsilon value must be in [0, 1) range and less then initial value"
        );

        Self {
            epsilon_decay,
            epsilon_min,
            epsilon_0: epsilon,
            epsilon,
            greedy: Default::default(),
            random: Default::default(),
        }
    }
}

impl Default for EpsilonDecayGreedy {
    fn default() -> Self {
        Self::new(0.1, 0.999, 0.01)
    }
}

impl Policy for EpsilonDecayGreedy {
    fn call<A, R, S, V, T>(&self, f: &V, state: &S, rng: &mut T) -> A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
        T: rand::Rng + ?Sized,
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

    fn update<A, R, S>(&mut self, _action: &A, _reward: &R, _state: &S, is_done: bool)
    where
        A: Action,
        R: Reward,
        S: State,
    {
        // If the episode has ended and epsilon has not reached the lower bound ...
        if is_done && self.epsilon > self.epsilon_min {
            // ... set epsilon to the maximum value between the lower bound and the updated epsilon.
            self.epsilon = f64::max(self.epsilon_min, self.epsilon * self.epsilon_decay);
        }
    }
}
