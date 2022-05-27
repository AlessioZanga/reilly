use std::fmt::Debug;

use rand::prelude::*;
use rand_distr::Uniform;

use super::{Greedy, Policy, Random};
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Epsilon-greedy policy.
#[derive(Clone, Debug)]
pub struct EpsilonGreedy<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    epsilon_0: f64,
    epsilon: f64,
    seed: Option<u64>,
    rng: T,
    // Helper policies.
    greedy: Greedy,
    random: Random<T>,
}

impl<T> EpsilonGreedy<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    /// Constructs a random policy given an optional seed.
    pub fn new(epsilon: f64, seed: Option<u64>) -> Self {
        // Seed the random number generator.
        let rng = match seed {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };
        // Initialize helper policies.
        let greedy = Greedy::default();
        let random = Random::new(seed);

        Self {
            epsilon_0: epsilon,
            epsilon,
            seed,
            rng,
            // Helper policies.
            greedy,
            random,
        }
    }
}

impl<T> Default for EpsilonGreedy<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    fn default() -> Self {
        // Seed the random number generator from entropy.
        let rng = SeedableRng::from_entropy();

        Self {
            epsilon_0: 0.1,
            epsilon: 0.1,
            seed: None,
            rng,
            // Default helper policies.
            greedy: Default::default(),
            random: Default::default(),
        }
    }
}

impl<T> Policy for EpsilonGreedy<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    fn call_mut<'a, A, R, S, V>(&mut self, f: &'a V, state: &S) -> &'a A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
    {
        // Sample probability.
        let p = Uniform::new(0., 1.).sample(&mut self.rng);
        // With probability (1 - epsilon) ...
        match p < (1. - self.epsilon) {
            // ... select an action greedily, otherwise ...
            false => self.greedy.call_mut(f, state),
            // ... select a random action form the action space.
            true => self.random.call_mut(f, state),
        }
    }

    fn reset(&mut self) {
        // Reset epsilon.
        self.epsilon = self.epsilon_0;
        // Re-seed the random number generator.
        self.rng = match self.seed {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };
        // Reset helper policies.
        self.greedy.reset();
        self.random.reset();
    }
}
