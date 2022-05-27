use std::fmt::Debug;

use rand::prelude::*;

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
    init_epsilon: f64,
    epsilon: f64,
    greedy: Greedy,
    random: Random<T>,
}

impl<T> EpsilonGreedy<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    /// Constructs a random policy given an optional seed.
    pub fn new(epsilon: f64, seed: Option<u64>) -> Self {
        // Initialize helper policies.
        let greedy = Greedy::default();
        let random = Random::new(seed);

        Self {
            init_epsilon: epsilon,
            epsilon,
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
        Self {
            init_epsilon: 0.1,
            epsilon: 0.1,
            greedy: Default::default(),
            random: Default::default(),
        }
    }
}

impl<T> Policy for EpsilonGreedy<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    fn call<'a, A, R, S, V>(&mut self, f: &'a V, state: &S) -> &'a A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
    {
        // FIXME: With probability (1 - epsilon) ...
        match true {
            // ... select an action greedily, otherwise ...
            false => self.greedy.call(f, state),
            // ... select a random action form the action space.
            true => self.random.call(f, state),
        }
    }

    fn reset(&mut self) {
        // Reset epsilon.
        self.epsilon = self.init_epsilon;
        // Reset helpers.
        self.greedy.reset();
        self.random.reset();
    }
}
