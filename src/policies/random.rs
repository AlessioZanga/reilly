use std::fmt::Debug;

use rand::{prelude::IteratorRandom, Rng, SeedableRng};

use super::Policy;
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Random policy.
#[derive(Clone, Debug)]
pub struct Random<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    seed: Option<u64>,
    rng: T,
}

impl<T> Random<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    /// Constructs a random policy given an optional seed.
    pub fn new(seed: Option<u64>) -> Self {
        // Seed the random number generator.
        let rng = match seed {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };

        Self { seed, rng }
    }
}

impl<T> Default for Random<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    fn default() -> Self {
        // Seed the random number generator from entropy.
        let rng = SeedableRng::from_entropy();

        Self { seed: None, rng }
    }
}

impl<T> Policy for Random<T>
where
    T: Clone + Debug + Rng + SeedableRng + ?Sized,
{
    fn call_mut<A, R, S, V>(&mut self, f: &V, _state: &S) -> A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
    {
        // Select a random action form the action space.
        f.actions_iter()
            .choose(&mut self.rng)
            .expect("Unable to choose an action")
            .clone()
    }

    fn reset(&mut self) {
        // Re-seed the random number generator.
        self.rng = match self.seed {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };
    }
}
