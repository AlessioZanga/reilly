use rand::Rng;
use serde::{Deserialize, Serialize};

use super::Policy;
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Greedy policy.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Greedy {}

impl Greedy {
    /// Constructs a greedy policy.
    pub fn new() -> Self {
        Self {}
    }
}

impl Policy for Greedy {
    fn call<A, R, S, V, T>(&self, f: &V, state: &S, rng: &mut T) -> A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
        T: Rng + ?Sized,
    {
        // For each action ...
        f.actions_iter()
            // ... evaluate the value function, then ...
            .map(|a| (a, f.call(a, state, rng)))
            // ... for each (action, reward) pair ...
            .reduce(|(a_i, r_i), (a_j, r_j)|
            // ... maximize the expected reward ...
            match r_i < r_j {
                false => (a_i, r_i),
                true => (a_j, r_j),
            })
            // ... and get the associated action ...
            .map(|(a, _)| a)
            // ... or panic if sequence is empty.
            .expect("Unable to choose an action")
            .clone()
    }

    fn reset(&mut self) {}

    fn update<A, R, S>(&mut self, _action: &A, _reward: &R, _state: &S, _is_done: bool)
    where
        A: Action,
        R: Reward,
        S: State,
    {
    }
}
