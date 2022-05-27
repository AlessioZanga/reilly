use rand::Rng;

use super::Policy;
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Greedy policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Greedy {}

impl Policy for Greedy {
    fn call<A, R, S, V, T>(&self, f: &V, state: &S, _rng: &mut T) -> A
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
            .map(|a| (a, f.call(a, state)))
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
}
