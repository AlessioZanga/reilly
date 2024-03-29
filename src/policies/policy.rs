use std::fmt::Debug;

use rand::Rng;

use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Definition of generic policy.
pub trait Policy: Clone + Debug + Default {
    /// Chooses the next action given a sequence of (action, expected_reward).
    fn call<A, R, S, V, T>(&self, f: &V, state: &S, rng: &mut T) -> A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
        T: Rng + ?Sized;

    /// Resets the function.
    fn reset(&mut self);
}
