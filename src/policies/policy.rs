use std::fmt::Debug;

use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Definition of generic policy.
pub trait Policy: Clone + Debug + Default {
    /// Chooses the next action given a sequence of (action, expected_reward).
    fn call_mut<A, R, S, V>(&mut self, f: &V, state: &S) -> A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>;

    /// Resets the function.
    fn reset(&mut self);
}
