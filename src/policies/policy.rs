use std::fmt::{Debug, Display};

use rand::Rng;

use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Definition of generic policy.
pub trait Policy: Clone + Debug + Display {
    /// Chooses the next action given the (state-)value function and the current state.
    fn call<A, R, S, V, T>(&self, f: &V, state: S, rng: &mut T) -> A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
        T: Rng + ?Sized;

    /// Updates the policy given performed action, obtained reward, next state and end-of-episode flag.
    fn update(&mut self, is_done: bool);
}
