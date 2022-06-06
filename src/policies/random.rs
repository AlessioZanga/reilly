use std::fmt::{Debug, Display, Formatter};

use rand::{prelude::IteratorRandom, Rng};
use serde::{Deserialize, Serialize};

use super::Policy;
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Random policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Random {}

impl Random {
    /// Constructs a random policy.
    pub fn new() -> Self {
        Self {}
    }
}

impl Display for Random {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Random")
    }
}

impl Policy for Random {
    fn call<A, R, S, V, T>(&self, f: &V, _state: &S, rng: &mut T) -> A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
        T: Rng + ?Sized,
    {
        // Select a random action form the action space.
        f.actions_iter().choose(rng).expect("Unable to choose an action")
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
