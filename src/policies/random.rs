use std::fmt::Debug;

use rand::{prelude::IteratorRandom, Rng};

use super::Policy;
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Random policy.
#[derive(Clone, Debug, Default)]
pub struct Random {}

impl Random {}

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
        f.actions_iter()
            .choose(rng)
            .expect("Unable to choose an action")
            .clone()
    }

    fn reset(&mut self) {}
}
