use std::fmt::Debug;

use rand::Rng;

use crate::types::{Action, Reward, State};

/// Definition of an environment.
pub trait Env<A, R, S>: Clone + Debug
where
    A: Action,
    R: Reward,
    S: State,
{
    /// Iterates of the action space.
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a A> + 'a>;

    /// Iterates of the state space.
    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a S> + 'a>;

    /// Gets current state of the environment.
    fn get_state(&self) -> S;

    /// Compute the effect of an action on the environment,
    /// returning the obtained reward, next state and end-of-episode flag.
    fn call_mut<T>(&mut self, action: &A, rng: &mut T) -> (R, S, bool)
    where
        T: Rng + ?Sized;

    /// Resets the environment state.
    fn reset(&mut self) -> &mut Self;
}
