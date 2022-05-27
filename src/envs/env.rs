use crate::types::{Action, Reward, State};
use std::fmt::Debug;

/// Definition of an environment.
pub trait Env<A, R, S>: Clone + Debug + Default
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

    /// Compute the effect of an action on the enviroment,
    /// returning the obtained reward, next state and end-of-episode flag.
    fn call_mut(&mut self, action: A) -> (R, S, bool);

    /// Resets the environment state.
    fn reset(&mut self);
}
