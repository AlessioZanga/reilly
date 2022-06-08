use std::fmt::{Debug, Display};

use rand::Rng;

use crate::{
    policies::Policy,
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Definition of a generic agent.
pub trait Agent<A, R, S, P, V>: Clone + Debug + Display
where
    A: Action,
    R: Reward,
    S: State,
    P: Policy,
    V: StateActionValue<A, R, S>,
{
    /// Constructs an agent given a policy and a (state-)action value function.
    fn new(pi: P, v: V) -> Self;

    /// Iterates of the action space.
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = A> + 'a>;

    /// Iterates of the state space.
    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = S> + 'a>;

    /// Computes the action for given state.
    fn call<T>(&self, state: S, rng: &mut T) -> A
    where
        T: Rng + ?Sized;

    /// Resets the agent.
    fn reset(&mut self) -> &mut Self;

    /// Updates the agent given performed action, obtained reward, next state and end-of-episode flag.
    fn update(&mut self, action: A, reward: R, state: S, is_done: bool);
}
