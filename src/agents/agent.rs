use crate::{
    policies::Policy,
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Definition of a generic agent.
pub trait Agent<A, R, S, P, V>
where
    A: Action,
    R: Reward,
    S: State,
    P: Policy,
    V: StateActionValue<A, R, S>,
{
    /// Constructs an agent given a policy and a (state-)action value function.
    fn new(pi: P, v: V) -> Self;

    /// Computes the action for given state.
    fn call_mut(&mut self, state: &S) -> &A;

    /// Resets the agent.
    fn reset(&mut self);

    /// Updates the agent given performed action, obtained reward, next state, and end-of-episode flag.
    fn update(&mut self, action: &A, reward: R, state: S, is_done: bool);
}
