use crate::{
    agents::Agent,
    envs::Env,
    policies::Policy,
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Definition of agent-environment experiment session.
pub trait Session {
    /// Execute the experiment session.
    fn call<A, R, S, P, V, T, E>(&self, agent: &mut T, environment: &mut E)
    where
        A: Action,
        R: Reward,
        S: State,
        P: Policy,
        V: StateActionValue<A, R, S>,
        T: Agent<A, R, S, P, V>,
        E: Env<A, R, S>;
}
