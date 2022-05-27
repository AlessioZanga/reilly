use super::Policy;
use crate::{
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Greedy policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Greedy {}

impl Policy for Greedy {
    fn call<'a, A, R, S, V>(&mut self, f: &'a V, state: &S) -> &'a A
    where
        A: Action,
        R: Reward,
        S: State,
        V: StateActionValue<A, R, S>,
    {
        // For each action ...
        f.actions_iter()
            // ... evaluate the value function, then ...
            .map(|a| (a, f.call(a, state)))
            // ... for each (action, reward) pair ...
            .reduce(|(a_i, r_i), (a_j, r_j)|
            // ... maximize the expected reward ...
            match r_i < r_j {
                false => (a_i, r_i),
                true => (a_j, r_j),
            })
            // ... and get the associated action ...
            .map(|(a, _)| a)
            // ... or panic if sequence is empty.
            .expect("Unable to choose an action")
    }

    fn reset(&mut self) {}
}
