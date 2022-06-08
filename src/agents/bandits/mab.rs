use std::{
    fmt::{Display, Formatter},
    marker::PhantomData,
};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    agents::Agent,
    policies::Policy,
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// (Contextual) multi armed bandit agent (MAB).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiArmedBandit<A, R, S, P, V>
where
    A: Action,
    R: Reward,
    S: State,
    P: Policy,
    V: StateActionValue<A, R, S>,
{
    #[serde(default, skip_serializing)]
    _a: PhantomData<A>,
    #[serde(default, skip_serializing)]
    _r: PhantomData<R>,
    #[serde(default, skip_serializing)]
    _s: PhantomData<S>,
    pi: P,
    v: V,
}

impl<A, R, S, P, V> Display for MultiArmedBandit<A, R, S, P, V>
where
    A: Action,
    R: Reward,
    S: State,
    P: Policy,
    V: StateActionValue<A, R, S>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}-MAB", self.pi, self.v)
    }
}

impl<A, R, S, P, V> Agent<A, R, S, P, V> for MultiArmedBandit<A, R, S, P, V>
where
    A: Action,
    R: Reward,
    S: State,
    P: Policy,
    V: StateActionValue<A, R, S>,
{
    fn new(pi: P, v: V) -> Self
    where
        P: Policy,
        V: StateActionValue<A, R, S>,
    {
        Self {
            _a: PhantomData,
            _r: PhantomData,
            _s: PhantomData,
            pi,
            v,
        }
    }

    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = A> + 'a> {
        self.v.actions_iter()
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = S> + 'a> {
        self.v.states_iter()
    }

    fn call<T>(&self, state: S, rng: &mut T) -> A
    where
        T: Rng + ?Sized,
    {
        // Evaluate the value function for each action.
        self.pi.call(&self.v, state, rng)
    }

    fn reset(&mut self) -> &mut Self {
        // Reset the (state-)action value function.
        self.v.reset();
        // Reset the policy.
        self.pi.reset();

        self
    }

    fn update(&mut self, action: A, reward: R, state: S, is_done: bool) {
        // Update the (state-)action value function.
        self.v.update(action, reward, state, is_done);
        // Update the policy.
        self.pi.update(is_done);
    }
}
