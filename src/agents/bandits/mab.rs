use std::{collections::HashMap, marker::PhantomData};

use rand::Rng;

use super::arms::Arm;
use crate::{
    agents::Agent,
    policies::Policy,
    types::{Action, Reward, State},
    values::{ActionValue, StateActionValue},
};

/// Action value function of a MAB.
pub struct Arms<A, R, V>
where
    A: Action,
    R: Reward,
    V: Arm<R>,
{
    _r_marker: PhantomData<R>,
    arms: HashMap<A, V>,
}

impl<A, R, V> Arms<A, R, V>
where
    A: Action,
    R: Reward,
    V: Arm<R>,
{
    /// Constructs a sequence of arms given the action space.
    pub fn new<I>(actions_iter: I) -> Self
    where
        I: Iterator<Item = A>,
    {
        let arms = actions_iter.map(|a| (a, Default::default())).collect();

        Self {
            _r_marker: PhantomData,
            arms,
        }
    }

    /// Constructs a sequence of arms given the (action, arm) pairs.
    pub fn from_actions_arms_iter<I>(actions_arms_iter: I) -> Self
    where
        I: Iterator<Item = (A, V)>,
    {
        let arms = actions_arms_iter.collect();

        Self {
            _r_marker: PhantomData,
            arms,
        }
    }
}

impl<A, R, V> ActionValue<A, R> for Arms<A, R, V>
where
    A: Action,
    R: Reward,
    V: Arm<R>,
{
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a A> + 'a> {
        Box::new(self.arms.keys())
    }

    fn call(&self, action: &A) -> R {
        self.arms[action].call()
    }

    fn reset(&mut self) -> &mut Self {
        self.arms.iter_mut().for_each(|(_, arm)| arm.reset());

        self
    }

    fn update(&mut self, action: &A, reward: &R) {
        // Update the arm association with performed action given the obtained reward.
        self.arms
            .get_mut(action)
            .expect("Unable to get bandit's arm for given action")
            .update(reward)
    }
}

/// (Contextual) multi armed bandit agent (MAB).
pub struct MultiArmedBandit<A, R, S, P, V>
where
    A: Action,
    R: Reward,
    S: State,
    P: Policy,
    V: StateActionValue<A, R, S>,
{
    _a_marker: PhantomData<A>,
    _r_marker: PhantomData<R>,
    _s_marker: PhantomData<S>,
    pi: P,
    v: V,
}

impl<A, R, S, P, V> Agent<A, R, S, P, V> for MultiArmedBandit<A, R, S, P, V>
where
    A: Action,
    R: Reward,
    S: State,
    P: Policy,
    V: StateActionValue<A, R, S>,
{
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a A> + 'a> {
        self.v.actions_iter()
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a S> + 'a> {
        self.v.states_iter()
    }

    fn new(pi: P, v: V) -> Self
    where
        P: Policy,
        V: StateActionValue<A, R, S>,
    {
        Self {
            _a_marker: PhantomData,
            _r_marker: PhantomData,
            _s_marker: PhantomData,
            pi,
            v,
        }
    }

    fn call<T>(&self, state: &S, rng: &mut T) -> A
    where
        T: Rng + ?Sized,
    {
        // Evaluate the value function for each action.
        self.pi.call(&self.v, state, rng)
    }

    fn reset(&mut self) -> &mut Self {
        self.pi.reset();
        self.v.reset();

        self
    }

    fn update(&mut self, action: &A, reward: &R, state: &S, _is_done: bool) {
        // Update the (state-)action value function.
        self.v.update(action, reward, state)
    }
}
