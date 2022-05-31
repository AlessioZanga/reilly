use std::{collections::HashMap, marker::PhantomData};

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::arms::Arm;
use crate::{
    agents::Agent,
    policies::Policy,
    types::{Action, Reward, State},
    values::{ActionValue, StateActionValue},
};

/// Arms algortithm pseudo-enumerator.
pub struct ArmsAlgorithm {}

impl ArmsAlgorithm {
    /// Choose an arm w.r.t. the maximum expected value.
    pub const EXPECTED_VALUE: usize = 0;
    /// Choose an arm w.r.t. the maximum sampled value.
    pub const THOMPSON_SAMPLING: usize = 1;
    /// Choose an arm w.r.t. the maximum expected value plus the upper confidence bound.
    pub const UCB1: usize = 2;
}

/// Action value function of a MAB.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Arms<A, R, V, const M: usize>
where
    A: Action,
    R: Reward,
    V: Arm<R>,
{
    #[serde(default, skip_serializing)]
    _r: PhantomData<R>,
    arms: HashMap<A, V>,
}

impl<A, R, V, const M: usize> Arms<A, R, V, M>
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

        Self { _r: PhantomData, arms }
    }

    /// Constructs a sequence of arms given the (action, arm) pairs.
    pub fn from_actions_arms_iter<I>(actions_arms_iter: I) -> Self
    where
        I: Iterator<Item = (A, V)>,
    {
        let arms = actions_arms_iter.collect();

        Self { _r: PhantomData, arms }
    }
}

impl<A, R, V, const M: usize> ActionValue<A, R> for Arms<A, R, V, M>
where
    A: Action,
    R: Reward,
    V: Arm<R>,
{
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a A> + 'a> {
        Box::new(self.arms.keys())
    }

    fn call<T>(&self, action: &A, rng: &mut T) -> R
    where
        T: Rng + ?Sized,
    {
        match M {
            // Compute the expected value.
            ArmsAlgorithm::EXPECTED_VALUE => self.arms[action].call(),
            // Sample from the distribution.
            ArmsAlgorithm::THOMPSON_SAMPLING => self.arms[action].sample(rng),
            // Invalid algorithm enumerator.
            _ => unreachable!(),
        }
    }

    fn reset(&mut self) -> &mut Self {
        self.arms.iter_mut().for_each(|(_, arm)| arm.reset());

        self
    }

    fn update(&mut self, action: &A, reward: &R, _is_done: bool) {
        // Update the arm association with performed action given the obtained reward.
        self.arms
            .get_mut(action)
            .expect("Unable to get bandit's arm for given action")
            .update(reward)
    }
}

/// Arms alias following the expected value algorithm.
pub type ExpectedValueArms<A, R, V> = Arms<A, R, V, { ArmsAlgorithm::EXPECTED_VALUE }>;
/// Arms alias following the Thompson sampling algorithm.
pub type ThompsonSamplingArms<A, R, V> = Arms<A, R, V, { ArmsAlgorithm::THOMPSON_SAMPLING }>;
/// Arms alias following the UCB1 algorithm.
pub type UCB1Arms<A, R, V> = Arms<A, R, V, { ArmsAlgorithm::UCB1 }>;

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

    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a A> + 'a> {
        self.v.actions_iter()
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a S> + 'a> {
        self.v.states_iter()
    }

    fn call<T>(&self, state: &S, rng: &mut T) -> A
    where
        T: Rng + ?Sized,
    {
        // Evaluate the value function for each action.
        self.pi.call(&self.v, state, rng)
    }

    fn reset(&mut self) -> &mut Self {
        // Reset the policy.
        self.pi.reset();
        // Reset the (state-)action value function.
        self.v.reset();

        self
    }

    fn update(&mut self, action: &A, reward: &R, state: &S, is_done: bool) {
        // Update the policy.
        self.pi.update(action, reward, state, is_done);
        // Update the (state-)action value function.
        self.v.update(action, reward, state, is_done);
    }
}
