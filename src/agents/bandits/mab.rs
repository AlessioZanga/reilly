use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
    marker::PhantomData,
};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    agents::{bandits::arms::Arm, Agent},
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
    /// Choose an arm w.r.t. the maximum expected value plus the upper confidence bound (UCB1[^1]).
    ///
    /// [^1]: [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem.](https://scholar.google.com/scholar?q=Finite-time+Analysis+of+the+Multiarmed+Bandit+Problem)
    pub const UCB_1: usize = 2;
    /// Choose an arm w.r.t. the maximum expected value plus the upper confidence bound (UCB1-Normal[^1]).
    ///
    /// [^1]: [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem.](https://scholar.google.com/scholar?q=Finite-time+Analysis+of+the+Multiarmed+Bandit+Problem)
    pub const UCB_1_NORMAL: usize = 3;
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
    count: usize,
}

impl<A, R, V, const M: usize> Arms<A, R, V, M>
where
    A: Action,
    R: Reward,
    V: Arm<R>,
{
    /// Constructs a sequence of arms given the (action, arm) pairs.
    pub fn new<I>(actions_arms_iter: I) -> Self
    where
        I: Iterator<Item = (A, V)>,
    {
        let arms = actions_arms_iter.collect();

        Self {
            _r: PhantomData,
            arms,
            count: 0,
        }
    }
}

impl<A, R, V, const M: usize> Display for Arms<A, R, V, M>
where
    A: Action,
    R: Reward,
    V: Arm<R>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO: Explicit arm type and parameters.
        write!(
            f,
            "{}",
            match M {
                ArmsAlgorithm::EXPECTED_VALUE => "ExpectedValue",
                ArmsAlgorithm::THOMPSON_SAMPLING => "ThompsonSampling",
                ArmsAlgorithm::UCB_1 => "UCB1",
                ArmsAlgorithm::UCB_1_NORMAL => "UCB1Normal",
                _ => unreachable!(),
            }
        )
    }
}

impl<A, R, V, const M: usize> ActionValue<A, R> for Arms<A, R, V, M>
where
    A: Action,
    R: Reward,
    V: Arm<R>,
{
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = A> + 'a> {
        Box::new(self.arms.keys().cloned())
    }

    fn call<T>(&self, action: A, rng: &mut T) -> R
    where
        T: Rng + ?Sized,
    {
        // Get the arm given action.
        let a = &self.arms[&action];
        // Execute the specified algorithm.
        match M {
            // Compute the expected value.
            ArmsAlgorithm::EXPECTED_VALUE => a.call(),
            // Sample from the distribution.
            ArmsAlgorithm::THOMPSON_SAMPLING => a.sample(rng),
            // Compute the expected value plus the upper confidence bound
            // following the UCB1: Q(a) + sqrt(2 * ln(t) / n).
            ArmsAlgorithm::UCB_1 => {
                // Cast t and n.
                let t = R::from(self.count).unwrap();
                let n = R::from(a.get_count()).unwrap();

                a.call() + R::sqrt(R::from(2.).unwrap() * R::ln(t) / n)
            }
            // Compute the expected value plus the upper confidence bound
            // following the UCB1-Normal: Q(a) + sqrt(16 * [(q - n * Q(a)^2) / (n - 1)] * [ln(t - 1) / n]).
            ArmsAlgorithm::UCB_1_NORMAL => {
                // Cast t and n.
                let t = R::from(self.count).unwrap();
                let n = R::from(a.get_count()).unwrap();

                // Compute Q(a) and Q(n).
                let q_a = a.call();
                let q_n = a.get_sum_squared_rewards();

                q_a + R::sqrt(
                    // Constant.
                    R::from(16.).unwrap() *
                    // First term.
                    ((q_n - n * R::powi(q_a, 2)) / (n - R::one())) *
                    // Second term.
                    R::ln(t - R::one()),
                )
            }
            // Invalid algorithm enumerator.
            _ => unreachable!(),
        }
    }

    fn reset(&mut self) -> &mut Self {
        // Reset each arm.
        self.arms.iter_mut().for_each(|(_, arm)| arm.reset());
        // Reset the counter.
        self.count = 0;

        self
    }

    fn update(&mut self, action: A, reward: R, _is_done: bool) {
        // Update the arm association with performed action given the obtained reward.
        self.arms
            .get_mut(&action)
            .expect("Unable to get bandit's arm for given action")
            .update(reward);
        // Increase the counter.
        self.count += 1;
    }
}

/// Arms alias following the expected value algorithm.
pub type ExpectedValueArms<A, R, V> = Arms<A, R, V, { ArmsAlgorithm::EXPECTED_VALUE }>;
/// Arms alias following the Thompson sampling algorithm.
pub type ThompsonSamplingArms<A, R, V> = Arms<A, R, V, { ArmsAlgorithm::THOMPSON_SAMPLING }>;
/// Arms alias following the UCB1 algorithm.
pub type UCB1Arms<A, R, V> = Arms<A, R, V, { ArmsAlgorithm::UCB_1 }>;
/// Arms alias following the UCB-Normal algorithm.
pub type UCB1NormalArms<A, R, V> = Arms<A, R, V, { ArmsAlgorithm::UCB_1_NORMAL }>;

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
