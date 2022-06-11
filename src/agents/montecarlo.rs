use std::{
    fmt::{Display, Formatter},
    marker::PhantomData,
};

use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

use super::Agent;
use crate::{
    policies::Policy,
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Monte Carlo Q function with a `first-visit` update approach.
#[derive(Clone, Debug)]
pub struct FirstVisit<A, R, S>
where
    A: Action,
    R: Reward,
    S: State,
{
    gamma: R,
    state: S,
    trajectory: Vec<(S, A, R)>,
    n: Array2<usize>,
    q: Array2<R>,
}

impl FirstVisit<usize, f64, usize> {
    /// Constructs the Q function with a `first-visit` update approach.
    pub fn new<I, J>(actions_iter: I, states_iter: J, gamma: f64) -> Self
    where
        I: ExactSizeIterator<Item = usize>,
        J: ExactSizeIterator<Item = usize>,
    {
        let n = ArrayBase::zeros((states_iter.len(), actions_iter.len()));
        let q = ArrayBase::zeros((states_iter.len(), actions_iter.len()));

        Self {
            gamma,
            state: 0,
            trajectory: Default::default(),
            n,
            q,
        }
    }
}

impl<A, R, S> Display for FirstVisit<A, R, S>
where
    A: Action,
    R: Reward,
    S: State,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FirstVisit(γ={:?})", self.gamma)
    }
}

impl StateActionValue<usize, f64, usize> for FirstVisit<usize, f64, usize> {
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..self.q.shape()[1])
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..self.q.shape()[0])
    }

    fn call<T>(&self, state: usize, action: usize, _rng: &mut T) -> f64
    where
        T: rand::Rng + ?Sized,
    {
        self.q[(state, action)]
    }

    fn reset(&mut self) -> &mut Self {
        // Reset current state.
        self.state = 0;
        // Reset trajectory.
        self.trajectory.clear();
        // Reset counter.
        self.n.fill(0);
        // Reset Q-function.
        self.q.fill(0.);

        self
    }

    fn update(&mut self, action: usize, reward: f64, next_state: usize, is_done: bool) {
        // Append to current trajectory.
        self.trajectory.push((self.state, action, reward));
        // Update current state.
        self.state = next_state;
        // If end of episode is reached ...
        if is_done {
            // Compute discounted cumulative gain.
            let mut g = 0.;
            // Iterate from T-1 down to 0.
            for (t, &(s_t, a_t, r_t)) in self.trajectory.iter().enumerate().rev() {
                // G = gamma * G + R_t+1. NOTE: It uses fused multiply-add (FMA) intrinsic.
                g = self.gamma.mul_add(g, r_t);
                // If there is a matching (state, action) pair in (0..t-1) ...
                if self.trajectory[0..t]
                    .iter()
                    .any(|&(s_k, a_k, _)| s_k == s_t && a_k == a_t)
                {
                    // Increment the counter.
                    self.n[(s_t, a_t)] += 1;
                    // Q(S_t, A_t) = Q(S_t, A_t) + (G - Q(S_t, A_t)) / n_(S_t, A_t)
                    self.q[(s_t, a_t)] += (g - self.q[(s_t, a_t)]) / self.n[(s_t, a_t)] as f64;
                }
            }
            // Clear the trajectory.
            self.trajectory.clear();
            // Reset the counter.
            self.n.fill(0);
        }
    }
}

/// Monte Carlo Q function with a `every-visit` update approach.
#[derive(Clone, Debug)]
pub struct EveryVisit<A, R, S>
where
    A: Action,
    R: Reward,
    S: State,
{
    gamma: R,
    state: S,
    trajectory: Vec<(S, A, R)>,
    n: Array2<usize>,
    q: Array2<R>,
}

impl EveryVisit<usize, f64, usize> {
    /// Constructs the Q function with a `every-visit` update approach.
    pub fn new<I, J>(actions_iter: I, states_iter: J, gamma: f64) -> Self
    where
        I: ExactSizeIterator<Item = usize>,
        J: ExactSizeIterator<Item = usize>,
    {
        let n = ArrayBase::zeros((states_iter.len(), actions_iter.len()));
        let q = ArrayBase::zeros((states_iter.len(), actions_iter.len()));

        Self {
            gamma,
            state: 0,
            trajectory: Default::default(),
            n,
            q,
        }
    }
}

impl<A, R, S> Display for EveryVisit<A, R, S>
where
    A: Action,
    R: Reward,
    S: State,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "EveryVisit(γ={:?})", self.gamma)
    }
}

impl StateActionValue<usize, f64, usize> for EveryVisit<usize, f64, usize> {
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..self.q.shape()[1])
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..self.q.shape()[0])
    }

    fn call<T>(&self, state: usize, action: usize, _rng: &mut T) -> f64
    where
        T: rand::Rng + ?Sized,
    {
        self.q[(state, action)]
    }

    fn reset(&mut self) -> &mut Self {
        // Reset current state.
        self.state = 0;
        // Reset trajectory.
        self.trajectory.clear();
        // Reset counter.
        self.n.fill(0);
        // Reset Q-function.
        self.q.fill(0.);

        self
    }

    fn update(&mut self, action: usize, reward: f64, next_state: usize, is_done: bool) {
        // Append to current trajectory.
        self.trajectory.push((self.state, action, reward));
        // Update current state.
        self.state = next_state;
        // If end of episode is reached ...
        if is_done {
            // Compute discounted cumulative gain.
            let mut g = 0.;
            // Iterate from T-1 down to 0.
            for (s_t, a_t, r_t) in self.trajectory.drain(..).rev() {
                // G = gamma * G + R_t+1. NOTE: It uses fused multiply-add (FMA) intrinsic.
                g = self.gamma.mul_add(g, r_t);
                // Increment the counter.
                self.n[(s_t, a_t)] += 1;
                // Q(S_t, A_t) = Q(S_t, A_t) + (G - Q(S_t, A_t)) / n_(S_t, A_t)
                self.q[(s_t, a_t)] += (g - self.q[(s_t, a_t)]) / self.n[(s_t, a_t)] as f64;
            }
            // Reset the counter.
            self.n.fill(0);
        }
    }
}

/// Monte Carlo agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonteCarlo<A, R, S, Q, P>
where
    A: Action,
    R: Reward,
    S: State,
    Q: StateActionValue<A, R, S>,
    P: Policy,
{
    #[serde(default, skip_serializing)]
    _a: PhantomData<A>,
    #[serde(default, skip_serializing)]
    _r: PhantomData<R>,
    #[serde(default, skip_serializing)]
    _s: PhantomData<S>,
    q: Q,
    pi: P,
}

impl<A, R, S, Q, P> MonteCarlo<A, R, S, Q, P>
where
    A: Action,
    R: Reward,
    S: State,
    Q: StateActionValue<A, R, S>,
    P: Policy,
{
}

impl<A, R, S, Q, P> Display for MonteCarlo<A, R, S, Q, P>
where
    A: Action,
    R: Reward,
    S: State,
    Q: StateActionValue<A, R, S>,
    P: Policy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}-MonteCarlo", self.pi, self.q)
    }
}

impl<A, R, S, Q, P> Agent<A, R, S, Q, P> for MonteCarlo<A, R, S, Q, P>
where
    A: Action,
    R: Reward,
    S: State,
    Q: StateActionValue<A, R, S>,
    P: Policy,
{
    fn new(q: Q, pi: P) -> Self {
        Self {
            _a: PhantomData,
            _r: PhantomData,
            _s: PhantomData,
            q,
            pi,
        }
    }

    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = A> + 'a> {
        self.q.actions_iter()
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = S> + 'a> {
        self.q.states_iter()
    }

    fn call<T>(&self, state: S, rng: &mut T) -> A
    where
        T: rand::Rng + ?Sized,
    {
        // Evaluate the value function for each action.
        self.pi.call(&self.q, state, rng)
    }

    fn reset(&mut self) -> &mut Self {
        // Reset the Q function.
        self.q.reset();
        // Reset the policy.
        self.pi.reset();

        self
    }

    fn update(&mut self, action: A, reward: R, next_state: S, is_done: bool) {
        // Update the Q function.
        self.q.update(action, reward, next_state, is_done);
        // Update the policy.
        self.pi.update(is_done);
    }
}
