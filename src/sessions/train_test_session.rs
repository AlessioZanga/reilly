use std::collections::HashSet;

use indicatif::ProgressBar;
use polars::prelude::*;
use rand::Rng;

use super::Session;
use crate::{
    agents::Agent,
    envs::Env,
    policies::Policy,
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// A train-test session, where an agent is trained for `n` episodes,
/// then tested for `m` episodes, repeating the train-test process for `k` times.
pub struct TrainTestSession {
    train: usize,
    test: usize,
    repeat: usize,
}

impl TrainTestSession {
    /// Constructs a train-test session.
    pub fn new(train: usize, test: usize, repeat: usize) -> Self {
        Self { train, test, repeat }
    }
}

impl Session for TrainTestSession {
    fn call<A, R, S, P, V, G, E, T>(&self, agent: &mut G, environment: &mut E, rng: &mut T) -> DataFrame
    where
        A: Action,
        R: Reward,
        S: State,
        P: Policy,
        V: StateActionValue<A, R, S>,
        G: Agent<A, R, S, P, V>,
        E: Env<A, R, S>,
        T: Rng + ?Sized,
    {
        // Assert same action- and state-space.
        assert_eq!(
            HashSet::<&A>::from_iter(agent.actions_iter()),
            HashSet::<&A>::from_iter(environment.actions_iter()),
            "Agent and environment have different actions-space"
        );
        assert_eq!(
            HashSet::<&S>::from_iter(agent.states_iter()),
            HashSet::<&S>::from_iter(environment.states_iter()),
            "Agent and environment have different states-space"
        );
        // Allocate memory for data collection.
        let capacity = self.repeat * self.test;
        let mut rewd = Vec::with_capacity(capacity);
        let mut test = Vec::with_capacity(capacity);
        let mut reps = Vec::with_capacity(capacity);
        // Initialize progress bar.
        let progress = ProgressBar::new((self.repeat * self.train) as u64);
        // For each fold ...
        for i in 0..self.repeat {
            // ... perform n train episodes, then ...
            for _ in 0..self.train {
                // Declare future reward.
                let mut reward;
                // Reset the environment and get its initial state.
                let mut state = environment.reset().get_state();
                // Set is_done flag to false.
                let mut is_done = false;
                // While the episode is not over ...
                while !is_done {
                    // ... get the action for the current state ...
                    let action = agent.call(&state, rng);
                    // ... perform the action ...
                    (reward, state, is_done) = environment.call_mut(&action, rng);
                    // ... update the agent.
                    agent.update(&action, &reward, &state, is_done);
                }
                // Update progress.
                progress.inc(1);
            }
            // ... perform m test episodes.
            for j in 0..self.test {
                // Reset the environment and get its initial state.
                let mut state = environment.reset().get_state();
                // Set is_done flag to false.
                let mut is_done = false;
                // Init the cumulative reward.
                let mut cum_reward = 0.;
                // Declare reward.
                let mut reward;
                // While the episode is not over ...
                while !is_done {
                    // ... get the action for the current state ...
                    let action = agent.call(&state, rng);
                    // ... perform the action ...
                    (reward, state, is_done) = environment.call_mut(&action, rng);
                    // ... update the cumulative reward.
                    cum_reward += reward.as_();
                }
                // Record the cumulative reward.
                rewd.push(cum_reward);
                test.push(j as u64);
                reps.push(i as u64);
            }
        }
        // Close progress.
        progress.finish();

        // Cast data to polars DataFrame.
        let rewd = ChunkedArray::<Float64Type>::from_vec("reward", rewd).into_series();
        let test = ChunkedArray::<UInt64Type>::from_vec("test", test).into_series();
        let reps = ChunkedArray::<UInt64Type>::from_vec("reps", reps).into_series();
        DataFrame::new(vec![reps, test, rewd]).expect("Unable to cast collected data to DataFrame")
    }
}
