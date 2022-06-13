use std::collections::HashSet;

use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainTest {
    train: usize,
    test: usize,
    repeat: usize,
    steps_max: usize,
}

impl TrainTest {
    /// Constructs a train-test session.
    pub fn new(train: usize, test: usize, repeat: usize) -> Self {
        Self {
            train,
            test,
            repeat,
            steps_max: usize::MAX,
        }
    }

    /// Sets the max number of steps for each episode.
    pub fn with_steps_max(mut self, steps_max: usize) -> Self {
        self.steps_max = steps_max;

        self
    }
}

impl Session for TrainTest {
    fn total_episodes(&self) -> usize {
        self.repeat * self.train
    }

    fn call_with_bar<A, R, S, V, P, G, E, T>(
        &self,
        agent: &mut G,
        environment: &mut E,
        rng: &mut T,
        progress: Option<ProgressBar>,
    ) -> DataFrame
    where
        A: Action,
        R: Reward,
        S: State,
        P: Policy,
        V: StateActionValue<A, R, S>,
        G: Agent<A, R, S, V, P>,
        E: Env<A, R, S>,
        T: Rng + ?Sized,
    {
        // Assert same action- and state-space.
        assert_eq!(
            HashSet::<A>::from_iter(agent.actions_iter()),
            HashSet::<A>::from_iter(environment.actions_iter()),
            "Agent and environment have different actions-space"
        );
        assert_eq!(
            HashSet::<S>::from_iter(agent.states_iter()),
            HashSet::<S>::from_iter(environment.states_iter()),
            "Agent and environment have different states-space"
        );
        // Allocate memory for data collection.
        let capacity = self.repeat * self.test;
        let mut rewd = Vec::with_capacity(capacity);
        let mut test = Vec::with_capacity(capacity);
        let mut reps = Vec::with_capacity(capacity);
        // Initialize progress bar.
        let progress = match progress {
            None => ProgressBar::new(self.total_episodes() as u64),
            Some(progress) => progress,
        }
        // Set progress message.
        .with_message("Executing train-test");
        // Set progress bar style.
        progress.set_style(ProgressStyle::default_bar().template(
            "{spinner} {msg}... ({percent}%) {wide_bar} [{pos}/{len}][{elapsed_precise}/{eta_precise}][{per_sec}]",
        ));
        // Print progress layout.
        progress.tick();
        // For each fold ...
        for i in 0..self.repeat {
            // ... perform n train episodes, then ...
            for _ in 0..self.train {
                // Declare future reward.
                let mut reward;
                // Reset the environment and get its initial state.
                let mut state = environment.reset(rng).state();
                // Reset the agent to the given initial state.
                agent.reset(state.clone());
                // Set is_done flag to false.
                let mut is_done = false;
                // Set the steps counter.
                let mut steps = 1;
                // While the episode is not over ...
                while !is_done {
                    // ... get the action for the current state ...
                    let action = agent.call(state, rng);
                    // ... perform the action ...
                    (reward, state, is_done) = environment.call_mut(action.clone(), rng);
                    // ... check if max number of steps is reached ...
                    is_done |= steps >= self.steps_max;
                    // ... update the agent ...
                    agent.update(action, reward, state.clone(), is_done);
                    // ... increment the steps counter.
                    steps += 1;
                }
                // Update progress.
                progress.inc(1);
            }
            // ... perform m test episodes.
            for j in 0..self.test {
                // Declare reward.
                let mut reward;
                // Reset the environment and get its initial state.
                let mut state = environment.reset(rng).state();
                // Reset the agent to the given initial state.
                agent.reset(state.clone());
                // Set is_done flag to false.
                let mut is_done = false;
                // Init the cumulative reward.
                let mut cum_reward = R::zero();
                // Set the steps counter.
                let mut steps = 1;
                // While the episode is not over ...
                while !is_done {
                    // ... get the action for the current state ...
                    let action = agent.call(state, rng);
                    // ... perform the action ...
                    (reward, state, is_done) = environment.call_mut(action, rng);
                    // ... update the cumulative reward ...
                    cum_reward = cum_reward + reward;
                    // ... check if max number of steps is reached.
                    is_done |= steps >= self.steps_max;
                    // ... increment the steps counter.
                    steps += 1;
                }
                // Record the cumulative reward.
                rewd.push(cum_reward.to_f64().unwrap());
                test.push(j as u64);
                reps.push(i as u64);
            }
        }
        // Close progress.
        progress.finish();

        // Cast data to polars DataFrame.
        let reps = Series::from_vec("rep", reps);
        let test = Series::from_vec("test", test);
        let rewd = Series::from_vec("reward", rewd);
        // Set agent and environment names.
        let mut agns = Series::from_iter(std::iter::repeat(agent.to_string()).take(capacity));
        agns.rename("agent");
        let mut envs = Series::from_iter(std::iter::repeat(environment.to_string()).take(capacity));
        envs.rename("environment");

        DataFrame::new(vec![envs, agns, reps, test, rewd]).expect("Unable to cast collected data to DataFrame")
    }
}
