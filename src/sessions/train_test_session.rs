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
    folds: usize,
}

impl TrainTestSession {
    /// Constructs a train-test session.
    pub fn new(train: usize, test: usize, folds: usize) -> Self {
        Self { train, test, folds }
    }
}

impl Session for TrainTestSession {
    fn call<A, R, S, P, V, G, E, T>(&self, agent: &mut G, environment: &mut E, rng: &mut T)
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
        // For each fold ...
        for _ in 0..self.folds {
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
                    (reward, state, is_done) = environment.call_mut(&action);
                    // ... update the agent.
                    agent.update(&action, &reward, &state, is_done);
                }
            }
            // ... perform m test episodes.
            for _ in 0..self.test {
                // Reset the environment and get its initial state.
                let mut state = environment.reset().get_state();
                // Set is_done flag to false.
                let mut is_done = false;
                // Declare reward.
                let mut reward;
                // While the episode is not over ...
                while !is_done {
                    // ... get the action for the current state ...
                    let action = agent.call(&state, rng);
                    // ... perform the action ...
                    (reward, state, is_done) = environment.call_mut(&action);
                    // TODO: ... record the obtained reward.
                    println!("r: {:?}", reward);
                }
            }
        }
    }
}
