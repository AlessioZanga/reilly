use std::fmt::Debug;

use polars::prelude::*;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    agents::Agent,
    envs::Env,
    policies::Policy,
    types::{Action, Reward, State},
    values::StateActionValue,
};

/// Definition of agent-environment experiment session.
pub trait Session: Clone + Debug {
    /// Gets the (expected) total number of episodes.
    fn total_episodes(&self) -> usize;

    /// Execute the experiment session.
    fn call<A, R, S, V, P, G, E, T>(&self, agent: &mut G, environment: &mut E, rng: &mut T) -> DataFrame
    where
        A: Action,
        R: Reward,
        S: State,
        P: Policy,
        V: StateActionValue<A, R, S>,
        G: Agent<A, R, S, V, P>,
        E: Env<A, R, S>,
        T: Rng + ?Sized;

    /// Execute the experiment session in parallel.
    fn par_call<'a, A, R, S, V, P, G, E, T, I>(&self, iter: I, rng: &mut T) -> DataFrame
    where
        Self: Sync,
        A: Action,
        R: Reward,
        S: State,
        P: Policy,
        V: StateActionValue<A, R, S>,
        G: Agent<A, R, S, V, P> + 'a,
        E: Env<A, R, S> + 'a,
        T: Rng + SeedableRng + ?Sized,
        I: IndexedParallelIterator<Item = &'a mut (G, E)>,
    {
        // Generate seeds for each random number generator.
        let seeds: Vec<_> = (0..iter.len()).map(|_| rng.next_u64()).collect();
        // For each (agent, environment) pair ...
        let iter = iter
            // ... link a seed to the pair ...
            .zip_eq(seeds)
            // ... distribute the workload across each thread.
            .map(|((agent, environment), seed)| {
                // Initialize a local random number generator given the seed.
                let mut rng: T = SeedableRng::seed_from_u64(seed);
                // Call train-test for each pair.
                self.call(agent, environment, &mut rng)
            });
        // Stack the resulting dataframes. NOTE: The reduce order is non-deterministic.
        let mut data = iter
            .reduce_with(|a, b| a.vstack(&b).expect("Unable to stack dataframes"))
            .expect("Unable to reduce the dataframes");
        // Re-chunk the dataframe after the reduction.
        data.rechunk();

        data
    }
}
