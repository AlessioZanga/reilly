use std::fmt::{Debug, Display, Formatter};

use rand::Rng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};

use super::Env;

/// Environment for multi-armed bandits given a sequence of distributions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FarWest<D>
where
    D: Clone + Debug + Distribution<f64>,
{
    distributions: Vec<D>,
    count: usize,
    end: usize,
}

impl<D> FarWest<D>
where
    D: Clone + Debug + Distribution<f64>,
{
    /// Constructs a far-west environment given a sequence of distributions and a time horizon.
    pub fn new<I>(distributions: I, end: usize) -> Self
    where
        I: Iterator<Item = D>,
    {
        let distributions = distributions.collect();

        Self {
            distributions,
            count: 0,
            end,
        }
    }
}

impl<D> Display for FarWest<D>
where
    D: Clone + Debug + Distribution<f64>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO: Explicit distributions type and parameters.
        write!(f, "FarWest")
    }
}

impl<D> Env<usize, f64, ()> for FarWest<D>
where
    D: Clone + Debug + Distribution<f64>,
{
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..self.distributions.len())
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = ()> + 'a> {
        Box::new([()].into_iter())
    }

    fn get_state(&self) {}

    fn call_mut<T>(&mut self, action: &usize, rng: &mut T) -> (f64, (), bool)
    where
        T: Rng + ?Sized,
    {
        // Check if we reached the end of the episode.
        let is_done = self.count >= self.end;
        // Increment counter.
        self.count += 1;

        (self.distributions[*action].sample(rng), (), is_done)
    }

    fn reset<T>(&mut self, _rng: &mut T) -> &mut Self
    where
        T: rand::Rng + ?Sized,
    {
        // Reset the time step counter.
        self.count = 0;

        self
    }
}
