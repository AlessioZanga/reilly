use std::fmt::Debug;

use rand::prelude::*;

use crate::types::Reward;

/// Definition of a generic bandit arm.
pub trait Arm<R>: Clone + Debug
where
    R: Reward,
{
    /// Gets the number of times the arm has been pulled.
    fn count(&self) -> usize;

    /// Gets the sum of the squared rewards obtained so far.
    fn sum_squared_rewards(&self) -> R;

    /// Gets expected reward.
    fn call(&self) -> R;

    /// Sample from underlying distribution with given random number generator.
    fn sample<T: Rng + ?Sized>(&self, rng: &mut T) -> R;

    /// Update the  underlying distribution parameters.
    fn update(&mut self, reward: R);
}
