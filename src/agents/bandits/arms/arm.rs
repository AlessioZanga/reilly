use std::fmt::Debug;

use rand::prelude::*;

use crate::types::Reward;

/// Definition of a generic bandit arm.
pub trait Arm<R>: Clone + Debug + Default
where
    R: Reward,
{
    /// Gets expected reward.
    fn call(&self) -> R;

    /// Resets the arm.
    fn reset(&mut self);

    /// Sample from underlying distribution with given random number generator.
    fn sample<T: Rng + ?Sized>(&self, rng: &mut T) -> R;

    /// Update the  underlying distribution parameters.
    fn update(&mut self, reward: R);
}
