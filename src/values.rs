use std::fmt::{Debug, Display};

use rand::Rng;

use crate::types::{Action, Reward, State};

/// Definition of the action value function.
pub trait ActionValue<A, R>: Clone + Debug + Display
where
    A: Action,
    R: Reward,
{
    /// Iterates of the action space.
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = A> + 'a>;

    /// Computes the expected reward of the given action.
    fn call<T>(&self, action: A, rng: &mut T) -> R
    where
        T: Rng + ?Sized;

    /// Resets the function.
    fn reset(&mut self) -> &mut Self;

    /// Updates the action-value function given performed action, obtained reward, next state and end-of-episode flag.
    fn update(&mut self, action: A, reward: R, is_done: bool);
}

/// Definition of the state-action value function.
pub trait StateActionValue<A, R, S>: Clone + Debug + Display
where
    A: Action,
    R: Reward,
    S: State,
{
    /// Iterates of the action space.
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = A> + 'a>;

    /// Iterates of the state space.
    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = S> + 'a>;

    /// Computes the expected reward of the given action-state pair.
    fn call<T>(&self, state: S, action: A, rng: &mut T) -> R
    where
        T: Rng + ?Sized;

    /// Resets the function.
    fn reset(&mut self) -> &mut Self;

    /// Updates the state-value function given performed action, obtained reward, next state and end-of-episode flag.
    fn update(&mut self, action: A, reward: R, next_state: S, is_done: bool);
}

// Auto-implements state-action value function for action value function
// using nil state as agent's state, i.e. S = ().
impl<A, R, V> StateActionValue<A, R, ()> for V
where
    A: Action,
    R: Reward,
    V: ActionValue<A, R>,
{
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = A> + 'a> {
        self.actions_iter()
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = ()> + 'a> {
        // The nil state is the only state.
        Box::new([()].into_iter())
    }

    fn call<T>(&self, _state: (), action: A, rng: &mut T) -> R
    where
        T: Rng + ?Sized,
    {
        self.call(action, rng)
    }

    fn reset(&mut self) -> &mut Self {
        self.reset()
    }

    fn update(&mut self, action: A, reward: R, _next_state: (), is_done: bool) {
        self.update(action, reward, is_done);
    }
}
