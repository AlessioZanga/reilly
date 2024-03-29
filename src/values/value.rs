use crate::types::{Action, Reward, State};

/// Definition of the action value function.
pub trait ActionValue<A, R>
where
    A: Action,
    R: Reward,
{
    /// Iterates of the action space.
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a A> + 'a>;

    /// Computes the expected reward of the given action.
    fn call(&self, action: &A) -> R;

    /// Resets the function.
    fn reset(&mut self) -> &mut Self;

    /// Updates the agent given performed action and obtained reward.
    fn update(&mut self, action: &A, reward: &R);
}

/// Definition of the state-action value function.
pub trait StateActionValue<A, R, S>
where
    A: Action,
    R: Reward,
    S: State,
{
    /// Iterates of the action space.
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a A> + 'a>;

    /// Iterates of the state space.
    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a S> + 'a>;

    /// Computes the expected reward of the given action-state pair.
    fn call(&self, action: &A, state: &S) -> R;

    /// Resets the function.
    fn reset(&mut self) -> &mut Self;

    /// Updates the agent given performed action, obtained reward and next state.
    fn update(&mut self, action: &A, reward: &R, state: &S);
}

// Auto-implements state-action value function for action value function
// using nil state as agent's state, i.e. S = ().
impl<A, R, V> StateActionValue<A, R, ()> for V
where
    A: Action,
    R: Reward,
    V: ActionValue<A, R>,
{
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a A> + 'a> {
        self.actions_iter()
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a ()> + 'a> {
        // The nil state is the only state.
        Box::new([()].iter())
    }

    fn call(&self, action: &A, _state: &()) -> R {
        self.call(action)
    }

    fn reset(&mut self) -> &mut Self {
        self.reset()
    }

    fn update(&mut self, action: &A, reward: &R, _state: &()) {
        self.update(action, reward);
    }
}
