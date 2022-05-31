use std::{fmt::Debug, hash::Hash};

use num::traits::Float;

/// Definition of a generic action.
pub trait Action: Clone + Debug + Eq + PartialEq + Hash {}

/// Definition of a generic reward.
pub trait Reward: Debug + Float {}

/// Definition of a generic state.
pub trait State: Clone + Debug + Eq + PartialEq + Hash {}

// FIXME: Restrict to primitive types.
impl<T> Action for T where T: Clone + Debug + Eq + PartialEq + Hash {}
impl<T> Reward for T where T: Debug + Float {}
impl<T> State for T where T: Clone + Debug + Eq + PartialEq + Hash {}
