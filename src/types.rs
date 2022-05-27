use std::{fmt::Debug, hash::Hash};

use num::cast::AsPrimitive;

/// Definition of a generic action.
pub trait Action: Clone + Debug + Eq + PartialEq + Hash {}

/// Definition of a generic reward.
pub trait Reward: AsPrimitive<f64> + Clone + Debug + PartialOrd {}

/// Definition of a generic state.
pub trait State: Clone + Debug + Eq + PartialEq + Hash {}

// FIXME: Restrict to primitive types.
impl<T> Action for T where T: Clone + Debug + Eq + PartialEq + Hash {}
impl<T> Reward for T where T: AsPrimitive<f64> + Clone + Debug + PartialOrd {}
impl<T> State for T where T: Clone + Debug + Eq + PartialEq + Hash {}
