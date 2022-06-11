use std::fmt::{Display, Formatter};

use ndarray::prelude::*;
use rand_distr::{Distribution, WeightedIndex};
use serde::{Deserialize, Serialize};

use super::Env;

/// Port of `FrozenLake-v1` from `OpenAI/Gym` as [here](https://github.com/openai/gym/blob/0263deb5ab8dce46b1056f5baa2c7c141fad9471/gym/envs/toy_text/frozen_lake.py).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenLake<const N: usize, const S: bool> {
    state: usize,
    p_states_0: Array1<f64>,
    probability_matrix: Array2<f64>,
    transition_matrix: Array2<usize>,
    reward_matrix: Array2<f64>,
    is_terminal: Array2<bool>,
}

impl<const N: usize, const S: bool> FrozenLake<N, S> {
    /// Textual representation of the environment map.
    #[rustfmt::skip]
    pub const MAP: &'static str = match N {
        4 => concat!(
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ),
        8 => concat!(
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ),
        _ => unreachable!(),
    };

    const ROWS: usize = N;
    const COLS: usize = N;

    /// Cardinality of the states space.
    pub const STATES: usize = Self::ROWS * Self::COLS;
    /// Cardinality of the states space.
    pub const ACTIONS: usize = 4;

    const fn to_s(row: usize, col: usize) -> usize {
        row * Self::COLS + col
    }

    const fn inc(mut row: usize, mut col: usize, action: usize) -> (usize, usize) {
        // Match action against enumerator.
        match action {
            // Left.
            0 => {
                // col = max(col - 1, 0).
                col = usize::saturating_sub(col, 1);
            }
            // Down.
            1 => {
                // row = min(row + 1, nrow - 1)
                // row = usize::min(row + 1, Self::ROWS - 1);
                // Const fn min/max implementation as [here](https://stackoverflow.com/a/53646925).
                let (a, b) = (row + 1, Self::ROWS - 1);
                row = [a, b][(a > b) as usize];
            }
            // Right.
            2 => {
                // col = min(col + 1, ncol - 1)
                // col = usize::min(col + 1, Self::COLS - 1);
                // Const fn min/max implementation as [here](https://stackoverflow.com/a/53646925).
                let (a, b) = (col + 1, Self::COLS - 1);
                col = [a, b][(a > b) as usize];
            }
            // Up.
            3 => {
                // row = max(row - 1, 0).
                row = usize::saturating_sub(row, 1);
            }
            // Invalid action.
            _ => unreachable!(),
        }

        (row, col)
    }

    const fn update_probability_matrix(row: usize, col: usize, action: usize) -> (f64, usize, bool) {
        let (new_row, new_col) = Self::inc(row, col, action);
        let new_state = Self::to_s(new_row, new_col);
        let new_char = Self::MAP.as_bytes()[new_state];
        let done = new_char == b'G' || new_char == b'H';
        let reward = (new_char == b'G') as u8 as f64;

        (reward, new_state, done)
    }

    /// Constructs a `FrozenLake-v1` environment.
    pub fn new() -> Self {
        // Probability distribution of the initial state.
        let mut p_states_0 = ArrayBase::from_iter(
            // Iter over MAP to find potential initial states.
            Self::MAP.as_bytes().iter().map(|&char| (char == b'S') as u8 as f64),
        );
        // Normalize initial state distribution probability.
        p_states_0 /= p_states_0.sum();

        // Probability matrix (state, action) -> probability of new action if slippery.
        let mut probability_matrix = ArrayBase::zeros((Self::STATES, Self::ACTIONS));
        // Transition function (state, action) -> next_state.
        let mut transition_matrix = ArrayBase::zeros((Self::STATES, Self::ACTIONS));
        // Reward function (state, action) -> reward.
        let mut reward_matrix = ArrayBase::zeros((Self::STATES, Self::ACTIONS));
        // Check if next state is terminal state.
        let mut is_terminal = ArrayBase::from_elem((Self::STATES, Self::ACTIONS), false);

        for row in 0..Self::ROWS {
            for col in 0..Self::COLS {
                let s = Self::to_s(row, col);
                for a in 0..Self::ACTIONS {
                    let char = Self::MAP.as_bytes()[s];
                    if char == b'G' || char == b'H' {
                        probability_matrix[(s, a)] = 1.;
                        transition_matrix[(s, a)] = s;
                        reward_matrix[(s, a)] = 0.;
                        is_terminal[(s, a)] = true;
                    } else {
                        // If is slippery ...
                        if S {
                            // ... assign each action a 1/(ACTIONS-1) probability ...
                            for b in 0..Self::ACTIONS {
                                // ... except the "non-perpendicular"/opposed action ...
                                if b != (a + 2) % Self::ACTIONS {
                                    let (reward, next_state, done) = Self::update_probability_matrix(row, col, b);
                                    probability_matrix[(s, b)] = 1. / (Self::ACTIONS - 1) as f64;
                                    transition_matrix[(s, b)] = next_state;
                                    reward_matrix[(s, b)] = reward;
                                    is_terminal[(s, b)] = done;
                                }
                            }
                        } else {
                            let (reward, next_state, done) = Self::update_probability_matrix(row, col, a);
                            probability_matrix[(s, a)] = 1.;
                            transition_matrix[(s, a)] = next_state;
                            reward_matrix[(s, a)] = reward;
                            is_terminal[(s, a)] = done;
                        }
                    }
                }
            }
        }

        Self {
            state: 0,
            p_states_0,
            probability_matrix,
            transition_matrix,
            reward_matrix,
            is_terminal,
        }
    }
}

impl<const N: usize, const S: bool> Default for FrozenLake<N, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const S: bool> Display for FrozenLake<N, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FrozenLake-{}x{}{}-v1", N, N, if S { "-Slippery" } else { "" })
    }
}

impl<const N: usize, const S: bool> Env<usize, f64, usize> for FrozenLake<N, S> {
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..Self::ACTIONS)
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..Self::STATES)
    }

    fn get_state(&self) -> usize {
        self.state
    }

    fn call_mut<T>(&mut self, action: usize, rng: &mut T) -> (f64, usize, bool)
    where
        T: rand::Rng + ?Sized,
    {
        // Copy current action.
        let mut action = action;
        // If slippery ...
        if S {
            // ... get new action distribution ...
            let actions = self.probability_matrix.row(self.state);
            let actions = WeightedIndex::new(actions).unwrap();
            // ... sample new action with given probability distribution.
            action = actions.sample(rng);
        }
        // Map transition matrix index.
        let idx = (self.state, action);
        // Get reward, next state and termination flag.
        let next_state = self.transition_matrix[idx];
        let reward = self.reward_matrix[idx];
        let done = self.is_terminal[idx];
        // Update current state.
        self.state = next_state;

        (reward, next_state, done)
    }

    fn reset<T>(&mut self, rng: &mut T) -> &mut Self
    where
        T: rand::Rng + ?Sized,
    {
        // Initialize new index distribution for initial state.
        let states = self.p_states_0.iter();
        let states = WeightedIndex::new(states).unwrap();
        // Sample new initial state.
        self.state = states.sample(rng);

        self
    }
}

/// `FrozenLake-v1` with 4x4 map and no slippery.
pub type FrozenLake4x4 = FrozenLake<4, false>;
/// `FrozenLake-v1` with 8x8 map and no slippery.
pub type FrozenLake8x8 = FrozenLake<8, false>;
/// `FrozenLake-v1` with 4x4 map and slippery.
pub type FrozenLake4x4Slippery = FrozenLake<4, true>;
/// `FrozenLake-v1` with 8x8 map and slippery.
pub type FrozenLake8x8Slippery = FrozenLake<8, true>;
