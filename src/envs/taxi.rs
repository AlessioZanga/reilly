use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::WeightedIndex;
use serde::{Deserialize, Serialize};

use super::Env;

/// Action enumerator for Taxi environment.
#[repr(usize)]
enum TaxiAction {
    MoveSouth = 0,
    MoveNorth = 1,
    MoveEast = 2,
    MoveWest = 3,
    PickUp = 4,
    DropOff = 5,
}

/// Port of `Taxi` from `OpenAI/Gym` as [here](https://github.com/openai/gym/blob/b704d4660e45edc7bb674a6c971d376990d340dc/gym/envs/toy_text/taxi.py).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Taxi {
    state: usize,
    #[serde(default, skip_serializing)]
    initial_states_distribution: Array1<f64>,
    #[serde(default, skip_serializing)]
    transition_matrix: HashMap<usize, HashMap<usize, Vec<(f64, usize, f64, bool)>>>,
}

impl Taxi {
    const MAP: [&'static str; 7] = [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ];

    /*
    self.desc = np.asarray(MAP, dtype="c")

    self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
    */

    const ROWS: usize = 5;
    const COLS: usize = 5;
    const MAX_ROW: usize = Self::ROWS - 1;
    const MAX_COL: usize = Self::COLS - 1;

    const STATES: usize = 500;
    const ACTIONS: [TaxiAction; 6] = [
        TaxiAction::MoveSouth,
        TaxiAction::MoveNorth,
        TaxiAction::MoveEast,
        TaxiAction::MoveWest,
        TaxiAction::PickUp,
        TaxiAction::DropOff,
    ];

    const LOCS: [(usize, usize); 4] = [(0, 0), (0, 4), (4, 0), (4, 3)];

    pub fn new() -> Self {
        let mut initial_states_distribution: Array1<f64> = ArrayBase::zeros((Self::STATES,));
        let mut transition_matrix: HashMap<usize, HashMap<usize, Vec<(f64, usize, f64, bool)>>> = Default::default();

        for s in 0..Self::STATES {
            for a in 0..Self::ACTIONS.len() {
                transition_matrix.entry(s).or_default().entry(a).or_default();
            }
        }

        for row in 0..Self::ROWS {
            for col in 0..Self::COLS {
                for pass_idx in 0..(Self::LOCS.len() + 1) {
                    for dest_idx in 0..Self::LOCS.len() {
                        let state = Self::encode(row, col, pass_idx, dest_idx);
                        if pass_idx < 4 && pass_idx != dest_idx {
                            initial_states_distribution[state] += 1.;
                            for action in Self::ACTIONS {
                                let (mut new_row, mut new_col, mut new_pass_idx) = (row, col, pass_idx);
                                let mut reward = -1.;
                                let mut done = false;
                                let taxi_loc = (row, col);
                                match action {
                                    TaxiAction::MoveSouth => {
                                        new_row = usize::min(row + 1, Self::MAX_ROW);
                                    }
                                    TaxiAction::MoveNorth => {
                                        new_row = usize::max(row - 1, 0);
                                    }
                                    TaxiAction::MoveEast => {
                                        if Self::MAP[1 + row].chars().nth(2 * col + 2) == Some(':') {
                                            new_col = usize::min(col + 1, Self::MAX_COL);
                                        }
                                    }
                                    TaxiAction::MoveWest => {
                                        if Self::MAP[1 + row].chars().nth(2 * col) == Some(':') {
                                            new_col = usize::max(col - 1, 0);
                                        }
                                    }
                                    TaxiAction::PickUp => {
                                        if pass_idx < 4 && taxi_loc == Self::LOCS[pass_idx] {
                                            new_pass_idx = 4;
                                        } else {
                                            // Passenger not at location.
                                            reward = -10.;
                                        }
                                    }
                                    TaxiAction::DropOff => {
                                        if pass_idx == 4 && taxi_loc == Self::LOCS[dest_idx] {
                                            new_pass_idx = dest_idx;
                                            reward = 20.;
                                            done = true;
                                        } else if pass_idx == 4 {
                                            if let Some(loc) = Self::LOCS.iter().position(|&loc| taxi_loc == loc) {
                                                new_pass_idx = loc;
                                            }
                                        } else {
                                            // Dropoff at wrong location.
                                            reward = -10.;
                                        }
                                    }
                                };
                                let new_state = Self::encode(new_row, new_col, new_pass_idx, dest_idx);
                                transition_matrix
                                    .get_mut(&state)
                                    .unwrap()
                                    .get_mut(&(action as usize))
                                    .unwrap()
                                    .push((1.0, new_state, reward, done));
                            }
                        }
                    }
                }
            }
        }

        initial_states_distribution /= initial_states_distribution.sum();

        Self {
            state: 0,
            initial_states_distribution,
            transition_matrix,
        }
    }

    fn encode(taxi_row: usize, taxi_col: usize, pass_loc: usize, dest_idx: usize) -> usize {
        let mut i = taxi_row;
        i *= 5;
        i += taxi_col;
        i *= 5;
        i += pass_loc;
        i *= 4;
        i += dest_idx;

        i
    }

    /*
    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)
    */
}

impl Default for Taxi {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Taxi {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Taxi-v3")
    }
}

impl Env<usize, f64, usize> for Taxi {
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a usize> + 'a> {
        todo!()
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = &'a usize> + 'a> {
        todo!()
    }

    fn get_state(&self) -> usize {
        self.state
    }

    fn call_mut<T>(&mut self, action: &usize, rng: &mut T) -> (f64, usize, bool)
    where
        T: rand::Rng + ?Sized,
    {
        let transitions = self.transition_matrix.get(&self.state).unwrap().get(action).unwrap();
        let weights = transitions.iter().map(|t| t.0);
        let idxs = WeightedIndex::new(weights).unwrap();
        let (_, next_state, reward, done) = transitions[idxs.sample(rng)];
        self.state = next_state;

        (reward, next_state, done)
    }

    fn reset<T>(&mut self, rng: &mut T) -> &mut Self
    where
        T: rand::Rng + ?Sized,
    {
        let weights = self.initial_states_distribution.iter();
        let idxs = WeightedIndex::new(weights).unwrap();
        self.state = idxs.sample(rng);

        self
    }
}
