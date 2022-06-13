use std::{
    fmt::{Display, Formatter},
    thread::sleep,
    time::Duration,
};

use console::{style, Term};
use ndarray::prelude::*;
use rand_distr::{Distribution, WeightedIndex};
use serde::{Deserialize, Serialize};

use super::Env;

/// Port of `Taxi-v3` from `OpenAI/Gym` as [here](https://github.com/openai/gym/blob/b704d4660e45edc7bb674a6c971d376990d340dc/gym/envs/toy_text/taxi.py).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Taxi<const D: bool> {
    state: usize,
    p_states_0: Array1<f64>,
    transition_matrix: Array2<usize>,
    reward_matrix: Array2<f64>,
    is_terminal: Array2<bool>,
}

impl<const D: bool> Taxi<D> {
    /// Textual representation of the environment map.
    pub const MAP: &'static str = concat!(
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    );

    /*
    self.desc = np.asarray(MAP, dtype="c")

    self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
    */

    const ROWS: usize = 5;
    const COLS: usize = 5;
    const MAX_ROW: usize = Self::ROWS - 1;
    const MAX_COL: usize = Self::COLS - 1;

    /// Cardinality of the states space.
    pub const STATES: usize = 500;
    /// Cardinality of the states space.
    pub const ACTIONS: usize = 6;

    const LOCS: [(usize, usize); 4] = [(0, 0), (0, 4), (4, 0), (4, 3)];

    const fn encode(taxi_row: usize, taxi_col: usize, pass_idx: usize, dest_idx: usize) -> usize {
        let mut i = taxi_row;
        i *= Self::ROWS;
        i += taxi_col;
        i *= Self::COLS;
        i += pass_idx;
        i *= Self::LOCS.len();
        i += dest_idx;

        i
    }

    fn decode(i: usize) -> (usize, usize, usize, usize) {
        let (dest_idx, i) = (i % Self::LOCS.len(), i / Self::LOCS.len());
        let (pass_idx, i) = (i % Self::COLS, i / Self::COLS);
        let (taxi_col, i) = (i % Self::ROWS, i / Self::ROWS);
        let taxi_row = i;
        assert!(i < 5);

        (taxi_row, taxi_col, pass_idx, dest_idx)
    }

    /// Constructs a `Taxi-v3` environment.
    pub fn new() -> Self {
        // Probability distribution of the initial state.
        let mut p_states_0 = ArrayBase::zeros((Self::STATES,));
        // Transition function (state, action) -> next_state.
        let mut transition_matrix = ArrayBase::zeros((Self::STATES, Self::ACTIONS));
        // Reward function (state, action) -> reward.
        let mut reward_matrix = ArrayBase::from_elem((Self::STATES, Self::ACTIONS), -1.);
        // Check if next state is terminal state.
        let mut is_terminal = ArrayBase::from_elem((Self::STATES, Self::ACTIONS), false);

        // For each row ...
        for row in 0..Self::ROWS {
            // ... for each column ...
            for col in 0..Self::COLS {
                // ... for each target location (plus the one in which the pick-up occur) ...
                for pass_idx in 0..(Self::LOCS.len() + 1) {
                    // ... for each target location ...
                    for dest_idx in 0..Self::LOCS.len() {
                        // Map the bi-dimensional location of the taxi to the one dimensional.
                        let state = Self::encode(row, col, pass_idx, dest_idx);
                        // Check is current location is not a target location.
                        if pass_idx < 4 && pass_idx != dest_idx {
                            // Set the current location as a potential starting state.
                            p_states_0[state] += 1.;
                        }
                        // ... for each action ...
                        for action in 0..Self::ACTIONS {
                            // Initialize location, reward, terminal state flag.
                            let mut reward = -1.;
                            let mut done = false;
                            let taxi_loc = (row, col);
                            // Copy current location.
                            let (mut new_row, mut new_col, mut new_pass_idx) = (row, col, pass_idx);

                            // Match current action.
                            match action {
                                // Move south.
                                0 => {
                                    new_row = usize::min(row + 1, Self::MAX_ROW);
                                }
                                // Move north.
                                1 => {
                                    // new_row = max(row - 1, 0)
                                    new_row = usize::saturating_sub(row, 1);
                                }
                                // Move east.
                                2 => {
                                    if Self::MAP.as_bytes()[(row + 1) * Self::COLS + (2 * (col + 1))] == b':' {
                                        new_col = usize::min(col + 1, Self::MAX_COL);
                                    }
                                }
                                // Move west.
                                3 => {
                                    if Self::MAP.as_bytes()[(row + 1) * Self::COLS + (2 * col)] == b':' {
                                        new_col = usize::max(col - 1, 0);
                                    }
                                }
                                // Pick-up.
                                4 => {
                                    if pass_idx < 4 && taxi_loc == Self::LOCS[pass_idx] {
                                        new_pass_idx = 4;
                                    } else {
                                        // Passenger not at location.
                                        reward = -10.;
                                    }
                                }
                                // Drop-off.
                                5 => {
                                    if pass_idx == 4 && taxi_loc == Self::LOCS[dest_idx] {
                                        new_pass_idx = dest_idx;
                                        reward = 20.;
                                        done = true;
                                    } else if pass_idx == 4 && Self::LOCS.iter().any(|&loc| taxi_loc == loc) {
                                        new_pass_idx = Self::LOCS.iter().position(|&loc| taxi_loc == loc).unwrap();
                                    } else {
                                        // Dropoff at wrong location.
                                        reward = -10.;
                                    }
                                }
                                // Invalid action.
                                _ => unreachable!(),
                            };

                            // Encode the new location.
                            let new_state = Self::encode(new_row, new_col, new_pass_idx, dest_idx);

                            // Update the transition matrix.
                            transition_matrix[(state, action)] = new_state;
                            reward_matrix[(state, action)] = reward;
                            is_terminal[(state, action)] = done;
                        }
                    }
                }
            }
        }

        // Normalize initial state distribution probability.
        p_states_0 /= p_states_0.sum();

        Self {
            state: 0,
            p_states_0,
            transition_matrix,
            reward_matrix,
            is_terminal,
        }
    }

    /// Renders the environment in a text-based mod.
    pub fn render(&self) -> std::io::Result<()> {
        // Decode current state ...
        let (taxi_row, taxi_col, pass_idx, dest_idx) = Self::decode(self.state);
        // ... and current passenger location (when not in the taxi) ...
        let (pass_row, pass_col) = Self::LOCS[pass_idx % Self::LOCS.len()];
        // ... and current passenger destination.
        let (dest_row, dest_col) = Self::LOCS[dest_idx];
        // Get terminal handle.
        let terminal = Term::stdout();
        // Disable cursor highlighting.
        terminal.hide_cursor()?;
        // Print environment.
        for row in 0..(Self::ROWS + 2) {
            let text = &Self::MAP[(row * (2 * Self::COLS + 1))..((row + 1) * (2 * Self::COLS + 1))];
            let text = text
                .chars()
                .enumerate()
                .map(|(col, char)| {
                    // Draw the taxi ...
                    if row == (1 + taxi_row) && col == (2 * taxi_col + 1) {
                        if pass_idx < 4 {
                            // ... without the passenger.
                            format!("{}", style(char).on_yellow())
                        } else {
                            // ... with the passenger.
                            format!("{}", style(char).on_green())
                        }
                    // Draw the passenger (when not in the taxi).
                    } else if row == (1 + pass_row) && col == (2 * pass_col + 1) && pass_idx < 4 {
                        format!("{}", style(char).on_blue())
                    // Draw the destination of the passenger.
                    } else if row == (1 + dest_row) && col == (2 * dest_col + 1) {
                        format!("{}", style(char).on_magenta())
                    } else {
                        format!("{}", char)
                    }
                })
                .collect::<Vec<String>>()
                .join("");
            terminal.write_line(&text)?;
        }
        // Reset cursor.
        terminal.move_cursor_up(Self::ROWS + 2)?;
        // Enable cursor highlighting.
        terminal.show_cursor()?;
        // Sleep for 20 milliseconds, i.e. set speed at 50 FPS.
        sleep(Duration::from_millis(1000 / 50));

        Ok(())
    }
}

impl<const D: bool> Default for Taxi<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: bool> Display for Taxi<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Taxi-v3")
    }
}

impl<const D: bool> Env<usize, f64, usize> for Taxi<D> {
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..Self::ACTIONS)
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..Self::STATES)
    }

    fn state(&self) -> usize {
        self.state
    }

    fn call_mut<T>(&mut self, action: usize, _rng: &mut T) -> (f64, usize, bool)
    where
        T: rand::Rng + ?Sized,
    {
        // Map transition matrix index.
        let idx = (self.state, action);
        // Get reward, next state and termination flag.
        let next_state = self.transition_matrix[idx];
        let reward = self.reward_matrix[idx];
        let done = self.is_terminal[idx];
        // Update current state.
        self.state = next_state;

        // If display is set, render current state.
        if D {
            self.render().expect("Unable to render current environment state");
        }

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

/// `Taxi-v3` with no display.
pub type TaxiNoDisplay = Taxi<false>;
/// `Taxi-v3` with display.
pub type TaxiWithDisplay = Taxi<true>;
