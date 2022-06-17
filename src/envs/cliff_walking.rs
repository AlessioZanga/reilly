use std::{
    fmt::{Display, Formatter},
    io::Write,
    thread::sleep,
    time::Duration,
};

use console::{style, Term};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

use super::Env;

/// Port of `CliffWalking-v0` from `OpenAI/Gym` as [here](https://github.com/openai/gym/blob/9acf9cd367fb1ea97ac1e394969df87fd9a0d5c9/gym/envs/toy_text/cliffwalking.py).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CliffWalkingGeneric<const D: bool> {
    state: usize,
    transition_matrix: Array2<usize>,
    reward_matrix: Array2<f64>,
    is_terminal: Array2<bool>,
}

impl<const D: bool> CliffWalkingGeneric<D> {
    const ROWS: usize = 4;
    const COLS: usize = 12;

    /// Cardinality of the states space.
    ///
    /// UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3.
    pub const ACTIONS: usize = 4;
    /// Cardinality of the states space.
    pub const STATES: usize = Self::ROWS * Self::COLS;

    const fn encode(row: usize, col: usize) -> usize {
        row * Self::COLS + col
    }

    const fn decode(i: usize) -> (usize, usize) {
        (i / Self::COLS, i % Self::COLS)
    }

    fn calculate_transition_prob(row: usize, col: usize, action: usize) -> (usize, f64, bool) {
        // Update current position.
        let (row, col) = match action {
            0 => (row.saturating_sub(1), col),
            1 => (row, col + 1),
            2 => (row + 1, col),
            3 => (row, col.saturating_sub(1)),
            _ => unreachable!(),
        };

        let (row, col) = (
            // Limit coordinates.
            usize::min(row, Self::ROWS - 1),
            usize::min(col, Self::COLS - 1),
        );

        let new_state = Self::encode(row, col);

        // If we reached the cliff[3, 1:-1] ...
        if row == 3 && (1..Self::COLS).contains(&col) {
            // ... then we fall.
            return (Self::encode(3, 0), -100., false);
        }

        // If we reached the terminal state, then the episode is over.
        let is_done = row == Self::ROWS - 1 && col == Self::COLS - 1;

        (new_state, -1., is_done)
    }

    /// Constructs a `CliffWalking-v0` environment.
    pub fn new() -> Self {
        // We always start in state (3, 0).
        let state = Self::encode(3, 0);
        // Transition function (state, action) -> next_state.
        let mut transition_matrix = ArrayBase::zeros((Self::STATES, Self::ACTIONS));
        // Reward function (state, action) -> reward.
        let mut reward_matrix = ArrayBase::from_elem((Self::STATES, Self::ACTIONS), -1.);
        // Check if next state is terminal state.
        let mut is_terminal = ArrayBase::from_elem((Self::STATES, Self::ACTIONS), false);

        for s in 0..Self::STATES {
            for a in 0..Self::ACTIONS {
                let (row, col) = Self::decode(s);
                let (next_state, reward, is_done) = Self::calculate_transition_prob(row, col, a);

                transition_matrix[(s, a)] = next_state;
                reward_matrix[(s, a)] = reward;
                is_terminal[(s, a)] = is_done;
            }
        }

        Self {
            state,
            transition_matrix,
            reward_matrix,
            is_terminal,
        }
    }

    /// Renders the environment in a text-based mode.
    pub fn render(&self) -> std::io::Result<()> {
        // Get terminal handle.
        let mut terminal = Term::stdout();
        // Disable cursor highlighting.
        terminal.hide_cursor()?;
        // Print environment.
        for row in 0..Self::ROWS {
            for col in 0..Self::COLS {
                let text = if (row, col) == (Self::ROWS - 1, 0) {
                    "S"
                } else if (row, col) == (Self::ROWS - 1, Self::COLS - 1) {
                    "T"
                } else if row == 3 && (1..Self::COLS).contains(&col) {
                    "C"
                } else {
                    "o"
                };

                let mut text = text.to_owned();
                if self.state == Self::encode(row, col) {
                    text = format!("{}", style(text).on_red());
                }

                terminal.write_all(text.as_bytes())?;
            }
            terminal.write_all(&[b'\n'])?;
        }
        // Reset cursor.
        terminal.move_cursor_up(Self::ROWS)?;
        // Enable cursor highlighting.
        terminal.show_cursor()?;
        // Sleep for 20 milliseconds, i.e. set speed at 50 FPS.
        sleep(Duration::from_millis(1000 / 50));

        Ok(())
    }
}

impl<const D: bool> Display for CliffWalkingGeneric<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CliffWalking-v0")
    }
}

impl<const D: bool> Env<usize, f64, usize> for CliffWalkingGeneric<D> {
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

    fn reset<T>(&mut self, _rng: &mut T) -> &mut Self
    where
        T: rand::Rng + ?Sized,
    {
        // We always start in state (3, 0).
        self.state = Self::encode(3, 0);

        self
    }
}

/// `CliffWalking-v0` with no display.
pub type CliffWalking = CliffWalkingGeneric<false>;
/// `CliffWalking-v0` with display.
pub type CliffWalkingWithDisplay = CliffWalkingGeneric<true>;
