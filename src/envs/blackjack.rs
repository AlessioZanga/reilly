use std::fmt::{Display, Formatter};

use rand::{prelude::SliceRandom, Rng};
use serde::{Deserialize, Serialize};

use super::Env;

/// Port of `Blackjack-v1` from `OpenAI/Gym` as [here](https://github.com/openai/gym/blob/9acf9cd367fb1ea97ac1e394969df87fd9a0d5c9/gym/envs/toy_text/blackjack.py).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlackjackGeneric<const N: bool, const S: bool> {
    state: usize,
    dealer: Vec<usize>,
    player: Vec<usize>,
}

impl<const N: bool, const S: bool> BlackjackGeneric<N, S> {
    /// Cardinality of the states space.
    pub const ACTIONS: usize = 2;
    /// Cardinality of the states space.
    pub const STATES: usize = 32 * 11 * 2;

    /// Deck: 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10.
    pub const DECK: [usize; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];

    fn draw_card<T>(rng: &mut T) -> usize
    where
        T: Rng + ?Sized,
    {
        *Self::DECK.choose(rng).unwrap()
    }

    fn draw_hand<T>(rng: &mut T) -> Vec<usize>
    where
        T: Rng + ?Sized,
    {
        vec![Self::draw_card(rng), Self::draw_card(rng)]
    }

    fn usable_ace(hand: &[usize]) -> bool {
        let (flag, sum) = hand
            .iter()
            .fold((false, 0), |(flag, sum), &h| (flag || h == 1, sum + h));

        flag && sum + 10 <= 21
    }

    fn sum_hand(hand: &[usize]) -> usize {
        let sum: usize = hand.iter().cloned().sum();

        sum + (Self::usable_ace(hand) as usize) * 10
    }

    fn is_bust(hand: &[usize]) -> bool {
        Self::sum_hand(hand) > 21
    }

    fn score(hand: &[usize]) -> f64 {
        if Self::is_bust(hand) {
            0.
        } else {
            Self::sum_hand(hand) as f64
        }
    }

    fn is_natural(hand: &[usize]) -> bool {
        hand == [1, 10] || hand == [10, 1]
    }

    fn cmp(a: f64, b: f64) -> f64 {
        (a > b) as usize as f64 - (a < b) as usize as f64
    }

    fn encode(player: &[usize], dealer: &[usize]) -> usize {
        let mut i = Self::sum_hand(player);
        i *= 11;
        i += dealer[0];
        i *= 2;
        i += Self::usable_ace(player) as usize;

        debug_assert!(i < Self::STATES);

        i
    }

    /// Constructs a `Blackjack-v1` environment.
    pub fn new() -> Self {
        Self {
            state: 0,
            dealer: vec![1, 2],
            player: vec![3, 4],
        }
    }
}

impl<const N: bool, const S: bool> Display for BlackjackGeneric<N, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Blackjack{}{}-v1",
            if N { "-Natural" } else { "" },
            if S { "-SAB" } else { "" }
        )
    }
}

impl<const N: bool, const S: bool> Env<usize, f64, usize> for BlackjackGeneric<N, S> {
    fn actions_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..Self::ACTIONS)
    }

    fn states_iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = usize> + 'a> {
        Box::new(0..Self::STATES)
    }

    fn state(&self) -> usize {
        self.state
    }

    fn call_mut<T>(&mut self, action: usize, rng: &mut T) -> (f64, usize, bool)
    where
        T: rand::Rng + ?Sized,
    {
        self.state = Self::encode(&self.player, &self.dealer);

        let mut reward = 0.;
        let mut done = false;

        match action {
            // Hit: add a card to players hand and return.
            1 => {
                self.player.push(Self::draw_card(rng));
                if Self::is_bust(&self.player) {
                    reward = -1.;
                    done = true;
                }
            }
            // Stick: play out the dealers hand and score.
            0 => {
                while Self::sum_hand(&self.dealer) < 17 {
                    self.dealer.push(Self::draw_card(rng));
                }

                reward = Self::cmp(Self::score(&self.player), Self::score(&self.dealer));

                if S && Self::is_natural(&self.player) && !Self::is_natural(&self.dealer) {
                    // Player automatically wins. Rules consistent with S&Bs.
                    reward = 1.;
                } else if !S && N && Self::is_natural(&self.player) && reward == 1. {
                    // Natural gives extra points, but doesn't autowin. Legacy implementation.
                    reward = 1.5;
                }

                done = true;
            }
            _ => unreachable!(),
        }

        (reward, self.state, done)
    }

    fn reset<T>(&mut self, rng: &mut T) -> &mut Self
    where
        T: rand::Rng + ?Sized,
    {
        self.dealer = Self::draw_hand(rng);
        self.player = Self::draw_hand(rng);

        self.state = Self::encode(&self.player, &self.dealer);

        self
    }
}

/// Blackjack.
pub type Blackjack = BlackjackGeneric<false, false>;
/// Blackjack with rules consistent with S&Bs.
pub type BlackjackSAB = BlackjackGeneric<false, true>;
/// Natural blackjack.
pub type BlackjackNatural = BlackjackGeneric<true, false>;
/// Natural blackjack with rules consistent with S&Bs.
pub type BlackjackNaturalSAB = BlackjackGeneric<true, true>;
