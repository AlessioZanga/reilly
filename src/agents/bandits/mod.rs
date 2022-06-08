/// Bandit's arms module.
pub mod arms;

mod mab;
pub use mab::{Arms, ExpectedValueArms, MultiArmedBandit, ThompsonSamplingArms, UCB1Arms, UCB1NormalArms};
