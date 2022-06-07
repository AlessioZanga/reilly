mod values {
    mod arms {
        mod expected_value {
            mod bernoulli {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::arms::Bernoulli,
                    values::{ExpectedValueArms, StateActionValue},
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ExpectedValueArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.actions_iter()), BTreeSet::from_iter(0..5));
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ExpectedValueArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ExpectedValueArms::new(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let mut arms = ExpectedValueArms::new(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let mut arms = ExpectedValueArms::new(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ExpectedValueArms::new(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ExpectedValueArms::new(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: ExpectedValueArms<i32, f64, Bernoulli> = serde_json::from_str(&json).unwrap();
                }
            }

            mod normal {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::arms::Normal,
                    values::{ExpectedValueArms, StateActionValue},
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ExpectedValueArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.actions_iter()), BTreeSet::from_iter(0..5));
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ExpectedValueArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ExpectedValueArms::new(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let mut arms = ExpectedValueArms::new(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let mut arms = ExpectedValueArms::new(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ExpectedValueArms::new(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ExpectedValueArms::new(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: ExpectedValueArms<i32, f64, Normal> = serde_json::from_str(&json).unwrap();
                }
            }
        }

        mod thompson_sampling {
            mod bernoulli {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::arms::Bernoulli,
                    values::{StateActionValue, ThompsonSamplingArms},
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ThompsonSamplingArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.actions_iter()), BTreeSet::from_iter(0..5));
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ThompsonSamplingArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ThompsonSamplingArms::new(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let mut arms = ThompsonSamplingArms::new(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let mut arms = ThompsonSamplingArms::new(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ThompsonSamplingArms::new(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = ThompsonSamplingArms::new(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: ThompsonSamplingArms<i32, f64, Bernoulli> = serde_json::from_str(&json).unwrap();
                }
            }

            mod normal {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::arms::Normal,
                    values::{StateActionValue, ThompsonSamplingArms},
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ThompsonSamplingArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.actions_iter()), BTreeSet::from_iter(0..5));
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ThompsonSamplingArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ThompsonSamplingArms::new(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let mut arms = ThompsonSamplingArms::new(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let mut arms = ThompsonSamplingArms::new(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ThompsonSamplingArms::new(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = ThompsonSamplingArms::new(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: ThompsonSamplingArms<i32, f64, Normal> = serde_json::from_str(&json).unwrap();
                }
            }
        }

        mod ucb_1 {
            mod bernoulli {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::arms::Bernoulli,
                    values::{StateActionValue, UCB1Arms},
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1Arms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.actions_iter()), BTreeSet::from_iter(0..5));
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1Arms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1Arms::new(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let mut arms = UCB1Arms::new(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let mut arms = UCB1Arms::new(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1Arms::new(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1Arms::new(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: UCB1Arms<i32, f64, Bernoulli> = serde_json::from_str(&json).unwrap();
                }
            }

            mod normal {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::arms::Normal,
                    values::{StateActionValue, UCB1Arms},
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1Arms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.actions_iter()), BTreeSet::from_iter(0..5));
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1Arms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1Arms::new(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let mut arms = UCB1Arms::new(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let mut arms = UCB1Arms::new(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1Arms::new(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1Arms::new(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: UCB1Arms<i32, f64, Normal> = serde_json::from_str(&json).unwrap();
                }
            }
        }

        mod ucb_1_normal {
            mod bernoulli {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::arms::Bernoulli,
                    values::{StateActionValue, UCB1NormalArms},
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1NormalArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.actions_iter()), BTreeSet::from_iter(0..5));
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1NormalArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1NormalArms::new(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let mut arms = UCB1NormalArms::new(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let mut arms = UCB1NormalArms::new(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1NormalArms::new(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::new(1., 1.)));
                    let arms = UCB1NormalArms::new(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: UCB1NormalArms<i32, f64, Bernoulli> = serde_json::from_str(&json).unwrap();
                }
            }

            mod normal {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::arms::Normal,
                    values::{StateActionValue, UCB1NormalArms},
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1NormalArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.actions_iter()), BTreeSet::from_iter(0..5));
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1NormalArms::new(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1NormalArms::new(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let mut arms = UCB1NormalArms::new(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let mut arms = UCB1NormalArms::new(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1NormalArms::new(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Normal::new()));
                    let arms = UCB1NormalArms::new(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: UCB1NormalArms<i32, f64, Normal> = serde_json::from_str(&json).unwrap();
                }
            }
        }
    }
}
