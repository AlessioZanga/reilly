mod values {
    mod arms {
        mod expected_value {
            mod bernoulli {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::{arms::Bernoulli, ExpectedValueArms},
                    values::StateActionValue,
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    assert_eq!(
                        BTreeSet::from_iter(arms.actions_iter().copied()),
                        BTreeSet::from_iter(0..5)
                    );
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([&()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let mut arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let mut arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: ExpectedValueArms<i32, f64, Bernoulli> = serde_json::from_str(&json).unwrap();
                }
            }

            mod sample_average {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::{arms::SampleAverage, ExpectedValueArms},
                    values::StateActionValue,
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    assert_eq!(
                        BTreeSet::from_iter(arms.actions_iter().copied()),
                        BTreeSet::from_iter(0..5)
                    );
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([&()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let mut arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let mut arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ExpectedValueArms::from_actions_arms_iter(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: ExpectedValueArms<i32, f64, SampleAverage> = serde_json::from_str(&json).unwrap();
                }
            }
        }

        mod thompson_sampling {
            mod bernoulli {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::{arms::Bernoulli, ThompsonSamplingArms},
                    values::StateActionValue,
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    assert_eq!(
                        BTreeSet::from_iter(arms.actions_iter().copied()),
                        BTreeSet::from_iter(0..5)
                    );
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([&()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let mut arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let mut arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: ThompsonSamplingArms<i32, f64, Bernoulli> = serde_json::from_str(&json).unwrap();
                }
            }

            mod sample_average {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::{arms::SampleAverage, ThompsonSamplingArms},
                    values::StateActionValue,
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    assert_eq!(
                        BTreeSet::from_iter(arms.actions_iter().copied()),
                        BTreeSet::from_iter(0..5)
                    );
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([&()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let mut arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let mut arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = ThompsonSamplingArms::from_actions_arms_iter(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: ThompsonSamplingArms<i32, f64, SampleAverage> = serde_json::from_str(&json).unwrap();
                }
            }
        }

        mod ucb1 {
            mod bernoulli {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::{arms::Bernoulli, UCB1Arms},
                    values::StateActionValue,
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    assert_eq!(
                        BTreeSet::from_iter(arms.actions_iter().copied()),
                        BTreeSet::from_iter(0..5)
                    );
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([&()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let mut arms = UCB1Arms::from_actions_arms_iter(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let mut arms = UCB1Arms::from_actions_arms_iter(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, Bernoulli::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: UCB1Arms<i32, f64, Bernoulli> = serde_json::from_str(&json).unwrap();
                }
            }

            mod sample_average {
                use std::collections::BTreeSet;

                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::bandits::{arms::SampleAverage, UCB1Arms},
                    values::StateActionValue,
                };

                #[test]
                fn actions_iter() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    assert_eq!(
                        BTreeSet::from_iter(arms.actions_iter().copied()),
                        BTreeSet::from_iter(0..5)
                    );
                }

                #[test]
                fn states_iter() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([&()]));
                }

                #[test]
                fn call() {
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    arms.call(&0, &(), &mut rng);
                }

                #[test]
                fn reset() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let mut arms = UCB1Arms::from_actions_arms_iter(arms);

                    arms.reset();
                }

                #[test]
                fn update() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let mut arms = UCB1Arms::from_actions_arms_iter(arms);

                    arms.update(&0, &0., &(), true);
                }

                #[test]
                fn serialize() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    serde_json::to_string(&arms).unwrap();
                }

                #[test]
                fn deserialize() {
                    let arms = (0..5).map(|a| (a, SampleAverage::default()));
                    let arms = UCB1Arms::from_actions_arms_iter(arms);

                    let json = serde_json::to_string(&arms).unwrap();
                    let _: UCB1Arms<i32, f64, SampleAverage> = serde_json::from_str(&json).unwrap();
                }
            }
        }
    }
}
