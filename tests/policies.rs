mod policies {

    mod greedy {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::{
            agents::bandits::{arms::Bernoulli, Arms},
            policies::{Greedy, Policy},
        };

        #[test]
        fn call() {
            // Initialize the random number generator.
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            // [(a, (alpha, beta))]
            let data = [
                (vec![(0, (1., 1.))], 0),
                (vec![(0, (2., 1.)), (1, (1., 1.))], 0),
                (vec![(0, (1., 1.)), (1, (2., 1.))], 1),
                (vec![(0, (2., 1.)), (1, (4., 2.)), (2, (3., 1.))], 2),
                (vec![(0, (3., 1.)), (1, (1., 1.)), (2, (2., 1.))], 0),
                (vec![(0, (1., 1.)), (1, (3., 1.)), (2, (2., 1.))], 1),
                (vec![(0, (1., 1.)), (1, (2., 1.)), (2, (3., 1.))], 2),
            ];

            for (i, j) in data {
                let pi: Greedy = Default::default();
                let v = Arms::from_actions_arms_iter(
                    i.into_iter().map(|(a, (alpha, beta))| (a, Bernoulli::new(alpha, beta))),
                );

                assert_eq!(pi.call(&v, &(), &mut rng), j);
            }
        }

        #[test]
        #[should_panic]
        fn call_should_panic() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let pi: Greedy = Default::default();
            let v = Arms::<usize, f64, Bernoulli>::from_actions_arms_iter([].into_iter());

            pi.call(&v, &(), &mut rng);
        }

        #[test]
        fn reset() {
            let mut pi: Greedy = Default::default();

            pi.reset();
        }

        #[test]
        fn update() {
            let mut pi = Greedy::new();

            pi.update(&0, &0., &(), true);
        }

        #[test]
        fn serialize() {
            let pi: Greedy = Default::default();

            serde_json::to_string(&pi).unwrap();
        }

        #[test]
        fn deserialize() {
            let pi: Greedy = Default::default();

            let json = serde_json::to_string(&pi).unwrap();
            let _: Greedy = serde_json::from_str(&json).unwrap();
        }
    }

    mod epsilon_greedy {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::{
            agents::bandits::{arms::Bernoulli, Arms},
            policies::{EpsilonGreedy, Policy},
        };

        #[test]
        fn call() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let data = [
                (vec![(0, (1., 1.))], 0),
                (vec![(0, (2., 1.)), (1, (1., 1.))], 0),
                (vec![(0, (1., 1.)), (1, (2., 1.))], 1),
                (vec![(0, (2., 1.)), (1, (4., 2.)), (2, (3., 1.))], 2),
                (vec![(0, (3., 1.)), (1, (1., 1.)), (2, (2., 1.))], 0),
                (vec![(0, (1., 1.)), (1, (3., 1.)), (2, (2., 1.))], 1),
                (vec![(0, (1., 1.)), (1, (2., 1.)), (2, (3., 1.))], 2),
            ];

            for (i, _) in data {
                let pi: EpsilonGreedy = Default::default();
                let v = Arms::from_actions_arms_iter(
                    i.into_iter().map(|(a, (alpha, beta))| (a, Bernoulli::new(alpha, beta))),
                );

                pi.call(&v, &(), &mut rng);
            }
        }

        #[test]
        #[should_panic]
        fn call_should_panic() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let pi = EpsilonGreedy::new(0.10);
            let v = Arms::<usize, f64, Bernoulli>::from_actions_arms_iter([].into_iter());

            pi.call(&v, &(), &mut rng);
        }

        #[test]
        fn reset() {
            let mut pi = EpsilonGreedy::new(0.10);

            pi.reset();
        }

        #[test]
        fn update() {
            let mut pi = EpsilonGreedy::new(0.10);

            pi.update(&0, &0., &(), true);
        }

        #[test]
        fn serialize() {
            let pi = EpsilonGreedy::new(0.10);

            serde_json::to_string(&pi).unwrap();
        }

        #[test]
        fn deserialize() {
            let pi = EpsilonGreedy::new(0.10);

            let json = serde_json::to_string(&pi).unwrap();
            let _: EpsilonGreedy = serde_json::from_str(&json).unwrap();
        }
    }

    mod epsilon_decay_greedy {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::{
            agents::bandits::{arms::Bernoulli, Arms},
            policies::{EpsilonDecayGreedy, Policy},
        };

        #[test]
        fn call() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let pi: EpsilonDecayGreedy = Default::default();
            let v = Arms::from_actions_arms_iter([(0, Bernoulli::new(1., 1.))].into_iter());

            pi.call(&v, &(), &mut rng);
        }

        #[test]
        #[should_panic]
        fn call_should_panic() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let pi: EpsilonDecayGreedy = Default::default();
            let v = Arms::<usize, f64, Bernoulli>::from_actions_arms_iter([].into_iter());

            pi.call(&v, &(), &mut rng);
        }

        #[test]
        fn reset() {
            let mut pi: EpsilonDecayGreedy = Default::default();

            pi.reset();
        }

        #[test]
        fn update() {
            let mut pi: EpsilonDecayGreedy = Default::default();

            pi.update(&0, &0., &(), true);
        }

        #[test]
        fn serialize() {
            let pi: EpsilonDecayGreedy = Default::default();

            serde_json::to_string(&pi).unwrap();
        }

        #[test]
        fn deserialize() {
            let pi: EpsilonDecayGreedy = Default::default();

            let json = serde_json::to_string(&pi).unwrap();
            let _: EpsilonDecayGreedy = serde_json::from_str(&json).unwrap();
        }
    }

    mod random {
        use std::collections::HashMap;

        use approx::*;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::{
            agents::bandits::{arms::Bernoulli, Arms},
            policies::{Policy, Random},
            values::StateActionValue,
        };

        #[test]
        fn call() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let data = [
                vec![(0, (1., 1.))],
                vec![(0, (2., 1.)), (1, (1., 1.))],
                vec![(0, (1., 1.)), (1, (2., 1.))],
                vec![(0, (2., 1.)), (1, (4., 2.)), (2, (3., 1.))],
                vec![(0, (3., 1.)), (1, (1., 1.)), (2, (2., 1.))],
                vec![(0, (1., 1.)), (1, (3., 1.)), (2, (2., 1.))],
                vec![(0, (1., 1.)), (1, (2., 1.)), (2, (3., 1.))],
            ];

            for i in data {
                let pi: Random = Default::default();
                let v = Arms::from_actions_arms_iter(
                    i.into_iter().map(|(a, (alpha, beta))| (a, Bernoulli::new(alpha, beta))),
                );

                let size = 100_000;
                let mut count: HashMap<i32, usize> = Default::default();
                let relative_frequency = 1. / v.actions_iter().len() as f64;

                for _ in 0..size {
                    let a = pi.call(&v, &(), &mut rng);
                    *count.entry(a).or_default() += 1;
                }

                for (_, c) in count {
                    assert_relative_eq!(
                        (c as f64 / size as f64),
                        relative_frequency,
                        max_relative = 0.01,
                        epsilon = 0.01
                    );
                }
            }
        }

        #[test]
        #[should_panic]
        fn call_should_panic() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let pi: Random = Default::default();
            let v = Arms::<usize, f64, Bernoulli>::from_actions_arms_iter([].into_iter());

            pi.call(&v, &(), &mut rng);
        }

        #[test]
        fn reset() {
            let mut pi: Random = Default::default();

            pi.reset();
        }

        #[test]
        fn update() {
            let mut pi = Random::new();

            pi.update(&0, &0., &(), true);
        }

        #[test]
        fn serialize() {
            let pi: Random = Default::default();

            serde_json::to_string(&pi).unwrap();
        }

        #[test]
        fn deserialize() {
            let pi: Random = Default::default();

            let json = serde_json::to_string(&pi).unwrap();
            let _: Random = serde_json::from_str(&json).unwrap();
        }
    }
}
