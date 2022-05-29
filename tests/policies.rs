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
            // Initialize the random number generator.
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let pi: Greedy = Default::default();
            let v = Arms::<usize, f64, Bernoulli>::from_actions_arms_iter([].into_iter());

            pi.call(&v, &(), &mut rng);
        }

        #[test]
        #[ignore]
        // TODO:
        fn reset() {
            todo!()
        }

        #[test]
        #[ignore]
        // TODO:
        fn serialize() {
            todo!()
        }

        #[test]
        #[ignore]
        // TODO:
        fn deserialize() {
            todo!()
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
            // Initialize the random number generator.
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            // [(a, (alpha, beta))]
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
            // Initialize the random number generator.
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let pi: Random = Default::default();
            let v = Arms::<usize, f64, Bernoulli>::from_actions_arms_iter([].into_iter());

            pi.call(&v, &(), &mut rng);
        }

        #[test]
        #[ignore]
        // TODO:
        fn reset() {
            todo!()
        }

        #[test]
        #[ignore]
        // TODO:
        fn serialize() {
            todo!()
        }

        #[test]
        #[ignore]
        // TODO:
        fn deserialize() {
            todo!()
        }
    }
}
