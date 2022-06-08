mod agents {
    mod mab {
        use std::collections::BTreeSet;

        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::{
            agents::{
                bandits::{arms::Bernoulli, ExpectedValueArms, MultiArmedBandit},
                Agent,
            },
            policies::EpsilonGreedy,
        };

        #[test]
        fn new() {
            let actions = vec![0, 1, 2, 3, 4];

            let mab = actions.iter().map(|&a| (a, Bernoulli::new(1., 1.)));
            let _ = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonGreedy::new(0.10),
                // Construct a action value function.
                ExpectedValueArms::new(mab),
            );
        }

        #[test]
        fn actions_iter() {
            let actions = vec![0, 1, 2, 3, 4];

            let mab = actions.iter().map(|&a| (a, Bernoulli::new(1., 1.)));
            let mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonGreedy::new(0.10),
                // Construct a action value function.
                ExpectedValueArms::new(mab),
            );

            assert_eq!(BTreeSet::from_iter(mab.actions_iter()), BTreeSet::from_iter(actions),);
        }

        #[test]
        fn states_iter() {
            let actions = vec![0, 1, 2, 3, 4];

            let mab = actions.iter().map(|&a| (a, Bernoulli::new(1., 1.)));
            let mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonGreedy::new(0.10),
                // Construct a action value function.
                ExpectedValueArms::new(mab),
            );

            assert!(mab.states_iter().eq([()].into_iter()));
        }

        #[test]
        fn call() {
            let actions = vec![0, 1, 2, 3, 4];

            let mab = actions.iter().map(|&a| (a, Bernoulli::new(1., 1.)));
            let mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonGreedy::new(0.10),
                // Construct a action value function.
                ExpectedValueArms::new(mab),
            );

            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let action = mab.call((), &mut rng);

            assert!(mab.actions_iter().any(|a| a == action));
        }

        #[test]
        fn reset() {
            let actions = vec![0, 1, 2, 3, 4];

            let mab = actions.iter().map(|&a| (a, Bernoulli::new(1., 1.)));
            let mut mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonGreedy::new(0.10),
                // Construct a action value function.
                ExpectedValueArms::new(mab),
            );

            mab.reset();
        }

        #[test]
        fn update() {
            let actions = vec![0, 1, 2, 3, 4];

            let mab = actions.iter().map(|&a| (a, Bernoulli::new(1., 1.)));
            let mut mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonGreedy::new(0.10),
                // Construct a action value function.
                ExpectedValueArms::new(mab),
            );

            mab.update(actions[0], 0., (), false);
        }

        #[test]
        fn serialize() {
            let actions = vec![0, 1, 2, 3, 4];

            let mab = actions.iter().map(|&a| (a, Bernoulli::new(1., 1.)));
            let mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonGreedy::new(0.10),
                // Construct a action value function.
                ExpectedValueArms::new(mab),
            );

            serde_json::to_string(&mab).unwrap();
        }

        #[test]
        fn deserialize() {
            let actions = vec![0, 1, 2, 3, 4];

            let mab = actions.iter().map(|&a| (a, Bernoulli::new(1., 1.)));
            let mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonGreedy::new(0.10),
                // Construct a action value function.
                ExpectedValueArms::new(mab),
            );

            let json = serde_json::to_string(&mab).unwrap();
            let _: MultiArmedBandit<i32, f64, (), EpsilonGreedy, ExpectedValueArms<i32, f64, Bernoulli>> =
                serde_json::from_str(&json).unwrap();
        }
    }
}
