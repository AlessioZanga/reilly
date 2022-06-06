mod envs {
    mod far_west {
        use std::collections::BTreeSet;

        use rand::SeedableRng;
        use rand_distr::Normal;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::envs::{Env, FarWest};

        #[test]
        fn actions_iter() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 2.),
                Normal::new(1., 6.),
                Normal::new(9., 4.),
                Normal::new(7., 3.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env, 1_000);

            assert_eq!(
                BTreeSet::from_iter(env.actions_iter()),
                BTreeSet::from_iter(vec![0, 1, 2, 3, 4].iter()),
            );
        }

        #[test]
        fn states_iter() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 2.),
                Normal::new(1., 6.),
                Normal::new(9., 4.),
                Normal::new(7., 3.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env, 1_000);

            assert!(env.states_iter().eq([&()].into_iter()));
        }

        #[test]
        fn get_state() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 2.),
                Normal::new(1., 6.),
                Normal::new(9., 4.),
                Normal::new(7., 3.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env, 1_000);

            env.get_state();
        }

        #[test]
        fn call_mut() {
            // Initialize the random number generator.
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            // Initialize the env.
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 2.),
                Normal::new(1., 6.),
                Normal::new(9., 4.),
                Normal::new(7., 3.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let mut env = FarWest::new(env, 1_000);

            env.call_mut(&0, &mut rng);
        }

        #[test]
        fn reset() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 2.),
                Normal::new(1., 6.),
                Normal::new(9., 4.),
                Normal::new(7., 3.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let mut env = FarWest::new(env, 1_000);

            env.reset(&mut rng);
        }

        #[test]
        #[ignore = "rand_distr are not serializable yet"]
        // TODO:
        fn serialize() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 2.),
                Normal::new(1., 6.),
                Normal::new(9., 4.),
                Normal::new(7., 3.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let _env = FarWest::new(env, 1_000);

            // FIXME: serde_json::to_string(&env).unwrap();
        }

        #[test]
        #[ignore = "rand_distr are not serializable yet"]
        // TODO:
        fn deserialize() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 2.),
                Normal::new(1., 6.),
                Normal::new(9., 4.),
                Normal::new(7., 3.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let _env = FarWest::new(env, 1_000);

            // FIXME: let json = serde_json::to_string(&env).unwrap();
            // FIXME: let _: FarWest<Normal<f64>> = serde_json::from_str(&json).unwrap();
        }
    }
}
