mod envs {
    mod far_west {
        use std::collections::BTreeSet;

        use rand::SeedableRng;
        use rand_distr::Normal;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::envs::{Env, FarWest};

        #[test]
        fn clone() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env);

            let _ = env.clone();
        }

        #[test]
        fn debug() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env);

            println!("{:?}", env);
        }

        #[test]
        fn display() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env);

            println!("{}", env);
        }

        #[test]
        fn actions_iter() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env);

            assert_eq!(
                BTreeSet::from_iter(env.actions_iter()),
                BTreeSet::from_iter(vec![0, 1, 2, 3, 4]),
            );
        }

        #[test]
        fn states_iter() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env);

            assert!(env.states_iter().eq([()].into_iter()));
        }

        #[test]
        fn state() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let env = FarWest::new(env);

            env.state();
        }

        #[test]
        fn call_mut() {
            // Initialize the random number generator.
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            // Initialize the env.
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let mut env = FarWest::new(env);

            env.call_mut(0, &mut rng);
        }

        #[test]
        fn reset() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let mut env = FarWest::new(env);

            env.reset(&mut rng);
        }

        #[test]
        #[ignore = "rand_distr are not serializable yet"]
        // TODO:
        fn serialize() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let _env = FarWest::new(env);

            // FIXME: serde_json::to_string(&env).unwrap();
        }

        #[test]
        #[ignore = "rand_distr are not serializable yet"]
        // TODO:
        fn deserialize() {
            let env = [
                Normal::new(0., 1.),
                Normal::new(5., 1.),
                Normal::new(1., 1.),
                Normal::new(9., 1.),
                Normal::new(7., 1.),
            ]
            .into_iter()
            .map(|d| d.unwrap());
            let _env = FarWest::new(env);

            // FIXME: let json = serde_json::to_string(&env).unwrap();
            // FIXME: let _: FarWest<Normal<f64>> = serde_json::from_str(&json).unwrap();
        }
    }

    mod taxi {
        use std::collections::BTreeSet;

        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::envs::{Env, Taxi, TaxiWithDisplay};

        #[test]
        fn clone() {
            let env = Taxi::new();

            let _ = env.clone();
        }

        #[test]
        fn debug() {
            let env = Taxi::new();

            println!("{:?}", env);
        }

        #[test]
        fn display() {
            let env = Taxi::new();

            println!("{}", env);
        }

        #[test]
        fn actions_iter() {
            let env = Taxi::new();

            assert_eq!(
                BTreeSet::from_iter(env.actions_iter()),
                BTreeSet::from_iter(0..Taxi::ACTIONS),
            );
        }

        #[test]
        fn states_iter() {
            let env = Taxi::new();

            assert_eq!(
                BTreeSet::from_iter(env.states_iter()),
                BTreeSet::from_iter(0..Taxi::STATES),
            );
        }

        #[test]
        fn state() {
            let env = Taxi::new();

            assert_eq!(env.state(), 0);
        }

        #[test]
        fn call_mut() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = Taxi::new();

            env.call_mut(0, &mut rng);
        }

        #[test]
        fn reset() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = Taxi::new();

            env.reset(&mut rng);
        }

        #[test]
        fn serialize() {
            let env = Taxi::new();

            serde_json::to_string(&env).unwrap();
        }

        #[test]
        fn deserialize() {
            let env = Taxi::new();

            let json = serde_json::to_string(&env).unwrap();
            let _: Taxi = serde_json::from_str(&json).unwrap();
        }

        #[test]
        fn render() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = TaxiWithDisplay::new();

            env.call_mut(0, &mut rng);
        }
    }

    mod frozen_lake_noslippery {
        use std::collections::BTreeSet;

        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::envs::{Env, FrozenLake4x4, FrozenLake4x4WithDisplay};

        #[test]
        fn clone() {
            let env = FrozenLake4x4::new();

            let _ = env.clone();
        }

        #[test]
        fn debug() {
            let env = FrozenLake4x4::new();

            println!("{:?}", env);
        }

        #[test]
        fn display() {
            let env = FrozenLake4x4::new();

            println!("{}", env);
        }

        #[test]
        fn actions_iter() {
            let env = FrozenLake4x4::new();

            assert_eq!(
                BTreeSet::from_iter(env.actions_iter()),
                BTreeSet::from_iter(0..FrozenLake4x4::ACTIONS),
            );
        }

        #[test]
        fn states_iter() {
            let env = FrozenLake4x4::new();

            assert_eq!(
                BTreeSet::from_iter(env.states_iter()),
                BTreeSet::from_iter(0..FrozenLake4x4::STATES),
            );
        }

        #[test]
        fn state() {
            let env = FrozenLake4x4::new();

            assert_eq!(env.state(), 0);
        }

        #[test]
        fn call_mut() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = FrozenLake4x4::new();

            env.call_mut(0, &mut rng);
        }

        #[test]
        fn reset() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = FrozenLake4x4::new();

            env.reset(&mut rng);
        }

        #[test]
        fn serialize() {
            let env = FrozenLake4x4::new();

            serde_json::to_string(&env).unwrap();
        }

        #[test]
        fn deserialize() {
            let env = FrozenLake4x4::new();

            let json = serde_json::to_string(&env).unwrap();
            let _: FrozenLake4x4 = serde_json::from_str(&json).unwrap();
        }

        #[test]
        fn render() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = FrozenLake4x4WithDisplay::new();

            env.call_mut(0, &mut rng);
        }
    }

    mod frozen_lake_slippery {
        use std::collections::BTreeSet;

        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::envs::{Env, FrozenLake4x4Slippery, FrozenLake4x4SlipperyWithDisplay};

        #[test]
        fn clone() {
            let env = FrozenLake4x4Slippery::new();

            let _ = env.clone();
        }

        #[test]
        fn debug() {
            let env = FrozenLake4x4Slippery::new();

            println!("{:?}", env);
        }

        #[test]
        fn display() {
            let env = FrozenLake4x4Slippery::new();

            println!("{}", env);
        }

        #[test]
        fn actions_iter() {
            let env = FrozenLake4x4Slippery::new();

            assert_eq!(
                BTreeSet::from_iter(env.actions_iter()),
                BTreeSet::from_iter(0..FrozenLake4x4Slippery::ACTIONS),
            );
        }

        #[test]
        fn states_iter() {
            let env = FrozenLake4x4Slippery::new();

            assert_eq!(
                BTreeSet::from_iter(env.states_iter()),
                BTreeSet::from_iter(0..FrozenLake4x4Slippery::STATES),
            );
        }

        #[test]
        fn state() {
            let env = FrozenLake4x4Slippery::new();

            assert_eq!(env.state(), 0);
        }

        #[test]
        fn call_mut() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = FrozenLake4x4Slippery::new();

            env.call_mut(0, &mut rng);
        }

        #[test]
        fn reset() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = FrozenLake4x4Slippery::new();

            env.reset(&mut rng);
        }

        #[test]
        fn serialize() {
            let env = FrozenLake4x4Slippery::new();

            serde_json::to_string(&env).unwrap();
        }

        #[test]
        fn deserialize() {
            let env = FrozenLake4x4Slippery::new();

            let json = serde_json::to_string(&env).unwrap();
            let _: FrozenLake4x4Slippery = serde_json::from_str(&json).unwrap();
        }

        #[test]
        fn render() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = FrozenLake4x4SlipperyWithDisplay::new();

            env.call_mut(0, &mut rng);
        }
    }

    mod blackjack {
        use std::collections::BTreeSet;

        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::envs::{Blackjack, Env};

        #[test]
        fn clone() {
            let env = Blackjack::new();

            let _ = env.clone();
        }

        #[test]
        fn debug() {
            let env = Blackjack::new();

            println!("{:?}", env);
        }

        #[test]
        fn display() {
            let env = Blackjack::new();

            println!("{}", env);
        }

        #[test]
        fn actions_iter() {
            let env = Blackjack::new();

            assert_eq!(
                BTreeSet::from_iter(env.actions_iter()),
                BTreeSet::from_iter(0..Blackjack::ACTIONS),
            );
        }

        #[test]
        fn states_iter() {
            let env = Blackjack::new();

            assert_eq!(
                BTreeSet::from_iter(env.states_iter()),
                BTreeSet::from_iter(0..Blackjack::STATES),
            );
        }

        #[test]
        fn state() {
            let env = Blackjack::new();

            assert_eq!(env.state(), 0);
        }

        #[test]
        fn call_mut() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = Blackjack::new();

            env.call_mut(0, &mut rng);
        }

        #[test]
        fn reset() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = Blackjack::new();

            env.reset(&mut rng);
        }

        #[test]
        fn serialize() {
            let env = Blackjack::new();

            serde_json::to_string(&env).unwrap();
        }

        #[test]
        fn deserialize() {
            let env = Blackjack::new();

            let json = serde_json::to_string(&env).unwrap();
            let _: Blackjack = serde_json::from_str(&json).unwrap();
        }
    }

    mod cliff_walking {
        use std::collections::BTreeSet;

        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::envs::{CliffWalking, CliffWalkingWithDisplay, Env};

        #[test]
        fn clone() {
            let env = CliffWalking::new();

            let _ = env.clone();
        }

        #[test]
        fn debug() {
            let env = CliffWalking::new();

            println!("{:?}", env);
        }

        #[test]
        fn display() {
            let env = CliffWalking::new();

            println!("{}", env);
        }

        #[test]
        fn actions_iter() {
            let env = CliffWalking::new();

            assert_eq!(
                BTreeSet::from_iter(env.actions_iter()),
                BTreeSet::from_iter(0..CliffWalking::ACTIONS),
            );
        }

        #[test]
        fn states_iter() {
            let env = CliffWalking::new();

            assert_eq!(
                BTreeSet::from_iter(env.states_iter()),
                BTreeSet::from_iter(0..CliffWalking::STATES),
            );
        }

        #[test]
        fn state() {
            let env = CliffWalking::new();

            assert_eq!(env.state(), 36);
        }

        #[test]
        fn call_mut() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = CliffWalking::new();

            env.call_mut(0, &mut rng);
        }

        #[test]
        fn reset() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = CliffWalking::new();

            env.reset(&mut rng);
        }

        #[test]
        fn serialize() {
            let env = CliffWalking::new();

            serde_json::to_string(&env).unwrap();
        }

        #[test]
        fn deserialize() {
            let env = CliffWalking::new();

            let json = serde_json::to_string(&env).unwrap();
            let _: CliffWalking = serde_json::from_str(&json).unwrap();
        }

        #[test]
        fn render() {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
            let mut env = CliffWalkingWithDisplay::new();

            env.call_mut(0, &mut rng);
        }
    }
}
