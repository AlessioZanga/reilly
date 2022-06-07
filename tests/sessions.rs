mod sessions {
    mod train_test {
        use std::fs::File;

        use polars::{io::SerWriter, prelude::CsvWriter};
        use rand::SeedableRng;
        use rand_distr::Normal;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use rayon::prelude::*;
        use reilly::{
            agents::{
                bandits::{arms, MultiArmedBandit},
                Agent,
            },
            envs::{Env, FarWest},
            policies::EpsilonDecayGreedy,
            sessions::{Session, TrainTest},
            values::UCB1NormalArms,
        };

        #[test]
        fn call() {
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
            // Initialize the MAB.
            let mab = env.actions_iter().map(|a| (a, arms::Normal::new()));
            let mut mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonDecayGreedy::default(),
                // Construct a action value function.
                UCB1NormalArms::new(mab),
            );
            // Execute the experiment session.
            let session = TrainTest::new(10, 3, 100);
            let mut data = session.call(&mut mab, &mut env, &mut rng);
            // Write data to CSV.
            let mut file = File::create("tests/out-train_test-call.csv").unwrap();
            CsvWriter::new(&mut file).has_header(true).finish(&mut data).unwrap();
        }

        #[test]
        fn par_call() {
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
            let env = FarWest::new(env, 1_000);
            // Initialize the MABs.
            let mabs: Vec<_> = env.actions_iter().map(|a| (a, arms::Normal::new())).collect();
            let mabs = [0.025, 0.050, 0.075, 0.10, 0.15, 0.20, 0.25].into_iter().map(|e| {
                MultiArmedBandit::new(
                    // Initialize an epsilon-greedy policy.
                    EpsilonDecayGreedy::new(e, 0.999, 0.01),
                    // Construct a action value function.
                    UCB1NormalArms::new(mabs.clone().into_iter()),
                )
            });
            // Pair each agent with its environment.
            let mut mabs_envs: Vec<_> = mabs.zip(std::iter::repeat(env)).collect();
            // Execute the experiment session.
            let session = TrainTest::new(10, 10, 100);
            let mut data = session.par_call(mabs_envs.par_iter_mut(), &mut rng);
            // Write data to CSV.
            let mut file = File::create("tests/out-train_test-par_call.csv").unwrap();
            CsvWriter::new(&mut file).has_header(true).finish(&mut data).unwrap();
        }

        #[test]
        fn serialize() {
            let session = TrainTest::new(10, 3, 100);

            serde_json::to_string(&session).unwrap();
        }

        #[test]
        fn deserialize() {
            let session = TrainTest::new(10, 3, 100);

            let json = serde_json::to_string(&session).unwrap();
            let _: TrainTest = serde_json::from_str(&json).unwrap();
        }
    }
}
