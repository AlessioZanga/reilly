mod sessions {
    mod train_test {
        use std::fs::File;

        use polars::{io::SerWriter, prelude::CsvWriter};
        use rand::SeedableRng;
        use rand_distr::Normal;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use reilly::{
            agents::{
                bandits::{arms::Bernoulli, Arms, MultiArmedBandit},
                Agent,
            },
            envs::{Env, FarWest},
            policies::EpsilonDecayGreedy,
            sessions::{Session, TrainTest},
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
            let mab = env.actions_iter().map(|&a| (a, Bernoulli::default()));
            let mut mab = MultiArmedBandit::new(
                // Initialize an epsilon-greedy policy.
                EpsilonDecayGreedy::default(),
                // Construct a action value function.
                Arms::from_actions_arms_iter(mab),
            );
            // Execute the experiment session.
            let session = TrainTest::new(10, 3, 500);
            let mut data = session.call(&mut mab, &mut env, &mut rng);
            // Write data to CSV.
            let mut file = File::create("tests/out.csv").unwrap();
            CsvWriter::new(&mut file).has_header(true).finish(&mut data).unwrap();
        }

        #[test]
        fn serialize() {
            let session = TrainTest::new(10, 3, 500);

            serde_json::to_string(&session).unwrap();
        }

        #[test]
        fn deserialize() {
            let session = TrainTest::new(10, 3, 500);

            let json = serde_json::to_string(&session).unwrap();
            let _: TrainTest = serde_json::from_str(&json).unwrap();
        }
    }
}
