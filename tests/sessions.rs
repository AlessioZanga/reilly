mod sessions {
    mod train_test {
        use std::fs::File;

        use polars::prelude::*;
        use rand::SeedableRng;
        use rand_distr::Normal;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use rayon::prelude::*;
        use reilly::{
            agents::{
                bandits::{arms, MultiArmedBandit, UCB1NormalArms},
                Agent,
            },
            envs::{Env, FarWest},
            policies::EpsilonDecayGreedy,
            sessions::{Session, TrainTest},
        };

        mod call {
            use std::fs::File;

            use polars::prelude::*;
            use rand::SeedableRng;
            use rand_distr::Normal;
            use rand_xoshiro::Xoshiro256PlusPlus;
            use reilly::{
                agents::{
                    bandits::{arms, MultiArmedBandit, UCB1NormalArms},
                    Agent,
                },
                envs::{Env, FarWest},
                policies::EpsilonDecayGreedy,
                sessions::{Session, TrainTest},
            };

            #[test]
            fn mab() {
                // Initialize the random number generator.
                let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
                // Initialize the environment.
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
                // Initialize the agent.
                let agent = env.actions_iter().map(|a| (a, arms::Bernoulli::new(1., 1.)));
                let mut agent = MultiArmedBandit::new(
                    // Construct a action value function.
                    UCB1NormalArms::new(agent),
                    // Initialize an epsilon-greedy policy.
                    EpsilonDecayGreedy::new(0.99, 0.999, 0.01),
                );
                // Execute the experiment session.
                let session = TrainTest::new(100, 30, 100).with_steps_max(500);
                let mut data = session.call(&mut agent, &mut env, &mut rng);
                // Write data to CSV.
                let mut file = File::create("tests/out-train_test-call-mab.csv").unwrap();
                CsvWriter::new(&mut file).has_header(true).finish(&mut data).unwrap();
            }

            mod monte_carlo {
                use std::fs::File;

                use polars::prelude::*;
                use rand::SeedableRng;
                use rand_xoshiro::Xoshiro256PlusPlus;
                use reilly::{
                    agents::{
                        montecarlo::{EveryVisit, FirstVisit, MonteCarlo},
                        Agent,
                    },
                    envs::{Env, Taxi},
                    policies::{EpsilonDecayGreedy, EpsilonGreedy},
                    sessions::{Session, TrainTest},
                };

                #[test]
                fn first_visit() {
                    // Initialize the random number generator.
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
                    // Initialize the environment.
                    let mut env = Taxi::new();
                    // Initialize the agent.
                    let mut agent = MonteCarlo::new(
                        FirstVisit::new(env.actions_iter(), env.states_iter(), 0.9),
                        EpsilonGreedy::new(0.4),
                    );
                    // Execute the experiment session.
                    let session = TrainTest::new(100, 10, 500).with_steps_max(500);
                    let mut data = session.call(&mut agent, &mut env, &mut rng);
                    // Write data to CSV.
                    let mut file = File::create("tests/out-train_test-call-monte_carlo-first_visit.csv").unwrap();
                    CsvWriter::new(&mut file).has_header(true).finish(&mut data).unwrap();
                }

                #[test]
                fn every_visit() {
                    // Initialize the random number generator.
                    let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();
                    // Initialize the environment.
                    let mut env = Taxi::new();
                    // Initialize the agent.
                    let mut agent = MonteCarlo::new(
                        EveryVisit::new(env.actions_iter(), env.states_iter(), 0.9),
                        EpsilonDecayGreedy::new(0.99, 0.9999, 0.01),
                    );
                    // Execute the experiment session with maximum number of steps per episode set to 500.
                    let session = TrainTest::new(100, 10, 500).with_steps_max(500);
                    let mut data = session.call(&mut agent, &mut env, &mut rng);
                    // Write data to CSV.
                    let mut file = File::create("tests/out-train_test-call-monte_carlo-every_visit.csv").unwrap();
                    CsvWriter::new(&mut file).has_header(true).finish(&mut data).unwrap();
                }
            }
        }

        #[test]
        fn par_call() {
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
            let env = FarWest::new(env);
            // Initialize the MABs.
            let mabs: Vec<_> = env.actions_iter().map(|a| (a, arms::Bernoulli::new(1., 1.))).collect();
            let mabs = [0.025, 0.050, 0.075, 0.10, 0.15, 0.20, 0.25].into_iter().map(|e| {
                MultiArmedBandit::new(
                    // Construct a action value function.
                    UCB1NormalArms::new(mabs.clone().into_iter()),
                    // Initialize an epsilon-greedy policy.
                    EpsilonDecayGreedy::new(e, 0.999, 0.01),
                )
            });
            // Pair each agent with its environment.
            let mut mabs_envs: Vec<_> = mabs.zip(std::iter::repeat(env)).collect();
            // Execute the experiment session.
            let session = TrainTest::new(10, 3, 100).with_steps_max(500);
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
