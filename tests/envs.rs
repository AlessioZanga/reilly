mod envs {
    use rand::SeedableRng;
    use rand_distr::Normal;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use reilly::{
        agents::{
            bandits::{arms::Bernoulli, Arms, MultiArmedBandit},
            Agent,
        },
        envs::FarWest,
        policies::EpsilonGreedy,
        sessions::{Session, TrainTestSession},
    };

    #[test]
    fn far_west() {
        let mab = (0..5).map(|a| (a, Bernoulli::default()));
        let mut mab = MultiArmedBandit::new(
            // Initialize an epsilon-greedy policy.
            EpsilonGreedy::new(0.10),
            // Construct a action value function.
            Arms::from_actions_arms_iter(mab),
        );

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

        let mut rng: Xoshiro256PlusPlus = SeedableRng::from_entropy();

        let session = TrainTestSession::new(1_000, 10, 5);
        session.call(&mut mab, &mut env, &mut rng);
    }
}
