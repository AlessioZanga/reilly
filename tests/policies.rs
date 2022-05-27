mod policies {
    use std::collections::HashMap;

    use approx::*;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use reilly::{
        agents::bandits::{arms::Bernoulli, Arms},
        policies::{Greedy, Policy, Random},
        values::StateActionValue,
    };

    #[test]
    pub fn greedy() {
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
            let mut pi: Greedy = Default::default();
            let v =
                Arms::from_actions_arms_iter(i.into_iter().map(|(a, (alpha, beta))| (a, Bernoulli::new(alpha, beta))));

            assert_eq!(pi.call_mut(&v, &()), j);
        }
    }

    #[test]
    #[should_panic]
    pub fn greedy_should_panic() {
        let mut pi: Greedy = Default::default();
        let v = Arms::<usize, f64, Bernoulli>::from_actions_arms_iter([].into_iter());

        pi.call_mut(&v, &());
    }

    #[test]
    pub fn random() {
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
            let mut pi: Random<Xoshiro256PlusPlus> = Default::default();
            let v =
                Arms::from_actions_arms_iter(i.into_iter().map(|(a, (alpha, beta))| (a, Bernoulli::new(alpha, beta))));

            let size = 100_000;
            let mut count: HashMap<i32, usize> = Default::default();
            let relative_frequency = 1. / v.actions_iter().len() as f64;

            for _ in 0..size {
                let a = pi.call_mut(&v, &());
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
    pub fn random_should_panic() {
        let mut pi: Random<Xoshiro256PlusPlus> = Default::default();
        let v = Arms::<usize, f64, Bernoulli>::from_actions_arms_iter([].into_iter());

        pi.call_mut(&v, &());
    }
}
