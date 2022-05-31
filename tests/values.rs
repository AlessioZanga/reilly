mod values {
    mod arms {
        use std::collections::BTreeSet;

        use reilly::{
            agents::bandits::{arms::Bernoulli, Arms},
            values::StateActionValue,
        };

        #[test]
        fn actions_iter() {
            let arms = (0..5).map(|a| (a, Bernoulli::default()));
            let arms = Arms::from_actions_arms_iter(arms);

            assert_eq!(
                BTreeSet::from_iter(arms.actions_iter().copied()),
                BTreeSet::from_iter(0..5)
            );
        }

        #[test]
        fn states_iter() {
            let arms = (0..5).map(|a| (a, Bernoulli::default()));
            let arms = Arms::from_actions_arms_iter(arms);

            assert_eq!(BTreeSet::from_iter(arms.states_iter()), BTreeSet::from_iter([&()]));
        }

        #[test]
        fn call() {
            let arms = (0..5).map(|a| (a, Bernoulli::default()));
            let arms = Arms::from_actions_arms_iter(arms);

            arms.call(&0, &());
        }

        #[test]
        fn reset() {
            let arms = (0..5).map(|a| (a, Bernoulli::default()));
            let mut arms = Arms::from_actions_arms_iter(arms);

            arms.reset();
        }

        #[test]
        fn update() {
            let arms = (0..5).map(|a| (a, Bernoulli::default()));
            let mut arms = Arms::from_actions_arms_iter(arms);

            arms.update(&0, &0., &(), true);
        }

        #[test]
        fn serialize() {
            let arms = (0..5).map(|a| (a, Bernoulli::default()));
            let arms = Arms::from_actions_arms_iter(arms);

            serde_json::to_string(&arms).unwrap();
        }

        #[test]
        fn deserialize() {
            let arms = (0..5).map(|a| (a, Bernoulli::default()));
            let arms = Arms::from_actions_arms_iter(arms);

            let json = serde_json::to_string(&arms).unwrap();
            let _: Arms<i32, f64, Bernoulli> = serde_json::from_str(&json).unwrap();
        }
    }
}
