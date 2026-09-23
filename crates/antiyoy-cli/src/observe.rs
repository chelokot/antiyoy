use antiyoy_core::Game;
use antiyoy_rl::BatchObservation;

pub(super) fn observe_games(games: &[&Game]) -> BatchObservation {
    let legal_actions = games
        .iter()
        .map(|game| {
            let mut actions = Vec::new();
            game.legal_actions(&mut actions);
            actions
        })
        .collect::<Vec<_>>();
    let observed = games
        .iter()
        .zip(&legal_actions)
        .map(|(&game, actions)| (game, actions.as_slice()))
        .collect::<Vec<_>>();
    let mut observation = BatchObservation::default();
    observation.observe_games(&observed, false);
    observation
}
