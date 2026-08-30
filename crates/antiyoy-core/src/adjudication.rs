use crate::{Game, PlayerId};

pub fn adjudicate(game: &Game) -> Option<PlayerId> {
    let mut scores = vec![0_i64; usize::from(game.player_count())];
    for cell in game.cells().iter().copied() {
        if cell.owner().is_neutral() {
            continue;
        }
        let value = 100 + i64::from(cell.unit().strength()) * 10;
        let score = &mut scores[cell.owner().index()];
        *score = score.saturating_add(value);
    }
    for province in game.provinces() {
        let score = &mut scores[province.owner().index()];
        *score = score.saturating_add(province.money());
    }
    let maximum = scores.iter().copied().max()?;
    let mut leaders = scores
        .iter()
        .enumerate()
        .filter(|(_, score)| **score == maximum);
    let (leader, _) = leaders.next()?;
    if leaders.next().is_some() {
        return None;
    }
    Some(PlayerId(
        u8::try_from(leader).expect("player count is bounded by u8"),
    ))
}

#[cfg(test)]
mod tests {
    use crate::{GeneratorConfig, Rules};

    use super::adjudicate;

    #[test]
    fn adjudication_handles_every_multiplayer_seat() {
        let mut scenario = GeneratorConfig {
            width: 15,
            height: 11,
            players: 4,
            seed: 47,
            ..GeneratorConfig::default()
        }
        .generate()
        .expect("valid procedural scenario");
        let fourth_player_cell = scenario
            .cells
            .iter()
            .position(|cell| cell.owner.0 == 3)
            .expect("fourth player start");
        scenario.cells[fourth_player_cell].unit_strength = 4;
        let game = crate::Game::new(Rules::online_default_v1(), scenario).expect("valid game");

        let winner = adjudicate(&game).expect("fourth player has the unique score lead");

        assert_eq!(winner.0, 3);
    }
}
