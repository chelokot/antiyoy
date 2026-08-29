#![forbid(unsafe_code)]

use antiyoy_agents::Agent;
use antiyoy_core::{Game, PlayerId, Rules, Scenario};
use antiyoy_protocol::{Replay, ReplayError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Termination {
    Victory,
    ActionLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MatchOutcome {
    pub winner: Option<PlayerId>,
    pub actions: u32,
    pub termination: Termination,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchReport {
    pub agents: [String; 2],
    pub seed: u64,
    pub outcome: MatchOutcome,
    pub replay: Replay,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rating {
    pub elo: f64,
    pub games: u64,
}

impl Default for Rating {
    fn default() -> Self {
        Self {
            elo: 1_000.0,
            games: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Elo {
    pub k_factor: f64,
}

impl Default for Elo {
    fn default() -> Self {
        Self { k_factor: 32.0 }
    }
}

impl Elo {
    pub fn update(self, first: &mut Rating, second: &mut Rating, first_score: f64) {
        let expected = 1.0 / (1.0 + 10.0_f64.powf((second.elo - first.elo) / 400.0));
        let delta = self.k_factor * (first_score - expected);
        first.elo += delta;
        second.elo -= delta;
        first.games += 1;
        second.games += 1;
    }
}

#[derive(Debug, Error)]
pub enum MatchError {
    #[error("map configuration failed: {0}")]
    InvalidMap(#[from] antiyoy_core::ConfigError),
    #[error("replay failed: {0}")]
    Replay(#[from] ReplayError),
}

pub fn symmetric_duel(width: u16, height: u16, seed: u64) -> Result<Scenario, MatchError> {
    Scenario::symmetric_duel(width, height, seed).map_err(MatchError::InvalidMap)
}

pub fn run_match<First: Agent, Second: Agent>(
    rules: Rules,
    scenario: Scenario,
    first: &mut First,
    second: &mut Second,
    action_limit: u32,
) -> Result<MatchReport, MatchError> {
    let seed = scenario.seed;
    let names = [first.name().to_owned(), second.name().to_owned()];
    let (mut replay, mut game) = Replay::new(rules, scenario)?;
    let mut legal_actions = Vec::new();

    for action_index in 0..action_limit {
        game.legal_actions(&mut legal_actions);
        let action = if game.active_player() == PlayerId(0) {
            first.select_action(&game, &legal_actions)
        } else {
            second.select_action(&game, &legal_actions)
        };
        let transition = replay.record(&mut game, action)?;
        if transition.terminal {
            return Ok(MatchReport {
                agents: names,
                seed,
                outcome: MatchOutcome {
                    winner: transition.winner,
                    actions: action_index + 1,
                    termination: Termination::Victory,
                },
                replay,
            });
        }
    }

    Ok(MatchReport {
        agents: names,
        seed,
        outcome: MatchOutcome {
            winner: adjudicate(&game),
            actions: action_limit,
            termination: Termination::ActionLimit,
        },
        replay,
    })
}

fn adjudicate(game: &Game) -> Option<PlayerId> {
    let mut scores = [0_i64; 2];
    for cell in game.cells().iter().copied() {
        if !cell.owner().is_neutral() {
            scores[cell.owner().index()] += 100 + i64::from(cell.unit().strength()) * 10;
        }
    }
    for province in game.provinces() {
        scores[province.owner().index()] += province.money();
    }
    match scores[0].cmp(&scores[1]) {
        std::cmp::Ordering::Greater => Some(PlayerId(0)),
        std::cmp::Ordering::Less => Some(PlayerId(1)),
        std::cmp::Ordering::Equal => None,
    }
}

pub fn score_for_first(outcome: MatchOutcome) -> f64 {
    match outcome.winner {
        Some(PlayerId(0)) => 1.0,
        Some(PlayerId(1)) => 0.0,
        _ => 0.5,
    }
}

#[cfg(test)]
mod tests {
    use antiyoy_agents::{GreedyAgent, RandomAgent};
    use antiyoy_core::Rules;

    use super::{Elo, Rating, run_match, score_for_first, symmetric_duel};

    #[test]
    fn rating_update_is_zero_sum() {
        let mut first = Rating::default();
        let mut second = Rating::default();
        Elo::default().update(&mut first, &mut second, 1.0);
        assert!((first.elo - 1_016.0).abs() < f64::EPSILON);
        assert!((second.elo - 984.0).abs() < f64::EPSILON);
        assert!((first.elo + second.elo - 2_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn seeded_match_produces_verifiable_replay() {
        let scenario = symmetric_duel(7, 5, 47).expect("valid duel");
        let mut first = GreedyAgent::new("greedy");
        let mut second = RandomAgent::new("random", 48);
        let report = run_match(
            Rules::classic_generic(),
            scenario,
            &mut first,
            &mut second,
            500,
        )
        .expect("valid match");
        report.replay.verify().expect("verifiable replay");
        assert!((0.0..=1.0).contains(&score_for_first(report.outcome)));
    }
}
