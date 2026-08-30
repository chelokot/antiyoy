#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use antiyoy_agents::Agent;
pub use antiyoy_core::adjudicate;
use antiyoy_core::{Game, PlayerId, Rules, Scenario};
use antiyoy_protocol::{Digest, Replay, ReplayError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const LEAGUE_SCHEMA_VERSION: u16 = 1;

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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Participant {
    pub rating: Rating,
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LeagueMatch {
    pub id: String,
    pub agents: [String; 2],
    pub seed: u64,
    pub outcome: MatchOutcome,
    pub final_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct League {
    pub schema_version: u16,
    pub elo: Elo,
    pub participants: BTreeMap<String, Participant>,
    pub matches: Vec<LeagueMatch>,
}

impl Default for League {
    fn default() -> Self {
        Self {
            schema_version: LEAGUE_SCHEMA_VERSION,
            elo: Elo::default(),
            participants: BTreeMap::new(),
            matches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Standing {
    pub rank: usize,
    pub name: String,
    pub rating: Rating,
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
}

#[derive(Debug, Error)]
pub enum LeagueError {
    #[error("league schema {actual} is unsupported, expected {expected}")]
    UnsupportedSchema { actual: u16, expected: u16 },
    #[error("league Elo K-factor must be finite and positive")]
    InvalidKFactor,
    #[error("league contains duplicate match {0}")]
    DuplicateMatch(String),
    #[error("league standings do not match its ordered match ledger")]
    CorruptStandings,
    #[error("both seats use the same agent identity {0}")]
    IdenticalAgents(String),
    #[error("match seed differs from its replay scenario")]
    SeedMismatch,
    #[error("match reports {reported} actions but replay contains {recorded}")]
    ActionCount { reported: u32, recorded: usize },
    #[error("match termination or winner differs from the replayed state")]
    OutcomeMismatch,
    #[error("replay verification failed: {0}")]
    Replay(#[from] ReplayError),
    #[error("match identity serialization failed: {0}")]
    Serialization(#[from] postcard::Error),
}

impl League {
    pub fn validate(&self) -> Result<(), LeagueError> {
        if self.schema_version != LEAGUE_SCHEMA_VERSION {
            return Err(LeagueError::UnsupportedSchema {
                actual: self.schema_version,
                expected: LEAGUE_SCHEMA_VERSION,
            });
        }
        if !self.elo.k_factor.is_finite() || self.elo.k_factor <= 0.0 {
            return Err(LeagueError::InvalidKFactor);
        }
        let mut match_ids = BTreeSet::new();
        let mut participants = BTreeMap::new();
        for record in &self.matches {
            if !match_ids.insert(&record.id) {
                return Err(LeagueError::DuplicateMatch(record.id.clone()));
            }
            if record.agents[0] == record.agents[1] {
                return Err(LeagueError::IdenticalAgents(record.agents[0].clone()));
            }
            Self::apply_result(
                &mut participants,
                self.elo,
                &record.agents,
                score_for_first(record.outcome),
            );
        }
        if participants != self.participants {
            return Err(LeagueError::CorruptStandings);
        }
        Ok(())
    }

    pub fn record(&mut self, report: &MatchReport) -> Result<LeagueMatch, LeagueError> {
        self.validate()?;
        if report.agents[0] == report.agents[1] {
            return Err(LeagueError::IdenticalAgents(report.agents[0].clone()));
        }
        if report.seed != report.replay.header.scenario.seed {
            return Err(LeagueError::SeedMismatch);
        }
        let verified = report.replay.play()?;
        let recorded_actions = verified.verification.frames;
        if usize::try_from(report.outcome.actions).ok() != Some(recorded_actions) {
            return Err(LeagueError::ActionCount {
                reported: report.outcome.actions,
                recorded: recorded_actions,
            });
        }
        let expected_winner = match report.outcome.termination {
            Termination::Victory if verified.game.is_terminal() => verified.game.winner(),
            Termination::ActionLimit if !verified.game.is_terminal() => adjudicate(&verified.game),
            _ => return Err(LeagueError::OutcomeMismatch),
        };
        if report.outcome.winner != expected_winner {
            return Err(LeagueError::OutcomeMismatch);
        }

        let id = blake3::hash(&postcard::to_allocvec(report)?)
            .to_hex()
            .to_string();
        if self.matches.iter().any(|record| record.id == id) {
            return Err(LeagueError::DuplicateMatch(id));
        }

        let first_score = score_for_first(report.outcome);
        Self::apply_result(
            &mut self.participants,
            self.elo,
            &report.agents,
            first_score,
        );
        let record = LeagueMatch {
            id,
            agents: report.agents.clone(),
            seed: report.seed,
            outcome: report.outcome,
            final_digest: verified.verification.final_digest,
        };
        self.matches.push(record.clone());
        Ok(record)
    }

    pub fn standings(&self) -> Vec<Standing> {
        let mut entries: Vec<_> = self.participants.iter().collect();
        entries.sort_by(|(first_name, first), (second_name, second)| {
            second
                .rating
                .elo
                .total_cmp(&first.rating.elo)
                .then_with(|| second.rating.games.cmp(&first.rating.games))
                .then_with(|| first_name.cmp(second_name))
        });
        entries
            .into_iter()
            .enumerate()
            .map(|(index, (name, participant))| Standing {
                rank: index + 1,
                name: name.clone(),
                rating: participant.rating,
                wins: participant.wins,
                draws: participant.draws,
                losses: participant.losses,
            })
            .collect()
    }

    fn record_score(participant: &mut Participant, score: f64) {
        if score > 0.5 {
            participant.wins += 1;
        } else if score < 0.5 {
            participant.losses += 1;
        } else {
            participant.draws += 1;
        }
    }

    fn apply_result(
        participants: &mut BTreeMap<String, Participant>,
        elo: Elo,
        agents: &[String; 2],
        first_score: f64,
    ) {
        let mut first = participants.get(&agents[0]).cloned().unwrap_or_default();
        let mut second = participants.get(&agents[1]).cloned().unwrap_or_default();
        elo.update(&mut first.rating, &mut second.rating, first_score);
        Self::record_score(&mut first, first_score);
        Self::record_score(&mut second, 1.0 - first_score);
        participants.insert(agents[0].clone(), first);
        participants.insert(agents[1].clone(), second);
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

pub fn run_match<First: Agent + ?Sized, Second: Agent + ?Sized>(
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
    use antiyoy_core::{PlayerId, Rules};

    use super::{Elo, League, LeagueError, Rating, run_match, score_for_first, symmetric_duel};

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

    #[test]
    fn league_records_only_verified_unique_matches() {
        let scenario = symmetric_duel(7, 5, 47).expect("valid duel");
        let mut report = run_match(
            Rules::classic_generic(),
            scenario,
            &mut GreedyAgent::new("greedy"),
            &mut RandomAgent::new("random", 48),
            500,
        )
        .expect("valid match");
        let mut league = League::default();
        let recorded = league.record(&report).expect("verified match");

        assert_eq!(recorded.seed, 47);
        assert_eq!(league.matches.len(), 1);
        assert_eq!(league.standings().len(), 2);
        assert_eq!(
            league
                .standings()
                .iter()
                .map(|standing| standing.rating.games)
                .sum::<u64>(),
            2
        );
        assert!(matches!(
            league.record(&report),
            Err(LeagueError::DuplicateMatch(_))
        ));

        let mut corrupted = league.clone();
        corrupted
            .participants
            .get_mut("greedy")
            .expect("participant")
            .wins += 1;
        assert!(matches!(
            corrupted.validate(),
            Err(LeagueError::CorruptStandings)
        ));

        report.outcome.winner = match report.outcome.winner {
            Some(PlayerId(0)) => Some(PlayerId(1)),
            _ => Some(PlayerId(0)),
        };
        assert!(matches!(
            League::default().record(&report),
            Err(LeagueError::OutcomeMismatch)
        ));
    }
}
