#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use antiyoy_agents::Agent;
pub use antiyoy_core::adjudicate;
use antiyoy_core::{PlayerId, Rules, Scenario};
use antiyoy_protocol::{Digest, Replay, ReplayError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const LEAGUE_SCHEMA_VERSION: u16 = 2;
pub const PREVIOUS_LEAGUE_SCHEMA_VERSION: u16 = 1;

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
    pub agents: Vec<String>,
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
    pub agents: Vec<String>,
    #[serde(default = "default_league_player_count")]
    pub player_count: u8,
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
    #[error("agent identity {0} appears in more than one seat")]
    DuplicateAgent(String),
    #[error("a rated match must contain at least two agents")]
    TooFewAgents,
    #[error("winner seat {winner} is outside a match with {agents} agents")]
    InvalidWinner { winner: u8, agents: usize },
    #[error("match names {agents} agents but its replay contains {players} players")]
    PlayerCount { agents: usize, players: u8 },
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
    pub fn upgrade(mut self) -> Result<Self, LeagueError> {
        if self.schema_version == PREVIOUS_LEAGUE_SCHEMA_VERSION {
            self.schema_version = LEAGUE_SCHEMA_VERSION;
        }
        self.validate()?;
        Ok(self)
    }

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
            Self::validate_agents(&record.agents, record.outcome)?;
            Self::validate_player_count(&record.agents, record.player_count)?;
            Self::apply_result(&mut participants, self.elo, &record.agents, record.outcome);
        }
        if participants != self.participants {
            return Err(LeagueError::CorruptStandings);
        }
        Ok(())
    }

    pub fn record(&mut self, report: &MatchReport) -> Result<LeagueMatch, LeagueError> {
        self.validate()?;
        Self::validate_agents(&report.agents, report.outcome)?;
        if report.seed != report.replay.header.scenario.seed {
            return Err(LeagueError::SeedMismatch);
        }
        let verified = report.replay.play()?;
        Self::validate_player_count(&report.agents, verified.game.player_count())?;
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

        Self::apply_result(
            &mut self.participants,
            self.elo,
            &report.agents,
            report.outcome,
        );
        let record = LeagueMatch {
            id,
            agents: report.agents.clone(),
            player_count: verified.game.player_count(),
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

    fn record_score(participant: &mut Participant, won: bool, drawn: bool) {
        if won {
            participant.wins += 1;
        } else if drawn {
            participant.draws += 1;
        } else {
            participant.losses += 1;
        }
    }

    fn apply_result(
        participants: &mut BTreeMap<String, Participant>,
        elo: Elo,
        agents: &[String],
        outcome: MatchOutcome,
    ) {
        let previous = agents
            .iter()
            .map(|agent| participants.get(agent).cloned().unwrap_or_default())
            .collect::<Vec<_>>();
        let mut deltas = vec![0.0; agents.len()];
        let opponents = u32::try_from(agents.len() - 1).expect("player count fits in u32");
        let pair_weight = 1.0 / f64::from(opponents);
        for first in 0..agents.len() {
            for second in first + 1..agents.len() {
                let expected = 1.0
                    / (1.0
                        + 10.0_f64.powf(
                            (previous[second].rating.elo - previous[first].rating.elo) / 400.0,
                        ));
                let score = match outcome.winner {
                    Some(winner) if winner.index() == first => 1.0,
                    Some(winner) if winner.index() == second => 0.0,
                    _ => 0.5,
                };
                let delta = elo.k_factor * pair_weight * (score - expected);
                deltas[first] += delta;
                deltas[second] -= delta;
            }
        }
        for (seat, agent) in agents.iter().enumerate() {
            let mut participant = previous[seat].clone();
            participant.rating.elo += deltas[seat];
            participant.rating.games += 1;
            Self::record_score(
                &mut participant,
                outcome.winner.is_some_and(|winner| winner.index() == seat),
                outcome.winner.is_none(),
            );
            participants.insert(agent.clone(), participant);
        }
    }

    fn validate_agents(agents: &[String], outcome: MatchOutcome) -> Result<(), LeagueError> {
        if agents.len() < 2 {
            return Err(LeagueError::TooFewAgents);
        }
        let mut unique = BTreeSet::new();
        for agent in agents {
            if !unique.insert(agent) {
                return Err(LeagueError::DuplicateAgent(agent.clone()));
            }
        }
        if let Some(winner) = outcome.winner
            && winner.index() >= agents.len()
        {
            return Err(LeagueError::InvalidWinner {
                winner: winner.0,
                agents: agents.len(),
            });
        }
        Ok(())
    }

    fn validate_player_count(agents: &[String], players: u8) -> Result<(), LeagueError> {
        if agents.len() != usize::from(players) {
            return Err(LeagueError::PlayerCount {
                agents: agents.len(),
                players,
            });
        }
        Ok(())
    }
}

const fn default_league_player_count() -> u8 {
    2
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
    let names = vec![first.name().to_owned(), second.name().to_owned()];
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
    use antiyoy_core::{Action, GENERATOR_SCHEMA_VERSION, GeneratorConfig, PlayerId, Rules};
    use antiyoy_protocol::Replay;

    use super::{
        Elo, League, LeagueError, MatchOutcome, MatchReport, Rating, Termination, adjudicate,
        run_match, score_for_first, symmetric_duel,
    };

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

    #[test]
    fn multiplayer_elo_is_zero_sum_and_counts_one_game_per_agent() {
        let scenario = GeneratorConfig {
            schema_version: GENERATOR_SCHEMA_VERSION,
            width: 17,
            height: 13,
            players: 4,
            seed: 91,
            ..GeneratorConfig::default()
        }
        .generate()
        .expect("valid multiplayer map");
        let (mut replay, mut game) =
            Replay::new(Rules::online_default_v1(), scenario).expect("valid multiplayer replay");
        for _ in 0..4 {
            replay
                .record(&mut game, Action::EndTurn)
                .expect("end turn is legal");
        }
        let report = MatchReport {
            agents: (0..4).map(|seat| format!("agent-{seat}")).collect(),
            seed: 91,
            outcome: MatchOutcome {
                winner: adjudicate(&game),
                actions: 4,
                termination: Termination::ActionLimit,
            },
            replay,
        };
        let mut incomplete = report.clone();
        incomplete.agents.pop();
        assert!(matches!(
            League::default().record(&incomplete),
            Err(LeagueError::PlayerCount {
                agents: 3,
                players: 4
            })
        ));
        let mut league = League::default();
        league.record(&report).expect("verified multiplayer result");
        let standings = league.standings();
        assert_eq!(standings.len(), 4);
        assert!(standings.iter().all(|standing| standing.rating.games == 1));
        let rating_total = standings
            .iter()
            .map(|standing| standing.rating.elo)
            .sum::<f64>();
        assert!((rating_total - 4_000.0).abs() < 1e-9);
        league
            .validate()
            .expect("reproducible multiplayer standings");
    }

    #[test]
    fn schema_one_two_player_league_upgrades_without_rating_drift() {
        let scenario = symmetric_duel(7, 5, 63).expect("valid duel");
        let report = run_match(
            Rules::classic_generic(),
            scenario,
            &mut GreedyAgent::new("legacy-greedy"),
            &mut RandomAgent::new("legacy-random", 64),
            100,
        )
        .expect("valid legacy match");
        let mut current = League::default();
        current.record(&report).expect("verified legacy result");
        let mut legacy = serde_json::to_value(&current).expect("serialized legacy league");
        legacy["schema_version"] = 1.into();
        legacy["matches"][0]
            .as_object_mut()
            .expect("legacy match object")
            .remove("player_count");
        let legacy: League = serde_json::from_value(legacy).expect("decoded schema-one league");
        assert_eq!(legacy.upgrade().expect("upgraded league"), current);
    }
}
