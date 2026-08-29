#![forbid(unsafe_code)]

use antiyoy_core::{
    Action, ActionError, ConfigError, Game, HexId, Object, PlayerId, Rules, Scenario, Structure,
    Transition,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const OBSERVATION_VERSION: u16 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ActionKind {
    EndTurn,
    Move,
    Recruit,
    Build,
    PlantTree,
}

impl ActionKind {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActionFeatures {
    pub kind: ActionKind,
    pub source: u16,
    pub target: u16,
    pub parameter: u8,
}

impl From<Action> for ActionFeatures {
    fn from(action: Action) -> Self {
        match action {
            Action::EndTurn => Self {
                kind: ActionKind::EndTurn,
                source: HexId::INVALID.0,
                target: HexId::INVALID.0,
                parameter: 0,
            },
            Action::Move { source, target } => Self {
                kind: ActionKind::Move,
                source: source.0,
                target: target.0,
                parameter: 0,
            },
            Action::Recruit {
                province,
                target,
                strength,
            } => Self {
                kind: ActionKind::Recruit,
                source: province.0,
                target: target.0,
                parameter: strength,
            },
            Action::Build { target, structure } => Self {
                kind: ActionKind::Build,
                source: HexId::INVALID.0,
                target: target.0,
                parameter: structure_code(structure),
            },
            Action::PlantTree { target } => Self {
                kind: ActionKind::PlantTree,
                source: HexId::INVALID.0,
                target: target.0,
                parameter: 0,
            },
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BatchObservation {
    pub version: u16,
    pub rules: Rules,
    pub cell_offsets: Vec<usize>,
    pub province_offsets: Vec<usize>,
    pub action_offsets: Vec<usize>,
    pub widths: Vec<u16>,
    pub heights: Vec<u16>,
    pub active_players: Vec<u8>,
    pub rounds: Vec<u32>,
    pub playable: Vec<u8>,
    pub owners: Vec<u8>,
    pub objects: Vec<u8>,
    pub unit_strengths: Vec<u8>,
    pub ready: Vec<u8>,
    pub defenses: Vec<u8>,
    pub province_ids: Vec<u16>,
    pub province_owners: Vec<u8>,
    pub province_money: Vec<i64>,
    pub province_profit: Vec<i64>,
    pub province_capitals: Vec<u16>,
    pub province_sizes: Vec<usize>,
    pub actions: Vec<ActionFeatures>,
}

impl BatchObservation {
    pub fn clear(&mut self) {
        self.version = OBSERVATION_VERSION;
        self.cell_offsets.clear();
        self.province_offsets.clear();
        self.action_offsets.clear();
        self.widths.clear();
        self.heights.clear();
        self.active_players.clear();
        self.rounds.clear();
        self.playable.clear();
        self.owners.clear();
        self.objects.clear();
        self.unit_strengths.clear();
        self.ready.clear();
        self.defenses.clear();
        self.province_ids.clear();
        self.province_owners.clear();
        self.province_money.clear();
        self.province_profit.clear();
        self.province_capitals.clear();
        self.province_sizes.clear();
        self.actions.clear();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RewardComponents {
    pub outcome: i8,
    pub territory_delta: i32,
    pub treasury_delta: i64,
    pub unit_strength_delta: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StepResult {
    pub actor: PlayerId,
    pub reward: RewardComponents,
    pub round: u32,
    pub terminal: bool,
    pub truncated: bool,
    pub winner: Option<PlayerId>,
}

impl StepResult {
    pub const fn done(self) -> bool {
        self.terminal || self.truncated
    }
}

#[derive(Debug, Error)]
pub enum BatchError {
    #[error("a batch must contain at least one scenario")]
    Empty,
    #[error("action limit must be greater than zero")]
    ZeroActionLimit,
    #[error("received {actual} action indices for {expected} environments")]
    ActionCount { actual: usize, expected: usize },
    #[error("environment index {index} is outside a batch of {environments}")]
    InvalidEnvironment { index: usize, environments: usize },
    #[error("environment {index} is already done and must be reset")]
    EnvironmentDone { index: usize },
    #[error(
        "action index {action} is outside {legal_actions} legal actions for environment {environment}"
    )]
    InvalidActionIndex {
        environment: usize,
        action: usize,
        legal_actions: usize,
    },
    #[error("scenario configuration failed: {0}")]
    Configuration(#[from] ConfigError),
    #[error("legal action failed: {0}")]
    Action(#[from] ActionError),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PlayerMetrics {
    territory: i32,
    treasury: i64,
    unit_strength: i32,
}

pub struct BatchEnv {
    rules: Rules,
    scenarios: Vec<Scenario>,
    games: Vec<Game>,
    legal_actions: Vec<Vec<Action>>,
    episode_steps: Vec<u32>,
    done: Vec<bool>,
    action_limit: u32,
}

impl BatchEnv {
    pub fn new(
        rules: Rules,
        scenarios: Vec<Scenario>,
        action_limit: u32,
    ) -> Result<Self, BatchError> {
        if scenarios.is_empty() {
            return Err(BatchError::Empty);
        }
        if action_limit == 0 {
            return Err(BatchError::ZeroActionLimit);
        }
        let mut games = Vec::with_capacity(scenarios.len());
        let mut legal_actions = Vec::with_capacity(scenarios.len());
        for scenario in &scenarios {
            let game = Game::new(rules.clone(), scenario.clone())?;
            let mut actions = Vec::with_capacity(scenario.topology.len() * 4);
            game.legal_actions(&mut actions);
            games.push(game);
            legal_actions.push(actions);
        }
        let environments = games.len();
        Ok(Self {
            rules,
            scenarios,
            games,
            legal_actions,
            episode_steps: vec![0; environments],
            done: vec![false; environments],
            action_limit,
        })
    }

    pub fn symmetric_duels(
        rules: Rules,
        environments: usize,
        width: u16,
        height: u16,
        first_seed: u64,
        action_limit: u32,
    ) -> Result<Self, BatchError> {
        if environments == 0 {
            return Err(BatchError::Empty);
        }
        let mut scenarios = Vec::with_capacity(environments);
        let mut seed = first_seed;
        for _ in 0..environments {
            scenarios.push(Scenario::symmetric_duel(width, height, seed)?);
            seed = seed.wrapping_add(1);
        }
        Self::new(rules, scenarios, action_limit)
    }

    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }

    pub fn game(&self, index: usize) -> Option<&Game> {
        self.games.get(index)
    }

    pub fn legal_actions(&self, index: usize) -> Option<&[Action]> {
        self.legal_actions.get(index).map(Vec::as_slice)
    }

    pub fn episode_steps(&self, index: usize) -> Option<u32> {
        self.episode_steps.get(index).copied()
    }

    pub fn is_done(&self, index: usize) -> Option<bool> {
        self.done.get(index).copied()
    }

    pub fn reset_with_seed(&mut self, index: usize, seed: u64) -> Result<(), BatchError> {
        let scenario = self
            .scenarios
            .get_mut(index)
            .ok_or(BatchError::InvalidEnvironment {
                index,
                environments: self.games.len(),
            })?;
        scenario.seed = seed;
        self.games[index] = Game::new(self.rules.clone(), scenario.clone())?;
        self.episode_steps[index] = 0;
        self.done[index] = false;
        self.games[index].legal_actions(&mut self.legal_actions[index]);
        Ok(())
    }

    pub fn step(
        &mut self,
        environment: usize,
        action_index: usize,
    ) -> Result<StepResult, BatchError> {
        if environment >= self.games.len() {
            return Err(BatchError::InvalidEnvironment {
                index: environment,
                environments: self.games.len(),
            });
        }
        if self.done[environment] {
            return Err(BatchError::EnvironmentDone { index: environment });
        }
        let action = self.legal_actions[environment]
            .get(action_index)
            .copied()
            .ok_or(BatchError::InvalidActionIndex {
                environment,
                action: action_index,
                legal_actions: self.legal_actions[environment].len(),
            })?;
        step_environment(
            &mut self.games[environment],
            &mut self.legal_actions[environment],
            &mut self.episode_steps[environment],
            &mut self.done[environment],
            self.action_limit,
            action,
        )
    }

    pub fn step_all(&mut self, action_indices: &[usize]) -> Result<Vec<StepResult>, BatchError> {
        if action_indices.len() != self.len() {
            return Err(BatchError::ActionCount {
                actual: action_indices.len(),
                expected: self.len(),
            });
        }
        for (environment, action_index) in action_indices.iter().copied().enumerate() {
            if self.done[environment] {
                return Err(BatchError::EnvironmentDone { index: environment });
            }
            if action_index >= self.legal_actions[environment].len() {
                return Err(BatchError::InvalidActionIndex {
                    environment,
                    action: action_index,
                    legal_actions: self.legal_actions[environment].len(),
                });
            }
        }
        self.games
            .par_iter_mut()
            .zip(self.legal_actions.par_iter_mut())
            .zip(self.episode_steps.par_iter_mut())
            .zip(self.done.par_iter_mut())
            .zip(action_indices.par_iter().copied())
            .map(|((((game, actions), episode_steps), done), action_index)| {
                let action = actions[action_index];
                step_environment(
                    game,
                    actions,
                    episode_steps,
                    done,
                    self.action_limit,
                    action,
                )
            })
            .collect()
    }

    pub fn observe(&self, output: &mut BatchObservation) {
        output.clear();
        output.rules = self.rules.clone();
        output.cell_offsets.reserve(self.len() + 1);
        output.province_offsets.reserve(self.len() + 1);
        output.action_offsets.reserve(self.len() + 1);
        output.cell_offsets.push(0);
        output.province_offsets.push(0);
        output.action_offsets.push(0);
        for (game, actions) in self.games.iter().zip(&self.legal_actions) {
            output.widths.push(game.topology().width());
            output.heights.push(game.topology().height());
            output.active_players.push(game.active_player().0);
            output.rounds.push(game.round());
            for (cell, hex_id) in game.cells().iter().copied().zip(0_u16..) {
                let hex = HexId(hex_id);
                output
                    .playable
                    .push(u8::from(game.topology().is_playable(hex)));
                output.owners.push(cell.owner().0);
                output.objects.push(object_code(cell.object()));
                output.unit_strengths.push(cell.unit().strength());
                output.ready.push(u8::from(cell.unit().is_ready()));
                output
                    .defenses
                    .push(game.hex_defense(hex).unwrap_or_default());
                output.province_ids.push(cell.province().0);
            }
            for province in game.provinces() {
                output.province_owners.push(province.owner().0);
                output.province_money.push(province.money());
                output
                    .province_profit
                    .push(game.province_profit(province.id()).unwrap_or_default());
                output.province_capitals.push(province.capital().0);
                output.province_sizes.push(province.hexes().len());
            }
            output
                .actions
                .extend(actions.iter().copied().map(ActionFeatures::from));
            output.cell_offsets.push(output.owners.len());
            output.province_offsets.push(output.province_money.len());
            output.action_offsets.push(output.actions.len());
        }
    }
}

fn structure_code(structure: Structure) -> u8 {
    match structure {
        Structure::Farm => 0,
        Structure::Tower => 1,
        Structure::StrongTower => 2,
    }
}

fn object_code(object: Object) -> u8 {
    match object {
        Object::Empty => 0,
        Object::Capital => 1,
        Object::Farm => 2,
        Object::Tower => 3,
        Object::StrongTower => 4,
        Object::Pine => 5,
        Object::Palm => 6,
        Object::Grave => 7,
    }
}

fn player_metrics(game: &Game, player: PlayerId) -> PlayerMetrics {
    let mut metrics = PlayerMetrics::default();
    for cell in game
        .cells()
        .iter()
        .copied()
        .filter(|cell| cell.owner() == player)
    {
        metrics.territory += 1;
        metrics.unit_strength += i32::from(cell.unit().strength());
    }
    metrics.treasury = game
        .provinces()
        .iter()
        .filter(|province| province.owner() == player)
        .map(antiyoy_core::Province::money)
        .sum();
    metrics
}

fn step_environment(
    game: &mut Game,
    legal_actions: &mut Vec<Action>,
    episode_steps: &mut u32,
    done: &mut bool,
    action_limit: u32,
    action: Action,
) -> Result<StepResult, BatchError> {
    let actor = game.active_player();
    let before = player_metrics(game, actor);
    let transition = game.step(action)?;
    *episode_steps += 1;
    let truncated = !transition.terminal && *episode_steps >= action_limit;
    *done = transition.terminal || truncated;
    let after = player_metrics(game, actor);
    let reward = reward_components(actor, before, after, transition);
    if *done {
        legal_actions.clear();
    } else {
        game.legal_actions(legal_actions);
    }
    Ok(StepResult {
        actor,
        reward,
        round: transition.round,
        terminal: transition.terminal,
        truncated,
        winner: transition.winner,
    })
}

fn reward_components(
    actor: PlayerId,
    before: PlayerMetrics,
    after: PlayerMetrics,
    transition: Transition,
) -> RewardComponents {
    RewardComponents {
        outcome: match transition.winner {
            Some(winner) if winner == actor => 1,
            Some(_) => -1,
            None => 0,
        },
        territory_delta: after.territory - before.territory,
        treasury_delta: after.treasury - before.treasury,
        unit_strength_delta: after.unit_strength - before.unit_strength,
    }
}

#[cfg(test)]
mod tests {
    use antiyoy_core::{Action, Rules};

    use super::{ActionFeatures, ActionKind, BatchEnv, BatchObservation};

    #[test]
    fn observation_offsets_partition_cells_and_legal_actions() {
        let environment = BatchEnv::symmetric_duels(Rules::classic_generic(), 3, 7, 5, 47, 500)
            .expect("valid batch");
        let mut observation = BatchObservation::default();
        environment.observe(&mut observation);
        assert_eq!(observation.cell_offsets, [0, 35, 70, 105]);
        assert_eq!(observation.province_offsets.len(), 4);
        assert_eq!(observation.action_offsets.len(), 4);
        assert_eq!(observation.owners.len(), 105);
        assert_eq!(observation.actions.len(), observation.action_offsets[3]);
    }

    #[test]
    fn action_features_preserve_core_action_identity() {
        assert_eq!(
            ActionFeatures::from(Action::EndTurn),
            ActionFeatures {
                kind: ActionKind::EndTurn,
                source: u16::MAX,
                target: u16::MAX,
                parameter: 0,
            }
        );
    }

    #[test]
    fn equal_seeds_and_actions_produce_equal_games() {
        let mut environment = BatchEnv::symmetric_duels(Rules::classic_generic(), 2, 7, 5, 91, 500)
            .expect("valid batch");
        environment.reset_with_seed(1, 91).expect("valid reset");
        for _ in 0..20 {
            let first_actions = environment.legal_actions(0).expect("known environment");
            let second_actions = environment.legal_actions(1).expect("known environment");
            assert_eq!(first_actions, second_actions);
            let selected = first_actions.len() / 2;
            let results = environment
                .step_all(&[selected, selected])
                .expect("legal steps");
            assert_eq!(results[0], results[1]);
            assert_eq!(environment.game(0), environment.game(1));
            if results[0].done() {
                break;
            }
        }
    }

    #[test]
    fn action_limit_truncates_without_claiming_core_terminal() {
        let mut environment = BatchEnv::symmetric_duels(Rules::classic_generic(), 1, 7, 5, 47, 1)
            .expect("valid batch");
        let result = environment.step(0, 0).expect("end turn is legal");
        assert!(result.truncated);
        assert!(!result.terminal);
        assert!(result.done());
    }
}
