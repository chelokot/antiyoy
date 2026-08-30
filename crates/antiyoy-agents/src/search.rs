use std::cmp::Ordering;
use std::collections::VecDeque;

use antiyoy_core::{Action, DiplomacyCommand, Game, Object, PlayerId, Structure};
use thiserror::Error;

use crate::evaluation::position_score;
use crate::{Agent, GreedyAgent};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchConfig {
    pub node_budget: usize,
    pub beam_width: usize,
    pub branch_width: usize,
    pub maximum_actions_per_turn: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            node_budget: 2_048,
            beam_width: 32,
            branch_width: 48,
            maximum_actions_per_turn: 24,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchStats {
    pub nodes: usize,
    pub completed_turns: usize,
    pub maximum_depth: usize,
    pub selected_score: i64,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum SearchConfigError {
    #[error("search node budget must be at least two")]
    NodeBudget,
    #[error("search beam width must be greater than zero")]
    BeamWidth,
    #[error("search branch width must be at least two")]
    BranchWidth,
    #[error("search turn depth must be greater than zero")]
    TurnDepth,
}

#[derive(Clone, Debug)]
struct Candidate {
    game: Game,
    actions: Vec<Action>,
    score: i64,
}

#[derive(Clone, Debug)]
struct PlannedAction {
    before: Game,
    action: Action,
}

#[derive(Clone, Debug)]
pub struct SearchAgent {
    name: String,
    config: SearchConfig,
    plan: VecDeque<PlannedAction>,
    last_stats: SearchStats,
    search_count: u64,
}

impl SearchAgent {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: SearchConfig::default(),
            plan: VecDeque::new(),
            last_stats: SearchStats::default(),
            search_count: 0,
        }
    }

    pub fn with_config(
        name: impl Into<String>,
        config: SearchConfig,
    ) -> Result<Self, SearchConfigError> {
        validate_config(config)?;
        Ok(Self {
            name: name.into(),
            config,
            plan: VecDeque::new(),
            last_stats: SearchStats::default(),
            search_count: 0,
        })
    }

    pub const fn config(&self) -> SearchConfig {
        self.config
    }

    pub const fn last_stats(&self) -> SearchStats {
        self.last_stats
    }

    pub const fn search_count(&self) -> u64 {
        self.search_count
    }

    pub fn clear_plan(&mut self) {
        self.plan.clear();
    }

    fn create_plan(&mut self, game: &Game) {
        self.search_count += 1;
        let player = game.active_player();
        let mut frontier = vec![Candidate {
            game: game.clone(),
            actions: Vec::new(),
            score: position_score(game, player),
        }];
        let mut completed = vec![greedy_turn_candidate(
            game,
            player,
            self.config.maximum_actions_per_turn,
        )];
        let mut nodes = 0;
        let mut maximum_depth = 0;

        for depth in 1..=self.config.maximum_actions_per_turn {
            if frontier.is_empty() || nodes >= self.config.node_budget {
                break;
            }
            let mut next = Vec::new();
            for candidate in frontier {
                for action in ordered_actions(&candidate.game, player, self.config.branch_width) {
                    if nodes >= self.config.node_budget {
                        break;
                    }
                    let mut successor = candidate.game.clone();
                    successor
                        .step(action)
                        .expect("engine-generated legal action must apply");
                    nodes += 1;
                    maximum_depth = maximum_depth.max(depth);
                    let mut actions = candidate.actions.clone();
                    actions.push(action);
                    let score = position_score(&successor, player);
                    let successor = Candidate {
                        game: successor,
                        actions,
                        score,
                    };
                    if successor.game.is_terminal() || successor.game.active_player() != player {
                        completed.push(successor);
                    } else {
                        next.push(successor);
                    }
                }
            }
            next.sort_by(compare_candidates);
            next.truncate(self.config.beam_width);
            frontier = next;
        }

        let completed_turns = completed.len();
        completed.sort_by(compare_candidates);
        let selected = completed
            .into_iter()
            .next()
            .expect("EndTurn always completes at least one candidate turn");
        self.last_stats = SearchStats {
            nodes,
            completed_turns,
            maximum_depth,
            selected_score: selected.score,
        };
        self.plan = materialize_plan(game, &selected.actions);
    }
}

fn greedy_turn_candidate(game: &Game, player: PlayerId, maximum_actions: usize) -> Candidate {
    let mut candidate = game.clone();
    let mut actions = Vec::new();
    let mut greedy = GreedyAgent::new("search-baseline");
    for action_index in 0..maximum_actions {
        let mut legal = Vec::new();
        candidate.legal_actions(&mut legal);
        let action = if action_index + 1 == maximum_actions {
            Action::EndTurn
        } else {
            greedy.select_action(&candidate, &legal)
        };
        candidate
            .step(action)
            .expect("greedy baseline action must apply");
        actions.push(action);
        if candidate.is_terminal() || candidate.active_player() != player {
            break;
        }
    }
    let score = position_score(&candidate, player);
    Candidate {
        game: candidate,
        actions,
        score,
    }
}

impl Agent for SearchAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, game: &Game, legal_actions: &[Action]) -> Action {
        let cached = self.plan.front().is_some_and(|planned| {
            planned.before == *game && legal_actions.contains(&planned.action)
        });
        if !cached {
            self.plan.clear();
            self.create_plan(game);
        }
        self.plan
            .pop_front()
            .map(|planned| planned.action)
            .expect("searched turn contains at least EndTurn")
    }
}

fn validate_config(config: SearchConfig) -> Result<(), SearchConfigError> {
    if config.node_budget < 2 {
        return Err(SearchConfigError::NodeBudget);
    }
    if config.beam_width == 0 {
        return Err(SearchConfigError::BeamWidth);
    }
    if config.branch_width < 2 {
        return Err(SearchConfigError::BranchWidth);
    }
    if config.maximum_actions_per_turn == 0 {
        return Err(SearchConfigError::TurnDepth);
    }
    Ok(())
}

fn compare_candidates(first: &Candidate, second: &Candidate) -> Ordering {
    second
        .score
        .cmp(&first.score)
        .then_with(|| first.actions.len().cmp(&second.actions.len()))
        .then_with(|| first.actions.cmp(&second.actions))
}

fn materialize_plan(game: &Game, actions: &[Action]) -> VecDeque<PlannedAction> {
    let mut state = game.clone();
    actions
        .iter()
        .copied()
        .map(|action| {
            let planned = PlannedAction {
                before: state.clone(),
                action,
            };
            state
                .step(action)
                .expect("selected search action must remain legal");
            planned
        })
        .collect()
}

fn ordered_actions(game: &Game, player: PlayerId, branch_width: usize) -> Vec<Action> {
    let mut legal = Vec::new();
    game.legal_actions(&mut legal);
    let mut ending = None;
    let mut continuing = Vec::with_capacity(legal.len());
    for action in legal {
        if action == Action::EndTurn {
            ending = Some(action);
        } else {
            continuing.push(action);
        }
    }
    continuing.sort_by(|first, second| {
        tactical_priority(game, player, *second)
            .cmp(&tactical_priority(game, player, *first))
            .then_with(|| first.cmp(second))
    });
    continuing.truncate(branch_width - 1);
    let mut ordered = Vec::with_capacity(continuing.len() + 1);
    ordered.push(ending.expect("non-terminal game always has EndTurn"));
    ordered.extend(continuing);
    ordered
}

fn tactical_priority(game: &Game, player: PlayerId, action: Action) -> i64 {
    match action {
        Action::EndTurn => i64::MIN,
        Action::Move { source, target } => {
            let target_cell = game.cell(target).expect("legal target exists");
            let capture = target_cell.owner() != player;
            i64::from(capture) * 1_000
                + i64::from(
                    game.cell(source)
                        .expect("legal source exists")
                        .unit()
                        .strength(),
                ) * 80
                + i64::from(matches!(target_cell.object(), Object::Pine | Object::Palm)) * 40
        }
        Action::Recruit {
            target, strength, ..
        } => {
            let capture = game.cell(target).expect("legal target exists").owner() != player;
            i64::from(capture) * 900 + i64::from(strength) * 70
        }
        Action::Build { structure, .. } => match structure {
            Structure::Farm => 500,
            Structure::Tower => 340,
            Structure::StrongTower => 420,
        },
        Action::PlantTree { .. } => 20,
        Action::Diplomacy { command, .. } => match command {
            DiplomacyCommand::DeclareWar => 600,
            DiplomacyCommand::Accept => 180,
            DiplomacyCommand::ProposeAlliance => 140,
            DiplomacyCommand::ProposeFriendship => 120,
            DiplomacyCommand::ProposeNeutral => 100,
            DiplomacyCommand::Reject => 40,
        },
    }
}
