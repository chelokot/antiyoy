use std::cmp::{Ordering, Reverse};
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

#[derive(Clone, Debug)]
pub struct SearchTurn {
    pub game: Game,
    pub actions: Vec<Action>,
    pub score: i64,
}

#[derive(Clone, Debug)]
pub struct SearchTurnSlate {
    pub turns: Vec<SearchTurn>,
    pub stats: SearchStats,
}

#[derive(Clone, Debug)]
pub struct SearchReply {
    pub score: i64,
    pub actions: Vec<Action>,
    pub game: Game,
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
    #[error("search turn slate size must be greater than zero")]
    SlateSize,
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
struct CachedScore {
    game: Game,
    root_player: PlayerId,
    score: i64,
}

#[derive(Clone, Debug)]
pub struct SearchAgent {
    name: String,
    config: SearchConfig,
    reply_nodes: usize,
    followup_nodes: usize,
    slate_size: usize,
    plan: VecDeque<PlannedAction>,
    score_cache: Vec<CachedScore>,
    score_cache_enabled: bool,
    cached_score_reuses: u64,
    last_stats: SearchStats,
    search_count: u64,
}

impl SearchAgent {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: SearchConfig::default(),
            reply_nodes: 0,
            followup_nodes: 0,
            slate_size: 1,
            plan: VecDeque::new(),
            score_cache: Vec::new(),
            score_cache_enabled: false,
            cached_score_reuses: 0,
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
            reply_nodes: 0,
            followup_nodes: 0,
            slate_size: 1,
            plan: VecDeque::new(),
            score_cache: Vec::new(),
            score_cache_enabled: false,
            cached_score_reuses: 0,
            last_stats: SearchStats::default(),
            search_count: 0,
        })
    }

    pub const fn config(&self) -> SearchConfig {
        self.config
    }

    pub fn with_reply_search(
        name: impl Into<String>,
        config: SearchConfig,
        slate_size: usize,
        reply_nodes: usize,
    ) -> Result<Self, SearchConfigError> {
        Self::with_three_turn_search(name, config, slate_size, reply_nodes, 0)
    }

    pub fn with_three_turn_search(
        name: impl Into<String>,
        config: SearchConfig,
        slate_size: usize,
        reply_nodes: usize,
        followup_nodes: usize,
    ) -> Result<Self, SearchConfigError> {
        if slate_size == 0 {
            return Err(SearchConfigError::SlateSize);
        }
        let reply_config = SearchConfig {
            node_budget: reply_nodes,
            ..config
        };
        validate_config(reply_config)?;
        if followup_nodes > 0 {
            validate_config(SearchConfig {
                node_budget: followup_nodes,
                ..config
            })?;
        }
        let mut agent = Self::with_config(name, config)?;
        agent.reply_nodes = reply_nodes;
        agent.followup_nodes = followup_nodes;
        agent.slate_size = slate_size;
        Ok(agent)
    }

    pub const fn last_stats(&self) -> SearchStats {
        self.last_stats
    }

    pub const fn search_count(&self) -> u64 {
        self.search_count
    }

    pub const fn cached_score_reuses(&self) -> u64 {
        self.cached_score_reuses
    }

    #[must_use]
    pub fn with_score_cache(mut self) -> Self {
        self.score_cache_enabled = true;
        self
    }

    pub fn enable_score_cache(&mut self) {
        self.score_cache_enabled = true;
    }

    pub fn clear_plan(&mut self) {
        self.plan.clear();
    }

    fn create_plan(&mut self, game: &Game) {
        self.search_count += 1;
        let slate = build_search_turn_slate(game, self.config, self.slate_size);
        let root_player = game.active_player();
        let selected = if self.reply_nodes == 0 {
            slate
                .turns
                .into_iter()
                .next()
                .expect("EndTurn always completes at least one candidate turn")
        } else {
            let previous_scores = std::mem::take(&mut self.score_cache);
            let mut current_scores = Vec::with_capacity(slate.turns.len());
            let mut cached_score_reuses = 0;
            let reply_config = SearchConfig {
                node_budget: self.reply_nodes,
                ..self.config
            };
            let followup_config = SearchConfig {
                node_budget: self.followup_nodes,
                ..self.config
            };
            let selected = slate
                .turns
                .into_iter()
                .enumerate()
                .max_by_key(|(index, turn)| {
                    let score = previous_scores
                        .iter()
                        .find(|cached| {
                            self.score_cache_enabled
                                && cached.root_player == root_player
                                && cached.game == turn.game
                        })
                        .map_or_else(
                            || {
                                if self.followup_nodes == 0 {
                                    reply_score(turn, root_player, reply_config)
                                } else {
                                    three_turn_score(
                                        turn,
                                        root_player,
                                        reply_config,
                                        followup_config,
                                    )
                                }
                            },
                            |cached| {
                                cached_score_reuses += 1;
                                cached.score
                            },
                        );
                    if self.score_cache_enabled {
                        current_scores.push(CachedScore {
                            game: turn.game.clone(),
                            root_player,
                            score,
                        });
                    }
                    (score, turn.score, Reverse(*index))
                })
                .map(|(_, turn)| turn)
                .expect("EndTurn always completes at least one candidate turn");
            self.score_cache = current_scores;
            self.cached_score_reuses += cached_score_reuses;
            selected
        };
        self.last_stats = SearchStats {
            selected_score: selected.score,
            ..slate.stats
        };
        self.plan = materialize_plan(game, &selected.actions);
    }
}

pub fn reply_score(turn: &SearchTurn, root_player: PlayerId, reply_config: SearchConfig) -> i64 {
    search_reply(turn, root_player, reply_config).score
}

fn three_turn_score(
    turn: &SearchTurn,
    root_player: PlayerId,
    reply_config: SearchConfig,
    followup_config: SearchConfig,
) -> i64 {
    let reply = search_reply(turn, root_player, reply_config);
    followup_score(&reply, root_player, followup_config)
}

pub fn followup_score(
    reply: &SearchReply,
    root_player: PlayerId,
    followup_config: SearchConfig,
) -> i64 {
    if reply.game.is_terminal() || reply.game.active_player() != root_player {
        return reply.score;
    }
    let followup = build_search_turn_slate(&reply.game, followup_config, 1);
    position_score(&followup.turns[0].game, root_player)
}

pub fn search_reply(
    turn: &SearchTurn,
    root_player: PlayerId,
    reply_config: SearchConfig,
) -> SearchReply {
    if turn.game.is_terminal() {
        return SearchReply {
            score: position_score(&turn.game, root_player),
            actions: Vec::new(),
            game: turn.game.clone(),
        };
    }
    let mut reply = build_search_turn_slate(&turn.game, reply_config, 1);
    let completed = reply.turns.remove(0);
    SearchReply {
        score: position_score(&completed.game, root_player),
        actions: completed.actions,
        game: completed.game,
    }
}

pub fn search_turn_slate(
    game: &Game,
    config: SearchConfig,
    size: usize,
) -> Result<SearchTurnSlate, SearchConfigError> {
    validate_config(config)?;
    if size == 0 {
        return Err(SearchConfigError::SlateSize);
    }
    Ok(build_search_turn_slate(game, config, size))
}

pub fn search_plan_indices(game: &Game, actions: &[Action], expected: &Game) -> Vec<usize> {
    replay_plan(
        game,
        actions,
        expected,
        None::<fn(&Game, Action, &Game) -> ()>,
    )
    .0
}

pub fn search_plan_indices_with<T>(
    game: &Game,
    actions: &[Action],
    expected: &Game,
    extract: impl FnMut(&Game, Action, &Game) -> T,
) -> (Vec<usize>, Vec<T>) {
    replay_plan(game, actions, expected, Some(extract))
}

fn replay_plan<T>(
    game: &Game,
    actions: &[Action],
    expected: &Game,
    mut extract: Option<impl FnMut(&Game, Action, &Game) -> T>,
) -> (Vec<usize>, Vec<T>) {
    let mut replay = game.clone();
    let mut legal = Vec::new();
    let mut indices = Vec::with_capacity(actions.len());
    let mut records = Vec::with_capacity(actions.len());
    for action in actions {
        replay.legal_actions(&mut legal);
        let index = legal
            .iter()
            .position(|candidate| candidate == action)
            .expect("searched turn action must be legal");
        indices.push(index);
        let before = extract.as_ref().map(|_| replay.clone());
        replay
            .step(*action)
            .expect("searched turn action must apply");
        if let (Some(before), Some(extract)) = (before.as_ref(), extract.as_mut()) {
            records.push(extract(before, *action, &replay));
        }
    }
    assert_eq!(
        replay, *expected,
        "indexed searched turn must replay exactly"
    );
    (indices, records)
}

fn build_search_turn_slate(game: &Game, config: SearchConfig, size: usize) -> SearchTurnSlate {
    let player = game.active_player();
    let mut frontier = vec![Candidate {
        game: game.clone(),
        actions: Vec::new(),
        score: position_score(game, player),
    }];
    let mut completed = vec![greedy_turn_candidate(
        game,
        player,
        config.maximum_actions_per_turn,
    )];
    let mut nodes = 0;
    let mut maximum_depth = 0;

    for depth in 1..=config.maximum_actions_per_turn {
        if frontier.is_empty() || nodes >= config.node_budget {
            break;
        }
        let mut next = Vec::new();
        for candidate in frontier {
            for action in ordered_actions(&candidate.game, player, config.branch_width) {
                if nodes >= config.node_budget {
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
        next.truncate(config.beam_width);
        frontier = next;
    }

    let completed_turns = completed.len();
    completed.sort_by(compare_candidates);
    let mut turns = Vec::with_capacity(size.min(completed_turns));
    for candidate in completed {
        if turns
            .iter()
            .all(|turn: &SearchTurn| turn.game != candidate.game)
        {
            turns.push(SearchTurn {
                game: candidate.game,
                actions: candidate.actions,
                score: candidate.score,
            });
            if turns.len() == size {
                break;
            }
        }
    }
    let selected_score = turns[0].score;
    SearchTurnSlate {
        turns,
        stats: SearchStats {
            nodes,
            completed_turns,
            maximum_depth,
            selected_score,
        },
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
    continuing
        .sort_by_cached_key(|action| (Reverse(tactical_priority(game, player, *action)), *action));
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

#[cfg(test)]
mod tests {
    use antiyoy_core::{Action, Game, Rules, Scenario};

    use super::{ordered_actions, tactical_priority};

    #[test]
    fn cached_tactical_keys_preserve_action_order() {
        for seed in [23, 29] {
            let scenario = Scenario::symmetric_duel(7, 5, seed).expect("valid duel");
            let mut game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
            let mut legal = Vec::new();
            for step in 0..64 {
                if game.is_terminal() {
                    break;
                }
                game.legal_actions(&mut legal);
                for branch_width in [2, 8, 48] {
                    let mut continuing: Vec<Action> = legal
                        .iter()
                        .copied()
                        .filter(|action| *action != Action::EndTurn)
                        .collect();
                    continuing.sort_by(|first, second| {
                        tactical_priority(&game, game.active_player(), *second)
                            .cmp(&tactical_priority(&game, game.active_player(), *first))
                            .then_with(|| first.cmp(second))
                    });
                    let mut expected = vec![Action::EndTurn];
                    expected.extend(continuing.into_iter().take(branch_width - 1));
                    assert_eq!(
                        ordered_actions(&game, game.active_player(), branch_width),
                        expected
                    );
                }
                game.step(legal[step % legal.len()])
                    .expect("legal action applies");
            }
        }
    }
}
