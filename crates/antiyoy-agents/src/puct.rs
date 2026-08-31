use std::collections::BTreeMap;

use antiyoy_core::{Action, Game, PlayerId};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PuctConfig {
    pub node_budget: usize,
    pub exploration: f64,
    pub virtual_loss: f64,
    pub maximum_depth: usize,
    pub root_value_weight: Option<f64>,
    pub search_opponent_turns: bool,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            node_budget: 256,
            exploration: 1.5,
            virtual_loss: 1.0,
            maximum_depth: 128,
            root_value_weight: None,
            search_opponent_turns: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PuctStats {
    pub nodes: usize,
    pub completed_simulations: u64,
    pub maximum_depth: usize,
    pub pending_leaves: usize,
    pub root_visits: u64,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum PuctError {
    #[error("PUCT node budget must be at least two")]
    NodeBudget,
    #[error("PUCT node budget exceeds the supported maximum")]
    NodeBudgetMaximum,
    #[error("PUCT exploration must be finite and positive")]
    Exploration,
    #[error("PUCT virtual loss must be finite and non-negative")]
    VirtualLoss,
    #[error("PUCT maximum depth must be positive")]
    MaximumDepth,
    #[error("PUCT root value weight must be finite and non-negative")]
    RootValueWeight,
    #[error("PUCT root must be a non-terminal state with legal actions")]
    TerminalRoot,
    #[error("unknown or already completed PUCT leaf token: {0}")]
    LeafToken(u64),
    #[error("PUCT leaf expected {expected} priors, received {actual}")]
    PriorCount { expected: usize, actual: usize },
    #[error("PUCT priors must be finite, non-negative, and have positive mass")]
    Priors,
    #[error("PUCT value must be finite")]
    Value,
    #[error("PUCT search has no visited root action")]
    NoRootAction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PuctLeaf {
    pub token: u64,
}

#[derive(Clone, Debug)]
struct Edge {
    action: Action,
    prior: f64,
    visits: u32,
    value_sum: f64,
    virtual_visits: u32,
    child: Option<usize>,
}

#[derive(Clone, Debug)]
enum NodeState {
    Unexpanded,
    Expanded(Vec<Edge>),
    Terminal(f64),
}

#[derive(Clone, Debug)]
struct Node {
    game: Game,
    legal_actions: Vec<Action>,
    state: NodeState,
    value_estimate: f64,
    pending: bool,
    expandable: bool,
}

#[derive(Clone, Debug)]
struct PendingLeaf {
    node: usize,
    path: Vec<(usize, usize)>,
    depth: usize,
}

#[derive(Clone, Debug)]
pub struct PuctSearch {
    config: PuctConfig,
    simulation_budget: u64,
    root_player: PlayerId,
    nodes: Vec<Node>,
    pending: BTreeMap<u64, PendingLeaf>,
    next_token: u64,
    completed_simulations: u64,
    maximum_depth: usize,
}

impl PuctSearch {
    pub fn new(
        game: &Game,
        legal_actions: &[Action],
        config: PuctConfig,
    ) -> Result<Self, PuctError> {
        validate_config(config)?;
        if game.is_terminal() || legal_actions.is_empty() {
            return Err(PuctError::TerminalRoot);
        }
        Ok(Self {
            config,
            simulation_budget: u64::from(
                u32::try_from(config.node_budget).map_err(|_| PuctError::NodeBudgetMaximum)?,
            ),
            root_player: game.active_player(),
            nodes: vec![Node {
                game: game.clone(),
                legal_actions: legal_actions.to_vec(),
                state: NodeState::Unexpanded,
                value_estimate: 0.0,
                pending: false,
                expandable: true,
            }],
            pending: BTreeMap::new(),
            next_token: 0,
            completed_simulations: 0,
            maximum_depth: 0,
        })
    }

    pub const fn config(&self) -> PuctConfig {
        self.config
    }

    pub fn is_complete(&self) -> bool {
        (self.nodes.len() >= self.config.node_budget
            || self.completed_simulations >= self.simulation_budget)
            && self.pending.is_empty()
    }

    pub fn select_leaves(&mut self, maximum_leaves: usize) -> Vec<PuctLeaf> {
        let mut leaves = Vec::with_capacity(maximum_leaves);
        while leaves.len() < maximum_leaves
            && self.nodes.len() < self.config.node_budget
            && self.completed_simulations < self.simulation_budget
        {
            let nodes = self.nodes.len();
            let simulations = self.completed_simulations;
            if let Some(leaf) = self.select_leaf() {
                leaves.push(leaf);
            } else if nodes == self.nodes.len() && simulations == self.completed_simulations {
                break;
            }
        }
        leaves
    }

    pub fn leaf(&self, token: u64) -> Option<(&Game, &[Action])> {
        let pending = self.pending.get(&token)?;
        let node = &self.nodes[pending.node];
        Some((&node.game, &node.legal_actions))
    }

    pub fn complete_leaf(
        &mut self,
        token: u64,
        priors: &[f64],
        value: f64,
    ) -> Result<(), PuctError> {
        if !value.is_finite() {
            return Err(PuctError::Value);
        }
        let pending = self
            .pending
            .remove(&token)
            .ok_or(PuctError::LeafToken(token))?;
        let expected = self.nodes[pending.node].legal_actions.len();
        if priors.len() != expected {
            self.pending.insert(token, pending);
            return Err(PuctError::PriorCount {
                expected,
                actual: priors.len(),
            });
        }
        let mass = priors.iter().copied().sum::<f64>();
        if !mass.is_finite()
            || mass <= 0.0
            || priors
                .iter()
                .any(|prior| !prior.is_finite() || *prior < 0.0)
        {
            self.pending.insert(token, pending);
            return Err(PuctError::Priors);
        }
        let root_value = self.root_value(pending.node, value);
        let node = &mut self.nodes[pending.node];
        node.pending = false;
        node.value_estimate = value;
        node.state = if node.expandable {
            NodeState::Expanded(
                node.legal_actions
                    .iter()
                    .copied()
                    .zip(priors.iter().copied())
                    .map(|(action, prior)| Edge {
                        action,
                        prior: prior / mass,
                        visits: 0,
                        value_sum: 0.0,
                        virtual_visits: 0,
                        child: None,
                    })
                    .collect(),
            )
        } else {
            NodeState::Terminal(root_value)
        };
        self.remove_virtual_visits(&pending.path);
        self.backup(&pending.path, root_value);
        self.completed_simulations += 1;
        self.maximum_depth = self.maximum_depth.max(pending.depth);
        Ok(())
    }

    pub fn selected_action_index(&self) -> Result<usize, PuctError> {
        let NodeState::Expanded(edges) = &self.nodes[0].state else {
            return Err(PuctError::NoRootAction);
        };
        if let Some(value_weight) = self.config.root_value_weight {
            return edges
                .iter()
                .enumerate()
                .max_by(|(first_index, first), (second_index, second)| {
                    root_policy_value_score(first, value_weight)
                        .total_cmp(&root_policy_value_score(second, value_weight))
                        .then_with(|| second_index.cmp(first_index))
                })
                .map(|(index, _)| index)
                .ok_or(PuctError::NoRootAction);
        }
        edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.visits > 0)
            .max_by(|(first_index, first), (second_index, second)| {
                first
                    .visits
                    .cmp(&second.visits)
                    .then_with(|| mean_value(first).total_cmp(&mean_value(second)))
                    .then_with(|| first.prior.total_cmp(&second.prior))
                    .then_with(|| second_index.cmp(first_index))
            })
            .map(|(index, _)| index)
            .ok_or(PuctError::NoRootAction)
    }

    pub fn root_target_probabilities(&self) -> Result<Vec<f64>, PuctError> {
        let NodeState::Expanded(edges) = &self.nodes[0].state else {
            return Err(PuctError::NoRootAction);
        };
        let weights = if let Some(value_weight) = self.config.root_value_weight {
            let scores = edges
                .iter()
                .map(|edge| root_policy_value_score(edge, value_weight))
                .collect::<Vec<_>>();
            let maximum = scores
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .ok_or(PuctError::NoRootAction)?;
            scores
                .into_iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>()
        } else {
            edges
                .iter()
                .map(|edge| f64::from(edge.visits))
                .collect::<Vec<_>>()
        };
        let mass = weights.iter().sum::<f64>();
        if !mass.is_finite() || mass <= 0.0 {
            return Err(PuctError::NoRootAction);
        }
        Ok(weights.into_iter().map(|weight| weight / mass).collect())
    }

    pub fn stats(&self) -> PuctStats {
        let root_visits = match &self.nodes[0].state {
            NodeState::Expanded(edges) => edges.iter().map(|edge| u64::from(edge.visits)).sum(),
            NodeState::Unexpanded | NodeState::Terminal(_) => 0,
        };
        PuctStats {
            nodes: self.nodes.len(),
            completed_simulations: self.completed_simulations,
            maximum_depth: self.maximum_depth,
            pending_leaves: self.pending.len(),
            root_visits,
        }
    }

    fn select_leaf(&mut self) -> Option<PuctLeaf> {
        let mut node_index = 0;
        let mut path = Vec::new();
        loop {
            let depth = path.len();
            match &self.nodes[node_index].state {
                NodeState::Terminal(value) => {
                    let value = *value;
                    self.backup(&path, value);
                    self.completed_simulations += 1;
                    self.maximum_depth = self.maximum_depth.max(depth);
                    return None;
                }
                NodeState::Unexpanded => {
                    if self.nodes[node_index].pending {
                        return None;
                    }
                    let token = self.next_token;
                    self.next_token = self.next_token.wrapping_add(1);
                    self.nodes[node_index].pending = true;
                    self.add_virtual_visits(&path);
                    self.pending.insert(
                        token,
                        PendingLeaf {
                            node: node_index,
                            path,
                            depth,
                        },
                    );
                    return Some(PuctLeaf { token });
                }
                NodeState::Expanded(_) if depth >= self.config.maximum_depth => {
                    let value = self.root_value(node_index, self.nodes[node_index].value_estimate);
                    self.backup(&path, value);
                    self.completed_simulations += 1;
                    self.maximum_depth = self.maximum_depth.max(depth);
                    return None;
                }
                NodeState::Expanded(_) => {
                    let edge_index = self.best_edge(node_index)?;
                    let child = self.child(node_index, edge_index);
                    path.push((node_index, edge_index));
                    node_index = child;
                }
            }
        }
    }

    fn best_edge(&self, node_index: usize) -> Option<usize> {
        let NodeState::Expanded(edges) = &self.nodes[node_index].state else {
            return None;
        };
        let parent_visits = edges
            .iter()
            .map(|edge| edge.visits + edge.virtual_visits)
            .sum::<u32>();
        edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.child.is_none_or(|child| !self.nodes[child].pending))
            .max_by(|(first_index, first), (second_index, second)| {
                edge_score(
                    first,
                    parent_visits,
                    self.config,
                    self.nodes[node_index].game.active_player() == self.root_player,
                )
                .total_cmp(&edge_score(
                    second,
                    parent_visits,
                    self.config,
                    self.nodes[node_index].game.active_player() == self.root_player,
                ))
                .then_with(|| second_index.cmp(first_index))
            })
            .map(|(index, _)| index)
    }

    fn child(&mut self, node_index: usize, edge_index: usize) -> usize {
        let existing = match &self.nodes[node_index].state {
            NodeState::Expanded(edges) => edges[edge_index].child,
            NodeState::Unexpanded | NodeState::Terminal(_) => None,
        };
        if let Some(child) = existing {
            return child;
        }
        let action = match &self.nodes[node_index].state {
            NodeState::Expanded(edges) => edges[edge_index].action,
            NodeState::Unexpanded | NodeState::Terminal(_) => unreachable!(),
        };
        let mut game = self.nodes[node_index].game.clone();
        game.step(action)
            .expect("PUCT expands only engine-generated legal actions");
        let mut legal_actions = Vec::new();
        let terminal = game.is_terminal();
        let state = if terminal {
            NodeState::Terminal(terminal_value(&game, self.root_player))
        } else {
            game.legal_actions(&mut legal_actions);
            NodeState::Unexpanded
        };
        let expandable = !terminal
            && (self.config.search_opponent_turns || game.active_player() == self.root_player);
        let child = self.nodes.len();
        self.nodes.push(Node {
            game,
            legal_actions,
            state,
            value_estimate: 0.0,
            pending: false,
            expandable,
        });
        let NodeState::Expanded(edges) = &mut self.nodes[node_index].state else {
            unreachable!();
        };
        edges[edge_index].child = Some(child);
        child
    }

    fn add_virtual_visits(&mut self, path: &[(usize, usize)]) {
        for &(node, edge) in path {
            let NodeState::Expanded(edges) = &mut self.nodes[node].state else {
                unreachable!();
            };
            edges[edge].virtual_visits += 1;
        }
    }

    fn remove_virtual_visits(&mut self, path: &[(usize, usize)]) {
        for &(node, edge) in path {
            let NodeState::Expanded(edges) = &mut self.nodes[node].state else {
                unreachable!();
            };
            edges[edge].virtual_visits -= 1;
        }
    }

    fn root_value(&self, leaf: usize, value: f64) -> f64 {
        if self.nodes[leaf].game.active_player() == self.root_player {
            value
        } else {
            -value
        }
    }

    fn backup(&mut self, path: &[(usize, usize)], value: f64) {
        for &(node, edge) in path.iter().rev() {
            let NodeState::Expanded(edges) = &mut self.nodes[node].state else {
                unreachable!();
            };
            edges[edge].visits += 1;
            edges[edge].value_sum += value;
        }
    }
}

fn validate_config(config: PuctConfig) -> Result<(), PuctError> {
    if config.node_budget < 2 {
        return Err(PuctError::NodeBudget);
    }
    if config.node_budget > usize::try_from(u32::MAX - 1).unwrap_or(usize::MAX) {
        return Err(PuctError::NodeBudgetMaximum);
    }
    if !config.exploration.is_finite() || config.exploration <= 0.0 {
        return Err(PuctError::Exploration);
    }
    if !config.virtual_loss.is_finite() || config.virtual_loss < 0.0 {
        return Err(PuctError::VirtualLoss);
    }
    if config.maximum_depth == 0 {
        return Err(PuctError::MaximumDepth);
    }
    if config
        .root_value_weight
        .is_some_and(|weight| !weight.is_finite() || weight < 0.0)
    {
        return Err(PuctError::RootValueWeight);
    }
    Ok(())
}

fn terminal_value(game: &Game, root_player: PlayerId) -> f64 {
    game.winner()
        .map_or(0.0, |winner| if winner == root_player { 1.0 } else { -1.0 })
}

fn mean_value(edge: &Edge) -> f64 {
    if edge.visits == 0 {
        0.0
    } else {
        edge.value_sum / f64::from(edge.visits)
    }
}

fn root_policy_value_score(edge: &Edge, value_weight: f64) -> f64 {
    edge.prior.ln() + value_weight * mean_value(edge)
}

fn edge_score(edge: &Edge, parent_visits: u32, config: PuctConfig, maximizing: bool) -> f64 {
    let visits = edge.visits + edge.virtual_visits;
    let virtual_penalty = config.virtual_loss * f64::from(edge.virtual_visits);
    let quality = if visits == 0 {
        0.0
    } else {
        let directed_value_sum = if maximizing {
            edge.value_sum
        } else {
            -edge.value_sum
        };
        (directed_value_sum - virtual_penalty) / f64::from(visits)
    };
    quality
        + config.exploration * edge.prior * f64::from(parent_visits + 1).sqrt()
            / f64::from(visits + 1)
}

#[cfg(test)]
mod tests {
    use antiyoy_core::{Game, Rules, Scenario};

    use super::{Edge, NodeState, PuctConfig, PuctError, PuctSearch, edge_score};

    fn search_with_uniform_evaluation(config: PuctConfig) -> PuctSearch {
        let scenario = Scenario::symmetric_duel(7, 5, 211).expect("valid duel");
        let game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let mut actions = Vec::new();
        game.legal_actions(&mut actions);
        let mut search = PuctSearch::new(&game, &actions, config).expect("valid search");
        while !search.is_complete() {
            let leaves = search.select_leaves(8);
            if leaves.is_empty() {
                assert!(search.is_complete());
            }
            for leaf in leaves {
                let action_count = search.leaf(leaf.token).expect("known leaf").1.len();
                search
                    .complete_leaf(leaf.token, &vec![1.0; action_count], 0.0)
                    .expect("valid evaluation");
            }
        }
        search
    }

    #[test]
    fn search_is_deterministic_and_uses_the_exact_node_budget() {
        let config = PuctConfig {
            node_budget: 64,
            maximum_depth: 32,
            ..PuctConfig::default()
        };
        let first = search_with_uniform_evaluation(config);
        let second = search_with_uniform_evaluation(config);
        assert_eq!(
            first.selected_action_index().expect("visited action"),
            second.selected_action_index().expect("visited action")
        );
        assert_eq!(first.stats(), second.stats());
        assert_eq!(first.stats().nodes, config.node_budget);
        assert_eq!(first.stats().pending_leaves, 0);
        assert!(first.stats().root_visits > 0);
    }

    #[test]
    fn root_priors_control_a_two_node_search() {
        let scenario = Scenario::symmetric_duel(7, 5, 223).expect("valid duel");
        let game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let mut actions = Vec::new();
        game.legal_actions(&mut actions);
        assert!(actions.len() > 2);
        let mut search = PuctSearch::new(
            &game,
            &actions,
            PuctConfig {
                node_budget: 2,
                ..PuctConfig::default()
            },
        )
        .expect("valid search");
        let root = search.select_leaves(8);
        assert_eq!(root.len(), 1);
        let mut priors = vec![0.0; actions.len()];
        priors[2] = 1.0;
        search
            .complete_leaf(root[0].token, &priors, 0.0)
            .expect("root evaluation");
        let child = search.select_leaves(8);
        assert_eq!(child.len(), 1);
        let action_count = search.leaf(child[0].token).expect("child").1.len();
        search
            .complete_leaf(child[0].token, &vec![1.0; action_count], 0.0)
            .expect("child evaluation");
        assert_eq!(search.selected_action_index().expect("visited action"), 2);
    }

    #[test]
    fn invalid_evaluations_preserve_the_pending_leaf() {
        let scenario = Scenario::symmetric_duel(7, 5, 227).expect("valid duel");
        let game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let mut actions = Vec::new();
        game.legal_actions(&mut actions);
        let mut search =
            PuctSearch::new(&game, &actions, PuctConfig::default()).expect("valid search");
        let leaf = search.select_leaves(1)[0];
        assert!(matches!(
            search.complete_leaf(leaf.token, &[], 0.0),
            Err(PuctError::PriorCount { .. })
        ));
        assert!(search.leaf(leaf.token).is_some());
        assert!(matches!(
            search.complete_leaf(leaf.token, &vec![0.0; actions.len()], 0.0),
            Err(PuctError::Priors)
        ));
        assert!(search.leaf(leaf.token).is_some());
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let scenario = Scenario::symmetric_duel(7, 5, 229).expect("valid duel");
        let game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let mut actions = Vec::new();
        game.legal_actions(&mut actions);
        assert!(matches!(
            PuctSearch::new(
                &game,
                &actions,
                PuctConfig {
                    node_budget: 1,
                    ..PuctConfig::default()
                }
            ),
            Err(PuctError::NodeBudget)
        ));
        assert!(matches!(
            PuctSearch::new(
                &game,
                &actions,
                PuctConfig {
                    root_value_weight: Some(-1.0),
                    ..PuctConfig::default()
                }
            ),
            Err(PuctError::RootValueWeight)
        ));
    }

    #[test]
    fn policy_value_root_selection_has_an_exact_policy_only_endpoint() {
        let scenario = Scenario::symmetric_duel(7, 5, 233).expect("valid duel");
        let game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let mut actions = Vec::new();
        game.legal_actions(&mut actions);
        let mut search = PuctSearch::new(
            &game,
            &actions,
            PuctConfig {
                root_value_weight: Some(0.0),
                ..PuctConfig::default()
            },
        )
        .expect("valid search");
        let root = search.select_leaves(1)[0];
        let mut priors = vec![0.0; actions.len()];
        priors[0] = 0.1;
        priors[1] = 0.9;
        search
            .complete_leaf(root.token, &priors, 0.0)
            .expect("root evaluation");
        let NodeState::Expanded(edges) = &mut search.nodes[0].state else {
            panic!("expanded root");
        };
        edges[0].visits = 100;
        edges[0].value_sum = 100.0;
        edges[1].visits = 1;
        edges[1].value_sum = -1.0;
        assert_eq!(search.selected_action_index().expect("root action"), 1);
        let targets = search
            .root_target_probabilities()
            .expect("root target probabilities");
        assert!((targets[0] - 0.1).abs() < 1e-12);
        assert!((targets[1] - 0.9).abs() < 1e-12);
        assert_eq!(targets[2..], vec![0.0; targets.len() - 2]);
        assert!((targets.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn maximum_depth_backups_do_not_touch_completed_virtual_visits() {
        let search = search_with_uniform_evaluation(PuctConfig {
            node_budget: 64,
            maximum_depth: 1,
            ..PuctConfig::default()
        });
        assert_eq!(search.stats().completed_simulations, 64);
        assert_eq!(search.stats().pending_leaves, 0);
    }

    #[test]
    fn opponent_leaf_horizon_evaluates_but_never_expands_other_turns() {
        let search = search_with_uniform_evaluation(PuctConfig {
            node_budget: 64,
            search_opponent_turns: false,
            ..PuctConfig::default()
        });
        let opponent_leaves = search
            .nodes
            .iter()
            .filter(|node| {
                !node.game.is_terminal() && node.game.active_player() != search.root_player
            })
            .collect::<Vec<_>>();
        assert!(!opponent_leaves.is_empty());
        assert!(
            opponent_leaves
                .iter()
                .all(|node| matches!(node.state, NodeState::Terminal(_)))
        );
    }

    #[test]
    fn virtual_loss_discourages_parallel_selection_for_every_player() {
        let config = PuctConfig {
            exploration: 1.0,
            virtual_loss: 1.0,
            ..PuctConfig::default()
        };
        let edge = Edge {
            action: antiyoy_core::Action::EndTurn,
            prior: 0.0,
            visits: 1,
            value_sum: 0.5,
            virtual_visits: 0,
            child: None,
        };
        let mut reserved = edge.clone();
        reserved.virtual_visits = 1;
        assert!(edge_score(&reserved, 1, config, true) < edge_score(&edge, 1, config, true));
        assert!(edge_score(&reserved, 1, config, false) < edge_score(&edge, 1, config, false));
    }
}
