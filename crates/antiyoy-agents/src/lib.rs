#![forbid(unsafe_code)]

mod evaluation;
mod puct;
mod search;

use antiyoy_core::{Action, DiplomacyCommand, Game};
use rand::{Rng, SeedableRng, rngs::SmallRng};

pub use evaluation::position_score;
pub use puct::{PuctConfig, PuctError, PuctLeaf, PuctSearch, PuctStats, PuctValueMode};
pub use search::{
    SearchAgent, SearchConfig, SearchConfigError, SearchReply, SearchStats, SearchTurn,
    SearchTurnSlate, followup_score, reply_score, search_plan_indices, search_plan_indices_with,
    search_reply, search_turn_slate,
};

pub trait Agent {
    fn name(&self) -> &str;
    fn select_action(&mut self, game: &Game, legal_actions: &[Action]) -> Action;
}

#[derive(Clone, Debug)]
pub struct RandomAgent {
    name: String,
    random: SmallRng,
}

impl RandomAgent {
    pub fn new(name: impl Into<String>, seed: u64) -> Self {
        Self {
            name: name.into(),
            random: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Agent for RandomAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, _game: &Game, legal_actions: &[Action]) -> Action {
        legal_actions[self.random.random_range(0..legal_actions.len())]
    }
}

#[derive(Clone, Debug)]
pub struct GreedyAgent {
    name: String,
}

impl GreedyAgent {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    fn action_priority(action: Action) -> u8 {
        match action {
            Action::Diplomacy {
                command: DiplomacyCommand::DeclareWar,
                ..
            } => 2,
            Action::EndTurn => 0,
            Action::Diplomacy { .. }
            | Action::Move { .. }
            | Action::Recruit { .. }
            | Action::Build { .. }
            | Action::PlantTree { .. } => 1,
        }
    }
}

impl Agent for GreedyAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, game: &Game, legal_actions: &[Action]) -> Action {
        let player = game.active_player();
        legal_actions
            .iter()
            .copied()
            .map(|action| {
                let mut candidate = game.clone();
                candidate
                    .step(action)
                    .expect("engine-generated legal action must apply");
                (
                    evaluation::position_score(&candidate, player),
                    Self::action_priority(action),
                    std::cmp::Reverse(action),
                )
            })
            .max()
            .map(|(_, _, action)| action.0)
            .expect("non-terminal game always has EndTurn")
    }
}

#[cfg(test)]
mod tests {
    use antiyoy_core::{
        Action, DiplomacyCommand, HexId, InitialCell, Object, PlayerId, Relation, Rules, Scenario,
        Topology,
    };

    use super::{
        Agent, GreedyAgent, SearchAgent, SearchConfig, SearchConfigError, position_score,
        reply_score, search_reply, search_turn_slate,
    };

    #[test]
    fn greedy_agent_prefers_free_capture_over_end_turn() {
        let topology = Topology::rectangle(5, 1).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 31);
        for hex in [0, 1] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        for hex in [3, 4] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[4].object = Object::Capital;
        let game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let legal = [
            Action::EndTurn,
            Action::Recruit {
                province: HexId(0),
                target: HexId(2),
                strength: 1,
            },
        ];
        let mut agent = GreedyAgent::new("greedy");
        assert_eq!(agent.select_action(&game, &legal), legal[1]);
    }

    #[test]
    fn greedy_agent_breaks_a_peace_stalemate() {
        let topology = Topology::rectangle(4, 1).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 37);
        for hex in [0, 1] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        for hex in [2, 3] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[3].object = Object::Capital;
        let mut rules = Rules::classic_generic();
        rules.diplomacy.enabled = true;
        rules.diplomacy.initial_relation = Relation::Neutral;
        let game = antiyoy_core::Game::new(rules, scenario).expect("valid game");
        let declaration = Action::Diplomacy {
            target: PlayerId(1),
            command: DiplomacyCommand::DeclareWar,
        };
        let legal = [Action::EndTurn, declaration];
        let mut agent = GreedyAgent::new("greedy");

        assert_eq!(agent.select_action(&game, &legal), declaration);
    }

    #[test]
    fn search_is_deterministic_and_respects_its_transition_budget() {
        let scenario = Scenario::symmetric_duel(7, 5, 101).expect("valid duel");
        let game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let mut legal = Vec::new();
        game.legal_actions(&mut legal);
        let config = SearchConfig {
            node_budget: 256,
            beam_width: 12,
            branch_width: 20,
            maximum_actions_per_turn: 12,
        };
        let mut first = SearchAgent::with_config("search", config).expect("valid search");
        let mut second = SearchAgent::with_config("search", config).expect("valid search");
        assert_eq!(
            first.select_action(&game, &legal),
            second.select_action(&game, &legal)
        );
        assert_eq!(first.last_stats(), second.last_stats());
        assert!(first.last_stats().nodes <= config.node_budget);
        assert!(first.last_stats().completed_turns > 0);
        assert!(first.last_stats().maximum_depth > 0);
    }

    #[test]
    fn search_turn_slate_is_ranked_distinct_and_preserves_the_played_plan() {
        let scenario = Scenario::symmetric_duel(7, 5, 101).expect("valid duel");
        let game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let config = SearchConfig {
            node_budget: 256,
            beam_width: 12,
            branch_width: 20,
            maximum_actions_per_turn: 12,
        };
        let first = search_turn_slate(&game, config, 8).expect("valid slate");
        let second = search_turn_slate(&game, config, 8).expect("repeatable slate");
        assert_eq!(first.stats, second.stats);
        assert_eq!(first.turns.len(), second.turns.len());
        assert!(first.turns.len() > 1);
        assert!(first.turns.len() <= 8);
        assert!(first.stats.completed_turns >= first.turns.len());
        for (index, turn) in first.turns.iter().enumerate() {
            assert_eq!(turn.actions, second.turns[index].actions);
            assert_eq!(turn.game, second.turns[index].game);
            assert!(turn.game.is_terminal() || turn.game.active_player() != game.active_player());
            assert!(
                first.turns[..index]
                    .iter()
                    .all(|other| other.game != turn.game)
            );
            if index > 0 {
                assert!(first.turns[index - 1].score >= turn.score);
            }
            let mut replay = game.clone();
            for action in &turn.actions {
                replay.step(*action).expect("slate actions remain legal");
            }
            assert_eq!(replay, turn.game);
        }
        let mut agent = SearchAgent::with_config("search", config).expect("valid search");
        let mut legal = Vec::new();
        game.legal_actions(&mut legal);
        assert_eq!(
            agent.select_action(&game, &legal),
            first.turns[0].actions[0]
        );
        assert_eq!(agent.last_stats(), first.stats);
    }

    #[test]
    fn search_turn_slate_rejects_zero_size() {
        let scenario = Scenario::symmetric_duel(7, 5, 101).expect("valid duel");
        let game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("valid game");
        assert!(matches!(
            search_turn_slate(&game, SearchConfig::default(), 0),
            Err(SearchConfigError::SlateSize)
        ));
    }

    #[test]
    fn reply_search_selects_the_best_completed_turn_after_opponent_response() {
        let scenario = Scenario::symmetric_duel(7, 5, 107).expect("valid duel");
        let game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let config = SearchConfig {
            node_budget: 64,
            beam_width: 12,
            branch_width: 20,
            maximum_actions_per_turn: 12,
        };
        let reply_config = SearchConfig {
            node_budget: 16,
            ..config
        };
        let slate = search_turn_slate(&game, config, 4).expect("valid slate");
        let expected = slate
            .turns
            .iter()
            .enumerate()
            .max_by_key(|(index, turn)| {
                let outcome = if turn.game.is_terminal() {
                    position_score(&turn.game, game.active_player())
                } else {
                    let reply = search_turn_slate(&turn.game, reply_config, 1)
                        .expect("valid opponent reply");
                    position_score(&reply.turns[0].game, game.active_player())
                };
                (outcome, turn.score, std::cmp::Reverse(*index))
            })
            .expect("completed turn");
        let mut legal = Vec::new();
        game.legal_actions(&mut legal);
        let mut first =
            SearchAgent::with_reply_search("reply", config, 4, 16).expect("valid reply search");
        let mut second =
            SearchAgent::with_reply_search("reply", config, 4, 16).expect("valid reply search");
        assert_eq!(first.select_action(&game, &legal), expected.1.actions[0]);
        assert_eq!(
            first.select_action(&game, &legal),
            second.select_action(&game, &legal)
        );
        assert_eq!(first.last_stats().nodes, slate.stats.nodes);
        assert_eq!(first.last_stats().selected_score, expected.1.score);
    }

    #[test]
    fn reply_plan_replays_to_the_reported_root_score() {
        let scenario = Scenario::symmetric_duel(7, 5, 109).expect("valid duel");
        let game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let config = SearchConfig {
            node_budget: 32,
            maximum_actions_per_turn: 8,
            ..SearchConfig::default()
        };
        let reply_config = SearchConfig {
            node_budget: 8,
            ..config
        };
        let slate = search_turn_slate(&game, config, 4).expect("valid slate");
        for turn in &slate.turns {
            let response = search_reply(turn, game.active_player(), reply_config);
            let mut replay = turn.game.clone();
            for action in &response.actions {
                replay
                    .step(*action)
                    .expect("searched opponent action is legal");
            }
            assert_eq!(response.game, replay);
            assert_eq!(
                response.score,
                position_score(&replay, game.active_player())
            );
            assert_eq!(
                response.score,
                reply_score(turn, game.active_player(), reply_config)
            );
            assert!(
                response.actions.is_empty()
                    || replay.is_terminal()
                    || replay.active_player() == game.active_player()
            );
        }
    }

    #[test]
    fn three_turn_search_selects_by_the_completed_root_followup() {
        let scenario = Scenario::symmetric_duel(7, 5, 111).expect("valid duel");
        let game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let config = SearchConfig {
            node_budget: 64,
            maximum_actions_per_turn: 12,
            ..SearchConfig::default()
        };
        let reply_config = SearchConfig {
            node_budget: 16,
            ..config
        };
        let followup_config = SearchConfig {
            node_budget: 8,
            ..config
        };
        let slate = search_turn_slate(&game, config, 4).expect("valid root slate");
        let expected = slate
            .turns
            .iter()
            .enumerate()
            .max_by_key(|(index, turn)| {
                let reply = search_reply(turn, game.active_player(), reply_config);
                let score = if reply.game.is_terminal() {
                    reply.score
                } else {
                    let followup =
                        search_turn_slate(&reply.game, followup_config, 1).expect("valid followup");
                    position_score(&followup.turns[0].game, game.active_player())
                };
                (score, turn.score, std::cmp::Reverse(*index))
            })
            .expect("completed turn");
        let mut legal = Vec::new();
        game.legal_actions(&mut legal);
        let mut agent = SearchAgent::with_three_turn_search("three-turn", config, 4, 16, 8)
            .expect("valid three-turn search");
        assert_eq!(agent.select_action(&game, &legal), expected.1.actions[0]);
        assert_eq!(agent.last_stats().selected_score, expected.1.score);
    }

    #[test]
    fn reply_search_validates_slate_and_reply_budgets() {
        assert!(matches!(
            SearchAgent::with_reply_search("reply", SearchConfig::default(), 0, 16),
            Err(SearchConfigError::SlateSize)
        ));
        assert!(matches!(
            SearchAgent::with_reply_search("reply", SearchConfig::default(), 4, 1),
            Err(SearchConfigError::NodeBudget)
        ));
        assert!(matches!(
            SearchAgent::with_three_turn_search("three-turn", SearchConfig::default(), 4, 16, 1),
            Err(SearchConfigError::NodeBudget)
        ));
    }

    #[test]
    fn search_reuses_only_a_plan_for_the_exact_expected_state() {
        let scenario = Scenario::symmetric_duel(7, 5, 103).expect("valid duel");
        let mut game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("game");
        let mut legal = Vec::new();
        game.legal_actions(&mut legal);
        let mut search = SearchAgent::new("search");
        let first = search.select_action(&game, &legal);
        assert_ne!(first, Action::EndTurn);
        let stats = search.last_stats();
        let searches = search.search_count();
        game.step(first).expect("searched action");
        game.legal_actions(&mut legal);
        search.select_action(&game, &legal);
        assert_eq!(search.last_stats(), stats);
        assert_eq!(search.search_count(), searches);
        search.clear_plan();
        search.select_action(&game, &legal);
        assert_eq!(search.search_count(), searches + 1);

        let reset = Scenario::symmetric_duel(7, 5, 103).expect("valid duel");
        let reset = antiyoy_core::Game::new(Rules::classic_generic(), reset).expect("game");
        reset.legal_actions(&mut legal);
        search.select_action(&reset, &legal);
        assert_eq!(search.search_count(), searches + 2);
    }

    #[test]
    fn exact_score_cache_preserves_every_replanned_action() {
        let scenario = Scenario::symmetric_duel(7, 5, 209).expect("valid duel");
        let mut game = antiyoy_core::Game::new(Rules::classic_generic(), scenario).expect("game");
        let config = SearchConfig {
            node_budget: 64,
            beam_width: 8,
            branch_width: 12,
            maximum_actions_per_turn: 12,
        };
        let mut uncached =
            SearchAgent::with_three_turn_search("plain", config, 4, 16, 8).expect("search config");
        let mut cached = SearchAgent::with_three_turn_search("cached", config, 4, 16, 8)
            .expect("search config")
            .with_score_cache();
        let mut legal = Vec::new();
        for _ in 0..32 {
            if game.is_terminal() {
                break;
            }
            game.legal_actions(&mut legal);
            uncached.clear_plan();
            cached.clear_plan();
            let expected = uncached.select_action(&game, &legal);
            assert_eq!(cached.select_action(&game, &legal), expected);
            game.step(expected).expect("searched action applies");
        }
        assert_eq!(uncached.search_count(), cached.search_count());
    }

    #[test]
    fn invalid_search_budget_is_rejected() {
        assert!(matches!(
            SearchAgent::with_config(
                "search",
                SearchConfig {
                    node_budget: 1,
                    ..SearchConfig::default()
                }
            ),
            Err(SearchConfigError::NodeBudget)
        ));
    }
}
