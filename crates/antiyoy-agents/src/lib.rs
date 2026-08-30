#![forbid(unsafe_code)]

mod evaluation;
mod search;

use antiyoy_core::{Action, DiplomacyCommand, Game};
use rand::{Rng, SeedableRng, rngs::SmallRng};

pub use search::{SearchAgent, SearchConfig, SearchConfigError, SearchStats};

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

    use super::{Agent, GreedyAgent, SearchAgent, SearchConfig, SearchConfigError};

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
