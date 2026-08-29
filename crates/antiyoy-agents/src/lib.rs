#![forbid(unsafe_code)]

use antiyoy_core::{Action, DiplomacyCommand, Game, Object, PlayerId};
use rand::{Rng, SeedableRng, rngs::SmallRng};

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

    fn evaluate(game: &Game, player: PlayerId) -> i64 {
        if game.is_terminal() {
            return if game.winner() == Some(player) {
                1_000_000_000
            } else {
                -1_000_000_000
            };
        }

        let mut score = 0;
        for cell in game.cells().iter().copied() {
            let direction = if cell.owner() == player { 1 } else { -1 };
            if cell.owner().is_neutral() {
                continue;
            }
            score += direction * 100;
            score += direction * i64::from(cell.unit().strength()) * 18;
            score += direction
                * match cell.object() {
                    Object::Capital => 20,
                    Object::Farm => 28,
                    Object::Tower => 16,
                    Object::StrongTower => 30,
                    _ => 0,
                };
        }
        for province in game.provinces() {
            let direction = if province.owner() == player { 1 } else { -1 };
            let profit = game.province_profit(province.id()).unwrap_or_default();
            score += direction * (province.money() + profit * 8);
        }
        score
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
                    Self::evaluate(&candidate, player),
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

    use super::{Agent, GreedyAgent};

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
}
