use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{Game, PlayerId, Relation};

pub const OBJECTIVE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EconomyMetric {
    GrossIncome,
    Profit,
    Treasury,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum VictoryCondition {
    Domination,
    DiplomaticVictory {
        player: PlayerId,
    },
    SurviveThroughRound {
        player: PlayerId,
        round: u32,
    },
    DestroyPlayer {
        player: PlayerId,
        target: PlayerId,
    },
    ReachEconomy {
        player: PlayerId,
        metric: EconomyMetric,
        minimum: i64,
    },
    EnsurePlayerVictory {
        player: PlayerId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub schema_version: u16,
    pub condition: VictoryCondition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveStatus {
    Active,
    Complete {
        satisfied: bool,
        winner: Option<PlayerId>,
    },
}

impl ObjectiveStatus {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    pub const fn is_satisfied(self) -> bool {
        matches!(
            self,
            Self::Complete {
                satisfied: true,
                ..
            }
        )
    }

    pub const fn winner(self) -> Option<PlayerId> {
        match self {
            Self::Active | Self::Complete { winner: None, .. } => None,
            Self::Complete {
                winner: Some(winner),
                ..
            } => Some(winner),
        }
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ObjectiveError {
    #[error("objective schema {actual} is unsupported, expected {expected}")]
    UnsupportedSchema { actual: u16, expected: u16 },
    #[error("objective player {player} is outside a game with {players} players")]
    InvalidPlayer { player: u8, players: u8 },
    #[error("objective player and target must differ")]
    IdenticalPlayers,
}

impl Default for Objective {
    fn default() -> Self {
        Self {
            schema_version: OBJECTIVE_SCHEMA_VERSION,
            condition: VictoryCondition::Domination,
        }
    }
}

impl Objective {
    pub fn evaluate(&self, game: &Game) -> Result<ObjectiveStatus, ObjectiveError> {
        self.validate(game.player_count())?;
        let status = match self.condition {
            VictoryCondition::Domination => core_completion(game),
            VictoryCondition::DiplomaticVictory { player } => {
                if is_alive(game, player) && all_survivors_are_allied(game, player) {
                    completed(true, Some(player))
                } else {
                    incomplete_or_failed(game)
                }
            }
            VictoryCondition::SurviveThroughRound { player, round } => {
                if is_alive(game, player) && (game.round() > round || game.winner() == Some(player))
                {
                    completed(true, Some(player))
                } else {
                    incomplete_or_failed(game)
                }
            }
            VictoryCondition::DestroyPlayer { player, target } => {
                if is_alive(game, player) && !is_alive(game, target) {
                    completed(true, Some(player))
                } else {
                    incomplete_or_failed(game)
                }
            }
            VictoryCondition::ReachEconomy {
                player,
                metric,
                minimum,
            } => {
                if is_alive(game, player) && economy_value(game, player, metric) >= minimum {
                    completed(true, Some(player))
                } else {
                    incomplete_or_failed(game)
                }
            }
            VictoryCondition::EnsurePlayerVictory { player } => {
                if game.is_terminal() {
                    completed(game.winner() == Some(player), game.winner())
                } else {
                    ObjectiveStatus::Active
                }
            }
        };
        Ok(status)
    }

    fn validate(&self, player_count: u8) -> Result<(), ObjectiveError> {
        if self.schema_version != OBJECTIVE_SCHEMA_VERSION {
            return Err(ObjectiveError::UnsupportedSchema {
                actual: self.schema_version,
                expected: OBJECTIVE_SCHEMA_VERSION,
            });
        }
        match self.condition {
            VictoryCondition::Domination => Ok(()),
            VictoryCondition::DiplomaticVictory { player }
            | VictoryCondition::SurviveThroughRound { player, .. }
            | VictoryCondition::ReachEconomy { player, .. }
            | VictoryCondition::EnsurePlayerVictory { player } => {
                validate_player(player, player_count)
            }
            VictoryCondition::DestroyPlayer { player, target } => {
                validate_player(player, player_count)?;
                validate_player(target, player_count)?;
                if player == target {
                    return Err(ObjectiveError::IdenticalPlayers);
                }
                Ok(())
            }
        }
    }
}

fn validate_player(player: PlayerId, players: u8) -> Result<(), ObjectiveError> {
    if player.0 >= players {
        return Err(ObjectiveError::InvalidPlayer {
            player: player.0,
            players,
        });
    }
    Ok(())
}

fn completed(satisfied: bool, winner: Option<PlayerId>) -> ObjectiveStatus {
    ObjectiveStatus::Complete { satisfied, winner }
}

fn core_completion(game: &Game) -> ObjectiveStatus {
    if game.is_terminal() {
        completed(true, game.winner())
    } else {
        ObjectiveStatus::Active
    }
}

fn incomplete_or_failed(game: &Game) -> ObjectiveStatus {
    if game.is_terminal() {
        completed(false, game.winner())
    } else {
        ObjectiveStatus::Active
    }
}

fn is_alive(game: &Game, player: PlayerId) -> bool {
    game.provinces()
        .iter()
        .any(|province| province.owner() == player)
}

fn all_survivors_are_allied(game: &Game, player: PlayerId) -> bool {
    game.provinces().iter().all(|province| {
        province.owner() == player
            || game.relation(player, province.owner()) == Some(Relation::Alliance)
    })
}

fn economy_value(game: &Game, player: PlayerId, metric: EconomyMetric) -> i64 {
    game.provinces()
        .iter()
        .filter(|province| province.owner() == player)
        .map(|province| match metric {
            EconomyMetric::GrossIncome => game.province_income(province.id()).unwrap_or_default(),
            EconomyMetric::Profit => game.province_profit(province.id()).unwrap_or_default(),
            EconomyMetric::Treasury => province.money(),
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use crate::{
        Action, HexId, InitialCell, Object, PlayerId, Rules, Scenario, Topology, Treasury,
    };

    use super::{
        EconomyMetric, OBJECTIVE_SCHEMA_VERSION, Objective, ObjectiveError, ObjectiveStatus,
        VictoryCondition,
    };

    fn fixture() -> crate::Game {
        let topology = Topology::rectangle(5, 1).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 17);
        for hex in [0, 1] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        for hex in [3, 4] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[4].object = Object::Capital;
        scenario.treasuries.push(Treasury {
            province: HexId(0),
            money: 50,
        });
        crate::Game::new(Rules::classic_generic(), scenario).expect("valid game")
    }

    fn objective(condition: VictoryCondition) -> Objective {
        Objective {
            schema_version: OBJECTIVE_SCHEMA_VERSION,
            condition,
        }
    }

    #[test]
    fn economic_goal_uses_the_selected_exact_metric() {
        let game = fixture();
        assert_eq!(
            objective(VictoryCondition::ReachEconomy {
                player: PlayerId(0),
                metric: EconomyMetric::GrossIncome,
                minimum: 2,
            })
            .evaluate(&game),
            Ok(ObjectiveStatus::Complete {
                satisfied: true,
                winner: Some(PlayerId(0)),
            })
        );
        assert_eq!(
            objective(VictoryCondition::ReachEconomy {
                player: PlayerId(0),
                metric: EconomyMetric::Treasury,
                minimum: 100,
            })
            .evaluate(&game),
            Ok(ObjectiveStatus::Active)
        );
    }

    #[test]
    fn destroy_target_completes_after_the_target_loses_its_province() {
        let mut game = fixture();
        let target = objective(VictoryCondition::DestroyPlayer {
            player: PlayerId(0),
            target: PlayerId(1),
        });
        assert_eq!(target.evaluate(&game), Ok(ObjectiveStatus::Active));
        game.step(Action::Recruit {
            province: HexId(0),
            target: HexId(2),
            strength: 2,
        })
        .expect("legal recruit");
        game.step(Action::EndTurn).expect("legal end turn");
        game.step(Action::EndTurn).expect("legal end turn");
        game.step(Action::Move {
            source: HexId(2),
            target: HexId(3),
        })
        .expect("legal capture");
        game.step(Action::EndTurn).expect("terminal end turn");
        assert_eq!(
            target.evaluate(&game),
            Ok(ObjectiveStatus::Complete {
                satisfied: true,
                winner: Some(PlayerId(0)),
            })
        );
    }

    #[test]
    fn ensure_victory_distinguishes_a_target_loss() {
        let mut game = fixture();
        game.step(Action::Recruit {
            province: HexId(0),
            target: HexId(2),
            strength: 2,
        })
        .expect("legal recruit");
        game.step(Action::EndTurn).expect("legal end turn");
        game.step(Action::EndTurn).expect("legal end turn");
        game.step(Action::Move {
            source: HexId(2),
            target: HexId(3),
        })
        .expect("legal capture");
        game.step(Action::EndTurn).expect("terminal end turn");
        assert_eq!(
            objective(VictoryCondition::EnsurePlayerVictory {
                player: PlayerId(1),
            })
            .evaluate(&game),
            Ok(ObjectiveStatus::Complete {
                satisfied: false,
                winner: Some(PlayerId(0)),
            })
        );
    }

    #[test]
    fn invalid_target_is_rejected_before_evaluation() {
        assert_eq!(
            objective(VictoryCondition::DestroyPlayer {
                player: PlayerId(0),
                target: PlayerId(2),
            })
            .evaluate(&fixture()),
            Err(ObjectiveError::InvalidPlayer {
                player: 2,
                players: 2,
            })
        );
    }
}
