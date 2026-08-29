#![forbid(unsafe_code)]

mod action;
mod error;
mod game;
mod rng;
mod rules;
mod state;
mod topology;
mod types;

pub use action::{Action, Transition};
pub use error::{ActionError, ConfigError};
pub use game::Game;
pub use rules::{CombatRules, EconomyRules, LifecycleRules, Rules, RulesProfile, VegetationRules};
pub use state::{Cell, InitialCell, Province, Scenario, Treasury};
pub use topology::{Axial, Topology};
pub use types::{HexId, Object, PlayerId, ProvinceId, Structure, Unit};

pub const ENGINE_VERSION: u16 = 3;
