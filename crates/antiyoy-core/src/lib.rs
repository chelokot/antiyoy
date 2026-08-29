#![forbid(unsafe_code)]

mod error;
mod rules;
mod topology;
mod types;

pub use error::ConfigError;
pub use rules::{CombatRules, EconomyRules, Rules, RulesProfile, VegetationRules};
pub use topology::{Axial, Topology};
pub use types::{HexId, Object, PlayerId, ProvinceId, Structure, Unit};

pub const ENGINE_VERSION: u16 = 1;
