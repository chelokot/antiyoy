use serde::{Deserialize, Serialize};

use crate::{HexId, PlayerId, Structure};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Action {
    EndTurn,
    Move {
        source: HexId,
        target: HexId,
    },
    Recruit {
        province: HexId,
        target: HexId,
        strength: u8,
    },
    Build {
        target: HexId,
        structure: Structure,
    },
    PlantTree {
        target: HexId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    pub active_player: PlayerId,
    pub round: u32,
    pub terminal: bool,
    pub winner: Option<PlayerId>,
}
