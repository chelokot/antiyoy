use serde::{Deserialize, Serialize};

use crate::{HexId, Object, PlayerId, ProvinceId, Topology, Unit};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InitialCell {
    pub owner: PlayerId,
    pub object: Object,
    pub unit_strength: u8,
}

impl InitialCell {
    pub const fn neutral() -> Self {
        Self {
            owner: PlayerId::NEUTRAL,
            object: Object::Empty,
            unit_strength: 0,
        }
    }

    pub const fn owned(owner: PlayerId) -> Self {
        Self {
            owner,
            object: Object::Empty,
            unit_strength: 0,
        }
    }
}

impl Default for InitialCell {
    fn default() -> Self {
        Self::neutral()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Treasury {
    pub province: HexId,
    pub money: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    pub topology: Topology,
    pub player_count: u8,
    pub cells: Vec<InitialCell>,
    pub treasuries: Vec<Treasury>,
    pub seed: u64,
}

impl Scenario {
    pub fn empty(topology: Topology, player_count: u8, seed: u64) -> Self {
        let cells = vec![InitialCell::neutral(); topology.len()];
        Self {
            topology,
            player_count,
            cells,
            treasuries: Vec::new(),
            seed,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub(crate) owner: PlayerId,
    pub(crate) object: Object,
    pub(crate) unit: Unit,
    pub(crate) province: ProvinceId,
    pub(crate) blocks_tree_spread: bool,
}

impl Cell {
    pub const fn owner(self) -> PlayerId {
        self.owner
    }

    pub const fn object(self) -> Object {
        self.object
    }

    pub const fn unit(self) -> Unit {
        self.unit
    }

    pub const fn province(self) -> ProvinceId {
        self.province
    }
}

impl From<InitialCell> for Cell {
    fn from(cell: InitialCell) -> Self {
        Self {
            owner: cell.owner,
            object: cell.object,
            unit: Unit::new(cell.unit_strength, false),
            province: ProvinceId::NONE,
            blocks_tree_spread: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Province {
    pub(crate) id: ProvinceId,
    pub(crate) owner: PlayerId,
    pub(crate) money: i64,
    pub(crate) capital: HexId,
    pub(crate) hexes: Vec<HexId>,
}

impl Province {
    pub const fn id(&self) -> ProvinceId {
        self.id
    }

    pub const fn owner(&self) -> PlayerId {
        self.owner
    }

    pub const fn money(&self) -> i64 {
        self.money
    }

    pub const fn capital(&self) -> HexId {
        self.capital
    }

    pub fn hexes(&self) -> &[HexId] {
        &self.hexes
    }
}
