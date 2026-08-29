use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct HexId(pub u16);

impl HexId {
    pub const INVALID: Self = Self(u16::MAX);

    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    pub fn is_valid(self) -> bool {
        self != Self::INVALID
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct PlayerId(pub u8);

impl PlayerId {
    pub const NEUTRAL: Self = Self(u8::MAX);

    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    pub fn is_neutral(self) -> bool {
        self == Self::NEUTRAL
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct ProvinceId(pub u16);

impl ProvinceId {
    pub const NONE: Self = Self(u16::MAX);

    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    pub fn is_some(self) -> bool {
        self != Self::NONE
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Object {
    #[default]
    Empty,
    Capital,
    Farm,
    Tower,
    StrongTower,
    Pine,
    Palm,
    Grave,
}

impl Object {
    pub fn is_tree(self) -> bool {
        matches!(self, Self::Pine | Self::Palm)
    }

    pub fn is_building(self) -> bool {
        matches!(
            self,
            Self::Capital | Self::Farm | Self::Tower | Self::StrongTower
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(u8)]
pub enum Structure {
    Farm,
    Tower,
    StrongTower,
}

impl From<Structure> for Object {
    fn from(structure: Structure) -> Self {
        match structure {
            Structure::Farm => Self::Farm,
            Structure::Tower => Self::Tower,
            Structure::StrongTower => Self::StrongTower,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    strength: u8,
    ready: bool,
}

impl Unit {
    pub const EMPTY: Self = Self {
        strength: 0,
        ready: false,
    };

    pub const fn new(strength: u8, ready: bool) -> Self {
        Self { strength, ready }
    }

    pub const fn strength(self) -> u8 {
        self.strength
    }

    pub const fn is_ready(self) -> bool {
        self.ready
    }

    pub const fn is_present(self) -> bool {
        self.strength != 0
    }

    pub(crate) fn set_ready(&mut self, ready: bool) {
        self.ready = ready;
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::EMPTY;
    }
}
