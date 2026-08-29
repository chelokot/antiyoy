use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ConfigError {
    #[error("map dimensions must be non-zero")]
    EmptyMap,
    #[error("map contains {cells} cells, exceeding the limit of {maximum}")]
    MapTooLarge { cells: usize, maximum: usize },
    #[error("playable mask has {actual} entries, expected {expected}")]
    PlayableMaskSize { actual: usize, expected: usize },
    #[error("at least two players are required")]
    TooFewPlayers,
    #[error("player count {players} exceeds the supported maximum {maximum}")]
    TooManyPlayers { players: u16, maximum: u16 },
    #[error("unit strength must be between one and {maximum}, got {strength}")]
    InvalidUnitStrength { strength: u8, maximum: u8 },
    #[error("probability per million cannot exceed 1,000,000")]
    InvalidProbability,
    #[error("movement range must be positive")]
    ZeroMovementRange,
    #[error("farm price increment must not be negative")]
    NegativeFarmPriceIncrement,
    #[error("scenario contains {actual} cells, expected {expected}")]
    ScenarioSize { actual: usize, expected: usize },
    #[error("player {player} on hex {hex} is outside the configured player range")]
    InvalidOwner { hex: u16, player: u8 },
    #[error("inactive hex {hex} contains game state")]
    OccupiedInactiveHex { hex: u16 },
    #[error("hex {hex} contains both a unit and an object")]
    ConflictingOccupants { hex: u16 },
    #[error("treasury anchor {hex} does not identify a province")]
    InvalidTreasuryAnchor { hex: u16 },
    #[error("more than one treasury is assigned to the same province")]
    DuplicateTreasury,
    #[error("symmetric duel maps must be at least 5 × 2")]
    InvalidDuelDimensions,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ActionError {
    #[error("the game is already finished")]
    GameFinished,
    #[error("hex {0} does not exist or is not playable")]
    InvalidHex(u16),
    #[error("the source does not contain a ready unit owned by the active player")]
    UnitNotReady,
    #[error("the selected province is not owned by the active player")]
    InvalidProvince,
    #[error("the target is outside the action's reachable zone")]
    Unreachable,
    #[error("the target is occupied by an incompatible object or unit")]
    Occupied,
    #[error("strength {strength} is outside the range 1..={maximum}")]
    InvalidStrength { strength: u8, maximum: u8 },
    #[error("the province cannot afford this action")]
    InsufficientFunds,
    #[error("the target's defense is too strong")]
    Defended,
    #[error("this action is disabled by the current rules")]
    Disabled,
    #[error("a farm must touch a capital or another farm in its province")]
    UnsupportedFarm,
}
