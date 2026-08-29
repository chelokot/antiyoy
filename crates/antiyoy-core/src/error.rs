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
}
