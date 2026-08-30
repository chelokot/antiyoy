#![forbid(unsafe_code)]

mod network;
mod replay;

pub use network::{
    CellView, ClaimSeatRequest, ClaimSeatResponse, ClientMessage, CreateMatchRequest,
    CreateMatchResponse, GameView, MAXIMUM_MATCH_PLAYERS, MINIMUM_MATCH_PLAYERS, MatchScenario,
    MatchSnapshot, MatchStatus, NETWORK_SCHEMA_VERSION, ProvinceView, RatingStatus, RelationView,
    SeatCredential, SeatKind, SeatRequest, ServerMessage, SubmitAction,
};
pub use replay::{
    Digest, Replay, ReplayError, ReplayFrame, ReplayHeader, Verification, VerifiedReplay,
};

pub const PROTOCOL_VERSION: u16 = 5;
