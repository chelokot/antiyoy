#![forbid(unsafe_code)]

mod network;
mod replay;

pub use network::{
    CellView, ClientMessage, CreateMatchRequest, CreateMatchResponse, GameView, MatchSnapshot,
    MatchStatus, NETWORK_SCHEMA_VERSION, ProvinceView, RatingStatus, RelationView, SeatCredential,
    SeatKind, SeatRequest, ServerMessage, SubmitAction,
};
pub use replay::{
    Digest, Replay, ReplayError, ReplayFrame, ReplayHeader, Verification, VerifiedReplay,
};

pub const PROTOCOL_VERSION: u16 = 5;
