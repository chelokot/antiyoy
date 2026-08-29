#![forbid(unsafe_code)]

mod replay;

pub use replay::{Digest, Replay, ReplayError, ReplayFrame, ReplayHeader, Verification};

pub const PROTOCOL_VERSION: u16 = 4;
