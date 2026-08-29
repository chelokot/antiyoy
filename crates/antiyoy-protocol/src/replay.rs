use antiyoy_core::{Action, ActionError, ENGINE_VERSION, Game, Rules, Scenario, Transition};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::PROTOCOL_VERSION;

const REPLAY_MAGIC: [u8; 8] = *b"ANTIYOY\0";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    pub fn of_game(game: &Game) -> Result<Self, ReplayError> {
        let bytes = postcard::to_allocvec(game)?;
        Ok(Self(*blake3::hash(&bytes).as_bytes()))
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReplayHeader {
    pub magic: [u8; 8],
    pub format_version: u16,
    pub engine_version: u16,
    pub rules: Rules,
    pub scenario: Scenario,
    pub initial_digest: Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReplayFrame {
    pub action: Action,
    pub transition: Transition,
    pub state_digest: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    pub header: ReplayHeader,
    pub frames: Vec<ReplayFrame>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Verification {
    pub frames: usize,
    pub final_digest: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedReplay {
    pub verification: Verification,
    pub game: Game,
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("replay serialization failed: {0}")]
    Serialization(#[from] postcard::Error),
    #[error("replay magic is invalid")]
    InvalidMagic,
    #[error("unsupported replay format {actual}, expected {expected}")]
    UnsupportedFormat { actual: u16, expected: u16 },
    #[error("replay requires engine version {actual}, this engine is {expected}")]
    UnsupportedEngine { actual: u16, expected: u16 },
    #[error("scenario is invalid: {0}")]
    InvalidScenario(#[from] antiyoy_core::ConfigError),
    #[error("action {frame} is invalid during replay: {source}")]
    InvalidAction { frame: usize, source: ActionError },
    #[error("initial state digest does not match")]
    InitialDivergence { expected: Digest, actual: Digest },
    #[error("state diverged after frame {frame}")]
    Divergence {
        frame: usize,
        expected: Digest,
        actual: Digest,
    },
    #[error("recording action failed: {0}")]
    RecordingAction(#[from] ActionError),
}

impl Replay {
    pub fn new(rules: Rules, scenario: Scenario) -> Result<(Self, Game), ReplayError> {
        let game = Game::new(rules.clone(), scenario.clone())?;
        let initial_digest = Digest::of_game(&game)?;
        Ok((
            Self {
                header: ReplayHeader {
                    magic: REPLAY_MAGIC,
                    format_version: PROTOCOL_VERSION,
                    engine_version: ENGINE_VERSION,
                    rules,
                    scenario,
                    initial_digest,
                },
                frames: Vec::new(),
            },
            game,
        ))
    }

    pub fn record(&mut self, game: &mut Game, action: Action) -> Result<Transition, ReplayError> {
        let transition = game.step(action)?;
        self.frames.push(ReplayFrame {
            action,
            transition,
            state_digest: Digest::of_game(game)?,
        });
        Ok(transition)
    }

    pub fn verify(&self) -> Result<Verification, ReplayError> {
        Ok(self.play()?.verification)
    }

    pub fn play(&self) -> Result<VerifiedReplay, ReplayError> {
        self.validate_header()?;
        let mut game = Game::new(self.header.rules.clone(), self.header.scenario.clone())?;
        let initial_digest = Digest::of_game(&game)?;
        if initial_digest != self.header.initial_digest {
            return Err(ReplayError::InitialDivergence {
                expected: self.header.initial_digest,
                actual: initial_digest,
            });
        }

        let mut final_digest = initial_digest;
        for (frame_index, frame) in self.frames.iter().enumerate() {
            let transition =
                game.step(frame.action)
                    .map_err(|source| ReplayError::InvalidAction {
                        frame: frame_index,
                        source,
                    })?;
            final_digest = Digest::of_game(&game)?;
            if final_digest != frame.state_digest || transition != frame.transition {
                return Err(ReplayError::Divergence {
                    frame: frame_index,
                    expected: frame.state_digest,
                    actual: final_digest,
                });
            }
        }

        Ok(VerifiedReplay {
            verification: Verification {
                frames: self.frames.len(),
                final_digest,
            },
            game,
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>, ReplayError> {
        self.validate_header()?;
        Ok(postcard::to_allocvec(self)?)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ReplayError> {
        let replay: Self = postcard::from_bytes(bytes)?;
        replay.validate_header()?;
        Ok(replay)
    }

    fn validate_header(&self) -> Result<(), ReplayError> {
        if self.header.magic != REPLAY_MAGIC {
            return Err(ReplayError::InvalidMagic);
        }
        if self.header.format_version != PROTOCOL_VERSION {
            return Err(ReplayError::UnsupportedFormat {
                actual: self.header.format_version,
                expected: PROTOCOL_VERSION,
            });
        }
        if self.header.engine_version != ENGINE_VERSION {
            return Err(ReplayError::UnsupportedEngine {
                actual: self.header.engine_version,
                expected: ENGINE_VERSION,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use antiyoy_core::{Action, HexId, InitialCell, Object, PlayerId, Rules, Scenario, Topology};

    use super::{Digest, Replay, ReplayError};

    fn replay_fixture() -> (Replay, antiyoy_core::Game) {
        let topology = Topology::rectangle(5, 1).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 91);
        for hex in [0, 1] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        for hex in [3, 4] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[4].object = Object::Capital;
        Replay::new(Rules::classic_generic(), scenario).expect("valid replay")
    }

    #[test]
    fn encoded_replay_verifies_after_round_trip() {
        let (mut replay, mut game) = replay_fixture();
        replay
            .record(
                &mut game,
                Action::Recruit {
                    province: HexId(0),
                    target: HexId(2),
                    strength: 1,
                },
            )
            .expect("legal action");
        replay
            .record(&mut game, Action::EndTurn)
            .expect("legal end turn");

        let bytes = replay.encode().expect("encodable replay");
        let decoded = Replay::decode(&bytes).expect("decodable replay");
        let verification = decoded.verify().expect("deterministic replay");
        assert_eq!(verification.frames, 2);
        assert_eq!(
            verification.final_digest,
            Digest::of_game(&game).expect("digest")
        );
    }

    #[test]
    fn verification_finds_first_tampered_frame() {
        let (mut replay, mut game) = replay_fixture();
        replay
            .record(
                &mut game,
                Action::Recruit {
                    province: HexId(0),
                    target: HexId(2),
                    strength: 1,
                },
            )
            .expect("legal action");
        replay.frames[0].state_digest = Digest([0; 32]);

        assert!(matches!(
            replay.verify(),
            Err(ReplayError::Divergence { frame: 0, .. })
        ));
    }
}
