#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

use antiyoy_agents::{Agent, GreedyAgent, RandomAgent};
use antiyoy_core::{Action, Game, Rules, Scenario};
use antiyoy_protocol::{
    CreateMatchRequest, CreateMatchResponse, MatchSnapshot, MatchStatus, NETWORK_SCHEMA_VERSION,
    Replay, SeatCredential, SeatKind, SubmitAction,
};
use thiserror::Error;
use tokio::sync::broadcast;

const UPDATE_CAPACITY: usize = 128;
const MAXIMUM_ACTION_LIMIT: u32 = 100_000;
const RANDOM_ID_BYTES: usize = 16;
const TOKEN_BYTES: usize = 32;

#[derive(Clone)]
pub struct MatchService {
    rooms: Arc<RwLock<BTreeMap<String, Arc<Mutex<MatchRoom>>>>>,
}

#[derive(Debug)]
struct MatchRoom {
    id: String,
    game: Game,
    replay: Replay,
    revision: u64,
    action_limit: u32,
    seats: [SeatController; 2],
    updates: broadcast::Sender<MatchSnapshot>,
}

#[derive(Debug)]
enum SeatController {
    Human { token: String },
    Greedy(GreedyAgent),
    Random(RandomAgent),
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ServiceError {
    #[error("match request is invalid: {0}")]
    InvalidRequest(String),
    #[error("match {0} was not found")]
    NotFound(String),
    #[error("seat credentials are invalid")]
    Unauthorized,
    #[error("client revision {actual} is stale; current revision is {expected}")]
    StaleRevision { actual: u64, expected: u64 },
    #[error("seat {seat} cannot act for active player {active}")]
    WrongTurn { seat: u8, active: u8 },
    #[error("match is no longer accepting actions")]
    Finished,
    #[error("action is illegal: {0}")]
    IllegalAction(String),
    #[error("multiplayer state failed: {0}")]
    Internal(String),
}

impl Default for MatchService {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchService {
    pub fn new() -> Self {
        Self {
            rooms: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    pub fn create_match(
        &self,
        request: &CreateMatchRequest,
    ) -> Result<CreateMatchResponse, ServiceError> {
        Self::validate_request(request)?;
        let rules = Rules::from_profile(request.rules_profile).ok_or_else(|| {
            ServiceError::InvalidRequest("custom rules require a full rules document".into())
        })?;
        let scenario = Scenario::symmetric_duel(request.width, request.height, request.seed)
            .map_err(|error| ServiceError::InvalidRequest(error.to_string()))?;
        let (replay, game) = Replay::new(rules, scenario)
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        let (updates, _) = broadcast::channel(UPDATE_CAPACITY);
        let mut credentials = Vec::new();
        let first = Self::seat_controller(request, 0, &mut credentials)?;
        let second = Self::seat_controller(request, 1, &mut credentials)?;
        let mut room = MatchRoom {
            id: String::new(),
            game,
            replay,
            revision: 0,
            action_limit: request.action_limit,
            seats: [first, second],
            updates,
        };
        room.advance_bots()?;
        let id = self.insert_room(room)?;
        let room = self.room(&id)?;
        let room = room
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        let snapshot = room.snapshot()?;
        Ok(CreateMatchResponse {
            snapshot,
            credentials,
        })
    }

    pub fn snapshot(&self, match_id: &str) -> Result<MatchSnapshot, ServiceError> {
        let room = self.room(match_id)?;
        let snapshot = room
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?
            .snapshot()?;
        Ok(snapshot)
    }

    pub fn replay(&self, match_id: &str) -> Result<Vec<u8>, ServiceError> {
        let room = self.room(match_id)?;
        let replay = room
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        replay
            .replay
            .encode()
            .map_err(|error| ServiceError::Internal(error.to_string()))
    }

    pub fn submit(
        &self,
        match_id: &str,
        seat: u8,
        token: &str,
        submission: SubmitAction,
    ) -> Result<MatchSnapshot, ServiceError> {
        if submission.schema_version != NETWORK_SCHEMA_VERSION {
            return Err(ServiceError::InvalidRequest(format!(
                "network schema {} is unsupported",
                submission.schema_version
            )));
        }
        let room = self.room(match_id)?;
        let mut room = room
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        room.authorize(seat, token)?;
        if submission.revision != room.revision {
            return Err(ServiceError::StaleRevision {
                actual: submission.revision,
                expected: room.revision,
            });
        }
        if room.status() != MatchStatus::Running {
            return Err(ServiceError::Finished);
        }
        if room.game.active_player().0 != seat {
            return Err(ServiceError::WrongTurn {
                seat,
                active: room.game.active_player().0,
            });
        }
        room.record(submission.action)?;
        room.advance_bots()?;
        let snapshot = room.snapshot()?;
        let _ = room.updates.send(snapshot.clone());
        Ok(snapshot)
    }

    pub fn subscribe(
        &self,
        match_id: &str,
    ) -> Result<(MatchSnapshot, broadcast::Receiver<MatchSnapshot>), ServiceError> {
        let room = self.room(match_id)?;
        let room = room
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        Ok((room.snapshot()?, room.updates.subscribe()))
    }

    pub fn authorize(&self, match_id: &str, seat: u8, token: &str) -> Result<(), ServiceError> {
        let room = self.room(match_id)?;
        let room = room
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        room.authorize(seat, token)
    }

    fn validate_request(request: &CreateMatchRequest) -> Result<(), ServiceError> {
        if request.schema_version != NETWORK_SCHEMA_VERSION {
            return Err(ServiceError::InvalidRequest(format!(
                "network schema {} is unsupported",
                request.schema_version
            )));
        }
        if request.action_limit == 0 || request.action_limit > MAXIMUM_ACTION_LIMIT {
            return Err(ServiceError::InvalidRequest(format!(
                "action limit must be within 1..={MAXIMUM_ACTION_LIMIT}"
            )));
        }
        for seat in &request.seats {
            if seat.name.is_empty() || seat.name.len() > 64 {
                return Err(ServiceError::InvalidRequest(
                    "seat names must contain 1..=64 bytes".into(),
                ));
            }
        }
        if request.seats[0].name == request.seats[1].name {
            return Err(ServiceError::InvalidRequest(
                "seat names must be distinct".into(),
            ));
        }
        Ok(())
    }

    fn seat_controller(
        request: &CreateMatchRequest,
        seat: usize,
        credentials: &mut Vec<SeatCredential>,
    ) -> Result<SeatController, ServiceError> {
        let requested = &request.seats[seat];
        Ok(match requested.kind {
            SeatKind::Human => {
                let token = random_hex(TOKEN_BYTES)?;
                credentials.push(SeatCredential {
                    seat: u8::try_from(seat).expect("two-seat matches fit in u8"),
                    name: requested.name.clone(),
                    token: token.clone(),
                });
                SeatController::Human { token }
            }
            SeatKind::Greedy => SeatController::Greedy(GreedyAgent::new(&requested.name)),
            SeatKind::Random => SeatController::Random(RandomAgent::new(
                &requested.name,
                request.seed ^ (u64::try_from(seat).expect("seat fits in u64") + 1),
            )),
        })
    }

    fn insert_room(&self, mut room: MatchRoom) -> Result<String, ServiceError> {
        let mut rooms = self
            .rooms
            .write()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        loop {
            let candidate = random_hex(RANDOM_ID_BYTES)?;
            if !rooms.contains_key(&candidate) {
                room.id.clone_from(&candidate);
                rooms.insert(candidate.clone(), Arc::new(Mutex::new(room)));
                return Ok(candidate);
            }
        }
    }

    fn room(&self, match_id: &str) -> Result<Arc<Mutex<MatchRoom>>, ServiceError> {
        self.rooms
            .read()
            .map_err(|error| ServiceError::Internal(error.to_string()))?
            .get(match_id)
            .cloned()
            .ok_or_else(|| ServiceError::NotFound(match_id.into()))
    }
}

impl MatchRoom {
    fn record(&mut self, action: Action) -> Result<(), ServiceError> {
        self.replay
            .record(&mut self.game, action)
            .map_err(|error| ServiceError::IllegalAction(error.to_string()))?;
        self.revision += 1;
        Ok(())
    }

    fn advance_bots(&mut self) -> Result<(), ServiceError> {
        let mut legal_actions = Vec::new();
        while self.status() == MatchStatus::Running {
            self.game.legal_actions(&mut legal_actions);
            let active = self.game.active_player().index();
            let action = match &mut self.seats[active] {
                SeatController::Human { .. } => break,
                SeatController::Greedy(agent) => agent.select_action(&self.game, &legal_actions),
                SeatController::Random(agent) => agent.select_action(&self.game, &legal_actions),
            };
            self.record(action)?;
        }
        Ok(())
    }

    fn authorize(&self, seat: u8, token: &str) -> Result<(), ServiceError> {
        let controller = self
            .seats
            .get(usize::from(seat))
            .ok_or(ServiceError::Unauthorized)?;
        match controller {
            SeatController::Human { token: expected } if expected == token => Ok(()),
            SeatController::Human { .. }
            | SeatController::Greedy(_)
            | SeatController::Random(_) => Err(ServiceError::Unauthorized),
        }
    }

    fn status(&self) -> MatchStatus {
        if self.game.is_terminal() {
            MatchStatus::Victory
        } else if self.replay.frames.len()
            >= usize::try_from(self.action_limit).expect("u32 action limit fits in usize")
        {
            MatchStatus::ActionLimit
        } else {
            MatchStatus::Running
        }
    }

    fn snapshot(&self) -> Result<MatchSnapshot, ServiceError> {
        MatchSnapshot::from_game(
            self.id.clone(),
            self.revision,
            self.status(),
            u32::try_from(self.replay.frames.len()).expect("replay length is capped by u32"),
            &self.game,
        )
        .map_err(|error| ServiceError::Internal(error.to_string()))
    }
}

fn random_hex(bytes: usize) -> Result<String, ServiceError> {
    let mut random = vec![0; bytes];
    getrandom::fill(&mut random).map_err(|error| ServiceError::Internal(error.to_string()))?;
    let mut encoded = String::with_capacity(bytes * 2);
    for byte in random {
        use std::fmt::Write;
        write!(encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use antiyoy_core::RulesProfile;
    use antiyoy_protocol::{
        CreateMatchRequest, MatchStatus, NETWORK_SCHEMA_VERSION, Replay, SeatKind, SeatRequest,
        SubmitAction,
    };

    use super::{MatchService, ServiceError};

    fn human_against_random() -> CreateMatchRequest {
        CreateMatchRequest {
            schema_version: NETWORK_SCHEMA_VERSION,
            rules_profile: RulesProfile::OnlineDuelV1,
            width: 7,
            height: 5,
            seed: 47,
            seats: [
                SeatRequest {
                    name: "human".into(),
                    kind: SeatKind::Human,
                },
                SeatRequest {
                    name: "random".into(),
                    kind: SeatKind::Random,
                },
            ],
            action_limit: 100,
        }
    }

    #[test]
    fn human_action_and_bot_reply_are_replay_verified() {
        let service = MatchService::new();
        let created = service
            .create_match(&human_against_random())
            .expect("valid match");
        let credential = &created.credentials[0];
        let action = created.snapshot.game.legal_actions[0];
        let snapshot = service
            .submit(
                &created.snapshot.match_id,
                credential.seat,
                &credential.token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: created.snapshot.revision,
                    action,
                },
            )
            .expect("legal human action");
        assert!(snapshot.revision >= 2);
        assert_eq!(snapshot.game.active_player, 0);
        assert_eq!(snapshot.game.relations.len(), 4);
        let replay = Replay::decode(
            &service
                .replay(&snapshot.match_id)
                .expect("encodable replay"),
        )
        .expect("decodable replay");
        let verified = replay.verify().expect("authoritative replay");
        assert_eq!(
            verified.frames,
            usize::try_from(snapshot.actions_played).unwrap()
        );
        assert_eq!(verified.final_digest, snapshot.digest);
    }

    #[test]
    fn stale_revision_does_not_mutate_the_match() {
        let service = MatchService::new();
        let created = service
            .create_match(&human_against_random())
            .expect("valid match");
        let credential = &created.credentials[0];
        let action = created.snapshot.game.legal_actions[0];
        let error = service
            .submit(
                &created.snapshot.match_id,
                credential.seat,
                &credential.token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: created.snapshot.revision + 1,
                    action,
                },
            )
            .expect_err("future revision must be rejected");
        assert_eq!(
            error,
            ServiceError::StaleRevision {
                actual: created.snapshot.revision + 1,
                expected: created.snapshot.revision,
            }
        );
        assert_eq!(
            service
                .snapshot(&created.snapshot.match_id)
                .expect("existing match")
                .revision,
            created.snapshot.revision
        );
    }

    #[test]
    fn bot_only_match_stops_at_the_exact_action_limit() {
        let service = MatchService::new();
        let mut request = human_against_random();
        request.seats[0].kind = SeatKind::Greedy;
        request.action_limit = 5;
        let created = service.create_match(&request).expect("valid bot match");
        assert!(created.credentials.is_empty());
        assert_eq!(created.snapshot.status, MatchStatus::ActionLimit);
        assert_eq!(created.snapshot.revision, 5);
        assert_eq!(created.snapshot.actions_played, 5);
    }

    #[test]
    fn invalid_token_cannot_mutate_a_match() {
        let service = MatchService::new();
        let created = service
            .create_match(&human_against_random())
            .expect("valid match");
        let error = service
            .submit(
                &created.snapshot.match_id,
                0,
                "invalid",
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: 0,
                    action: created.snapshot.game.legal_actions[0],
                },
            )
            .expect_err("invalid token must be rejected");
        assert_eq!(error, ServiceError::Unauthorized);
        assert_eq!(
            service
                .snapshot(&created.snapshot.match_id)
                .expect("existing match")
                .revision,
            0
        );
    }
}
