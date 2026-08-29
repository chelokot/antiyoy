#![forbid(unsafe_code)]

mod storage;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use antiyoy_agents::{Agent, GreedyAgent, RandomAgent};
use antiyoy_core::{Action, Game, Rules, Scenario};
use antiyoy_eval::{League, LeagueError, MatchOutcome, MatchReport, Termination, adjudicate};
use antiyoy_protocol::{
    CreateMatchRequest, CreateMatchResponse, MatchSnapshot, MatchStatus, NETWORK_SCHEMA_VERSION,
    RatingStatus, Replay, SeatCredential, SeatKind, SubmitAction,
};
use thiserror::Error;
use tokio::sync::broadcast;

use crate::storage::{ROOM_STORAGE_SCHEMA_VERSION, Storage, StoredRoom};

const RANDOM_ID_BYTES: usize = 16;
const TOKEN_BYTES: usize = 32;

#[derive(Clone)]
pub struct MatchService {
    rooms: Arc<RwLock<BTreeMap<String, Arc<Mutex<MatchRoom>>>>>,
    limits: ServiceLimits,
    storage: Option<Arc<Storage>>,
    league: Arc<Mutex<League>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServiceLimits {
    pub maximum_rooms: usize,
    pub maximum_cells: usize,
    pub maximum_action_limit: u32,
    pub update_capacity: usize,
}

#[derive(Clone, Debug)]
struct MatchRoom {
    id: String,
    request: CreateMatchRequest,
    game: Game,
    replay: Replay,
    revision: u64,
    action_limit: u32,
    seats: [SeatController; 2],
    updates: broadcast::Sender<MatchSnapshot>,
    closed: bool,
    rated: bool,
    duplicate: bool,
}

#[derive(Clone, Debug)]
enum SeatController {
    Human { token_hash: [u8; 32] },
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
    #[error("match was closed")]
    Closed,
    #[error("multiplayer room capacity {limit} has been reached")]
    Capacity { limit: usize },
    #[error("action is illegal: {0}")]
    IllegalAction(String),
    #[error("stored multiplayer state is invalid: {0}")]
    CorruptStorage(String),
    #[error("multiplayer storage failed: {0}")]
    Storage(String),
    #[error("multiplayer state failed: {0}")]
    Internal(String),
}

impl Default for MatchService {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ServiceLimits {
    fn default() -> Self {
        Self {
            maximum_rooms: 1_024,
            maximum_cells: 4_096,
            maximum_action_limit: 10_000,
            update_capacity: 32,
        }
    }
}

impl MatchService {
    pub fn new() -> Self {
        Self {
            rooms: Arc::new(RwLock::new(BTreeMap::new())),
            limits: ServiceLimits::default(),
            storage: None,
            league: Arc::new(Mutex::new(League::default())),
        }
    }

    pub fn with_limits(limits: ServiceLimits) -> Result<Self, ServiceError> {
        if limits.maximum_rooms == 0
            || limits.maximum_cells == 0
            || limits.maximum_action_limit == 0
            || limits.update_capacity == 0
        {
            return Err(ServiceError::InvalidRequest(
                "service limits must be positive".into(),
            ));
        }
        Ok(Self {
            rooms: Arc::new(RwLock::new(BTreeMap::new())),
            limits,
            storage: None,
            league: Arc::new(Mutex::new(League::default())),
        })
    }

    pub fn persistent(
        limits: ServiceLimits,
        directory: impl AsRef<Path>,
    ) -> Result<Self, ServiceError> {
        let mut service = Self::with_limits(limits)?;
        let storage = Arc::new(
            Storage::new(directory.as_ref())
                .map_err(|error| ServiceError::Storage(error.to_string()))?,
        );
        service.league = Arc::new(Mutex::new(
            storage
                .load_league()
                .map_err(|error| ServiceError::Storage(error.to_string()))?,
        ));
        service.storage = Some(storage);
        service.restore_rooms()?;
        service.reconcile_ratings()?;
        Ok(service)
    }

    pub fn create_match(
        &self,
        request: &CreateMatchRequest,
    ) -> Result<CreateMatchResponse, ServiceError> {
        self.validate_request(request)?;
        let rules = Rules::from_profile(request.rules_profile).ok_or_else(|| {
            ServiceError::InvalidRequest("custom rules require a full rules document".into())
        })?;
        let scenario = Scenario::symmetric_duel(request.width, request.height, request.seed)
            .map_err(|error| ServiceError::InvalidRequest(error.to_string()))?;
        let (replay, game) = Replay::new(rules, scenario)
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        let (updates, _) = broadcast::channel(self.limits.update_capacity);
        let mut credentials = Vec::new();
        let first = Self::seat_controller(request, 0, &mut credentials)?;
        let second = Self::seat_controller(request, 1, &mut credentials)?;
        let mut room = MatchRoom {
            id: String::new(),
            request: request.clone(),
            game,
            replay,
            revision: 0,
            action_limit: request.action_limit,
            seats: [first, second],
            updates,
            closed: false,
            rated: false,
            duplicate: false,
        };
        room.advance_bots()?;
        let id = self.insert_room(room)?;
        let room = self.room(&id)?;
        let mut room = room
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        if let Err(error) = self.persist_room(&room) {
            drop(room);
            self.rooms
                .write()
                .map_err(|lock_error| ServiceError::Internal(lock_error.to_string()))?
                .remove(&id);
            return Err(error);
        }
        let _ = self.rate_room(&mut room);
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
        if room.closed {
            return Err(ServiceError::Closed);
        }
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
        let previous = room.clone();
        room.record(submission.action)?;
        room.advance_bots()?;
        if let Err(error) = self.persist_room(&room) {
            *room = previous;
            return Err(error);
        }
        let _ = self.rate_room(&mut room);
        let snapshot = room.snapshot()?;
        let _ = room.updates.send(snapshot.clone());
        Ok(snapshot)
    }

    pub fn delete(&self, match_id: &str, seat: u8, token: &str) -> Result<(), ServiceError> {
        let room = self.room(match_id)?;
        {
            let mut room = room
                .lock()
                .map_err(|error| ServiceError::Internal(error.to_string()))?;
            room.authorize(seat, token)?;
            self.rate_room(&mut room)?;
            if let Some(storage) = &self.storage {
                storage
                    .delete(match_id)
                    .map_err(|error| ServiceError::Storage(error.to_string()))?;
            }
            room.closed = true;
        }
        self.rooms
            .write()
            .map_err(|error| ServiceError::Internal(error.to_string()))?
            .remove(match_id);
        Ok(())
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

    pub fn league(&self) -> Result<League, ServiceError> {
        self.reconcile_ratings()?;
        self.league
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))
            .map(|league| league.clone())
    }

    fn validate_request(&self, request: &CreateMatchRequest) -> Result<(), ServiceError> {
        if request.schema_version != NETWORK_SCHEMA_VERSION {
            return Err(ServiceError::InvalidRequest(format!(
                "network schema {} is unsupported",
                request.schema_version
            )));
        }
        if request.action_limit == 0 || request.action_limit > self.limits.maximum_action_limit {
            return Err(ServiceError::InvalidRequest(format!(
                "action limit must be within 1..={}",
                self.limits.maximum_action_limit
            )));
        }
        let cells = usize::from(request.width) * usize::from(request.height);
        if cells > self.limits.maximum_cells {
            return Err(ServiceError::InvalidRequest(format!(
                "map contains {cells} cells, limit is {}",
                self.limits.maximum_cells
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
                SeatController::Human {
                    token_hash: token_hash(&token),
                }
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
        if rooms.len() >= self.limits.maximum_rooms {
            return Err(ServiceError::Capacity {
                limit: self.limits.maximum_rooms,
            });
        }
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

    fn persist_room(&self, room: &MatchRoom) -> Result<(), ServiceError> {
        let Some(storage) = &self.storage else {
            return Ok(());
        };
        storage
            .save(&room.stored())
            .map_err(|error| ServiceError::Storage(error.to_string()))
    }

    fn restore_rooms(&self) -> Result<(), ServiceError> {
        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| ServiceError::Internal("persistent service has no storage".into()))?;
        for stored in storage
            .load()
            .map_err(|error| ServiceError::Storage(error.to_string()))?
        {
            let mut room = MatchRoom::restore(stored, self.limits.update_capacity)?;
            self.validate_request(&room.request)?;
            room.advance_bots()?;
            self.persist_room(&room)?;
            self.insert_restored_room(room)?;
        }
        Ok(())
    }

    fn reconcile_ratings(&self) -> Result<(), ServiceError> {
        let rooms = self
            .rooms
            .read()
            .map_err(|error| ServiceError::Internal(error.to_string()))?
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for room in rooms {
            let mut room = room
                .lock()
                .map_err(|error| ServiceError::Internal(error.to_string()))?;
            self.rate_room(&mut room)?;
        }
        Ok(())
    }

    fn rate_room(&self, room: &mut MatchRoom) -> Result<(), ServiceError> {
        let Some(report) = room.report() else {
            return Ok(());
        };
        if room.rated || room.duplicate {
            return Ok(());
        }
        let mut league = self
            .league
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        let previous = league.clone();
        match league.record(&report) {
            Ok(_) => {
                if let Some(storage) = &self.storage {
                    if let Err(error) = storage.save_league(&league) {
                        *league = previous;
                        return Err(ServiceError::Storage(error.to_string()));
                    }
                }
            }
            Err(LeagueError::DuplicateMatch(_)) => room.duplicate = true,
            Err(error) => return Err(ServiceError::CorruptStorage(error.to_string())),
        }
        if !room.duplicate {
            room.rated = true;
        }
        let _ = self.persist_room(room);
        Ok(())
    }

    fn insert_restored_room(&self, room: MatchRoom) -> Result<(), ServiceError> {
        let mut rooms = self
            .rooms
            .write()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        if rooms.len() >= self.limits.maximum_rooms {
            return Err(ServiceError::Capacity {
                limit: self.limits.maximum_rooms,
            });
        }
        if rooms.contains_key(&room.id) {
            return Err(ServiceError::CorruptStorage(format!(
                "duplicate room {}",
                room.id
            )));
        }
        rooms.insert(room.id.clone(), Arc::new(Mutex::new(room)));
        Ok(())
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
            SeatController::Human {
                token_hash: expected,
            } if *expected == token_hash(token) => Ok(()),
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
            self.rating_status(),
            u32::try_from(self.replay.frames.len()).expect("replay length is capped by u32"),
            &self.game,
        )
        .map_err(|error| ServiceError::Internal(error.to_string()))
    }

    fn stored(&self) -> StoredRoom {
        StoredRoom {
            schema_version: ROOM_STORAGE_SCHEMA_VERSION,
            id: self.id.clone(),
            request: self.request.clone(),
            human_token_hashes: std::array::from_fn(|seat| match &self.seats[seat] {
                SeatController::Human { token_hash } => Some(*token_hash),
                SeatController::Greedy(_) | SeatController::Random(_) => None,
            }),
            replay: self.replay.clone(),
            rated: self.rated,
            duplicate: self.duplicate,
        }
    }

    fn restore(stored: StoredRoom, update_capacity: usize) -> Result<Self, ServiceError> {
        if stored.schema_version != ROOM_STORAGE_SCHEMA_VERSION {
            return Err(ServiceError::CorruptStorage(format!(
                "room {} has schema {}, expected {}",
                stored.id, stored.schema_version, ROOM_STORAGE_SCHEMA_VERSION
            )));
        }
        let rules = Rules::from_profile(stored.request.rules_profile).ok_or_else(|| {
            ServiceError::CorruptStorage("stored room uses custom rules without a document".into())
        })?;
        let scenario = Scenario::symmetric_duel(
            stored.request.width,
            stored.request.height,
            stored.request.seed,
        )
        .map_err(|error| ServiceError::CorruptStorage(error.to_string()))?;
        if stored.replay.header.rules != rules || stored.replay.header.scenario != scenario {
            return Err(ServiceError::CorruptStorage(format!(
                "room {} request differs from its replay header",
                stored.id
            )));
        }
        let verified = stored
            .replay
            .play()
            .map_err(|error| ServiceError::CorruptStorage(error.to_string()))?;
        let mut seats = Self::restore_seats(&stored)?;
        let mut reconstructed = Game::new(rules, scenario)
            .map_err(|error| ServiceError::CorruptStorage(error.to_string()))?;
        let mut legal_actions = Vec::new();
        for frame in &stored.replay.frames {
            reconstructed.legal_actions(&mut legal_actions);
            let active = reconstructed.active_player().index();
            let predicted = match &mut seats[active] {
                SeatController::Human { .. } => None,
                SeatController::Greedy(agent) => {
                    Some(agent.select_action(&reconstructed, &legal_actions))
                }
                SeatController::Random(agent) => {
                    Some(agent.select_action(&reconstructed, &legal_actions))
                }
            };
            if predicted.is_some_and(|action| action != frame.action) {
                return Err(ServiceError::CorruptStorage(format!(
                    "room {} bot action diverged during restoration",
                    stored.id
                )));
            }
            reconstructed
                .step(frame.action)
                .map_err(|error| ServiceError::CorruptStorage(error.to_string()))?;
        }
        if reconstructed != verified.game {
            return Err(ServiceError::CorruptStorage(format!(
                "room {} reconstruction diverged",
                stored.id
            )));
        }
        let revision = u64::try_from(stored.replay.frames.len())
            .map_err(|error| ServiceError::CorruptStorage(error.to_string()))?;
        let action_limit = stored.request.action_limit;
        let (updates, _) = broadcast::channel(update_capacity);
        Ok(Self {
            id: stored.id,
            request: stored.request,
            game: reconstructed,
            replay: stored.replay,
            revision,
            action_limit,
            seats,
            updates,
            closed: false,
            rated: stored.rated,
            duplicate: stored.duplicate,
        })
    }

    fn restore_seats(stored: &StoredRoom) -> Result<[SeatController; 2], ServiceError> {
        let mut restored = Vec::with_capacity(2);
        for (seat, requested) in stored.request.seats.iter().enumerate() {
            let controller = match requested.kind {
                SeatKind::Human => SeatController::Human {
                    token_hash: stored.human_token_hashes[seat].ok_or_else(|| {
                        ServiceError::CorruptStorage(format!(
                            "room {} human seat {seat} has no token hash",
                            stored.id
                        ))
                    })?,
                },
                SeatKind::Greedy => SeatController::Greedy(GreedyAgent::new(&requested.name)),
                SeatKind::Random => SeatController::Random(RandomAgent::new(
                    &requested.name,
                    stored.request.seed ^ (u64::try_from(seat).expect("seat fits in u64") + 1),
                )),
            };
            restored.push(controller);
        }
        restored.try_into().map_err(|_| {
            ServiceError::CorruptStorage(format!("room {} seat count is invalid", stored.id))
        })
    }

    fn rating_status(&self) -> RatingStatus {
        if self.status() == MatchStatus::Running {
            RatingStatus::NotFinished
        } else if self.rated {
            RatingStatus::Recorded
        } else if self.duplicate {
            RatingStatus::Duplicate
        } else {
            RatingStatus::Pending
        }
    }

    fn report(&self) -> Option<MatchReport> {
        let termination = match self.status() {
            MatchStatus::Running => return None,
            MatchStatus::Victory => Termination::Victory,
            MatchStatus::ActionLimit => Termination::ActionLimit,
        };
        let winner = match termination {
            Termination::Victory => self.game.winner(),
            Termination::ActionLimit => adjudicate(&self.game),
        };
        Some(MatchReport {
            agents: [
                self.request.seats[0].name.clone(),
                self.request.seats[1].name.clone(),
            ],
            seed: self.request.seed,
            outcome: MatchOutcome {
                winner,
                actions: u32::try_from(self.replay.frames.len())
                    .expect("replay length is capped by u32"),
                termination,
            },
            replay: self.replay.clone(),
        })
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

fn token_hash(token: &str) -> [u8; 32] {
    *blake3::hash(token.as_bytes()).as_bytes()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use antiyoy_core::{Action, RulesProfile};
    use antiyoy_protocol::{
        CreateMatchRequest, MatchStatus, NETWORK_SCHEMA_VERSION, RatingStatus, Replay, SeatKind,
        SeatRequest, SubmitAction,
    };

    use super::{MatchService, ServiceError, ServiceLimits, random_hex};

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

    fn human_duel() -> CreateMatchRequest {
        let mut request = human_against_random();
        request.seats[1] = SeatRequest {
            name: "second-human".into(),
            kind: SeatKind::Human,
        };
        request
    }

    fn temporary_directory() -> PathBuf {
        std::env::temp_dir().join(format!(
            "antiyoy-server-test-{}",
            random_hex(8).expect("operating system random source")
        ))
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

    #[test]
    fn room_capacity_is_released_only_by_authorized_deletion() {
        let service = MatchService::with_limits(ServiceLimits {
            maximum_rooms: 1,
            ..ServiceLimits::default()
        })
        .expect("valid limits");
        let created = service
            .create_match(&human_against_random())
            .expect("first room fits");
        assert_eq!(
            service
                .create_match(&human_against_random())
                .expect_err("second room exceeds capacity"),
            ServiceError::Capacity { limit: 1 }
        );
        assert_eq!(
            service
                .delete(&created.snapshot.match_id, 0, "invalid")
                .expect_err("invalid token cannot delete"),
            ServiceError::Unauthorized
        );
        service
            .delete(&created.snapshot.match_id, 0, &created.credentials[0].token)
            .expect("authorized deletion");
        assert!(matches!(
            service.snapshot(&created.snapshot.match_id),
            Err(ServiceError::NotFound(_))
        ));
        service
            .create_match(&human_against_random())
            .expect("deleted room releases capacity");
    }

    #[test]
    fn configured_map_and_action_limits_are_enforced() {
        let service = MatchService::with_limits(ServiceLimits {
            maximum_cells: 34,
            maximum_action_limit: 99,
            ..ServiceLimits::default()
        })
        .expect("valid limits");
        let request = human_against_random();
        assert!(matches!(
            service.create_match(&request),
            Err(ServiceError::InvalidRequest(_))
        ));
        let mut request = request;
        request.width = 5;
        request.height = 2;
        assert!(matches!(
            service.create_match(&request),
            Err(ServiceError::InvalidRequest(_))
        ));
        request.action_limit = 99;
        service.create_match(&request).expect("request fits limits");
    }

    #[test]
    fn persistent_room_restores_state_tokens_and_future_actions() {
        let directory = temporary_directory();
        let limits = ServiceLimits::default();
        let service = MatchService::persistent(limits, &directory).expect("writable storage");
        let created = service.create_match(&human_duel()).expect("valid room");
        let first = &created.credentials[0];
        let advanced = service
            .submit(
                &created.snapshot.match_id,
                first.seat,
                &first.token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: created.snapshot.revision,
                    action: created.snapshot.game.legal_actions[0],
                },
            )
            .expect("persisted action");
        let stored_path = directory.join(format!("{}.room", created.snapshot.match_id));
        let stored_bytes = fs::read(stored_path).expect("stored room");
        assert!(
            !stored_bytes
                .windows(first.token.len())
                .any(|window| window == first.token.as_bytes())
        );
        drop(service);

        let restored = MatchService::persistent(limits, &directory).expect("restorable storage");
        let snapshot = restored
            .snapshot(&created.snapshot.match_id)
            .expect("restored room");
        assert_eq!(snapshot, advanced);
        let active = snapshot.game.active_player;
        let credential = created
            .credentials
            .iter()
            .find(|credential| credential.seat == active)
            .expect("active human credential");
        restored
            .submit(
                &snapshot.match_id,
                active,
                &credential.token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: snapshot.revision,
                    action: snapshot.game.legal_actions[0],
                },
            )
            .expect("token remains valid after restart");
        restored
            .delete(&snapshot.match_id, active, &credential.token)
            .expect("persistent deletion");
        drop(restored);

        let empty = MatchService::persistent(limits, &directory).expect("empty storage");
        assert!(matches!(
            empty.snapshot(&created.snapshot.match_id),
            Err(ServiceError::NotFound(_))
        ));
        drop(empty);
        fs::remove_dir_all(directory).expect("remove test storage");
    }

    #[test]
    fn random_bot_rng_stream_survives_restart() {
        let directory = temporary_directory();
        let limits = ServiceLimits::default();
        let request = human_against_random();
        let persistent = MatchService::persistent(limits, &directory).expect("writable storage");
        let created = persistent.create_match(&request).expect("valid room");
        let credential = &created.credentials[0];
        let first_end_turn = created
            .snapshot
            .game
            .legal_actions
            .iter()
            .copied()
            .find(|action| *action == Action::EndTurn)
            .expect("end turn is legal");
        let first = persistent
            .submit(
                &created.snapshot.match_id,
                0,
                &credential.token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: 0,
                    action: first_end_turn,
                },
            )
            .expect("first turn");
        drop(persistent);
        let restored = MatchService::persistent(limits, &directory).expect("restored room");
        let second_end_turn = first
            .game
            .legal_actions
            .iter()
            .copied()
            .find(|action| *action == Action::EndTurn)
            .expect("end turn is legal");
        let resumed = restored
            .submit(
                &first.match_id,
                0,
                &credential.token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: first.revision,
                    action: second_end_turn,
                },
            )
            .expect("resumed turn");

        let reference = MatchService::new();
        let reference_created = reference.create_match(&request).expect("reference room");
        let reference_token = &reference_created.credentials[0].token;
        let reference_first = reference
            .submit(
                &reference_created.snapshot.match_id,
                0,
                reference_token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: 0,
                    action: first_end_turn,
                },
            )
            .expect("reference first turn");
        let reference_resumed = reference
            .submit(
                &reference_first.match_id,
                0,
                reference_token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: reference_first.revision,
                    action: second_end_turn,
                },
            )
            .expect("reference second turn");
        assert_eq!(resumed.digest, reference_resumed.digest);
        assert_eq!(resumed.game, reference_resumed.game);
        drop(restored);
        fs::remove_dir_all(directory).expect("remove test storage");
    }

    #[test]
    fn completed_match_is_rated_once_across_restart() {
        let directory = temporary_directory();
        let limits = ServiceLimits::default();
        let mut request = human_against_random();
        request.seats[0].kind = SeatKind::Greedy;
        request.action_limit = 5;
        let service = MatchService::persistent(limits, &directory).expect("writable storage");
        let created = service.create_match(&request).expect("completed bot room");
        assert_eq!(created.snapshot.status, MatchStatus::ActionLimit);
        assert_eq!(created.snapshot.rating_status, RatingStatus::Recorded);
        let league = service.league().expect("valid league");
        assert_eq!(league.matches.len(), 1);
        assert_eq!(league.standings().len(), 2);
        let duplicate = service
            .create_match(&request)
            .expect("duplicate room remains playable");
        assert_eq!(duplicate.snapshot.rating_status, RatingStatus::Duplicate);
        assert_eq!(service.league().expect("unchanged league").matches.len(), 1);
        drop(service);

        let restored = MatchService::persistent(limits, &directory).expect("restored service");
        let restored_league = restored.league().expect("restored league");
        assert_eq!(restored_league, league);
        assert_eq!(restored_league.matches.len(), 1);
        assert_eq!(
            restored
                .snapshot(&created.snapshot.match_id)
                .expect("restored completed room")
                .rating_status,
            RatingStatus::Recorded
        );
        drop(restored);
        fs::remove_dir_all(directory).expect("remove test storage");
    }
}
