#![forbid(unsafe_code)]

mod storage;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use antiyoy_agents::{Agent, GreedyAgent, RandomAgent, SearchAgent, SearchConfig};
use antiyoy_core::{Action, Game, Rules, Scenario};
use antiyoy_eval::{League, LeagueError, MatchOutcome, MatchReport, Termination, adjudicate};
use antiyoy_protocol::{
    ClaimSeatRequest, ClaimSeatResponse, CreateMatchRequest, CreateMatchResponse,
    MAXIMUM_MATCH_PLAYERS, MINIMUM_MATCH_PLAYERS, MatchScenario, MatchSnapshot, MatchStatus,
    NETWORK_SCHEMA_VERSION, RatingStatus, Replay, SeatCredential, SeatKind, SubmitAction,
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
    pub search_nodes: usize,
}

#[derive(Clone, Debug)]
struct MatchRoom {
    id: String,
    request: CreateMatchRequest,
    game: Game,
    replay: Replay,
    revision: u64,
    action_limit: u32,
    seats: Vec<SeatController>,
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
    Search(SearchAgent),
    Open,
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
    #[error("seat {seat} is not available to claim")]
    SeatUnavailable { seat: u8 },
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
            search_nodes: 2_048,
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
            || limits.search_nodes < 2
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
        let scenario = scenario_from_request(request).map_err(ServiceError::InvalidRequest)?;
        let (replay, game) = Replay::new(rules, scenario)
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        let (updates, _) = broadcast::channel(self.limits.update_capacity);
        let mut credentials = Vec::new();
        let seats = request
            .seats
            .iter()
            .enumerate()
            .map(|(seat, _)| self.seat_controller(request, seat, &mut credentials))
            .collect::<Result<Vec<_>, _>>()?;
        let mut room = MatchRoom {
            id: String::new(),
            request: request.clone(),
            game,
            replay,
            revision: 0,
            action_limit: request.action_limit,
            seats,
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

    pub fn claim_seat(
        &self,
        match_id: &str,
        seat: u8,
        request: ClaimSeatRequest,
    ) -> Result<ClaimSeatResponse, ServiceError> {
        if request.schema_version != NETWORK_SCHEMA_VERSION {
            return Err(ServiceError::InvalidRequest(format!(
                "network schema {} is unsupported",
                request.schema_version
            )));
        }
        validate_seat_name(&request.name)?;
        let room = self.room(match_id)?;
        let mut room = room
            .lock()
            .map_err(|error| ServiceError::Internal(error.to_string()))?;
        if room.closed {
            return Err(ServiceError::Closed);
        }
        let seat_index = usize::from(seat);
        if !matches!(room.seats.get(seat_index), Some(SeatController::Open)) {
            return Err(ServiceError::SeatUnavailable { seat });
        }
        if room
            .request
            .seats
            .iter()
            .enumerate()
            .any(|(index, existing)| index != seat_index && existing.name == request.name)
        {
            return Err(ServiceError::InvalidRequest(
                "seat names must be distinct".into(),
            ));
        }
        let token = random_hex(TOKEN_BYTES)?;
        let previous = room.clone();
        room.request.seats[seat_index] = antiyoy_protocol::SeatRequest {
            name: request.name.clone(),
            kind: SeatKind::Human,
        };
        room.seats[seat_index] = SeatController::Human {
            token_hash: token_hash(&token),
        };
        room.advance_bots()?;
        if let Err(error) = self.persist_room(&room) {
            *room = previous;
            return Err(error);
        }
        let snapshot = room.snapshot()?;
        let _ = room.updates.send(snapshot.clone());
        Ok(ClaimSeatResponse {
            snapshot,
            credential: SeatCredential {
                seat,
                name: request.name,
                token,
            },
        })
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
        if !(MINIMUM_MATCH_PLAYERS..=MAXIMUM_MATCH_PLAYERS).contains(&request.seats.len()) {
            return Err(ServiceError::InvalidRequest(format!(
                "seat count must be within {MINIMUM_MATCH_PLAYERS}..={MAXIMUM_MATCH_PLAYERS}"
            )));
        }
        if usize::from(request.scenario.players()) != request.seats.len() {
            return Err(ServiceError::InvalidRequest(format!(
                "scenario has {} players but {} seats were supplied",
                request.scenario.players(),
                request.seats.len()
            )));
        }
        let cells = usize::from(request.scenario.width()) * usize::from(request.scenario.height());
        if cells > self.limits.maximum_cells {
            return Err(ServiceError::InvalidRequest(format!(
                "map contains {cells} cells, limit is {}",
                self.limits.maximum_cells
            )));
        }
        for seat in &request.seats {
            validate_seat_name(&seat.name)?;
        }
        let mut names = std::collections::BTreeSet::new();
        for seat in &request.seats {
            if !names.insert(&seat.name) {
                return Err(ServiceError::InvalidRequest(
                    "seat names must be distinct".into(),
                ));
            }
        }
        Ok(())
    }

    fn seat_controller(
        &self,
        request: &CreateMatchRequest,
        seat: usize,
        credentials: &mut Vec<SeatCredential>,
    ) -> Result<SeatController, ServiceError> {
        let requested = &request.seats[seat];
        Ok(match requested.kind {
            SeatKind::Human => {
                let token = random_hex(TOKEN_BYTES)?;
                credentials.push(SeatCredential {
                    seat: u8::try_from(seat).expect("match seat count fits in u8"),
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
                request.scenario.seed() ^ (u64::try_from(seat).expect("seat fits in u64") + 1),
            )),
            SeatKind::Search => SeatController::Search(
                SearchAgent::with_config(
                    &requested.name,
                    SearchConfig {
                        node_budget: self.limits.search_nodes,
                        ..SearchConfig::default()
                    },
                )
                .expect("service limits contain a valid search budget"),
            ),
            SeatKind::Open => SeatController::Open,
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
            let mut room = MatchRoom::restore(
                stored,
                self.limits.update_capacity,
                self.limits.search_nodes,
            )?;
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
                SeatController::Human { .. } | SeatController::Open => break,
                SeatController::Greedy(agent) => agent.select_action(&self.game, &legal_actions),
                SeatController::Random(agent) => agent.select_action(&self.game, &legal_actions),
                SeatController::Search(agent) => agent.select_action(&self.game, &legal_actions),
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
            | SeatController::Random(_)
            | SeatController::Search(_)
            | SeatController::Open => Err(ServiceError::Unauthorized),
        }
    }

    fn status(&self) -> MatchStatus {
        if self
            .seats
            .iter()
            .any(|seat| matches!(seat, SeatController::Open))
        {
            MatchStatus::Waiting
        } else if self.game.is_terminal() {
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
            &self.request,
            &self.game,
        )
        .map_err(|error| ServiceError::Internal(error.to_string()))
    }

    fn stored(&self) -> StoredRoom {
        StoredRoom {
            schema_version: ROOM_STORAGE_SCHEMA_VERSION,
            id: self.id.clone(),
            request: self.request.clone(),
            human_token_hashes: self
                .seats
                .iter()
                .map(|seat| match seat {
                    SeatController::Human { token_hash } => Some(*token_hash),
                    SeatController::Greedy(_)
                    | SeatController::Random(_)
                    | SeatController::Search(_)
                    | SeatController::Open => None,
                })
                .collect(),
            replay: self.replay.clone(),
            rated: self.rated,
            duplicate: self.duplicate,
        }
    }

    fn restore(
        stored: StoredRoom,
        update_capacity: usize,
        search_nodes: usize,
    ) -> Result<Self, ServiceError> {
        if stored.schema_version != ROOM_STORAGE_SCHEMA_VERSION {
            return Err(ServiceError::CorruptStorage(format!(
                "room {} has schema {}, expected {}",
                stored.id, stored.schema_version, ROOM_STORAGE_SCHEMA_VERSION
            )));
        }
        let rules = Rules::from_profile(stored.request.rules_profile).ok_or_else(|| {
            ServiceError::CorruptStorage("stored room uses custom rules without a document".into())
        })?;
        let scenario =
            scenario_from_request(&stored.request).map_err(ServiceError::CorruptStorage)?;
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
        let mut seats = Self::restore_seats(&stored, search_nodes)?;
        let mut reconstructed = Game::new(rules, scenario)
            .map_err(|error| ServiceError::CorruptStorage(error.to_string()))?;
        let mut legal_actions = Vec::new();
        for frame in &stored.replay.frames {
            reconstructed.legal_actions(&mut legal_actions);
            let active = reconstructed.active_player().index();
            let predicted = match &mut seats[active] {
                SeatController::Human { .. } | SeatController::Open => None,
                SeatController::Greedy(agent) => {
                    Some(agent.select_action(&reconstructed, &legal_actions))
                }
                SeatController::Random(agent) => {
                    Some(agent.select_action(&reconstructed, &legal_actions))
                }
                SeatController::Search(agent) => {
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

    fn restore_seats(
        stored: &StoredRoom,
        search_nodes: usize,
    ) -> Result<Vec<SeatController>, ServiceError> {
        if stored.human_token_hashes.len() != stored.request.seats.len() {
            return Err(ServiceError::CorruptStorage(format!(
                "room {} credential count differs from its seat count",
                stored.id
            )));
        }
        let mut restored = Vec::with_capacity(stored.request.seats.len());
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
                    stored.request.scenario.seed()
                        ^ (u64::try_from(seat).expect("seat fits in u64") + 1),
                )),
                SeatKind::Search => SeatController::Search(
                    SearchAgent::with_config(
                        &requested.name,
                        SearchConfig {
                            node_budget: search_nodes,
                            ..SearchConfig::default()
                        },
                    )
                    .expect("service limits contain a valid search budget"),
                ),
                SeatKind::Open => {
                    if stored.human_token_hashes[seat].is_some() {
                        return Err(ServiceError::CorruptStorage(format!(
                            "room {} open seat {seat} has a token hash",
                            stored.id
                        )));
                    }
                    SeatController::Open
                }
            };
            restored.push(controller);
        }
        Ok(restored)
    }

    fn rating_status(&self) -> RatingStatus {
        if matches!(self.status(), MatchStatus::Waiting | MatchStatus::Running) {
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
            MatchStatus::Waiting | MatchStatus::Running => return None,
            MatchStatus::Victory => Termination::Victory,
            MatchStatus::ActionLimit => Termination::ActionLimit,
        };
        let winner = match termination {
            Termination::Victory => self.game.winner(),
            Termination::ActionLimit => adjudicate(&self.game),
        };
        Some(MatchReport {
            agents: self
                .request
                .seats
                .iter()
                .map(|seat| seat.name.clone())
                .collect(),
            seed: self.request.scenario.seed(),
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

fn scenario_from_request(request: &CreateMatchRequest) -> Result<Scenario, String> {
    match &request.scenario {
        MatchScenario::SymmetricDuel {
            width,
            height,
            seed,
        } => Scenario::symmetric_duel(*width, *height, *seed).map_err(|error| error.to_string()),
        MatchScenario::Procedural(config) => config.generate().map_err(|error| error.to_string()),
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

fn validate_seat_name(name: &str) -> Result<(), ServiceError> {
    if name.is_empty() || name.len() > 64 {
        return Err(ServiceError::InvalidRequest(
            "seat names must contain 1..=64 bytes".into(),
        ));
    }
    Ok(())
}

fn token_hash(token: &str) -> [u8; 32] {
    *blake3::hash(token.as_bytes()).as_bytes()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use antiyoy_core::{Action, GENERATOR_SCHEMA_VERSION, GeneratorConfig, RulesProfile};
    use antiyoy_protocol::{
        ClaimSeatRequest, CreateMatchRequest, MatchScenario, MatchStatus, NETWORK_SCHEMA_VERSION,
        RatingStatus, Replay, SeatKind, SeatRequest, SubmitAction,
    };

    use super::storage::{LegacyCreateMatchRequest, LegacyStoredRoom, StoredRoom};
    use super::{MatchService, ServiceError, ServiceLimits, random_hex};

    fn human_against_random() -> CreateMatchRequest {
        CreateMatchRequest {
            schema_version: NETWORK_SCHEMA_VERSION,
            rules_profile: RulesProfile::OnlineDuelV1,
            scenario: MatchScenario::SymmetricDuel {
                width: 7,
                height: 5,
                seed: 47,
            },
            seats: vec![
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

    fn joinable_duel() -> CreateMatchRequest {
        let mut request = human_duel();
        request.seats[1] = SeatRequest {
            name: "open-seat-2".into(),
            kind: SeatKind::Open,
        };
        request
    }

    fn human_against_search() -> CreateMatchRequest {
        let mut request = human_against_random();
        request.seats[1] = SeatRequest {
            name: "search".into(),
            kind: SeatKind::Search,
        };
        request
    }

    fn four_player_procedural() -> CreateMatchRequest {
        CreateMatchRequest {
            schema_version: NETWORK_SCHEMA_VERSION,
            rules_profile: RulesProfile::OnlineDefaultV1,
            scenario: MatchScenario::Procedural(GeneratorConfig {
                schema_version: GENERATOR_SCHEMA_VERSION,
                width: 17,
                height: 13,
                players: 4,
                seed: 700,
                land_density_per_million: 650_000,
                ..GeneratorConfig::default()
            }),
            seats: vec![
                SeatRequest {
                    name: "human-four".into(),
                    kind: SeatKind::Human,
                },
                SeatRequest {
                    name: "random-four".into(),
                    kind: SeatKind::Random,
                },
                SeatRequest {
                    name: "greedy-four".into(),
                    kind: SeatKind::Greedy,
                },
                SeatRequest {
                    name: "search-four".into(),
                    kind: SeatKind::Search,
                },
            ],
            action_limit: 1_000,
        }
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
    fn open_seat_blocks_play_until_exactly_one_guest_claims_it() {
        let service = MatchService::new();
        let created = service.create_match(&joinable_duel()).expect("valid room");
        assert_eq!(created.snapshot.status, MatchStatus::Waiting);
        assert!(created.snapshot.game.legal_actions.is_empty());
        assert_eq!(created.credentials.len(), 1);
        let (_, mut updates) = service
            .subscribe(&created.snapshot.match_id)
            .expect("subscribable room");
        let duplicate_name = service
            .claim_seat(
                &created.snapshot.match_id,
                1,
                ClaimSeatRequest {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    name: created.credentials[0].name.clone(),
                },
            )
            .expect_err("seat names remain unique");
        assert!(matches!(duplicate_name, ServiceError::InvalidRequest(_)));
        let joined = service
            .claim_seat(
                &created.snapshot.match_id,
                1,
                ClaimSeatRequest {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    name: "guest".into(),
                },
            )
            .expect("open seat claimed");
        assert_eq!(joined.credential.seat, 1);
        assert_eq!(joined.credential.name, "guest");
        assert_eq!(joined.snapshot.status, MatchStatus::Running);
        assert!(!joined.snapshot.game.legal_actions.is_empty());
        assert_eq!(joined.snapshot.seats[1].kind, SeatKind::Human);
        assert_eq!(
            updates.try_recv().expect("claim broadcast"),
            joined.snapshot
        );
        assert_eq!(
            service
                .claim_seat(
                    &created.snapshot.match_id,
                    1,
                    ClaimSeatRequest {
                        schema_version: NETWORK_SCHEMA_VERSION,
                        name: "late-guest".into(),
                    },
                )
                .expect_err("claimed seat cannot be stolen"),
            ServiceError::SeatUnavailable { seat: 1 }
        );
    }

    #[test]
    fn procedural_multiplayer_room_advances_every_bot_and_verifies_replay() {
        let service = MatchService::with_limits(ServiceLimits {
            search_nodes: 32,
            ..ServiceLimits::default()
        })
        .expect("valid service limits");
        let request = four_player_procedural();
        let created = service
            .create_match(&request)
            .expect("valid four-player room");
        assert_eq!(created.credentials.len(), 1);
        assert_eq!(created.snapshot.seats.len(), 4);
        assert_eq!(created.snapshot.rules_profile, request.rules_profile);
        assert_eq!(created.snapshot.scenario, request.scenario);
        assert_eq!(created.snapshot.game.relations.len(), 16);
        assert_eq!(created.snapshot.game.active_player, 0);
        let owners = created
            .snapshot
            .game
            .cells
            .iter()
            .filter_map(|cell| cell.owner)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(owners, std::collections::BTreeSet::from([0, 1, 2, 3]));
        let end_turn = created
            .snapshot
            .game
            .legal_actions
            .iter()
            .copied()
            .find(|action| *action == Action::EndTurn)
            .expect("end turn is legal");
        let advanced = service
            .submit(
                &created.snapshot.match_id,
                0,
                &created.credentials[0].token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: created.snapshot.revision,
                    action: end_turn,
                },
            )
            .expect("all three bots reply");
        assert_eq!(advanced.game.active_player, 0);
        assert!(advanced.revision >= 4);
        let replay = Replay::decode(
            &service
                .replay(&advanced.match_id)
                .expect("encodable multiplayer replay"),
        )
        .expect("decodable multiplayer replay");
        assert_eq!(
            replay
                .verify()
                .expect("verified multiplayer replay")
                .final_digest,
            advanced.digest
        );
    }

    #[test]
    fn every_bundled_profile_creates_a_four_player_procedural_room() {
        let service = MatchService::new();
        for profile in [
            RulesProfile::ClassicGeneric,
            RulesProfile::ClassicSlay,
            RulesProfile::OnlineDefaultV1,
            RulesProfile::OnlineClassicV1,
            RulesProfile::OnlineDuelV1,
            RulesProfile::OnlineExperimentalV1,
            RulesProfile::OnlineExperimentalV2_260801,
        ] {
            let mut request = four_player_procedural();
            request.rules_profile = profile;
            let created = service
                .create_match(&request)
                .expect("bundled profile creates a multiplayer room");
            assert_eq!(created.snapshot.rules_profile, profile);
            assert_eq!(created.snapshot.scenario.players(), 4);
        }
    }

    #[test]
    fn multiplayer_seat_count_and_generator_players_must_match() {
        let service = MatchService::new();
        let mut request = four_player_procedural();
        request.seats.pop();
        assert!(matches!(
            service.create_match(&request),
            Err(ServiceError::InvalidRequest(_))
        ));
        request.seats.extend((3..9).map(|seat| SeatRequest {
            name: format!("overflow-{seat}"),
            kind: SeatKind::Random,
        }));
        assert!(matches!(
            service.create_match(&request),
            Err(ServiceError::InvalidRequest(_))
        ));
    }

    #[test]
    fn search_bot_reply_and_restoration_are_replay_verified() {
        let directory = temporary_directory();
        let limits = ServiceLimits {
            search_nodes: 256,
            ..ServiceLimits::default()
        };
        let service = MatchService::persistent(limits, &directory).expect("writable storage");
        let created = service
            .create_match(&human_against_search())
            .expect("valid search match");
        let credential = &created.credentials[0];
        let end_turn = created
            .snapshot
            .game
            .legal_actions
            .iter()
            .copied()
            .find(|action| *action == Action::EndTurn)
            .expect("end turn is legal");
        let advanced = service
            .submit(
                &created.snapshot.match_id,
                credential.seat,
                &credential.token,
                SubmitAction {
                    schema_version: NETWORK_SCHEMA_VERSION,
                    revision: created.snapshot.revision,
                    action: end_turn,
                },
            )
            .expect("search bot replies");
        assert!(advanced.revision > 1);
        assert_eq!(advanced.game.active_player, 0);
        drop(service);

        let restored = MatchService::persistent(limits, &directory).expect("restorable storage");
        assert_eq!(
            restored
                .snapshot(&created.snapshot.match_id)
                .expect("restored search room"),
            advanced
        );
        let replay = Replay::decode(
            &restored
                .replay(&created.snapshot.match_id)
                .expect("encoded search replay"),
        )
        .expect("decoded search replay");
        assert_eq!(
            replay
                .verify()
                .expect("verified search replay")
                .final_digest,
            advanced.digest
        );
        drop(restored);
        fs::remove_dir_all(directory).expect("remove test storage");
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
        assert!(matches!(
            MatchService::with_limits(ServiceLimits {
                search_nodes: 1,
                ..ServiceLimits::default()
            }),
            Err(ServiceError::InvalidRequest(_))
        ));
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
        request.scenario = MatchScenario::SymmetricDuel {
            width: 5,
            height: 2,
            seed: 47,
        };
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
    fn legacy_fixed_seat_room_files_are_upgraded_during_restoration() {
        let directory = temporary_directory();
        let limits = ServiceLimits::default();
        let service = MatchService::persistent(limits, &directory).expect("writable storage");
        let created = service.create_match(&human_duel()).expect("valid room");
        drop(service);
        let stored_path = directory.join(format!("{}.room", created.snapshot.match_id));
        let stored: StoredRoom =
            postcard::from_bytes(&fs::read(&stored_path).expect("stored room"))
                .expect("decoded stored room");
        let MatchScenario::SymmetricDuel {
            width,
            height,
            seed,
        } = stored.request.scenario
        else {
            panic!("legacy fixture is a duel");
        };
        let legacy = LegacyStoredRoom {
            schema_version: 1,
            id: stored.id,
            request: LegacyCreateMatchRequest {
                schema_version: 3,
                rules_profile: stored.request.rules_profile,
                width,
                height,
                seed,
                seats: stored
                    .request
                    .seats
                    .try_into()
                    .expect("legacy fixture has two seats"),
                action_limit: stored.request.action_limit,
            },
            human_token_hashes: stored
                .human_token_hashes
                .try_into()
                .expect("legacy fixture has two credentials"),
            replay: stored.replay,
            rated: stored.rated,
            duplicate: stored.duplicate,
        };
        fs::write(
            &stored_path,
            postcard::to_allocvec(&legacy).expect("encoded legacy room"),
        )
        .expect("legacy room written");

        let restored = MatchService::persistent(limits, &directory).expect("legacy room restored");
        assert_eq!(
            restored
                .snapshot(&created.snapshot.match_id)
                .expect("upgraded room")
                .schema_version,
            NETWORK_SCHEMA_VERSION
        );
        drop(restored);
        fs::remove_dir_all(directory).expect("remove test storage");
    }

    #[test]
    fn supported_network_schema_rooms_are_upgraded_during_restoration() {
        for previous_schema in [4, 5] {
            let directory = temporary_directory();
            let limits = ServiceLimits::default();
            let service = MatchService::persistent(limits, &directory).expect("writable storage");
            let created = service.create_match(&joinable_duel()).expect("valid room");
            drop(service);
            let stored_path = directory.join(format!("{}.room", created.snapshot.match_id));
            let mut stored: StoredRoom =
                postcard::from_bytes(&fs::read(&stored_path).expect("stored room"))
                    .expect("decoded stored room");
            stored.request.schema_version = previous_schema;
            fs::write(
                &stored_path,
                postcard::to_allocvec(&stored).expect("encoded previous room"),
            )
            .expect("previous room written");

            let restored = MatchService::persistent(limits, &directory).expect("room restored");
            let snapshot = restored
                .snapshot(&created.snapshot.match_id)
                .expect("upgraded room");
            assert_eq!(snapshot.schema_version, NETWORK_SCHEMA_VERSION);
            assert_eq!(snapshot.status, MatchStatus::Waiting);
            drop(restored);
            fs::remove_dir_all(directory).expect("remove test storage");
        }
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

    #[test]
    fn completed_multiplayer_match_updates_every_rating_once() {
        let mut request = four_player_procedural();
        request.action_limit = 8;
        for seat in &mut request.seats {
            seat.kind = SeatKind::Greedy;
        }
        let service = MatchService::new();
        let created = service
            .create_match(&request)
            .expect("completed multiplayer room");
        assert_eq!(created.snapshot.status, MatchStatus::ActionLimit);
        assert_eq!(created.snapshot.rating_status, RatingStatus::Recorded);
        let league = service.league().expect("valid multiplayer league");
        let standings = league.standings();
        assert_eq!(league.matches.len(), 1);
        assert_eq!(standings.len(), 4);
        assert!(standings.iter().all(|standing| standing.rating.games == 1));
        let rating_total = standings
            .iter()
            .map(|standing| standing.rating.elo)
            .sum::<f64>();
        assert!((rating_total - 4_000.0).abs() < 1e-9);
    }
}
