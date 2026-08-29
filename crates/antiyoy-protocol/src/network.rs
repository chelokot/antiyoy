use antiyoy_core::{Action, Game, HexId, Object, Relation, RulesProfile};
use serde::{Deserialize, Serialize};

use crate::{Digest, ReplayError};

pub const NETWORK_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SeatKind {
    Human,
    Greedy,
    Random,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SeatRequest {
    pub name: String,
    pub kind: SeatKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CreateMatchRequest {
    pub schema_version: u16,
    pub rules_profile: RulesProfile,
    pub width: u16,
    pub height: u16,
    pub seed: u64,
    pub seats: [SeatRequest; 2],
    pub action_limit: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SeatCredential {
    pub seat: u8,
    pub name: String,
    pub token: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CreateMatchResponse {
    pub snapshot: MatchSnapshot,
    pub credentials: Vec<SeatCredential>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SubmitAction {
    pub schema_version: u16,
    pub revision: u64,
    pub action: Action,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ClientMessage {
    Authenticate {
        schema_version: u16,
        seat: u8,
        token: String,
    },
    Submit(SubmitAction),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ServerMessage {
    Snapshot(MatchSnapshot),
    Authenticated { seat: u8 },
    Error { code: String, message: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MatchStatus {
    Running,
    Victory,
    ActionLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CellView {
    pub id: u16,
    pub playable: bool,
    pub owner: Option<u8>,
    pub object: Object,
    pub strength: u8,
    pub ready: bool,
    pub province: Option<u16>,
    pub defense: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProvinceView {
    pub id: u16,
    pub owner: u8,
    pub money: i64,
    pub income: i64,
    pub upkeep: i64,
    pub profit: i64,
    pub capital: u16,
    pub size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RelationView {
    pub first: u8,
    pub second: u8,
    pub relation: Relation,
    pub proposal: Option<Relation>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GameView {
    pub width: u16,
    pub height: u16,
    pub round: u32,
    pub active_player: u8,
    pub terminal: bool,
    pub winner: Option<u8>,
    pub cells: Vec<CellView>,
    pub provinces: Vec<ProvinceView>,
    pub relations: Vec<RelationView>,
    pub legal_actions: Vec<Action>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MatchSnapshot {
    pub schema_version: u16,
    pub match_id: String,
    pub revision: u64,
    pub status: MatchStatus,
    pub actions_played: u32,
    pub digest: Digest,
    pub game: GameView,
}

impl GameView {
    pub fn from_game(game: &Game) -> Self {
        let mut legal_actions = Vec::new();
        game.legal_actions(&mut legal_actions);
        let topology = game.topology();
        let cells = game
            .cells()
            .iter()
            .copied()
            .zip(0_u16..)
            .map(|(cell, raw_id)| {
                let id = HexId(raw_id);
                let playable = topology.is_playable(id);
                CellView {
                    id: id.0,
                    playable,
                    owner: (playable && !cell.owner().is_neutral()).then_some(cell.owner().0),
                    object: cell.object(),
                    strength: cell.unit().strength(),
                    ready: cell.unit().is_ready(),
                    province: cell.province().is_some().then_some(cell.province().0),
                    defense: if playable {
                        game.hex_defense(id).unwrap_or_default()
                    } else {
                        0
                    },
                }
            })
            .collect();
        let provinces = game
            .provinces()
            .iter()
            .map(|province| ProvinceView {
                id: province.id().0,
                owner: province.owner().0,
                money: province.money(),
                income: game.province_income(province.id()).unwrap_or_default(),
                upkeep: game.province_upkeep(province.id()).unwrap_or_default(),
                profit: game.province_profit(province.id()).unwrap_or_default(),
                capital: province.capital().0,
                size: province.hexes().len(),
            })
            .collect();
        let mut relations = Vec::with_capacity(usize::from(game.player_count()).pow(2));
        for first in 0..game.player_count() {
            for second in 0..game.player_count() {
                let first_player = antiyoy_core::PlayerId(first);
                let second_player = antiyoy_core::PlayerId(second);
                if let Some(relation) = game.relation(first_player, second_player) {
                    relations.push(RelationView {
                        first,
                        second,
                        relation,
                        proposal: game.proposal(first_player, second_player),
                    });
                }
            }
        }
        Self {
            width: topology.width(),
            height: topology.height(),
            round: game.round(),
            active_player: game.active_player().0,
            terminal: game.is_terminal(),
            winner: game.winner().map(|winner| winner.0),
            cells,
            provinces,
            relations,
            legal_actions,
        }
    }
}

impl MatchSnapshot {
    pub fn from_game(
        match_id: String,
        revision: u64,
        status: MatchStatus,
        actions_played: u32,
        game: &Game,
    ) -> Result<Self, ReplayError> {
        Ok(Self {
            schema_version: NETWORK_SCHEMA_VERSION,
            match_id,
            revision,
            status,
            actions_played,
            digest: Digest::of_game(game)?,
            game: GameView::from_game(game),
        })
    }
}
