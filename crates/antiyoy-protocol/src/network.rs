use antiyoy_core::{Action, Game, GeneratorConfig, HexId, Object, Relation, RulesProfile};
use serde::{Deserialize, Serialize};

use crate::{Digest, ReplayError};

pub const NETWORK_SCHEMA_VERSION: u16 = 6;
pub const MINIMUM_MATCH_PLAYERS: usize = 2;
pub const MAXIMUM_MATCH_PLAYERS: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SeatKind {
    Human,
    Greedy,
    Random,
    Search,
    Open,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SeatRequest {
    pub name: String,
    pub kind: SeatKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MatchScenario {
    SymmetricDuel {
        width: u16,
        height: u16,
        #[serde(with = "u64_wire")]
        seed: u64,
    },
    Procedural(#[serde(with = "generator_config_wire")] GeneratorConfig),
}

mod u64_wire {
    use serde::{Deserialize, Deserializer, Serializer, de};

    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            serializer.serialize_str(&value.to_string())
        } else {
            serializer.serialize_u64(*value)
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            String::deserialize(deserializer)?
                .parse()
                .map_err(de::Error::custom)
        } else {
            u64::deserialize(deserializer)
        }
    }
}

mod generator_config_wire {
    use antiyoy_core::GeneratorConfig;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

    #[derive(Serialize)]
    struct SerializableGeneratorConfig {
        schema_version: u16,
        width: u16,
        height: u16,
        players: u8,
        seed: String,
        land_density_per_million: u32,
        starting_province_size: u16,
        starting_money: i64,
        tree_density_per_million: u32,
        neutral_tower_density_per_million: u32,
        neutral_capital_density_per_million: u32,
        grave_density_per_million: u32,
    }

    #[derive(Deserialize)]
    struct DeserializableGeneratorConfig {
        schema_version: u16,
        width: u16,
        height: u16,
        players: u8,
        seed: String,
        land_density_per_million: u32,
        starting_province_size: u16,
        starting_money: i64,
        tree_density_per_million: u32,
        neutral_tower_density_per_million: u32,
        neutral_capital_density_per_million: u32,
        grave_density_per_million: u32,
    }

    pub fn serialize<S>(config: &GeneratorConfig, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if !serializer.is_human_readable() {
            return config.serialize(serializer);
        }
        SerializableGeneratorConfig {
            schema_version: config.schema_version,
            width: config.width,
            height: config.height,
            players: config.players,
            seed: config.seed.to_string(),
            land_density_per_million: config.land_density_per_million,
            starting_province_size: config.starting_province_size,
            starting_money: config.starting_money,
            tree_density_per_million: config.tree_density_per_million,
            neutral_tower_density_per_million: config.neutral_tower_density_per_million,
            neutral_capital_density_per_million: config.neutral_capital_density_per_million,
            grave_density_per_million: config.grave_density_per_million,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<GeneratorConfig, D::Error>
    where
        D: Deserializer<'de>,
    {
        if !deserializer.is_human_readable() {
            return GeneratorConfig::deserialize(deserializer);
        }
        let config = DeserializableGeneratorConfig::deserialize(deserializer)?;
        Ok(GeneratorConfig {
            schema_version: config.schema_version,
            width: config.width,
            height: config.height,
            players: config.players,
            seed: config.seed.parse().map_err(de::Error::custom)?,
            land_density_per_million: config.land_density_per_million,
            starting_province_size: config.starting_province_size,
            starting_money: config.starting_money,
            tree_density_per_million: config.tree_density_per_million,
            neutral_tower_density_per_million: config.neutral_tower_density_per_million,
            neutral_capital_density_per_million: config.neutral_capital_density_per_million,
            grave_density_per_million: config.grave_density_per_million,
        })
    }
}

impl MatchScenario {
    pub const fn width(&self) -> u16 {
        match self {
            Self::SymmetricDuel { width, .. } => *width,
            Self::Procedural(config) => config.width,
        }
    }

    pub const fn height(&self) -> u16 {
        match self {
            Self::SymmetricDuel { height, .. } => *height,
            Self::Procedural(config) => config.height,
        }
    }

    pub const fn seed(&self) -> u64 {
        match self {
            Self::SymmetricDuel { seed, .. } => *seed,
            Self::Procedural(config) => config.seed,
        }
    }

    pub const fn players(&self) -> u8 {
        match self {
            Self::SymmetricDuel { .. } => 2,
            Self::Procedural(config) => config.players,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CreateMatchRequest {
    pub schema_version: u16,
    pub rules_profile: RulesProfile,
    pub scenario: MatchScenario,
    pub seats: Vec<SeatRequest>,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ClaimSeatRequest {
    pub schema_version: u16,
    pub name: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ClaimSeatResponse {
    pub snapshot: MatchSnapshot,
    pub credential: SeatCredential,
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
    Snapshot(Box<MatchSnapshot>),
    Authenticated { seat: u8 },
    Error { code: String, message: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MatchStatus {
    Waiting,
    Running,
    Victory,
    ActionLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RatingStatus {
    NotFinished,
    Pending,
    Recorded,
    Duplicate,
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
    pub rating_status: RatingStatus,
    pub actions_played: u32,
    pub digest: Digest,
    pub rules_profile: RulesProfile,
    pub scenario: MatchScenario,
    pub seats: Vec<SeatRequest>,
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
        rating_status: RatingStatus,
        actions_played: u32,
        request: &CreateMatchRequest,
        game: &Game,
    ) -> Result<Self, ReplayError> {
        let mut game_view = GameView::from_game(game);
        if status == MatchStatus::Waiting {
            game_view.legal_actions.clear();
        }
        Ok(Self {
            schema_version: NETWORK_SCHEMA_VERSION,
            match_id,
            revision,
            status,
            rating_status,
            actions_played,
            digest: Digest::of_game(game)?,
            rules_profile: request.rules_profile,
            scenario: request.scenario.clone(),
            seats: request.seats.clone(),
            game: game_view,
        })
    }
}

#[cfg(test)]
mod tests {
    use antiyoy_core::{GENERATOR_SCHEMA_VERSION, GeneratorConfig, RulesProfile};

    use super::{CreateMatchRequest, MatchScenario, NETWORK_SCHEMA_VERSION, SeatKind, SeatRequest};

    #[test]
    fn procedural_match_request_round_trips_every_seat() {
        let request = CreateMatchRequest {
            schema_version: NETWORK_SCHEMA_VERSION,
            rules_profile: RulesProfile::OnlineDefaultV1,
            scenario: MatchScenario::Procedural(GeneratorConfig {
                schema_version: GENERATOR_SCHEMA_VERSION,
                width: 21,
                height: 15,
                players: 4,
                seed: u64::MAX,
                ..GeneratorConfig::default()
            }),
            seats: (0..4)
                .map(|seat| SeatRequest {
                    name: format!("player-{seat}"),
                    kind: SeatKind::Human,
                })
                .collect(),
            action_limit: 2_000,
        };
        let encoded = serde_json::to_vec(&request).expect("serializable request");
        let json: serde_json::Value = serde_json::from_slice(&encoded).expect("json request");
        assert_eq!(json["scenario"]["Procedural"]["seed"], u64::MAX.to_string());
        let decoded: CreateMatchRequest =
            serde_json::from_slice(&encoded).expect("deserializable request");
        assert_eq!(decoded, request);
        let binary = postcard::to_allocvec(&request).expect("binary request");
        assert_eq!(
            postcard::from_bytes::<CreateMatchRequest>(&binary).expect("binary round trip"),
            request
        );
        assert_eq!(request.scenario.players(), 4);
        assert_eq!(request.scenario.width(), 21);
        assert_eq!(request.scenario.height(), 15);
        assert_eq!(request.scenario.seed(), u64::MAX);
    }
}
