#![forbid(unsafe_code)]

use antiyoy_agents::{Agent, GreedyAgent, RandomAgent};
use antiyoy_core::{Action, Game, HexId, Object, PlayerId, Rules, Scenario};
use antiyoy_protocol::Replay;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct CellView {
    id: u16,
    owner: Option<u8>,
    object: Object,
    strength: u8,
    ready: bool,
    province: Option<u16>,
    defense: u8,
}

#[derive(Serialize)]
struct ProvinceView {
    id: u16,
    owner: u8,
    money: i64,
    income: i64,
    upkeep: i64,
    profit: i64,
    capital: u16,
    size: usize,
}

#[derive(Serialize)]
struct StateView {
    width: u16,
    height: u16,
    round: u32,
    active_player: u8,
    terminal: bool,
    winner: Option<u8>,
    cells: Vec<CellView>,
    provinces: Vec<ProvinceView>,
    legal_actions: usize,
}

#[derive(Serialize)]
struct ReplayMetadata {
    seed: String,
    frames: usize,
    engine_version: u16,
    format_version: u16,
    rules_profile: &'static str,
}

#[wasm_bindgen]
pub struct WasmGame {
    seed: u64,
    rules: Rules,
    scenario: Scenario,
    game: Game,
    legal_actions: Vec<Action>,
    greedy: GreedyAgent,
    random: RandomAgent,
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u16, height: u16, seed: u64) -> Result<Self, JsError> {
        let rules = Rules::classic_generic();
        let scenario = Scenario::symmetric_duel(width, height, seed)
            .map_err(|error| JsError::new(&error.to_string()))?;
        let game = Game::new(rules.clone(), scenario.clone())
            .map_err(|error| JsError::new(&error.to_string()))?;
        Ok(Self {
            seed,
            rules,
            scenario,
            game,
            legal_actions: Vec::new(),
            greedy: GreedyAgent::new("greedy"),
            random: RandomAgent::new("random", seed ^ 1),
        })
    }

    pub fn reset(&mut self) -> Result<String, JsError> {
        self.game = Game::new(self.rules.clone(), self.scenario.clone())
            .map_err(|error| JsError::new(&error.to_string()))?;
        self.random = RandomAgent::new("random", self.seed ^ 1);
        self.state_json()
    }

    pub fn state_json(&mut self) -> Result<String, JsError> {
        state_json(&self.game, &mut self.legal_actions)
    }

    pub fn legal_actions_json(&mut self) -> Result<String, JsError> {
        self.game.legal_actions(&mut self.legal_actions);
        serde_json::to_string(&self.legal_actions).map_err(|error| JsError::new(&error.to_string()))
    }

    pub fn step(&mut self, action_index: usize) -> Result<String, JsError> {
        self.game.legal_actions(&mut self.legal_actions);
        let action = self
            .legal_actions
            .get(action_index)
            .copied()
            .ok_or_else(|| JsError::new("action index is outside the legal action list"))?;
        self.game
            .step(action)
            .map_err(|error| JsError::new(&error.to_string()))?;
        self.state_json()
    }

    pub fn step_bot(&mut self) -> Result<String, JsError> {
        if self.game.is_terminal() {
            return self.state_json();
        }
        self.game.legal_actions(&mut self.legal_actions);
        let action = if self.game.active_player() == PlayerId(0) {
            self.greedy.select_action(&self.game, &self.legal_actions)
        } else {
            self.random.select_action(&self.game, &self.legal_actions)
        };
        self.game
            .step(action)
            .map_err(|error| JsError::new(&error.to_string()))?;
        self.state_json()
    }
}

#[wasm_bindgen]
pub struct WasmReplay {
    replay: Replay,
    game: Game,
    frame: usize,
    legal_actions: Vec<Action>,
}

#[wasm_bindgen]
impl WasmReplay {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Self, JsError> {
        let replay = Replay::decode(bytes).map_err(|error| JsError::new(&error.to_string()))?;
        replay
            .verify()
            .map_err(|error| JsError::new(&error.to_string()))?;
        let game = Game::new(replay.header.rules.clone(), replay.header.scenario.clone())
            .map_err(|error| JsError::new(&error.to_string()))?;
        Ok(Self {
            replay,
            game,
            frame: 0,
            legal_actions: Vec::new(),
        })
    }

    pub fn frame_count(&self) -> usize {
        self.replay.frames.len()
    }

    pub fn metadata_json(&self) -> Result<String, JsError> {
        let metadata = ReplayMetadata {
            seed: self.replay.header.scenario.seed.to_string(),
            frames: self.replay.frames.len(),
            engine_version: self.replay.header.engine_version,
            format_version: self.replay.header.format_version,
            rules_profile: profile_name(self.replay.header.rules.profile),
        };
        serde_json::to_string(&metadata).map_err(|error| JsError::new(&error.to_string()))
    }

    pub fn seek(&mut self, frame: usize) -> Result<String, JsError> {
        if frame > self.replay.frames.len() {
            return Err(JsError::new("frame is outside the replay"));
        }
        if frame < self.frame {
            self.game = Game::new(
                self.replay.header.rules.clone(),
                self.replay.header.scenario.clone(),
            )
            .map_err(|error| JsError::new(&error.to_string()))?;
            self.frame = 0;
        }
        while self.frame < frame {
            self.game
                .step(self.replay.frames[self.frame].action)
                .map_err(|error| JsError::new(&error.to_string()))?;
            self.frame += 1;
        }
        state_json(&self.game, &mut self.legal_actions)
    }
}

fn state_json(game: &Game, legal_actions: &mut Vec<Action>) -> Result<String, JsError> {
    game.legal_actions(legal_actions);
    let view = StateView {
        width: game.topology().width(),
        height: game.topology().height(),
        round: game.round(),
        active_player: game.active_player().0,
        terminal: game.is_terminal(),
        winner: game.winner().map(|player| player.0),
        cells: game
            .cells()
            .iter()
            .copied()
            .zip(0_u16..)
            .map(|(cell, id)| CellView {
                id,
                owner: (!cell.owner().is_neutral()).then_some(cell.owner().0),
                object: cell.object(),
                strength: cell.unit().strength(),
                ready: cell.unit().is_ready(),
                province: cell.province().is_some().then_some(cell.province().0),
                defense: game.hex_defense(HexId(id)).unwrap_or_default(),
            })
            .collect(),
        provinces: game
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
            .collect(),
        legal_actions: legal_actions.len(),
    };
    serde_json::to_string(&view).map_err(|error| JsError::new(&error.to_string()))
}

fn profile_name(profile: antiyoy_core::RulesProfile) -> &'static str {
    match profile {
        antiyoy_core::RulesProfile::ClassicGeneric => "classic_generic_2022",
        antiyoy_core::RulesProfile::ClassicSlay => "classic_slay_2022",
        antiyoy_core::RulesProfile::OnlineDefaultV1 => "online_default_v1",
        antiyoy_core::RulesProfile::OnlineClassicV1 => "online_classic_v1",
        antiyoy_core::RulesProfile::OnlineDuelV1 => "online_duel_v1",
        antiyoy_core::RulesProfile::OnlineExperimentalV1 => "online_experimental_v1",
        antiyoy_core::RulesProfile::OnlineExperimentalV2_260801 => "online_experimental_v2_260801",
        antiyoy_core::RulesProfile::Custom => "custom",
    }
}

#[wasm_bindgen]
pub fn engine_version() -> u16 {
    antiyoy_core::ENGINE_VERSION
}
