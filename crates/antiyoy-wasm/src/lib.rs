#![forbid(unsafe_code)]

use antiyoy_agents::{Agent, GreedyAgent, RandomAgent};
use antiyoy_core::{Action, Game, HexId, Object, PlayerId, Rules, Scenario};
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
        self.game.legal_actions(&mut self.legal_actions);
        let view = StateView {
            width: self.game.topology().width(),
            height: self.game.topology().height(),
            round: self.game.round(),
            active_player: self.game.active_player().0,
            terminal: self.game.is_terminal(),
            winner: self.game.winner().map(|player| player.0),
            cells: self
                .game
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
                    defense: self.game.hex_defense(HexId(id)).unwrap_or_default(),
                })
                .collect(),
            provinces: self
                .game
                .provinces()
                .iter()
                .map(|province| ProvinceView {
                    id: province.id().0,
                    owner: province.owner().0,
                    money: province.money(),
                    income: self.game.province_income(province.id()).unwrap_or_default(),
                    upkeep: self.game.province_upkeep(province.id()).unwrap_or_default(),
                    profit: self.game.province_profit(province.id()).unwrap_or_default(),
                    capital: province.capital().0,
                    size: province.hexes().len(),
                })
                .collect(),
            legal_actions: self.legal_actions.len(),
        };
        serde_json::to_string(&view).map_err(|error| JsError::new(&error.to_string()))
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
pub fn engine_version() -> u16 {
    antiyoy_core::ENGINE_VERSION
}
