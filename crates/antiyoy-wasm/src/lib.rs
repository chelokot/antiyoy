#![forbid(unsafe_code)]

use antiyoy_agents::{Agent, GreedyAgent, SearchAgent, SearchConfig};
use antiyoy_core::{Action, Game, GeneratorConfig, PlayerId, Rules, Scenario};
use antiyoy_protocol::{GameView, Replay};
use serde::Serialize;
use wasm_bindgen::prelude::*;

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
    rules: Rules,
    scenario: Scenario,
    game: Game,
    legal_actions: Vec<Action>,
    greedy: GreedyAgent,
    search: SearchAgent,
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u16, height: u16, seed: u64) -> Result<Self, JsError> {
        Self::with_profile(width, height, seed, "classic_generic_2022")
    }

    pub fn with_profile(
        width: u16,
        height: u16,
        seed: u64,
        profile: &str,
    ) -> Result<Self, JsError> {
        let scenario = Scenario::symmetric_duel(width, height, seed)
            .map_err(|error| JsError::new(&error.to_string()))?;
        Self::from_scenario(rules_for_profile(profile)?, scenario)
    }

    pub fn procedural(
        width: u16,
        height: u16,
        players: u8,
        seed: u64,
        land_density_per_million: u32,
    ) -> Result<Self, JsError> {
        Self::procedural_with_profile(
            width,
            height,
            players,
            seed,
            land_density_per_million,
            "classic_generic_2022",
        )
    }

    pub fn procedural_with_profile(
        width: u16,
        height: u16,
        players: u8,
        seed: u64,
        land_density_per_million: u32,
        profile: &str,
    ) -> Result<Self, JsError> {
        let config = GeneratorConfig {
            width,
            height,
            players,
            seed,
            land_density_per_million,
            ..GeneratorConfig::default()
        };
        let scenario = config
            .generate()
            .map_err(|error| JsError::new(&error.to_string()))?;
        Self::from_scenario(rules_for_profile(profile)?, scenario)
    }

    pub fn rules_profile(&self) -> String {
        profile_name(self.rules.profile).to_owned()
    }

    pub fn reset(&mut self) -> Result<String, JsError> {
        self.game = Game::new(self.rules.clone(), self.scenario.clone())
            .map_err(|error| JsError::new(&error.to_string()))?;
        self.search = SearchAgent::new("turn-search");
        self.state_json()
    }

    pub fn state_json(&mut self) -> Result<String, JsError> {
        state_json(&self.game)
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
        self.step_with_policy(self.game.active_player() != PlayerId(0))
    }

    pub fn step_search(&mut self) -> Result<String, JsError> {
        self.step_with_policy(true)
    }

    pub fn step_search_with_budget(&mut self, node_budget: usize) -> Result<String, JsError> {
        if self.search.config().node_budget != node_budget {
            let config = SearchConfig {
                node_budget,
                ..self.search.config()
            };
            self.search = SearchAgent::with_config("turn-search", config)
                .map_err(|error| JsError::new(&error.to_string()))?;
        }
        self.step_with_policy(true)
    }

    pub fn search_node_budget(&self) -> usize {
        self.search.config().node_budget
    }

    pub fn search_nodes(&self) -> usize {
        self.search.last_stats().nodes
    }

    pub fn search_count(&self) -> u64 {
        self.search.search_count()
    }
}

impl WasmGame {
    fn step_with_policy(&mut self, use_search: bool) -> Result<String, JsError> {
        if self.game.is_terminal() {
            return self.state_json();
        }
        self.game.legal_actions(&mut self.legal_actions);
        let action = if use_search {
            self.search.select_action(&self.game, &self.legal_actions)
        } else {
            self.greedy.select_action(&self.game, &self.legal_actions)
        };
        self.game
            .step(action)
            .map_err(|error| JsError::new(&error.to_string()))?;
        self.state_json()
    }
    fn from_scenario(rules: Rules, scenario: Scenario) -> Result<Self, JsError> {
        let game = Game::new(rules.clone(), scenario.clone())
            .map_err(|error| JsError::new(&error.to_string()))?;
        Ok(Self {
            rules,
            scenario,
            game,
            legal_actions: Vec::new(),
            greedy: GreedyAgent::new("greedy"),
            search: SearchAgent::new("turn-search"),
        })
    }
}

#[wasm_bindgen]
pub struct WasmReplay {
    replay: Replay,
    game: Game,
    frame: usize,
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
        state_json(&self.game)
    }
}

fn state_json(game: &Game) -> Result<String, JsError> {
    serde_json::to_string(&GameView::from_game(game))
        .map_err(|error| JsError::new(&error.to_string()))
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

fn rules_for_profile(profile: &str) -> Result<Rules, JsError> {
    match profile {
        "classic_generic_2022" => Ok(Rules::classic_generic()),
        "classic_slay_2022" => Ok(Rules::classic_slay()),
        "online_default_v1" => Ok(Rules::online_default_v1()),
        "online_classic_v1" => Ok(Rules::online_classic_v1()),
        "online_duel_v1" => Ok(Rules::online_duel_v1()),
        "online_experimental_v1" => Ok(Rules::online_experimental_v1()),
        "online_experimental_v2_260801" => Ok(Rules::online_experimental_v2_260801()),
        _ => Err(JsError::new("unknown rules profile")),
    }
}

#[wasm_bindgen]
pub fn engine_version() -> u16 {
    antiyoy_core::ENGINE_VERSION
}
