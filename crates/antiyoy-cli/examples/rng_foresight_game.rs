use antiyoy_agents::{Agent, SearchAgent, SearchConfig};
use antiyoy_core::{Action, GENERATOR_ROTATED_SCHEMA_VERSION, Game, GeneratorConfig, Rules};
use antiyoy_eval::{Termination, run_match};
use antiyoy_rl::BatchObservation;
use serde::Serialize;

#[path = "support/rng_state.rs"]
mod rng_state;

const FIRST_SEED: u64 = 6_521_000;
const MAPS: u64 = 64;
const ACTION_LIMIT: u32 = 2_400;

struct ReplannedAgent {
    name: &'static str,
    search: SearchAgent,
    shifted_rng: bool,
}

impl ReplannedAgent {
    fn new(name: &'static str, shifted_rng: bool) -> Self {
        let search = SearchAgent::with_three_turn_search(
            name,
            SearchConfig {
                node_budget: 256,
                ..SearchConfig::default()
            },
            8,
            64,
            32,
        )
        .expect("valid search configuration");
        Self {
            name,
            search,
            shifted_rng,
        }
    }
}

impl Agent for ReplannedAgent {
    fn name(&self) -> &str {
        self.name
    }

    fn select_action(&mut self, game: &Game, legal_actions: &[Action]) -> Action {
        self.search.clear_plan();
        if !self.shifted_rng {
            return self.search.select_action(game, legal_actions);
        }
        let shifted = rng_state::shifted_random(game);
        let mut shifted_legal = Vec::new();
        shifted.legal_actions(&mut shifted_legal);
        assert_eq!(legal_actions, shifted_legal);
        let mut original_observation = BatchObservation::default();
        original_observation.observe_game(game, legal_actions, false);
        let mut shifted_observation = BatchObservation::default();
        shifted_observation.observe_game(&shifted, &shifted_legal, false);
        assert_eq!(original_observation, shifted_observation);
        self.search.select_action(&shifted, legal_actions)
    }
}

#[derive(Serialize)]
struct GameResult {
    seed: u64,
    exact_seat: u8,
    winner_seat: Option<u8>,
    actions: u32,
    termination: Termination,
}

#[derive(Serialize)]
struct Report {
    first_seed: u64,
    maps: u64,
    action_limit: u32,
    games: Vec<GameResult>,
}

fn main() {
    let mut games = Vec::with_capacity(usize::try_from(MAPS * 2).expect("small map count"));
    for seed in FIRST_SEED..FIRST_SEED + MAPS {
        for exact_seat in 0..2 {
            let scenario = GeneratorConfig {
                schema_version: GENERATOR_ROTATED_SCHEMA_VERSION,
                width: 11,
                height: 9,
                players: 2,
                seed,
                ..GeneratorConfig::default()
            }
            .generate()
            .expect("valid procedural map");
            let mut exact = ReplannedAgent::new("exact-rng-search", false);
            let mut shifted = ReplannedAgent::new("shifted-rng-search", true);
            let report = if exact_seat == 0 {
                run_match(
                    Rules::classic_generic(),
                    scenario,
                    &mut exact,
                    &mut shifted,
                    ACTION_LIMIT,
                )
            } else {
                run_match(
                    Rules::classic_generic(),
                    scenario,
                    &mut shifted,
                    &mut exact,
                    ACTION_LIMIT,
                )
            }
            .expect("legal search match");
            games.push(GameResult {
                seed,
                exact_seat,
                winner_seat: report.outcome.winner.map(|winner| winner.0),
                actions: report.outcome.actions,
                termination: report.outcome.termination,
            });
        }
    }
    println!(
        "{}",
        serde_json::to_string(&Report {
            first_seed: FIRST_SEED,
            maps: MAPS,
            action_limit: ACTION_LIMIT,
            games,
        })
        .expect("report serializes")
    );
}
