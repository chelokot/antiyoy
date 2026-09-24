use antiyoy_agents::{Agent, GreedyAgent, SearchAgent, SearchConfig};
use antiyoy_core::{Action, GENERATOR_ROTATED_SCHEMA_VERSION, Game, GeneratorConfig, Rules};
use antiyoy_rl::BatchObservation;
use serde::Serialize;

const FIRST_SEED: u64 = 6_520_000;
const MAPS: u64 = 64;
const RNG_SHIFT: u64 = 11_400_714_819_323_198_485;

#[derive(Serialize)]
struct ChangedAction {
    seed: u64,
    seat: u8,
    original: Action,
    shifted: Action,
}

#[derive(Serialize)]
struct Report {
    first_seed: u64,
    maps: u64,
    sampled_by_seat: [u32; 2],
    skipped_by_seat: [u32; 2],
    changed_by_seat: [u32; 2],
    maps_with_change: u32,
    changed_actions: Vec<ChangedAction>,
}

fn shifted_random(game: &Game) -> Game {
    let mut value = serde_json::to_value(game).expect("game serializes");
    let random = value["random"]["state"]
        .as_u64()
        .expect("game RNG state is a u64");
    value["random"]["state"] = serde_json::json!(random.wrapping_add(RNG_SHIFT));
    serde_json::from_value(value).expect("shifted game deserializes")
}

fn teacher_action(game: &Game, legal: &[Action]) -> Action {
    let mut agent = SearchAgent::with_three_turn_search(
        "observation-alias-audit",
        SearchConfig {
            node_budget: 256,
            ..SearchConfig::default()
        },
        8,
        64,
        32,
    )
    .expect("valid search configuration");
    agent.select_action(game, legal)
}

fn sample(game: &Game) -> Option<(Action, Action)> {
    let mut legal = Vec::new();
    game.legal_actions(&mut legal);
    let shifted = shifted_random(game);
    let mut shifted_legal = Vec::new();
    shifted.legal_actions(&mut shifted_legal);
    assert_eq!(legal, shifted_legal);
    let mut original_observation = BatchObservation::default();
    original_observation.observe_game(game, &legal, false);
    let mut shifted_observation = BatchObservation::default();
    shifted_observation.observe_game(&shifted, &shifted_legal, false);
    assert_eq!(original_observation, shifted_observation);
    let original = teacher_action(game, &legal);
    let altered = teacher_action(&shifted, &legal);
    (original != altered).then_some((original, altered))
}

fn main() {
    let mut report = Report {
        first_seed: FIRST_SEED,
        maps: MAPS,
        sampled_by_seat: [0; 2],
        skipped_by_seat: [0; 2],
        changed_by_seat: [0; 2],
        maps_with_change: 0,
        changed_actions: Vec::new(),
    };
    for seed in FIRST_SEED..FIRST_SEED + MAPS {
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
        let mut game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let mut greedy = GreedyAgent::new("rollin");
        let mut sampled = [false; 2];
        let mut changed = false;
        for _ in 0..2_400 {
            if game.is_terminal() || sampled.iter().all(|done| *done) {
                break;
            }
            let seat = game.active_player().index();
            if game.round() >= 4 && !sampled[seat] {
                sampled[seat] = true;
                report.sampled_by_seat[seat] += 1;
                if let Some((original, shifted)) = sample(&game) {
                    report.changed_by_seat[seat] += 1;
                    changed = true;
                    report.changed_actions.push(ChangedAction {
                        seed,
                        seat: game.active_player().0,
                        original,
                        shifted,
                    });
                }
            }
            let mut legal = Vec::new();
            game.legal_actions(&mut legal);
            let action = greedy.select_action(&game, &legal);
            game.step(action).expect("greedy action is legal");
        }
        if sampled.iter().any(|done| !done) {
            for (seat, done) in sampled.into_iter().enumerate() {
                report.skipped_by_seat[seat] += u32::from(!done);
            }
        }
        report.maps_with_change += u32::from(changed);
    }
    println!(
        "{}",
        serde_json::to_string(&report).expect("report serializes")
    );
}
