use std::time::Instant;

use antiyoy_agents::{Agent, SearchAgent, SearchConfig};
use antiyoy_core::{Game, GeneratorConfig, PlayerId, Rules};
use clap::Parser;
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    maps: u32,
    #[arg(long, default_value_t = 256)]
    root_nodes: usize,
    #[arg(long, default_value_t = 64)]
    reply_nodes: usize,
    #[arg(long, default_value_t = 32)]
    followup_nodes: usize,
    #[arg(long, default_value_t = 8)]
    slate_size: usize,
    #[arg(long, default_value_t = 1_000)]
    action_limit: u32,
    #[arg(long)]
    all_turns: bool,
}

#[derive(Default)]
struct Measurements {
    baseline_ms: Vec<f64>,
    candidate_ms: Vec<f64>,
    samples_by_seat: [u32; 2],
    censored_rollins: u32,
}

impl Measurements {
    fn sample(&mut self, game: &Game, config: SearchConfig, args: &Args, candidate_first: bool) {
        let (baseline, candidate) = measure(game, config, args, candidate_first);
        self.baseline_ms.push(baseline);
        self.candidate_ms.push(candidate);
        self.samples_by_seat[game.active_player().index()] += 1;
    }
}

fn measure(game: &Game, config: SearchConfig, args: &Args, candidate_first: bool) -> (f64, f64) {
    let mut legal = Vec::new();
    game.legal_actions(&mut legal);
    let mut baseline =
        SearchAgent::with_reply_search("reply-search", config, args.slate_size, args.reply_nodes)
            .expect("valid baseline search");
    let mut candidate = SearchAgent::with_three_turn_search(
        "three-turn-search",
        config,
        args.slate_size,
        args.reply_nodes,
        args.followup_nodes,
    )
    .expect("valid three-turn search");
    if candidate_first {
        let started = Instant::now();
        candidate.select_action(game, &legal);
        let candidate_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        baseline.select_action(game, &legal);
        (started.elapsed().as_secs_f64() * 1_000.0, candidate_ms)
    } else {
        let started = Instant::now();
        baseline.select_action(game, &legal);
        let baseline_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        candidate.select_action(game, &legal);
        (baseline_ms, started.elapsed().as_secs_f64() * 1_000.0)
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

fn percentile_95(values: &[f64]) -> f64 {
    values[values.len().saturating_mul(95).div_ceil(100) - 1]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    assert!(
        args.maps > 0 && args.action_limit > 0,
        "map and action limits must be positive"
    );
    let config = SearchConfig {
        node_budget: args.root_nodes,
        ..SearchConfig::default()
    };
    let mut measurements = Measurements::default();
    for map in 0..args.maps {
        let scenario = GeneratorConfig {
            schema_version: 2,
            width: 11,
            height: 9,
            players: 2,
            seed: args.seed + u64::from(map),
            ..GeneratorConfig::default()
        }
        .generate()?;
        let mut game = Game::new(Rules::classic_generic(), scenario)?;
        if args.all_turns {
            let mut agents = [
                SearchAgent::with_reply_search(
                    "seat-zero",
                    config,
                    args.slate_size,
                    args.reply_nodes,
                )?,
                SearchAgent::with_reply_search(
                    "seat-one",
                    config,
                    args.slate_size,
                    args.reply_nodes,
                )?,
            ];
            let mut prior_seat = None;
            for _ in 0..args.action_limit {
                if game.is_terminal() {
                    break;
                }
                let seat = game.active_player().index();
                if prior_seat != Some(seat) {
                    let candidate_first = measurements.baseline_ms.len().is_multiple_of(2);
                    measurements.sample(&game, config, &args, candidate_first);
                }
                let mut legal = Vec::new();
                game.legal_actions(&mut legal);
                let action = agents[seat].select_action(&game, &legal);
                game.step(action)?;
                prior_seat = Some(seat);
            }
            measurements.censored_rollins += u32::from(!game.is_terminal());
        } else {
            for seat in 0..2_u8 {
                if game.is_terminal() || game.active_player() != PlayerId(seat) {
                    break;
                }
                measurements.sample(&game, config, &args, (map + u32::from(seat)) % 2 == 0);
                if seat == 0 {
                    let mut agent = SearchAgent::with_reply_search(
                        "rollin",
                        config,
                        args.slate_size,
                        args.reply_nodes,
                    )?;
                    while !game.is_terminal() && game.active_player() == PlayerId(0) {
                        let mut legal = Vec::new();
                        game.legal_actions(&mut legal);
                        let action = agent.select_action(&game, &legal);
                        game.step(action)?;
                    }
                }
            }
        }
    }
    let baseline_median = median(&mut measurements.baseline_ms);
    let candidate_median = median(&mut measurements.candidate_ms);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "seed_first": args.seed,
            "maps": args.maps,
            "samples_by_seat": measurements.samples_by_seat,
            "all_turns": args.all_turns,
            "action_limit": args.action_limit,
            "censored_rollins": measurements.censored_rollins,
            "root_nodes": args.root_nodes,
            "reply_nodes": args.reply_nodes,
            "followup_nodes": args.followup_nodes,
            "slate_size": args.slate_size,
            "baseline_median_root_decision_ms": baseline_median,
            "candidate_median_root_decision_ms": candidate_median,
            "median_ratio": candidate_median / baseline_median,
            "baseline_p95_root_decision_ms": percentile_95(&measurements.baseline_ms),
            "candidate_p95_root_decision_ms": percentile_95(&measurements.candidate_ms)
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{median, percentile_95};

    #[test]
    fn median_handles_odd_and_even_samples() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(percentile_95(&[1.0, 2.0, 3.0, 4.0]), 4.0);
    }
}
