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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    assert!(args.maps > 0, "map count must be positive");
    let config = SearchConfig {
        node_budget: args.root_nodes,
        ..SearchConfig::default()
    };
    let mut baseline_ms = Vec::new();
    let mut candidate_ms = Vec::new();
    let mut samples_by_seat = [0_u32; 2];
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
        for seat in 0..2_u8 {
            if game.is_terminal() || game.active_player() != PlayerId(seat) {
                break;
            }
            let (baseline, candidate) =
                measure(&game, config, &args, (map + u32::from(seat)) % 2 == 0);
            baseline_ms.push(baseline);
            candidate_ms.push(candidate);
            samples_by_seat[usize::from(seat)] += 1;
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
    let baseline_median = median(&mut baseline_ms);
    let candidate_median = median(&mut candidate_ms);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "seed_first": args.seed,
            "maps": args.maps,
            "samples_by_seat": samples_by_seat,
            "root_nodes": args.root_nodes,
            "reply_nodes": args.reply_nodes,
            "followup_nodes": args.followup_nodes,
            "slate_size": args.slate_size,
            "baseline_median_root_decision_ms": baseline_median,
            "candidate_median_root_decision_ms": candidate_median,
            "median_ratio": candidate_median / baseline_median
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_handles_odd_and_even_samples() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
