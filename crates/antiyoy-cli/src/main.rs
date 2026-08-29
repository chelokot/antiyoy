use std::path::{Path, PathBuf};
use std::time::Instant;

use antiyoy_agents::{GreedyAgent, RandomAgent};
use antiyoy_core::Rules;
use antiyoy_eval::{Elo, MatchOutcome, Rating, run_match, score_for_first, symmetric_duel};
use antiyoy_protocol::{Digest, Replay};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(name = "antiyoy", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Version,
    Play {
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 11)]
        width: u16,
        #[arg(long, default_value_t = 9)]
        height: u16,
        #[arg(long, value_enum, default_value_t = AgentKind::Greedy)]
        first: AgentKind,
        #[arg(long, value_enum, default_value_t = AgentKind::Random)]
        second: AgentKind,
        #[arg(long, default_value_t = 2_000)]
        action_limit: u32,
        #[arg(long)]
        replay: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    Tournament {
        #[arg(long, default_value_t = 100)]
        games: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 11)]
        width: u16,
        #[arg(long, default_value_t = 9)]
        height: u16,
        #[arg(long, default_value_t = 2_000)]
        action_limit: u32,
        #[arg(long)]
        json: bool,
    },
    Verify {
        replay: PathBuf,
    },
    Bench {
        #[arg(long, default_value_t = 20)]
        games: u32,
        #[arg(long, default_value_t = 1_000)]
        action_limit: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum AgentKind {
    Greedy,
    Random,
}

#[derive(Debug, Serialize)]
struct PlaySummary {
    first: String,
    second: String,
    seed: u64,
    outcome: MatchOutcome,
    final_digest: String,
}

#[derive(Debug, Serialize)]
struct TournamentSummary {
    games: u32,
    greedy_wins: u32,
    random_wins: u32,
    draws: u32,
    greedy: Rating,
    random: Rating,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Version => println!("{}", antiyoy_core::ENGINE_VERSION),
        Command::Play {
            seed,
            width,
            height,
            first,
            second,
            action_limit,
            replay,
            json,
        } => play(
            seed,
            width,
            height,
            first,
            second,
            action_limit,
            replay,
            json,
        )?,
        Command::Tournament {
            games,
            seed,
            width,
            height,
            action_limit,
            json,
        } => tournament(games, seed, width, height, action_limit, json)?,
        Command::Verify { replay } => verify(&replay)?,
        Command::Bench {
            games,
            action_limit,
        } => bench(games, action_limit)?,
    }
    Ok(())
}

#[expect(clippy::too_many_arguments)]
fn play(
    seed: u64,
    width: u16,
    height: u16,
    first: AgentKind,
    second: AgentKind,
    action_limit: u32,
    replay_path: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let scenario = symmetric_duel(width, height, seed)?;
    let report = match (first, second) {
        (AgentKind::Greedy, AgentKind::Greedy) => run_match(
            Rules::classic_generic(),
            scenario,
            &mut GreedyAgent::new("greedy-0"),
            &mut GreedyAgent::new("greedy-1"),
            action_limit,
        )?,
        (AgentKind::Greedy, AgentKind::Random) => run_match(
            Rules::classic_generic(),
            scenario,
            &mut GreedyAgent::new("greedy"),
            &mut RandomAgent::new("random", seed ^ 1),
            action_limit,
        )?,
        (AgentKind::Random, AgentKind::Greedy) => run_match(
            Rules::classic_generic(),
            scenario,
            &mut RandomAgent::new("random", seed ^ 1),
            &mut GreedyAgent::new("greedy"),
            action_limit,
        )?,
        (AgentKind::Random, AgentKind::Random) => run_match(
            Rules::classic_generic(),
            scenario,
            &mut RandomAgent::new("random-0", seed ^ 1),
            &mut RandomAgent::new("random-1", seed ^ 2),
            action_limit,
        )?,
    };
    let verification = report.replay.verify()?;
    if let Some(path) = replay_path {
        std::fs::write(&path, report.replay.encode()?)
            .with_context(|| format!("failed to write replay to {}", path.display()))?;
    }
    let summary = PlaySummary {
        first: report.agents[0].clone(),
        second: report.agents[1].clone(),
        seed,
        outcome: report.outcome,
        final_digest: verification.final_digest.to_string(),
    };
    print_value(&summary, json)?;
    Ok(())
}

fn tournament(
    games: u32,
    seed: u64,
    width: u16,
    height: u16,
    action_limit: u32,
    json: bool,
) -> Result<()> {
    let mut greedy_rating = Rating::default();
    let mut random_rating = Rating::default();
    let mut greedy_wins = 0;
    let mut random_wins = 0;
    let mut draws = 0;
    for game_index in 0..games {
        let game_seed = seed + u64::from(game_index);
        let scenario = symmetric_duel(width, height, game_seed)?;
        let greedy_first = game_index % 2 == 0;
        let greedy_score = if greedy_first {
            let report = run_match(
                Rules::classic_generic(),
                scenario,
                &mut GreedyAgent::new("greedy"),
                &mut RandomAgent::new("random", game_seed ^ 1),
                action_limit,
            )?;
            score_for_first(report.outcome)
        } else {
            let report = run_match(
                Rules::classic_generic(),
                scenario,
                &mut RandomAgent::new("random", game_seed ^ 1),
                &mut GreedyAgent::new("greedy"),
                action_limit,
            )?;
            1.0 - score_for_first(report.outcome)
        };
        if greedy_score > 0.5 {
            greedy_wins += 1;
        } else if greedy_score < 0.5 {
            random_wins += 1;
        } else {
            draws += 1;
        }
        Elo::default().update(&mut greedy_rating, &mut random_rating, greedy_score);
    }
    let summary = TournamentSummary {
        games,
        greedy_wins,
        random_wins,
        draws,
        greedy: greedy_rating,
        random: random_rating,
    };
    print_value(&summary, json)?;
    Ok(())
}

fn verify(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read replay from {}", path.display()))?;
    let replay = Replay::decode(&bytes)?;
    let verification = replay.verify()?;
    println!(
        "verified {} frames, final digest {}",
        verification.frames, verification.final_digest
    );
    Ok(())
}

#[expect(clippy::cast_precision_loss)]
fn bench(games: u32, action_limit: u32) -> Result<()> {
    let started = Instant::now();
    let mut actions = 0_u64;
    let mut digest = Digest([0; 32]);
    for game_index in 0..games {
        let seed = u64::from(game_index) + 1;
        let scenario = symmetric_duel(11, 9, seed)?;
        let report = run_match(
            Rules::classic_generic(),
            scenario,
            &mut RandomAgent::new("random-0", seed),
            &mut RandomAgent::new("random-1", seed ^ 1),
            action_limit,
        )?;
        actions += u64::from(report.outcome.actions);
        digest = report
            .replay
            .frames
            .last()
            .map_or(report.replay.header.initial_digest, |frame| {
                frame.state_digest
            });
    }
    let elapsed = started.elapsed();
    let actions_per_second = actions as f64 / elapsed.as_secs_f64();
    println!(
        "{actions} actions in {:.3}s ({actions_per_second:.0} actions/s), digest {digest}",
        elapsed.as_secs_f64()
    );
    Ok(())
}

fn print_value(value: &impl Serialize, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{value}", value = serde_json::to_string(value)?);
    }
    Ok(())
}
