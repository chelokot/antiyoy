use std::path::{Path, PathBuf};
use std::time::Instant;

use antiyoy_agents::{GreedyAgent, RandomAgent};
use antiyoy_core::Rules;
use antiyoy_eval::{
    Elo, League as RatingLeague, MatchOutcome, Rating, Standing, run_match, score_for_first,
    symmetric_duel,
};
use antiyoy_protocol::{Digest, Replay};
use antiyoy_rl::{BatchEnv, BatchObservation};
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
    League {
        #[arg(long, default_value = "league.json")]
        state: PathBuf,
        #[arg(long, default_value = "replays")]
        replay_dir: PathBuf,
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
        #[arg(long, value_enum, default_value_t = RulesKind::ClassicGeneric)]
        rules: RulesKind,
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
    RlBench {
        #[arg(long, default_value_t = 64)]
        environments: usize,
        #[arg(long, default_value_t = 100_000)]
        transitions: u64,
        #[arg(long, default_value_t = 1_000)]
        action_limit: u32,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum AgentKind {
    Greedy,
    Random,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum RulesKind {
    ClassicGeneric,
    ClassicSlay,
    OnlineDefaultV1,
    OnlineClassicV1,
    OnlineDuelV1,
    OnlineExperimentalV1,
    OnlineExperimentalV2,
}

impl RulesKind {
    fn rules(self) -> Rules {
        match self {
            Self::ClassicGeneric => Rules::classic_generic(),
            Self::ClassicSlay => Rules::classic_slay(),
            Self::OnlineDefaultV1 => Rules::online_default_v1(),
            Self::OnlineClassicV1 => Rules::online_classic_v1(),
            Self::OnlineDuelV1 => Rules::online_duel_v1(),
            Self::OnlineExperimentalV1 => Rules::online_experimental_v1(),
            Self::OnlineExperimentalV2 => Rules::online_experimental_v2_260801(),
        }
    }
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

#[derive(Debug, Serialize)]
struct LeagueSummary {
    games_added: u32,
    total_games: usize,
    standings: Vec<Standing>,
}

#[derive(Debug, Serialize)]
struct RlBenchmarkSummary {
    environments: usize,
    transitions: u64,
    observation_batches: u64,
    elapsed_seconds: f64,
    transitions_per_second: f64,
    checksum: u64,
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
        Command::League {
            state,
            replay_dir,
            games,
            seed,
            width,
            height,
            action_limit,
            rules,
            json,
        } => league(
            &state,
            &replay_dir,
            games,
            seed,
            width,
            height,
            action_limit,
            rules,
            json,
        )?,
        Command::Verify { replay } => verify(&replay)?,
        Command::Bench {
            games,
            action_limit,
        } => bench(games, action_limit)?,
        Command::RlBench {
            environments,
            transitions,
            action_limit,
            json,
        } => rl_bench(environments, transitions, action_limit, json)?,
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

#[expect(clippy::too_many_arguments)]
fn league(
    state_path: &Path,
    replay_dir: &Path,
    games: u32,
    seed: u64,
    width: u16,
    height: u16,
    action_limit: u32,
    rules_kind: RulesKind,
    json: bool,
) -> Result<()> {
    let mut league = if state_path.try_exists()? {
        let bytes = std::fs::read(state_path)
            .with_context(|| format!("failed to read league from {}", state_path.display()))?;
        serde_json::from_slice::<RatingLeague>(&bytes)
            .with_context(|| format!("failed to decode league from {}", state_path.display()))?
    } else {
        RatingLeague::default()
    };
    league.validate()?;
    std::fs::create_dir_all(replay_dir)
        .with_context(|| format!("failed to create replay directory {}", replay_dir.display()))?;

    for game_index in 0..games {
        let game_seed = seed + u64::from(game_index);
        let scenario = symmetric_duel(width, height, game_seed)?;
        let report = if game_index % 2 == 0 {
            run_match(
                rules_kind.rules(),
                scenario,
                &mut GreedyAgent::new("greedy"),
                &mut RandomAgent::new("random", game_seed ^ 1),
                action_limit,
            )?
        } else {
            run_match(
                rules_kind.rules(),
                scenario,
                &mut RandomAgent::new("random", game_seed ^ 1),
                &mut GreedyAgent::new("greedy"),
                action_limit,
            )?
        };
        let match_id = league.record(&report)?.id;
        let replay_path = replay_dir.join(format!("{match_id}.antiyoy"));
        write_atomic(&replay_path, &report.replay.encode()?)?;
    }

    let encoded = serde_json::to_vec_pretty(&league)?;
    write_atomic(state_path, &encoded)?;
    let summary = LeagueSummary {
        games_added: games,
        total_games: league.matches.len(),
        standings: league.standings(),
    };
    print_value(&summary, json)?;
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, bytes)
        .with_context(|| format!("failed to write temporary file {}", temporary.display()))?;
    std::fs::rename(&temporary, path)
        .with_context(|| format!("failed to replace {}", path.display()))?;
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

#[expect(clippy::cast_precision_loss)]
fn rl_bench(environments: usize, transitions: u64, action_limit: u32, json: bool) -> Result<()> {
    anyhow::ensure!(transitions > 0, "transitions must be greater than zero");
    let mut batch = BatchEnv::symmetric_duels(
        Rules::classic_generic(),
        environments,
        11,
        9,
        1,
        action_limit,
    )?;
    let mut observation = BatchObservation::default();
    let mut random_state = 0x4d59_5df4_d0f3_3173_u64;
    let mut completed = 0_u64;
    let mut observation_batches = 0_u64;
    let mut reset_seed = 1_000_000_u64;
    let mut checksum = 0_u64;
    let mut action_indices = vec![0_usize; batch.len()];
    let started = Instant::now();
    while completed < transitions {
        batch.observe(&mut observation);
        observation_batches += 1;
        checksum = checksum
            .wrapping_mul(31)
            .wrapping_add(observation.actions.len() as u64)
            .wrapping_add(observation.owners.len() as u64);
        for (environment, action_index) in action_indices.iter_mut().enumerate() {
            let legal_actions = batch
                .legal_actions(environment)
                .expect("batch index is generated from its length")
                .len();
            *action_index = usize::try_from(next_random(&mut random_state)).unwrap_or(usize::MAX)
                % legal_actions;
        }
        let remaining = transitions - completed;
        if remaining < batch.len() as u64 {
            for (environment, action_index) in action_indices
                .iter()
                .copied()
                .enumerate()
                .take(usize::try_from(remaining).unwrap_or(usize::MAX))
            {
                let result = batch.step(environment, action_index)?;
                checksum = checksum
                    .wrapping_add(u64::from(result.round))
                    .wrapping_add(result.reward.treasury_delta.unsigned_abs());
                completed += 1;
                if result.done() {
                    batch.reset_with_seed(environment, reset_seed)?;
                    reset_seed = reset_seed.wrapping_add(1);
                }
            }
            continue;
        }
        let results = batch.step_all(&action_indices)?;
        completed += batch.len() as u64;
        for (environment, result) in results.into_iter().enumerate() {
            checksum = checksum
                .wrapping_add(u64::from(result.round))
                .wrapping_add(result.reward.treasury_delta.unsigned_abs());
            if result.done() {
                batch.reset_with_seed(environment, reset_seed)?;
                reset_seed = reset_seed.wrapping_add(1);
            }
        }
    }
    let elapsed = started.elapsed();
    let summary = RlBenchmarkSummary {
        environments,
        transitions: completed,
        observation_batches,
        elapsed_seconds: elapsed.as_secs_f64(),
        transitions_per_second: completed as f64 / elapsed.as_secs_f64(),
        checksum,
    };
    print_value(&summary, json)?;
    Ok(())
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn print_value(value: &impl Serialize, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{value}", value = serde_json::to_string(value)?);
    }
    Ok(())
}
