use std::path::{Path, PathBuf};
use std::time::Instant;

use antiyoy_agents::{Agent, GreedyAgent, RandomAgent, SearchAgent, SearchConfig};
use antiyoy_core::{GeneratorConfig, Rules, RulesProfile};
use antiyoy_eval::{
    Elo, League as RatingLeague, MatchOutcome, Rating, Standing, run_match, score_for_first,
    symmetric_duel,
};
use antiyoy_protocol::{Digest, Replay};
use antiyoy_rl::{BatchEnv, BatchObservation};
use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
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
        #[arg(long, default_value_t = 2_048)]
        search_nodes: usize,
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
    Compare {
        #[arg(long, default_value_t = 8)]
        pairs: u32,
        #[arg(long, default_value_t = 10_000)]
        seed: u64,
        #[arg(long, default_value_t = 7)]
        width: u16,
        #[arg(long, default_value_t = 5)]
        height: u16,
        #[arg(long, default_value_t = 500)]
        action_limit: u32,
        #[arg(long, value_enum, default_value_t = AgentKind::Search)]
        first: AgentKind,
        #[arg(long, value_enum, default_value_t = AgentKind::Greedy)]
        second: AgentKind,
        #[arg(long, default_value_t = 2_048)]
        search_nodes: usize,
        #[arg(long, value_enum, default_value_t = RulesKind::ClassicGeneric)]
        rules: RulesKind,
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
        #[command(flatten)]
        map: RlMapArgs,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum AgentKind {
    Greedy,
    Random,
    Search,
}

impl AgentKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Random => "random",
            Self::Search => "turn-search",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum RlMapKind {
    Symmetric,
    Procedural,
}

#[derive(Clone, Debug, Args)]
struct RlMapArgs {
    #[arg(long, value_enum, default_value_t = RlMapKind::Symmetric)]
    map: RlMapKind,
    #[arg(long, default_value_t = 11)]
    width: u16,
    #[arg(long, default_value_t = 9)]
    height: u16,
    #[arg(long, default_value_t = 2)]
    players: u8,
    #[arg(long, default_value_t = 650_000)]
    land_density_per_million: u32,
    #[arg(long, default_value_t = 5)]
    starting_province_size: u16,
    #[arg(long, default_value_t = 10)]
    starting_money: i64,
    #[arg(long, default_value_t = 150_000)]
    tree_density_per_million: u32,
    #[arg(long, default_value_t = 20_000)]
    neutral_tower_density_per_million: u32,
    #[arg(long, default_value_t = 10_000)]
    neutral_capital_density_per_million: u32,
    #[arg(long, default_value_t = 15_000)]
    grave_density_per_million: u32,
}

impl RlMapArgs {
    fn create_batch(&self, environments: usize, action_limit: u32) -> Result<BatchEnv> {
        match self.map {
            RlMapKind::Symmetric => {
                anyhow::ensure!(
                    self.players == 2,
                    "symmetric maps require exactly two players"
                );
                Ok(BatchEnv::symmetric_duels(
                    Rules::classic_generic(),
                    environments,
                    self.width,
                    self.height,
                    1,
                    action_limit,
                )?)
            }
            RlMapKind::Procedural => Ok(BatchEnv::procedural(
                Rules::classic_generic(),
                environments,
                &GeneratorConfig {
                    schema_version: antiyoy_core::GENERATOR_SCHEMA_VERSION,
                    width: self.width,
                    height: self.height,
                    players: self.players,
                    seed: 1,
                    land_density_per_million: self.land_density_per_million,
                    starting_province_size: self.starting_province_size,
                    starting_money: self.starting_money,
                    tree_density_per_million: self.tree_density_per_million,
                    neutral_tower_density_per_million: self.neutral_tower_density_per_million,
                    neutral_capital_density_per_million: self.neutral_capital_density_per_million,
                    grave_density_per_million: self.grave_density_per_million,
                },
                action_limit,
            )?),
        }
    }

    const fn name(&self) -> &'static str {
        match self.map {
            RlMapKind::Symmetric => "symmetric_duel_v1",
            RlMapKind::Procedural => "procedural_v1",
        }
    }
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
        let profile = match self {
            Self::ClassicGeneric => RulesProfile::ClassicGeneric,
            Self::ClassicSlay => RulesProfile::ClassicSlay,
            Self::OnlineDefaultV1 => RulesProfile::OnlineDefaultV1,
            Self::OnlineClassicV1 => RulesProfile::OnlineClassicV1,
            Self::OnlineDuelV1 => RulesProfile::OnlineDuelV1,
            Self::OnlineExperimentalV1 => RulesProfile::OnlineExperimentalV1,
            Self::OnlineExperimentalV2 => RulesProfile::OnlineExperimentalV2_260801,
        };
        Rules::from_profile(profile).expect("CLI exposes only bundled rules profiles")
    }

    const fn name(self) -> &'static str {
        match self {
            Self::ClassicGeneric => "classic_generic_2022",
            Self::ClassicSlay => "classic_slay_2022",
            Self::OnlineDefaultV1 => "online_default_v1",
            Self::OnlineClassicV1 => "online_classic_v1",
            Self::OnlineDuelV1 => "online_duel_v1",
            Self::OnlineExperimentalV1 => "online_experimental_v1",
            Self::OnlineExperimentalV2 => "online_experimental_v2_260801",
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
struct ComparisonSummary {
    first: String,
    second: String,
    rules: String,
    seed: u64,
    width: u16,
    height: u16,
    action_limit: u32,
    search_nodes: usize,
    pairs: u32,
    games: u32,
    first_wins: u32,
    second_wins: u32,
    draws: u32,
    first_score: f64,
    relative_elo: f64,
    first_rating: Rating,
    second_rating: Rating,
    actions: u64,
    elapsed_seconds: f64,
    actions_per_second: f64,
}

#[derive(Debug, Serialize)]
struct LeagueSummary {
    games_added: u32,
    total_games: usize,
    standings: Vec<Standing>,
}

#[derive(Debug, Serialize)]
struct RlBenchmarkSummary {
    map: &'static str,
    width: u16,
    height: u16,
    players: u8,
    playable_hexes_per_environment: usize,
    environments: usize,
    transitions: u64,
    observation_batches: u64,
    setup_seconds: f64,
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
            search_nodes,
            replay,
            json,
        } => play(
            seed,
            width,
            height,
            first,
            second,
            action_limit,
            search_nodes,
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
        Command::Compare {
            pairs,
            seed,
            width,
            height,
            action_limit,
            first,
            second,
            search_nodes,
            rules,
            json,
        } => compare_agents(
            pairs,
            seed,
            width,
            height,
            action_limit,
            first,
            second,
            search_nodes,
            rules,
            json,
        )?,
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
            map,
            json,
        } => rl_bench(environments, transitions, action_limit, &map, json)?,
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
    search_nodes: usize,
    replay_path: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let scenario = symmetric_duel(width, height, seed)?;
    let first_name = format!("{}-0", first.name());
    let second_name = format!("{}-1", second.name());
    let mut first_agent = create_agent(first, &first_name, seed ^ 1, search_nodes)?;
    let mut second_agent = create_agent(second, &second_name, seed ^ 2, search_nodes)?;
    let report = run_match(
        Rules::classic_generic(),
        scenario,
        &mut *first_agent,
        &mut *second_agent,
        action_limit,
    )?;
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

fn create_agent(
    kind: AgentKind,
    name: &str,
    seed: u64,
    search_nodes: usize,
) -> Result<Box<dyn Agent>> {
    let agent: Box<dyn Agent> = match kind {
        AgentKind::Greedy => Box::new(GreedyAgent::new(name)),
        AgentKind::Random => Box::new(RandomAgent::new(name, seed)),
        AgentKind::Search => Box::new(
            SearchAgent::with_config(
                name,
                SearchConfig {
                    node_budget: search_nodes,
                    ..SearchConfig::default()
                },
            )
            .context("invalid search configuration")?,
        ),
    };
    Ok(agent)
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

#[expect(clippy::cast_precision_loss)]
#[expect(clippy::too_many_arguments)]
fn compare_agents(
    pairs: u32,
    seed: u64,
    width: u16,
    height: u16,
    action_limit: u32,
    first_kind: AgentKind,
    second_kind: AgentKind,
    search_nodes: usize,
    rules_kind: RulesKind,
    json: bool,
) -> Result<()> {
    anyhow::ensure!(pairs > 0, "pairs must be greater than zero");
    anyhow::ensure!(
        first_kind != second_kind,
        "comparison agents must be different"
    );
    let first_name = first_kind.name();
    let second_name = second_kind.name();
    let mut first_rating = Rating::default();
    let mut second_rating = Rating::default();
    let mut first_wins = 0;
    let mut second_wins = 0;
    let mut draws = 0;
    let mut actions = 0_u64;
    let started = Instant::now();

    for pair in 0..pairs {
        let game_seed = seed.wrapping_add(u64::from(pair));
        for first_starts in [true, false] {
            let scenario = symmetric_duel(width, height, game_seed)?;
            let first_agent_seed = game_seed ^ 0xa076_1d64_78bd_642f;
            let second_agent_seed = game_seed ^ 0xe703_7ed1_a0b4_28db;
            let mut first = create_agent(first_kind, first_name, first_agent_seed, search_nodes)?;
            let mut second =
                create_agent(second_kind, second_name, second_agent_seed, search_nodes)?;
            let report = if first_starts {
                run_match(
                    rules_kind.rules(),
                    scenario,
                    &mut *first,
                    &mut *second,
                    action_limit,
                )?
            } else {
                run_match(
                    rules_kind.rules(),
                    scenario,
                    &mut *second,
                    &mut *first,
                    action_limit,
                )?
            };
            actions += u64::from(report.outcome.actions);
            let first_score = if first_starts {
                score_for_first(report.outcome)
            } else {
                1.0 - score_for_first(report.outcome)
            };
            if first_score > 0.5 {
                first_wins += 1;
            } else if first_score < 0.5 {
                second_wins += 1;
            } else {
                draws += 1;
            }
            Elo::default().update(&mut first_rating, &mut second_rating, first_score);
        }
    }

    let elapsed_seconds = started.elapsed().as_secs_f64();
    let games = pairs * 2;
    let first_score = (f64::from(first_wins) + f64::from(draws) * 0.5) / f64::from(games);
    let edge = 0.5 / f64::from(games);
    let clipped_score = first_score.clamp(edge, 1.0 - edge);
    let relative_elo = 400.0 * (clipped_score / (1.0 - clipped_score)).log10();
    let summary = ComparisonSummary {
        first: first_name.to_owned(),
        second: second_name.to_owned(),
        rules: rules_kind.name().to_owned(),
        seed,
        width,
        height,
        action_limit,
        search_nodes,
        pairs,
        games,
        first_wins,
        second_wins,
        draws,
        first_score,
        relative_elo,
        first_rating,
        second_rating,
        actions,
        elapsed_seconds,
        actions_per_second: actions as f64 / elapsed_seconds,
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
fn rl_bench(
    environments: usize,
    transitions: u64,
    action_limit: u32,
    map: &RlMapArgs,
    json: bool,
) -> Result<()> {
    anyhow::ensure!(transitions > 0, "transitions must be greater than zero");
    let setup_started = Instant::now();
    let mut batch = map.create_batch(environments, action_limit)?;
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    let playable_hexes_per_environment = batch
        .game(0)
        .context("benchmark batch must contain an environment")?
        .topology()
        .playable_hexes()
        .len();
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
        map: map.name(),
        width: map.width,
        height: map.height,
        players: map.players,
        playable_hexes_per_environment,
        environments,
        transitions: completed,
        observation_batches,
        setup_seconds,
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
