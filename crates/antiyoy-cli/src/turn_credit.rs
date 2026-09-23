use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Instant;

use antiyoy_agents::{
    Agent, GreedyAgent, SearchAgent, SearchConfig, position_score, search_turn_slate,
};
use antiyoy_core::{Action, Game, GeneratorConfig, PlayerId, Rules, adjudicate};
use antiyoy_rl::BatchObservation;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::{RlMapKind, TurnCreditArgs, outcome_score, print_value};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Continuation {
    winner: Option<u8>,
    truncated: bool,
    actions: u32,
    reply_score: Option<i64>,
}

#[derive(Debug, Serialize)]
struct BranchRecord {
    actions: Vec<Action>,
    static_score: i64,
    continuation: Continuation,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_index: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ObservationRecord {
    root: BatchObservation,
    post_turn: BatchObservation,
}

#[derive(Debug, Serialize)]
struct AlternativeRecord {
    search_nodes: usize,
    search_expansions: usize,
    different_end_state: bool,
    branch: BranchRecord,
}

#[derive(Debug, Serialize)]
struct BeamRecord {
    rank: usize,
    different_end_state: bool,
    branch: BranchRecord,
}

#[derive(Debug, Serialize)]
struct CreditRecord {
    seed: u64,
    seat: u8,
    round: u32,
    search_expansions: usize,
    different_end_state: bool,
    distinct_end_states: usize,
    greedy: BranchRecord,
    search: BranchRecord,
    alternatives: Vec<AlternativeRecord>,
    beam_candidates: Vec<BeamRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_search_continuations: Option<Vec<Continuation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    opponent_search_continuations: Option<Vec<Continuation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observations: Option<ObservationRecord>,
}

#[derive(Debug, Serialize)]
struct TurnCreditSummary {
    generator: GeneratorConfig,
    rules: &'static str,
    maps: u32,
    maps_with_samples: u32,
    target_round: u32,
    rollin_seat: Option<u8>,
    reply_rollin: bool,
    rollout_limit: u32,
    search_nodes: usize,
    alternative_search_nodes: Vec<usize>,
    beam_slate_size: usize,
    include_observations: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_search_nodes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    opponent_search_nodes: Option<usize>,
    maximum_actions_per_turn: usize,
    positions: usize,
    different_end_states: u32,
    search_static_improvements: u32,
    search_outcome_better: u32,
    search_outcome_worse: u32,
    search_outcome_same: u32,
    censored_positions: u32,
    continuation_truncations: u32,
    slate_censored_positions: u32,
    slate_continuation_truncations: u32,
    elapsed_seconds: f64,
    records: Vec<CreditRecord>,
}

struct CreditConfig {
    target_round: u32,
    rollin_seat: Option<u8>,
    reply_rollin: bool,
    rollout_limit: u32,
    search: SearchConfig,
    alternative_search_nodes: Vec<usize>,
    beam_slate_size: usize,
    include_observations: bool,
    root_search_nodes: Option<usize>,
    opponent_search_nodes: Option<usize>,
}

struct ProbeContinuations {
    root: Option<Vec<Continuation>>,
    opponent: Option<Vec<Continuation>>,
}

pub(super) fn run(arguments: &TurnCreditArgs) -> Result<()> {
    ensure!(
        arguments.map.map == RlMapKind::Procedural,
        "turn credit requires a procedural map"
    );
    let summary = diagnose(
        arguments.map.generator_config(arguments.seed),
        &arguments.rules.rules(),
        arguments.rules.name(),
        arguments.maps,
        CreditConfig {
            target_round: arguments.target_round,
            rollin_seat: arguments.rollin_seat,
            reply_rollin: arguments.reply_rollin,
            rollout_limit: arguments.rollout_limit,
            search: SearchConfig {
                node_budget: arguments.search_nodes,
                maximum_actions_per_turn: arguments.maximum_actions_per_turn,
                ..SearchConfig::default()
            },
            alternative_search_nodes: arguments.alternative_search_nodes.clone(),
            beam_slate_size: arguments.beam_slate_size,
            include_observations: arguments.include_observations,
            root_search_nodes: arguments.root_search_nodes,
            opponent_search_nodes: arguments.opponent_search_nodes,
        },
    )?;
    print_value(&summary, arguments.json)
}

fn diagnose(
    generator: GeneratorConfig,
    rules: &Rules,
    rules_name: &'static str,
    maps: u32,
    config: CreditConfig,
) -> Result<TurnCreditSummary> {
    validate_credit_config(&generator, maps, &config)?;
    let mut records = Vec::new();
    let mut maps_with_samples = 0;
    let mut different_end_states = 0;
    let mut search_static_improvements = 0;
    let mut search_outcome_better = 0;
    let mut search_outcome_worse = 0;
    let mut search_outcome_same = 0;
    let mut censored_positions = 0;
    let mut continuation_truncations = 0;
    let mut slate_censored_positions = 0;
    let mut slate_continuation_truncations = 0;
    let started = Instant::now();

    for map_index in 0..maps {
        let mut map_config = generator.clone();
        map_config.seed = generator.seed.wrapping_add(u64::from(map_index));
        let game = Game::new(rules.clone(), map_config.generate()?)?;
        let states = sample_states(game, &config)?;
        maps_with_samples += u32::from(!states.is_empty());
        for state in states {
            let record = analyze_turn(&state, map_config.seed, &config)?;
            different_end_states += u32::from(record.different_end_state);
            search_static_improvements +=
                u32::from(record.search.static_score > record.greedy.static_score);
            let pair_truncations = u32::from(record.greedy.continuation.truncated)
                + u32::from(record.search.continuation.truncated);
            continuation_truncations += pair_truncations;
            let extra_truncations = extra_truncations(&record)?;
            slate_continuation_truncations += pair_truncations + extra_truncations;
            slate_censored_positions += u32::from(pair_truncations + extra_truncations > 0);
            if record.greedy.continuation.truncated || record.search.continuation.truncated {
                censored_positions += 1;
            } else {
                match outcome_score(record.search.continuation.winner, record.seat).cmp(
                    &outcome_score(record.greedy.continuation.winner, record.seat),
                ) {
                    Ordering::Greater => search_outcome_better += 1,
                    Ordering::Less => search_outcome_worse += 1,
                    Ordering::Equal => search_outcome_same += 1,
                }
            }
            records.push(record);
        }
    }

    Ok(TurnCreditSummary {
        generator,
        rules: rules_name,
        maps,
        maps_with_samples,
        target_round: config.target_round,
        rollin_seat: config.rollin_seat,
        reply_rollin: config.reply_rollin,
        rollout_limit: config.rollout_limit,
        search_nodes: config.search.node_budget,
        alternative_search_nodes: config.alternative_search_nodes,
        beam_slate_size: config.beam_slate_size,
        include_observations: config.include_observations,
        root_search_nodes: config.root_search_nodes,
        opponent_search_nodes: config.opponent_search_nodes,
        maximum_actions_per_turn: config.search.maximum_actions_per_turn,
        positions: records.len(),
        different_end_states,
        search_static_improvements,
        search_outcome_better,
        search_outcome_worse,
        search_outcome_same,
        censored_positions,
        continuation_truncations,
        slate_censored_positions,
        slate_continuation_truncations,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        records,
    })
}

fn validate_credit_config(
    generator: &GeneratorConfig,
    maps: u32,
    config: &CreditConfig,
) -> Result<()> {
    ensure!(maps > 0, "map count must be positive");
    ensure!(config.target_round > 0, "target round must be positive");
    ensure!(config.rollout_limit > 0, "rollout limit must be positive");
    ensure!(
        !config.reply_rollin || config.rollin_seat.is_none(),
        "reply roll-in cannot be combined with a fixed divergence seat"
    );
    ensure!(
        !config.reply_rollin
            || (config.beam_slate_size > 0 && config.opponent_search_nodes.is_some()),
        "reply roll-in requires a beam slate and opponent search nodes"
    );
    ensure!(
        config
            .rollin_seat
            .is_none_or(|seat| seat < generator.players),
        "roll-in seat must belong to the player range"
    );
    SearchAgent::with_config("validation", config.search)
        .context("invalid search configuration")?;
    for (nodes, role) in [
        (config.root_search_nodes, "root"),
        (config.opponent_search_nodes, "opponent"),
    ] {
        let Some(nodes) = nodes else { continue };
        ensure!(
            config.include_observations,
            "{role} search probes require post-turn observations"
        );
        SearchAgent::with_config(
            role,
            SearchConfig {
                node_budget: nodes,
                ..config.search
            },
        )
        .with_context(|| format!("invalid {role} search configuration"))?;
    }
    let mut alternative_budgets = HashSet::new();
    for &budget in &config.alternative_search_nodes {
        ensure!(
            budget != config.search.node_budget && alternative_budgets.insert(budget),
            "alternative search budgets must be unique and differ from the primary budget"
        );
        SearchAgent::with_config(
            "alternative-validation",
            SearchConfig {
                node_budget: budget,
                ..config.search
            },
        )
        .context("invalid alternative search configuration")?;
    }
    Ok(())
}

fn sample_states(game: Game, config: &CreditConfig) -> Result<Vec<Game>> {
    if let Some(seat) = config.rollin_seat {
        Ok(sample_first_search_divergence(game, seat, config)?
            .into_iter()
            .collect())
    } else if config.reply_rollin {
        let mut agent = SearchAgent::with_reply_search(
            "reply-rollin",
            config.search,
            config.beam_slate_size,
            config
                .opponent_search_nodes
                .expect("reply roll-in requires opponent search nodes"),
        )
        .context("invalid reply roll-in search configuration")?;
        sample_round_states(game, config.target_round, config.rollout_limit, &mut agent)
    } else {
        sample_round_states(
            game,
            config.target_round,
            config.rollout_limit,
            &mut GreedyAgent::new("greedy-sampling"),
        )
    }
}

fn extra_truncations(record: &CreditRecord) -> Result<u32> {
    let alternatives = record
        .alternatives
        .iter()
        .filter(|candidate| candidate.branch.continuation.truncated)
        .count();
    let beam = record
        .beam_candidates
        .iter()
        .filter(|candidate| candidate.branch.continuation.truncated)
        .count();
    Ok(u32::try_from(alternatives + beam)?)
}

fn analyze_turn(state: &Game, seed: u64, config: &CreditConfig) -> Result<CreditRecord> {
    let seat = state.active_player().0;
    let round = state.round();
    let search_slate = search_turn_slate(state, config.search, config.beam_slate_size.max(1))
        .context("invalid search configuration")?;
    let greedy_turn = finish_turn(
        state.clone(),
        &mut GreedyAgent::new("greedy"),
        config.search.maximum_actions_per_turn,
    )?;
    let selected = &search_slate.turns[0];
    let search_turn = TurnBranch {
        game: selected.game.clone(),
        actions: selected.actions.clone(),
        score: selected.score,
    };
    ensure!(
        search_turn.score >= greedy_turn.score,
        "search score fell below its greedy-turn fallback"
    );
    let different = greedy_turn.game != search_turn.game;
    let greedy_result = continue_with_greedy(
        greedy_turn.game.clone(),
        PlayerId(seat),
        config.rollout_limit,
    )?;
    let search_result = if different {
        continue_with_greedy(
            search_turn.game.clone(),
            PlayerId(seat),
            config.rollout_limit,
        )?
    } else {
        greedy_result
    };
    let mut completed_states = vec![(greedy_turn.game.clone(), greedy_result)];
    if different {
        completed_states.push((search_turn.game.clone(), search_result));
    }
    let alternatives =
        compare_alternative_budgets(state, config, &greedy_turn.game, &mut completed_states)?;
    let mut beam_candidates = Vec::with_capacity(search_slate.turns.len().saturating_sub(1));
    for (rank, turn) in search_slate.turns.into_iter().enumerate().skip(1) {
        let (state_index, continuation) = continuation_for_state(
            &turn.game,
            PlayerId(seat),
            config.rollout_limit,
            &mut completed_states,
        )?;
        beam_candidates.push(BeamRecord {
            rank,
            different_end_state: turn.game != greedy_turn.game,
            branch: BranchRecord {
                actions: turn.actions,
                static_score: turn.score,
                continuation,
                state_index: config.include_observations.then_some(state_index),
            },
        });
    }
    let observations = config
        .include_observations
        .then(|| observe_states(state, &completed_states));
    let probes = probe_continuations(&completed_states, PlayerId(seat), config)?;
    Ok(CreditRecord {
        seed,
        seat,
        round,
        search_expansions: search_slate.stats.nodes,
        different_end_state: different,
        distinct_end_states: completed_states.len(),
        greedy: BranchRecord {
            actions: greedy_turn.actions,
            static_score: greedy_turn.score,
            continuation: greedy_result,
            state_index: config.include_observations.then_some(0),
        },
        search: BranchRecord {
            actions: search_turn.actions,
            static_score: search_turn.score,
            continuation: search_result,
            state_index: config
                .include_observations
                .then_some(usize::from(different)),
        },
        alternatives,
        beam_candidates,
        root_search_continuations: probes.root,
        opponent_search_continuations: probes.opponent,
        observations,
    })
}

fn probe_continuations(
    states: &[(Game, Continuation)],
    player: PlayerId,
    config: &CreditConfig,
) -> Result<ProbeContinuations> {
    Ok(ProbeContinuations {
        root: probe_completed_states(states, player, config, config.root_search_nodes, None)?,
        opponent: probe_completed_states(
            states,
            player,
            config,
            None,
            config.opponent_search_nodes,
        )?,
    })
}

fn probe_completed_states(
    states: &[(Game, Continuation)],
    player: PlayerId,
    config: &CreditConfig,
    root_search_nodes: Option<usize>,
    opponent_search_nodes: Option<usize>,
) -> Result<Option<Vec<Continuation>>> {
    if root_search_nodes.is_none() && opponent_search_nodes.is_none() {
        return Ok(None);
    }
    let search_config = |nodes: Option<usize>| {
        nodes.map(|node_budget| SearchConfig {
            node_budget,
            ..config.search
        })
    };
    states
        .iter()
        .map(|(game, _)| {
            continue_with_policy(
                game.clone(),
                player,
                config.rollout_limit,
                search_config(root_search_nodes),
                search_config(opponent_search_nodes),
            )
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn compare_alternative_budgets(
    state: &Game,
    config: &CreditConfig,
    greedy_end_state: &Game,
    completed_states: &mut Vec<(Game, Continuation)>,
) -> Result<Vec<AlternativeRecord>> {
    let mut alternatives = Vec::with_capacity(config.alternative_search_nodes.len());
    for &nodes in &config.alternative_search_nodes {
        let mut agent = SearchAgent::with_config(
            "alternative-search",
            SearchConfig {
                node_budget: nodes,
                ..config.search
            },
        )
        .context("invalid alternative search configuration")?;
        let turn = finish_turn(
            state.clone(),
            &mut agent,
            config.search.maximum_actions_per_turn,
        )?;
        let (state_index, continuation) = continuation_for_state(
            &turn.game,
            state.active_player(),
            config.rollout_limit,
            completed_states,
        )?;
        alternatives.push(AlternativeRecord {
            search_nodes: nodes,
            search_expansions: agent.last_stats().nodes,
            different_end_state: turn.game != *greedy_end_state,
            branch: BranchRecord {
                actions: turn.actions,
                static_score: turn.score,
                continuation,
                state_index: config.include_observations.then_some(state_index),
            },
        });
    }
    Ok(alternatives)
}

fn continuation_for_state(
    game: &Game,
    player: PlayerId,
    action_limit: u32,
    completed_states: &mut Vec<(Game, Continuation)>,
) -> Result<(usize, Continuation)> {
    if let Some((index, (_, result))) = completed_states
        .iter()
        .enumerate()
        .find(|(_, (state, _))| state == game)
    {
        return Ok((index, *result));
    }
    let result = continue_with_greedy(game.clone(), player, action_limit)?;
    let index = completed_states.len();
    completed_states.push((game.clone(), result));
    Ok((index, result))
}

fn observe_states(root: &Game, states: &[(Game, Continuation)]) -> ObservationRecord {
    let mut root_actions = Vec::new();
    root.legal_actions(&mut root_actions);
    let mut root_observation = BatchObservation::default();
    root_observation.observe_game(root, &root_actions, false);

    let legal_actions = states
        .iter()
        .map(|(game, _)| {
            let mut actions = Vec::new();
            game.legal_actions(&mut actions);
            actions
        })
        .collect::<Vec<_>>();
    let games = states
        .iter()
        .zip(&legal_actions)
        .map(|((game, _), actions)| (game, actions.as_slice()))
        .collect::<Vec<_>>();
    let mut post_turn = BatchObservation::default();
    post_turn.observe_games(&games, false);
    ObservationRecord {
        root: root_observation,
        post_turn,
    }
}

fn sample_round_states(
    mut game: Game,
    target_round: u32,
    action_limit: u32,
    agent: &mut dyn Agent,
) -> Result<Vec<Game>> {
    let mut samples = vec![None; usize::from(game.player_count())];
    let mut legal_actions = Vec::new();
    for _ in 0..action_limit {
        if game.is_terminal() || game.round() > target_round {
            break;
        }
        let seat = game.active_player().index();
        if game.round() == target_round && samples[seat].is_none() {
            samples[seat] = Some(game.clone());
        }
        game.legal_actions(&mut legal_actions);
        let action = agent.select_action(&game, &legal_actions);
        game.step(action)?;
    }
    Ok(samples.into_iter().flatten().collect())
}

fn sample_first_search_divergence(
    mut game: Game,
    seat: u8,
    config: &CreditConfig,
) -> Result<Option<Game>> {
    let mut search = SearchAgent::with_config("search-rollin", config.search)
        .context("invalid search configuration")?;
    let mut actions = 0;
    while !game.is_terminal()
        && game.round() <= config.target_round
        && actions < config.rollout_limit
    {
        let remaining = usize::try_from(config.rollout_limit - actions)?;
        let turn = if game.active_player().0 == seat {
            let turn_limit = config.search.maximum_actions_per_turn.min(remaining);
            let greedy_turn = finish_turn(
                game.clone(),
                &mut GreedyAgent::new("greedy-alternative"),
                turn_limit,
            )?;
            let search_turn = finish_turn(game.clone(), &mut search, turn_limit)?;
            if greedy_turn.game != search_turn.game {
                return Ok(Some(game));
            }
            search_turn
        } else {
            finish_turn(game, &mut GreedyAgent::new("greedy-opponent"), remaining)?
        };
        actions += u32::try_from(turn.actions.len())?;
        game = turn.game;
    }
    Ok(None)
}

struct TurnBranch {
    game: Game,
    actions: Vec<Action>,
    score: i64,
}

fn finish_turn(
    mut game: Game,
    agent: &mut dyn Agent,
    maximum_actions: usize,
) -> Result<TurnBranch> {
    let player = game.active_player();
    let mut actions = Vec::new();
    let mut legal_actions = Vec::new();
    for action_index in 0..maximum_actions {
        game.legal_actions(&mut legal_actions);
        let action = if action_index + 1 == maximum_actions {
            Action::EndTurn
        } else {
            agent.select_action(&game, &legal_actions)
        };
        game.step(action)?;
        actions.push(action);
        if game.is_terminal() || game.active_player() != player {
            break;
        }
    }
    Ok(TurnBranch {
        score: position_score(&game, player),
        game,
        actions,
    })
}

fn continue_with_greedy(game: Game, player: PlayerId, action_limit: u32) -> Result<Continuation> {
    continue_with_policy(game, player, action_limit, None, None)
}

fn continue_with_policy(
    mut game: Game,
    player: PlayerId,
    action_limit: u32,
    root_search: Option<SearchConfig>,
    opponent_search: Option<SearchConfig>,
) -> Result<Continuation> {
    if game.is_terminal() {
        return Ok(Continuation {
            winner: game.winner().map(|winner| winner.0),
            truncated: false,
            actions: 0,
            reply_score: None,
        });
    }
    let mut greedy = GreedyAgent::new("greedy-continuation");
    let mut root_search = root_search
        .map(|config| SearchAgent::with_config("root-search-continuation", config))
        .transpose()
        .context("invalid root search configuration")?;
    let mut search = opponent_search
        .map(|config| SearchAgent::with_config("opponent-search-continuation", config))
        .transpose()
        .context("invalid opponent search configuration")?;
    let mut legal_actions = Vec::new();
    let mut reply_score = None;
    for action_index in 0..action_limit {
        game.legal_actions(&mut legal_actions);
        let action = if game.active_player() == player {
            if let Some(search) = root_search.as_mut() {
                search.select_action(&game, &legal_actions)
            } else {
                greedy.select_action(&game, &legal_actions)
            }
        } else if let Some(search) = search.as_mut() {
            search.select_action(&game, &legal_actions)
        } else {
            greedy.select_action(&game, &legal_actions)
        };
        let transition = game.step(action)?;
        if transition.terminal {
            return Ok(Continuation {
                winner: transition.winner.map(|winner| winner.0),
                truncated: false,
                actions: action_index + 1,
                reply_score,
            });
        }
        if game.active_player() == player && reply_score.is_none() {
            reply_score = Some(position_score(&game, player));
        }
    }
    Ok(Continuation {
        winner: adjudicate(&game).map(|winner| winner.0),
        truncated: true,
        actions: action_limit,
        reply_score,
    })
}

#[cfg(test)]
mod tests {
    use antiyoy_agents::{GreedyAgent, SearchAgent, SearchConfig};
    use antiyoy_core::{Game, GeneratorConfig, Rules};

    use super::{
        CreditConfig, continue_with_greedy, diagnose, finish_turn, sample_first_search_divergence,
        sample_round_states,
    };

    #[test]
    fn samples_each_surviving_seat_at_the_start_of_the_requested_round() {
        let game = Game::new(
            Rules::classic_generic(),
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 3,
                seed: 504,
                ..GeneratorConfig::default()
            }
            .generate()
            .expect("valid map"),
        )
        .expect("valid game");
        let states = sample_round_states(game, 1, 100, &mut GreedyAgent::new("greedy-sampling"))
            .expect("valid sampling");
        assert_eq!(states.len(), 3);
        for (seat, state) in states.iter().enumerate() {
            assert_eq!(state.round(), 1);
            assert_eq!(state.active_player().index(), seat);
        }
    }

    #[test]
    fn reply_rollin_samples_each_surviving_seat_at_the_target_round() {
        let game = Game::new(
            Rules::classic_generic(),
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 2,
                seed: 6260000,
                ..GeneratorConfig::default()
            }
            .generate()
            .expect("valid map"),
        )
        .expect("valid game");
        let config = SearchConfig {
            node_budget: 32,
            maximum_actions_per_turn: 8,
            ..SearchConfig::default()
        };
        let mut agent = SearchAgent::with_reply_search("reply-rollin", config, 4, 8)
            .expect("valid reply search");
        let states = sample_round_states(game, 1, 100, &mut agent).expect("valid sampling");
        assert_eq!(states.len(), 2);
        for (seat, state) in states.iter().enumerate() {
            assert_eq!(state.round(), 1);
            assert_eq!(state.active_player().index(), seat);
        }
    }

    #[test]
    fn reply_rollin_diagnostic_records_the_sampling_policy() {
        let report = diagnose(
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 2,
                seed: 6260000,
                ..GeneratorConfig::default()
            },
            &Rules::classic_generic(),
            "classic_generic_2022",
            1,
            CreditConfig {
                target_round: 1,
                rollin_seat: None,
                reply_rollin: true,
                rollout_limit: 64,
                search: SearchConfig {
                    node_budget: 32,
                    maximum_actions_per_turn: 8,
                    ..SearchConfig::default()
                },
                alternative_search_nodes: Vec::new(),
                beam_slate_size: 4,
                include_observations: true,
                root_search_nodes: None,
                opponent_search_nodes: Some(8),
            },
        )
        .expect("valid reply roll-in diagnostic");
        assert!(report.reply_rollin);
        assert_eq!(report.positions, 2);
        assert!(report.records.iter().all(|record| record.round == 1));
    }

    #[test]
    fn search_turn_never_scores_below_its_greedy_fallback() {
        let game = Game::new(
            Rules::classic_generic(),
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 3,
                seed: 505,
                ..GeneratorConfig::default()
            }
            .generate()
            .expect("valid map"),
        )
        .expect("valid game");
        let config = SearchConfig {
            node_budget: 32,
            maximum_actions_per_turn: 8,
            ..SearchConfig::default()
        };
        let player = game.active_player();
        let greedy = finish_turn(game.clone(), &mut GreedyAgent::new("greedy"), 8)
            .expect("valid greedy turn");
        let search = finish_turn(
            game,
            &mut SearchAgent::with_config("search", config).expect("valid search"),
            8,
        )
        .expect("valid searched turn");
        assert!(search.score >= greedy.score);
        assert!(!greedy.actions.is_empty());
        assert!(!search.actions.is_empty());
        let first =
            continue_with_greedy(greedy.game.clone(), player, 32).expect("valid continuation");
        let second = continue_with_greedy(greedy.game, player, 32).expect("same continuation");
        assert_eq!(first, second);
    }

    #[test]
    fn diagnostic_pairs_every_sample_and_preserves_the_rollout_contract() {
        let report = diagnose(
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 3,
                seed: 506,
                ..GeneratorConfig::default()
            },
            &Rules::classic_generic(),
            "classic_generic_2022",
            1,
            CreditConfig {
                target_round: 1,
                rollin_seat: None,
                reply_rollin: false,
                rollout_limit: 32,
                search: SearchConfig {
                    node_budget: 2,
                    maximum_actions_per_turn: 8,
                    ..SearchConfig::default()
                },
                alternative_search_nodes: vec![4, 8],
                beam_slate_size: 4,
                include_observations: true,
                root_search_nodes: Some(2),
                opponent_search_nodes: Some(2),
            },
        )
        .expect("valid diagnostic");
        assert_eq!(report.positions, 3);
        assert_eq!(
            report.search_outcome_better
                + report.search_outcome_worse
                + report.search_outcome_same
                + report.censored_positions,
            3
        );
        assert!(report.censored_positions > 0);
        assert!(report.slate_censored_positions >= report.censored_positions);
        assert!(report.slate_continuation_truncations >= report.continuation_truncations);
        assert!(report.records.iter().all(|record| {
            record.greedy.continuation.actions <= 32
                && record.search.continuation.actions <= 32
                && record.search.static_score >= record.greedy.static_score
                && record.alternatives.len() == 2
                && record.beam_candidates.len() <= 3
                && record.distinct_end_states <= 7
        }));
        for record in &report.records {
            let observations = record.observations.as_ref().expect("opt-in observations");
            assert_eq!(observations.root.active_players, vec![record.seat]);
            assert_eq!(
                observations.post_turn.widths.len(),
                record.distinct_end_states
            );
            assert_eq!(
                observations.post_turn.cell_offsets.len(),
                record.distinct_end_states + 1
            );
            let probes = record
                .opponent_search_continuations
                .as_ref()
                .expect("opt-in opponent search probes");
            assert_eq!(probes.len(), record.distinct_end_states);
            assert!(probes.iter().all(|probe| probe.actions <= 32));
            let root_probes = record
                .root_search_continuations
                .as_ref()
                .expect("opt-in root search probes");
            assert_eq!(root_probes.len(), record.distinct_end_states);
            assert!(root_probes.iter().all(|probe| probe.actions <= 32));
            let branches = std::iter::once(&record.greedy)
                .chain(std::iter::once(&record.search))
                .chain(
                    record
                        .alternatives
                        .iter()
                        .map(|candidate| &candidate.branch),
                )
                .chain(
                    record
                        .beam_candidates
                        .iter()
                        .map(|candidate| &candidate.branch),
                );
            let mut continuations = vec![None; record.distinct_end_states];
            for branch in branches {
                let index = branch.state_index.expect("mapped branch");
                assert!(index < record.distinct_end_states);
                match continuations[index] {
                    Some(continuation) => assert_eq!(continuation, branch.continuation),
                    None => continuations[index] = Some(branch.continuation),
                }
            }
            assert!(continuations.into_iter().all(|value| value.is_some()));
        }
    }

    #[test]
    fn observations_are_absent_unless_requested() {
        let report = diagnose(
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 3,
                seed: 506,
                ..GeneratorConfig::default()
            },
            &Rules::classic_generic(),
            "classic_generic_2022",
            1,
            CreditConfig {
                target_round: 1,
                rollin_seat: None,
                reply_rollin: false,
                rollout_limit: 32,
                search: SearchConfig {
                    node_budget: 2,
                    maximum_actions_per_turn: 8,
                    ..SearchConfig::default()
                },
                alternative_search_nodes: Vec::new(),
                beam_slate_size: 2,
                include_observations: false,
                root_search_nodes: None,
                opponent_search_nodes: None,
            },
        )
        .expect("valid diagnostic");
        for record in &report.records {
            assert!(record.observations.is_none());
            assert!(record.root_search_continuations.is_none());
            assert!(record.opponent_search_continuations.is_none());
            assert!(record.greedy.state_index.is_none());
            assert!(record.search.state_index.is_none());
            assert!(
                record
                    .beam_candidates
                    .iter()
                    .all(|candidate| candidate.branch.state_index.is_none())
            );
        }
        let serialized = serde_json::to_value(report).expect("serializable report");
        assert!(serialized["records"][0].get("observations").is_none());
        assert!(serialized.get("opponent_search_nodes").is_none());
        assert!(serialized.get("root_search_nodes").is_none());
        assert!(
            serialized["records"][0]
                .get("root_search_continuations")
                .is_none()
        );
        assert!(
            serialized["records"][0]
                .get("opponent_search_continuations")
                .is_none()
        );
        assert!(
            serialized["records"][0]["greedy"]
                .get("state_index")
                .is_none()
        );
    }

    #[test]
    fn rollin_seat_must_exist_on_the_map() {
        let result = diagnose(
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 3,
                ..GeneratorConfig::default()
            },
            &Rules::classic_generic(),
            "classic_generic_2022",
            1,
            CreditConfig {
                target_round: 1,
                rollin_seat: Some(3),
                reply_rollin: false,
                rollout_limit: 32,
                search: SearchConfig::default(),
                alternative_search_nodes: Vec::new(),
                beam_slate_size: 0,
                include_observations: false,
                root_search_nodes: None,
                opponent_search_nodes: None,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn alternative_search_budgets_must_be_distinct_and_valid() {
        for budgets in [vec![2], vec![4, 4], vec![1]] {
            let result = diagnose(
                GeneratorConfig {
                    schema_version: 2,
                    width: 11,
                    height: 9,
                    players: 3,
                    ..GeneratorConfig::default()
                },
                &Rules::classic_generic(),
                "classic_generic_2022",
                1,
                CreditConfig {
                    target_round: 1,
                    rollin_seat: None,
                    reply_rollin: false,
                    rollout_limit: 32,
                    search: SearchConfig {
                        node_budget: 2,
                        ..SearchConfig::default()
                    },
                    alternative_search_nodes: budgets,
                    beam_slate_size: 0,
                    include_observations: false,
                    root_search_nodes: None,
                    opponent_search_nodes: None,
                },
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn rollin_stops_at_the_action_limit() {
        let game = Game::new(
            Rules::classic_generic(),
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 3,
                seed: 505,
                ..GeneratorConfig::default()
            }
            .generate()
            .expect("valid map"),
        )
        .expect("valid game");
        let result = sample_first_search_divergence(
            game,
            0,
            &CreditConfig {
                target_round: 100,
                rollin_seat: Some(0),
                reply_rollin: false,
                rollout_limit: 1,
                search: SearchConfig::default(),
                alternative_search_nodes: Vec::new(),
                beam_slate_size: 0,
                include_observations: false,
                root_search_nodes: None,
                opponent_search_nodes: None,
            },
        )
        .expect("valid roll-in");
        assert!(result.is_none());
    }
}
