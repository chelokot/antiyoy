use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Instant;

use antiyoy_agents::{Agent, GreedyAgent, SearchAgent, SearchConfig, position_score};
use antiyoy_core::{Action, Game, GeneratorConfig, PlayerId, Rules, adjudicate};
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
}

#[derive(Debug, Serialize)]
struct AlternativeRecord {
    search_nodes: usize,
    search_expansions: usize,
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
}

#[derive(Debug, Serialize)]
struct TurnCreditSummary {
    generator: GeneratorConfig,
    rules: &'static str,
    maps: u32,
    maps_with_samples: u32,
    target_round: u32,
    rollin_seat: Option<u8>,
    rollout_limit: u32,
    search_nodes: usize,
    alternative_search_nodes: Vec<usize>,
    maximum_actions_per_turn: usize,
    positions: usize,
    different_end_states: u32,
    search_static_improvements: u32,
    search_outcome_better: u32,
    search_outcome_worse: u32,
    search_outcome_same: u32,
    censored_positions: u32,
    continuation_truncations: u32,
    elapsed_seconds: f64,
    records: Vec<CreditRecord>,
}

struct CreditConfig {
    target_round: u32,
    rollin_seat: Option<u8>,
    rollout_limit: u32,
    search: SearchConfig,
    alternative_search_nodes: Vec<usize>,
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
            rollout_limit: arguments.rollout_limit,
            search: SearchConfig {
                node_budget: arguments.search_nodes,
                maximum_actions_per_turn: arguments.maximum_actions_per_turn,
                ..SearchConfig::default()
            },
            alternative_search_nodes: arguments.alternative_search_nodes.clone(),
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
    ensure!(maps > 0, "map count must be positive");
    ensure!(config.target_round > 0, "target round must be positive");
    ensure!(config.rollout_limit > 0, "rollout limit must be positive");
    ensure!(
        config
            .rollin_seat
            .is_none_or(|seat| seat < generator.players),
        "roll-in seat must belong to the player range"
    );
    SearchAgent::with_config("validation", config.search)
        .context("invalid search configuration")?;
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
    let mut records = Vec::new();
    let mut maps_with_samples = 0;
    let mut different_end_states = 0;
    let mut search_static_improvements = 0;
    let mut search_outcome_better = 0;
    let mut search_outcome_worse = 0;
    let mut search_outcome_same = 0;
    let mut censored_positions = 0;
    let mut continuation_truncations = 0;
    let started = Instant::now();

    for map_index in 0..maps {
        let mut map_config = generator.clone();
        map_config.seed = generator.seed.wrapping_add(u64::from(map_index));
        let game = Game::new(rules.clone(), map_config.generate()?)?;
        let states = if let Some(seat) = config.rollin_seat {
            sample_first_search_divergence(game, seat, &config)?
                .into_iter()
                .collect()
        } else {
            sample_round_states(game, config.target_round, config.rollout_limit)?
        };
        maps_with_samples += u32::from(!states.is_empty());
        for state in states {
            let record = analyze_turn(&state, map_config.seed, &config)?;
            different_end_states += u32::from(record.different_end_state);
            search_static_improvements +=
                u32::from(record.search.static_score > record.greedy.static_score);
            continuation_truncations += u32::from(record.greedy.continuation.truncated)
                + u32::from(record.search.continuation.truncated);
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
        rollout_limit: config.rollout_limit,
        search_nodes: config.search.node_budget,
        alternative_search_nodes: config.alternative_search_nodes,
        maximum_actions_per_turn: config.search.maximum_actions_per_turn,
        positions: records.len(),
        different_end_states,
        search_static_improvements,
        search_outcome_better,
        search_outcome_worse,
        search_outcome_same,
        censored_positions,
        continuation_truncations,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        records,
    })
}

fn analyze_turn(state: &Game, seed: u64, config: &CreditConfig) -> Result<CreditRecord> {
    let seat = state.active_player().0;
    let round = state.round();
    let mut search_agent = SearchAgent::with_config("search", config.search)
        .context("invalid search configuration")?;
    let greedy_turn = finish_turn(
        state.clone(),
        &mut GreedyAgent::new("greedy"),
        config.search.maximum_actions_per_turn,
    )?;
    let search_turn = finish_turn(
        state.clone(),
        &mut search_agent,
        config.search.maximum_actions_per_turn,
    )?;
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
    let mut alternatives = Vec::with_capacity(config.alternative_search_nodes.len());
    for &nodes in &config.alternative_search_nodes {
        let mut alternative_agent = SearchAgent::with_config(
            "alternative-search",
            SearchConfig {
                node_budget: nodes,
                ..config.search
            },
        )
        .context("invalid alternative search configuration")?;
        let alternative_turn = finish_turn(
            state.clone(),
            &mut alternative_agent,
            config.search.maximum_actions_per_turn,
        )?;
        let continuation = if let Some((_, result)) = completed_states
            .iter()
            .find(|(game, _)| game == &alternative_turn.game)
        {
            *result
        } else {
            let result = continue_with_greedy(
                alternative_turn.game.clone(),
                PlayerId(seat),
                config.rollout_limit,
            )?;
            completed_states.push((alternative_turn.game.clone(), result));
            result
        };
        alternatives.push(AlternativeRecord {
            search_nodes: nodes,
            search_expansions: alternative_agent.last_stats().nodes,
            different_end_state: alternative_turn.game != greedy_turn.game,
            branch: BranchRecord {
                actions: alternative_turn.actions,
                static_score: alternative_turn.score,
                continuation,
            },
        });
    }
    Ok(CreditRecord {
        seed,
        seat,
        round,
        search_expansions: search_agent.last_stats().nodes,
        different_end_state: different,
        distinct_end_states: completed_states.len(),
        greedy: BranchRecord {
            actions: greedy_turn.actions,
            static_score: greedy_turn.score,
            continuation: greedy_result,
        },
        search: BranchRecord {
            actions: search_turn.actions,
            static_score: search_turn.score,
            continuation: search_result,
        },
        alternatives,
    })
}

fn sample_round_states(mut game: Game, target_round: u32, action_limit: u32) -> Result<Vec<Game>> {
    let mut samples = vec![None; usize::from(game.player_count())];
    let mut greedy = GreedyAgent::new("greedy-sampling");
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
        let action = greedy.select_action(&game, &legal_actions);
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

fn continue_with_greedy(
    mut game: Game,
    player: PlayerId,
    action_limit: u32,
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
    let mut legal_actions = Vec::new();
    let mut reply_score = None;
    for action_index in 0..action_limit {
        game.legal_actions(&mut legal_actions);
        let action = greedy.select_action(&game, &legal_actions);
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
        let states = sample_round_states(game, 1, 100).expect("valid sampling");
        assert_eq!(states.len(), 3);
        for (seat, state) in states.iter().enumerate() {
            assert_eq!(state.round(), 1);
            assert_eq!(state.active_player().index(), seat);
        }
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
                rollout_limit: 32,
                search: SearchConfig {
                    node_budget: 2,
                    maximum_actions_per_turn: 8,
                    ..SearchConfig::default()
                },
                alternative_search_nodes: vec![4, 8],
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
        assert!(report.records.iter().all(|record| {
            record.greedy.continuation.actions <= 32
                && record.search.continuation.actions <= 32
                && record.search.static_score >= record.greedy.static_score
                && record.alternatives.len() == 2
                && record.distinct_end_states <= 4
        }));
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
                rollout_limit: 32,
                search: SearchConfig::default(),
                alternative_search_nodes: Vec::new(),
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
                    rollout_limit: 32,
                    search: SearchConfig {
                        node_budget: 2,
                        ..SearchConfig::default()
                    },
                    alternative_search_nodes: budgets,
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
                rollout_limit: 1,
                search: SearchConfig::default(),
                alternative_search_nodes: Vec::new(),
            },
        )
        .expect("valid roll-in");
        assert!(result.is_none());
    }
}
