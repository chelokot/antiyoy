use std::time::Instant;

use antiyoy_agents::{Agent, SearchAgent, SearchConfig};
use antiyoy_core::{Game, GeneratorConfig, Rules, Scenario, adjudicate};
use anyhow::{Result, ensure};
use serde::Serialize;

use crate::{AgentKind, MultiCompareArgs, RlMapKind, create_agent, outcome_score, print_value};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Episode {
    winner: Option<u8>,
    truncated: bool,
    actions: u32,
}

#[derive(Clone, Copy, Debug)]
struct SearchMatchSettings {
    nodes: usize,
    baseline_nodes: usize,
    reply_nodes: usize,
    slate_size: usize,
}

#[derive(Debug, Serialize)]
struct MatchRecord {
    seed: u64,
    candidate_seat: u8,
    candidate: Episode,
    baseline: Episode,
}

#[derive(Debug, Serialize)]
struct MultiCompareSummary {
    generator: GeneratorConfig,
    rules: &'static str,
    candidate: &'static str,
    baseline: &'static str,
    search_nodes: usize,
    baseline_search_nodes: usize,
    candidate_reply_nodes: usize,
    candidate_slate_size: usize,
    maps: u32,
    games: u32,
    baseline_reference_games: u32,
    action_limit: u32,
    candidate_wins_by_seat: Vec<u32>,
    baseline_wins_by_seat: Vec<u32>,
    paired_better: u32,
    paired_worse: u32,
    paired_same: u32,
    candidate_truncations: u32,
    baseline_reference_truncations: u32,
    actions: u64,
    elapsed_seconds: f64,
    records: Vec<MatchRecord>,
}

pub(super) fn run(arguments: &MultiCompareArgs) -> Result<()> {
    ensure!(
        arguments.map.map == RlMapKind::Procedural,
        "multiplayer comparison requires a procedural map"
    );
    ensure!(
        arguments.candidate != arguments.baseline
            || (arguments.candidate == AgentKind::Search
                && (arguments.candidate_reply_nodes > 0
                    || arguments
                        .baseline_search_nodes
                        .is_some_and(|nodes| nodes != arguments.search_nodes))),
        "comparison agents must be different"
    );
    ensure!(
        arguments.candidate_reply_nodes == 0 || arguments.candidate == AgentKind::Search,
        "reply search requires the search candidate"
    );
    ensure!(
        arguments.candidate_reply_nodes == 0 || arguments.map.players == 2,
        "reply search requires two-player maps"
    );
    ensure!(
        arguments.candidate_reply_nodes == 0 || arguments.candidate_slate_size > 0,
        "reply search slate size must be positive"
    );
    let summary = compare(
        arguments.map.generator_config(arguments.seed),
        &arguments.rules.rules(),
        arguments.rules.name(),
        arguments.maps,
        arguments.action_limit,
        arguments.candidate,
        arguments.baseline,
        SearchMatchSettings {
            nodes: arguments.search_nodes,
            baseline_nodes: arguments
                .baseline_search_nodes
                .unwrap_or(arguments.search_nodes),
            reply_nodes: arguments.candidate_reply_nodes,
            slate_size: arguments.candidate_slate_size,
        },
    )?;
    print_value(&summary, arguments.json)
}

#[expect(clippy::too_many_arguments)]
fn compare(
    generator: GeneratorConfig,
    rules: &Rules,
    rules_name: &'static str,
    maps: u32,
    action_limit: u32,
    candidate: AgentKind,
    baseline: AgentKind,
    search: SearchMatchSettings,
) -> Result<MultiCompareSummary> {
    ensure!(
        maps > 0 && action_limit > 0,
        "maps and action limit must be positive"
    );
    let games = maps
        .checked_mul(u32::from(generator.players))
        .ok_or_else(|| anyhow::anyhow!("map count and player count exceed the comparison limit"))?;
    let players = usize::from(generator.players);
    let mut candidate_wins_by_seat = vec![0; players];
    let mut baseline_wins_by_seat = vec![0; players];
    let mut paired_better = 0;
    let mut paired_worse = 0;
    let mut paired_same = 0;
    let mut candidate_truncations = 0;
    let mut baseline_reference_truncations = 0;
    let mut actions = 0_u64;
    let mut records = Vec::with_capacity(games as usize);
    let started = Instant::now();

    for map_index in 0..maps {
        let mut map_config = generator.clone();
        map_config.seed = generator.seed.wrapping_add(u64::from(map_index));
        let scenario = map_config.generate()?;
        let reference = play_episode(
            &scenario,
            rules,
            action_limit,
            baseline,
            None,
            SearchMatchSettings {
                reply_nodes: 0,
                ..search
            },
            map_config.seed,
        )?;
        actions += u64::from(reference.actions);
        baseline_reference_truncations += u32::from(reference.truncated);
        for player in 0..generator.players {
            let seat = usize::from(player);
            let outcome = play_episode(
                &scenario,
                rules,
                action_limit,
                baseline,
                Some((seat, candidate)),
                search,
                map_config.seed,
            )?;
            actions += u64::from(outcome.actions);
            candidate_truncations += u32::from(outcome.truncated);
            candidate_wins_by_seat[seat] += u32::from(outcome.winner == Some(player));
            baseline_wins_by_seat[seat] += u32::from(reference.winner == Some(player));
            match outcome_score(outcome.winner, player)
                .cmp(&outcome_score(reference.winner, player))
            {
                std::cmp::Ordering::Greater => paired_better += 1,
                std::cmp::Ordering::Less => paired_worse += 1,
                std::cmp::Ordering::Equal => paired_same += 1,
            }
            records.push(MatchRecord {
                seed: map_config.seed,
                candidate_seat: player,
                candidate: outcome,
                baseline: reference,
            });
        }
    }

    Ok(MultiCompareSummary {
        generator,
        rules: rules_name,
        candidate: if search.reply_nodes > 0 {
            "reply-search"
        } else {
            candidate.name()
        },
        baseline: baseline.name(),
        search_nodes: search.nodes,
        baseline_search_nodes: search.baseline_nodes,
        candidate_reply_nodes: search.reply_nodes,
        candidate_slate_size: search.slate_size,
        maps,
        games,
        baseline_reference_games: maps,
        action_limit,
        candidate_wins_by_seat,
        baseline_wins_by_seat,
        paired_better,
        paired_worse,
        paired_same,
        candidate_truncations,
        baseline_reference_truncations,
        actions,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        records,
    })
}

fn play_episode(
    scenario: &Scenario,
    rules: &Rules,
    action_limit: u32,
    baseline: AgentKind,
    candidate: Option<(usize, AgentKind)>,
    search: SearchMatchSettings,
    seed: u64,
) -> Result<Episode> {
    let mut game = Game::new(rules.clone(), scenario.clone())?;
    let mut agents: Vec<Box<dyn Agent>> = (0..usize::from(scenario.player_count))
        .map(|seat| {
            let is_candidate = candidate.is_some_and(|(candidate_seat, _)| candidate_seat == seat);
            let kind = candidate
                .filter(|(candidate_seat, _)| *candidate_seat == seat)
                .map_or(baseline, |(_, kind)| kind);
            if candidate == Some((seat, AgentKind::Search)) && search.reply_nodes > 0 {
                Ok(Box::new(
                    SearchAgent::with_reply_search(
                        "reply-search",
                        SearchConfig {
                            node_budget: search.nodes,
                            ..SearchConfig::default()
                        },
                        search.slate_size,
                        search.reply_nodes,
                    )
                    .map_err(anyhow::Error::from)?,
                ) as Box<dyn Agent>)
            } else {
                create_agent(
                    kind,
                    kind.name(),
                    seed ^ (seat as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    if is_candidate {
                        search.nodes
                    } else {
                        search.baseline_nodes
                    },
                )
            }
        })
        .collect::<Result<_>>()?;
    let mut legal_actions = Vec::new();
    for action_index in 0..action_limit {
        game.legal_actions(&mut legal_actions);
        let seat = game.active_player().index();
        let action = agents[seat].select_action(&game, &legal_actions);
        let transition = game.step(action)?;
        if transition.terminal {
            return Ok(Episode {
                winner: transition.winner.map(|winner| winner.0),
                truncated: false,
                actions: action_index + 1,
            });
        }
    }
    Ok(Episode {
        winner: adjudicate(&game).map(|winner| winner.0),
        truncated: true,
        actions: action_limit,
    })
}

#[cfg(test)]
mod tests {
    use antiyoy_core::{GeneratorConfig, Rules};

    use super::{AgentKind, SearchMatchSettings, compare, play_episode};
    use crate::outcome_score;

    #[test]
    fn identical_agents_reproduce_the_reference_for_every_seat() {
        let scenario = GeneratorConfig {
            schema_version: 2,
            width: 11,
            height: 9,
            players: 3,
            seed: 503,
            ..GeneratorConfig::default()
        }
        .generate()
        .expect("valid map");
        let rules = Rules::classic_generic();
        let search = SearchMatchSettings {
            nodes: 2,
            baseline_nodes: 2,
            reply_nodes: 0,
            slate_size: 8,
        };
        let reference = play_episode(&scenario, &rules, 100, AgentKind::Greedy, None, search, 503)
            .expect("valid reference");
        for seat in 0..3 {
            let candidate = play_episode(
                &scenario,
                &rules,
                100,
                AgentKind::Greedy,
                Some((seat, AgentKind::Greedy)),
                search,
                503,
            )
            .expect("valid candidate");
            assert_eq!(candidate, reference);
        }
    }

    #[test]
    fn comparison_counts_every_seed_and_seat() {
        let summary = compare(
            GeneratorConfig {
                schema_version: 2,
                width: 11,
                height: 9,
                players: 3,
                seed: 504,
                ..GeneratorConfig::default()
            },
            &Rules::classic_generic(),
            "classic_generic_2022",
            2,
            8,
            AgentKind::Random,
            AgentKind::Greedy,
            SearchMatchSettings {
                nodes: 2,
                baseline_nodes: 2,
                reply_nodes: 0,
                slate_size: 8,
            },
        )
        .expect("valid comparison");
        assert_eq!(summary.games, 6);
        assert_eq!(summary.records.len(), 6);
        assert_eq!(
            summary.paired_better + summary.paired_worse + summary.paired_same,
            6
        );
        assert_eq!(summary.baseline_reference_games, 2);
        assert_eq!(summary.baseline_reference_truncations, 2);
        assert_eq!(summary.candidate_truncations, 6);
        assert_eq!(summary.records[0].seed, 504);
        assert_eq!(summary.records[5].seed, 505);
    }

    #[test]
    fn reply_search_compares_against_the_same_plain_search_budget() {
        let summary = compare(
            GeneratorConfig {
                schema_version: 2,
                width: 9,
                height: 7,
                players: 2,
                seed: 505,
                ..GeneratorConfig::default()
            },
            &Rules::classic_generic(),
            "classic_generic_2022",
            2,
            100,
            AgentKind::Search,
            AgentKind::Search,
            SearchMatchSettings {
                nodes: 32,
                baseline_nodes: 32,
                reply_nodes: 8,
                slate_size: 4,
            },
        )
        .expect("reply search match");
        assert_eq!(summary.candidate, "reply-search");
        assert_eq!(summary.baseline, "turn-search");
        assert_eq!(summary.search_nodes, 32);
        assert_eq!(summary.baseline_search_nodes, 32);
        assert_eq!(summary.candidate_reply_nodes, 8);
        assert_eq!(summary.candidate_slate_size, 4);
        assert_eq!(summary.records.len(), 4);
    }

    #[test]
    fn baseline_search_budget_is_used_by_reference_games() {
        let generator = GeneratorConfig {
            schema_version: 2,
            width: 7,
            height: 5,
            players: 2,
            seed: 507,
            ..GeneratorConfig::default()
        };
        let rules = Rules::classic_generic();
        let search = SearchMatchSettings {
            nodes: 16,
            baseline_nodes: 32,
            reply_nodes: 0,
            slate_size: 1,
        };
        let summary = compare(
            generator.clone(),
            &rules,
            "classic_generic_2022",
            1,
            100,
            AgentKind::Search,
            AgentKind::Search,
            search,
        )
        .expect("unequal search budgets");
        let scenario = generator.generate().expect("valid map");
        let reference = play_episode(
            &scenario,
            &rules,
            100,
            AgentKind::Search,
            None,
            search,
            generator.seed,
        )
        .expect("baseline reference");

        assert_eq!(summary.search_nodes, 16);
        assert_eq!(summary.baseline_search_nodes, 32);
        assert_eq!(summary.records.len(), 2);
        assert_eq!(summary.records[0].baseline.winner, reference.winner);
        assert_eq!(summary.records[0].baseline.actions, reference.actions);
    }

    #[test]
    fn draw_scores_between_win_and_loss() {
        assert_eq!(outcome_score(Some(2), 2), 2);
        assert_eq!(outcome_score(None, 2), 1);
        assert_eq!(outcome_score(Some(1), 2), 0);
    }
}
