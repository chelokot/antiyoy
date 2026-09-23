use std::cmp::Reverse;
use std::time::Instant;

use antiyoy_agents::{
    SearchAgent, SearchConfig, SearchReply, SearchTurnSlate, search_reply, search_turn_slate,
};
use antiyoy_core::{Action, Game, GeneratorConfig, Rules};
use antiyoy_rl::BatchObservation;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::{RlMapKind, SlateDatasetArgs, observe, print_value};

#[derive(Debug, Eq, PartialEq, Serialize)]
struct SlateRecord {
    seed: u64,
    seat: u8,
    round: u32,
    search_expansions: usize,
    selected_index: usize,
    static_scores: Vec<i64>,
    reply_scores: Vec<i64>,
    actions: Vec<Vec<Action>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    opponent_actions: Option<Vec<Vec<Action>>>,
    post_turn: BatchObservation,
    #[serde(skip_serializing_if = "Option::is_none")]
    post_reply: Option<BatchObservation>,
}

#[derive(Debug, Serialize)]
struct SlateDataset {
    schema_version: u16,
    generator: GeneratorConfig,
    rules: &'static str,
    maps: u32,
    completed_maps: u32,
    truncated_maps: u32,
    action_limit: u32,
    search_nodes: usize,
    beam_slate_size: usize,
    opponent_search_nodes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_opponent_actions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_opponent_observations: Option<bool>,
    maximum_actions_per_turn: usize,
    sample_round_modulus: u32,
    sample_round_remainder: u32,
    positions: usize,
    teacher_overrides: u32,
    first_action_changes: u32,
    actions_played: u64,
    elapsed_seconds: f64,
    records: Vec<SlateRecord>,
}

struct ScoredSlate {
    slate: SearchTurnSlate,
    replies: Vec<SearchReply>,
    selected: usize,
}

pub(super) fn run(arguments: &SlateDatasetArgs) -> Result<()> {
    ensure!(
        arguments.map.map == RlMapKind::Procedural && arguments.map.players == 2,
        "slate dataset requires a procedural two-player map"
    );
    let summary = collect(
        arguments.map.generator_config(arguments.seed),
        &arguments.rules.rules(),
        arguments.rules.name(),
        arguments,
    )?;
    print_value(&summary, arguments.json)
}

fn collect(
    generator: GeneratorConfig,
    rules: &Rules,
    rules_name: &'static str,
    arguments: &SlateDatasetArgs,
) -> Result<SlateDataset> {
    ensure!(
        generator.players == 2 && arguments.sample_round_modulus > 0,
        "slate dataset requires two players and a positive sample modulus"
    );
    ensure!(
        arguments.sample_round_remainder < arguments.sample_round_modulus,
        "sample remainder must be smaller than modulus"
    );
    let search = SearchConfig {
        node_budget: arguments.search_nodes,
        maximum_actions_per_turn: arguments.maximum_actions_per_turn,
        ..SearchConfig::default()
    };
    SearchAgent::with_reply_search(
        "slate-validation",
        search,
        arguments.beam_slate_size,
        arguments.opponent_search_nodes,
    )
    .context("invalid reply-search configuration")?;
    let started = Instant::now();
    let mut records = Vec::new();
    let mut completed_maps = 0;
    let mut truncated_maps = 0;
    let mut teacher_overrides = 0;
    let mut first_action_changes = 0;
    let mut actions_played = 0;
    for map_index in 0..arguments.maps {
        let mut map_generator = generator.clone();
        map_generator.seed = generator
            .seed
            .checked_add(u64::from(map_index))
            .context("map seed overflow")?;
        let mut game = Game::new(rules.clone(), map_generator.generate()?)?;
        let mut actions = 0;
        while !game.is_terminal() && actions < arguments.action_limit {
            let scored = score_slate(
                &game,
                search,
                arguments.beam_slate_size,
                arguments.opponent_search_nodes,
            )?;
            let chosen = &scored.slate.turns[scored.selected];
            let turn_actions = u32::try_from(chosen.actions.len())?;
            if turn_actions > arguments.action_limit - actions {
                break;
            }
            let next_game = chosen.game.clone();
            if game.round() % arguments.sample_round_modulus == arguments.sample_round_remainder {
                let record = record_slate(
                    &game,
                    map_generator.seed,
                    &scored,
                    arguments.include_opponent_actions,
                    arguments.include_opponent_observations,
                );
                teacher_overrides += u32::from(record.selected_index != 0);
                first_action_changes +=
                    u32::from(record.actions[record.selected_index][0] != record.actions[0][0]);
                records.push(record);
            }
            actions += turn_actions;
            game = next_game;
        }
        actions_played += u64::from(actions);
        completed_maps += u32::from(game.is_terminal());
        truncated_maps += u32::from(!game.is_terminal());
    }
    Ok(SlateDataset {
        schema_version: 1,
        generator,
        rules: rules_name,
        maps: arguments.maps,
        completed_maps,
        truncated_maps,
        action_limit: arguments.action_limit,
        search_nodes: arguments.search_nodes,
        beam_slate_size: arguments.beam_slate_size,
        opponent_search_nodes: arguments.opponent_search_nodes,
        include_opponent_actions: arguments.include_opponent_actions.then_some(true),
        include_opponent_observations: arguments.include_opponent_observations.then_some(true),
        maximum_actions_per_turn: arguments.maximum_actions_per_turn,
        sample_round_modulus: arguments.sample_round_modulus,
        sample_round_remainder: arguments.sample_round_remainder,
        positions: records.len(),
        teacher_overrides,
        first_action_changes,
        actions_played,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        records,
    })
}

fn score_slate(
    game: &Game,
    search: SearchConfig,
    size: usize,
    reply_nodes: usize,
) -> Result<ScoredSlate> {
    let slate = search_turn_slate(game, search, size)?;
    let reply_config = SearchConfig {
        node_budget: reply_nodes,
        ..search
    };
    let replies = slate
        .turns
        .iter()
        .map(|turn| search_reply(turn, game.active_player(), reply_config))
        .collect::<Vec<_>>();
    let selected = slate
        .turns
        .iter()
        .enumerate()
        .max_by_key(|(index, turn)| (replies[*index].score, turn.score, Reverse(*index)))
        .map(|(index, _)| index)
        .context("search did not produce a completed turn")?;
    Ok(ScoredSlate {
        slate,
        replies,
        selected,
    })
}

fn record_slate(
    game: &Game,
    seed: u64,
    scored: &ScoredSlate,
    include_opponent_actions: bool,
    include_opponent_observations: bool,
) -> SlateRecord {
    let games = scored
        .slate
        .turns
        .iter()
        .map(|turn| &turn.game)
        .collect::<Vec<_>>();
    SlateRecord {
        seed,
        seat: game.active_player().0,
        round: game.round(),
        search_expansions: scored.slate.stats.nodes,
        selected_index: scored.selected,
        static_scores: scored.slate.turns.iter().map(|turn| turn.score).collect(),
        reply_scores: scored.replies.iter().map(|reply| reply.score).collect(),
        actions: scored
            .slate
            .turns
            .iter()
            .map(|turn| turn.actions.clone())
            .collect(),
        opponent_actions: include_opponent_actions.then(|| {
            scored
                .replies
                .iter()
                .map(|reply| reply.actions.clone())
                .collect()
        }),
        post_turn: observe::observe_games(&games),
        post_reply: include_opponent_observations.then(|| {
            observe::observe_games(
                &scored
                    .replies
                    .iter()
                    .map(|reply| &reply.game)
                    .collect::<Vec<_>>(),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use antiyoy_agents::{Agent, SearchAgent, SearchConfig, position_score};
    use antiyoy_core::{Game, GeneratorConfig, Rules};

    use super::{SlateDatasetArgs, collect, score_slate};
    use crate::{RlMapArgs, RlMapKind, RulesKind};

    #[test]
    fn scored_slate_matches_reply_search_agent_turn() {
        let game = Game::new(
            Rules::classic_generic(),
            GeneratorConfig {
                schema_version: 2,
                players: 2,
                seed: 6_270_000,
                ..GeneratorConfig::default()
            }
            .generate()
            .expect("valid map"),
        )
        .expect("valid game");
        let search = SearchConfig {
            node_budget: 32,
            maximum_actions_per_turn: 8,
            ..SearchConfig::default()
        };
        let scored = score_slate(&game, search, 4, 8).expect("valid slate");
        for (turn, reply) in scored.slate.turns.iter().zip(&scored.replies) {
            let mut response = turn.game.clone();
            for action in &reply.actions {
                response.step(*action).expect("searched reply is legal");
            }
            assert_eq!(response, reply.game);
            assert_eq!(reply.score, position_score(&response, game.active_player()));
        }
        let mut agent =
            SearchAgent::with_reply_search("teacher", search, 4, 8).expect("valid agent");
        let mut replay = game.clone();
        let player = replay.active_player();
        let mut legal = Vec::new();
        for _ in 0..search.maximum_actions_per_turn {
            replay.legal_actions(&mut legal);
            let action = agent.select_action(&replay, &legal);
            replay.step(action).expect("legal action");
            if replay.is_terminal() || replay.active_player() != player {
                break;
            }
        }
        assert_eq!(replay, scored.slate.turns[scored.selected].game);
    }

    #[test]
    fn collected_slates_are_deterministic_and_aligned() {
        let generator = GeneratorConfig {
            schema_version: 2,
            players: 2,
            seed: 6_270_100,
            ..GeneratorConfig::default()
        };
        let arguments = SlateDatasetArgs {
            maps: 1,
            seed: generator.seed,
            action_limit: 128,
            map: RlMapArgs {
                map: RlMapKind::Procedural,
                generator_schema_version: 2,
                width: generator.width,
                height: generator.height,
                players: 2,
                land_density_per_million: generator.land_density_per_million,
                starting_province_size: generator.starting_province_size,
                starting_money: generator.starting_money,
                tree_density_per_million: generator.tree_density_per_million,
                neutral_tower_density_per_million: generator.neutral_tower_density_per_million,
                neutral_capital_density_per_million: generator.neutral_capital_density_per_million,
                grave_density_per_million: generator.grave_density_per_million,
            },
            search_nodes: 32,
            beam_slate_size: 4,
            opponent_search_nodes: 8,
            include_opponent_actions: true,
            include_opponent_observations: true,
            sample_round_modulus: 2,
            sample_round_remainder: 0,
            maximum_actions_per_turn: 8,
            rules: RulesKind::ClassicGeneric,
            json: true,
        };
        let first = collect(
            generator.clone(),
            &Rules::classic_generic(),
            "classic_generic_2022",
            &arguments,
        )
        .expect("valid dataset");
        let second = collect(
            generator.clone(),
            &Rules::classic_generic(),
            "classic_generic_2022",
            &arguments,
        )
        .expect("same dataset");
        assert_eq!(first.records, second.records);
        assert_eq!(first.include_opponent_actions, Some(true));
        assert!(!first.records.is_empty());
        assert!(first.records.iter().all(|record| {
            let candidates = record.static_scores.len();
            record.selected_index < candidates
                && record.reply_scores.len() == candidates
                && record.actions.len() == candidates
                && record.opponent_actions.as_ref().map(Vec::len) == Some(candidates)
                && record.post_turn.widths.len() == candidates
                && record.post_turn.cell_offsets.len() == candidates + 1
                && record.post_reply.as_ref().map(|reply| reply.widths.len()) == Some(candidates)
                && record
                    .post_reply
                    .as_ref()
                    .map(|reply| reply.cell_offsets.len())
                    == Some(candidates + 1)
        }));
        let mut without_replies = arguments;
        without_replies.include_opponent_actions = false;
        without_replies.include_opponent_observations = false;
        let plain = collect(
            generator,
            &Rules::classic_generic(),
            "classic_generic_2022",
            &without_replies,
        )
        .expect("plain dataset");
        assert!(
            plain
                .records
                .iter()
                .all(|record| record.opponent_actions.is_none())
        );
        assert!(
            plain
                .records
                .iter()
                .all(|record| record.post_reply.is_none())
        );
        let serialized = serde_json::to_value(&plain.records[0]).expect("serializable record");
        assert!(serialized.get("opponent_actions").is_none());
        assert!(serialized.get("post_reply").is_none());
        let summary = serde_json::to_value(&plain).expect("serializable summary");
        assert!(summary.get("include_opponent_actions").is_none());
        assert!(summary.get("include_opponent_observations").is_none());
    }
}
