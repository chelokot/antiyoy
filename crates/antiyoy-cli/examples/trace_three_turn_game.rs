use std::collections::HashMap;

use antiyoy_agents::{
    Agent, SearchAgent, SearchConfig, position_score, search_reply, search_turn_slate,
};
use antiyoy_core::{Action, Game, GeneratorConfig, PlayerId, Rules, adjudicate};
use clap::Parser;
use serde::Serialize;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    candidate_seat: u8,
    #[arg(long, default_value_t = 2_400)]
    action_limit: u32,
    #[arg(long)]
    inspect_action: Option<u32>,
}

#[derive(Serialize)]
struct Snapshot {
    action: u32,
    round: u32,
    active_seat: u8,
    owned_cells: [usize; 2],
    units: [usize; 2],
    unit_strengths: [Vec<u8>; 2],
    province_money: [i64; 2],
    position_score_seat_one: i64,
    end_turns: [u32; 2],
    moves: [u32; 2],
    recruits: [u32; 2],
    builds: [u32; 2],
    provinces: Vec<ProvinceStatus>,
    legal_actions: Vec<Action>,
    legal_capture_moves: usize,
    legal_neutral_moves: usize,
}

#[derive(Serialize)]
struct ProvinceStatus {
    seat: u8,
    cells: usize,
    money: i64,
    income: i64,
    upkeep: i64,
}

#[derive(Serialize)]
struct RepeatedBoard {
    first_action: u32,
    last_action: u32,
    occurrences: u32,
}

#[derive(Serialize)]
struct CandidateScore {
    actions: Vec<Action>,
    root_score: i64,
    reply_score: i64,
    followup_score: i64,
    policy_reply_score: i64,
    policy_reply_followup_score: i64,
    reply_actions: Vec<Action>,
    followup_actions: Vec<Action>,
    policy_reply_actions: Vec<Action>,
}

#[derive(Serialize)]
struct InspectedAction {
    action_index: u32,
    seat: u8,
    action: Action,
}

fn followup(game: &Game, player: PlayerId, config: SearchConfig) -> (i64, Vec<Action>) {
    if game.is_terminal() {
        return (position_score(game, player), Vec::new());
    }
    let turn = search_turn_slate(
        game,
        SearchConfig {
            node_budget: 32,
            ..config
        },
        1,
    )
    .expect("valid followup search")
    .turns
    .remove(0);
    (position_score(&turn.game, player), turn.actions)
}

fn policy_reply(game: &Game, player: PlayerId, config: SearchConfig) -> (Game, Vec<Action>) {
    let mut opponent = SearchAgent::with_reply_search("policy-opponent", config, 8, 64)
        .expect("valid opponent search");
    let mut state = game.clone();
    let mut actions = Vec::new();
    while !state.is_terminal() && state.active_player() != player {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        let action = opponent.select_action(&state, &legal);
        state
            .step(action)
            .expect("selected opponent action is legal");
        actions.push(action);
        assert!(actions.len() <= config.maximum_actions_per_turn);
    }
    (state, actions)
}

fn candidate_scores(game: &Game, config: SearchConfig) -> Vec<CandidateScore> {
    let player = game.active_player();
    search_turn_slate(game, config, 8)
        .expect("valid root search")
        .turns
        .into_iter()
        .map(|turn| {
            let reply = search_reply(
                &turn,
                player,
                SearchConfig {
                    node_budget: 64,
                    ..config
                },
            );
            let (followup_score, followup_actions) = followup(&reply.game, player, config);
            let (policy_reply_game, policy_reply_actions) =
                policy_reply(&turn.game, player, config);
            let policy_reply_score = position_score(&policy_reply_game, player);
            let (policy_reply_followup_score, _) = followup(&policy_reply_game, player, config);
            CandidateScore {
                actions: turn.actions,
                root_score: turn.score,
                reply_score: reply.score,
                followup_score,
                policy_reply_score,
                policy_reply_followup_score,
                reply_actions: reply.actions,
                followup_actions,
                policy_reply_actions,
            }
        })
        .collect()
}

fn snapshot(game: &Game, action: u32, counts: &ActionCounts) -> Snapshot {
    let mut owned_cells = [0; 2];
    let mut units = [0; 2];
    let mut unit_strengths = [Vec::new(), Vec::new()];
    let mut province_money = [0; 2];
    for cell in game.cells() {
        if cell.owner().is_neutral() {
            continue;
        }
        let seat = cell.owner().index();
        owned_cells[seat] += 1;
        units[seat] += usize::from(cell.unit().is_present());
        if cell.unit().is_present() {
            unit_strengths[seat].push(cell.unit().strength());
        }
    }
    for province in game.provinces() {
        province_money[province.owner().index()] += province.money();
    }
    let provinces = game
        .provinces()
        .iter()
        .map(|province| ProvinceStatus {
            seat: province.owner().0,
            cells: province.hexes().len(),
            money: province.money(),
            income: game.province_income(province.id()).expect("live province"),
            upkeep: game.province_upkeep(province.id()).expect("live province"),
        })
        .collect();
    let mut legal_actions = Vec::new();
    game.legal_actions(&mut legal_actions);
    let legal_capture_moves = legal_actions
        .iter()
        .filter(|action| match action {
            Action::Move { target, .. } => game.cell(*target).is_some_and(|cell| {
                !cell.owner().is_neutral() && cell.owner() != game.active_player()
            }),
            _ => false,
        })
        .count();
    let legal_neutral_moves = legal_actions
        .iter()
        .filter(|action| match action {
            Action::Move { target, .. } => game
                .cell(*target)
                .is_some_and(|cell| cell.owner().is_neutral()),
            _ => false,
        })
        .count();
    Snapshot {
        action,
        round: game.round(),
        active_seat: game.active_player().0,
        owned_cells,
        units,
        unit_strengths,
        province_money,
        position_score_seat_one: position_score(game, PlayerId(1)),
        end_turns: counts.end_turns,
        moves: counts.moves,
        recruits: counts.recruits,
        builds: counts.builds,
        provinces,
        legal_actions,
        legal_capture_moves,
        legal_neutral_moves,
    }
}

fn board_key(game: &Game) -> Vec<u8> {
    game.cells()
        .iter()
        .flat_map(|cell| [cell.owner().0, cell.object() as u8, cell.unit().strength()])
        .collect()
}

fn position_key(game: &Game) -> Vec<u8> {
    let mut key = board_key(game);
    key.push(game.active_player().0);
    key
}

#[derive(Default)]
struct ActionCounts {
    end_turns: [u32; 2],
    moves: [u32; 2],
    recruits: [u32; 2],
    builds: [u32; 2],
}

impl ActionCounts {
    fn add(&mut self, seat: usize, action: Action) {
        match action {
            Action::EndTurn => self.end_turns[seat] += 1,
            Action::Move { .. } => self.moves[seat] += 1,
            Action::Recruit { .. } => self.recruits[seat] += 1,
            Action::Build { .. } | Action::PlantTree { .. } => self.builds[seat] += 1,
            Action::Diplomacy { .. } => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    assert!(args.candidate_seat < 2 && args.action_limit > 0);
    let scenario = GeneratorConfig {
        schema_version: 2,
        width: 11,
        height: 9,
        players: 2,
        seed: args.seed,
        ..GeneratorConfig::default()
    }
    .generate()?;
    let mut game = Game::new(Rules::classic_generic(), scenario)?;
    let config = SearchConfig {
        node_budget: 256,
        ..SearchConfig::default()
    };
    let baseline = SearchAgent::with_reply_search("baseline", config, 8, 64)?;
    let candidate = SearchAgent::with_three_turn_search("candidate", config, 8, 64, 32)?;
    let mut agents = if args.candidate_seat == 0 {
        [candidate, baseline]
    } else {
        [baseline, candidate]
    };
    let report = trace(&args, &mut game, &mut agents, config)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn trace(
    args: &Args,
    game: &mut Game,
    agents: &mut [SearchAgent; 2],
    config: SearchConfig,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut counts = ActionCounts::default();
    let mut snapshots = vec![snapshot(game, 0, &counts)];
    let mut boards = HashMap::<Vec<u8>, RepeatedBoard>::new();
    let mut last_owner_change = None;
    let mut last_board_change = None;
    let mut inspected_candidate_scores = Vec::new();
    let mut inspected_actions = Vec::new();
    let mut actions = 0;
    let mut prior_owners: Vec<_> = game.cells().iter().map(|cell| cell.owner()).collect();
    let mut prior_board = board_key(game);
    for action_index in 0..args.action_limit {
        if game.is_terminal() {
            break;
        }
        let key = position_key(game);
        boards
            .entry(key)
            .and_modify(|record| {
                record.last_action = action_index;
                record.occurrences += 1;
            })
            .or_insert(RepeatedBoard {
                first_action: action_index,
                last_action: action_index,
                occurrences: 1,
            });
        let seat = game.active_player().index();
        if args.inspect_action == Some(action_index) && seat == usize::from(args.candidate_seat) {
            inspected_candidate_scores = candidate_scores(game, config);
        }
        let mut legal = Vec::new();
        game.legal_actions(&mut legal);
        let action = agents[seat].select_action(game, &legal);
        if args
            .inspect_action
            .is_some_and(|inspected| action_index.abs_diff(inspected) <= 2)
        {
            inspected_actions.push(InspectedAction {
                action_index,
                seat: u8::try_from(seat).expect("two-player seat"),
                action,
            });
        }
        counts.add(seat, action);
        game.step(action)?;
        actions = action_index + 1;
        let owners: Vec<_> = game.cells().iter().map(|cell| cell.owner()).collect();
        if owners != prior_owners {
            last_owner_change = Some(actions);
            prior_owners = owners;
        }
        let board = board_key(game);
        if board != prior_board {
            last_board_change = Some(actions);
            prior_board = board;
        }
        if actions.is_multiple_of(100)
            || args
                .inspect_action
                .is_some_and(|inspected| actions.abs_diff(inspected) <= 2)
            || game.is_terminal()
        {
            snapshots.push(snapshot(game, actions, &counts));
        }
    }
    if snapshots
        .last()
        .is_some_and(|snapshot| snapshot.action != actions)
    {
        snapshots.push(snapshot(game, actions, &counts));
    }
    let mut repeated_boards: Vec<_> = boards.into_values().filter(|r| r.occurrences > 1).collect();
    repeated_boards.sort_by_key(|record| std::cmp::Reverse(record.occurrences));
    repeated_boards.truncate(10);
    Ok(serde_json::json!({
            "seed": args.seed,
            "candidate_seat": args.candidate_seat,
            "action_limit": args.action_limit,
            "inspect_action": args.inspect_action,
            "actions": actions,
            "terminal": game.is_terminal(),
            "winner": game.winner().map(|winner| winner.0),
            "adjudicated_winner": adjudicate(game).map(|winner| winner.0),
            "last_owner_change_action": last_owner_change,
            "last_board_change_action": last_board_change,
            "repeated_boards": repeated_boards,
            "inspected_candidate_scores": inspected_candidate_scores,
            "inspected_actions": inspected_actions,
            "snapshots": snapshots,
    }))
}
