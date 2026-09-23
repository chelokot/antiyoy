use std::time::Instant;

use antiyoy_agents::{Agent, GreedyAgent};
use antiyoy_core::{Game, GeneratorConfig, PlayerId, Rules, Scenario, adjudicate};
use anyhow::{Result, ensure};
use serde::Serialize;

use crate::{RlMapKind, SeatAuditArgs, print_value};

#[derive(Debug, Serialize)]
struct SeatAuditSummary {
    generator: GeneratorConfig,
    rules: &'static str,
    maps: u32,
    rotations_per_map: u8,
    games: u32,
    action_limit: u32,
    wins_by_seat: Vec<u32>,
    wins_by_original_start: Vec<u32>,
    draws: u32,
    truncations: u32,
    actions: u64,
    elapsed_seconds: f64,
}

pub(super) fn run(arguments: &SeatAuditArgs) -> Result<()> {
    ensure!(
        arguments.map.map == RlMapKind::Procedural,
        "seat audit requires a procedural map"
    );
    let result = audit(
        arguments.map.generator_config(arguments.seed),
        &arguments.rules.rules(),
        arguments.rules.name(),
        arguments.maps,
        arguments.action_limit,
        arguments.rotate_starts,
    )?;
    print_value(&result, arguments.json)
}

fn audit(
    generator: GeneratorConfig,
    rules: &Rules,
    rules_name: &'static str,
    maps: u32,
    action_limit: u32,
    rotate_starts: bool,
) -> Result<SeatAuditSummary> {
    ensure!(
        maps > 0 && action_limit > 0,
        "maps and action limit must be positive"
    );
    let players = generator.players;
    let rotations = if rotate_starts { players } else { 1 };
    let games = maps
        .checked_mul(u32::from(rotations))
        .ok_or_else(|| anyhow::anyhow!("map count and player count exceed the audit limit"))?;
    let mut wins_by_seat = vec![0; usize::from(players)];
    let mut wins_by_original_start = vec![0; usize::from(players)];
    let mut draws = 0;
    let mut truncations = 0;
    let mut actions = 0_u64;
    let started = Instant::now();
    for map_index in 0..maps {
        let mut map_config = generator.clone();
        map_config.seed = generator.seed.wrapping_add(u64::from(map_index));
        let scenario = map_config.generate()?;
        for offset in 0..rotations {
            let rotated = rotate_owners(&scenario, offset);
            let mut game = Game::new(rules.clone(), rotated)?;
            let mut agent = GreedyAgent::new("greedy");
            let mut legal_actions = Vec::new();
            let mut terminal_result = None;
            for _ in 0..action_limit {
                game.legal_actions(&mut legal_actions);
                let action = agent.select_action(&game, &legal_actions);
                let transition = game.step(action)?;
                actions += 1;
                if transition.terminal {
                    terminal_result = Some(transition.winner);
                    break;
                }
            }
            let winner = if let Some(winner) = terminal_result {
                winner
            } else {
                truncations += 1;
                adjudicate(&game)
            };
            if let Some(winner) = winner {
                wins_by_seat[winner.index()] += 1;
                let original = (u16::from(winner.0) + u16::from(players) - u16::from(offset))
                    % u16::from(players);
                wins_by_original_start[usize::from(original)] += 1;
            } else {
                draws += 1;
            }
        }
    }
    Ok(SeatAuditSummary {
        generator,
        rules: rules_name,
        maps,
        rotations_per_map: rotations,
        games,
        action_limit,
        wins_by_seat,
        wins_by_original_start,
        draws,
        truncations,
        actions,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn rotate_owners(scenario: &Scenario, offset: u8) -> Scenario {
    let mut rotated = scenario.clone();
    for cell in &mut rotated.cells {
        if !cell.owner.is_neutral() {
            let owner =
                (u16::from(cell.owner.0) + u16::from(offset)) % u16::from(scenario.player_count);
            cell.owner = PlayerId(u8::try_from(owner).expect("owner is less than player count"));
        }
    }
    rotated
}

#[cfg(test)]
mod tests {
    use antiyoy_core::{Game, GeneratorConfig, InitialCell, PlayerId, Rules, Scenario, Topology};

    use super::{audit, rotate_owners};

    #[test]
    fn rotating_start_owners_preserves_map_and_treasury() {
        let scenario = GeneratorConfig {
            width: 11,
            height: 9,
            players: 3,
            seed: 501,
            ..GeneratorConfig::default()
        }
        .generate()
        .expect("valid map");
        let rotated = rotate_owners(&scenario, 1);
        assert_eq!(rotated.topology, scenario.topology);
        assert_eq!(rotated.treasuries, scenario.treasuries);
        for (original, selected) in scenario.cells.iter().zip(&rotated.cells) {
            assert_eq!(original.object, selected.object);
            assert_eq!(original.unit_strength, selected.unit_strength);
            let expected = if original.owner.is_neutral() {
                original.owner
            } else {
                PlayerId((original.owner.0 + 1) % 3)
            };
            assert_eq!(selected.owner, expected);
        }
        let game = Game::new(Rules::classic_generic(), rotated).expect("valid rotation");
        assert_eq!(game.provinces().len(), 3);
    }

    #[test]
    fn rotating_large_player_ids_does_not_wrap_before_modulo() {
        let topology = Topology::rectangle(1, 1).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 250, 1);
        scenario.cells[0] = InitialCell::owned(PlayerId(249));
        let rotated = rotate_owners(&scenario, 249);
        assert_eq!(rotated.cells[0].owner, PlayerId(248));
    }

    #[test]
    fn rotated_audit_counts_every_map_and_seat_once() {
        let config = GeneratorConfig {
            width: 11,
            height: 9,
            players: 3,
            seed: 502,
            ..GeneratorConfig::default()
        };
        let summary = audit(
            config,
            &Rules::classic_generic(),
            "classic_generic_2022",
            2,
            8,
            true,
        )
        .expect("valid audit");
        assert_eq!(summary.games, 6);
        assert_eq!(summary.rotations_per_map, 3);
        assert_eq!(summary.wins_by_seat.iter().sum::<u32>() + summary.draws, 6);
        assert_eq!(
            summary.wins_by_original_start.iter().sum::<u32>() + summary.draws,
            6
        );
        assert_eq!(summary.truncations, 6);
    }
}
