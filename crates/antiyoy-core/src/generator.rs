use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::rng::DeterministicRng;
use crate::{ConfigError, HexId, InitialCell, Object, PlayerId, Scenario, Topology, Treasury};

pub const GENERATOR_SCHEMA_VERSION: u16 = 1;
const MAXIMUM_PLACEMENT_ATTEMPTS: usize = 64;
const PER_MILLION: u64 = 1_000_000;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GeneratorConfig {
    pub schema_version: u16,
    pub width: u16,
    pub height: u16,
    pub players: u8,
    pub seed: u64,
    pub land_density_per_million: u32,
    pub starting_province_size: u16,
    pub starting_money: i64,
    pub tree_density_per_million: u32,
    pub neutral_tower_density_per_million: u32,
    pub neutral_capital_density_per_million: u32,
    pub grave_density_per_million: u32,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum GenerationError {
    #[error("generator schema {actual} is unsupported, expected {expected}")]
    UnsupportedSchema { actual: u16, expected: u16 },
    #[error("map configuration failed: {0}")]
    Config(#[from] ConfigError),
    #[error("at least two players are required")]
    TooFewPlayers,
    #[error("starting province size must be at least two")]
    StartingProvinceTooSmall,
    #[error("generator probability cannot exceed 1,000,000")]
    InvalidProbability,
    #[error("neutral object densities sum to more than 1,000,000")]
    DensityOverflow,
    #[error("requested land has {actual} hexes but starts require {required}")]
    InsufficientLand { actual: usize, required: usize },
    #[error("could not place balanced starting provinces after deterministic retries")]
    PlacementFailed,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            schema_version: GENERATOR_SCHEMA_VERSION,
            width: 31,
            height: 21,
            players: 2,
            seed: 1,
            land_density_per_million: 650_000,
            starting_province_size: 5,
            starting_money: 10,
            tree_density_per_million: 150_000,
            neutral_tower_density_per_million: 20_000,
            neutral_capital_density_per_million: 10_000,
            grave_density_per_million: 15_000,
        }
    }
}

impl GeneratorConfig {
    pub fn generate(&self) -> Result<Scenario, GenerationError> {
        self.validate()?;
        let full = Topology::rectangle(self.width, self.height)?;
        let target_land = usize::try_from(
            (u64::from(self.width)
                * u64::from(self.height)
                * u64::from(self.land_density_per_million))
            .div_ceil(PER_MILLION),
        )
        .map_err(|_| GenerationError::PlacementFailed)?;
        let required = usize::from(self.players) * usize::from(self.starting_province_size);
        if target_land < required {
            return Err(GenerationError::InsufficientLand {
                actual: target_land,
                required,
            });
        }
        let mut random = DeterministicRng::new(self.seed);
        let playable = grow_land(&full, target_land, &mut random)?;
        let topology = Topology::masked_rectangle(self.width, self.height, playable)?;
        let starts = place_starts(
            &topology,
            self.players,
            usize::from(self.starting_province_size),
            &mut random,
        )?;
        let mut scenario = Scenario::empty(topology.clone(), self.players, self.seed);
        for (owner, cluster) in (0..self.players).map(PlayerId).zip(&starts) {
            for hex in cluster.iter().copied() {
                scenario.cells[hex.index()] = InitialCell::owned(owner);
            }
            let capital = cluster
                .first()
                .copied()
                .ok_or(GenerationError::PlacementFailed)?;
            scenario.cells[capital.index()].object = Object::Capital;
            scenario.treasuries.push(Treasury {
                province: capital,
                money: self.starting_money,
            });
        }
        populate_neutral_objects(self, &topology, &mut scenario.cells, &mut random);
        Ok(scenario)
    }

    fn validate(&self) -> Result<(), GenerationError> {
        if self.schema_version != GENERATOR_SCHEMA_VERSION {
            return Err(GenerationError::UnsupportedSchema {
                actual: self.schema_version,
                expected: GENERATOR_SCHEMA_VERSION,
            });
        }
        if self.players < 2 {
            return Err(GenerationError::TooFewPlayers);
        }
        if self.starting_province_size < 2 {
            return Err(GenerationError::StartingProvinceTooSmall);
        }
        let densities = [
            self.land_density_per_million,
            self.tree_density_per_million,
            self.neutral_tower_density_per_million,
            self.neutral_capital_density_per_million,
            self.grave_density_per_million,
        ];
        if densities.iter().any(|density| *density > 1_000_000) {
            return Err(GenerationError::InvalidProbability);
        }
        let object_density = u64::from(self.tree_density_per_million)
            + u64::from(self.neutral_tower_density_per_million)
            + u64::from(self.neutral_capital_density_per_million)
            + u64::from(self.grave_density_per_million);
        if object_density > PER_MILLION {
            return Err(GenerationError::DensityOverflow);
        }
        Ok(())
    }
}

fn grow_land(
    topology: &Topology,
    target: usize,
    random: &mut DeterministicRng,
) -> Result<Vec<bool>, GenerationError> {
    let mut playable = vec![false; topology.len()];
    if target == 0 {
        return Ok(playable);
    }
    let first = topology
        .playable_hexes()
        .get(random.index(topology.playable_hexes().len()))
        .copied()
        .ok_or(GenerationError::PlacementFailed)?;
    playable[first.index()] = true;
    let mut frontier = Vec::new();
    let mut queued = vec![false; topology.len()];
    add_frontier(topology, first, &playable, &mut queued, &mut frontier);
    let mut generated = 1;
    while generated < target {
        if frontier.is_empty() {
            return Err(GenerationError::PlacementFailed);
        }
        let selected = random.index(frontier.len());
        let hex = frontier.swap_remove(selected);
        queued[hex.index()] = false;
        if playable[hex.index()] {
            continue;
        }
        playable[hex.index()] = true;
        generated += 1;
        add_frontier(topology, hex, &playable, &mut queued, &mut frontier);
    }
    Ok(playable)
}

fn add_frontier(
    topology: &Topology,
    source: HexId,
    playable: &[bool],
    queued: &mut [bool],
    frontier: &mut Vec<HexId>,
) {
    for neighbour in topology
        .neighbours(source)
        .into_iter()
        .flatten()
        .copied()
        .filter(|neighbour| neighbour.is_valid())
    {
        if !playable[neighbour.index()] && !queued[neighbour.index()] {
            queued[neighbour.index()] = true;
            frontier.push(neighbour);
        }
    }
}

fn place_starts(
    topology: &Topology,
    players: u8,
    province_size: usize,
    random: &mut DeterministicRng,
) -> Result<Vec<Vec<HexId>>, GenerationError> {
    for _ in 0..MAXIMUM_PLACEMENT_ATTEMPTS {
        let first = topology
            .playable_hexes()
            .get(random.index(topology.playable_hexes().len()))
            .copied()
            .ok_or(GenerationError::PlacementFailed)?;
        let seeds = farthest_seeds(topology, players, first)?;
        let regions = voronoi_regions(topology, &seeds)?;
        if regions.iter().any(|region| region.len() < province_size) {
            continue;
        }
        let starts = regions
            .iter()
            .zip(seeds)
            .map(|(region, seed)| connected_prefix(topology, region, seed, province_size))
            .collect::<Option<Vec<_>>>();
        if let Some(starts) = starts {
            return Ok(starts);
        }
    }
    Err(GenerationError::PlacementFailed)
}

fn farthest_seeds(
    topology: &Topology,
    players: u8,
    first: HexId,
) -> Result<Vec<HexId>, GenerationError> {
    let mut seeds = Vec::with_capacity(usize::from(players));
    seeds.push(first);
    while seeds.len() < usize::from(players) {
        let distances = distances_from(topology, &seeds);
        let next = topology
            .playable_hexes()
            .iter()
            .copied()
            .filter(|candidate| !seeds.contains(candidate))
            .max_by_key(|candidate| (distances[candidate.index()], std::cmp::Reverse(*candidate)))
            .ok_or(GenerationError::PlacementFailed)?;
        seeds.push(next);
    }
    Ok(seeds)
}

fn distances_from(topology: &Topology, sources: &[HexId]) -> Vec<u16> {
    let mut distances = vec![u16::MAX; topology.len()];
    let mut queue = VecDeque::new();
    for source in sources.iter().copied() {
        distances[source.index()] = 0;
        queue.push_back(source);
    }
    while let Some(hex) = queue.pop_front() {
        let next_distance = distances[hex.index()].saturating_add(1);
        for neighbour in topology
            .neighbours(hex)
            .into_iter()
            .flatten()
            .copied()
            .filter(|neighbour| neighbour.is_valid())
        {
            if distances[neighbour.index()] == u16::MAX {
                distances[neighbour.index()] = next_distance;
                queue.push_back(neighbour);
            }
        }
    }
    distances
}

fn voronoi_regions(
    topology: &Topology,
    seeds: &[HexId],
) -> Result<Vec<Vec<HexId>>, GenerationError> {
    let mut owners = vec![None; topology.len()];
    let mut queue = VecDeque::new();
    for (owner, seed) in seeds.iter().copied().enumerate() {
        owners[seed.index()] = Some(owner);
        queue.push_back(seed);
    }
    while let Some(hex) = queue.pop_front() {
        let owner = owners[hex.index()].ok_or(GenerationError::PlacementFailed)?;
        for neighbour in topology
            .neighbours(hex)
            .into_iter()
            .flatten()
            .copied()
            .filter(|neighbour| neighbour.is_valid())
        {
            if owners[neighbour.index()].is_none() {
                owners[neighbour.index()] = Some(owner);
                queue.push_back(neighbour);
            }
        }
    }
    let mut regions = vec![Vec::new(); seeds.len()];
    for hex in topology.playable_hexes().iter().copied() {
        let owner = owners[hex.index()].ok_or(GenerationError::PlacementFailed)?;
        regions[owner].push(hex);
    }
    Ok(regions)
}

fn connected_prefix(
    topology: &Topology,
    region: &[HexId],
    seed: HexId,
    length: usize,
) -> Option<Vec<HexId>> {
    let mut included = vec![false; topology.len()];
    for hex in region.iter().copied() {
        included[hex.index()] = true;
    }
    let mut selected = Vec::with_capacity(length);
    let mut visited = vec![false; topology.len()];
    let mut queue = VecDeque::from([seed]);
    visited[seed.index()] = true;
    while let Some(hex) = queue.pop_front() {
        selected.push(hex);
        if selected.len() == length {
            break;
        }
        for neighbour in topology
            .neighbours(hex)
            .into_iter()
            .flatten()
            .copied()
            .filter(|neighbour| neighbour.is_valid())
        {
            if included[neighbour.index()] && !visited[neighbour.index()] {
                visited[neighbour.index()] = true;
                queue.push_back(neighbour);
            }
        }
    }
    (selected.len() == length).then_some(selected)
}

fn populate_neutral_objects(
    config: &GeneratorConfig,
    topology: &Topology,
    cells: &mut [InitialCell],
    random: &mut DeterministicRng,
) {
    let tree_end = u64::from(config.tree_density_per_million);
    let tower_end = tree_end + u64::from(config.neutral_tower_density_per_million);
    let capital_end = tower_end + u64::from(config.neutral_capital_density_per_million);
    let grave_end = capital_end + u64::from(config.grave_density_per_million);
    for hex in topology.playable_hexes().iter().copied() {
        if !cells[hex.index()].owner.is_neutral() {
            continue;
        }
        let roll = random.next_u64() % PER_MILLION;
        cells[hex.index()].object = if roll < tree_end {
            let coastal = topology
                .neighbours(hex)
                .is_some_and(|neighbours| neighbours.iter().any(|neighbour| !neighbour.is_valid()));
            if coastal { Object::Palm } else { Object::Pine }
        } else if roll < tower_end {
            Object::Tower
        } else if roll < capital_end {
            Object::Capital
        } else if roll < grave_end {
            Object::Grave
        } else {
            Object::Empty
        };
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::{Game, Object, PlayerId, Rules, RulesProfile};

    use super::{GenerationError, GeneratorConfig};

    #[test]
    fn equal_configuration_produces_equal_connected_scenarios() {
        let config = GeneratorConfig {
            width: 17,
            height: 13,
            players: 4,
            seed: 47,
            land_density_per_million: 600_000,
            ..GeneratorConfig::default()
        };
        let first = config.generate().expect("valid generation");
        let second = config.generate().expect("valid generation");
        assert_eq!(first, second);
        assert_eq!(first.topology.playable_hexes().len(), 133);

        let mut visited = vec![false; first.topology.len()];
        let origin = first.topology.playable_hexes()[0];
        let mut queue = VecDeque::from([origin]);
        visited[origin.index()] = true;
        while let Some(hex) = queue.pop_front() {
            for neighbour in first
                .topology
                .neighbours(hex)
                .expect("known hex")
                .iter()
                .copied()
                .filter(|neighbour| neighbour.is_valid())
            {
                if !visited[neighbour.index()] {
                    visited[neighbour.index()] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        assert!(
            first
                .topology
                .playable_hexes()
                .iter()
                .all(|hex| visited[hex.index()])
        );
    }

    #[test]
    fn every_player_receives_connected_start_and_exact_treasury() {
        let config = GeneratorConfig {
            width: 19,
            height: 15,
            players: 6,
            seed: 91,
            starting_province_size: 7,
            starting_money: 23,
            ..GeneratorConfig::default()
        };
        let scenario = config.generate().expect("valid generation");
        assert_eq!(scenario.treasuries.len(), 6);
        for player in 0..config.players {
            let owner = PlayerId(player);
            let cells = scenario
                .cells
                .iter()
                .filter(|cell| cell.owner == owner)
                .count();
            assert_eq!(cells, usize::from(config.starting_province_size));
            assert_eq!(
                scenario
                    .cells
                    .iter()
                    .filter(|cell| cell.owner == owner && cell.object == Object::Capital)
                    .count(),
                1
            );
        }
        assert!(
            scenario
                .treasuries
                .iter()
                .all(|treasury| treasury.money == 23)
        );
        let game = Game::new(Rules::online_default_v1(), scenario).expect("playable scenario");
        assert_eq!(game.provinces().len(), usize::from(config.players));
        for player in 0..config.players {
            let province = game
                .provinces()
                .iter()
                .find(|province| province.owner() == PlayerId(player))
                .expect("one connected province per player");
            assert_eq!(
                province.hexes().len(),
                usize::from(config.starting_province_size)
            );
            assert_eq!(province.money(), config.starting_money);
        }
    }

    #[test]
    fn generated_scenario_is_valid_for_every_bundled_profile() {
        let scenario = GeneratorConfig::default()
            .generate()
            .expect("valid generation");
        for profile in [
            RulesProfile::ClassicGeneric,
            RulesProfile::ClassicSlay,
            RulesProfile::OnlineDefaultV1,
            RulesProfile::OnlineClassicV1,
            RulesProfile::OnlineDuelV1,
            RulesProfile::OnlineExperimentalV1,
            RulesProfile::OnlineExperimentalV2_260801,
        ] {
            Game::new(
                Rules::from_profile(profile).expect("bundled profile"),
                scenario.clone(),
            )
            .expect("profile accepts generated scenario");
        }
    }

    #[test]
    fn impossible_land_budget_is_rejected() {
        let config = GeneratorConfig {
            width: 5,
            height: 2,
            players: 4,
            starting_province_size: 3,
            land_density_per_million: 1_000_000,
            ..GeneratorConfig::default()
        };
        assert_eq!(
            config.generate(),
            Err(GenerationError::InsufficientLand {
                actual: 10,
                required: 12,
            })
        );
    }
}
