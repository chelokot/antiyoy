use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::rng::DeterministicRng;
use crate::{
    Action, ActionError, Cell, ConfigError, HexId, Object, PlayerId, Province, ProvinceId, Rules,
    Scenario, Structure, Topology, Transition, Unit,
};

const MAXIMUM_PLAYERS: u16 = u8::MAX as u16;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Component {
    owner: PlayerId,
    hexes: Vec<HexId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Game {
    rules: Rules,
    topology: Topology,
    player_count: u8,
    cells: Vec<Cell>,
    provinces: Vec<Province>,
    active_player: PlayerId,
    round: u32,
    random: DeterministicRng,
    terminal: bool,
    winner: Option<PlayerId>,
}

impl Game {
    pub fn new(rules: Rules, scenario: Scenario) -> Result<Self, ConfigError> {
        rules.validate()?;
        Self::validate_scenario(&rules, &scenario)?;

        let mut game = Self {
            rules,
            topology: scenario.topology,
            player_count: scenario.player_count,
            cells: scenario.cells.into_iter().map(Cell::from).collect(),
            provinces: Vec::new(),
            active_player: PlayerId(0),
            round: 1,
            random: DeterministicRng::new(scenario.seed),
            terminal: false,
            winner: None,
        };

        game.rebuild_provinces(true);
        game.apply_treasuries(&scenario.treasuries)?;
        game.balance_first_round();
        game.begin_turn(false);
        Ok(game)
    }

    pub const fn rules(&self) -> &Rules {
        &self.rules
    }

    pub const fn topology(&self) -> &Topology {
        &self.topology
    }

    pub const fn active_player(&self) -> PlayerId {
        self.active_player
    }

    pub const fn round(&self) -> u32 {
        self.round
    }

    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub const fn winner(&self) -> Option<PlayerId> {
        self.winner
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn cell(&self, hex: HexId) -> Option<Cell> {
        self.cells
            .get(hex.index())
            .copied()
            .filter(|_| self.topology.is_playable(hex))
    }

    pub fn provinces(&self) -> &[Province] {
        &self.provinces
    }

    pub fn province(&self, id: ProvinceId) -> Option<&Province> {
        self.provinces.get(id.index())
    }

    pub fn province_at(&self, hex: HexId) -> Option<&Province> {
        let province = self.cell(hex)?.province;
        self.province(province)
    }

    pub fn province_income(&self, id: ProvinceId) -> Option<i64> {
        self.province(id).map(|province| {
            province
                .hexes
                .iter()
                .copied()
                .map(|hex| self.hex_income(hex))
                .sum()
        })
    }

    pub fn province_upkeep(&self, id: ProvinceId) -> Option<i64> {
        self.province(id).map(|province| {
            province
                .hexes
                .iter()
                .copied()
                .map(|hex| self.hex_upkeep(hex))
                .sum()
        })
    }

    pub fn province_profit(&self, id: ProvinceId) -> Option<i64> {
        Some(self.province_income(id)? - self.province_upkeep(id)?)
    }

    pub fn hex_defense(&self, hex: HexId) -> Option<u8> {
        self.topology.is_playable(hex).then(|| self.defense(hex))
    }

    pub fn farm_price(&self, id: ProvinceId) -> Option<i64> {
        let province = self.province(id)?;
        let farms = province
            .hexes
            .iter()
            .filter(|hex| self.cells[hex.index()].object == Object::Farm)
            .count();
        let farms = i64::try_from(farms).ok()?;
        Some(self.rules.economy.farm_base_price + farms * self.rules.economy.farm_price_increment)
    }

    pub fn step(&mut self, action: Action) -> Result<Transition, ActionError> {
        if self.terminal {
            return Err(ActionError::GameFinished);
        }

        match action {
            Action::EndTurn => self.end_turn(),
            Action::Move { source, target } => self.move_unit(source, target)?,
            Action::Recruit {
                province,
                target,
                strength,
            } => self.recruit(province, target, strength)?,
            Action::Build { target, structure } => self.build(target, structure)?,
            Action::PlantTree { target } => self.plant_tree(target)?,
        }

        Ok(self.transition())
    }

    pub fn legal_actions(&self, output: &mut Vec<Action>) {
        output.clear();
        if self.terminal {
            return;
        }
        output.push(Action::EndTurn);

        let mut distances = vec![usize::MAX; self.cells.len()];
        let mut queue = VecDeque::with_capacity(self.cells.len());

        for province in self
            .provinces
            .iter()
            .filter(|province| province.owner == self.active_player)
        {
            for target in self.topology.playable_hexes().iter().copied() {
                if !self.is_recruitment_zone_target(province.id, target) {
                    continue;
                }
                for strength in 1..=self.rules.combat.maximum_unit_strength {
                    let cost = self.rules.economy.unit_price_per_level * i64::from(strength);
                    if province.money < cost {
                        break;
                    }
                    let action = Action::Recruit {
                        province: province.capital,
                        target,
                        strength,
                    };
                    if self
                        .validate_destination(strength, province.id, target)
                        .is_ok()
                    {
                        output.push(action);
                    }
                }
            }

            for target in province.hexes.iter().copied() {
                for structure in [Structure::Farm, Structure::Tower, Structure::StrongTower] {
                    if self.can_build(target, structure).is_ok() {
                        output.push(Action::Build { target, structure });
                    }
                }
                if self.can_plant_tree(target).is_ok() {
                    output.push(Action::PlantTree { target });
                }
            }
        }

        for source in self.topology.playable_hexes().iter().copied() {
            let cell = self.cells[source.index()];
            if cell.owner != self.active_player || !cell.unit.is_ready() {
                continue;
            }
            self.mark_movement_targets(
                source,
                usize::from(self.rules.combat.movement_range),
                &mut distances,
                &mut queue,
            );
            for target in self.topology.playable_hexes().iter().copied() {
                if source != target
                    && distances[target.index()] != usize::MAX
                    && self
                        .validate_destination(cell.unit.strength(), cell.province, target)
                        .is_ok()
                {
                    output.push(Action::Move { source, target });
                }
            }
        }
    }

    fn validate_scenario(rules: &Rules, scenario: &Scenario) -> Result<(), ConfigError> {
        if scenario.player_count < 2 {
            return Err(ConfigError::TooFewPlayers);
        }
        if u16::from(scenario.player_count) >= MAXIMUM_PLAYERS {
            return Err(ConfigError::TooManyPlayers {
                players: u16::from(scenario.player_count),
                maximum: MAXIMUM_PLAYERS - 1,
            });
        }
        if scenario.cells.len() != scenario.topology.len() {
            return Err(ConfigError::ScenarioSize {
                actual: scenario.cells.len(),
                expected: scenario.topology.len(),
            });
        }

        for (index, cell) in scenario.cells.iter().copied().enumerate() {
            let hex = HexId(u16::try_from(index).expect("validated topology index fits in u16"));
            if !cell.owner.is_neutral() && cell.owner.0 >= scenario.player_count {
                return Err(ConfigError::InvalidOwner {
                    hex: hex.0,
                    player: cell.owner.0,
                });
            }
            if !scenario.topology.is_playable(hex)
                && (!cell.owner.is_neutral()
                    || cell.object != Object::Empty
                    || cell.unit_strength != 0)
            {
                return Err(ConfigError::OccupiedInactiveHex { hex: hex.0 });
            }
            if cell.unit_strength > rules.combat.maximum_unit_strength {
                return Err(ConfigError::InvalidUnitStrength {
                    strength: cell.unit_strength,
                    maximum: rules.combat.maximum_unit_strength,
                });
            }
            if cell.object != Object::Empty && cell.unit_strength != 0 {
                return Err(ConfigError::ConflictingOccupants { hex: hex.0 });
            }
        }
        Ok(())
    }

    fn apply_treasuries(&mut self, treasuries: &[crate::Treasury]) -> Result<(), ConfigError> {
        let mut assigned = vec![false; self.provinces.len()];
        for treasury in treasuries {
            let province_id = self
                .cell(treasury.province)
                .map(|cell| cell.province)
                .filter(|province| province.is_some())
                .ok_or(ConfigError::InvalidTreasuryAnchor {
                    hex: treasury.province.0,
                })?;
            if assigned[province_id.index()] {
                return Err(ConfigError::DuplicateTreasury);
            }
            assigned[province_id.index()] = true;
            self.provinces[province_id.index()].money = treasury.money;
        }
        Ok(())
    }

    fn balance_first_round(&mut self) {
        let profits: Vec<i64> = self
            .provinces
            .iter()
            .map(|province| self.province_profit(province.id).unwrap_or_default())
            .collect();
        for (province, profit) in self.provinces.iter_mut().zip(profits) {
            if province.owner != PlayerId(0) {
                province.money -= profit;
            }
        }
    }

    fn transition(&self) -> Transition {
        Transition {
            active_player: self.active_player,
            round: self.round,
            terminal: self.terminal,
            winner: self.winner,
        }
    }

    fn require_hex(&self, hex: HexId) -> Result<Cell, ActionError> {
        self.cell(hex).ok_or(ActionError::InvalidHex(hex.0))
    }

    fn owned_province_id(&self, anchor: HexId) -> Result<ProvinceId, ActionError> {
        let province_id = self.require_hex(anchor)?.province;
        let province = self
            .province(province_id)
            .ok_or(ActionError::InvalidProvince)?;
        if province.owner != self.active_player {
            return Err(ActionError::InvalidProvince);
        }
        Ok(province_id)
    }

    fn can_move(&self, source: HexId, target: HexId) -> Result<(), ActionError> {
        let source_cell = self.require_hex(source)?;
        self.require_hex(target)?;
        if source_cell.owner != self.active_player || !source_cell.unit.is_ready() {
            return Err(ActionError::UnitNotReady);
        }
        if source == target
            || !self.is_reachable(
                source,
                target,
                usize::from(self.rules.combat.movement_range),
            )
        {
            return Err(ActionError::Unreachable);
        }
        self.validate_destination(source_cell.unit.strength(), source_cell.province, target)
    }

    fn validate_destination(
        &self,
        strength: u8,
        source_province: ProvinceId,
        target_hex: HexId,
    ) -> Result<(), ActionError> {
        let target = self.cells[target_hex.index()];
        if target.owner == self.active_player {
            if target.province != source_province || target.object.is_building() {
                return Err(ActionError::Occupied);
            }
            if target.unit.is_present()
                && target.unit.strength() + strength > self.rules.combat.maximum_unit_strength
            {
                return Err(ActionError::Occupied);
            }
            return Ok(());
        }
        if self.can_attack(strength, target_hex) {
            Ok(())
        } else {
            Err(ActionError::Defended)
        }
    }

    fn move_unit(&mut self, source: HexId, target: HexId) -> Result<(), ActionError> {
        self.can_move(source, target)?;
        let moving_unit = self.cells[source.index()].unit;
        let source_province = self.cells[source.index()].province;
        self.cells[source.index()].unit.clear();

        if self.cells[target.index()].owner == self.active_player {
            if self.cells[target.index()].object.is_tree() {
                self.provinces[source_province.index()].money += self.rules.economy.tree_cut_reward;
            }
            self.cells[target.index()].object = Object::Empty;
            let target_unit = self.cells[target.index()].unit;
            let strength = moving_unit.strength() + target_unit.strength();
            let ready =
                target_unit.is_present() && moving_unit.is_ready() && target_unit.is_ready();
            self.cells[target.index()].unit = Unit::new(strength, ready);
        } else {
            self.cells[target.index()].owner = self.active_player;
            self.cells[target.index()].object = Object::Empty;
            self.cells[target.index()].unit = Unit::new(moving_unit.strength(), false);
            self.rebuild_provinces(false);
        }
        Ok(())
    }

    fn can_recruit(
        &self,
        province_anchor: HexId,
        target: HexId,
        strength: u8,
    ) -> Result<(), ActionError> {
        if strength == 0 || strength > self.rules.combat.maximum_unit_strength {
            return Err(ActionError::InvalidStrength {
                strength,
                maximum: self.rules.combat.maximum_unit_strength,
            });
        }
        let province_id = self.owned_province_id(province_anchor)?;
        self.require_hex(target)?;
        let cost = self.rules.economy.unit_price_per_level * i64::from(strength);
        if self.provinces[province_id.index()].money < cost {
            return Err(ActionError::InsufficientFunds);
        }
        if !self.is_recruitment_zone_target(province_id, target) {
            return Err(ActionError::Unreachable);
        }
        self.validate_destination(strength, province_id, target)
    }

    fn recruit(
        &mut self,
        province_anchor: HexId,
        target: HexId,
        strength: u8,
    ) -> Result<(), ActionError> {
        self.can_recruit(province_anchor, target, strength)?;
        let province_id = self.cells[province_anchor.index()].province;
        self.provinces[province_id.index()].money -=
            self.rules.economy.unit_price_per_level * i64::from(strength);

        if self.cells[target.index()].owner == self.active_player {
            let had_object = self.cells[target.index()].object != Object::Empty;
            if self.cells[target.index()].object.is_tree() {
                self.provinces[province_id.index()].money += self.rules.economy.tree_cut_reward;
            }
            self.cells[target.index()].object = Object::Empty;
            let merged_strength = self.cells[target.index()].unit.strength() + strength;
            let ready = if self.cells[target.index()].unit.is_present() {
                self.rules.combat.recruited_merge_preserves_readiness
                    && self.cells[target.index()].unit.is_ready()
            } else {
                self.rules.combat.recruited_units_ready_on_owned_empty
                    && !had_object
                    && self.has_friendly_neighbour(target, self.active_player)
            };
            self.cells[target.index()].unit = Unit::new(merged_strength, ready);
        } else {
            self.cells[target.index()].owner = self.active_player;
            self.cells[target.index()].object = Object::Empty;
            self.cells[target.index()].unit = Unit::new(strength, false);
            self.rebuild_provinces(false);
        }
        Ok(())
    }

    fn can_build(&self, target: HexId, structure: Structure) -> Result<(), ActionError> {
        let cell = self.require_hex(target)?;
        let province_id = self.owned_province_id(target)?;
        if cell.unit.is_present() {
            return Err(ActionError::Occupied);
        }

        let price = match structure {
            Structure::Farm => {
                if !self.rules.combat.farms_enabled {
                    return Err(ActionError::Disabled);
                }
                if cell.object != Object::Empty {
                    return Err(ActionError::Occupied);
                }
                if !self.farm_is_supported(target, province_id) {
                    return Err(ActionError::UnsupportedFarm);
                }
                self.farm_price(province_id)
                    .ok_or(ActionError::InvalidProvince)?
            }
            Structure::Tower => {
                if !self.rules.combat.towers_enabled {
                    return Err(ActionError::Disabled);
                }
                if cell.object != Object::Empty {
                    return Err(ActionError::Occupied);
                }
                self.rules.economy.tower_price
            }
            Structure::StrongTower => {
                if !self.rules.combat.strong_towers_enabled {
                    return Err(ActionError::Disabled);
                }
                if !matches!(cell.object, Object::Empty | Object::Tower) {
                    return Err(ActionError::Occupied);
                }
                self.rules.economy.strong_tower_price
            }
        };

        if self.provinces[province_id.index()].money < price {
            return Err(ActionError::InsufficientFunds);
        }
        Ok(())
    }

    fn build(&mut self, target: HexId, structure: Structure) -> Result<(), ActionError> {
        self.can_build(target, structure)?;
        let province_id = self.cells[target.index()].province;
        let price = match structure {
            Structure::Farm => self
                .farm_price(province_id)
                .ok_or(ActionError::InvalidProvince)?,
            Structure::Tower => self.rules.economy.tower_price,
            Structure::StrongTower => self.rules.economy.strong_tower_price,
        };
        self.provinces[province_id.index()].money -= price;
        self.cells[target.index()].object = structure.into();
        Ok(())
    }

    fn can_plant_tree(&self, target: HexId) -> Result<(), ActionError> {
        if !self.rules.combat.tree_planting_enabled {
            return Err(ActionError::Disabled);
        }
        let cell = self.require_hex(target)?;
        let province_id = self.owned_province_id(target)?;
        if cell.object != Object::Empty || cell.unit.is_present() {
            return Err(ActionError::Occupied);
        }
        if self.provinces[province_id.index()].money < self.rules.economy.planted_tree_price {
            return Err(ActionError::InsufficientFunds);
        }
        Ok(())
    }

    fn plant_tree(&mut self, target: HexId) -> Result<(), ActionError> {
        self.can_plant_tree(target)?;
        let province_id = self.cells[target.index()].province;
        self.provinces[province_id.index()].money -= self.rules.economy.planted_tree_price;
        self.cells[target.index()].object = self.tree_for_hex(target);
        Ok(())
    }

    fn farm_is_supported(&self, target: HexId, province_id: ProvinceId) -> bool {
        self.topology
            .neighbours(target)
            .into_iter()
            .flatten()
            .copied()
            .filter(|hex| hex.is_valid())
            .any(|hex| {
                let cell = self.cells[hex.index()];
                cell.province == province_id
                    && matches!(cell.object, Object::Capital | Object::Farm)
            })
    }

    fn is_reachable(&self, source: HexId, target: HexId, maximum_distance: usize) -> bool {
        if !self.topology.is_playable(source) || !self.topology.is_playable(target) {
            return false;
        }
        let owner = self.cells[source.index()].owner;
        let source_province = self.cells[source.index()].province;
        let mut distances = vec![usize::MAX; self.cells.len()];
        let mut queue = VecDeque::with_capacity(self.cells.len());
        distances[source.index()] = 0;
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            let distance = distances[current.index()];
            if current == target {
                return true;
            }
            if distance == maximum_distance {
                continue;
            }
            for neighbour in self
                .topology
                .neighbours(current)
                .into_iter()
                .flatten()
                .copied()
                .filter(|hex| hex.is_valid())
            {
                if distances[neighbour.index()] != usize::MAX {
                    continue;
                }
                let cell = self.cells[neighbour.index()];
                if cell.owner == owner && cell.province == source_province {
                    distances[neighbour.index()] = distance + 1;
                    queue.push_back(neighbour);
                } else if neighbour == target {
                    return true;
                }
            }
        }
        false
    }

    fn is_recruitment_target(&self, province: ProvinceId, target: HexId) -> bool {
        self.cells[target.index()].province == province
            || self
                .topology
                .neighbours(target)
                .into_iter()
                .flatten()
                .copied()
                .filter(|hex| hex.is_valid())
                .any(|hex| self.cells[hex.index()].province == province)
    }

    fn is_recruitment_zone_target(&self, province: ProvinceId, target: HexId) -> bool {
        self.is_recruitment_target(province, target)
            && (self.cells[target.index()].owner == self.active_player
                || !self
                    .rules
                    .combat
                    .foreign_recruit_requires_economic_neighbour
                || self.has_friendly_economic_neighbour(target))
    }

    fn mark_movement_targets(
        &self,
        source: HexId,
        maximum_distance: usize,
        distances: &mut [usize],
        queue: &mut VecDeque<HexId>,
    ) {
        distances.fill(usize::MAX);
        queue.clear();
        let owner = self.cells[source.index()].owner;
        let source_province = self.cells[source.index()].province;
        distances[source.index()] = 0;
        queue.push_back(source);
        while let Some(current) = queue.pop_front() {
            let next_distance = distances[current.index()] + 1;
            if next_distance > maximum_distance {
                continue;
            }
            for neighbour in self
                .topology
                .neighbours(current)
                .into_iter()
                .flatten()
                .copied()
                .filter(|hex| hex.is_valid())
            {
                if distances[neighbour.index()] != usize::MAX {
                    continue;
                }
                distances[neighbour.index()] = next_distance;
                let cell = self.cells[neighbour.index()];
                if cell.owner == owner && cell.province == source_province {
                    queue.push_back(neighbour);
                }
            }
        }
    }

    fn can_attack(&self, strength: u8, target: HexId) -> bool {
        self.rules.combat.strongest_unit_ignores_defense
            && strength == self.rules.combat.maximum_unit_strength
            || strength > self.defense(target)
    }

    fn defense(&self, target_hex: HexId) -> u8 {
        let target = self.cells[target_hex.index()];
        let mut defense = Self::cell_defense(target);
        for neighbour in self
            .topology
            .neighbours(target_hex)
            .into_iter()
            .flatten()
            .copied()
            .filter(|hex| hex.is_valid())
        {
            let cell = self.cells[neighbour.index()];
            if cell.owner == target.owner {
                defense = defense.max(Self::cell_defense(cell));
            }
        }
        defense
    }

    fn cell_defense(cell: Cell) -> u8 {
        let object_defense = match cell.object {
            Object::Capital => 1,
            Object::Tower => 2,
            Object::StrongTower => 3,
            _ => 0,
        };
        object_defense.max(cell.unit.strength())
    }

    fn end_turn(&mut self) {
        for cell in &mut self.cells {
            if cell.owner == self.active_player {
                cell.unit.set_ready(false);
            }
        }
        self.update_terminal();
        if self.terminal {
            return;
        }

        let previous = self.active_player;
        self.active_player = self.next_player_with_province();
        if self.active_player.0 <= previous.0 {
            self.round += 1;
            self.expand_trees();
        }
        self.begin_turn(true);
    }

    fn next_player_with_province(&self) -> PlayerId {
        for offset in 1..=self.player_count {
            let candidate = PlayerId((self.active_player.0 + offset) % self.player_count);
            if self
                .provinces
                .iter()
                .any(|province| province.owner == candidate)
            {
                return candidate;
            }
        }
        self.active_player
    }

    fn begin_turn(&mut self, collect_income: bool) {
        self.transform_graves();
        let province_ids: Vec<ProvinceId> = self
            .provinces
            .iter()
            .filter(|province| province.owner == self.active_player)
            .map(|province| province.id)
            .collect();
        for province_id in province_ids {
            let profit = self.province_profit(province_id).unwrap_or_default();
            if collect_income {
                self.provinces[province_id.index()].money += profit;
            }
            if self.provinces[province_id.index()].money < 0 {
                self.provinces[province_id.index()].money = 0;
                let hexes = self.provinces[province_id.index()].hexes.clone();
                for hex in hexes {
                    if self.cells[hex.index()].unit.is_present() {
                        self.starve_unit(hex);
                    }
                }
            }
        }

        let isolated: Vec<HexId> = self
            .topology
            .playable_hexes()
            .iter()
            .copied()
            .filter(|hex| {
                let cell = self.cells[hex.index()];
                cell.owner == self.active_player
                    && cell.unit.is_present()
                    && !self.has_friendly_neighbour(*hex, self.active_player)
            })
            .collect();
        for hex in isolated {
            self.starve_unit(hex);
        }

        let readiness: Vec<(HexId, bool)> = self
            .topology
            .playable_hexes()
            .iter()
            .copied()
            .map(|hex| {
                let cell = self.cells[hex.index()];
                (
                    hex,
                    cell.owner == self.active_player
                        && cell.unit.is_present()
                        && self.has_friendly_neighbour(hex, self.active_player),
                )
            })
            .collect();
        for (hex, ready) in readiness {
            self.cells[hex.index()].unit.set_ready(ready);
        }
    }

    fn starve_unit(&mut self, hex: HexId) {
        self.cells[hex.index()].unit.clear();
        self.cells[hex.index()].object = Object::Grave;
    }

    fn transform_graves(&mut self) {
        let graves: Vec<HexId> = self
            .topology
            .playable_hexes()
            .iter()
            .copied()
            .filter(|hex| {
                let cell = self.cells[hex.index()];
                cell.owner == self.active_player && cell.object == Object::Grave
            })
            .collect();
        for grave in graves {
            self.cells[grave.index()].object = self.tree_for_hex(grave);
            self.cells[grave.index()].blocks_tree_spread = true;
        }
    }

    fn expand_trees(&mut self) {
        if !self.rules.vegetation.enabled {
            return;
        }
        let mut palms = Vec::new();
        let mut pines = Vec::new();
        for hex in self.topology.playable_hexes().iter().copied() {
            if self.cells[hex.index()].object != Object::Empty
                || self.cells[hex.index()].unit.is_present()
            {
                continue;
            }
            let neighbours = self.tree_neighbours(hex);
            let palm_source = neighbours.iter().copied().any(|neighbour| {
                self.cells[neighbour.index()].object == Object::Palm
                    && !self.cells[neighbour.index()].blocks_tree_spread
            });
            if self.is_near_water(hex)
                && palm_source
                && self
                    .random
                    .occurs_per_million(self.rules.vegetation.palm_spread_per_million)
            {
                palms.push(hex);
            }
            let pine_sources = neighbours
                .iter()
                .copied()
                .filter(|neighbour| self.cells[neighbour.index()].object.is_tree())
                .count();
            let expandable_pine = neighbours.iter().copied().any(|neighbour| {
                self.cells[neighbour.index()].object == Object::Pine
                    && !self.cells[neighbour.index()].blocks_tree_spread
            });
            if pine_sources >= usize::from(self.rules.vegetation.pine_minimum_neighbours)
                && expandable_pine
                && self
                    .random
                    .occurs_per_million(self.rules.vegetation.pine_spread_per_million)
            {
                pines.push(hex);
            }
        }
        for hex in palms {
            self.cells[hex.index()].object = Object::Palm;
        }
        for hex in pines {
            self.cells[hex.index()].object = Object::Pine;
        }
        for cell in &mut self.cells {
            if cell.object.is_tree() {
                cell.blocks_tree_spread = false;
            }
        }
    }

    fn tree_neighbours(&self, hex: HexId) -> Vec<HexId> {
        self.topology
            .neighbours(hex)
            .into_iter()
            .flatten()
            .copied()
            .filter(|neighbour| neighbour.is_valid())
            .collect()
    }

    fn tree_for_hex(&self, hex: HexId) -> Object {
        if self.is_near_water(hex) {
            Object::Palm
        } else {
            Object::Pine
        }
    }

    fn is_near_water(&self, hex: HexId) -> bool {
        self.topology
            .neighbours(hex)
            .is_some_and(|neighbours| neighbours.iter().any(|neighbour| !neighbour.is_valid()))
    }

    fn has_friendly_neighbour(&self, hex: HexId, owner: PlayerId) -> bool {
        self.topology
            .neighbours(hex)
            .into_iter()
            .flatten()
            .copied()
            .filter(|neighbour| neighbour.is_valid())
            .any(|neighbour| self.cells[neighbour.index()].owner == owner)
    }

    fn has_friendly_economic_neighbour(&self, hex: HexId) -> bool {
        self.topology
            .neighbours(hex)
            .into_iter()
            .flatten()
            .copied()
            .filter(|neighbour| neighbour.is_valid())
            .any(|neighbour| {
                let cell = self.cells[neighbour.index()];
                cell.owner == self.active_player
                    && matches!(cell.object, Object::Capital | Object::Farm)
            })
    }

    fn hex_income(&self, hex: HexId) -> i64 {
        match self.cells[hex.index()].object {
            Object::Pine | Object::Palm => 0,
            Object::Farm => self.rules.economy.farm_hex_income,
            _ => self.rules.economy.clear_hex_income,
        }
    }

    fn hex_upkeep(&self, hex: HexId) -> i64 {
        let cell = self.cells[hex.index()];
        if cell.unit.is_present() {
            return self.rules.economy.unit_upkeep[usize::from(cell.unit.strength())];
        }
        match cell.object {
            Object::Tower => self.rules.economy.tower_upkeep,
            Object::StrongTower => self.rules.economy.strong_tower_upkeep,
            _ => 0,
        }
    }

    fn update_terminal(&mut self) {
        let mut alive = Vec::new();
        for province in &self.provinces {
            if !alive.contains(&province.owner) {
                alive.push(province.owner);
            }
        }
        if alive.len() <= 1 {
            self.terminal = true;
            self.winner = alive.first().copied();
        }
    }

    fn rebuild_provinces(&mut self, initial: bool) {
        let old_provinces = self.provinces.clone();
        let old_membership: Vec<ProvinceId> = self.cells.iter().map(|cell| cell.province).collect();
        for cell in &mut self.cells {
            cell.province = ProvinceId::NONE;
        }

        let mut visited = vec![false; self.cells.len()];
        let mut components = Vec::new();
        let playable_hexes = self.topology.playable_hexes().to_vec();
        for start in playable_hexes {
            if visited[start.index()] || self.cells[start.index()].owner.is_neutral() {
                continue;
            }
            let owner = self.cells[start.index()].owner;
            let mut hexes = Vec::new();
            let mut queue = VecDeque::new();
            visited[start.index()] = true;
            queue.push_back(start);
            while let Some(hex) = queue.pop_front() {
                hexes.push(hex);
                for neighbour in self
                    .topology
                    .neighbours(hex)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|neighbour| neighbour.is_valid())
                {
                    if !visited[neighbour.index()] && self.cells[neighbour.index()].owner == owner {
                        visited[neighbour.index()] = true;
                        queue.push_back(neighbour);
                    }
                }
            }
            hexes.sort_unstable();
            if hexes.len() >= usize::from(self.rules.minimum_province_size) {
                components.push(Component { owner, hexes });
            } else {
                self.destroy_singleton_buildings(&hexes);
            }
        }
        components.sort_by_key(|component| component.hexes[0]);

        let mut money = vec![0_i64; components.len()];
        if initial {
            money.fill(self.rules.economy.starting_money);
        } else {
            for old_province in &old_provinces {
                if self.cells[old_province.capital.index()].owner != old_province.owner {
                    continue;
                }
                let destination = components
                    .iter()
                    .enumerate()
                    .filter(|(_, component)| component.owner == old_province.owner)
                    .filter(|(_, component)| {
                        component
                            .hexes
                            .iter()
                            .any(|hex| old_membership[hex.index()] == old_province.id)
                    })
                    .max_by_key(|(_, component)| {
                        (component.hexes.len(), std::cmp::Reverse(component.hexes[0]))
                    })
                    .map(|(index, _)| index);
                if let Some(destination) = destination {
                    money[destination] += old_province.money;
                }
            }
        }

        let mut provinces = Vec::with_capacity(components.len());
        for (index, component) in components.into_iter().enumerate() {
            let province_id = ProvinceId(u16::try_from(index).expect("province count fits in u16"));
            let capital = self.select_capital(&component, &old_provinces, &old_membership);
            for hex in &component.hexes {
                self.cells[hex.index()].province = province_id;
                if self.cells[hex.index()].object == Object::Capital && *hex != capital {
                    self.cells[hex.index()].object = Object::Empty;
                }
            }
            self.cells[capital.index()].object = Object::Capital;
            self.cells[capital.index()].unit.clear();
            provinces.push(Province {
                id: province_id,
                owner: component.owner,
                money: money[index],
                capital,
                hexes: component.hexes,
            });
        }
        self.provinces = provinces;
    }

    fn select_capital(
        &mut self,
        component: &Component,
        old_provinces: &[Province],
        old_membership: &[ProvinceId],
    ) -> HexId {
        let existing = component
            .hexes
            .iter()
            .copied()
            .filter(|hex| self.cells[hex.index()].object == Object::Capital)
            .max_by_key(|hex| {
                let old_size = old_membership
                    .get(hex.index())
                    .filter(|id| id.is_some())
                    .and_then(|id| old_provinces.get(id.index()))
                    .map_or(0, |province| province.hexes.len());
                (old_size, std::cmp::Reverse(*hex))
            });
        if let Some(existing) = existing {
            return existing;
        }

        let free: Vec<HexId> = component
            .hexes
            .iter()
            .copied()
            .filter(|hex| {
                let cell = self.cells[hex.index()];
                cell.object == Object::Empty && !cell.unit.is_present()
            })
            .collect();
        if !free.is_empty() {
            return free[self.random.index(free.len())];
        }
        let without_towers: Vec<HexId> = component
            .hexes
            .iter()
            .copied()
            .filter(|hex| {
                !matches!(
                    self.cells[hex.index()].object,
                    Object::Tower | Object::StrongTower
                )
            })
            .collect();
        if !without_towers.is_empty() {
            return without_towers[self.random.index(without_towers.len())];
        }
        component.hexes[self.random.index(component.hexes.len())]
    }

    fn destroy_singleton_buildings(&mut self, hexes: &[HexId]) {
        for hex in hexes {
            let object = self.cells[hex.index()].object;
            if object == Object::Capital {
                self.cells[hex.index()].object = self.tree_for_hex(*hex);
            } else if object.is_building() {
                self.cells[hex.index()].object = Object::Empty;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Action, ActionError, Game, HexId, InitialCell, Object, PlayerId, Rules, Scenario,
        Structure, Topology, Treasury,
    };

    fn balanced_game() -> Game {
        let topology = Topology::rectangle(5, 2).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 7);
        for hex in [0, 1, 5, 6] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        for hex in [3, 4, 8, 9] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[4].object = Object::Capital;
        scenario.treasuries = vec![
            Treasury {
                province: HexId(0),
                money: 100,
            },
            Treasury {
                province: HexId(4),
                money: 100,
            },
        ];
        Game::new(Rules::classic_generic(), scenario).expect("valid game")
    }

    #[test]
    fn farm_price_and_income_follow_classic_economy() {
        let mut game = balanced_game();
        let province = game.province_at(HexId(0)).expect("player province").id;
        assert_eq!(game.farm_price(province), Some(12));
        assert_eq!(game.province_income(province), Some(4));

        game.step(Action::Build {
            target: HexId(1),
            structure: Structure::Farm,
        })
        .expect("legal farm");

        let province = game.province_at(HexId(0)).expect("player province");
        assert_eq!(province.money, 88);
        assert_eq!(game.farm_price(province.id), Some(14));
        assert_eq!(game.province_income(province.id), Some(8));
    }

    #[test]
    fn capture_merges_province_treasuries_after_purchase() {
        let topology = Topology::rectangle(5, 1).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 11);
        for hex in [0, 1, 3, 4] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[3].object = Object::Capital;
        scenario.treasuries = vec![
            Treasury {
                province: HexId(0),
                money: 70,
            },
            Treasury {
                province: HexId(3),
                money: 40,
            },
        ];
        let mut game = Game::new(Rules::classic_generic(), scenario).expect("valid game");

        game.step(Action::Recruit {
            province: HexId(0),
            target: HexId(2),
            strength: 1,
        })
        .expect("legal capture");

        let player_provinces: Vec<_> = game
            .provinces()
            .iter()
            .filter(|province| province.owner == PlayerId(0))
            .collect();
        assert_eq!(player_provinces.len(), 1);
        assert_eq!(player_provinces[0].money, 100);
        assert_eq!(player_provinces[0].hexes.len(), 5);
    }

    #[test]
    fn noncapital_capture_splits_money_into_largest_fragment() {
        let mut game = split_fixture();
        let enemy_money = game.province_at(HexId(0)).expect("enemy province").money;

        game.step(Action::Recruit {
            province: HexId(5),
            target: HexId(2),
            strength: 1,
        })
        .expect("legal center capture");

        let enemy: Vec<_> = game
            .provinces()
            .iter()
            .filter(|province| province.owner == PlayerId(1))
            .collect();
        assert_eq!(enemy.len(), 2);
        assert_eq!(
            enemy.iter().map(|province| province.money).sum::<i64>(),
            enemy_money
        );
        assert!(enemy.iter().any(|province| province.money == 0));
        assert!(enemy.iter().all(|province| province.hexes.len() == 2));
    }

    #[test]
    fn captured_capital_destroys_legacy_treasury() {
        let mut game = split_fixture();
        assert!(game.province_at(HexId(0)).expect("enemy province").money > 0);

        game.step(Action::Recruit {
            province: HexId(5),
            target: HexId(0),
            strength: 4,
        })
        .expect("knight captures capital");

        let enemy = game
            .province_at(HexId(1))
            .expect("surviving enemy province");
        assert_eq!(enemy.money, 0);
        assert_ne!(enemy.capital, HexId(0));
    }

    #[test]
    fn bankruptcy_turns_every_unit_into_a_grave() {
        let topology = Topology::rectangle(4, 2).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 19);
        for hex in [0, 1] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        for hex in [4, 5] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[4].object = Object::Capital;
        scenario.cells[5].unit_strength = 4;
        scenario.treasuries.push(Treasury {
            province: HexId(4),
            money: 10,
        });
        let mut game = Game::new(Rules::classic_generic(), scenario).expect("valid game");

        game.step(Action::EndTurn).expect("player zero ends turn");
        assert!(game.cell(HexId(5)).expect("unit hex").unit().is_present());
        game.step(Action::EndTurn).expect("player one ends turn");
        game.step(Action::EndTurn)
            .expect("player zero ends second turn");

        let starved = game.cell(HexId(5)).expect("starved hex");
        assert!(!starved.unit().is_present());
        assert_eq!(starved.object(), Object::Grave);
        assert_eq!(game.province_at(HexId(4)).expect("province").money, 0);
    }

    #[test]
    fn illegal_attack_does_not_mutate_state() {
        let topology = Topology::rectangle(3, 2).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 23);
        for hex in [0, 1, 2] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        for hex in [3, 4, 5] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[1].object = Object::Tower;
        scenario.cells[3].object = Object::Capital;
        scenario.treasuries.push(Treasury {
            province: HexId(3),
            money: 100,
        });
        let mut game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let before = game.clone();

        let result = game.step(Action::Recruit {
            province: HexId(3),
            target: HexId(0),
            strength: 2,
        });
        assert_eq!(result, Err(ActionError::Defended));
        assert_eq!(game, before);

        game.step(Action::Recruit {
            province: HexId(3),
            target: HexId(0),
            strength: 4,
        })
        .expect("generic knight bypasses defense");
    }

    #[test]
    fn duel_recruits_are_not_ready_on_owned_empty_hexes() {
        let scenario = recruitment_fixture(false);
        let mut default_game =
            Game::new(Rules::online_default_v1(), scenario.clone()).expect("valid game");
        default_game
            .step(Action::Recruit {
                province: HexId(0),
                target: HexId(1),
                strength: 1,
            })
            .expect("default recruit");
        assert!(
            default_game
                .cell(HexId(1))
                .expect("target")
                .unit()
                .is_ready()
        );

        let mut duel_game = Game::new(Rules::online_duel_v1(), scenario).expect("valid game");
        duel_game
            .step(Action::Recruit {
                province: HexId(0),
                target: HexId(1),
                strength: 1,
            })
            .expect("duel recruit");
        assert!(!duel_game.cell(HexId(1)).expect("target").unit().is_ready());
    }

    #[test]
    fn duel_foreign_recruit_requires_adjacent_capital_or_farm() {
        let scenario = recruitment_fixture(false);
        let mut default_game =
            Game::new(Rules::online_default_v1(), scenario.clone()).expect("valid game");
        default_game
            .step(Action::Recruit {
                province: HexId(0),
                target: HexId(2),
                strength: 1,
            })
            .expect("default permits boundary recruit");

        let mut duel_game = Game::new(Rules::online_duel_v1(), scenario).expect("valid duel game");
        assert_eq!(
            duel_game.step(Action::Recruit {
                province: HexId(0),
                target: HexId(2),
                strength: 1,
            }),
            Err(ActionError::Unreachable)
        );

        let supported = recruitment_fixture(true);
        let mut supported_game =
            Game::new(Rules::online_duel_v1(), supported).expect("valid supported duel");
        supported_game
            .step(Action::Recruit {
                province: HexId(0),
                target: HexId(2),
                strength: 1,
            })
            .expect("economic building supports foreign recruit");
    }

    #[test]
    fn moving_unit_merge_keeps_readiness_only_when_both_units_are_ready() {
        let mut scenario = recruitment_fixture(false);
        scenario.cells[0].object = Object::Empty;
        scenario.cells[0].unit_strength = 1;
        scenario.cells[1].unit_strength = 1;
        scenario.cells[5].object = Object::Capital;
        let mut game = Game::new(Rules::classic_generic(), scenario).expect("valid game");

        game.step(Action::Move {
            source: HexId(0),
            target: HexId(1),
        })
        .expect("legal merge");

        let merged = game.cell(HexId(1)).expect("merged target").unit();
        assert_eq!(merged.strength(), 2);
        assert!(merged.is_ready());
    }

    fn recruitment_fixture(with_farm: bool) -> Scenario {
        let topology = Topology::rectangle(5, 2).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 101);
        for hex in [0, 1, 5, 6] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        for hex in [3, 4, 8, 9] {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[4].object = Object::Capital;
        if with_farm {
            scenario.cells[1].object = Object::Farm;
        }
        scenario.treasuries = vec![
            Treasury {
                province: HexId(0),
                money: 100,
            },
            Treasury {
                province: HexId(4),
                money: 100,
            },
        ];
        scenario
    }

    #[test]
    fn optimized_legal_actions_match_reference_trajectory() {
        let scenario = Scenario::symmetric_duel(11, 9, 73).expect("valid duel");
        let mut game = Game::new(Rules::classic_generic(), scenario).expect("valid game");
        let mut optimized = Vec::new();
        for step in 0..200 {
            game.legal_actions(&mut optimized);
            let reference = legal_actions_reference(&game);
            assert_eq!(optimized, reference, "legal action mismatch at step {step}");
            if game.is_terminal() {
                break;
            }
            let selected = (step * 17 + 3) % optimized.len();
            game.step(optimized[selected])
                .expect("listed action is legal");
        }
    }

    #[test]
    fn duel_legal_actions_match_validation() {
        let scenario = Scenario::symmetric_duel(11, 9, 83).expect("valid duel");
        let mut game = Game::new(Rules::online_duel_v1(), scenario).expect("valid game");
        let mut optimized = Vec::new();
        for step in 0..200 {
            game.legal_actions(&mut optimized);
            let reference = legal_actions_reference(&game);
            assert_eq!(optimized, reference, "legal action mismatch at step {step}");
            if game.is_terminal() {
                break;
            }
            let selected = (step * 19 + 5) % optimized.len();
            game.step(optimized[selected])
                .expect("listed action is legal");
        }
    }

    fn legal_actions_reference(game: &Game) -> Vec<Action> {
        let mut actions = Vec::new();
        if game.terminal {
            return actions;
        }
        actions.push(Action::EndTurn);
        for province in game
            .provinces
            .iter()
            .filter(|province| province.owner == game.active_player)
        {
            for target in game.topology.playable_hexes().iter().copied() {
                for strength in 1..=game.rules.combat.maximum_unit_strength {
                    let action = Action::Recruit {
                        province: province.capital,
                        target,
                        strength,
                    };
                    if game.can_recruit(province.capital, target, strength).is_ok() {
                        actions.push(action);
                    }
                }
            }
            for target in province.hexes.iter().copied() {
                for structure in [Structure::Farm, Structure::Tower, Structure::StrongTower] {
                    if game.can_build(target, structure).is_ok() {
                        actions.push(Action::Build { target, structure });
                    }
                }
                if game.can_plant_tree(target).is_ok() {
                    actions.push(Action::PlantTree { target });
                }
            }
        }
        for source in game.topology.playable_hexes().iter().copied() {
            let cell = game.cells[source.index()];
            if cell.owner != game.active_player || !cell.unit.is_ready() {
                continue;
            }
            for target in game.topology.playable_hexes().iter().copied() {
                if game.can_move(source, target).is_ok() {
                    actions.push(Action::Move { source, target });
                }
            }
        }
        actions
    }

    fn split_fixture() -> Game {
        let topology = Topology::rectangle(5, 2).expect("valid topology");
        let mut scenario = Scenario::empty(topology, 2, 13);
        for hex in 0..5 {
            scenario.cells[hex] = InitialCell::owned(PlayerId(1));
        }
        for hex in 5..10 {
            scenario.cells[hex] = InitialCell::owned(PlayerId(0));
        }
        scenario.cells[0].object = Object::Capital;
        scenario.cells[5].object = Object::Capital;
        scenario.treasuries = vec![
            Treasury {
                province: HexId(0),
                money: 50,
            },
            Treasury {
                province: HexId(5),
                money: 100,
            },
        ];
        Game::new(Rules::classic_generic(), scenario).expect("valid game")
    }
}
