use serde::{Deserialize, Serialize};

use crate::ConfigError;

const MAXIMUM_CONFIGURED_UNIT_STRENGTH: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RulesProfile {
    ClassicGeneric,
    ClassicSlay,
    OnlineDefaultV1,
    OnlineClassicV1,
    OnlineDuelV1,
    OnlineExperimentalV1,
    OnlineExperimentalV2_260801,
    Custom,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EconomyRules {
    pub starting_money: i64,
    pub clear_hex_income: i64,
    pub farm_hex_income: i64,
    pub unit_price_per_level: i64,
    pub unit_upkeep: [i64; 5],
    pub farm_base_price: i64,
    pub farm_price_increment: i64,
    pub tower_price: i64,
    pub strong_tower_price: i64,
    pub tower_upkeep: i64,
    pub strong_tower_upkeep: i64,
    pub planted_tree_price: i64,
    pub tree_cut_reward: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CombatRules {
    pub maximum_unit_strength: u8,
    pub movement_range: u8,
    pub strongest_unit_ignores_defense: bool,
    pub farms_enabled: bool,
    pub towers_enabled: bool,
    pub strong_towers_enabled: bool,
    pub tree_planting_enabled: bool,
    pub recruited_units_ready_on_owned_empty: bool,
    pub recruited_merge_preserves_readiness: bool,
    pub foreign_recruit_requires_economic_neighbour: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VegetationRules {
    pub enabled: bool,
    pub pine_minimum_neighbours: u8,
    pub pine_spread_per_million: u32,
    pub palm_spread_per_million: u32,
    pub target_based_spread: bool,
    pub target_spread_per_million: u32,
    pub charge_player_zero_per_spawn: bool,
    pub grave_tree_skips_next_cycle: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LifecycleRules {
    pub split_money_follows_capital_then_farms: bool,
    pub merge_capital_prefers_farm_support: bool,
    pub singleton_buildings_persist: bool,
    pub eliminate_singleton_units_after_capture: bool,
    pub skip_first_round_income: bool,
    pub income_before_grave_conversion: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rules {
    pub schema_version: u16,
    pub profile: RulesProfile,
    pub minimum_province_size: u16,
    pub economy: EconomyRules,
    pub combat: CombatRules,
    pub vegetation: VegetationRules,
    pub lifecycle: LifecycleRules,
}

impl Rules {
    pub fn classic_generic() -> Self {
        Self {
            schema_version: 4,
            profile: RulesProfile::ClassicGeneric,
            minimum_province_size: 2,
            economy: EconomyRules {
                starting_money: 10,
                clear_hex_income: 1,
                farm_hex_income: 5,
                unit_price_per_level: 10,
                unit_upkeep: [0, 2, 6, 18, 36],
                farm_base_price: 12,
                farm_price_increment: 2,
                tower_price: 15,
                strong_tower_price: 35,
                tower_upkeep: 1,
                strong_tower_upkeep: 6,
                planted_tree_price: 10,
                tree_cut_reward: 3,
            },
            combat: CombatRules {
                maximum_unit_strength: 4,
                movement_range: 4,
                strongest_unit_ignores_defense: true,
                farms_enabled: true,
                towers_enabled: true,
                strong_towers_enabled: true,
                tree_planting_enabled: true,
                recruited_units_ready_on_owned_empty: true,
                recruited_merge_preserves_readiness: true,
                foreign_recruit_requires_economic_neighbour: false,
            },
            vegetation: VegetationRules {
                enabled: true,
                pine_minimum_neighbours: 2,
                pine_spread_per_million: 200_000,
                palm_spread_per_million: 300_000,
                target_based_spread: false,
                target_spread_per_million: 0,
                charge_player_zero_per_spawn: false,
                grave_tree_skips_next_cycle: true,
            },
            lifecycle: LifecycleRules {
                split_money_follows_capital_then_farms: false,
                merge_capital_prefers_farm_support: false,
                singleton_buildings_persist: false,
                eliminate_singleton_units_after_capture: false,
                skip_first_round_income: false,
                income_before_grave_conversion: false,
            },
        }
    }

    pub fn classic_slay() -> Self {
        let mut rules = Self::classic_generic();
        rules.profile = RulesProfile::ClassicSlay;
        rules.economy.farm_hex_income = 1;
        rules.economy.unit_upkeep[4] = 54;
        rules.economy.tree_cut_reward = 0;
        rules.economy.tower_upkeep = 0;
        rules.combat.strongest_unit_ignores_defense = false;
        rules.combat.farms_enabled = false;
        rules.combat.strong_towers_enabled = false;
        rules.combat.tree_planting_enabled = false;
        rules.vegetation.pine_spread_per_million = 800_000;
        rules.vegetation.palm_spread_per_million = 1_000_000;
        rules
    }

    pub fn online_default_v1() -> Self {
        let mut rules = Self::classic_generic();
        rules.profile = RulesProfile::OnlineDefaultV1;
        rules.lifecycle = LifecycleRules {
            split_money_follows_capital_then_farms: true,
            merge_capital_prefers_farm_support: true,
            singleton_buildings_persist: true,
            eliminate_singleton_units_after_capture: true,
            skip_first_round_income: true,
            income_before_grave_conversion: true,
        };
        rules.vegetation.target_based_spread = true;
        rules.vegetation.target_spread_per_million = 330_000;
        rules.vegetation.charge_player_zero_per_spawn = true;
        rules.vegetation.grave_tree_skips_next_cycle = false;
        rules
    }

    pub fn online_classic_v1() -> Self {
        let mut rules = Self::classic_slay();
        rules.profile = RulesProfile::OnlineClassicV1;
        rules
    }

    pub fn online_duel_v1() -> Self {
        let mut rules = Self::online_default_v1();
        rules.profile = RulesProfile::OnlineDuelV1;
        rules.combat.recruited_units_ready_on_owned_empty = false;
        rules.combat.recruited_merge_preserves_readiness = false;
        rules.combat.foreign_recruit_requires_economic_neighbour = true;
        rules
    }

    pub fn online_experimental_v1() -> Self {
        let mut rules = Self::online_duel_v1();
        rules.profile = RulesProfile::OnlineExperimentalV1;
        rules
    }

    pub fn online_experimental_v2_260801() -> Self {
        let mut rules = Self::online_experimental_v1();
        rules.profile = RulesProfile::OnlineExperimentalV2_260801;
        rules.economy.clear_hex_income = 0;
        rules.economy.farm_hex_income = 7;
        rules.economy.farm_base_price = 8;
        rules
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.combat.maximum_unit_strength == 0
            || usize::from(self.combat.maximum_unit_strength) >= self.economy.unit_upkeep.len()
        {
            return Err(ConfigError::InvalidUnitStrength {
                strength: self.combat.maximum_unit_strength,
                maximum: MAXIMUM_CONFIGURED_UNIT_STRENGTH,
            });
        }
        if self.combat.movement_range == 0 {
            return Err(ConfigError::ZeroMovementRange);
        }
        if self.vegetation.pine_spread_per_million > 1_000_000
            || self.vegetation.palm_spread_per_million > 1_000_000
            || self.vegetation.target_spread_per_million > 1_000_000
        {
            return Err(ConfigError::InvalidProbability);
        }
        if self.economy.farm_price_increment < 0 {
            return Err(ConfigError::NegativeFarmPriceIncrement);
        }
        Ok(())
    }
}

impl Default for Rules {
    fn default() -> Self {
        Self::classic_generic()
    }
}

#[cfg(test)]
mod tests {
    use super::{Rules, RulesProfile};

    #[test]
    fn classic_profiles_match_observed_prices_and_upkeep() {
        let generic = Rules::classic_generic();
        assert_eq!(generic.economy.unit_upkeep, [0, 2, 6, 18, 36]);
        assert_eq!(generic.economy.farm_base_price, 12);
        assert_eq!(generic.economy.farm_hex_income, 5);
        assert!(generic.combat.strongest_unit_ignores_defense);

        let slay = Rules::classic_slay();
        assert_eq!(slay.profile, RulesProfile::ClassicSlay);
        assert_eq!(slay.economy.unit_upkeep[4], 54);
        assert!(!slay.combat.farms_enabled);
        assert!(slay.combat.towers_enabled);
        assert!(!slay.combat.strong_towers_enabled);
        assert!(!slay.combat.strongest_unit_ignores_defense);

        let duel = Rules::online_duel_v1();
        assert!(!duel.combat.recruited_units_ready_on_owned_empty);
        assert!(!duel.combat.recruited_merge_preserves_readiness);
        assert!(duel.combat.foreign_recruit_requires_economic_neighbour);
        assert!(duel.vegetation.target_based_spread);
        assert_eq!(duel.vegetation.target_spread_per_million, 330_000);
        assert!(duel.vegetation.charge_player_zero_per_spawn);
        assert!(!duel.vegetation.grave_tree_skips_next_cycle);

        let experimental = Rules::online_experimental_v2_260801();
        assert_eq!(experimental.economy.clear_hex_income, 0);
        assert_eq!(experimental.economy.farm_hex_income, 7);
        assert_eq!(experimental.economy.farm_base_price, 8);
    }

    #[test]
    fn bundled_profiles_are_valid() {
        assert_eq!(Rules::classic_generic().validate(), Ok(()));
        assert_eq!(Rules::classic_slay().validate(), Ok(()));
        assert_eq!(Rules::online_default_v1().validate(), Ok(()));
        assert_eq!(Rules::online_classic_v1().validate(), Ok(()));
        assert_eq!(Rules::online_duel_v1().validate(), Ok(()));
        assert_eq!(Rules::online_experimental_v1().validate(), Ok(()));
        assert_eq!(Rules::online_experimental_v2_260801().validate(), Ok(()));
    }
}
