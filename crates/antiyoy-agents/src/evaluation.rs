use antiyoy_core::{Game, Object, PlayerId};

const WIN_SCORE: i64 = 1_000_000_000_000;
pub const SCORE_COMPONENT_WEIGHTS: [i64; 16] = [
    128, 28, 82, 210, 430, 42, 58, 46, 96, -10, -16, 5, 1, 11, -15, -14,
];

pub fn position_score(game: &Game, player: PlayerId) -> i64 {
    if game.is_terminal() {
        return if game.winner() == Some(player) {
            WIN_SCORE
        } else {
            -WIN_SCORE
        };
    }
    evaluate::<false>(game, player).0
}

pub fn position_components(game: &Game, player: PlayerId) -> [i64; 16] {
    evaluate::<true>(game, player).1
}

pub fn position_score_from_components(components: &[i64; 16]) -> i64 {
    components
        .iter()
        .zip(SCORE_COMPONENT_WEIGHTS)
        .map(|(value, weight)| value * weight)
        .sum()
}

fn evaluate<const COMPONENTS: bool>(game: &Game, player: PlayerId) -> (i64, [i64; 16]) {
    let mut score = 0;
    let mut components = [0; 16];
    for (raw_hex, cell) in game.cells().iter().copied().enumerate() {
        if cell.owner().is_neutral() {
            continue;
        }
        let direction = if cell.owner() == player { 1 } else { -1 };
        let unit = cell.unit();
        let unit_component = match unit.strength() {
            0 => None,
            1 => Some(1),
            2 => Some(2),
            3 => Some(3),
            _ => Some(4),
        };
        let object_component = match cell.object() {
            Object::Empty => None,
            Object::Capital => Some(5),
            Object::Farm => Some(6),
            Object::Tower => Some(7),
            Object::StrongTower => Some(8),
            Object::Pine | Object::Palm => Some(9),
            Object::Grave => Some(10),
        };
        let defense = u16::try_from(raw_hex)
            .ok()
            .and_then(|hex| game.hex_defense(antiyoy_core::HexId(hex)))
            .unwrap_or_default();
        score += direction
            * (128
                + unit_component.map_or(0, |index| SCORE_COMPONENT_WEIGHTS[index])
                + object_component.map_or(0, |index| SCORE_COMPONENT_WEIGHTS[index])
                + i64::from(defense) * 5);
        if COMPONENTS {
            components[0] += direction;
            if let Some(index) = unit_component {
                components[index] += direction;
            }
            if let Some(index) = object_component {
                components[index] += direction;
            }
            components[11] += direction * i64::from(defense);
        }
    }
    for province in game.provinces() {
        let direction = if province.owner() == player { 1 } else { -1 };
        let income = game.province_income(province.id()).unwrap_or_default();
        let upkeep = game.province_upkeep(province.id()).unwrap_or_default();
        score += direction * (province.money() + income * 11 - upkeep * 15);
        score -= direction * 14;
        if COMPONENTS {
            components[12] += direction * province.money();
            components[13] += direction * income;
            components[14] += direction * upkeep;
            components[15] += direction;
        }
    }
    (score, components)
}
