use antiyoy_core::{Game, Object, PlayerId};

const WIN_SCORE: i64 = 1_000_000_000_000;

pub(crate) fn position_score(game: &Game, player: PlayerId) -> i64 {
    if game.is_terminal() {
        return if game.winner() == Some(player) {
            WIN_SCORE
        } else {
            -WIN_SCORE
        };
    }

    let mut score = 0;
    for (raw_hex, cell) in game.cells().iter().copied().enumerate() {
        if cell.owner().is_neutral() {
            continue;
        }
        let direction = if cell.owner() == player { 1 } else { -1 };
        let unit = cell.unit();
        let unit_score = match unit.strength() {
            0 => 0,
            1 => 28,
            2 => 82,
            3 => 210,
            _ => 430,
        } + i64::from(unit.is_ready()) * 6;
        let object_score = match cell.object() {
            Object::Empty => 0,
            Object::Capital => 42,
            Object::Farm => 58,
            Object::Tower => 46,
            Object::StrongTower => 96,
            Object::Pine | Object::Palm => -10,
            Object::Grave => -16,
        };
        let defense = u16::try_from(raw_hex)
            .ok()
            .and_then(|hex| game.hex_defense(antiyoy_core::HexId(hex)))
            .unwrap_or_default();
        score += direction * (128 + unit_score + object_score + i64::from(defense) * 5);
    }
    for province in game.provinces() {
        let direction = if province.owner() == player { 1 } else { -1 };
        let income = game.province_income(province.id()).unwrap_or_default();
        let upkeep = game.province_upkeep(province.id()).unwrap_or_default();
        score += direction * (province.money() + income * 11 - upkeep * 15);
        score -= direction * 14;
    }
    score
}
