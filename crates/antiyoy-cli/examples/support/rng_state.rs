use antiyoy_core::Game;

const RNG_SHIFT: u64 = 11_400_714_819_323_198_485;

pub fn shifted_random(game: &Game) -> Game {
    let mut value = serde_json::to_value(game).expect("game serializes");
    let random = value["random"]["state"]
        .as_u64()
        .expect("game RNG state is a u64");
    value["random"]["state"] = serde_json::json!(random.wrapping_add(RNG_SHIFT));
    serde_json::from_value(value).expect("shifted game deserializes")
}
