use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    pub(crate) fn index(&mut self, length: usize) -> usize {
        let length = u64::try_from(length).expect("collection length fits in u64");
        usize::try_from(self.next_u64() % length).expect("random index fits in usize")
    }

    pub(crate) fn occurs_per_million(&mut self, probability: u32) -> bool {
        self.next_u64() % 1_000_000 < u64::from(probability)
    }
}

#[cfg(test)]
mod tests {
    use super::DeterministicRng;

    #[test]
    fn stream_is_stable() {
        let mut random = DeterministicRng::new(42);
        assert_eq!(random.next_u64(), 2_949_826_092_126_892_291);
        assert_eq!(random.next_u64(), 5_139_283_748_462_763_858);
        assert_eq!(random.next_u64(), 6_349_198_060_258_255_764);
    }
}
