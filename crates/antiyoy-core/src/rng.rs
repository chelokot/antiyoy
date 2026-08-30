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

pub(crate) struct JavaRandom {
    state: u64,
}

impl JavaRandom {
    const MULTIPLIER: u64 = 0x0005_DEEC_E66D;
    const ADDEND: u64 = 0xB;
    const MASK: u64 = (1_u64 << 48) - 1;

    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: (seed ^ Self::MULTIPLIER) & Self::MASK,
        }
    }

    pub(crate) fn index(&mut self, length: usize) -> usize {
        let bound = u32::try_from(length).expect("collection length fits in u32");
        assert!(bound > 0, "cannot sample an empty collection");
        if bound.is_power_of_two() {
            return usize::try_from((u64::from(bound) * u64::from(self.next(31))) >> 31)
                .expect("random index fits in usize");
        }
        loop {
            let bits = self.next(31);
            let value = bits % bound;
            if u64::from(bits - value) + u64::from(bound - 1) < 1_u64 << 31 {
                return usize::try_from(value).expect("random index fits in usize");
            }
        }
    }

    fn next(&mut self, bits: u8) -> u32 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::ADDEND)
            & Self::MASK;
        u32::try_from(self.state >> (48 - bits)).expect("requested bits fit in u32")
    }
}

#[cfg(test)]
mod tests {
    use super::{DeterministicRng, JavaRandom};

    #[test]
    fn stream_is_stable() {
        let mut random = DeterministicRng::new(42);
        assert_eq!(random.next_u64(), 2_949_826_092_126_892_291);
        assert_eq!(random.next_u64(), 5_139_283_748_462_763_858);
        assert_eq!(random.next_u64(), 6_349_198_060_258_255_764);
    }

    #[test]
    fn java_bounded_stream_matches_java_util_random() {
        let mut random = JavaRandom::new(0);
        assert_eq!(random.index(100), 60);
        assert_eq!(random.index(100), 48);
        assert_eq!(random.index(100), 29);
        assert_eq!(random.index(100), 47);
    }
}
