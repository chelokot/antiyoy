use serde::{Deserialize, Serialize};

use crate::{ConfigError, HexId};

const MAXIMUM_HEXES: usize = u16::MAX as usize - 1;
const DIRECTIONS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Axial {
    pub q: i32,
    pub r: i32,
}

impl Axial {
    pub fn distance(self, other: Self) -> u32 {
        let delta_q = self.q - other.q;
        let delta_r = self.r - other.r;
        let delta_s = -delta_q - delta_r;
        (delta_q.unsigned_abs() + delta_r.unsigned_abs() + delta_s.unsigned_abs()) / 2
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Topology {
    width: u16,
    height: u16,
    coordinates: Vec<Axial>,
    neighbours: Vec<[HexId; 6]>,
    playable: Vec<bool>,
    playable_hexes: Vec<HexId>,
}

impl Topology {
    pub fn rectangle(width: u16, height: u16) -> Result<Self, ConfigError> {
        let cells = Self::cell_count(width, height)?;
        Self::masked_rectangle(width, height, vec![true; cells])
    }

    pub fn masked_rectangle(
        width: u16,
        height: u16,
        playable: Vec<bool>,
    ) -> Result<Self, ConfigError> {
        let cells = Self::cell_count(width, height)?;
        if playable.len() != cells {
            return Err(ConfigError::PlayableMaskSize {
                actual: playable.len(),
                expected: cells,
            });
        }

        let mut coordinates = Vec::with_capacity(cells);
        let mut neighbours = vec![[HexId::INVALID; 6]; cells];
        let mut playable_hexes = Vec::with_capacity(cells);

        for row in 0..height {
            for column in 0..width {
                let id = HexId(row * width + column);
                coordinates.push(Axial {
                    q: i32::from(column),
                    r: i32::from(row),
                });
                if playable[id.index()] {
                    playable_hexes.push(id);
                }
            }
        }

        for id in playable_hexes.iter().copied() {
            let coordinate = coordinates[id.index()];
            for (direction, (delta_q, delta_r)) in DIRECTIONS.iter().copied().enumerate() {
                let q = coordinate.q + delta_q;
                let r = coordinate.r + delta_r;
                if q < 0 || r < 0 || q >= i32::from(width) || r >= i32::from(height) {
                    continue;
                }
                let Ok(neighbour_row) = u16::try_from(r) else {
                    continue;
                };
                let Ok(neighbour_column) = u16::try_from(q) else {
                    continue;
                };
                let neighbour = HexId(neighbour_row * width + neighbour_column);
                if playable[neighbour.index()] {
                    neighbours[id.index()][direction] = neighbour;
                }
            }
        }

        Ok(Self {
            width,
            height,
            coordinates,
            neighbours,
            playable,
            playable_hexes,
        })
    }

    pub const fn width(&self) -> u16 {
        self.width
    }

    pub const fn height(&self) -> u16 {
        self.height
    }

    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.playable_hexes.is_empty()
    }

    pub fn is_playable(&self, id: HexId) -> bool {
        id.index() < self.playable.len() && self.playable[id.index()]
    }

    pub fn coordinate(&self, id: HexId) -> Option<Axial> {
        self.coordinates.get(id.index()).copied()
    }

    pub fn neighbours(&self, id: HexId) -> Option<&[HexId; 6]> {
        self.neighbours.get(id.index())
    }

    pub fn playable_hexes(&self) -> &[HexId] {
        &self.playable_hexes
    }

    fn cell_count(width: u16, height: u16) -> Result<usize, ConfigError> {
        if width == 0 || height == 0 {
            return Err(ConfigError::EmptyMap);
        }
        let cells = usize::from(width) * usize::from(height);
        if cells > MAXIMUM_HEXES {
            return Err(ConfigError::MapTooLarge {
                cells,
                maximum: MAXIMUM_HEXES,
            });
        }
        Ok(cells)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Axial, HexId, Topology};

    #[test]
    fn axial_distance_matches_hex_geometry() {
        assert_eq!(Axial { q: 0, r: 0 }.distance(Axial { q: 3, r: -2 }), 3);
        assert_eq!(Axial { q: 5, r: 4 }.distance(Axial { q: 5, r: 4 }), 0);
    }

    #[test]
    fn rectangle_precomputes_symmetric_neighbours() {
        let topology = Topology::rectangle(3, 3).expect("valid topology");
        let center = HexId(4);
        let center_neighbours = topology.neighbours(center).expect("known hex");
        assert_eq!(
            center_neighbours.iter().filter(|id| id.is_valid()).count(),
            6
        );

        for neighbour in center_neighbours.iter().copied() {
            let reverse = topology.neighbours(neighbour).expect("known neighbour");
            assert!(reverse.contains(&center));
        }
    }

    #[test]
    fn mask_removes_edges_to_inactive_hexes() {
        let mut playable = vec![true; 9];
        playable[4] = false;
        let topology = Topology::masked_rectangle(3, 3, playable).expect("valid topology");
        assert!(!topology.is_playable(HexId(4)));
        assert_eq!(topology.playable_hexes().len(), 8);
        assert!(
            topology
                .playable_hexes()
                .iter()
                .flat_map(|id| topology.neighbours(*id).expect("known hex"))
                .all(|neighbour| *neighbour != HexId(4))
        );
    }
}
