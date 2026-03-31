/// Game configuration for a specific board size.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GameConfig {
    pub size: u8,
    pub stones: u8,
    pub capstones: u8,
    pub carry_limit: u8,
    pub komi: i8,
    pub half_komi: bool,
}

impl GameConfig {
    /// Standard Tak configuration for the given board size.
    pub fn standard(size: u8) -> Self {
        let (stones, caps) = match size {
            3 => (10, 0),
            4 => (15, 0),
            5 => (21, 1),
            6 => (30, 1),
            7 => (40, 2),
            8 => (50, 2),
            _ => panic!("unsupported board size: {}", size),
        };
        GameConfig {
            size,
            stones,
            capstones: caps,
            carry_limit: size,
            komi: 0,
            half_komi: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_configs() {
        let cases = [
            (3, 10, 0),
            (4, 15, 0),
            (5, 21, 1),
            (6, 30, 1),
            (7, 40, 2),
            (8, 50, 2),
        ];
        for (size, stones, caps) in cases {
            let c = GameConfig::standard(size);
            assert_eq!(c.size, size);
            assert_eq!(c.stones, stones, "stones for size {}", size);
            assert_eq!(c.capstones, caps, "caps for size {}", size);
            assert_eq!(c.carry_limit, size);
            assert_eq!(c.komi, 0);
            assert!(!c.half_komi);
        }
    }
}
