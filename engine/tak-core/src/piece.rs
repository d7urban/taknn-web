#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum PieceType {
    Flat = 0,
    Wall = 1,
    Cap = 2,
}

/// A piece on the board. Encodes (Color, PieceType) as color * 3 + piece_type.
/// Stack layers below top are always Flat, so only Color is needed there.
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Piece {
    WhiteFlat = 0,
    WhiteWall = 1,
    WhiteCap = 2,
    BlackFlat = 3,
    BlackWall = 4,
    BlackCap = 5,
}

impl Piece {
    #[inline]
    pub fn new(color: Color, piece_type: PieceType) -> Piece {
        // SAFETY: color * 3 + piece_type is always in 0..6
        unsafe { std::mem::transmute(color as u8 * 3 + piece_type as u8) }
    }

    #[inline]
    pub fn color(self) -> Color {
        if (self as u8) < 3 {
            Color::White
        } else {
            Color::Black
        }
    }

    #[inline]
    pub fn piece_type(self) -> PieceType {
        // SAFETY: self as u8 % 3 is always 0, 1, or 2
        unsafe { std::mem::transmute(self as u8 % 3) }
    }

    #[inline]
    pub fn is_flat(self) -> bool {
        self.piece_type() == PieceType::Flat
    }

    #[inline]
    pub fn is_wall(self) -> bool {
        self.piece_type() == PieceType::Wall
    }

    #[inline]
    pub fn is_cap(self) -> bool {
        self.piece_type() == PieceType::Cap
    }

    /// Can another piece be stacked on top of this piece?
    /// Flats: yes. Walls and caps: no (walls can be flattened by caps, handled separately).
    #[inline]
    pub fn can_stack_on(self) -> bool {
        self.is_flat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_roundtrip() {
        for &color in &[Color::White, Color::Black] {
            for &pt in &[PieceType::Flat, PieceType::Wall, PieceType::Cap] {
                let p = Piece::new(color, pt);
                assert_eq!(p.color(), color);
                assert_eq!(p.piece_type(), pt);
            }
        }
    }

    #[test]
    fn piece_discriminants() {
        assert_eq!(Piece::WhiteFlat as u8, 0);
        assert_eq!(Piece::WhiteWall as u8, 1);
        assert_eq!(Piece::WhiteCap as u8, 2);
        assert_eq!(Piece::BlackFlat as u8, 3);
        assert_eq!(Piece::BlackWall as u8, 4);
        assert_eq!(Piece::BlackCap as u8, 5);
    }
}
