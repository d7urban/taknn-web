use arrayvec::ArrayVec;

use crate::piece::{Color, Piece};

/// Square index: 0..63, row-major on the 8x8 grid.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Square(pub u8);

impl Square {
    #[inline]
    pub fn from_rc(r: u8, c: u8) -> Self {
        debug_assert!(r < 8 && c < 8);
        Square(r * 8 + c)
    }

    #[inline]
    pub fn row(self) -> u8 {
        self.0 / 8
    }

    #[inline]
    pub fn col(self) -> u8 {
        self.0 % 8
    }
}

/// Fixed-capacity stack. Stores top piece + up to 7 interior colors + buried counts.
/// Interior pieces (below top) are always flat, so only Color is stored.
#[derive(Clone, Debug)]
pub struct Stack {
    /// Top piece, if any. None = empty square.
    pub top: Option<Piece>,
    /// Pieces below top, from second-from-top downward.
    /// Length 0..=7. Only the top min(height-1, 7) are stored.
    pub below: ArrayVec<Color, 7>,
    /// Count of white flats buried below the explicit layers.
    pub buried_white: u8,
    /// Count of black flats buried below the explicit layers.
    pub buried_black: u8,
    /// Total height of the stack.
    pub height: u8,
}

impl Stack {
    pub fn empty() -> Self {
        Stack {
            top: None,
            below: ArrayVec::new(),
            buried_white: 0,
            buried_black: 0,
            height: 0,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.top.is_none()
    }

    /// Place a piece on top of this stack. The caller must verify legality
    /// (e.g., cannot stack on walls/caps).
    pub fn push(&mut self, piece: Piece) {
        if let Some(old_top) = self.top {
            // Old top becomes an interior piece (must be a flat for normal stacking,
            // or a wall being flattened by a capstone — caller handles type change).
            if self.below.len() == 7 {
                // Overflow: move the bottommost explicit layer to buried counts.
                let buried_color = self.below[6];
                match buried_color {
                    Color::White => self.buried_white += 1,
                    Color::Black => self.buried_black += 1,
                }
                self.below.pop();
            }
            self.below.insert(0, old_top.color());
        }
        self.top = Some(piece);
        self.height += 1;
    }

    /// Remove the top piece from this stack. Returns it, or None if empty.
    ///
    /// When buried pieces exist and the explicit `below` array is drained,
    /// a buried piece is promoted as the new top. The color chosen is
    /// arbitrary (white first), which is acceptable because `apply_spread`
    /// always snapshots the stack in `UndoInfo` before mutating, so undo
    /// restores the exact original ordering.
    pub fn pop(&mut self) -> Option<Piece> {
        let piece = self.top.take()?;
        self.height -= 1;
        if self.below.is_empty() {
            if self.buried_white > 0 || self.buried_black > 0 {
                // Promote a buried piece into the new top. Exact color order
                // is unknown, but UndoInfo restores correctness on undo.
                let color = if self.buried_white > 0 {
                    self.buried_white -= 1;
                    Color::White
                } else {
                    self.buried_black -= 1;
                    Color::Black
                };
                self.top = Some(Piece::new(color, crate::piece::PieceType::Flat));
            } else {
                self.top = None;
            }
        } else {
            // Promote the first below-top piece to the new top (as a flat).
            let color = self.below.remove(0);
            self.top = Some(Piece::new(color, crate::piece::PieceType::Flat));

            // If there are buried pieces and we have room, promote one
            // into the bottom of the explicit layer.
            if self.below.len() < 7 && (self.buried_white > 0 || self.buried_black > 0) {
                let buried_color = if self.buried_white > 0 {
                    self.buried_white -= 1;
                    Color::White
                } else {
                    self.buried_black -= 1;
                    Color::Black
                };
                self.below.push(buried_color);
            }
        }
        Some(piece)
    }

    /// Get the color of the top piece, if any.
    #[inline]
    pub fn top_color(&self) -> Option<Color> {
        self.top.map(|p| p.color())
    }

    /// Iterate colors from top to bottom (top piece color, then below, then buried).
    /// Only yields the explicit layers, not buried.
    pub fn colors_top_down(&self) -> impl Iterator<Item = Color> + '_ {
        self.top
            .map(|p| p.color())
            .into_iter()
            .chain(self.below.iter().copied())
    }
}

/// 8x8 board. Squares outside the active NxN region are always empty.
/// Row-major indexing: square(r, c) = board.squares[r * 8 + c].
#[derive(Clone, Debug)]
pub struct Board {
    pub squares: [Stack; 64],
}

impl Board {
    pub fn empty() -> Self {
        Board {
            squares: std::array::from_fn(|_| Stack::empty()),
        }
    }

    #[inline]
    pub fn get(&self, sq: Square) -> &Stack {
        &self.squares[sq.0 as usize]
    }

    #[inline]
    pub fn get_mut(&mut self, sq: Square) -> &mut Stack {
        &mut self.squares[sq.0 as usize]
    }

    /// Count of flats on top for each color, within the active NxN region.
    pub fn flat_counts(&self, size: u8) -> (u32, u32) {
        let mut white = 0u32;
        let mut black = 0u32;
        for r in 0..size {
            for c in 0..size {
                let sq = Square::from_rc(r, c);
                if let Some(piece) = self.get(sq).top {
                    if piece.is_flat() {
                        match piece.color() {
                            Color::White => white += 1,
                            Color::Black => black += 1,
                        }
                    }
                }
            }
        }
        (white, black)
    }

    /// Count empty squares within the active NxN region.
    pub fn empty_count(&self, size: u8) -> u32 {
        let mut count = 0u32;
        for r in 0..size {
            for c in 0..size {
                if self.get(Square::from_rc(r, c)).is_empty() {
                    count += 1;
                }
            }
        }
        count
    }

    /// Apply D4 transform to the entire board.
    pub fn transform(&self, sym: crate::symmetry::D4, size: u8) -> Self {
        let mut new_board = Board::empty();
        for r in 0..size {
            for c in 0..size {
                let old_sq = Square::from_rc(r, c);
                let new_sq = sym.transform_square(old_sq, size);
                new_board.squares[new_sq.0 as usize] = self.squares[old_sq.0 as usize].clone();
            }
        }
        new_board
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::PieceType;

    #[test]
    fn square_roundtrip() {
        for r in 0..8u8 {
            for c in 0..8u8 {
                let sq = Square::from_rc(r, c);
                assert_eq!(sq.row(), r);
                assert_eq!(sq.col(), c);
            }
        }
    }

    #[test]
    fn stack_push_pop() {
        let mut s = Stack::empty();
        assert!(s.is_empty());
        assert_eq!(s.height, 0);

        s.push(Piece::new(Color::White, PieceType::Flat));
        assert_eq!(s.height, 1);
        assert_eq!(s.top, Some(Piece::WhiteFlat));
        assert!(s.below.is_empty());

        s.push(Piece::new(Color::Black, PieceType::Flat));
        assert_eq!(s.height, 2);
        assert_eq!(s.top, Some(Piece::BlackFlat));
        assert_eq!(s.below.len(), 1);
        assert_eq!(s.below[0], Color::White);

        let popped = s.pop().unwrap();
        assert_eq!(popped, Piece::BlackFlat);
        assert_eq!(s.height, 1);
        assert_eq!(s.top, Some(Piece::WhiteFlat));
    }

    #[test]
    fn board_empty_count() {
        let board = Board::empty();
        assert_eq!(board.empty_count(6), 36);
        assert_eq!(board.empty_count(8), 64);
    }
}
