use crate::board::Square;
use crate::moves::{Direction, Move};

// ---------------------------------------------------------------------------
// D4 — dihedral group of the square
// ---------------------------------------------------------------------------

/// The 8 elements of the dihedral group D4 on an NxN board.
///
/// Rotations are counter-clockwise. Reflections are self-inverse.
/// The coordinate system is row-major: row 0 is the top, column 0 is the left.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(u8)]
pub enum D4 {
    Identity = 0,
    Rot90 = 1,       // 90° counter-clockwise
    Rot180 = 2,
    Rot270 = 3,      // 270° counter-clockwise (= 90° clockwise)
    ReflectH = 4,    // reflection across horizontal axis (flip rows)
    ReflectV = 5,    // reflection across vertical axis (flip columns)
    ReflectMain = 6, // reflection across main diagonal: (r,c) -> (c,r)
    ReflectAnti = 7, // reflection across anti-diagonal
}

impl D4 {
    /// All 8 elements in discriminant order.
    pub const ALL: [D4; 8] = [
        D4::Identity,
        D4::Rot90,
        D4::Rot180,
        D4::Rot270,
        D4::ReflectH,
        D4::ReflectV,
        D4::ReflectMain,
        D4::ReflectAnti,
    ];

    // -- Square transform --------------------------------------------------

    /// Transform a square on an NxN board.
    ///
    /// The active region is rows `0..size`, columns `0..size` within the 8x8
    /// backing grid.  The input square must lie in the active region.
    #[inline]
    pub fn transform_square(self, sq: Square, size: u8) -> Square {
        let r = sq.row();
        let c = sq.col();
        let n = size - 1; // maximum index
        let (r2, c2) = match self {
            D4::Identity => (r, c),
            D4::Rot90 => (c, n - r),
            D4::Rot180 => (n - r, n - c),
            D4::Rot270 => (n - c, r),
            D4::ReflectH => (n - r, c),
            D4::ReflectV => (r, n - c),
            D4::ReflectMain => (c, r),
            D4::ReflectAnti => (n - c, n - r),
        };
        Square::from_rc(r2, c2)
    }

    // -- Direction transform -----------------------------------------------

    /// Transform a direction.
    ///
    /// The mapping is derived from how the coordinate transform acts on the
    /// unit displacement vectors:
    ///
    /// | Transform    | N→ | E→ | S→ | W→ |
    /// |--------------|----|----|----|----|
    /// | Identity     | N  | E  | S  | W  |
    /// | Rot90 (CCW)  | E  | S  | W  | N  |
    /// | Rot180       | S  | W  | N  | E  |
    /// | Rot270 (CCW) | W  | N  | E  | S  |
    /// | ReflectH     | S  | E  | N  | W  |
    /// | ReflectV     | N  | W  | S  | E  |
    /// | ReflectMain  | W  | S  | E  | N  |
    /// | ReflectAnti  | E  | N  | W  | S  |
    #[inline]
    pub fn transform_direction(self, dir: Direction) -> Direction {
        // Lookup table indexed by (transform, direction).
        // Outer index: D4 variant (0..8).  Inner index: Direction variant (0..4).
        const TABLE: [[Direction; 4]; 8] = [
            // Identity:    N       E       S       W
            [Direction::North, Direction::East, Direction::South, Direction::West],
            // Rot90 (CCW): N→E    E→S     S→W     W→N
            [Direction::East, Direction::South, Direction::West, Direction::North],
            // Rot180:      N→S    E→W     S→N     W→E
            [Direction::South, Direction::West, Direction::North, Direction::East],
            // Rot270 (CCW):N→W    E→N     S→E     W→S
            [Direction::West, Direction::North, Direction::East, Direction::South],
            // ReflectH:    N→S    E→E     S→N     W→W
            [Direction::South, Direction::East, Direction::North, Direction::West],
            // ReflectV:    N→N    E→W     S→S     W→E
            [Direction::North, Direction::West, Direction::South, Direction::East],
            // ReflectMain: N→W    E→S     S→E     W→N
            [Direction::West, Direction::South, Direction::East, Direction::North],
            // ReflectAnti: N→E    E→N     S→W     W→S
            [Direction::East, Direction::North, Direction::West, Direction::South],
        ];
        TABLE[self as usize][dir as usize]
    }

    // -- Move transform ----------------------------------------------------

    /// Transform a complete `Move`.
    ///
    /// For placements the square is transformed and the piece type is
    /// preserved.  For spreads the source square and direction are
    /// transformed; pickup count and template id (which encode the drop
    /// sequence along the ray) are preserved because the ordering along
    /// the ray is maintained by the consistent square+direction transform.
    #[inline]
    pub fn transform_move(self, mv: Move, size: u8) -> Move {
        match mv {
            Move::Place { square, piece_type } => Move::Place {
                square: self.transform_square(square, size),
                piece_type,
            },
            Move::Spread {
                src,
                dir,
                pickup,
                template,
            } => Move::Spread {
                src: self.transform_square(src, size),
                dir: self.transform_direction(dir),
                pickup,
                template,
            },
        }
    }

    // -- Group operations --------------------------------------------------

    /// The inverse of this transform.
    ///
    /// Rotations: Rot90⁻¹ = Rot270, Rot180⁻¹ = Rot180, Rot270⁻¹ = Rot90.
    /// All reflections are involutions (self-inverse).
    #[inline]
    pub fn inverse(self) -> D4 {
        match self {
            D4::Identity => D4::Identity,
            D4::Rot90 => D4::Rot270,
            D4::Rot180 => D4::Rot180,
            D4::Rot270 => D4::Rot90,
            D4::ReflectH => D4::ReflectH,
            D4::ReflectV => D4::ReflectV,
            D4::ReflectMain => D4::ReflectMain,
            D4::ReflectAnti => D4::ReflectAnti,
        }
    }

    /// Compose two transforms: apply `self` first, then `other`.
    ///
    /// That is, `self.compose(other)` is the transform `T` such that
    /// `T(x) = other(self(x))` for all x.
    #[inline]
    pub fn compose(self, other: D4) -> D4 {
        // Full 8x8 Cayley table for D4.
        // Row = self (applied first), Column = other (applied second).
        //
        // Element encoding:
        //   0=Id, 1=R90, 2=R180, 3=R270, 4=RH, 5=RV, 6=RM, 7=RA
        //
        // The table is derived from the group presentation:
        //   r^4 = 1, s^2 = 1, s*r*s = r^{-1}
        // where r = Rot90 and s = ReflectH.
        const T: [[u8; 8]; 8] = [
            //        Id  R90 R180 R270  RH   RV   RM   RA
            /* Id  */ [0, 1, 2, 3, 4, 5, 6, 7],
            /* R90 */ [1, 2, 3, 0, 7, 6, 4, 5],
            /* R180*/ [2, 3, 0, 1, 5, 4, 7, 6],
            /* R270*/ [3, 0, 1, 2, 6, 7, 5, 4],
            /* RH  */ [4, 6, 5, 7, 0, 2, 1, 3],
            /* RV  */ [5, 7, 4, 6, 2, 0, 3, 1],
            /* RM  */ [6, 5, 7, 4, 3, 1, 0, 2],
            /* RA  */ [7, 4, 6, 5, 1, 3, 2, 0],
        ];
        // SAFETY: both indices are in 0..8, result is in 0..8.
        unsafe { std::mem::transmute(T[self as usize][other as usize]) }
    }

    /// Construct a D4 from its discriminant value (0..8). Returns `None` for
    /// out-of-range values.
    #[inline]
    pub fn from_u8(v: u8) -> Option<D4> {
        if v < 8 {
            // SAFETY: v is in 0..8 which covers all variants.
            Some(unsafe { std::mem::transmute::<u8, D4>(v) })
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// D4Tables — precomputed lookup tables for a given board size
// ---------------------------------------------------------------------------

/// Precomputed lookup tables for fast D4 transforms on a given board size.
///
/// `square_map[transform][square_index]` gives the transformed square.
/// `direction_map[transform][direction_index]` gives the transformed direction.
///
/// Squares outside the active NxN region have undefined mappings (they are set
/// to `Square(0)` by default, but should not be used).
pub struct D4Tables {
    pub square_map: [[Square; 64]; 8],
    pub direction_map: [[Direction; 4]; 8],
}

impl D4Tables {
    /// Build lookup tables for an NxN board.
    pub fn build(size: u8) -> Self {
        let mut square_map = [[Square(0); 64]; 8];
        let mut direction_map = [[Direction::North; 4]; 8];

        for &sym in &D4::ALL {
            let si = sym as usize;

            // Squares in the active region.
            for r in 0..size {
                for c in 0..size {
                    let sq = Square::from_rc(r, c);
                    square_map[si][sq.0 as usize] = sym.transform_square(sq, size);
                }
            }

            // Directions.
            for &dir in &Direction::ALL {
                direction_map[si][dir as usize] = sym.transform_direction(dir);
            }
        }

        D4Tables {
            square_map,
            direction_map,
        }
    }

    /// Look up the transformed square.
    #[inline]
    pub fn transform_square(&self, sym: D4, sq: Square) -> Square {
        self.square_map[sym as usize][sq.0 as usize]
    }

    /// Look up the transformed direction.
    #[inline]
    pub fn transform_direction(&self, sym: D4, dir: Direction) -> Direction {
        self.direction_map[sym as usize][dir as usize]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Convenience: all board sizes we want to test.
    const SIZES: [u8; 6] = [3, 4, 5, 6, 7, 8];

    // -- Test 1: round-trip (transform then inverse) -----------------------

    #[test]
    fn roundtrip_square_all_transforms() {
        for &size in &SIZES {
            for &sym in &D4::ALL {
                let inv = sym.inverse();
                for r in 0..size {
                    for c in 0..size {
                        let sq = Square::from_rc(r, c);
                        let fwd = sym.transform_square(sq, size);
                        let back = inv.transform_square(fwd, size);
                        assert_eq!(
                            sq, back,
                            "size={size}, sym={sym:?}: ({r},{c}) -> ({},{}) -> ({},{})",
                            fwd.row(), fwd.col(), back.row(), back.col()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn roundtrip_direction_all_transforms() {
        for &sym in &D4::ALL {
            let inv = sym.inverse();
            for &dir in &Direction::ALL {
                let fwd = sym.transform_direction(dir);
                let back = inv.transform_direction(fwd);
                assert_eq!(
                    dir, back,
                    "sym={sym:?}: {dir:?} -> {fwd:?} -> {back:?}"
                );
            }
        }
    }

    // -- Test 2: group axioms (Cayley table) --------------------------------

    #[test]
    fn identity_element() {
        for &sym in &D4::ALL {
            assert_eq!(D4::Identity.compose(sym), sym);
            assert_eq!(sym.compose(D4::Identity), sym);
        }
    }

    #[test]
    fn inverse_property() {
        for &sym in &D4::ALL {
            assert_eq!(sym.compose(sym.inverse()), D4::Identity);
            assert_eq!(sym.inverse().compose(sym), D4::Identity);
        }
    }

    #[test]
    fn associativity() {
        for &a in &D4::ALL {
            for &b in &D4::ALL {
                for &c in &D4::ALL {
                    let ab_c = a.compose(b).compose(c);
                    let a_bc = a.compose(b.compose(c));
                    assert_eq!(
                        ab_c, a_bc,
                        "associativity: ({a:?}*{b:?})*{c:?} = {ab_c:?} != \
                         {a:?}*({b:?}*{c:?}) = {a_bc:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn group_has_8_elements() {
        assert_eq!(D4::ALL.len(), 8);

        // All distinct.
        for i in 0..8 {
            for j in (i + 1)..8 {
                assert_ne!(D4::ALL[i], D4::ALL[j]);
            }
        }
    }

    #[test]
    fn closure() {
        // Every product is one of the 8 elements.
        for &a in &D4::ALL {
            for &b in &D4::ALL {
                let c = a.compose(b);
                assert!(
                    D4::ALL.contains(&c),
                    "{a:?} * {b:?} = {c:?} not in D4::ALL"
                );
            }
        }
    }

    // -- Test 3: D4 has exactly 8 elements ---------------------------------
    // (covered by group_has_8_elements above)

    // -- Test 4: transforms preserve the active region ---------------------

    #[test]
    fn transform_stays_in_active_region() {
        for &size in &SIZES {
            for &sym in &D4::ALL {
                for r in 0..size {
                    for c in 0..size {
                        let sq = Square::from_rc(r, c);
                        let t = sym.transform_square(sq, size);
                        assert!(
                            t.row() < size && t.col() < size,
                            "size={size}, sym={sym:?}: ({r},{c}) -> ({},{}) out of range",
                            t.row(),
                            t.col()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn transform_is_bijection_on_active_region() {
        for &size in &SIZES {
            for &sym in &D4::ALL {
                let mut seen = [false; 64];
                for r in 0..size {
                    for c in 0..size {
                        let sq = Square::from_rc(r, c);
                        let t = sym.transform_square(sq, size);
                        assert!(
                            !seen[t.0 as usize],
                            "size={size}, sym={sym:?}: duplicate mapping to ({},{})",
                            t.row(),
                            t.col()
                        );
                        seen[t.0 as usize] = true;
                    }
                }
            }
        }
    }

    // -- Test 5: direction transforms consistent with square transforms ----

    #[test]
    fn direction_consistent_with_square() {
        // For every (sym, square, direction), check that:
        //   transform(sq + delta(dir)) == transform(sq) + delta(transform(dir))
        // where "sq + delta(dir)" means stepping one square in direction dir.
        for &size in &SIZES {
            for &sym in &D4::ALL {
                for r in 0..size {
                    for c in 0..size {
                        for &dir in &Direction::ALL {
                            let (dr, dc) = dir.delta();
                            let r2 = r as i8 + dr;
                            let c2 = c as i8 + dc;
                            // Skip if the step goes off the board.
                            if r2 < 0
                                || r2 >= size as i8
                                || c2 < 0
                                || c2 >= size as i8
                            {
                                continue;
                            }

                            let sq = Square::from_rc(r, c);
                            let sq_next = Square::from_rc(r2 as u8, c2 as u8);

                            // Transform both squares.
                            let tsq = sym.transform_square(sq, size);
                            let tsq_next = sym.transform_square(sq_next, size);

                            // Step from tsq in the transformed direction.
                            let tdir = sym.transform_direction(dir);
                            let (tdr, tdc) = tdir.delta();
                            let expected = Square::from_rc(
                                (tsq.row() as i8 + tdr) as u8,
                                (tsq.col() as i8 + tdc) as u8,
                            );

                            assert_eq!(
                                tsq_next, expected,
                                "size={size}, sym={sym:?}, sq=({r},{c}), dir={dir:?}: \
                                 transform(sq+dir)=({},{}) but \
                                 transform(sq)+transform(dir)=({},{})",
                                tsq_next.row(),
                                tsq_next.col(),
                                expected.row(),
                                expected.col()
                            );
                        }
                    }
                }
            }
        }
    }

    // -- Test 6: D4Tables matches D4 methods --------------------------------

    #[test]
    fn d4tables_matches_methods() {
        for &size in &SIZES {
            let tables = D4Tables::build(size);
            for &sym in &D4::ALL {
                // Squares.
                for r in 0..size {
                    for c in 0..size {
                        let sq = Square::from_rc(r, c);
                        assert_eq!(
                            tables.transform_square(sym, sq),
                            sym.transform_square(sq, size),
                            "size={size}, sym={sym:?}, sq=({r},{c})"
                        );
                    }
                }
                // Directions.
                for &dir in &Direction::ALL {
                    assert_eq!(
                        tables.transform_direction(sym, dir),
                        sym.transform_direction(dir),
                        "size={size}, sym={sym:?}, dir={dir:?}"
                    );
                }
            }
        }
    }

    // -- Test 7: compose matches sequential application on squares ----------

    #[test]
    fn compose_matches_sequential_application() {
        for &size in &SIZES {
            for &a in &D4::ALL {
                for &b in &D4::ALL {
                    let ab = a.compose(b);
                    for r in 0..size {
                        for c in 0..size {
                            let sq = Square::from_rc(r, c);
                            let sequential = b.transform_square(
                                a.transform_square(sq, size),
                                size,
                            );
                            let composed = ab.transform_square(sq, size);
                            assert_eq!(
                                sequential, composed,
                                "size={size}, a={a:?}, b={b:?}, sq=({r},{c}): \
                                 b(a(sq))=({},{}) but (a*b)(sq)=({},{})",
                                sequential.row(),
                                sequential.col(),
                                composed.row(),
                                composed.col()
                            );
                        }
                    }
                }
            }
        }
    }

    // -- Test 8: specific transform spot-checks -----------------------------

    #[test]
    fn spot_check_rot90_size5() {
        // (0,0) -> (0, 4)
        assert_eq!(
            D4::Rot90.transform_square(Square::from_rc(0, 0), 5),
            Square::from_rc(0, 4)
        );
        // (0,4) -> (4, 4)
        assert_eq!(
            D4::Rot90.transform_square(Square::from_rc(0, 4), 5),
            Square::from_rc(4, 4)
        );
        // (4,4) -> (4, 0)
        assert_eq!(
            D4::Rot90.transform_square(Square::from_rc(4, 4), 5),
            Square::from_rc(4, 0)
        );
        // (4,0) -> (0, 0) — full cycle
        assert_eq!(
            D4::Rot90.transform_square(Square::from_rc(4, 0), 5),
            Square::from_rc(0, 0)
        );
        // Centre stays: (2,2) -> (2, 2)
        assert_eq!(
            D4::Rot90.transform_square(Square::from_rc(2, 2), 5),
            Square::from_rc(2, 2)
        );
    }

    #[test]
    fn spot_check_reflect_main_size4() {
        // (0,0) -> (0,0)
        assert_eq!(
            D4::ReflectMain.transform_square(Square::from_rc(0, 0), 4),
            Square::from_rc(0, 0)
        );
        // (0,3) -> (3,0)
        assert_eq!(
            D4::ReflectMain.transform_square(Square::from_rc(0, 3), 4),
            Square::from_rc(3, 0)
        );
        // (1,2) -> (2,1)
        assert_eq!(
            D4::ReflectMain.transform_square(Square::from_rc(1, 2), 4),
            Square::from_rc(2, 1)
        );
    }

    // -- Test 9: rotation orders -------------------------------------------

    #[test]
    fn rotation_orders() {
        // Rot90^4 = Identity
        let mut t = D4::Rot90;
        for _ in 0..3 {
            t = t.compose(D4::Rot90);
        }
        assert_eq!(t, D4::Identity);

        // Rot180^2 = Identity
        assert_eq!(D4::Rot180.compose(D4::Rot180), D4::Identity);

        // All reflections have order 2
        for &refl in &[
            D4::ReflectH,
            D4::ReflectV,
            D4::ReflectMain,
            D4::ReflectAnti,
        ] {
            assert_eq!(refl.compose(refl), D4::Identity, "{refl:?}^2 != Id");
        }
    }

    // -- Test 10: transform_move preserves variant -------------------------

    #[test]
    fn transform_move_place() {
        use crate::piece::PieceType;

        let mv = Move::Place {
            square: Square::from_rc(1, 2),
            piece_type: PieceType::Cap,
        };
        let t = D4::Rot90.transform_move(mv, 5);
        match t {
            Move::Place { square, piece_type } => {
                assert_eq!(
                    square,
                    D4::Rot90.transform_square(Square::from_rc(1, 2), 5)
                );
                assert_eq!(piece_type, PieceType::Cap);
            }
            _ => panic!("expected Place"),
        }
    }

    #[test]
    fn transform_move_spread() {
        use crate::templates::DropTemplateId;

        let mv = Move::Spread {
            src: Square::from_rc(3, 1),
            dir: Direction::North,
            pickup: 3,
            template: DropTemplateId(42),
        };
        let t = D4::ReflectMain.transform_move(mv, 6);
        match t {
            Move::Spread {
                src,
                dir,
                pickup,
                template,
            } => {
                assert_eq!(
                    src,
                    D4::ReflectMain.transform_square(Square::from_rc(3, 1), 6)
                );
                assert_eq!(
                    dir,
                    D4::ReflectMain.transform_direction(Direction::North)
                );
                assert_eq!(pickup, 3);
                assert_eq!(template, DropTemplateId(42));
            }
            _ => panic!("expected Spread"),
        }
    }

    // -- Test 11: from_u8 ---------------------------------------------------

    #[test]
    fn from_u8_valid() {
        for i in 0u8..8 {
            let d = D4::from_u8(i).unwrap();
            assert_eq!(d as u8, i);
        }
    }

    #[test]
    fn from_u8_invalid() {
        assert!(D4::from_u8(8).is_none());
        assert!(D4::from_u8(255).is_none());
    }
}
