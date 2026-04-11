//! Game state for Tak: board, reserves, side to move, game result,
//! apply/undo move, road detection, and flat win detection.

use crate::board::{Board, Square, Stack};
use crate::moves::{Direction, Move, MoveList, MoveGen};
use crate::piece::{Color, Piece, PieceType};
use crate::rules::GameConfig;
use crate::templates::TemplateTable;
use crate::zobrist;

// ---------------------------------------------------------------------------
// GameResult
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum GameResult {
    Ongoing,
    RoadWin(Color),
    FlatWin(Color),
    Draw,
}

impl GameResult {
    #[inline]
    pub fn is_terminal(self) -> bool {
        self != GameResult::Ongoing
    }
}

// ---------------------------------------------------------------------------
// UndoInfo
// ---------------------------------------------------------------------------

/// Information needed to undo a move.
#[derive(Clone, Debug)]
pub struct UndoInfo {
    pub old_zobrist: u64,
    pub old_result: GameResult,
    /// Snapshots of all stacks affected by the move (src + path squares for
    /// spreads, or just the target square for placements).
    pub stack_snapshots: Vec<(Square, Stack)>,
    /// Was a wall flattened during a spread? If so, which square.
    pub flattened_wall: Option<Square>,
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub config: GameConfig,
    pub side_to_move: Color,
    pub ply: u16,
    /// [white_stones, white_caps, black_stones, black_caps]
    pub reserves: [u8; 4],
    pub result: GameResult,
    pub zobrist: u64,
    pub hash_history: Vec<u64>,
    /// Template table for this board size, used for move generation.
    pub templates: TemplateTable,
}

impl GameState {
    // -- Reserve index helpers -----------------------------------------------

    /// Index into `reserves` for stone count of the given color.
    #[inline]
    fn stones_idx(color: Color) -> usize {
        match color {
            Color::White => 0,
            Color::Black => 2,
        }
    }

    /// Index into `reserves` for capstone count of the given color.
    #[inline]
    fn caps_idx(color: Color) -> usize {
        match color {
            Color::White => 1,
            Color::Black => 3,
        }
    }

    /// Reserve index for a given color and piece type.
    /// Stones (Flat/Wall) use the stone reserve; Caps use the cap reserve.
    #[inline]
    fn reserve_idx(color: Color, piece_type: PieceType) -> usize {
        match piece_type {
            PieceType::Flat | PieceType::Wall => Self::stones_idx(color),
            PieceType::Cap => Self::caps_idx(color),
        }
    }

    // -- Constructor ---------------------------------------------------------

    /// Create a new game in the starting position.
    pub fn new(config: GameConfig) -> Self {
        let board = Board::empty();
        let reserves = [config.stones, config.capstones, config.stones, config.capstones];
        let zobrist = zobrist::compute_full(&board, config.size, Color::White, &reserves);

        let templates = TemplateTable::build(config.size);

        GameState {
            board,
            config,
            side_to_move: Color::White,
            ply: 0,
            reserves,
            result: GameResult::Ongoing,
            zobrist,
            hash_history: vec![zobrist],
            templates,
        }
    }

    // -- Queries -------------------------------------------------------------

    /// Returns true during the opening phase (first two plies).
    /// During the opening, each player places one of the OPPONENT's flats.
    #[inline]
    pub fn is_opening_phase(&self) -> bool {
        self.ply < 2
    }

    /// Returns white_flats - black_flats + komi at game end.
    pub fn flat_margin(&self) -> i16 {
        let (white, black) = self.board.flat_counts(self.config.size);
        let mut margin = white as i16 - black as i16;
        margin += self.config.komi as i16;
        // If half-komi is set, and it's a draw on whole points, it's effectively 0.5.
        // We handle this by returning non-zero. For TrainingRecord we mostly need the sign
        // and a rough magnitude.
        margin
    }

    /// Generate all legal moves for the current side to move.
    pub fn legal_moves(&self) -> MoveList {
        MoveGen::legal_moves(
            &self.board,
            &self.config,
            self.side_to_move,
            self.ply,
            &self.reserves,
            &self.templates,
        )
    }

    // -- Apply move ----------------------------------------------------------

    /// Apply a move to the game state and return the information needed to undo it.
    pub fn apply_move(&mut self, mv: Move) -> UndoInfo {
        let mut undo = UndoInfo {
            old_zobrist: self.zobrist,
            old_result: self.result,
            stack_snapshots: Vec::new(),
            flattened_wall: None,
        };

        let mover = self.side_to_move;

        match mv {
            Move::Place { square, piece_type } => {
                self.apply_place(square, piece_type, mover, &mut undo);
            }
            Move::Spread { src, dir, pickup, template } => {
                self.apply_spread(src, dir, pickup, template, &mut undo);
            }
        }

        // Toggle side to move.
        self.side_to_move = self.side_to_move.opposite();
        self.ply += 1;
        self.zobrist ^= zobrist::hash_side();

        // Push new hash to history.
        self.hash_history.push(self.zobrist);

        // Check for game-ending conditions.
        self.check_game_over(mover);

        undo
    }

    /// Apply a Place move.
    fn apply_place(
        &mut self,
        square: Square,
        piece_type: PieceType,
        mover: Color,
        undo: &mut UndoInfo,
    ) {
        // Snapshot the target stack.
        undo.stack_snapshots.push((square, self.board.get(square).clone()));

        // Determine which piece to place and whose reserves to decrement.
        let (place_color, place_type) = if self.is_opening_phase() {
            // Opening rule: place opponent's flat.
            (mover.opposite(), PieceType::Flat)
        } else {
            (mover, piece_type)
        };

        let piece = Piece::new(place_color, place_type);

        let sq_idx = square.0 as usize;

        // Decrement reserves.
        let res_idx = Self::reserve_idx(place_color, place_type);
        let old_reserve = self.reserves[res_idx];
        debug_assert!(old_reserve > 0, "no reserves left for {:?} {:?}", place_color, place_type);
        self.zobrist ^= zobrist::hash_reserve_diff(res_idx, old_reserve, old_reserve - 1);
        self.reserves[res_idx] = old_reserve - 1;

        // XOR out old stack, push piece, XOR in new stack.
        let old_stack = self.board.get(square).clone();
        self.board.get_mut(square).push(piece);
        self.zobrist ^= zobrist::hash_stack_diff(sq_idx, &old_stack, self.board.get(square));
    }

    /// Apply a Spread move.
    fn apply_spread(
        &mut self,
        src: Square,
        dir: Direction,
        pickup: u8,
        template: crate::templates::DropTemplateId,
        undo: &mut UndoInfo,
    ) {
        let size = self.config.size;
        let (dr, dc) = dir.delta();

        // Look up the drop sequence.
        let drop_seq = self.templates.get_sequence(template);
        let drops = &drop_seq.drops;
        let num_steps = drops.len();

        // Snapshot source stack.
        undo.stack_snapshots.push((src, self.board.get(src).clone()));

        // Snapshot all target squares along the path.
        let mut target_squares: Vec<Square> = Vec::with_capacity(num_steps);
        {
            let mut r = src.row() as i8;
            let mut c = src.col() as i8;
            for _ in 0..num_steps {
                r += dr;
                c += dc;
                debug_assert!(
                    r >= 0 && r < size as i8 && c >= 0 && c < size as i8,
                    "spread goes off board"
                );
                let sq = Square::from_rc(r as u8, c as u8);
                undo.stack_snapshots.push((sq, self.board.get(sq).clone()));
                target_squares.push(sq);
            }
        }

        // XOR out old hash for source stack.
        let src_idx = src.0 as usize;
        let old_src_stack = self.board.get(src).clone();

        // Pop `pickup` pieces from source. This gives [top, second, ...].
        let mut carried: Vec<Piece> = Vec::with_capacity(pickup as usize);
        for _ in 0..pickup {
            carried.push(
                self.board
                    .get_mut(src)
                    .pop()
                    .expect("not enough pieces on source stack"),
            );
        }
        // Reverse so carried[0] = bottom of carried sub-stack,
        // carried.last() = top (original top of source).
        carried.reverse();

        // Drop pieces onto each target square.
        let mut carry_idx: usize = 0;
        for (step, &drop_count) in drops.iter().enumerate() {
            let target_sq = target_squares[step];

            // Check for wall-flattening on the last step.
            let is_last = step == num_steps - 1;
            if is_last && drop_count == 1 {
                if let Some(top_piece) = self.board.get(target_sq).top {
                    if top_piece.is_wall() && carried.last().unwrap().is_cap() {
                        // Flatten the wall: change it to a flat of the same color.
                        let wall_color = top_piece.color();
                        let stack = self.board.get_mut(target_sq);
                        stack.top = Some(Piece::new(wall_color, PieceType::Flat));
                        undo.flattened_wall = Some(target_sq);
                    }
                }
            }

            // Drop `drop_count` pieces from the bottom of the carried sub-stack.
            let end_idx = carry_idx + drop_count as usize;
            for item in carried.iter().take(end_idx).skip(carry_idx) {
                self.board.get_mut(target_sq).push(*item);
            }
            carry_idx = end_idx;
        }
        debug_assert_eq!(carry_idx, carried.len(), "all carried pieces must be dropped");

        // Update zobrist for all affected stacks.
        self.zobrist ^= zobrist::hash_stack_diff(src_idx, &old_src_stack, self.board.get(src));
        for &(sq, ref old_stack) in &undo.stack_snapshots[1..] {
            // stack_snapshots[0] is the source (already handled above with old_src_stack).
            self.zobrist ^= zobrist::hash_stack_diff(sq.0 as usize, old_stack, self.board.get(sq));
        }
    }

    // -- Undo move -----------------------------------------------------------

    /// Undo a move, restoring the game state to what it was before.
    pub fn undo_move(&mut self, mv: Move, undo: &UndoInfo) {
        // Restore all stack snapshots.
        for (sq, snapshot) in &undo.stack_snapshots {
            *self.board.get_mut(*sq) = snapshot.clone();
        }

        // Adjust reserves back for Place moves.
        if let Move::Place { square: _, piece_type } = mv {
            let mover = self.side_to_move.opposite(); // side that made the move
            let (place_color, place_type) = if self.ply <= 2 {
                // Opening: was placing opponent's flat. ply is already incremented,
                // so opening plies were 0 and 1, now ply is 1 or 2.
                (mover.opposite(), PieceType::Flat)
            } else {
                (mover, piece_type)
            };
            let res_idx = Self::reserve_idx(place_color, place_type);
            self.reserves[res_idx] += 1;
        }

        // Decrement ply and toggle side back.
        self.ply -= 1;
        self.side_to_move = self.side_to_move.opposite();

        // Restore zobrist and result.
        self.zobrist = undo.old_zobrist;
        self.result = undo.old_result;

        // Pop hash_history.
        self.hash_history.pop();
    }

    // -- Game-over detection -------------------------------------------------

    /// Check whether the game has ended after a move by `mover`.
    fn check_game_over(&mut self, mover: Color) {
        // 1. Check for repetition (third occurrence of the current hash).
        let current_hash = self.zobrist;
        let count = self.hash_history.iter().filter(|&&h| h == current_hash).count();
        if count >= 3 {
            self.result = GameResult::Draw;
            return;
        }

        // 2. Road win check.
        let mover_road = check_road(&self.board, self.config.size, mover);
        let opponent_road = check_road(&self.board, self.config.size, mover.opposite());

        if mover_road {
            self.result = GameResult::RoadWin(mover);
            return;
        }
        if opponent_road {
            self.result = GameResult::RoadWin(mover.opposite());
            return;
        }

        // 3. Flat win check: board full or either player exhausted reserves.
        let board_full = self.board.empty_count(self.config.size) == 0;
        let white_exhausted = self.reserves[0] == 0 && self.reserves[1] == 0;
        let black_exhausted = self.reserves[2] == 0 && self.reserves[3] == 0;

        if board_full || white_exhausted || black_exhausted {
            self.result = flat_winner(&self.board, &self.config);
        }
    }
}

// ---------------------------------------------------------------------------
// Road detection
// ---------------------------------------------------------------------------

/// Check whether `color` has a road (connected path of flats or capstones
/// spanning opposite edges) on the active NxN region of the board.
///
/// A road connects the north edge to the south edge, or the west edge to the
/// east edge, through orthogonally adjacent squares whose top piece belongs
/// to `color` and is a flat or capstone (walls do NOT count for roads).
pub fn check_road(board: &Board, size: u8, color: Color) -> bool {
    let n = size as usize;

    // We use a bitset over the NxN grid. Max 8x8 = 64 bits.
    // Bit index = r * size + c.
    // First, build the set of all "road-contributing" squares for this color.
    let mut road_squares: u64 = 0;
    for r in 0..size {
        for c in 0..size {
            let sq = Square::from_rc(r, c);
            if let Some(top) = board.get(sq).top {
                if top.color() == color && (top.is_flat() || top.is_cap()) {
                    let bit = r as usize * n + c as usize;
                    road_squares |= 1u64 << bit;
                }
            }
        }
    }

    // Check north-south road: flood fill from every road square on the north
    // edge (row 0) and see if any reaches the south edge (row size-1).
    if check_road_direction(road_squares, n, true) {
        return true;
    }

    // Check east-west road: flood fill from west edge (col 0) to east edge (col size-1).
    if check_road_direction(road_squares, n, false) {
        return true;
    }

    false
}

/// Check for a road connecting two opposite edges.
/// `north_south`: if true, check north (row 0) to south (row n-1);
///                if false, check west (col 0) to east (col n-1).
fn check_road_direction(road_squares: u64, n: usize, north_south: bool) -> bool {
    // Seed: road squares on the starting edge.
    let mut visited: u64 = 0;
    let mut frontier: u64;

    if north_south {
        // Starting edge: row 0.
        let mut seed: u64 = 0;
        for c in 0..n {
            seed |= 1u64 << c; // bit for (0, c)
        }
        frontier = road_squares & seed;
    } else {
        // Starting edge: col 0.
        let mut seed: u64 = 0;
        for r in 0..n {
            seed |= 1u64 << (r * n); // bit for (r, 0)
        }
        frontier = road_squares & seed;
    }

    if frontier == 0 {
        return false;
    }

    // Target edge.
    let mut target: u64 = 0;
    if north_south {
        // row n-1
        for c in 0..n {
            target |= 1u64 << ((n - 1) * n + c);
        }
    } else {
        // col n-1
        for r in 0..n {
            target |= 1u64 << (r * n + (n - 1));
        }
    }

    // BFS using bitwise flood fill.
    while frontier != 0 {
        visited |= frontier;

        // Check if we reached the target edge.
        if visited & target != 0 {
            return true;
        }

        // Expand frontier to all orthogonal neighbors within the road set.
        let mut next: u64 = 0;

        // For each bit in frontier, add its neighbors.
        // We do this by shifting the whole frontier in each of the 4 directions,
        // masking out invalid wrap-arounds.
        // North: shift down by n (row decreases). bit >> n.
        next |= frontier >> n;
        // South: shift up by n (row increases). bit << n.
        next |= frontier << n;
        // East: shift right by 1 (col increases). bit << 1, but mask out col 0 wrap.
        // When we shift left by 1, a bit at col n-1 wraps to col 0 of the next row.
        // We need to mask out those bits.
        let not_last_col = build_col_mask(n, n - 1);
        next |= (frontier & not_last_col) << 1;
        // West: shift left by 1 (col decreases). bit >> 1, mask out col n-1 wrap.
        let not_first_col = build_col_mask(n, 0);
        next |= (frontier & not_first_col) >> 1;

        // Only keep squares that are road squares and not yet visited.
        let valid_mask = if n < 8 { (1u64 << (n * n)) - 1 } else { u64::MAX };
        frontier = next & road_squares & !visited & valid_mask;
    }

    false
}

/// Build a mask that has all bits SET except those in column `exclude_col`.
/// Used to prevent wrap-around during bitwise flood fill.
#[inline]
fn build_col_mask(n: usize, exclude_col: usize) -> u64 {
    let mut mask: u64 = 0;
    for r in 0..n {
        for c in 0..n {
            if c != exclude_col {
                mask |= 1u64 << (r * n + c);
            }
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// Fast road-in-1 detection (bitwise)
// ---------------------------------------------------------------------------

/// Expand a bitset by one orthogonal step in all four directions.
#[inline]
fn expand_one_step(bits: u64, n: usize) -> u64 {
    let valid_mask = if n < 8 { (1u64 << (n * n)) - 1 } else { u64::MAX };
    let not_last_col = build_col_mask(n, n - 1);
    let not_first_col = build_col_mask(n, 0);

    let mut exp = bits;
    exp |= bits >> n; // north
    exp |= bits << n; // south
    exp |= (bits & not_last_col) << 1; // east
    exp |= (bits & not_first_col) >> 1; // west
    exp & valid_mask
}

/// Return the bitset of empty squares where placing a road-eligible piece
/// would complete a road for `color`.
///
/// Each set bit `b` corresponds to grid position `(b / size, b % size)`.
/// Returns 0 when no single-placement road completion exists.
///
/// Runs in O(board_area) time using bitwise flood fill — much cheaper than
/// the O(legal_moves) approach of cloning state for every move.
pub fn road_bridging_squares(board: &Board, size: u8, color: Color) -> u64 {
    let n = size as usize;
    let valid_mask = if n < 8 { (1u64 << (n * n)) - 1 } else { u64::MAX };
    let mut road_sq: u64 = 0;
    let mut occupied: u64 = 0;

    for r in 0..n {
        for c in 0..n {
            let sq = Square::from_rc(r as u8, c as u8);
            let bit = r * n + c;
            if let Some(top) = board.get(sq).top {
                occupied |= 1u64 << bit;
                if top.color() == color && !top.is_wall() {
                    road_sq |= 1u64 << bit;
                }
            }
        }
    }

    // Need at least size-1 road-eligible squares for a one-placement road.
    if road_sq.count_ones() < n as u32 - 1 {
        return 0;
    }

    // Candidate squares: empty squares adjacent to existing road squares.
    let empty = !occupied & valid_mask;
    let candidates = expand_one_step(road_sq, n) & empty;

    // Keep only candidates that actually complete a road.
    let mut bridging: u64 = 0;
    let mut remaining = candidates;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as usize;
        let test = road_sq | (1u64 << bit);
        if check_road_direction(test, n, true) || check_road_direction(test, n, false) {
            bridging |= 1u64 << bit;
        }
        remaining &= remaining - 1; // clear lowest set bit
    }

    bridging
}

/// Fast bitwise check: can `color` complete a road by placing a single
/// flat or capstone on an empty square?
///
/// Thin wrapper around [`road_bridging_squares`] that also verifies the
/// player has reserves remaining.
pub fn has_road_in_1_placement(
    board: &Board,
    size: u8,
    color: Color,
    reserves: &[u8; 4],
) -> bool {
    let (stones, caps) = match color {
        Color::White => (reserves[0], reserves[1]),
        Color::Black => (reserves[2], reserves[3]),
    };
    if stones == 0 && caps == 0 {
        return false;
    }
    road_bridging_squares(board, size, color) != 0
}

// ---------------------------------------------------------------------------
// Flat win detection
// ---------------------------------------------------------------------------

/// Determine the winner (or draw) by flat count, applying komi.
///
/// Komi is added to White's score. If `half_komi` is true, White gets an
/// additional 0.5 points (breaking ties in White's favor when komi is applied).
pub fn flat_winner(board: &Board, config: &GameConfig) -> GameResult {
    let (white_flats, black_flats) = board.flat_counts(config.size);

    // Compare using doubled values to avoid floating point.
    // White's doubled score: 2 * white_flats + 2 * komi + (1 if half_komi else 0)
    let white_doubled =
        (white_flats as i32) * 2 + (config.komi as i32) * 2 + if config.half_komi { 1 } else { 0 };
    let black_doubled = (black_flats as i32) * 2;

    if white_doubled > black_doubled {
        GameResult::FlatWin(Color::White)
    } else if black_doubled > white_doubled {
        GameResult::FlatWin(Color::Black)
    } else {
        GameResult::Draw
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Square};
    use crate::piece::{Color, Piece, PieceType};
    use crate::rules::GameConfig;
    use crate::templates::DropTemplateId;

    // -----------------------------------------------------------------------
    // Helper: build a board from a description
    // -----------------------------------------------------------------------

    /// Place a flat on the board at (r, c).
    fn place_flat(board: &mut Board, r: u8, c: u8, color: Color) {
        board
            .get_mut(Square::from_rc(r, c))
            .push(Piece::new(color, PieceType::Flat));
    }

    /// Place a capstone on the board at (r, c).
    fn place_cap(board: &mut Board, r: u8, c: u8, color: Color) {
        board
            .get_mut(Square::from_rc(r, c))
            .push(Piece::new(color, PieceType::Cap));
    }

    /// Place a wall on the board at (r, c).
    fn place_wall(board: &mut Board, r: u8, c: u8, color: Color) {
        board
            .get_mut(Square::from_rc(r, c))
            .push(Piece::new(color, PieceType::Wall));
    }

    // -----------------------------------------------------------------------
    // Road detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn road_horizontal_row0() {
        // 5x5 board: White flats spanning row 0, columns 0..5.
        // This should be a north-south road? No -- row 0 to row 0 is just one edge.
        // A horizontal road (same row across all columns) connects west to east.
        let mut board = Board::empty();
        for c in 0..5u8 {
            place_flat(&mut board, 0, c, Color::White);
        }
        // West-to-east road via row 0.
        assert!(check_road(&board, 5, Color::White));
        assert!(!check_road(&board, 5, Color::Black));
    }

    #[test]
    fn road_vertical_col0() {
        // 5x5 board: White flats spanning col 0, rows 0..5.
        // This connects north to south.
        let mut board = Board::empty();
        for r in 0..5u8 {
            place_flat(&mut board, r, 0, Color::White);
        }
        assert!(check_road(&board, 5, Color::White));
        assert!(!check_road(&board, 5, Color::Black));
    }

    #[test]
    fn road_diagonal_not_a_road() {
        // Diagonal path does not count (not orthogonally connected).
        let mut board = Board::empty();
        for i in 0..5u8 {
            place_flat(&mut board, i, i, Color::White);
        }
        // This connects (0,0) to (4,4) diagonally, but pieces are NOT
        // orthogonally connected, so no road.
        assert!(!check_road(&board, 5, Color::White));
    }

    #[test]
    fn road_l_shaped() {
        // L-shaped path on 5x5: row 0 cols 0-2, then col 2 rows 0-4.
        // Connects west to east? No -- col 2 is not the east edge (col 4).
        // Connects north to south: row 0 to row 4 via col 2. Yes!
        let mut board = Board::empty();
        for c in 0..3u8 {
            place_flat(&mut board, 0, c, Color::White);
        }
        for r in 1..5u8 {
            place_flat(&mut board, r, 2, Color::White);
        }
        assert!(check_road(&board, 5, Color::White));
    }

    #[test]
    fn road_not_quite_connected() {
        // 5x5: col 0 rows 0..3 and col 0 row 4 missing. Gap at (3,0).
        let mut board = Board::empty();
        for r in 0..3u8 {
            place_flat(&mut board, r, 0, Color::White);
        }
        place_flat(&mut board, 4, 0, Color::White);
        // Gap at row 3 => no north-south road.
        assert!(!check_road(&board, 5, Color::White));
    }

    #[test]
    fn road_wall_blocks() {
        // 5x5: col 0 fully occupied, but (2,0) is a wall.
        // Walls do NOT count for roads.
        let mut board = Board::empty();
        for r in 0..5u8 {
            if r == 2 {
                place_wall(&mut board, r, 0, Color::White);
            } else {
                place_flat(&mut board, r, 0, Color::White);
            }
        }
        assert!(!check_road(&board, 5, Color::White));
    }

    #[test]
    fn road_capstone_counts() {
        // 5x5: col 0 fully occupied, (2,0) is a capstone. Should still be a road.
        let mut board = Board::empty();
        for r in 0..5u8 {
            if r == 2 {
                place_cap(&mut board, r, 0, Color::White);
            } else {
                place_flat(&mut board, r, 0, Color::White);
            }
        }
        assert!(check_road(&board, 5, Color::White));
    }

    #[test]
    fn road_both_directions() {
        // 5x5: plus sign centered at (2,2). Connects N-S and W-E.
        let mut board = Board::empty();
        for r in 0..5u8 {
            place_flat(&mut board, r, 2, Color::White);
        }
        for c in 0..5u8 {
            if c != 2 {
                place_flat(&mut board, 2, c, Color::White);
            }
        }
        assert!(check_road(&board, 5, Color::White));
    }

    #[test]
    fn road_empty_board() {
        let board = Board::empty();
        assert!(!check_road(&board, 5, Color::White));
        assert!(!check_road(&board, 5, Color::Black));
    }

    #[test]
    fn road_3x3_minimal() {
        // 3x3: col 1 rows 0,1,2 all white flats. North-south road.
        let mut board = Board::empty();
        for r in 0..3u8 {
            place_flat(&mut board, r, 1, Color::White);
        }
        assert!(check_road(&board, 3, Color::White));
    }

    // -----------------------------------------------------------------------
    // Flat win tests
    // -----------------------------------------------------------------------

    #[test]
    fn flat_win_no_komi() {
        let config = GameConfig::standard(5);
        let mut board = Board::empty();
        // 3 white flats, 2 black flats.
        place_flat(&mut board, 0, 0, Color::White);
        place_flat(&mut board, 0, 1, Color::White);
        place_flat(&mut board, 0, 2, Color::White);
        place_flat(&mut board, 1, 0, Color::Black);
        place_flat(&mut board, 1, 1, Color::Black);

        assert_eq!(flat_winner(&board, &config), GameResult::FlatWin(Color::White));
    }

    #[test]
    fn flat_win_black_more() {
        let config = GameConfig::standard(5);
        let mut board = Board::empty();
        place_flat(&mut board, 0, 0, Color::White);
        place_flat(&mut board, 1, 0, Color::Black);
        place_flat(&mut board, 1, 1, Color::Black);

        assert_eq!(flat_winner(&board, &config), GameResult::FlatWin(Color::Black));
    }

    #[test]
    fn flat_win_tie_is_draw() {
        let config = GameConfig::standard(5);
        let mut board = Board::empty();
        place_flat(&mut board, 0, 0, Color::White);
        place_flat(&mut board, 1, 0, Color::Black);

        assert_eq!(flat_winner(&board, &config), GameResult::Draw);
    }

    #[test]
    fn flat_win_with_komi() {
        let mut config = GameConfig::standard(6);
        config.komi = 2; // White gets +2

        let mut board = Board::empty();
        // 3 white, 5 black. With komi: white=5, black=5 => draw.
        for c in 0..3u8 {
            place_flat(&mut board, 0, c, Color::White);
        }
        for c in 0..5u8 {
            place_flat(&mut board, 1, c, Color::Black);
        }
        assert_eq!(flat_winner(&board, &config), GameResult::Draw);
    }

    #[test]
    fn flat_win_with_komi_white_wins() {
        let mut config = GameConfig::standard(6);
        config.komi = 4;

        let mut board = Board::empty();
        // 2 white + 4 komi = 6, vs 5 black. White wins.
        place_flat(&mut board, 0, 0, Color::White);
        place_flat(&mut board, 0, 1, Color::White);
        for c in 0..5u8 {
            place_flat(&mut board, 1, c, Color::Black);
        }
        assert_eq!(flat_winner(&board, &config), GameResult::FlatWin(Color::White));
    }

    #[test]
    fn flat_win_with_half_komi() {
        let mut config = GameConfig::standard(6);
        config.komi = 2;
        config.half_komi = true; // White gets +2.5

        let mut board = Board::empty();
        // 3 white + 2.5 komi = 5.5, vs 5 black. White wins.
        for c in 0..3u8 {
            place_flat(&mut board, 0, c, Color::White);
        }
        for c in 0..5u8 {
            place_flat(&mut board, 1, c, Color::Black);
        }
        assert_eq!(flat_winner(&board, &config), GameResult::FlatWin(Color::White));
    }

    #[test]
    fn flat_win_walls_and_caps_dont_count() {
        let config = GameConfig::standard(5);
        let mut board = Board::empty();
        // White has 1 flat + 1 wall + 1 cap = only 1 flat counts.
        place_flat(&mut board, 0, 0, Color::White);
        place_wall(&mut board, 0, 1, Color::White);
        place_cap(&mut board, 0, 2, Color::White);
        // Black has 2 flats.
        place_flat(&mut board, 1, 0, Color::Black);
        place_flat(&mut board, 1, 1, Color::Black);

        assert_eq!(flat_winner(&board, &config), GameResult::FlatWin(Color::Black));
    }

    // -----------------------------------------------------------------------
    // Opening rule tests
    // -----------------------------------------------------------------------

    #[test]
    fn opening_phase_detection() {
        let config = GameConfig::standard(5);
        let state = GameState::new(config);
        assert!(state.is_opening_phase());
        assert_eq!(state.ply, 0);
        assert_eq!(state.side_to_move, Color::White);
    }

    #[test]
    fn opening_places_opponent_flat() {
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);

        // Ply 0: White places -- should result in Black flat on the board.
        let sq = Square::from_rc(2, 2);
        let undo0 = state.apply_move(Move::Place {
            square: sq,
            piece_type: PieceType::Flat,
        });

        // After ply 0: Black flat at (2,2), Black's stones decremented.
        let top = state.board.get(sq).top.unwrap();
        assert_eq!(top.color(), Color::Black);
        assert_eq!(top.piece_type(), PieceType::Flat);
        // Black stones should be decremented by 1.
        assert_eq!(state.reserves[2], config.stones - 1); // black stones
        assert_eq!(state.reserves[0], config.stones); // white stones unchanged
        assert_eq!(state.side_to_move, Color::Black);
        assert_eq!(state.ply, 1);
        assert!(state.is_opening_phase());

        // Ply 1: Black places -- should result in White flat on the board.
        let sq2 = Square::from_rc(3, 3);
        let undo1 = state.apply_move(Move::Place {
            square: sq2,
            piece_type: PieceType::Flat,
        });

        let top2 = state.board.get(sq2).top.unwrap();
        assert_eq!(top2.color(), Color::White);
        assert_eq!(top2.piece_type(), PieceType::Flat);
        assert_eq!(state.reserves[0], config.stones - 1); // white stones decremented
        assert_eq!(state.reserves[2], config.stones - 1); // black stones unchanged from before
        assert_eq!(state.side_to_move, Color::White);
        assert_eq!(state.ply, 2);
        assert!(!state.is_opening_phase());

        // Undo both moves and verify state is restored.
        state.undo_move(
            Move::Place {
                square: sq2,
                piece_type: PieceType::Flat,
            },
            &undo1,
        );
        assert_eq!(state.ply, 1);
        assert_eq!(state.side_to_move, Color::Black);
        assert!(state.board.get(sq2).is_empty());
        assert_eq!(state.reserves[0], config.stones); // white stones restored

        state.undo_move(
            Move::Place {
                square: sq,
                piece_type: PieceType::Flat,
            },
            &undo0,
        );
        assert_eq!(state.ply, 0);
        assert_eq!(state.side_to_move, Color::White);
        assert!(state.board.get(sq).is_empty());
        assert_eq!(state.reserves[2], config.stones); // black stones restored
    }

    // -----------------------------------------------------------------------
    // Apply/undo round-trip for placements
    // -----------------------------------------------------------------------

    #[test]
    fn apply_undo_place_roundtrip() {
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        let initial_zobrist = state.zobrist;
        let initial_reserves = state.reserves;

        // Opening moves.
        let sq0 = Square::from_rc(0, 0);
        let sq1 = Square::from_rc(4, 4);
        let undo0 = state.apply_move(Move::Place { square: sq0, piece_type: PieceType::Flat });
        let undo1 = state.apply_move(Move::Place { square: sq1, piece_type: PieceType::Flat });

        // Normal moves.
        let sq2 = Square::from_rc(1, 1);
        let undo2 = state.apply_move(Move::Place { square: sq2, piece_type: PieceType::Wall });
        let sq3 = Square::from_rc(2, 2);
        let undo3 = state.apply_move(Move::Place { square: sq3, piece_type: PieceType::Cap });

        // Undo everything.
        state.undo_move(Move::Place { square: sq3, piece_type: PieceType::Cap }, &undo3);
        state.undo_move(Move::Place { square: sq2, piece_type: PieceType::Wall }, &undo2);
        state.undo_move(Move::Place { square: sq1, piece_type: PieceType::Flat }, &undo1);
        state.undo_move(Move::Place { square: sq0, piece_type: PieceType::Flat }, &undo0);

        // Verify state matches initial.
        assert_eq!(state.ply, 0);
        assert_eq!(state.side_to_move, Color::White);
        assert_eq!(state.reserves, initial_reserves);
        assert_eq!(state.zobrist, initial_zobrist);
        assert_eq!(state.board.empty_count(5), 25);
    }

    // -----------------------------------------------------------------------
    // Spread move tests
    // -----------------------------------------------------------------------

    #[test]
    fn spread_simple_one_step() {
        // Set up: 5x5, skip opening by manually configuring state.
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        // Manually place a white flat at (2,2).
        let src = Square::from_rc(2, 2);
        state.board.get_mut(src).push(Piece::new(Color::White, PieceType::Flat));
        state.reserves[0] -= 1; // decrement white stones
        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);

        let initial_zobrist = state.zobrist;

        // Spread 1 piece north (from (2,2) to (1,2)).
        // Template for pickup=1, travel=1: drops = [1].
        let range = state.templates.lookup_range(1, 1);
        let template_id = crate::templates::DropTemplateId(range.base_id);

        let mv = Move::Spread {
            src,
            dir: Direction::North,
            pickup: 1,
            template: template_id,
        };
        let undo = state.apply_move(mv);

        // Source should be empty.
        assert!(state.board.get(src).is_empty());
        // Target (1,2) should have the white flat.
        let target = Square::from_rc(1, 2);
        let top = state.board.get(target).top.unwrap();
        assert_eq!(top, Piece::WhiteFlat);
        assert_eq!(state.board.get(target).height, 1);

        // Undo.
        state.undo_move(mv, &undo);
        assert!(!state.board.get(src).is_empty());
        assert_eq!(state.board.get(src).top.unwrap(), Piece::WhiteFlat);
        assert!(state.board.get(target).is_empty());
        assert_eq!(state.zobrist, initial_zobrist);
    }

    #[test]
    fn spread_multi_step() {
        // 5x5: Stack of [WhiteFlat, BlackFlat, WhiteFlat] at (2,2).
        // Spread 2 east with template [1, 1] (pickup top 2 pieces).
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        let src = Square::from_rc(2, 2);
        // Push bottom-to-top: WF, BF, WF.
        state.board.get_mut(src).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(src).push(Piece::new(Color::Black, PieceType::Flat));
        state.board.get_mut(src).push(Piece::new(Color::White, PieceType::Flat));
        state.reserves[0] -= 2;
        state.reserves[2] -= 1;
        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);

        let initial_zobrist = state.zobrist;

        // Template for pickup=2, travel=2: [1, 1].
        let range = state.templates.lookup_range(2, 2);
        let template_id = DropTemplateId(range.base_id);
        let seq = state.templates.get_sequence(template_id);
        assert_eq!(seq.drops.as_slice(), &[1, 1]);

        let mv = Move::Spread {
            src,
            dir: Direction::East,
            pickup: 2,
            template: template_id,
        };
        let undo = state.apply_move(mv);

        // Source (2,2) should still have the bottom piece (WhiteFlat, the first one pushed).
        assert_eq!(state.board.get(src).top.unwrap(), Piece::WhiteFlat);
        assert_eq!(state.board.get(src).height, 1);
        // Carried: pop top 2 = [WF(top), BF]. Reverse = [BF, WF].
        // Step 1 (drop 1 from front): BF goes to (2,3).
        let t1 = Square::from_rc(2, 3);
        assert_eq!(state.board.get(t1).top.unwrap(), Piece::BlackFlat);
        assert_eq!(state.board.get(t1).height, 1);
        // Step 2 (drop 1 from front): WF goes to (2,4).
        let t2 = Square::from_rc(2, 4);
        assert_eq!(state.board.get(t2).top.unwrap(), Piece::WhiteFlat);
        assert_eq!(state.board.get(t2).height, 1);

        // Undo and verify.
        state.undo_move(mv, &undo);
        assert_eq!(state.board.get(src).height, 3);
        assert_eq!(state.board.get(src).top.unwrap(), Piece::WhiteFlat);
        assert!(state.board.get(t1).is_empty());
        assert!(state.board.get(t2).is_empty());
        assert_eq!(state.zobrist, initial_zobrist);

        // Now test pickup=3, travel=3 going south from (0,2) so there's room.
        let src2 = Square::from_rc(0, 2);
        // Move the stack from (2,2) to (0,2) for the test.
        // Reset board first.
        state.board = Board::empty();
        state.board.get_mut(src2).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(src2).push(Piece::new(Color::Black, PieceType::Flat));
        state.board.get_mut(src2).push(Piece::new(Color::White, PieceType::Flat));
        state.reserves = [config.stones - 2, config.capstones, config.stones - 1, config.capstones];
        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        let initial_zobrist2 = state.zobrist;

        // Template for pickup=3, travel=3: [1, 1, 1].
        let range3 = state.templates.lookup_range(3, 3);
        let template_id3 = DropTemplateId(range3.base_id);
        let seq3 = state.templates.get_sequence(template_id3);
        assert_eq!(seq3.drops.as_slice(), &[1, 1, 1]);

        let mv3 = Move::Spread {
            src: src2,
            dir: Direction::South,
            pickup: 3,
            template: template_id3,
        };
        let undo3 = state.apply_move(mv3);

        // Source (0,2) should be empty.
        assert!(state.board.get(src2).is_empty());
        // Carried: pop top 3 = [WF, BF, WF]. Reverse = [WF, BF, WF].
        // Step 1 (drop 1): WF(bottom of carried) goes to (1,2).
        assert_eq!(state.board.get(Square::from_rc(1, 2)).top.unwrap(), Piece::WhiteFlat);
        // Step 2 (drop 1): BF goes to (2,2).
        assert_eq!(state.board.get(Square::from_rc(2, 2)).top.unwrap(), Piece::BlackFlat);
        // Step 3 (drop 1): WF(top of carried, original top) goes to (3,2).
        assert_eq!(state.board.get(Square::from_rc(3, 2)).top.unwrap(), Piece::WhiteFlat);

        // Undo.
        state.undo_move(mv3, &undo3);
        assert_eq!(state.board.get(src2).height, 3);
        assert_eq!(state.board.get(src2).top.unwrap(), Piece::WhiteFlat);
        assert!(state.board.get(Square::from_rc(1, 2)).is_empty());
        assert!(state.board.get(Square::from_rc(2, 2)).is_empty());
        assert!(state.board.get(Square::from_rc(3, 2)).is_empty());
        assert_eq!(state.zobrist, initial_zobrist2);
    }

    #[test]
    fn spread_capstone_flatten() {
        // 5x5: White capstone at (2,2), Black wall at (2,3).
        // Spread 1 piece east with pickup=1: capstone should flatten the wall.
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        let src = Square::from_rc(2, 2);
        state.board.get_mut(src).push(Piece::new(Color::White, PieceType::Cap));
        state.reserves[1] -= 1;

        let target = Square::from_rc(2, 3);
        state.board.get_mut(target).push(Piece::new(Color::Black, PieceType::Wall));
        state.reserves[2] -= 1;

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        let initial_zobrist = state.zobrist;

        let range = state.templates.lookup_range(1, 1);
        let template_id = DropTemplateId(range.base_id);

        let mv = Move::Spread {
            src,
            dir: Direction::East,
            pickup: 1,
            template: template_id,
        };
        let undo = state.apply_move(mv);

        // Source should be empty.
        assert!(state.board.get(src).is_empty());
        // Target should have: bottom = BlackFlat (was wall, now flattened), top = WhiteCap.
        let target_stack = state.board.get(target);
        assert_eq!(target_stack.height, 2);
        assert_eq!(target_stack.top.unwrap(), Piece::WhiteCap);
        // The below should be Black (the flattened wall).
        assert_eq!(target_stack.below[0], Color::Black);
        // The flattened_wall should be recorded.
        assert_eq!(undo.flattened_wall, Some(target));

        // Undo.
        state.undo_move(mv, &undo);
        assert_eq!(state.board.get(src).top.unwrap(), Piece::WhiteCap);
        assert_eq!(state.board.get(target).top.unwrap(), Piece::BlackWall);
        assert_eq!(state.board.get(target).height, 1);
        assert_eq!(state.zobrist, initial_zobrist);
    }

    #[test]
    fn spread_onto_existing_stack() {
        // 5x5: White flat at (2,2), Black flat at (2,3).
        // Spread 1 east: White flat goes on top of Black flat.
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        let src = Square::from_rc(2, 2);
        state.board.get_mut(src).push(Piece::new(Color::White, PieceType::Flat));
        state.reserves[0] -= 1;

        let target = Square::from_rc(2, 3);
        state.board.get_mut(target).push(Piece::new(Color::Black, PieceType::Flat));
        state.reserves[2] -= 1;

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);

        let range = state.templates.lookup_range(1, 1);
        let template_id = DropTemplateId(range.base_id);

        let mv = Move::Spread {
            src,
            dir: Direction::East,
            pickup: 1,
            template: template_id,
        };
        let undo = state.apply_move(mv);

        // Target should have height 2: BlackFlat at bottom, WhiteFlat on top.
        let target_stack = state.board.get(target);
        assert_eq!(target_stack.height, 2);
        assert_eq!(target_stack.top.unwrap(), Piece::WhiteFlat);
        assert_eq!(target_stack.below[0], Color::Black);

        state.undo_move(mv, &undo);
        assert_eq!(state.board.get(src).height, 1);
        assert_eq!(state.board.get(target).height, 1);
    }

    // -----------------------------------------------------------------------
    // Spread with drop template [2, 1]
    // -----------------------------------------------------------------------

    #[test]
    fn spread_drop_template_2_1() {
        // 5x5: Stack at (0,0) = [WF, BF, WF] (bottom to top).
        // Spread 3 south with template [2, 1].
        // carried (bottom-to-top after reverse): [WF(bottom), BF, WF(top)]
        // But we're picking up 3, and template is [2, 1], so travel = 2.
        // Step 1: drop 2 from bottom => WF, BF go to (1,0).
        //   (1,0) gets: WF (first), then BF on top.
        // Step 2: drop 1 from bottom (remaining) => WF goes to (2,0).
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        let src = Square::from_rc(0, 0);
        state.board.get_mut(src).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(src).push(Piece::new(Color::Black, PieceType::Flat));
        state.board.get_mut(src).push(Piece::new(Color::White, PieceType::Flat));
        state.reserves[0] -= 2;
        state.reserves[2] -= 1;

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        let initial_zobrist = state.zobrist;

        // Template for pickup=3, travel=2: first partition is [1, 2], second is [2, 1].
        let range = state.templates.lookup_range(3, 2);
        assert_eq!(range.count, 2);
        // We want [2, 1] which is the second template.
        let template_id = DropTemplateId(range.base_id + 1);
        let seq = state.templates.get_sequence(template_id);
        assert_eq!(seq.drops.as_slice(), &[2, 1]);

        let mv = Move::Spread {
            src,
            dir: Direction::South,
            pickup: 3,
            template: template_id,
        };
        let undo = state.apply_move(mv);

        // Source should be empty.
        assert!(state.board.get(src).is_empty());

        // (1,0): got 2 pieces from bottom of carried.
        // Carried was [WF, BF, WF] (bottom to top after reverse of popping).
        // Pop order: WF(top), BF, WF(bottom). Reverse: [WF(bottom), BF, WF(top)].
        // Drop 2 from front: carried[0]=WF(bottom), carried[1]=BF.
        // Push WF then BF onto (1,0). So (1,0) has WF at bottom, BF on top.
        let t1 = Square::from_rc(1, 0);
        assert_eq!(state.board.get(t1).height, 2);
        assert_eq!(state.board.get(t1).top.unwrap(), Piece::BlackFlat);
        assert_eq!(state.board.get(t1).below[0], Color::White);

        // (2,0): got 1 piece = carried[2] = WF(top of original stack).
        let t2 = Square::from_rc(2, 0);
        assert_eq!(state.board.get(t2).height, 1);
        assert_eq!(state.board.get(t2).top.unwrap(), Piece::WhiteFlat);

        // Undo and verify.
        state.undo_move(mv, &undo);
        assert_eq!(state.board.get(src).height, 3);
        assert_eq!(state.board.get(src).top.unwrap(), Piece::WhiteFlat);
        assert!(state.board.get(t1).is_empty());
        assert!(state.board.get(t2).is_empty());
        assert_eq!(state.zobrist, initial_zobrist);
    }

    // -----------------------------------------------------------------------
    // Repetition detection
    // -----------------------------------------------------------------------

    #[test]
    fn repetition_detection() {
        // Create a position where moves can repeat.
        // On a 5x5 board, place stones to shuttle back and forth.
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);

        // Skip opening: manually set up.
        state.ply = 2;
        state.side_to_move = Color::White;
        state.reserves[0] -= 1; // white used 1 stone
        state.reserves[2] -= 1; // black used 1 stone

        // White flat at (0,0), Black flat at (4,4).
        state.board.get_mut(Square::from_rc(0, 0)).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(Square::from_rc(4, 4)).push(Piece::new(Color::Black, PieceType::Flat));

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        state.hash_history = vec![state.zobrist];

        let range11 = state.templates.lookup_range(1, 1);
        let tmpl = DropTemplateId(range11.base_id);

        // Cycle: White moves (0,0)->(0,1), Black moves (4,4)->(4,3),
        //        White moves (0,1)->(0,0), Black moves (4,3)->(4,4).
        // After 2 full cycles (8 half-moves), the position repeats 3 times.

        let mut undos = Vec::new();
        let mut moves = Vec::new();

        for _cycle in 0..2 {
            // White: (0,0) -> (0,1) East
            let mv = Move::Spread { src: Square::from_rc(0, 0), dir: Direction::East, pickup: 1, template: tmpl };
            moves.push(mv);
            undos.push(state.apply_move(mv));
            if state.result.is_terminal() { break; }

            // Black: (4,4) -> (4,3) West
            let mv = Move::Spread { src: Square::from_rc(4, 4), dir: Direction::West, pickup: 1, template: tmpl };
            moves.push(mv);
            undos.push(state.apply_move(mv));
            if state.result.is_terminal() { break; }

            // White: (0,1) -> (0,0) West
            let mv = Move::Spread { src: Square::from_rc(0, 1), dir: Direction::West, pickup: 1, template: tmpl };
            moves.push(mv);
            undos.push(state.apply_move(mv));
            if state.result.is_terminal() { break; }

            // Black: (4,3) -> (4,4) East
            let mv = Move::Spread { src: Square::from_rc(4, 3), dir: Direction::East, pickup: 1, template: tmpl };
            moves.push(mv);
            undos.push(state.apply_move(mv));
            if state.result.is_terminal() { break; }
        }

        // After 2 full cycles, the hash should have appeared 3 times => Draw.
        assert_eq!(state.result, GameResult::Draw);

        // Undo all moves and verify we're back to Ongoing.
        for (mv, undo) in moves.iter().rev().zip(undos.iter().rev()) {
            state.undo_move(*mv, undo);
        }
        assert_eq!(state.result, GameResult::Ongoing);
    }

    // -----------------------------------------------------------------------
    // Road win detection through apply_move
    // -----------------------------------------------------------------------

    #[test]
    fn road_win_detected_on_place() {
        // 5x5 board: White has flats on col 0, rows 0..3. Place at (4,0) completes the road.
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);

        // Skip opening.
        state.ply = 2;
        state.side_to_move = Color::White;
        state.reserves[0] -= 4; // white already placed 4 stones

        for r in 0..4u8 {
            state.board.get_mut(Square::from_rc(r, 0)).push(Piece::new(Color::White, PieceType::Flat));
        }
        // Black has a flat somewhere.
        state.board.get_mut(Square::from_rc(2, 4)).push(Piece::new(Color::Black, PieceType::Flat));
        state.reserves[2] -= 1;

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        state.hash_history = vec![state.zobrist];

        // Place white flat at (4,0).
        let mv = Move::Place { square: Square::from_rc(4, 0), piece_type: PieceType::Flat };
        let undo = state.apply_move(mv);

        assert_eq!(state.result, GameResult::RoadWin(Color::White));

        // Undo should restore Ongoing.
        state.undo_move(mv, &undo);
        assert_eq!(state.result, GameResult::Ongoing);
    }

    #[test]
    fn road_win_detected_on_spread() {
        // 5x5: White has flats on col 2, rows 0..4 except row 2.
        // White flat at (2,1). Spread east to (2,2) completes the road.
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        for r in 0..5u8 {
            if r != 2 {
                state.board.get_mut(Square::from_rc(r, 2)).push(Piece::new(Color::White, PieceType::Flat));
            }
        }
        state.board.get_mut(Square::from_rc(2, 1)).push(Piece::new(Color::White, PieceType::Flat));
        state.reserves[0] -= 5;

        // Black somewhere.
        state.board.get_mut(Square::from_rc(0, 4)).push(Piece::new(Color::Black, PieceType::Flat));
        state.reserves[2] -= 1;

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        state.hash_history = vec![state.zobrist];

        let range = state.templates.lookup_range(1, 1);
        let tmpl = DropTemplateId(range.base_id);
        let mv = Move::Spread {
            src: Square::from_rc(2, 1),
            dir: Direction::East,
            pickup: 1,
            template: tmpl,
        };
        let undo = state.apply_move(mv);

        assert_eq!(state.result, GameResult::RoadWin(Color::White));

        state.undo_move(mv, &undo);
        assert_eq!(state.result, GameResult::Ongoing);
    }

    // -----------------------------------------------------------------------
    // Flat win detected when board is full
    // -----------------------------------------------------------------------

    #[test]
    fn flat_win_on_full_board() {
        // 3x3 board (10 stones, 0 caps each).
        let config = GameConfig::standard(3);
        let mut state = GameState::new(config);

        // Skip opening.
        state.ply = 2;
        state.side_to_move = Color::White;

        // Fill 8 of 9 squares.
        // White flats: (0,0), (0,1), (0,2), (1,0), (1,1) = 5
        // Black flats: (1,2), (2,0), (2,1) = 3
        // Leave (2,2) empty.
        let white_sqs = [(0,0), (0,1), (0,2), (1,0), (1,1)];
        let black_sqs = [(1,2), (2,0), (2,1)];
        for &(r,c) in &white_sqs {
            state.board.get_mut(Square::from_rc(r, c)).push(Piece::new(Color::White, PieceType::Flat));
            state.reserves[0] -= 1;
        }
        for &(r,c) in &black_sqs {
            state.board.get_mut(Square::from_rc(r, c)).push(Piece::new(Color::Black, PieceType::Flat));
            state.reserves[2] -= 1;
        }

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        state.hash_history = vec![state.zobrist];

        // But first we need to check: white has 5 in a row on rows 0 and 1.
        // Row 0 is complete (0,0)-(0,1)-(0,2) => west-to-east road for white!
        // That would be a road win, not a flat win. Let's rearrange.
        // Break the road: put a black piece in the middle of row 0.
        // Actually, let me place walls to block roads.
        // Simpler: use non-road-forming pattern.

        // Reset board.
        state.board = Board::empty();
        state.reserves = [config.stones, config.capstones, config.stones, config.capstones];
        state.reserves[0] -= 5;
        state.reserves[2] -= 3;

        // Checkerboard-like pattern that doesn't form roads.
        // W B W
        // B W B
        // W B _
        let pattern = [
            (0, 0, Color::White), (0, 1, Color::Black), (0, 2, Color::White),
            (1, 0, Color::Black), (1, 1, Color::White), (1, 2, Color::Black),
            (2, 0, Color::White), (2, 1, Color::Black),
        ];
        for &(r, c, color) in &pattern {
            state.board.get_mut(Square::from_rc(r, c)).push(Piece::new(color, PieceType::Flat));
        }
        // Reserves: white placed 4, black placed 4.
        state.reserves = [config.stones - 4, config.capstones, config.stones - 4, config.capstones];

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        state.hash_history = vec![state.zobrist];

        // White places flat at (2,2) to fill the board. White flats=5, Black flats=4. White wins.
        let mv = Move::Place { square: Square::from_rc(2, 2), piece_type: PieceType::Flat };
        let undo = state.apply_move(mv);

        // No road should exist in the checkerboard.
        // White flats: (0,0), (0,2), (1,1), (2,0), (2,2) = 5
        // Black flats: (0,1), (1,0), (1,2), (2,1) = 4
        assert_eq!(state.result, GameResult::FlatWin(Color::White));

        state.undo_move(mv, &undo);
        assert_eq!(state.result, GameResult::Ongoing);
    }

    // -----------------------------------------------------------------------
    // Flat win when reserves exhausted
    // -----------------------------------------------------------------------

    #[test]
    fn flat_win_on_reserves_exhausted() {
        // 3x3 with config: 2 stones, 0 caps each (custom config for testing).
        let config = GameConfig {
            size: 3,
            stones: 2,
            capstones: 0,
            carry_limit: 3,
            komi: 0,
            half_komi: false,
        };
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        // White has 1 stone left, black has 1 stone left.
        state.reserves = [1, 0, 1, 0];

        // White flat at (0,0), Black flat at (1,1).
        state.board.get_mut(Square::from_rc(0, 0)).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(Square::from_rc(1, 1)).push(Piece::new(Color::Black, PieceType::Flat));

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        state.hash_history = vec![state.zobrist];

        // White places last stone.
        let mv = Move::Place { square: Square::from_rc(2, 2), piece_type: PieceType::Flat };
        let undo = state.apply_move(mv);

        // White exhausted (0 stones + 0 caps). White flats=2, Black flats=1. White wins.
        assert_eq!(state.reserves[0], 0);
        assert_eq!(state.reserves[1], 0);
        assert_eq!(state.result, GameResult::FlatWin(Color::White));

        state.undo_move(mv, &undo);
        assert_eq!(state.result, GameResult::Ongoing);
        assert_eq!(state.reserves[0], 1);
    }

    // -----------------------------------------------------------------------
    // Zobrist consistency through apply/undo
    // -----------------------------------------------------------------------

    #[test]
    fn zobrist_incremental_matches_full_recompute() {
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);

        // Opening moves.
        let mv0 = Move::Place { square: Square::from_rc(2, 2), piece_type: PieceType::Flat };
        let _u0 = state.apply_move(mv0);
        let expected = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        assert_eq!(state.zobrist, expected, "zobrist mismatch after opening move 0");

        let mv1 = Move::Place { square: Square::from_rc(3, 3), piece_type: PieceType::Flat };
        let _u1 = state.apply_move(mv1);
        let expected = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        assert_eq!(state.zobrist, expected, "zobrist mismatch after opening move 1");

        // Normal placement.
        let mv2 = Move::Place { square: Square::from_rc(0, 0), piece_type: PieceType::Wall };
        let _u2 = state.apply_move(mv2);
        let expected = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        assert_eq!(state.zobrist, expected, "zobrist mismatch after normal wall placement");

        // Black placement.
        let mv3 = Move::Place { square: Square::from_rc(4, 4), piece_type: PieceType::Flat };
        let _u3 = state.apply_move(mv3);
        let expected = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        assert_eq!(state.zobrist, expected, "zobrist mismatch after black flat placement");

        // Spread: move the white flat at (2,2) -- wait, that's a black flat (opening rule).
        // Actually (2,2) has Black flat, (3,3) has White flat.
        // White to move (ply 4). Move the White flat at (3,3) north to (2,3).
        let range = state.templates.lookup_range(1, 1);
        let tmpl = DropTemplateId(range.base_id);
        let mv4 = Move::Spread {
            src: Square::from_rc(3, 3),
            dir: Direction::North,
            pickup: 1,
            template: tmpl,
        };
        let _u4 = state.apply_move(mv4);
        let expected = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        assert_eq!(state.zobrist, expected, "zobrist mismatch after spread move");
    }

    // -----------------------------------------------------------------------
    // GameResult utility
    // -----------------------------------------------------------------------

    #[test]
    fn game_result_is_terminal() {
        assert!(!GameResult::Ongoing.is_terminal());
        assert!(GameResult::RoadWin(Color::White).is_terminal());
        assert!(GameResult::RoadWin(Color::Black).is_terminal());
        assert!(GameResult::FlatWin(Color::White).is_terminal());
        assert!(GameResult::FlatWin(Color::Black).is_terminal());
        assert!(GameResult::Draw.is_terminal());
    }

    // -----------------------------------------------------------------------
    // Road win: mover vs opponent priority
    // -----------------------------------------------------------------------

    #[test]
    fn road_win_mover_takes_priority() {
        // If both players have a road after a move, the mover wins.
        // Set up a 5x5 where White's move creates roads for both.
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        // Black road: col 4, rows 0..5.
        for r in 0..5u8 {
            state.board.get_mut(Square::from_rc(r, 4)).push(Piece::new(Color::Black, PieceType::Flat));
            state.reserves[2] -= 1;
        }

        // White near-road: col 0, rows 0..4 (missing row 4).
        for r in 0..4u8 {
            state.board.get_mut(Square::from_rc(r, 0)).push(Piece::new(Color::White, PieceType::Flat));
            state.reserves[0] -= 1;
        }
        // White flat at (4,1) to spread west to (4,0).
        state.board.get_mut(Square::from_rc(4, 1)).push(Piece::new(Color::White, PieceType::Flat));
        state.reserves[0] -= 1;

        state.zobrist = zobrist::compute_full(&state.board, config.size, state.side_to_move, &state.reserves);
        state.hash_history = vec![state.zobrist];

        // White spreads (4,1) west to (4,0), completing White's north-south road.
        // Black already has a road on col 4.
        let range = state.templates.lookup_range(1, 1);
        let tmpl = DropTemplateId(range.base_id);
        let mv = Move::Spread {
            src: Square::from_rc(4, 1),
            dir: Direction::West,
            pickup: 1,
            template: tmpl,
        };
        let undo = state.apply_move(mv);

        // Both have roads, but White (mover) should win.
        assert_eq!(state.result, GameResult::RoadWin(Color::White));

        state.undo_move(mv, &undo);
        assert_eq!(state.result, GameResult::Ongoing);
    }

    // -----------------------------------------------------------------------
    // New game state sanity
    // -----------------------------------------------------------------------

    #[test]
    fn new_game_state_sanity() {
        for size in 3..=8u8 {
            let config = GameConfig::standard(size);
            let state = GameState::new(config);
            assert_eq!(state.ply, 0);
            assert_eq!(state.side_to_move, Color::White);
            assert_eq!(state.result, GameResult::Ongoing);
            assert_eq!(state.reserves[0], config.stones);
            assert_eq!(state.reserves[1], config.capstones);
            assert_eq!(state.reserves[2], config.stones);
            assert_eq!(state.reserves[3], config.capstones);
            assert_eq!(state.board.empty_count(size), (size as u32) * (size as u32));
            assert!(state.is_opening_phase());
            assert_eq!(state.flat_margin(), 0); // initial margin is 0 (even with 0 komi)
            let _ = state.zobrist; // just check it doesn't panic
        }
    }

    #[test]
    fn flat_margin_calculation() {
        let mut state = GameState::new(GameConfig::standard(5));
        state.ply = 2;
        // 3 white flats, 1 black flat
        state.board.get_mut(Square::from_rc(0, 0)).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(Square::from_rc(0, 1)).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(Square::from_rc(0, 2)).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(Square::from_rc(4, 4)).push(Piece::new(Color::Black, PieceType::Flat));
        
        assert_eq!(state.flat_margin(), 2);

        // Add komi of 2
        state.config.komi = 2;
        assert_eq!(state.flat_margin(), 4);
    }

    // -----------------------------------------------------------------------
    // AC 1.19: 3x3 forced-win regression tests
    // -----------------------------------------------------------------------

    /// Helper: play a PTN move sequence and return the final state.
    fn play_sequence(size: u8, ptn_moves: &[&str]) -> GameState {
        let config = GameConfig::standard(size);
        let mut state = GameState::new(config);
        for ptn_str in ptn_moves {
            let mv = crate::ptn::parse_move(ptn_str, &state)
                .unwrap_or_else(|e| panic!("Failed to parse '{}': {}", ptn_str, e));
            state.apply_move(mv);
        }
        state
    }

    #[test]
    fn forced_win_3x3_line_1() {
        // White road via row 1 (horizontal):
        //   Ply 0: W places B flat at a1 (0,0)
        //   Ply 1: B places W flat at b2 (1,1)
        //   Ply 2: W places W flat at a2 (1,0)
        //   Ply 3: B places B flat at a3 (2,0)
        //   Ply 4: W places W flat at c2 (1,2)
        // White flats at (1,0), (1,1), (1,2) = west-to-east road.
        let state = play_sequence(3, &["a1", "b2", "a2", "a3", "c2"]);
        assert_eq!(state.result, GameResult::RoadWin(Color::White));
    }

    #[test]
    fn forced_win_3x3_line_2() {
        // White road via column 0 (vertical):
        //   Ply 0: W places B flat at b2 (1,1)
        //   Ply 1: B places W flat at a1 (0,0)
        //   Ply 2: W places W flat at a2 (1,0)
        //   Ply 3: B places B flat at c3 (2,2)
        //   Ply 4: W places W flat at a3 (2,0)
        // White flats at (0,0), (1,0), (2,0) = north-to-south road.
        let state = play_sequence(3, &["b2", "a1", "a2", "c3", "a3"]);
        assert_eq!(state.result, GameResult::RoadWin(Color::White));
    }

    #[test]
    fn forced_win_3x3_line_3() {
        // White road via column 1 (vertical):
        //   Ply 0: W places B flat at a1 (0,0)
        //   Ply 1: B places W flat at b2 (1,1)
        //   Ply 2: W places W flat at b1 (0,1)
        //   Ply 3: B places B flat at c3 (2,2)
        //   Ply 4: W places W flat at b3 (2,1)
        // White flats at (0,1), (1,1), (2,1) = north-to-south road.
        let state = play_sequence(3, &["a1", "b2", "b1", "c3", "b3"]);
        assert_eq!(state.result, GameResult::RoadWin(Color::White));
    }

    // -----------------------------------------------------------------------
    // AC 1.20: Undo correctness — apply N random moves, undo all, verify
    // -----------------------------------------------------------------------

    #[test]
    fn undo_n_random_moves_restores_initial_state() {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        for size in 3..=6u8 {
            for seed in 0..5u64 {
                let config = GameConfig::standard(size);
                let initial_state = GameState::new(config);
                let mut state = initial_state.clone();
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed * 100 + size as u64);
                let mut history: Vec<(Move, UndoInfo)> = Vec::new();

                // Play up to 80 random moves (or until game over).
                let max_moves = 80;
                for _ in 0..max_moves {
                    if state.result.is_terminal() {
                        break;
                    }
                    let moves = state.legal_moves();
                    if moves.is_empty() {
                        break;
                    }
                    let idx = rng.random_range(0..moves.len());
                    let mv = moves[idx];
                    let undo = state.apply_move(mv);
                    history.push((mv, undo));
                }

                let num_played = history.len();

                // Undo all moves in reverse order.
                while let Some((mv, undo)) = history.pop() {
                    state.undo_move(mv, &undo);
                }

                // Verify state matches initial state.
                assert_eq!(state.ply, 0, "size={} seed={} played={}", size, seed, num_played);
                assert_eq!(state.side_to_move, Color::White);
                assert_eq!(state.result, GameResult::Ongoing);
                assert_eq!(state.zobrist, initial_state.zobrist);
                assert_eq!(state.reserves, initial_state.reserves);
                assert_eq!(
                    state.board.empty_count(size),
                    (size as u32) * (size as u32),
                    "board should be empty after full undo"
                );
            }
        }
    }
}
