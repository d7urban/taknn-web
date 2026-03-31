# TakNN-Web: Build Specification

Derived from PLAN.md. This is the implementation contract. All module boundaries, data formats, tensor shapes, and acceptance criteria are defined here.

---

## 1) Repository Structure

```
taknn-web/
├── PLAN.md
├── BUILD_SPEC.md
├── engine/                         # Rust workspace
│   ├── Cargo.toml                  # workspace root
│   ├── tak-core/                   # crate: rules, board, moves, formats
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── board.rs            # Board, Square, Stack
│   │   │   ├── piece.rs            # Piece, Color, PieceType
│   │   │   ├── rules.rs            # GameConfig, reserves/capstones per size
│   │   │   ├── state.rs            # GameState, apply_move, undo_move, game result
│   │   │   ├── moves.rs            # Move enum, MoveGen, legal move enumeration
│   │   │   ├── descriptor.rs       # MoveDescriptor, descriptor builder
│   │   │   ├── templates.rs        # DropTemplate tables, per-size precomputation
│   │   │   ├── symmetry.rs         # D4 transforms for boards, squares, moves
│   │   │   ├── tactical.rs         # road-in-1, forced defense, capstone flatten, endgame
│   │   │   ├── zobrist.rs          # Zobrist hashing
│   │   │   ├── tps.rs              # TPS parsing/serialization
│   │   │   ├── ptn.rs              # PTN parsing/serialization
│   │   │   └── tensor.rs           # board-to-tensor encoding for NN input
│   │   └── tests/
│   ├── tak-search/                 # crate: search engine
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── pvs.rs              # iterative deepening PVS
│   │   │   ├── tt.rs               # transposition table
│   │   │   ├── eval.rs             # heuristic eval (CP2), trait for neural eval
│   │   │   ├── ordering.rs         # killer, history, policy-prior ordering
│   │   │   ├── extensions.rs       # road-threat, forced-response extensions
│   │   │   └── time.rs             # time management
│   │   └── tests/
│   ├── tak-data/                   # crate: data generation (native only)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── shard.rs            # binary shard writer/reader
│   │   │   ├── selfplay.rs         # self-play driver
│   │   │   ├── sampling.rs         # thinning, tactical flag evaluation
│   │   │   └── opening.rs          # opening family tracking, D4 canonicalization
│   │   └── tests/
│   └── tak-wasm/                   # crate: wasm-bindgen bindings
│       ├── src/
│       │   └── lib.rs              # JS-facing API
│       └── Cargo.toml
├── train/                          # Python package
│   ├── pyproject.toml
│   └── taknn/
│       ├── __init__.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── encoder.py          # input tensor construction
│       │   ├── trunk.py            # residual trunk with FiLM
│       │   ├── policy.py           # legal-move scorer MLP
│       │   ├── value.py            # WDL + scalar + auxiliary heads
│       │   ├── teacher.py          # teacher model assembly
│       │   └── student.py          # student model assembly
│       ├── data/
│       │   ├── __init__.py
│       │   ├── shard.py            # shard reader (calls Rust via PyO3 or reads binary)
│       │   ├── dataset.py          # manifest, sampling, replay buffer
│       │   ├── augment.py          # D4 augmentation on tensors + targets
│       │   └── loader.py           # async prefetch dataloader
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── elo.py              # Elo harness, match runner
│       │   ├── tactical.py         # tactical suite runner
│       │   ├── gate.py             # promotion gating logic
│       │   └── browser_perf.py     # latency/throughput measurement stubs
│       └── train/
│           ├── __init__.py
│           ├── loop.py             # main training loop
│           ├── distill.py          # teacher→student distillation
│           ├── curriculum.py       # size scheduling, opening family balancing
│           ├── export.py           # ONNX export, INT8 quantization
│           └── config.py           # hyperparameter configs
├── web/                            # Next.js app (Checkpoint 7)
│   ├── package.json
│   ├── next.config.js              # cross-origin isolation headers
│   ├── public/
│   │   └── models/                 # quantized ONNX files, opening book
│   └── src/
│       ├── app/
│       ├── components/
│       │   ├── Board.tsx
│       │   ├── Controls.tsx
│       │   └── GameInfo.tsx
│       ├── engine/
│       │   ├── worker.ts           # Web Worker: search + inference
│       │   ├── wasm.ts             # WASM module loader
│       │   ├── inference.ts        # ORT Web session management
│       │   └── book.ts             # opening book lookup
│       └── lib/
│           ├── game.ts             # game state management
│           └── types.ts            # shared TypeScript types
└── data/                           # gitignored
    ├── shards/                     # training data
    ├── manifests/                  # shard indices
    ├── models/                     # checkpointed weights
    ├── eval/                       # match logs, tactical results
    └── books/                      # opening books
```

---

## 2) Rust Crate: tak-core

### 2.1) Core Types

```rust
// piece.rs
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Color { White = 0, Black = 1 }

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum PieceType { Flat = 0, Wall = 1, Cap = 2 }

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Piece {
    WhiteFlat  = 0,
    WhiteWall  = 1,
    WhiteCap   = 2,
    BlackFlat  = 3,
    BlackWall  = 4,
    BlackCap   = 5,
}
// Piece encodes (Color, PieceType) as color * 3 + piece_type.
// 6 values. Stack layers below top are always Flat, so only need Color there.
```

```rust
// board.rs

/// Fixed-capacity stack. Stores top 8 pieces explicitly + buried counts.
/// Interior pieces (below top) are always flat, so only Color is stored.
#[derive(Clone)]
pub struct Stack {
    /// Top piece, if any. None = empty square.
    pub top: Option<Piece>,
    /// Pieces below top, from second-from-top downward.
    /// Stored as Color only (interior pieces are always flat).
    /// Length 0..=7. Only the top min(height-1, 7) are stored.
    pub below: ArrayVec<Color, 7>,
    /// Count of white flats buried below the explicit layers.
    pub buried_white: u8,
    /// Count of black flats buried below the explicit layers.
    pub buried_black: u8,
    /// Total height of the stack.
    pub height: u8,
}

/// 8x8 board. Squares outside the active NxN region are always empty.
/// Row-major indexing: square(r, c) = board.squares[r * 8 + c].
/// Active region for an NxN game: rows 0..N, cols 0..N.
/// Padding squares (row >= N or col >= N) are permanently empty.
pub struct Board {
    pub squares: [Stack; 64],
}

/// Square index: 0..63, row-major on the 8x8 grid.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Square(pub u8);

impl Square {
    pub fn from_rc(r: u8, c: u8) -> Self { Square(r * 8 + c) }
    pub fn row(self) -> u8 { self.0 / 8 }
    pub fn col(self) -> u8 { self.0 % 8 }
}
```

```rust
// rules.rs

#[derive(Copy, Clone)]
pub struct GameConfig {
    pub size: u8,           // 3..=8
    pub stones: u8,         // per player
    pub capstones: u8,      // per player
    pub carry_limit: u8,    // == size
    pub komi: i8,           // typically 0 or 2 for 6x6
    pub half_komi: bool,    // for 0.5 increments
}

impl GameConfig {
    pub fn standard(size: u8) -> Self {
        let (stones, caps) = match size {
            3 => (10, 0),
            4 => (15, 0),
            5 => (21, 1),
            6 => (30, 1),
            7 => (40, 2),
            8 => (50, 2),
            _ => panic!("unsupported size"),
        };
        GameConfig {
            size, stones, capstones: caps,
            carry_limit: size, komi: 0, half_komi: false,
        }
    }
}
```

```rust
// state.rs

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum GameResult {
    Ongoing,
    RoadWin(Color),
    FlatWin(Color),
    Draw,
}

pub struct GameState {
    pub board: Board,
    pub config: GameConfig,
    pub side_to_move: Color,
    pub ply: u16,
    /// Reserves remaining: [white_stones, white_caps, black_stones, black_caps]
    pub reserves: [u8; 4],
    pub result: GameResult,
    pub zobrist: u64,
    /// History of zobrist hashes for repetition detection.
    pub hash_history: Vec<u64>,
}

impl GameState {
    pub fn new(config: GameConfig) -> Self;
    pub fn apply_move(&mut self, mv: Move);
    pub fn undo_move(&mut self, mv: Move, undo: &UndoInfo);
    pub fn is_opening_phase(&self) -> bool { self.ply < 2 }
    pub fn legal_moves(&self) -> MoveList;
}
```

### 2.2) Move Representation

```rust
// moves.rs

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum Direction { North = 0, East = 1, South = 2, West = 3 }

/// Drop template: how many stones to drop at each step along the ray.
/// For pickup_count k and travel_length t, the template is a sequence
/// [d_1, d_2, ..., d_t] where sum(d_i) = k and each d_i >= 1.
/// template_id indexes into the precomputed table for (carry_limit, k, t).
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct DropTemplateId(pub u16);

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Move {
    Place {
        square: Square,
        piece_type: PieceType,  // Flat, Wall, or Cap
    },
    Spread {
        src: Square,
        dir: Direction,
        pickup: u8,             // 1..=carry_limit
        template: DropTemplateId,
    },
}

/// Move list. Grammar-maximum action count is 32,704 on 8x8.
/// Real positions will typically have 50–500 legal moves on 6x6,
/// but worst-case 8x8 positions with many tall stacks can exceed 1,000.
/// Use Vec, not a fixed-capacity ArrayVec.
pub type MoveList = Vec<Move>;

pub struct MoveGen;
impl MoveGen {
    /// Generate all legal moves for the current side to move.
    /// Moves are returned in canonical order (see §2.8 Canonical Move Enumeration).
    pub fn legal_moves(state: &GameState) -> MoveList;
    /// Generate legal moves for a specific side (for tactical detection).
    /// Same canonical ordering.
    pub fn legal_moves_for(state: &GameState, color: Color) -> MoveList;
}
```

### 2.3) Drop Template Tables

```rust
// templates.rs

/// For a given carry_limit, precompute all valid (pickup_count, travel_length, template)
/// combinations. A drop template for pickup k over distance t is an ordered partition
/// of k into t parts, each >= 1.
///
/// The number of such partitions is C(k-1, t-1) (stars and bars).
///
/// Total movement actions per directed ray of distance d (with carry_limit L):
///   sum over k=1..=L of sum over t=1..=d of C(k-1, t-1)
///
/// 8x8 totals per ray distance d:
///   d=1:   8     d=2:  36    d=3:  92    d=4: 162
///   d=5: 218     d=6: 246    d=7: 254
///
/// Each (carry_limit, pickup_count, travel_length) triple maps to a contiguous
/// range of template IDs. Template ID 0 is reserved for "not a spread move."

pub struct TemplateTable {
    /// For each (k, t): base template_id and count.
    /// Indexed as entries[k - 1][t - 1] for k in 1..=carry_limit, t in 1..=max_travel.
    pub entries: Vec<Vec<TemplateRange>>,
    /// The actual drop sequences, indexed by DropTemplateId.
    pub sequences: Vec<DropSequence>,
}

pub struct TemplateRange {
    pub base_id: u16,
    pub count: u16,
}

/// A drop sequence: how many to drop at each step.
/// Length = travel_length. Sum = pickup_count.
pub struct DropSequence {
    pub drops: ArrayVec<u8, 8>,
}

impl TemplateTable {
    /// Build table for a given carry_limit (== board size).
    pub fn build(carry_limit: u8) -> Self;

    /// Total template count for this carry_limit.
    pub fn total_templates(&self) -> u16;
}

/// Per-size action counts (for regression tests):
///   size  placements  movements   total
///   3x3     9×3=27        320       347
///   4x4    16×3=48      1,744     1,792
///   5x5    25×3=75      5,200     5,275
///   6x6    36×3=108    11,552    11,660
///   7x7    49×3=147    21,664    21,811
///   8x8    64×3=192    32,512    32,704
///
/// These are grammar-maximum counts (empty board, all pieces available).
/// Actual legal moves in any position will be much smaller.
```

### 2.4) Move Descriptor (for Policy Head)

```rust
// descriptor.rs

/// Compact descriptor for one legal move, used as input to the policy MLP.
/// Built by the engine, consumed by the neural network.
#[derive(Clone)]
pub struct MoveDescriptor {
    /// Source square index (0..63). For placements: the target square.
    pub src: u8,
    /// Destination square index (0..63).
    /// For placements: same as src.
    /// For spreads: the final square the moving stack reaches.
    pub dst: u8,
    /// Squares traversed by a spread (excluding src, including dst).
    /// Empty for placements. Length = travel_length.
    pub path: ArrayVec<u8, 7>,

    // Discrete features (indices into learned embedding tables):
    pub move_type: u8,          // 0=placement, 1=spread
    pub piece_type: u8,         // 0=flat, 1=wall, 2=cap, 3=N/A (spread)
    pub direction: u8,          // 0=N, 1=E, 2=S, 3=W, 4=N/A (placement)
    pub pickup_count: u8,       // 0 (placement) or 1..=carry_limit
    pub drop_template_id: u16,  // 0 = N/A (placement)
    pub travel_length: u8,      // 0 (placement) or 1..=size-1

    // Binary flags:
    pub capstone_flatten: bool,
    pub enters_occupied: bool,
    pub opening_phase: bool,
}

/// Build descriptors for all legal moves in a position.
pub fn build_descriptors(state: &GameState, moves: &MoveList) -> Vec<MoveDescriptor>;
```

### 2.5) D4 Symmetry

```rust
// symmetry.rs

/// The 8 elements of the dihedral group D4 on an NxN board.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum D4 {
    Identity     = 0,
    Rot90        = 1,
    Rot180       = 2,
    Rot270       = 3,
    ReflectH     = 4,  // horizontal axis
    ReflectV     = 5,  // vertical axis
    ReflectMain  = 6,  // main diagonal
    ReflectAnti  = 7,  // anti-diagonal
}

impl D4 {
    pub const ALL: [D4; 8] = [ /* all variants */ ];

    /// Transform a square on an NxN board.
    pub fn transform_square(self, sq: Square, size: u8) -> Square;

    /// Transform a direction.
    pub fn transform_direction(self, dir: Direction) -> Direction;

    /// Transform a complete Move (remaps square + direction, preserves pickup/template).
    pub fn transform_move(self, mv: Move, size: u8) -> Move;

    /// Transform a MoveDescriptor.
    pub fn transform_descriptor(self, desc: &MoveDescriptor, size: u8) -> MoveDescriptor;

    /// Inverse of this transform.
    pub fn inverse(self) -> D4;

    /// Compose two transforms.
    pub fn compose(self, other: D4) -> D4;
}

/// Precomputed lookup tables for a given board size.
pub struct D4Tables {
    /// square_map[transform][square] -> transformed square
    pub square_map: [[Square; 64]; 8],
    /// direction_map[transform][direction] -> transformed direction
    pub direction_map: [[Direction; 4]; 8],
}

impl D4Tables {
    pub fn build(size: u8) -> Self;
}
```

### 2.6) Tactical Detection

```rust
// tactical.rs

#[derive(Copy, Clone)]
pub enum TacticalPhase {
    Tactical,       // keep every ply
    SemiTactical,   // keep every 2 plies
    Quiet,          // keep every 4 plies
}

pub struct TacticalFlags {
    pub road_in_1_white: bool,
    pub road_in_1_black: bool,
    pub forced_defense: bool,   // stm has <= 2 blocking replies
    pub capstone_flatten: bool, // any legal cap flatten exists for either side
    pub endgame: bool,          // empty squares <= 2*size or reserves <= size
}

impl TacticalFlags {
    /// Compute tactical flags purely from move generation. No eval needed.
    pub fn compute(state: &GameState) -> Self;

    pub fn phase(&self) -> TacticalPhase {
        if self.road_in_1_white || self.road_in_1_black
            || self.forced_defense || self.capstone_flatten {
            TacticalPhase::Tactical
        } else if self.endgame {
            TacticalPhase::SemiTactical
        } else {
            TacticalPhase::Quiet
        }
    }
}
```

### 2.7) Canonical Move Enumeration

Every subsystem that refers to a move by index (shard policy targets, opening families,
WASM `apply_move(move_index)`, search root distributions) must use the same ordering.
This is the single canonical ordering produced by `MoveGen::legal_moves`.

```rust
// moves.rs — canonical ordering contract

/// MoveGen::legal_moves returns moves in this deterministic order:
///
/// 1. PLACEMENTS (before all movements)
///    Sorted by (square row-major index ASC, piece_type ASC).
///    piece_type order: Flat=0, Wall=1, Cap=2.
///    During opening phase (ply < 2): only opponent Flat placements,
///    same square ordering.
///
/// 2. MOVEMENTS (after all placements)
///    Sorted by (src square row-major ASC, direction ASC, pickup ASC, template_id ASC).
///    Direction order: North=0, East=1, South=2, West=3.
///    Within a (src, dir, pickup) group, template_id is the index from
///    TemplateTable, which enumerates drop partitions in lexicographic order
///    of the drop sequence [d_1, d_2, ..., d_t].
///
/// This ordering is the same in Rust (native + WASM) and is the ordering
/// that policy target move_index values refer to.
/// Python code must either call into Rust via PyO3 or reimplement this
/// exact ordering. The canonical ordering is verified by cross-language
/// round-trip tests (see acceptance criterion 3.13).

/// Grammar-maximum action enumeration (ignoring position legality):
/// Returns all structurally valid moves for a given board size.
/// Used only for regression testing template/count correctness.
/// This is NOT the same as legal_moves on any particular GameState.
pub fn grammar_actions(size: u8) -> Vec<Move>;
```

### 2.8) Repetition Rule

```rust
// state.rs — repetition detection contract

/// Tak uses third-occurrence draw: if the same position (board + side to move)
/// occurs for the third time in a game, the game is drawn.
///
/// Position identity is determined by zobrist hash. hash_history stores the
/// zobrist hash after every move. On apply_move, check if the new hash
/// appears twice already in hash_history; if so, set result = Draw.
///
/// For search: a position that appears twice in hash_history is scored as
/// SCORE_DRAW (0) at the search leaf, regardless of depth. This prevents
/// the engine from entering repetition loops.
///
/// The zobrist hash includes: board contents (piece types and colors at all
/// stack positions), side to move, and reserves remaining. It does NOT
/// include ply count, komi, or hash_history itself.
///
/// Undo restores the previous hash by popping hash_history.
```

### 2.9) Board-to-Tensor Encoding

```rust
// tensor.rs

/// Encode a GameState into the NN input tensor.
///
/// Output shape: [C_in, 8, 8] where C_in is defined below.
///
/// Channel layout (per square on the 8x8 grid):
///
///   Channels  0..6    top piece one-hot (6 channels): WhiteFlat(0), WhiteWall(1),
///                     WhiteCap(2), BlackFlat(3), BlackWall(4), BlackCap(5),
///                     plus is_occupied(6).
///                     All zero if square is empty.
///
///   Channels  7..20   interior stack layers 2 through 8 (7 layers × 2 channels each).
///                     Per interior layer: [is_white_flat, is_black_flat].
///                     Both zero if that layer is empty/absent.
///                     Layer 2 (second from top): channels 7, 8.
///                     Layer 3: channels 9, 10.
///                     Layer 4: channels 11, 12.
///                     Layer 5: channels 13, 14.
///                     Layer 6: channels 15, 16.
///                     Layer 7: channels 17, 18.
///                     Layer 8: channels 19, 20.
///
///   Channels 21..22   buried flat counts (below the 8 explicit layers):
///                     buried_white / 50.0 (ch 21), buried_black / 50.0 (ch 22).
///
/// Total spatial channels: 7 + 14 + 2 = 23
///
/// Global feature planes (constant value broadcast to all 8x8 squares):
///
///   Channel 23        side to move (1.0 = white, 0.0 = black)
///   Channel 24        opening placement phase (1.0 if ply < 2)
///   Channel 25        white stone reserves / max_stones
///   Channel 26        black stone reserves / max_stones
///   Channel 27        white capstone reserves / max_caps (0.0 if size has no caps)
///   Channel 28        black capstone reserves / max_caps (0.0 if size has no caps)
///   Channel 29        komi / 4.0
///   Channel 30        ply count / 200.0
///
/// Total: C_in = 31
///
/// Board size is NOT a spatial channel. It is consumed by FiLM conditioning
/// as a separate integer input: size_id ∈ {0,1,2,3,4,5} for sizes {3,4,5,6,7,8}.
///
/// Padding: squares outside the active NxN region have all-zero spatial channels.
/// Global planes are still broadcast to all 64 squares.

pub const C_IN: usize = 31;

pub struct BoardTensor {
    /// Shape: [C_IN, 8, 8] in CHW order, row-major.
    pub data: [f32; C_IN * 64],
    /// Board size index: 0..5 for sizes 3..8.
    pub size_id: u8,
}

impl BoardTensor {
    pub fn encode(state: &GameState) -> Self;
}
```

---

## 3) Rust Crate: tak-search

### 3.1) Evaluation Trait

```rust
// eval.rs

/// Score in centipawns. Positive = good for side to move.
pub type Score = i32;

pub const SCORE_INF: Score = 30000;
pub const SCORE_ROAD_WIN: Score = 29000;
pub const SCORE_FLAT_WIN: Score = 28000;

/// Trait for position evaluation. Heuristic eval (CP2) and neural eval (CP4+)
/// both implement this.
pub trait Evaluator {
    /// Evaluate a leaf position. Returns score from side-to-move perspective.
    fn evaluate(&self, state: &GameState) -> Score;

    /// Optional: return move ordering scores for legal moves.
    /// Default: no neural ordering (returns None).
    fn move_scores(&self, _state: &GameState, _moves: &MoveList) -> Option<Vec<f32>> {
        None
    }
}

/// Heuristic evaluator for Checkpoint 2.
pub struct HeuristicEval;

impl Evaluator for HeuristicEval {
    fn evaluate(&self, state: &GameState) -> Score {
        // Components:
        // - flat count difference
        // - road connectivity (longest path toward opposite edges)
        // - center control
        // - stack ownership (top pieces)
        // - wall placement quality
        // - capstone mobility
        // - reserve pressure
        // Weights tuned by hand initially.
        todo!()
    }
}
```

### 3.2) Search

```rust
// pvs.rs

pub struct SearchConfig {
    pub max_depth: u8,
    pub max_time_ms: u64,
    pub tt_size_mb: usize,      // default: 64 for browser, 256 for native
}

pub struct SearchResult {
    pub best_move: Move,
    pub score: Score,
    pub depth: u8,
    pub nodes: u64,
    pub pv: Vec<Move>,
    /// Root policy target for training. One entry per legal move, in canonical order.
    /// Produced by softmax over root move scores with temperature T=1.0:
    ///   For each root move i, compute s_i = score from a depth-1 PVS call.
    ///   Normalize: p_i = exp(s_i / C) / sum(exp(s_j / C))
    ///   where C = 100.0 (centipawn scaling factor).
    /// Moves not searched (pruned at root) get probability 0.
    /// This distribution is what gets stored in shard policy_target.
    pub root_policy: Vec<f32>,
}

pub struct PvsSearch<E: Evaluator> {
    pub config: SearchConfig,
    pub eval: E,
    tt: TranspositionTable,
    killers: KillerTable,
    history: HistoryTable,
    nodes: u64,
}

impl<E: Evaluator> PvsSearch<E> {
    pub fn new(config: SearchConfig, eval: E) -> Self;
    pub fn search(&mut self, state: &mut GameState) -> SearchResult;

    // Internal:
    // fn pvs(&mut self, state, alpha, beta, depth, ply) -> Score;
    // fn quiesce(&mut self, state, alpha, beta) -> Score;
}
```

### 3.3) Transposition Table

```rust
// tt.rs

#[derive(Copy, Clone)]
pub enum TTFlag { Exact, LowerBound, UpperBound }

#[derive(Copy, Clone)]
pub struct TTEntry {
    pub key: u32,       // upper 32 bits of zobrist (lower 32 used as index)
    pub score: i16,
    pub depth: u8,
    pub flag: TTFlag,
    pub best_move: Move, // or a compact 16-bit move encoding
}

pub struct TranspositionTable {
    entries: Vec<TTEntry>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self;
    pub fn probe(&self, zobrist: u64) -> Option<&TTEntry>;
    pub fn store(&mut self, zobrist: u64, entry: TTEntry);
    pub fn clear(&mut self);
}
```

---

## 4) Rust Crate: tak-data

### 4.1) Training Record Format

```rust
// shard.rs

/// Shard file format.
///
/// Each shard file begins with a 16-byte header:
///
///   bytes   field
///   ─────   ─────
///   0..3    magic: [u8; 4] = [0x54, 0x4B, 0x4E, 0x4E]  ("TKNN")
///   4..5    format_version: u16 = 1
///   6       endian: u8 = 0 (little-endian; all multi-byte fields are LE)
///   7..15   reserved: [u8; 9] = zeroes
///
/// After the header: a sequence of training records, then the entire
/// payload (header + records) is compressed with zstd level 3.
/// Target shard size: 64–256 MB uncompressed.
///
/// One training record. Serialized as fixed-size header + variable-length data.
///
/// Binary layout per record:
///
///   bytes   field
///   ─────   ─────
///   0..3    record_length: u32 (total bytes including this field)
///   4       board_size: u8 (3..8)
///   5       side_to_move: u8 (0=white, 1=black)
///   6..7    ply: u16
///   8       white_stones: u8
///   9       white_caps: u8
///   10      black_stones: u8
///   11      black_caps: u8
///   12      komi: i8 (signed; 0 or 2 typically)
///   13      half_komi: u8 (0=no, 1=yes; if 1, effective komi = komi + 0.5)
///   14      game_result: u8 (0=white_road, 1=black_road, 2=white_flat,
///                            3=black_flat, 4=draw)
///   15..16  flat_margin: i16 (white flats - black flats at game end)
///   17      search_depth: u8
///   18..21  search_nodes: u32
///   22..25  game_id: u32
///   26..27  source_model_id: u16
///   28      tactical_phase: u8 (0=quiet, 1=semi, 2=tactical)
///   29..30  num_policy_entries: u16 (number of nonzero policy entries, see below)
///   31..36  teacher_wdl: [f16; 3] = 6 bytes (win, draw, loss from white perspective)
///   37..40  teacher_margin: f32 (flat margin prediction, normalized to [-1, 1])
///   41..N   board_data: packed stack representation (see below)
///   N..M    policy_target: sparse entries (see below)
///
/// Board data packing (per square, row-major, only active NxN squares):
///   1 byte: top piece (0..5 for pieces, 255 for empty)
///   1 byte: stack height
///   if height > 1:
///     ceil((min(height-1, 7)) / 4) bytes: interior colors packed 2 bits each
///       (0 = white, 1 = black)
///     if height > 8:
///       1 byte: buried_white
///       1 byte: buried_black
///
/// Policy target: SPARSE format. Only moves with nonzero probability are stored.
/// num_policy_entries × 4 bytes each: (move_index: u16, probability: f16).
/// move_index is the position of this move in the canonical legal-move
/// enumeration (see §2.7). Probabilities sum to 1.0 over stored entries.
/// Typical entry count: 5–30 for focused search results, up to ~200 for
/// shallow/noisy searches. Not bounded by total legal move count.
///
/// Auxiliary training labels (road_threat, block_threat, cap_flatten, endgame)
/// are NOT stored in shards. They are recomputed deterministically at load
/// time from the board state using TacticalFlags::compute() and
/// MoveGen::legal_moves_for(). See §5.4a for the exact labeling functions.
```

### 4.2) Self-Play Driver

```rust
// selfplay.rs

pub struct SelfPlayConfig {
    pub board_size: u8,
    pub search_config: SearchConfig,
    pub temperature_schedule: TemperatureSchedule,
    pub opening_family_quotas: bool,
    pub thinning: bool,
}

pub struct TemperatureSchedule {
    /// Temperature by ply range.
    /// Plies 1-4: sample from underrepresented opening families.
    /// Plies 1-6: add Dirichlet root noise (alpha=0.3, weight=0.25).
    /// Plies 1-8: temperature = 1.0.
    /// Plies 9+: temperature = 0.1 (near-greedy).
    pub noise_plies: u16,       // default: 6
    pub warm_plies: u16,        // default: 8
    pub warm_temp: f32,         // default: 1.0
    pub cool_temp: f32,         // default: 0.1
    pub dirichlet_alpha: f32,   // default: 0.3
    pub dirichlet_weight: f32,  // default: 0.25
}

/// Opening family: D4-canonicalized first 4 plies.
/// Stored as a 4-tuple of move indices under D4 canonical form.
pub struct OpeningFamily([u16; 4]);

impl OpeningFamily {
    /// Canonicalize by trying all 8 D4 transforms and taking the lexicographic minimum.
    pub fn canonicalize(plies: &[Move; 4], size: u8) -> Self;
}
```

---

## 5) Python Package: taknn

### 5.1) Model Architecture — Teacher

```
Input
  board_tensor: [B, 31, 8, 8]          float32
  size_id:      [B]                     int64, values 0..5

FiLM Conditioning
  size_embed:   Embedding(6, 32)        → e: [B, 32]
  Per residual block b:
    gamma_b = Linear(32, 128)(e)        → [B, 128]
    beta_b  = Linear(32, 128)(e)        → [B, 128]

Stem
  Conv2d(31, 128, 3, padding=1)         → [B, 128, 8, 8]
  BatchNorm → ReLU

Trunk: 10 residual blocks
  Each block:
    Conv2d(128, 128, 3, padding=1)
    BatchNorm
    FiLM: x = gamma_b * x + beta_b     (gamma/beta reshaped to [B, 128, 1, 1])
    ReLU
    Conv2d(128, 128, 3, padding=1)
    BatchNorm
    FiLM: x = gamma_b2 * x + beta_b2
    Skip connection + ReLU

Global Pool
  g = mean(trunk_out, dim=[2,3])        → [B, 128]

Spatial Output
  h = trunk_out                         → [B, 128, 8, 8]
  (used to extract h_src, h_dst, path_pool per move)

Policy Head (Legal-Move Scorer)
  For each legal move m in batch element i:
    h_src_m = h[i, :, src_r, src_c]                     → [128]
    h_dst_m = h[i, :, dst_r, dst_c]                     → [128]
    path_pool_m = mean(h[i, :, path_squares])            → [128]  (zero if placement)
    discrete_m = concat(
      Embedding(2,  8)(move_type),                       → [8]
      Embedding(4, 8)(piece_type),                       → [8]    (3 types + N/A)
      Embedding(5, 8)(direction),                        → [8]    (4 dirs + N/A)
      Embedding(9, 16)(pickup_count),                    → [16]   (0..8)
      Embedding(256, 16)(drop_template_id),              → [16]   (0 = N/A)
      Embedding(8, 8)(travel_length),                    → [8]    (0..7)
    )                                                    → [64]
    flags_m = [capstone_flatten, enters_occupied,
               opening_phase]                            → [3]    float

    input_m = concat(g[i], h_src_m, h_dst_m,
                     path_pool_m, discrete_m, flags_m)   → [128+128+128+128+64+3 = 579]
    logit_m = MLP(579 → 256 → 1)                        → scalar

  logits = [logit_m for m in legal_moves]
  policy = softmax(logits)

Value Head
  v_in = concat(g, size_embed)                           → [B, 160]
  v_hidden = Linear(160, 128) → ReLU                     → [B, 128]

  wdl    = Linear(128, 3) → softmax                      → [B, 3]   (win/draw/loss from white POV)
  margin = Linear(128, 1) → tanh                          → [B, 1]   (flat margin, normalized to [-1,1])
  scalar_value = wdl[:, 0] - wdl[:, 2] + 0.5 * margin    → [B, 1]   (derived, not a separate head)

  scalar_value is the single scalar used for:
    - search leaf evaluation (from side-to-move perspective: negate if black)
    - distillation value target
    - shard teacher_margin stores the raw margin prediction
    - shard teacher_wdl stores the raw WDL vector
  No separate "teacher_value" scalar exists. Distillation reconstructs
  scalar_value from teacher_wdl and teacher_margin at load time.

Auxiliary Heads
  aux_in = h (spatial trunk output)                      → [B, 128, 8, 8]
  road_threat  = Conv2d(128, 2, 1) → sigmoid              → [B, 2, 8, 8]  (white/black per square)
  block_threat = Conv2d(128, 2, 1) → sigmoid              → [B, 2, 8, 8]
  cap_flatten  = Conv2d(128, 1, 1) → sigmoid              → [B, 1, 8, 8]
  endgame      = Linear(160, 1) → sigmoid                 → [B, 1]         (from v_in)
```

### 5.2) Model Architecture — Student

Same structure, different dimensions:

```
Channels:          64 (was 128)
Residual blocks:   6  (was 10)
FiLM embed dim:    16 (was 32)

Policy MLP input:  64+64+64+64+64+3 = 323   (embedding dims stay the same)
Policy MLP:        323 → 128 → 1

Value hidden:      Linear(80, 64)            (64 + 16 size embed = 80)
WDL:               Linear(64, 3)
Margin:            Linear(64, 1)
scalar_value:      derived = wdl[0] - wdl[2] + 0.5 * margin   (same formula as teacher)

Auxiliary heads: same structure, 64 input channels instead of 128.
```

### 5.3) Parameter Counts (approximate)

```
Teacher:
  Stem:             31 × 128 × 3 × 3 + 128      ≈    36K
  Trunk:        10 × 2 × (128 × 128 × 3 × 3)    ≈ 2,949K
  FiLM:         10 × 2 × (32 × 128 + 128)        ≈    85K
  Policy MLP:       579 × 256 + 256 × 1           ≈   148K
  Policy embeds:    2×8 + 4×8 + 5×8 + 9×16
                    + 256×16 + 8×8                 ≈     4K
  Value head:       160 × 128 + 128 × 4           ≈    21K
  Aux heads:        128 × 5 + 1                    ≈     1K
  BatchNorm:    10 × 2 × 2 × 128 + 2 × 128       ≈     5K
  Size embed:       6 × 32                         =   192
  ─────────────────────────────────────────────────────────
  Total teacher:                                   ≈ 3.25M parameters

Student:
  Stem:             31 × 64 × 3 × 3 + 64          ≈    18K
  Trunk:         6 × 2 × (64 × 64 × 3 × 3)       ≈   442K
  FiLM:          6 × 2 × (16 × 64 + 64)           ≈    13K
  Policy MLP:       323 × 128 + 128 × 1           ≈    41K
  Policy embeds:                                   ≈     4K
  Value + aux:                                     ≈     6K
  BatchNorm + size embed:                          ≈     2K
  ─────────────────────────────────────────────────────────
  Total student:                                   ≈  526K parameters
  INT8 model size:                                 ≈  526 KB
```

### 5.4) Training Loss

```python
loss = (
    w_policy  * cross_entropy(pred_policy, teacher_policy_target)
  + w_wdl     * cross_entropy(pred_wdl, game_result_onehot)
  + w_margin  * mse(pred_margin, normalized_flat_margin)
  + w_road    * bce(pred_road_threat, road_threat_label)
  + w_block   * bce(pred_block_threat, block_threat_label)
  + w_cap     * bce(pred_cap_flatten, cap_flatten_label)
  + w_endgame * bce(pred_endgame, endgame_label)
)

# Default weights (tune later):
w_policy  = 1.0
w_wdl     = 1.0
w_margin  = 0.5
w_road    = 0.2
w_block   = 0.2
w_cap     = 0.1
w_endgame = 0.1
```

### 5.4a) Auxiliary Label Production

Auxiliary labels are NOT stored in shards. They are recomputed deterministically
at training load time from the board state stored in each shard record.
This avoids bloating the shard format and ensures labels stay consistent
with the current labeling logic even if it is refined.

```python
# train/taknn/data/labels.py

def compute_aux_labels(state: GameState) -> dict:
    """
    Compute auxiliary training labels from a GameState.
    Called by the dataloader after reconstructing state from shard board_data.

    Returns:
        road_threat:  [2, 8, 8] float32 — per-square binary label.
                      road_threat[0, r, c] = 1.0 if white has a legal move
                      from or through (r,c) that creates a road win.
                      road_threat[1, r, c] = same for black.
                      Computed by: for each legal move that is a road win,
                      mark the source square and all destination/path squares.

        block_threat: [2, 8, 8] float32 — per-square binary label.
                      block_threat[color, r, c] = 1.0 if the opponent has a
                      road-in-1, and a legal move by `color` that occupies (r,c)
                      blocks all opponent road-in-1 moves.
                      Requires: enumerate opponent road-in-1 moves, find squares
                      whose occupation blocks all of them.

        cap_flatten:  [1, 8, 8] float32 — per-square binary label.
                      cap_flatten[0, r, c] = 1.0 if any legal move is a
                      capstone-flatten spread whose source is (r,c).
                      Either side's perspective; combined into one plane.

        endgame:      [1] float32 — scalar binary label.
                      1.0 if empty_squares <= 2 * board_size
                      OR either player's stone reserves <= board_size.
    """
```

The labeler depends only on `GameState` + `MoveGen` + `TacticalFlags`, all from
tak-core. Python calls into Rust via PyO3 for the actual computation, or
reimplements the same logic with identical results (verified by cross-language
test, acceptance criterion 3.14).

### 5.5) Distillation Loss (teacher → student)

```python
# scalar_value is derived: wdl[0] - wdl[2] + 0.5 * margin
# teacher_scalar is reconstructed from shard teacher_wdl and teacher_margin.
teacher_scalar = teacher_wdl[0] - teacher_wdl[2] + 0.5 * teacher_margin
student_scalar = student_wdl[0] - student_wdl[2] + 0.5 * student_margin

distill_loss = (
    w_policy_distill * kl_div(student_policy, teacher_policy_from_shard)
  + w_wdl_distill    * kl_div(student_wdl, teacher_wdl_from_shard)
  + w_margin_distill * mse(student_margin, teacher_margin_from_shard)
  + w_game           * cross_entropy(student_wdl, game_result_onehot)
)

# Default weights:
w_policy_distill = 1.0
w_wdl_distill    = 0.7   # soft teacher WDL
w_margin_distill = 0.3   # soft teacher margin
w_game           = 0.3   # hard game outcome

# The scalar_value derivation is deterministic from WDL + margin,
# so matching WDL and margin is sufficient. No separate scalar loss.
```

### 5.6) Training Hyperparameters

```yaml
# train/taknn/train/config.py defaults

optimizer: AdamW
learning_rate: 2e-3           # teacher initial
lr_schedule: cosine_with_warmup
warmup_steps: 1000
weight_decay: 1e-4
batch_size: 1024              # teacher
gradient_accumulation: 1
mixed_precision: fp16

# Student distillation:
student_lr: 1e-3
student_batch_size: 2048

# Replay buffer:
replay_window_size: 5_000_000   # positions
recent_fraction: 0.7
old_fraction: 0.3

# Size quotas in replay (Checkpoint 6):
size_quota:
  4x4: 0.10
  5x5: 0.25
  6x6: 0.55
  3x3: 0.02    # regression only
  7x7: 0.05    # stretch
  8x8: 0.03    # stretch

# D4 augmentation: apply random D4 transform to each sample.
# 8× effective dataset size.
```

### 5.7) ONNX Export

```python
# train/taknn/train/export.py

def export_student_trunk(model, output_path):
    """
    Export the student trunk (stem + residual blocks + value/aux heads) to ONNX.
    The policy MLP is NOT included — it runs in WASM (see below).

    Inputs:
      board_tensor: float32 [1, 31, 8, 8]
      size_id: int64 [1]

    Outputs:
      trunk_spatial: float32 [1, 64, 8, 8]   # h: per-square embeddings
      global_pool: float32 [1, 64]            # g: global board embedding
      wdl: float32 [1, 3]
      margin: float32 [1, 1]
      road_threat: float32 [1, 2, 8, 8]
      block_threat: float32 [1, 2, 8, 8]
      cap_flatten: float32 [1, 1, 8, 8]
      endgame: float32 [1, 1]

    Two export steps:
      1. FP32 ONNX: used for numerical parity verification.
      2. Dynamic INT8 ONNX: used for browser deployment.

    Target file sizes:
      FP32: ~2.1 MB
      INT8: < 600 KB
    """

def export_policy_weights(model, output_path):
    """
    Export the policy MLP weights as a flat binary file.
    The policy MLP runs in WASM, not in ORT, because the number of legal
    moves varies per position and ORT dynamic shapes add overhead.

    Binary format:
      - policy MLP weights: Linear(323, 128) + bias, Linear(128, 1) + bias
      - discrete embedding tables: move_type(2×8), piece_type(4×8),
        direction(5×8), pickup(9×16), template(256×16), travel(8×8)
      All stored as float32, little-endian, in the order listed.
      Total: ~170 KB

    The WASM policy scorer loads these weights once at startup and
    scores legal moves by:
      1. Receiving trunk_spatial [64, 8, 8] and global_pool [64] from ORT.
      2. For each legal move, building the descriptor (from tak-core).
      3. Gathering h_src, h_dst, path_pool from trunk_spatial.
      4. Looking up discrete embeddings.
      5. Concatenating [g, h_src, h_dst, path_pool, embeds, flags] → [323].
      6. Running the 2-layer MLP → scalar logit.
      7. Softmax over all legal-move logits.
    """
```

**Policy execution boundary contract:**

| Component | Runs in | Input | Output |
|-----------|---------|-------|--------|
| Trunk (stem + resblocks + value/aux) | ORT Web (WebGPU or WASM) | `[1, 31, 8, 8]` + `size_id` | `trunk_spatial`, `global_pool`, `wdl`, `margin`, aux |
| Policy MLP | WASM (tak-wasm) | `trunk_spatial`, `global_pool`, legal moves | `Vec<f32>` logits, one per legal move |
| Softmax + move selection | WASM or JS | logits | move index |

The Web Worker orchestrates: call ORT for the trunk, pass results to WASM for
policy scoring, return best move to the main thread.

This split means:
- ORT handles only fixed-shape tensors (no dynamic batch dimension for moves).
- The policy MLP runs in WASM at native-like speed over a small, variable-length loop.
- One round-trip per position, not per move.
- Discrete embedding tables and MLP weights live in WASM linear memory.

---

## 6) WASM Bindings: tak-wasm

The WASM API uses opaque handles for game state (stored in WASM linear memory)
and typed arrays for bulk data transfer. No JSON serialization on the hot path.

```rust
// tak-wasm/src/lib.rs

use wasm_bindgen::prelude::*;

/// Opaque handle to a GameState stored in a global arena inside WASM memory.
/// JS holds only the integer handle; all mutation happens via exported functions.
#[wasm_bindgen]
pub struct GameHandle(u32);

// ── Lifecycle ──

/// Create a new game. Returns an opaque handle.
#[wasm_bindgen]
pub fn new_game(size: u8) -> GameHandle;

/// Create a game from a TPS string. Returns handle.
#[wasm_bindgen]
pub fn from_tps(tps: &str) -> GameHandle;

/// Destroy a game handle and free its memory.
#[wasm_bindgen]
pub fn free_game(handle: GameHandle);

// ── State queries ──

/// Serialize current state to TPS string.
#[wasm_bindgen]
pub fn to_tps(handle: &GameHandle) -> String;

/// Game result: 0=ongoing, 1=white_road, 2=black_road,
/// 3=white_flat, 4=black_flat, 5=draw.
#[wasm_bindgen]
pub fn game_result(handle: &GameHandle) -> u8;

/// Current ply.
#[wasm_bindgen]
pub fn ply(handle: &GameHandle) -> u16;

/// Side to move: 0=white, 1=black.
#[wasm_bindgen]
pub fn side_to_move(handle: &GameHandle) -> u8;

/// Size ID (0..5) for FiLM conditioning.
#[wasm_bindgen]
pub fn size_id(handle: &GameHandle) -> u8;

// ── Move generation (canonical order, see §2.7) ──

/// Number of legal moves in the current position.
#[wasm_bindgen]
pub fn legal_move_count(handle: &GameHandle) -> u32;

/// Apply move by index into the canonical legal-move list.
#[wasm_bindgen]
pub fn apply_move(handle: &GameHandle, move_index: u32);

/// Undo the last move.
#[wasm_bindgen]
pub fn undo_move(handle: &GameHandle);

/// Get human-readable move notation (PTN) for a move index.
#[wasm_bindgen]
pub fn move_to_ptn(handle: &GameHandle, move_index: u32) -> String;

/// Parse a PTN move string and return its canonical index, or -1 if invalid.
#[wasm_bindgen]
pub fn ptn_to_move_index(handle: &GameHandle, ptn: &str) -> i32;

// ── Tensor encoding (for NN trunk input) ──

/// Encode board as NN input tensor. Writes into caller-provided Float32Array.
/// Buffer must be at least 31 * 8 * 8 = 1984 floats.
/// Layout: CHW, channels 0..30, row-major per channel (see §2.9).
#[wasm_bindgen]
pub fn encode_board(handle: &GameHandle, out: &mut [f32]);

// ── Policy scoring (runs the policy MLP in WASM) ──

/// Load policy MLP weights from a binary buffer (see export_policy_weights format).
/// Call once at startup.
#[wasm_bindgen]
pub fn load_policy_weights(weights: &[u8]);

/// Score all legal moves using the policy MLP.
///
/// Inputs:
///   trunk_spatial: Float32Array, length 64*8*8 = 4096 (CHW, from ORT trunk output)
///   global_pool:   Float32Array, length 64 (from ORT trunk output)
///
/// Writes softmax probabilities into `out`, one per legal move in canonical order.
/// Buffer must be at least legal_move_count() floats.
#[wasm_bindgen]
pub fn score_moves(
    handle: &GameHandle,
    trunk_spatial: &[f32],
    global_pool: &[f32],
    out: &mut [f32],
);

// ── Heuristic search (no NN) ──

/// Run heuristic PVS search. Returns best move index (canonical order).
#[wasm_bindgen]
pub fn search_heuristic(handle: &GameHandle, time_ms: u32, tt_size_mb: u32) -> u32;

// ── Tactical queries ──

/// Compute tactical flags. Returns packed byte:
/// bit 0: road_in_1_white, bit 1: road_in_1_black,
/// bit 2: forced_defense, bit 3: capstone_flatten, bit 4: endgame.
#[wasm_bindgen]
pub fn tactical_flags(handle: &GameHandle) -> u8;

// ── Utilities ──

/// Parse a full PTN game string. Returns a JSON array of
/// {move_ptn, move_index} objects. This is UI-facing, not hot-path.
#[wasm_bindgen]
pub fn parse_ptn_game(ptn: &str, size: u8) -> JsValue;

/// Apply D4 transform to a board tensor in-place.
/// Buffer: 31 * 8 * 8 = 1984 floats.
#[wasm_bindgen]
pub fn d4_transform_tensor(tensor: &mut [f32], transform: u8, size: u8);
```

---

## 7) Acceptance Criteria

### Checkpoint 1 — Full Legal Engine + Browser UI

| # | Criterion | How to verify |
|---|-----------|---------------|
| 1.1 | `GameConfig::standard(n)` returns correct reserves/capstones for all n in 3..8 | Unit test |
| 1.2a | `grammar_actions(n)` returns correct grammar-max counts per size: 3→347, 4→1792, 5→5275, 6→11660, 7→21811, 8→32704 | Unit test (synthetic enumeration, no GameState) |
| 1.2b | `MoveGen::legal_moves` on a fresh GameState (ply=0) returns only opponent flat placements: 3→9, 4→16, 5→25, 6→36, 7→49, 8→64 | Unit test (opening rule applied) |
| 1.3 | Opening rule enforced: plies 0 and 1 allow only opponent flat placement | Unit test |
| 1.4 | Capstone flatten: capstone can flatten a wall only as the final step of a spread where the capstone is on top and pickup=1 for that final drop | Unit tests covering legal and illegal flatten attempts |
| 1.5 | Road detection: correctly identifies row/column-spanning connected paths of flats+capstones for both colors | Unit tests on hand-constructed boards with roads, near-roads, and non-roads |
| 1.6 | Flat win: when board is full or either player exhausts reserves, correct flat counting including komi | Unit test |
| 1.7 | Stack carry limit enforced: cannot pick up more than N stones on NxN board | Unit test |
| 1.8 | TPS round-trip: `to_tps(from_tps(s)) == s` for a corpus of 100+ positions | Property test |
| 1.9 | PTN parsing: parse a corpus of 50+ real game PTN files and verify final position matches TPS | Integration test |
| 1.10 | D4 transform correctness: for all 8 transforms, `transform(transform_inverse(board)) == board` | Property test on random boards |
| 1.11 | D4 move transform: legal moves on transformed board == transformed legal moves on original board | Property test |
| 1.12 | Move descriptor builder: descriptors for all legal moves match the move they describe (round-trip through descriptor fields) | Property test |
| 1.13 | Drop template tables match expected counts per distance (d=1→8, d=2→36, ..., d=7→254) for carry_limit=8 | Unit test |
| 1.14 | Tensor encoding: non-active squares (outside NxN) are all-zero in spatial channels | Unit test |
| 1.15 | Tensor encoding round-trip: encode a position, decode it back (or verify key properties), confirm consistency | Integration test |
| 1.16 | WASM build: `wasm-pack build` succeeds, produces <1 MB .wasm file | Build test |
| 1.17 | Browser UI: can start a game on each size 3-8, place pieces, make moves, and see game result | Manual test |
| 1.18 | Zobrist hashing: no collisions in a set of 10,000 random positions (statistical test) | Property test |
| 1.19 | 3x3 regression: Player 1 (white) can win from the initial position. Verify at least 3 known forced-win lines | Regression test |
| 1.20 | Undo correctness: apply N random legal moves, undo all N, verify state equals initial state | Property test |

### Checkpoint 2 — Heuristic Search Bot

| # | Criterion | How to verify |
|---|-----------|---------------|
| 2.1 | PVS returns a legal move for every non-terminal position | Fuzz test on 10,000 random positions |
| 2.2 | Search finds road-in-1 within depth 1 on all sizes | Unit test on constructed positions |
| 2.3 | Search finds road-in-2 (forced) within depth 3 on 5x5 and 6x6 | Unit test on constructed positions |
| 2.4 | TT hit rate > 30% at depth 6 on 6x6 midgame positions | Logged metric |
| 2.5 | Search reaches depth 6 within 1 second on 6x6 midgame positions (native) | Benchmark |
| 2.6 | Search reaches depth 4 within 2 seconds on 6x6 midgame positions (WASM, Chrome) | Benchmark |
| 2.7 | Heuristic eval prefers positions with more friendly top-flats | Unit test |
| 2.8 | Heuristic eval gives bonus for road connectivity | Unit test |
| 2.9 | Move ordering: TT move searched first, then captures/threats, then quiet | Logged metric: first-move cutoff rate > 60% at depth >= 4 |
| 2.10 | Web Worker search: UI remains responsive (no jank) during 5-second search | Manual test |
| 2.11 | Bot beats random player >99% on 5x5 at 1s/move | Automated match |
| 2.12 | Tactical extensions: search extends when road-in-1 exists for either side | Logged metric |

### Checkpoint 3 — Data Pipeline + Trainer + Evaluator

| # | Criterion | How to verify |
|---|-----------|---------------|
| 3.1 | Shard writer produces valid zstd-compressed binary; reader reconstructs identical records | Round-trip test on 10,000 records |
| 3.2 | Tactical thinning: quiet positions sampled at 1/4 rate, tactical at 1/1 rate, measured over 1000 games | Statistical test |
| 3.3 | D4 augmentation: augmented board tensor + policy target, when inverse-transformed, matches original | Property test |
| 3.4 | Teacher model forward pass: input [B=4, 31, 8, 8] + size_id [B=4] → policy logits for legal moves, WDL [B,3], margin [B,1], aux heads all correct shapes | Shape test |
| 3.5 | Student model forward pass: same as 3.4 with student dimensions | Shape test |
| 3.6 | Training loop runs 100 steps on synthetic data without crash, loss decreases | Smoke test |
| 3.7 | Replay buffer respects size quotas within ±5% of target fractions | Unit test |
| 3.8 | Elo harness: runs a 50-game match between two random players, computes Elo difference with confidence interval | Integration test |
| 3.9 | Tactical suite runner: loads a suite of 20+ positions, runs model inference, reports accuracy | Integration test |
| 3.10 | Promotion gate: correctly blocks a candidate that regresses on tactical suite by >5% | Unit test on mock data |
| 3.11 | Mixed precision training: fp16 forward/backward pass produces finite loss on real data | Smoke test |
| 3.12 | Dataloader throughput: > 10,000 positions/sec on a single CPU core | Benchmark |
| 3.13 | Canonical move ordering: Rust native, WASM, and Python (PyO3 or reimplemented) produce identical move lists for 1,000 random positions across all sizes | Cross-language round-trip test |
| 3.14 | Auxiliary label parity: Python labeler and Rust TacticalFlags produce identical road_threat, block_threat, cap_flatten, endgame labels for 1,000 random positions | Cross-language test |
| 3.15 | Shard format: reader rejects files with wrong magic number or unsupported version | Unit test |

### Checkpoint 4 — Strong 4x4 / Competent 5x5

| # | Criterion | How to verify |
|---|-----------|---------------|
| 4.1 | Teacher net on 4x4: Elo > Anchor C (strong heuristic PVS) at 1s/move | Elo match, 200+ games |
| 4.2 | Student net on 4x4: Elo > Anchor B (shallow heuristic PVS) at 0.5s/move | Elo match, 200+ games |
| 4.3 | Teacher net on 5x5: Elo > Anchor B at 1s/move | Elo match, 200+ games |
| 4.4 | Policy-guided move ordering: teacher search NPS improves >20% vs heuristic-only ordering | Benchmark |
| 4.5 | Distillation: student policy KL-divergence from teacher < 0.5 nats on held-out 4x4 positions | Metric |
| 4.6 | Browser inference: student model eval latency < 10ms in Chrome on reference machine | Benchmark |
| 4.7a | FP32 ONNX export: trunk outputs match PyTorch within 1e-4 on 100 test positions | Numerical test |
| 4.7b | INT8 ONNX export: WDL prediction agrees with FP32 on >95% of 100 test positions (argmax match); margin MSE < 0.01 | Quality regression test |
| 4.8 | Training loss converges: final policy loss < 3.0, WDL loss < 0.8 | Training log |
| 4.9 | WASM policy scorer: `score_moves` output matches PyTorch policy head within 1e-3 on 100 test positions (same trunk outputs) | Numerical parity test |

### Checkpoint 5 — Capstone-Competent 5x5

| # | Criterion | How to verify |
|---|-----------|---------------|
| 5.1 | Teacher net on 5x5: Elo > Anchor C at 1s/move | Elo match |
| 5.2 | Student net on 5x5: Elo > Anchor B at 0.5s/move in browser | Elo match |
| 5.3 | Tactical suite accuracy on capstone-flatten positions: > 70% | Suite runner |
| 5.4 | Tactical suite accuracy on road-win-in-1 positions: > 90% | Suite runner |
| 5.5 | Tactical suite accuracy on forced-road-block positions: > 75% | Suite runner |
| 5.6 | Auxiliary road-threat head: AUC > 0.85 on held-out data | Metric |
| 5.7 | No 4x4 regression: teacher 4x4 Elo does not drop by more than 50 Elo | Elo match |
| 5.8 | Student INT8 model size < 600 KB | File size check |
| 5.9 | Browser inference + search: median move time < 2s on 5x5 at reasonable quality | Benchmark |

### Checkpoint 6 — 6x6 Mainline Push

| # | Criterion | How to verify |
|---|-----------|---------------|
| 6.1 | Teacher net on 6x6: Elo > Anchor C + 400 at 5s/move | Elo match, 400+ games |
| 6.2 | Student net on 6x6: Elo > Anchor C at 2s/move in browser | Elo match |
| 6.3 | Tactical suite accuracy (all categories combined) on 6x6: > 80% | Suite runner |
| 6.4 | Road-win-in-1 accuracy on 6x6: > 95% | Suite runner |
| 6.5 | Opening diversity: at least 50 distinct D4-canonical opening families observed in last 1000 training games | Metric |
| 6.6 | No 5x5 regression: teacher 5x5 Elo does not drop by more than 75 Elo | Elo match |
| 6.7 | Opening book generated: covers ≥ 100 positions at ply ≤ 8, each with ≥ 10 support games | Book stats |
| 6.8 | Replay buffer: 6x6 fraction within 50–60% of total window | Metric |
| 6.9 | Training throughput: > 5,000 positions/sec on 3090 with mixed precision | Benchmark |
| 6.10 | Gating: at least 3 successful promotions during the 6x6 push, each passing tactical + latency gates | Promotion log |

### Checkpoint 7 — Production Browser Player

| # | Criterion | How to verify |
|---|-----------|---------------|
| 7.1 | Next.js app builds and deploys to Vercel without errors | CI/CD |
| 7.2 | WASM + ONNX model loads in < 3 seconds on 50 Mbps connection | Browser test |
| 7.3 | SharedArrayBuffer available (cross-origin isolation headers set correctly) | Browser console check |
| 7.4 | Game playable on all sizes 3–8: start, play moves, undo, see result | Manual test |
| 7.5 | Bot plays a legal move within 5 seconds on 6x6 in Chrome on mid-range laptop | Benchmark |
| 7.6 | Opening book consulted: bot follows book moves in early game when available | Logged metric |
| 7.7 | WebGPU path used when available; falls back to WASM without error | Browser test on Chrome + Firefox |
| 7.8 | PTN/TPS import/export works from browser UI | Manual test |
| 7.9 | No memory leaks: 100-game autoplay session, memory stays < 500 MB | Browser profiler |
| 7.10 | Lighthouse performance score > 80 on initial page load | Lighthouse audit |

### Checkpoint 8 — Optional 7x7 / 8x8 (Stretch)

| # | Criterion | How to verify |
|---|-----------|---------------|
| 8.1 | Teacher on 7x7: Elo > Anchor C at 5s/move | Elo match |
| 8.2 | Teacher on 8x8: Elo > Anchor B at 10s/move | Elo match |
| 8.3 | Student on 7x7: playable in browser within 10s/move | Benchmark |
| 8.4 | No 6x6 regression: Elo drop < 50 | Elo match |
| 8.5 | FiLM conditioning transfers: 7x7/8x8 training converges 2× faster starting from 6x6 checkpoint vs random init | Training log comparison |

---

## 8) Build Commands

```bash
# Rust engine (native)
cd engine && cargo build --release

# Rust engine (WASM)
cd engine/tak-wasm && wasm-pack build --target web --release

# Python training
cd train && pip install -e ".[dev]"
python -m taknn.train.loop --config configs/teacher_4x4.yaml

# Self-play data generation (native Rust)
cd engine && cargo run --release --bin selfplay -- --size 6 --games 10000 --output ../data/shards/

# Elo evaluation
python -m taknn.eval.elo --model models/teacher_v3.pt --anchor heuristic --size 6 --games 200

# ONNX export
python -m taknn.train.export --model models/student_v3.pt --output web/public/models/student.onnx --quantize int8

# Web app
cd web && npm install && npm run dev

# Vercel deployment
cd web && vercel --prod
```

---

## 9) Dependency Summary

### Rust Crates

```toml
# engine/Cargo.toml workspace dependencies
[workspace.dependencies]
arrayvec = "0.7"
rand = "0.8"
rand_xoshiro = "0.6"     # fast deterministic RNG for self-play
serde = { version = "1", features = ["derive"] }
serde_json = "1"
zstd = "0.13"
wasm-bindgen = "0.2"
js-sys = "0.3"
```

### Python

```toml
# train/pyproject.toml
[project]
dependencies = [
    "torch>=2.1",
    "numpy>=1.24",
    "onnx>=1.14",
    "onnxruntime>=1.16",
    "zstandard>=0.21",
    "tqdm",
    "pyyaml",
]
```

### Web

```json
// web/package.json (key dependencies)
{
    "next": "^14",
    "react": "^18",
    "onnxruntime-web": "^1.17",
    "typescript": "^5"
}
```
