//! WASM bindings for the Tak engine, exposing a JS-friendly API.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use tak_core::board::Square;
use tak_core::moves::Move;
use tak_core::piece::PieceType;
use tak_core::rules::GameConfig;
use tak_core::state::{GameResult, GameState};
use tak_core::tps;
use tak_core::ptn;

/// Install panic hook so WASM panics produce readable console errors.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// JS-facing game wrapper
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct TakGame {
    state: GameState,
    move_history: Vec<Move>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveInfo {
    pub index: usize,
    pub ptn: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameInfo {
    pub size: u8,
    pub ply: u16,
    pub side_to_move: String,
    pub result: String,
    pub tps: String,
    pub white_stones: u8,
    pub white_caps: u8,
    pub black_stones: u8,
    pub black_caps: u8,
}

#[wasm_bindgen]
impl TakGame {
    /// Create a new game with the given board size (3-8).
    #[wasm_bindgen(constructor)]
    pub fn new(size: u8) -> Result<TakGame, JsError> {
        if !(3..=8).contains(&size) {
            return Err(JsError::new(&format!("unsupported board size: {}", size)));
        }
        Ok(TakGame {
            state: GameState::new(GameConfig::standard(size)),
            move_history: Vec::new(),
        })
    }

    /// Create a game from a TPS string.
    #[wasm_bindgen(js_name = "fromTps")]
    pub fn from_tps(tps_str: &str) -> Result<TakGame, JsError> {
        let state = tps::from_tps(tps_str).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(TakGame {
            state,
            move_history: Vec::new(),
        })
    }

    /// Get game info as a JSON object.
    #[wasm_bindgen(js_name = "getInfo")]
    pub fn get_info(&self) -> Result<JsValue, JsError> {
        let info = GameInfo {
            size: self.state.config.size,
            ply: self.state.ply,
            side_to_move: match self.state.side_to_move {
                tak_core::piece::Color::White => "white".into(),
                tak_core::piece::Color::Black => "black".into(),
            },
            result: format_result(self.state.result),
            tps: tps::to_tps(&self.state),
            white_stones: self.state.reserves[0],
            white_caps: self.state.reserves[1],
            black_stones: self.state.reserves[2],
            black_caps: self.state.reserves[3],
        };
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get list of legal moves as JSON array of {index, ptn}.
    #[wasm_bindgen(js_name = "legalMoves")]
    pub fn legal_moves(&self) -> Result<JsValue, JsError> {
        let moves = self.state.legal_moves();
        let infos: Vec<MoveInfo> = moves
            .iter()
            .enumerate()
            .map(|(i, &mv)| MoveInfo {
                index: i,
                ptn: ptn::format_move(mv, &self.state),
            })
            .collect();
        serde_wasm_bindgen::to_value(&infos).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Apply a move by its index in the legal moves list.
    #[wasm_bindgen(js_name = "applyMoveIndex")]
    pub fn apply_move_index(&mut self, index: usize) -> Result<(), JsError> {
        let moves = self.state.legal_moves();
        let mv = moves
            .get(index)
            .ok_or_else(|| JsError::new(&format!("move index {} out of range", index)))?;
        let mv = *mv;
        self.state.apply_move(mv);
        self.move_history.push(mv);
        Ok(())
    }

    /// Apply a move by PTN string.
    #[wasm_bindgen(js_name = "applyMovePtn")]
    pub fn apply_move_ptn(&mut self, ptn_str: &str) -> Result<(), JsError> {
        let mv =
            ptn::parse_move(ptn_str, &self.state).map_err(|e| JsError::new(&e.to_string()))?;
        // Verify the move is legal.
        let legal = self.state.legal_moves();
        if !legal.contains(&mv) {
            return Err(JsError::new(&format!("illegal move: {}", ptn_str)));
        }
        self.state.apply_move(mv);
        self.move_history.push(mv);
        Ok(())
    }

    /// Undo the last move. Returns false if no moves to undo.
    pub fn undo(&mut self) -> bool {
        if self.move_history.is_empty() {
            return false;
        }
        // Rebuild from scratch (simpler than maintaining undo stack through WASM boundary).
        let size = self.state.config.size;
        let moves = self.move_history.clone();
        self.state = GameState::new(GameConfig::standard(size));
        self.move_history.clear();
        for &mv in &moves[..moves.len() - 1] {
            self.state.apply_move(mv);
            self.move_history.push(mv);
        }
        true
    }

    /// Get the board state as a flat array for rendering.
    /// Returns a JSON array of 64 square objects (8x8 grid, row-major).
    /// Each square: { pieces: [{color, type}], active: bool }
    #[wasm_bindgen(js_name = "getBoard")]
    pub fn get_board(&self) -> Result<JsValue, JsError> {
        let size = self.state.config.size;
        let mut squares: Vec<SquareInfo> = Vec::with_capacity(64);

        for r in 0..8u8 {
            for c in 0..8u8 {
                let active = r < size && c < size;
                let sq = Square::from_rc(r, c);
                let stack = self.state.board.get(sq);
                let mut pieces = Vec::new();

                if active && !stack.is_empty() {
                    // Build pieces bottom to top.
                    // Buried pieces (order unknown).
                    for _ in 0..stack.buried_white {
                        pieces.push(PieceInfo {
                            color: "white".into(),
                            piece_type: "flat".into(),
                        });
                    }
                    for _ in 0..stack.buried_black {
                        pieces.push(PieceInfo {
                            color: "black".into(),
                            piece_type: "flat".into(),
                        });
                    }
                    // Explicit interior layers (bottom to top = reversed below array).
                    for &color in stack.below.iter().rev() {
                        pieces.push(PieceInfo {
                            color: match color {
                                tak_core::piece::Color::White => "white".into(),
                                tak_core::piece::Color::Black => "black".into(),
                            },
                            piece_type: "flat".into(),
                        });
                    }
                    // Top piece.
                    if let Some(top) = stack.top {
                        pieces.push(PieceInfo {
                            color: match top.color() {
                                tak_core::piece::Color::White => "white".into(),
                                tak_core::piece::Color::Black => "black".into(),
                            },
                            piece_type: match top.piece_type() {
                                PieceType::Flat => "flat".into(),
                                PieceType::Wall => "wall".into(),
                                PieceType::Cap => "cap".into(),
                            },
                        });
                    }
                }

                squares.push(SquareInfo { pieces, active });
            }
        }

        serde_wasm_bindgen::to_value(&squares).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get TPS string for the current position.
    #[wasm_bindgen(js_name = "getTps")]
    pub fn get_tps(&self) -> String {
        tps::to_tps(&self.state)
    }

    /// Get the move history as PTN.
    #[wasm_bindgen(js_name = "getMoveHistory")]
    pub fn get_move_history(&self) -> String {
        ptn::format_game(&self.state.config, &self.move_history)
    }

    /// Check if the game is over.
    #[wasm_bindgen(js_name = "isGameOver")]
    pub fn is_game_over(&self) -> bool {
        self.state.result.is_terminal()
    }

    /// Get the board size.
    pub fn size(&self) -> u8 {
        self.state.config.size
    }

    /// Get current ply.
    pub fn ply(&self) -> u16 {
        self.state.ply
    }

    /// Run heuristic search and return the best move + info.
    /// `max_depth` and `time_ms` control search limits.
    #[wasm_bindgen(js_name = "searchMove")]
    pub fn search_move(&mut self, max_depth: u8, time_ms: u32) -> Result<JsValue, JsError> {
        use tak_search::eval::HeuristicEval;
        use tak_search::pvs::{PvsSearch, SearchConfig};

        if self.state.result.is_terminal() {
            return Err(JsError::new("game is already over"));
        }

        let config = SearchConfig {
            max_depth,
            max_time_ms: time_ms as u64,
            tt_size_mb: 4,
        };
        let mut search = PvsSearch::new(config, HeuristicEval);
        let result = search.search(&mut self.state);

        let best_ptn = result.best_move.map(|mv| ptn::format_move(mv, &self.state));
        let pv_ptn: Vec<String> = result.pv.iter().map(|mv| ptn::format_move(*mv, &self.state)).collect();

        let info = SearchResultInfo {
            best_move: best_ptn.unwrap_or_default(),
            score: result.score,
            depth: result.depth,
            nodes: result.nodes as u32,
            pv: pv_ptn,
            tt_hits: result.tt_hits as u32,
        };

        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run search and apply the best move. Returns search info.
    #[wasm_bindgen(js_name = "botMove")]
    pub fn bot_move(&mut self, max_depth: u8, time_ms: u32) -> Result<JsValue, JsError> {
        use tak_search::eval::HeuristicEval;
        use tak_search::pvs::{PvsSearch, SearchConfig};

        if self.state.result.is_terminal() {
            return Err(JsError::new("game is already over"));
        }

        let config = SearchConfig {
            max_depth,
            max_time_ms: time_ms as u64,
            tt_size_mb: 4,
        };
        let mut search = PvsSearch::new(config, HeuristicEval);
        let result = search.search(&mut self.state);

        let mv = result
            .best_move
            .ok_or_else(|| JsError::new("no move found"))?;

        let best_ptn = ptn::format_move(mv, &self.state);
        let pv_ptn: Vec<String> = result.pv.iter().map(|m| ptn::format_move(*m, &self.state)).collect();

        // Apply the move.
        self.state.apply_move(mv);
        self.move_history.push(mv);

        let info = SearchResultInfo {
            best_move: best_ptn,
            score: result.score,
            depth: result.depth,
            nodes: result.nodes as u32,
            pv: pv_ptn,
            tt_hits: result.tt_hits as u32,
        };

        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultInfo {
    best_move: String,
    score: i32,
    depth: u8,
    nodes: u32,
    pv: Vec<String>,
    tt_hits: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PieceInfo {
    color: String,
    piece_type: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquareInfo {
    pieces: Vec<PieceInfo>,
    active: bool,
}

fn format_result(result: GameResult) -> String {
    match result {
        GameResult::Ongoing => "ongoing".into(),
        GameResult::RoadWin(c) => format!("road_win_{}", color_str(c)),
        GameResult::FlatWin(c) => format!("flat_win_{}", color_str(c)),
        GameResult::Draw => "draw".into(),
    }
}

fn color_str(c: tak_core::piece::Color) -> &'static str {
    match c {
        tak_core::piece::Color::White => "white",
        tak_core::piece::Color::Black => "black",
    }
}
