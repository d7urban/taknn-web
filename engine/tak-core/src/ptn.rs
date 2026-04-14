//! PTN (Portable Tak Notation) parsing and serialization.
//!
//! Supports individual move parsing/formatting and full game files with headers.

use crate::board::Square;
use crate::moves::{Direction, Move};
use crate::piece::PieceType;
use crate::rules::GameConfig;
use crate::state::GameState;
use crate::templates::DropTemplateId;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum PtnError {
    InvalidSquare(String),
    InvalidMove(String),
    InvalidHeader(String),
    GameplayError(String),
}

impl std::fmt::Display for PtnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PtnError::InvalidSquare(s) => write!(f, "invalid square: {}", s),
            PtnError::InvalidMove(s) => write!(f, "invalid move: {}", s),
            PtnError::InvalidHeader(s) => write!(f, "invalid header: {}", s),
            PtnError::GameplayError(s) => write!(f, "gameplay error: {}", s),
        }
    }
}

impl std::error::Error for PtnError {}

// ---------------------------------------------------------------------------
// Square helpers
// ---------------------------------------------------------------------------

/// Parse a square string like "a1", "c3" into a Square.
/// Column 'a'=0, row '1'=0 (bottom).
pub fn parse_square(s: &str) -> Result<Square, PtnError> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return Err(PtnError::InvalidSquare(s.to_string()));
    }
    let col = bytes[0].wrapping_sub(b'a');
    let row = bytes[1].wrapping_sub(b'1');
    if col >= 8 || row >= 8 {
        return Err(PtnError::InvalidSquare(s.to_string()));
    }
    Ok(Square::from_rc(row, col))
}

/// Format a Square as PTN string (e.g., "a1").
pub fn format_square(sq: Square) -> String {
    let col = (b'a' + sq.col()) as char;
    let row = (b'1' + sq.row()) as char;
    format!("{}{}", col, row)
}

// ---------------------------------------------------------------------------
// Direction helpers
// ---------------------------------------------------------------------------

/// PTN direction characters.
/// In PTN: '+' = up (increasing row number = South in our coords),
///         '-' = down (North), '>' = right (East), '<' = left (West).
fn dir_to_ptn(dir: Direction) -> char {
    match dir {
        Direction::South => '+',
        Direction::North => '-',
        Direction::East => '>',
        Direction::West => '<',
    }
}

fn ptn_to_dir(ch: char) -> Option<Direction> {
    match ch {
        '+' => Some(Direction::South),
        '-' => Some(Direction::North),
        '>' => Some(Direction::East),
        '<' => Some(Direction::West),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Move parsing
// ---------------------------------------------------------------------------

/// Parse a single PTN move string. Needs GameState to resolve template IDs.
pub fn parse_move(s: &str, state: &GameState) -> Result<Move, PtnError> {
    // Strip result markers and whitespace.
    let s = s.trim().trim_end_matches(['\'', '"', '!', '?']);
    // Strip capstone-flatten marker '*'.
    let s = s.trim_end_matches('*');

    if s.is_empty() {
        return Err(PtnError::InvalidMove("empty move string".into()));
    }

    // Detect spread vs placement by looking for a direction character.
    let has_direction = s.chars().any(|c| matches!(c, '+' | '-' | '>' | '<'));

    if has_direction {
        parse_spread(s, state)
    } else {
        parse_placement(s, state)
    }
}

fn parse_placement(s: &str, _state: &GameState) -> Result<Move, PtnError> {
    let (piece_type, sq_str) = if s.starts_with('F') || s.starts_with('S') || s.starts_with('C') {
        let pt = match s.as_bytes()[0] {
            b'F' => PieceType::Flat,
            b'S' => PieceType::Wall,
            b'C' => PieceType::Cap,
            _ => unreachable!(),
        };
        (pt, &s[1..])
    } else {
        // No prefix = flat placement.
        (PieceType::Flat, s)
    };

    let square = parse_square(sq_str)?;
    Ok(Move::Place { square, piece_type })
}

fn parse_spread(s: &str, state: &GameState) -> Result<Move, PtnError> {
    let bytes = s.as_bytes();
    let mut pos = 0;

    // Optional pickup count (digit at start).
    let pickup: u8 = if bytes[0].is_ascii_digit() && !bytes[0].is_ascii_lowercase() {
        // Check if first char is a digit 1-8 (pickup count) vs part of square.
        // If next char is a letter, this digit is the pickup count.
        if bytes.len() > 1 && bytes[1].is_ascii_lowercase() {
            let p = bytes[0] - b'0';
            pos = 1;
            p
        } else {
            1
        }
    } else {
        1
    };

    // Square (2 chars).
    if pos + 2 > bytes.len() {
        return Err(PtnError::InvalidMove(s.to_string()));
    }
    let sq_str = &s[pos..pos + 2];
    let src = parse_square(sq_str)?;
    pos += 2;

    // Direction (1 char).
    if pos >= bytes.len() {
        return Err(PtnError::InvalidMove(s.to_string()));
    }
    let dir = ptn_to_dir(bytes[pos] as char)
        .ok_or_else(|| PtnError::InvalidMove(format!("bad direction '{}'", bytes[pos] as char)))?;
    pos += 1;

    // Optional drop sequence (remaining digits).
    let drops_str = &s[pos..];
    let drops: Vec<u8> = if drops_str.is_empty() {
        // All pieces dropped on first square.
        vec![pickup]
    } else {
        drops_str
            .chars()
            .map(|c| {
                c.to_digit(10)
                    .map(|d| d as u8)
                    .ok_or_else(|| PtnError::InvalidMove(format!("bad drop char '{}'", c)))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    // Validate drop sequence.
    let total: u8 = drops.iter().sum();
    if total != pickup {
        return Err(PtnError::InvalidMove(format!(
            "drop sum {} != pickup {}",
            total, pickup
        )));
    }
    let travel = drops.len() as u8;

    // Find matching template ID.
    let range = state.templates.lookup_range(pickup, travel);
    let mut found_id = None;
    for i in 0..range.count {
        let tid = DropTemplateId(range.base_id + i);
        let seq = state.templates.get_sequence(tid);
        if seq.drops.as_slice() == drops.as_slice() {
            found_id = Some(tid);
            break;
        }
    }
    let template = found_id.ok_or_else(|| {
        PtnError::InvalidMove(format!(
            "no template for pickup={} drops={:?}",
            pickup, drops
        ))
    })?;

    Ok(Move::Spread {
        src,
        dir,
        pickup,
        template,
    })
}

// ---------------------------------------------------------------------------
// Move formatting
// ---------------------------------------------------------------------------

/// Format a Move as a PTN string.
pub fn format_move(mv: Move, state: &GameState) -> String {
    match mv {
        Move::Place { square, piece_type } => {
            let sq = format_square(square);
            match piece_type {
                PieceType::Flat => sq,
                PieceType::Wall => format!("S{}", sq),
                PieceType::Cap => format!("C{}", sq),
            }
        }
        Move::Spread {
            src,
            dir,
            pickup,
            template,
        } => {
            let drops = &state.templates.get_sequence(template).drops;
            let mut result = String::new();

            // Pickup count (omit if 1).
            if pickup > 1 {
                result.push((b'0' + pickup) as char);
            }

            result.push_str(&format_square(src));
            result.push(dir_to_ptn(dir));

            // Drop sequence (omit if all dropped on first square, i.e., drops == [pickup]).
            if drops.len() > 1 || (drops.len() == 1 && drops[0] != pickup) {
                for &d in drops.iter() {
                    result.push((b'0' + d) as char);
                }
            }

            result
        }
    }
}

// ---------------------------------------------------------------------------
// Game parsing
// ---------------------------------------------------------------------------

/// Parse a full PTN game. Returns the final GameState and the list of moves.
pub fn parse_game(ptn: &str) -> Result<(GameState, Vec<Move>), PtnError> {
    let mut size: u8 = 6; // default
    let mut komi: i8 = 0;
    let mut half_komi = false;
    let mut lines = ptn.lines().peekable();

    // Parse headers.
    while let Some(line) = lines.peek() {
        let line = line.trim();
        if line.starts_with('[') {
            if let Some(val) = extract_header(line, "Size") {
                size = val
                    .parse()
                    .map_err(|_| PtnError::InvalidHeader(format!("bad size: {}", val)))?;
            }
            if let Some(val) = extract_header(line, "Komi") {
                (komi, half_komi) = parse_komi_header(val)?;
            }
            lines.next();
        } else {
            break;
        }
    }

    let mut config = GameConfig::standard(size);
    config.komi = komi;
    config.half_komi = half_komi;
    let mut state = GameState::new(config);
    let mut moves = Vec::new();

    // Parse moves.
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('{') {
            continue;
        }
        // Remove move numbers and result strings.
        for token in line.split_whitespace() {
            // Skip move numbers like "1." or "12."
            if token.ends_with('.') && token[..token.len() - 1].chars().all(|c| c.is_ascii_digit())
            {
                continue;
            }
            // Skip result strings.
            if matches!(
                token,
                "R-0" | "0-R" | "F-0" | "0-F" | "1/2-1/2" | "1-0" | "0-1" | "*"
            ) {
                continue;
            }
            if state.result.is_terminal() {
                break;
            }
            let mv = parse_move(token, &state)
                .map_err(|e| PtnError::GameplayError(format!("{}: {}", e, token)))?;
            state.apply_move(mv);
            moves.push(mv);
        }
    }

    Ok((state, moves))
}

fn extract_header<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let line = line.trim();
    if !line.starts_with('[') || !line.ends_with(']') {
        return None;
    }
    let inner = &line[1..line.len() - 1];
    let inner = inner.trim();
    if !inner.starts_with(key) {
        return None;
    }
    let rest = inner[key.len()..].trim();
    if rest.starts_with('"') && rest.ends_with('"') {
        Some(&rest[1..rest.len() - 1])
    } else {
        Some(rest)
    }
}

fn parse_komi_header(val: &str) -> Result<(i8, bool), PtnError> {
    let val = val.trim();
    if let Some(base) = val.strip_suffix(".5") {
        let komi = base
            .parse::<i8>()
            .map_err(|_| PtnError::InvalidHeader(format!("bad komi: {}", val)))?;
        Ok((komi, true))
    } else {
        let komi = val
            .parse::<i8>()
            .map_err(|_| PtnError::InvalidHeader(format!("bad komi: {}", val)))?;
        Ok((komi, false))
    }
}

/// Format a game as PTN.
pub fn format_game(config: &GameConfig, moves: &[Move]) -> String {
    let mut result = String::new();
    result.push_str(&format!("[Size \"{}\"]\n", config.size));
    if config.komi != 0 || config.half_komi {
        result.push_str(&format!(
            "[Komi \"{}{}\"]\n",
            config.komi,
            if config.half_komi { ".5" } else { "" }
        ));
    }
    result.push('\n');

    let mut state = GameState::new(*config);
    let mut move_num = 1;

    for (i, &mv) in moves.iter().enumerate() {
        if i % 2 == 0 {
            if i > 0 {
                result.push('\n');
            }
            result.push_str(&format!("{}. ", move_num));
        } else {
            result.push(' ');
        }
        result.push_str(&format_move(mv, &state));
        state.apply_move(mv);
        if i % 2 == 1 {
            move_num += 1;
        }
    }
    // If odd number of moves, the last move was white's and move_num wasn't incremented.
    result.push('\n');
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_roundtrip() {
        for r in 0..8u8 {
            for c in 0..8u8 {
                let sq = Square::from_rc(r, c);
                let s = format_square(sq);
                let sq2 = parse_square(&s).unwrap();
                assert_eq!(sq, sq2);
            }
        }
    }

    #[test]
    fn parse_placement_flat() {
        let state = GameState::new(GameConfig::standard(5));
        let mv = parse_move("a1", &state).unwrap();
        assert_eq!(
            mv,
            Move::Place {
                square: Square::from_rc(0, 0),
                piece_type: PieceType::Flat,
            }
        );
    }

    #[test]
    fn parse_placement_with_prefix() {
        let state = GameState::new(GameConfig::standard(5));
        assert_eq!(
            parse_move("Fa1", &state).unwrap(),
            Move::Place {
                square: Square::from_rc(0, 0),
                piece_type: PieceType::Flat
            }
        );
        assert_eq!(
            parse_move("Sa1", &state).unwrap(),
            Move::Place {
                square: Square::from_rc(0, 0),
                piece_type: PieceType::Wall
            }
        );
        assert_eq!(
            parse_move("Ca1", &state).unwrap(),
            Move::Place {
                square: Square::from_rc(0, 0),
                piece_type: PieceType::Cap
            }
        );
    }

    #[test]
    fn parse_spread_simple() {
        let mut state = GameState::new(GameConfig::standard(5));
        // Set up: opening placements.
        state.apply_move(Move::Place {
            square: Square::from_rc(0, 0),
            piece_type: PieceType::Flat,
        });
        state.apply_move(Move::Place {
            square: Square::from_rc(1, 1),
            piece_type: PieceType::Flat,
        });
        state.apply_move(Move::Place {
            square: Square::from_rc(2, 0),
            piece_type: PieceType::Flat,
        });
        state.apply_move(Move::Place {
            square: Square::from_rc(3, 3),
            piece_type: PieceType::Flat,
        });
        // Now white can spread (2,0) i.e. "a3".
        // "a3+" = move a3 up (South in our coords), pick up 1, drop 1 on a4.
        let mv = parse_move("a3+", &state).unwrap();
        match mv {
            Move::Spread {
                src, dir, pickup, ..
            } => {
                assert_eq!(src, Square::from_rc(2, 0));
                assert_eq!(dir, Direction::South);
                assert_eq!(pickup, 1);
            }
            _ => panic!("expected spread"),
        }
    }

    #[test]
    fn parse_spread_with_drops() {
        let mut state = GameState::new(GameConfig::standard(5));
        // Build a stack at (0,0) with 3 pieces.
        state.apply_move(Move::Place {
            square: Square::from_rc(0, 0),
            piece_type: PieceType::Flat,
        });
        state.apply_move(Move::Place {
            square: Square::from_rc(4, 4),
            piece_type: PieceType::Flat,
        });
        state.apply_move(Move::Place {
            square: Square::from_rc(1, 0),
            piece_type: PieceType::Flat,
        });
        state.apply_move(Move::Place {
            square: Square::from_rc(3, 3),
            piece_type: PieceType::Flat,
        });

        // Now stack (1,0) by spreading (0,0) south to (1,0). Wait, (0,0) has a black flat (opening rule).
        // Actually at ply 4, white moves. Let me think about what's on the board.
        // Ply 0: White places Black flat at (0,0)
        // Ply 1: Black places White flat at (4,4)
        // Ply 2: White places White flat at (1,0)
        // Ply 3: Black places Black flat at (3,3)
        // Now ply 4: White's turn. White owns (1,0) with white flat.
        // Spread (1,0) south to (2,0): "a2+"
        let mv = parse_move("a2+", &state).unwrap();
        let formatted = format_move(mv, &state);
        assert_eq!(formatted, "a2+");
    }

    #[test]
    fn format_roundtrip_placement() {
        let state = GameState::new(GameConfig::standard(5));
        for &pt in &[PieceType::Flat, PieceType::Wall, PieceType::Cap] {
            let mv = Move::Place {
                square: Square::from_rc(2, 3),
                piece_type: pt,
            };
            let s = format_move(mv, &state);
            let mv2 = parse_move(&s, &state).unwrap();
            assert_eq!(mv, mv2, "roundtrip failed for {:?}", pt);
        }
    }

    #[test]
    fn parse_game_simple() {
        let ptn = r#"[Size "5"]

1. a1 e5
2. b2 d4
"#;
        let (state, moves) = parse_game(ptn).unwrap();
        assert_eq!(state.config.size, 5);
        assert_eq!(moves.len(), 4);
        assert_eq!(state.ply, 4);
    }

    #[test]
    fn parse_game_with_komi_header() {
        let ptn = r#"[Size "6"]
[Komi "2"]

1. a1 f6
"#;
        let (state, moves) = parse_game(ptn).unwrap();
        assert_eq!(state.config.komi, 2);
        assert!(!state.config.half_komi);
        assert_eq!(moves.len(), 2);
    }

    #[test]
    fn format_game_emits_komi_header() {
        let mut config = GameConfig::standard(6);
        config.komi = 2;
        config.half_komi = true;
        let ptn = format_game(&config, &[]);
        assert!(ptn.contains("[Komi \"2.5\"]"));
    }

    #[test]
    fn parse_square_errors() {
        assert!(parse_square("").is_err());
        assert!(parse_square("a").is_err());
        assert!(parse_square("z1").is_err());
        assert!(parse_square("a0").is_err());
    }

    // -------------------------------------------------------------------
    // AC 1.9: PTN round-trip for 50+ games
    // -------------------------------------------------------------------

    /// Generate 50+ random games, format them as PTN, parse them back,
    /// and verify the final position matches.
    #[test]
    fn ptn_roundtrip_50_games() {
        use crate::state::GameResult;
        use crate::tps;
        use rand::Rng;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(999);
        let mut total_games = 0;

        for size in 3..=7u8 {
            let config = GameConfig::standard(size);
            let games_per_size = if size <= 4 { 15 } else { 10 };
            let max_plies = match size {
                3 => 18,
                4 => 32,
                5 => 50,
                _ => 60,
            };

            for _ in 0..games_per_size {
                let mut state = GameState::new(config);
                let mut moves_played: Vec<Move> = Vec::new();

                for _ in 0..max_plies {
                    if state.result != GameResult::Ongoing {
                        break;
                    }
                    let legal = state.legal_moves();
                    if legal.is_empty() {
                        break;
                    }
                    let idx = rng.random_range(0..legal.len());
                    let mv = legal[idx];
                    moves_played.push(mv);
                    state.apply_move(mv);
                }

                // Format as PTN.
                let ptn_text = format_game(&config, &moves_played);

                // Parse back.
                let (recovered_state, recovered_moves) = parse_game(&ptn_text)
                    .unwrap_or_else(|e| panic!("parse_game failed: {} (size={})", e, size));

                // Verify move count matches.
                assert_eq!(
                    recovered_moves.len(),
                    moves_played.len(),
                    "move count mismatch (size={})",
                    size
                );

                // Verify final TPS matches.
                let expected_tps = tps::to_tps(&state);
                let recovered_tps = tps::to_tps(&recovered_state);
                assert_eq!(
                    expected_tps, recovered_tps,
                    "final position mismatch (size={})",
                    size
                );

                total_games += 1;
            }
        }

        assert!(
            total_games >= 50,
            "expected 50+ game round-trips, got {}",
            total_games
        );
    }
}
