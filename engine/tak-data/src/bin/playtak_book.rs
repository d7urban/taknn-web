use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use clap::Parser;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use tak_core::moves::Move;
use tak_core::piece::Color;
use tak_core::ptn;
use tak_core::rules::GameConfig;
use tak_core::state::GameState;
use tak_core::symmetry::D4;
use tak_core::tps;

#[derive(Parser, Debug)]
#[command(name = "playtak-book")]
#[command(about = "Generate an opening book from live PlayTak game history")]
struct Args {
    #[arg(long, default_value = "https://api.playtak.com/v1")]
    api_base: String,

    #[arg(long, default_value = "../web/public/models/opening_book.json")]
    out: PathBuf,

    #[arg(long, default_value_t = 6)]
    board_size: u8,

    /// PlayTak API komi uses half-points: 4 => 2.0 komi.
    #[arg(long, default_value_t = 4)]
    komi_half_points: i16,

    #[arg(long, default_value_t = 8)]
    max_ply: u16,

    #[arg(long, default_value_t = 3)]
    min_support: u32,

    #[arg(long, default_value_t = 4)]
    max_moves_per_position: usize,

    #[arg(long, default_value_t = 100)]
    per_page: u32,

    /// Stop after this many pages. 0 means fetch all available pages.
    #[arg(long, default_value_t = 0)]
    page_limit: u32,

    #[arg(long, default_value_t = 1500)]
    min_rating: i32,

    #[arg(long, default_value_t = false)]
    include_bots: bool,
}

#[derive(Deserialize)]
struct GamesResponse {
    items: Vec<PlayTakGame>,
    #[serde(rename = "totalPages")]
    total_pages: u32,
}

#[derive(Deserialize)]
struct PlayTakGame {
    id: u64,
    size: u8,
    player_white: String,
    player_black: String,
    notation: String,
    result: String,
    rating_white: i32,
    rating_black: i32,
    tournament: u8,
    komi: i16,
}

#[derive(Default)]
struct MoveAggregate {
    support: f64,
    total_value: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OpeningBook {
    version: u8,
    max_ply: u16,
    entries: HashMap<String, BookEntry>,
}

#[derive(Serialize)]
struct BookEntry {
    moves: Vec<BookMove>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BookMove {
    move_key: String,
    support: u32,
    value: f64,
}

#[derive(Copy, Clone)]
enum Outcome {
    WhiteWin,
    BlackWin,
    Draw,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let (komi, half_komi) = decode_half_point_komi(args.komi_half_points)?;

    let client = Client::builder()
        .user_agent("TakNN-web opening-book generator")
        .build()?;

    let mut aggregates: HashMap<String, HashMap<String, MoveAggregate>> = HashMap::new();
    let mut accepted_games = 0u32;
    let mut skipped_games = 0u32;

    let mut page = 0u32;
    let mut total_pages = 1u32;
    while page < total_pages && (args.page_limit == 0 || page < args.page_limit) {
        let response = fetch_games_page(&client, &args, page)?;
        total_pages = response.total_pages.max(1);

        for game in response.items {
            if !should_include_game(&game, &args) {
                skipped_games += 1;
                continue;
            }
            match process_game(&game, args.max_ply, komi, half_komi, &mut aggregates) {
                Ok(true) => accepted_games += 1,
                Ok(false) => skipped_games += 1,
                Err(err) => {
                    skipped_games += 1;
                    eprintln!("Skipping game {}: {}", game.id, err);
                }
            }
        }

        page += 1;
    }

    let entries = finalize_entries(aggregates, args.min_support, args.max_moves_per_position);

    let book = OpeningBook {
        version: 1,
        max_ply: args.max_ply,
        entries,
    };

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_string_pretty(&book)?)?;

    println!(
        "Wrote {} opening-book position(s) from {} PlayTak game(s) ({} skipped) to {}",
        book.entries.len(),
        accepted_games,
        skipped_games,
        args.out.display()
    );

    Ok(())
}

fn fetch_games_page(
    client: &Client,
    args: &Args,
    page: u32,
) -> Result<GamesResponse, Box<dyn std::error::Error>> {
    let response = client
        .get(format!(
            "{}/games-history",
            args.api_base.trim_end_matches('/')
        ))
        .query(&[
            ("page", page.to_string()),
            ("limit", args.per_page.to_string()),
            ("size", args.board_size.to_string()),
            ("type", "tournament".to_string()),
            ("komi", args.komi_half_points.to_string()),
            ("sort", "id".to_string()),
            ("order", "DESC".to_string()),
        ])
        .send()?
        .error_for_status()?;
    Ok(response.json()?)
}

fn should_include_game(game: &PlayTakGame, args: &Args) -> bool {
    if game.size != args.board_size || game.tournament != 1 || game.komi != args.komi_half_points {
        return false;
    }
    if game.notation.trim().is_empty() {
        return false;
    }
    if game.rating_white < args.min_rating || game.rating_black < args.min_rating {
        return false;
    }
    if !args.include_bots && (is_bot_name(&game.player_white) || is_bot_name(&game.player_black)) {
        return false;
    }
    true
}

fn process_game(
    game: &PlayTakGame,
    max_ply: u16,
    komi: i8,
    half_komi: bool,
    aggregates: &mut HashMap<String, HashMap<String, MoveAggregate>>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let outcome = parse_result(&game.result)?;
    let mut config = GameConfig::standard(game.size);
    config.komi = komi;
    config.half_komi = half_komi;
    let mut state = GameState::new(config);

    let mut seen = Vec::new();
    for token in game
        .notation
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if state.ply >= max_ply {
            break;
        }

        let ptn_move = convert_playtak_move(token)?;
        let mv = ptn::parse_move(&ptn_move, &state)?;
        if !state.legal_moves().contains(&mv) {
            return Err(format!(
                "illegal converted move {} ({}) at ply {}",
                ptn_move, token, state.ply
            )
            .into());
        }
        let (key, sym) = canonical_position_key(&state);
        let canonical_move = sym.transform_move(mv, state.config.size);
        seen.push((
            qualify_book_key(&key, state.config.komi, state.config.half_komi),
            encode_book_move_key(canonical_move),
            state.side_to_move,
        ));
        state.apply_move(mv);
    }

    if seen.is_empty() {
        return Ok(false);
    }

    let avg_rating = (game.rating_white + game.rating_black) as f64 / 2.0;
    let weight = 2.0f64.powf((avg_rating - 1500.0) / 200.0).max(0.1);

    for (key, move_key, side_to_move) in seen {
        let value = outcome_value(outcome, side_to_move);
        let moves = aggregates.entry(key).or_default();
        let entry = moves.entry(move_key).or_default();
        entry.support += weight;
        entry.total_value += value * weight;
    }

    Ok(true)
}

fn finalize_entries(
    aggregates: HashMap<String, HashMap<String, MoveAggregate>>,
    min_support: u32,
    max_moves_per_position: usize,
) -> HashMap<String, BookEntry> {
    let mut entries = HashMap::new();

    for (key, moves) in aggregates {
        let mut book_moves: Vec<BookMove> = moves
            .into_iter()
            .filter_map(|(move_key, aggregate)| {
                let support_int = aggregate.support.round() as u32;
                if support_int < min_support {
                    return None;
                }
                Some(BookMove {
                    move_key,
                    support: support_int,
                    value: ((aggregate.total_value / aggregate.support) * 1000.0).round() / 1000.0,
                })
            })
            .collect();

        if book_moves.is_empty() {
            continue;
        }

        book_moves.sort_by(|left, right| {
            right
                .support
                .cmp(&left.support)
                .then_with(|| {
                    right
                        .value
                        .partial_cmp(&left.value)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| left.move_key.cmp(&right.move_key))
        });
        book_moves.truncate(max_moves_per_position);
        entries.insert(key, BookEntry { moves: book_moves });
    }

    entries
}

fn decode_half_point_komi(half_points: i16) -> Result<(i8, bool), Box<dyn std::error::Error>> {
    if !(-16..=16).contains(&half_points) {
        return Err(format!("komi half-points out of supported range: {}", half_points).into());
    }
    Ok(((half_points / 2) as i8, half_points % 2 != 0))
}

fn qualify_book_key(key: &str, komi: i8, half_komi: bool) -> String {
    format!("{}|k={}{}", key, komi, if half_komi { ".5" } else { "" })
}

fn canonical_position_key(state: &GameState) -> (String, D4) {
    let mut best_key = String::new();
    let mut best_sym = D4::Identity;
    let mut initialized = false;

    for &sym in &D4::ALL {
        let mut transformed = state.clone();
        transformed.board = state.board.transform(sym, state.config.size);
        let key = tps::to_tps(&transformed);
        if !initialized || key < best_key {
            best_key = key;
            best_sym = sym;
            initialized = true;
        }
    }

    (best_key, best_sym)
}

fn encode_book_move_key(mv: Move) -> String {
    match mv {
        Move::Place { square, piece_type } => format!("P:{}:{}", square.0, piece_type as u8),
        Move::Spread {
            src,
            dir,
            pickup,
            template,
        } => format!("S:{}:{}:{}:{}", src.0, dir as u8, pickup, template.0),
    }
}

fn outcome_value(outcome: Outcome, side_to_move: Color) -> f64 {
    match outcome {
        Outcome::Draw => 0.0,
        Outcome::WhiteWin => {
            if side_to_move == Color::White {
                1.0
            } else {
                -1.0
            }
        }
        Outcome::BlackWin => {
            if side_to_move == Color::Black {
                1.0
            } else {
                -1.0
            }
        }
    }
}

fn parse_result(result: &str) -> Result<Outcome, Box<dyn std::error::Error>> {
    match result {
        "R-0" | "F-0" | "1-0" | "X-0" => Ok(Outcome::WhiteWin),
        "0-R" | "0-F" | "0-1" | "0-X" => Ok(Outcome::BlackWin),
        "1/2-1/2" | "0-0" => Ok(Outcome::Draw),
        _ => Err(format!("unsupported result: {}", result).into()),
    }
}

fn convert_playtak_move(token: &str) -> Result<String, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = token.split_whitespace().collect();
    match parts.as_slice() {
        ["P", square] => Ok(square.to_ascii_lowercase()),
        ["P", square, stone] => {
            let prefix = if *stone == "C" { 'C' } else { 'S' };
            Ok(format!("{}{}", prefix, square.to_ascii_lowercase()))
        }
        ["M", from, to, drops @ ..] if !drops.is_empty() => {
            let dir = move_direction(from, to)?;
            let mut pickup = 0u8;
            let mut drop_str = String::new();
            for drop in drops {
                let value = drop.parse::<u8>()?;
                pickup += value;
                drop_str.push_str(drop);
            }

            let mut ptn = String::new();
            if pickup > 1 {
                ptn.push_str(&pickup.to_string());
            }
            ptn.push_str(&from.to_ascii_lowercase());
            ptn.push(dir);
            if !(drops.len() == 1 && pickup == 1) {
                ptn.push_str(&drop_str);
            }
            Ok(ptn)
        }
        _ => Err(format!("unsupported PlayTak notation token: {}", token).into()),
    }
}

fn move_direction(from: &str, to: &str) -> Result<char, Box<dyn std::error::Error>> {
    let from = from.as_bytes();
    let to = to.as_bytes();
    if from.len() != 2 || to.len() != 2 {
        return Err(format!(
            "invalid move endpoints: {} -> {}",
            String::from_utf8_lossy(from),
            String::from_utf8_lossy(to)
        )
        .into());
    }

    Ok(if from[0] == to[0] {
        if to[1] > from[1] {
            '+'
        } else {
            '-'
        }
    } else if to[0] > from[0] {
        '>'
    } else {
        '<'
    })
}

fn is_bot_name(name: &str) -> bool {
    name.ends_with("Bot") || name.contains("_Bot")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_playtak_place_moves() {
        assert_eq!(convert_playtak_move("P D4").unwrap(), "d4");
        assert_eq!(convert_playtak_move("P C2 C").unwrap(), "Cc2");
        assert_eq!(convert_playtak_move("P D6 W").unwrap(), "Sd6");
    }

    #[test]
    fn converts_playtak_spread_moves() {
        assert_eq!(convert_playtak_move("M D4 C4 1").unwrap(), "d4<");
        assert_eq!(convert_playtak_move("M B6 D6 1 3").unwrap(), "4b6>13");
    }

    #[test]
    fn parses_results() {
        assert!(matches!(parse_result("R-0").unwrap(), Outcome::WhiteWin));
        assert!(matches!(parse_result("0-F").unwrap(), Outcome::BlackWin));
        assert!(matches!(parse_result("1/2-1/2").unwrap(), Outcome::Draw));
    }

    #[test]
    fn qualifies_book_key_with_komi() {
        assert_eq!(
            qualify_book_key("x6/x6/x6/x6/x6/x6 1 1", 2, false),
            "x6/x6/x6/x6/x6/x6 1 1|k=2"
        );
        assert_eq!(qualify_book_key("key", 2, true), "key|k=2.5");
    }
}
