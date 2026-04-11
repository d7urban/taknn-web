//! Training record format and shard file management.

use std::io::{Read, Write};
use half::f16;
use tak_core::board::Square;
use tak_core::piece::{Color, Piece};
use tak_core::rules::GameConfig;
use tak_core::state::{GameResult, GameState};
use tak_core::tactical::TacticalPhase;
use tak_core::templates::TemplateTable;
use tak_core::zobrist;

/// One training record. Serialized as fixed-size header + variable-length data.
///
/// Contains both ground-truth labels (`game_result`, `flat_margin`) and
/// search-derived soft labels (`teacher_wdl`, `teacher_margin`).  The Rust
/// training pipeline uses the ground-truth labels; the search-derived fields
/// are consumed by the Python training loop and preserved for future use.
#[derive(Clone, Debug)]
pub struct TrainingRecord {
    pub board_size: u8,
    pub side_to_move: Color,
    pub ply: u16,
    pub reserves: [u8; 4],
    pub komi: i8,
    pub half_komi: bool,
    /// Ground-truth game outcome (used by Rust trainer for WDL labels).
    pub game_result: GameResult,
    /// Ground-truth flat-count difference at game end (used by Rust trainer
    /// for margin labels, normalized to [-1, 1] via `/50`).
    pub flat_margin: i16,
    pub search_depth: u8,
    pub search_nodes: u32,
    pub game_id: u32,
    pub source_model_id: u16,
    pub tactical_phase: TacticalPhase,
    /// Search-score soft WDL (logistic curve). Used by Python trainer;
    /// intentionally unused by Rust trainer (see `tak-train/data.rs`).
    pub teacher_wdl: [f16; 3],
    /// Search-score soft margin. Same consumer note as `teacher_wdl`.
    pub teacher_margin: f32,
    /// (move_index, probability)
    pub policy_target: Vec<(u16, f16)>,
    /// We store the board state in a packed format.
    pub board_data: Vec<u8>,
}

impl TrainingRecord {
    /// Serialize this record to a binary buffer.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        // placeholder for record_length
        buf.extend_from_slice(&[0u8; 4]);

        buf.push(self.board_size);
        buf.push(self.side_to_move as u8);
        buf.extend_from_slice(&self.ply.to_le_bytes());
        buf.extend_from_slice(&self.reserves);
        buf.push(self.komi as u8);
        buf.push(if self.half_komi { 1 } else { 0 });
        buf.push(match self.game_result {
            GameResult::RoadWin(Color::White) => 0,
            GameResult::RoadWin(Color::Black) => 1,
            GameResult::FlatWin(Color::White) => 2,
            GameResult::FlatWin(Color::Black) => 3,
            GameResult::Draw => 4,
            GameResult::Ongoing => 255,
        });
        buf.extend_from_slice(&self.flat_margin.to_le_bytes());
        buf.push(self.search_depth);
        buf.extend_from_slice(&self.search_nodes.to_le_bytes());
        buf.extend_from_slice(&self.game_id.to_le_bytes());
        buf.extend_from_slice(&self.source_model_id.to_le_bytes());
        buf.push(match self.tactical_phase {
            TacticalPhase::Quiet => 0,
            TacticalPhase::SemiTactical => 1,
            TacticalPhase::Tactical => 2,
        });

        buf.extend_from_slice(&(self.policy_target.len() as u16).to_le_bytes());
        
        // teacher_wdl [f16; 3] (6 bytes)
        for wdl in &self.teacher_wdl {
            buf.extend_from_slice(&wdl.to_bits().to_le_bytes());
        }
        // teacher_margin [f32] (4 bytes)
        buf.extend_from_slice(&self.teacher_margin.to_le_bytes());

        buf.extend_from_slice(&self.board_data);

        for (idx, prob) in &self.policy_target {
            buf.extend_from_slice(&idx.to_le_bytes());
            buf.extend_from_slice(&prob.to_bits().to_le_bytes());
        }

        let len = buf.len() as u32;
        buf[0..4].copy_from_slice(&len.to_le_bytes());
        buf
    }

    /// Deserialize a record from a binary buffer.
    pub fn from_bytes(b: &[u8]) -> std::io::Result<Self> {
        if b.len() < 41 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "record too short"));
        }
        let _record_len = u32::from_le_bytes(b[0..4].try_into().unwrap());
        let board_size = b[4];
        let side_to_move = match b[5] {
            0 => Color::White,
            1 => Color::Black,
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid side_to_move")),
        };
        let ply = u16::from_le_bytes(b[6..8].try_into().unwrap());
        let mut reserves = [0u8; 4];
        reserves.copy_from_slice(&b[8..12]);
        let komi = b[12] as i8;
        let half_komi = b[13] == 1;
        let game_result = match b[14] {
            0 => GameResult::RoadWin(Color::White),
            1 => GameResult::RoadWin(Color::Black),
            2 => GameResult::FlatWin(Color::White),
            3 => GameResult::FlatWin(Color::Black),
            4 => GameResult::Draw,
            255 => GameResult::Ongoing,
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid game_result")),
        };
        let flat_margin = i16::from_le_bytes(b[15..17].try_into().unwrap());
        let search_depth = b[17];
        let search_nodes = u32::from_le_bytes(b[18..22].try_into().unwrap());
        let game_id = u32::from_le_bytes(b[22..26].try_into().unwrap());
        let source_model_id = u16::from_le_bytes(b[26..28].try_into().unwrap());
        let tactical_phase = match b[28] {
            0 => TacticalPhase::Quiet,
            1 => TacticalPhase::SemiTactical,
            2 => TacticalPhase::Tactical,
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid tactical_phase")),
        };
        let num_policy = u16::from_le_bytes(b[29..31].try_into().unwrap());
        
        let teacher_wdl = [
            f16::from_bits(u16::from_le_bytes(b[31..33].try_into().unwrap())),
            f16::from_bits(u16::from_le_bytes(b[33..35].try_into().unwrap())),
            f16::from_bits(u16::from_le_bytes(b[35..37].try_into().unwrap())),
        ];
        let teacher_margin = f32::from_le_bytes(b[37..41].try_into().unwrap());

        let policy_len = num_policy as usize * 4;
        if b.len() < 41 + policy_len {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "record too short for policy"));
        }
        
        let board_data_end = b.len() - policy_len;
        let board_data = b[41..board_data_end].to_vec();
        
        let mut policy_target = Vec::with_capacity(num_policy as usize);
        let mut offset = board_data_end;
        for _ in 0..num_policy {
            let idx = u16::from_le_bytes(b[offset..offset+2].try_into().unwrap());
            let prob_bits = u16::from_le_bytes(b[offset+2..offset+4].try_into().unwrap());
            policy_target.push((idx, f16::from_bits(prob_bits)));
            offset += 4;
        }

        Ok(TrainingRecord {
            board_size,
            side_to_move,
            ply,
            reserves,
            komi,
            half_komi,
            game_result,
            flat_margin,
            search_depth,
            search_nodes,
            game_id,
            source_model_id,
            tactical_phase,
            teacher_wdl,
            teacher_margin,
            policy_target,
            board_data,
        })
    }

    /// Pack a board state into the binary format.
    pub fn pack_board(state: &GameState) -> Vec<u8> {
        let size = state.config.size;
        let mut data = Vec::with_capacity((size * size * 2) as usize);
        for r in 0..size {
            for c in 0..size {
                let sq = Square::from_rc(r, c);
                let stack = state.board.get(sq);
                if stack.is_empty() {
                    data.push(255);
                    data.push(0);
                    continue;
                }
                let top = stack.top.unwrap();
                data.push(top as u8);
                data.push(stack.height);
                if stack.height > 1 {
                    let interior_count = (stack.height - 1).min(7);
                    let byte_count = interior_count.div_ceil(4);
                    for i in 0..byte_count {
                        let mut byte = 0u8;
                        for j in 0..4 {
                            let idx = i * 4 + j;
                            if idx < interior_count {
                                let color = stack.below[idx as usize];
                                byte |= (color as u8) << (j * 2);
                            }
                        }
                        data.push(byte);
                    }
                    if stack.height > 8 {
                        data.push(stack.buried_white);
                        data.push(stack.buried_black);
                    }
                }
            }
        }
        data
    }

    /// Unpack board data into a GameState.
    pub fn unpack_board(&self) -> std::io::Result<GameState> {
        let size = self.board_size;
        let mut config = GameConfig::standard(size);
        config.komi = self.komi;
        config.half_komi = self.half_komi;
        let mut state = GameState::new(config);
        state.side_to_move = self.side_to_move;
        state.ply = self.ply;
        state.reserves = self.reserves;
        state.result = GameResult::Ongoing;
        
        let mut offset = 0;
        for r in 0..size {
            for c in 0..size {
                let top_raw = self.board_data[offset];
                let height = self.board_data[offset+1];
                offset += 2;
                if top_raw == 255 {
                    continue;
                }
                
                let sq = Square::from_rc(r, c);
                let stack = state.board.get_mut(sq);
                
                let top_piece = unsafe { std::mem::transmute::<u8, Piece>(top_raw) };
                
                if height > 1 {
                    let interior_count = (height - 1).min(7);
                    let byte_count = interior_count.div_ceil(4);
                    let mut interior_colors = Vec::new();
                    for _ in 0..byte_count {
                        let byte = self.board_data[offset];
                        offset += 1;
                        for j in 0..4 {
                            if interior_colors.len() < interior_count as usize {
                                let color_raw = (byte >> (j * 2)) & 0x03;
                                let color = match color_raw {
                                    0 => Color::White,
                                    1 => Color::Black,
                                    _ => unreachable!(),
                                };
                                interior_colors.push(color);
                            }
                        }
                    }
                    
                    let (bw, bb) = if height > 8 {
                        let w = self.board_data[offset];
                        let b = self.board_data[offset+1];
                        offset += 2;
                        (w, b)
                    } else {
                        (0, 0)
                    };
                    
                    stack.buried_white = bw;
                    stack.buried_black = bb;
                    for color in interior_colors {
                        stack.below.push(color);
                    }
                    stack.height = height;
                    stack.top = Some(top_piece);
                } else {
                    stack.top = Some(top_piece);
                    stack.height = 1;
                }
            }
        }
        
        state.zobrist = zobrist::compute_full(&state.board, size, state.side_to_move, &state.reserves);
        state.templates = TemplateTable::build(size);
        Ok(state)
    }
}

pub struct ShardWriter<W: Write> {
    encoder: zstd::Encoder<'static, W>,
}

impl<W: Write> ShardWriter<W> {
    pub fn new(writer: W) -> std::io::Result<Self> {
        let mut encoder = zstd::Encoder::new(writer, 3)?;
        encoder.write_all(b"TKNN")?;
        encoder.write_all(&1u16.to_le_bytes())?;
        encoder.write_all(&[0u8])?;
        encoder.write_all(&[0u8; 9])?;
        Ok(ShardWriter { encoder })
    }

    pub fn write_record(&mut self, record: &TrainingRecord) -> std::io::Result<()> {
        let bytes = record.to_bytes();
        self.encoder.write_all(&bytes)
    }

    pub fn finish(self) -> std::io::Result<W> {
        self.encoder.finish()
    }
}

pub struct ShardReader<R: Read> {
    decoder: zstd::Decoder<'static, std::io::BufReader<R>>,
}

impl<R: Read> ShardReader<R> {
    pub fn new(reader: R) -> std::io::Result<Self> {
        let mut decoder = zstd::Decoder::new(reader)?;
        let mut magic = [0u8; 4];
        decoder.read_exact(&mut magic)?;
        if &magic != b"TKNN" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid magic"));
        }
        let mut version = [0u8; 2];
        decoder.read_exact(&mut version)?;
        let mut endian = [0u8; 1];
        decoder.read_exact(&mut endian)?;
        let mut reserved = [0u8; 9];
        decoder.read_exact(&mut reserved)?;
        Ok(ShardReader { decoder })
    }

    pub fn next_record(&mut self) -> std::io::Result<Option<TrainingRecord>> {
        let mut len_bytes = [0u8; 4];
        if let Err(e) = self.decoder.read_exact(&mut len_bytes) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                return Ok(None);
            }
            return Err(e);
        }
        let len = u32::from_le_bytes(len_bytes);
        let mut buf = vec![0u8; len as usize];
        buf[0..4].copy_from_slice(&len_bytes);
        self.decoder.read_exact(&mut buf[4..])?;
        Ok(Some(TrainingRecord::from_bytes(&buf)?))
    }
}
