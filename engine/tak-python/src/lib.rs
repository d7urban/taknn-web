use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tak_core::state::{GameResult, GameState};
use tak_core::rules::GameConfig;
use tak_core::symmetry::D4;
use tak_core::descriptor;
use tak_core::tactical::{TacticalFlags, SpatialLabels};
use tak_core::tps;
use tak_data::shard::{ShardReader, ShardWriter};
use tak_data::selfplay::{SelfPlayConfig, SelfPlayEngine, TemperatureSchedule};
use tak_search::pvs::{PvsSearch, SearchConfig};
use tak_search::eval::HeuristicEval;
use numpy::PyArray3;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[pyclass]
pub struct PyGameState {
    inner: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (size, tps_str=None))]
    fn new(size: u8, tps_str: Option<&str>) -> PyResult<Self> {
        let inner = if let Some(s) = tps_str {
            tps::from_tps(s).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid TPS: {e}")))?
        } else {
            GameState::new(GameConfig::standard(size))
        };
        Ok(PyGameState { inner })
    }

    fn size(&self) -> u8 { self.inner.config.size }
    fn ply(&self) -> u16 { self.inner.ply }
    fn side_to_move(&self) -> u8 { self.inner.side_to_move as u8 }

    fn is_terminal(&self) -> bool { self.inner.result.is_terminal() }

    fn result(&self) -> u8 {
        match self.inner.result {
            GameResult::RoadWin(tak_core::piece::Color::White) => 0,
            GameResult::RoadWin(tak_core::piece::Color::Black) => 1,
            GameResult::FlatWin(tak_core::piece::Color::White) => 2,
            GameResult::FlatWin(tak_core::piece::Color::Black) => 3,
            GameResult::Draw => 4,
            GameResult::Ongoing => 255,
        }
    }

    fn legal_move_count(&self) -> usize {
        self.inner.legal_moves().len()
    }

    fn legal_moves(&self) -> Vec<usize> {
        let moves = self.inner.legal_moves();
        (0..moves.len()).collect()
    }

    fn apply_move(&mut self, move_index: usize) -> PyResult<()> {
        let moves = self.inner.legal_moves();
        if move_index >= moves.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("move index out of range"));
        }
        self.inner.apply_move(moves[move_index]);
        Ok(())
    }

    fn to_tps(&self) -> String {
        tps::to_tps(&self.inner)
    }

    fn flat_margin(&self) -> i16 {
        self.inner.flat_margin()
    }

    /// Returns a mapping: original_move_index -> transformed_move_index
    /// for D4 symmetry transform (0..7).
    fn get_transformation_map(&self, transform: u8) -> PyResult<Vec<usize>> {
        let sym = D4::from_u8(transform).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("transform must be 0..7")
        })?;
        let size = self.inner.config.size;
        let original_moves = self.inner.legal_moves();
        let mut transformed_state = self.inner.clone();
        transformed_state.board = transformed_state.board.transform(sym, size);
        let transformed_moves = transformed_state.legal_moves();

        let mut map = Vec::with_capacity(original_moves.len());
        for &m in original_moves.iter() {
            let tm = sym.transform_move(m, size);
            let target_idx = transformed_moves
                .iter()
                .position(|&x| x == tm)
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Transformed move not found")
                })?;
            map.push(target_idx);
        }
        Ok(map)
    }

    /// Encode current board state as a [31, 8, 8] numpy array.
    fn encode_tensor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let tensor = tak_core::tensor::BoardTensor::encode(&self.inner);
        Ok(PyArray3::from_vec3(
            py,
            &tensor_to_vec3(&tensor.data),
        )?)
    }

    /// Run PVS search with heuristic eval. Returns (best_move_index, score, depth, nodes).
    #[pyo3(signature = (max_depth=20, max_time_ms=1000, tt_size_mb=16))]
    fn search_move(&mut self, max_depth: u8, max_time_ms: u64, tt_size_mb: usize) -> PyResult<(usize, i32, u8, u64)> {
        let config = SearchConfig { max_depth, max_time_ms, tt_size_mb };
        let mut search = PvsSearch::new(config, HeuristicEval);
        let result = search.search(&mut self.inner);
        let moves = self.inner.legal_moves();
        if moves.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("No legal moves"));
        }
        let move_idx = match result.best_move {
            Some(best_move) => moves.iter().position(|&m| m == best_move).unwrap_or(0),
            None => 0, // fallback to first legal move
        };
        Ok((move_idx, result.score, result.depth, result.nodes))
    }

    /// Encode board tensors for all positions after each legal move.
    /// Returns list of [31, 8, 8] numpy arrays (one per legal move).
    fn encode_children_tensors<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray3<f32>>>> {
        let moves = self.inner.legal_moves();
        let mut arrays = Vec::with_capacity(moves.len());
        for &mv in &moves {
            let mut child = self.inner.clone();
            child.apply_move(mv);
            let tensor = tak_core::tensor::BoardTensor::encode(&child);
            let arr = PyArray3::from_vec3(py, &tensor_to_vec3(&tensor.data))?;
            arrays.push(arr);
        }
        Ok(arrays)
    }

    /// Returns a list of move descriptor dicts for all legal moves.
    fn get_move_descriptors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let moves = self.inner.legal_moves();
        let descs = descriptor::build_descriptors(&self.inner, &moves);
        let list = PyList::empty(py);
        for d in &descs {
            let dict = PyDict::new(py);
            dict.set_item("src", d.src)?;
            dict.set_item("dst", d.dst)?;
            dict.set_item("path", d.path.as_slice())?;
            dict.set_item("move_type", d.move_type)?;
            dict.set_item("piece_type", d.piece_type)?;
            dict.set_item("direction", d.direction)?;
            dict.set_item("pickup_count", d.pickup_count)?;
            dict.set_item("drop_template_id", d.drop_template_id)?;
            dict.set_item("travel_length", d.travel_length)?;
            dict.set_item("capstone_flatten", d.capstone_flatten)?;
            dict.set_item("enters_occupied", d.enters_occupied)?;
            dict.set_item("opening_phase", d.opening_phase)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Compute per-square spatial labels for auxiliary head training.
    /// Returns dict with numpy arrays: road_threat [2,8,8], block_threat [2,8,8],
    /// cap_flatten [1,8,8], endgame scalar.
    fn compute_spatial_labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let labels = SpatialLabels::compute(&self.inner);
        let dict = PyDict::new(py);

        // road_threat [2, 8, 8]
        let rt = PyArray3::from_vec3(py, &spatial_2x64_to_vec3(&labels.road_threat))?;
        dict.set_item("road_threat", rt)?;

        // block_threat [2, 8, 8]
        let bt = PyArray3::from_vec3(py, &spatial_2x64_to_vec3(&labels.block_threat))?;
        dict.set_item("block_threat", bt)?;

        // cap_flatten [1, 8, 8]
        let cf = PyArray3::from_vec3(py, &spatial_1x64_to_vec3(&labels.cap_flatten))?;
        dict.set_item("cap_flatten", cf)?;

        // endgame scalar
        dict.set_item("endgame", labels.endgame)?;

        Ok(dict)
    }

    /// Returns a dict of tactical labels for the current position.
    fn get_tactical_labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let flags = TacticalFlags::compute(&self.inner);
        let dict = PyDict::new(py);
        dict.set_item("road_in_1_white", flags.road_in_1_white)?;
        dict.set_item("road_in_1_black", flags.road_in_1_black)?;
        dict.set_item("forced_defense", flags.forced_defense)?;
        dict.set_item("capstone_flatten", flags.capstone_flatten)?;
        dict.set_item("endgame", flags.endgame)?;
        dict.set_item("phase", flags.phase() as u8)?;
        Ok(dict)
    }
}

#[pyclass]
pub struct PyShardReader {
    reader: ShardReader<std::fs::File>,
}

#[pymethods]
impl PyShardReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let file = std::fs::File::open(path)?;
        let reader = ShardReader::new(file)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Shard error: {e}")))?;
        Ok(PyShardReader { reader })
    }

    fn next<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self
            .reader
            .next_record()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Read error: {e}")))?
        {
            Some(record) => {
                let dict = PyDict::new(py);
                dict.set_item("board_size", record.board_size)?;
                dict.set_item("side_to_move", record.side_to_move as u8)?;
                dict.set_item("ply", record.ply)?;
                dict.set_item(
                    "game_result",
                    match record.game_result {
                        GameResult::RoadWin(tak_core::piece::Color::White) => 0u8,
                        GameResult::RoadWin(tak_core::piece::Color::Black) => 1,
                        GameResult::FlatWin(tak_core::piece::Color::White) => 2,
                        GameResult::FlatWin(tak_core::piece::Color::Black) => 3,
                        GameResult::Draw => 4,
                        GameResult::Ongoing => 255,
                    },
                )?;
                dict.set_item("flat_margin", record.flat_margin)?;
                dict.set_item(
                    "teacher_wdl",
                    record
                        .teacher_wdl
                        .iter()
                        .map(|w| w.to_f32())
                        .collect::<Vec<_>>(),
                )?;
                dict.set_item("teacher_margin", record.teacher_margin)?;

                let policy_indices: Vec<u16> =
                    record.policy_target.iter().map(|(i, _)| *i).collect();
                let policy_probs: Vec<f32> =
                    record.policy_target.iter().map(|(_, p)| p.to_f32()).collect();
                dict.set_item("policy_indices", policy_indices)?;
                dict.set_item("policy_probs", policy_probs)?;

                // Unpack board to get tensor and TPS
                let state = record.unpack_board().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Unpack error: {e}"))
                })?;
                let tensor = tak_core::tensor::BoardTensor::encode(&state);
                let arr = PyArray3::from_vec3(py, &tensor_to_vec3(&tensor.data))?;
                dict.set_item("board_tensor", arr)?;
                dict.set_item("tps", tps::to_tps(&state))?;
                dict.set_item("tactical_phase", match record.tactical_phase {
                    tak_core::tactical::TacticalPhase::Quiet => 0u8,
                    tak_core::tactical::TacticalPhase::SemiTactical => 1,
                    tak_core::tactical::TacticalPhase::Tactical => 2,
                })?;

                Ok(Some(dict))
            }
            None => Ok(None),
        }
    }
}

/// Convert flat [C_IN * 64] array to Vec<Vec<Vec<f32>>> shape [31][8][8].
fn tensor_to_vec3(data: &[f32]) -> Vec<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(31);
    for c in 0..31 {
        let mut plane = Vec::with_capacity(8);
        for r in 0..8 {
            let mut row = Vec::with_capacity(8);
            for col in 0..8 {
                row.push(data[c * 64 + r * 8 + col]);
            }
            plane.push(row);
        }
        out.push(plane);
    }
    out
}

/// Convert [2][64] spatial label to Vec3 shape [2][8][8].
fn spatial_2x64_to_vec3(data: &[[f32; 64]; 2]) -> Vec<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(2);
    for channel in data {
        let mut plane = Vec::with_capacity(8);
        for r in 0..8 {
            let mut row = Vec::with_capacity(8);
            for col in 0..8 {
                row.push(channel[r * 8 + col]);
            }
            plane.push(row);
        }
        out.push(plane);
    }
    out
}

/// Convert [64] spatial label to Vec3 shape [1][8][8].
fn spatial_1x64_to_vec3(data: &[f32; 64]) -> Vec<Vec<Vec<f32>>> {
    let mut plane = Vec::with_capacity(8);
    for r in 0..8 {
        let mut row = Vec::with_capacity(8);
        for col in 0..8 {
            row.push(data[r * 8 + col]);
        }
        plane.push(row);
    }
    vec![plane]
}

/// Generate self-play games and write them to a zstd-compressed shard file.
///
/// Returns the number of training records written.
#[pyfunction]
#[pyo3(signature = (output_path, board_size, num_games, max_depth=4, max_time_ms=200, seed=42))]
fn generate_shard(
    output_path: &str,
    board_size: u8,
    num_games: u32,
    max_depth: u8,
    max_time_ms: u64,
    seed: u64,
) -> PyResult<usize> {
    let config = SelfPlayConfig {
        board_size,
        search_config: SearchConfig {
            max_depth,
            max_time_ms,
            tt_size_mb: 4,
        },
        temp_schedule: TemperatureSchedule::default(),
    };
    let engine = SelfPlayEngine::new(config);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let file = std::fs::File::create(output_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?;
    let mut writer = ShardWriter::new(file)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?;

    let mut total_records = 0;
    for game_id in 0..num_games {
        let records = engine.play_game(&mut rng, game_id);
        for record in &records {
            writer
                .write_record(record)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?;
        }
        total_records += records.len();
    }
    writer
        .finish()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?;

    Ok(total_records)
}

#[pymodule]
fn tak_python(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyShardReader>()?;
    m.add_function(wrap_pyfunction!(generate_shard, m)?)?;
    Ok(())
}
