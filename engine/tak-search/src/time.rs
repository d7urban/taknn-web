//! Time management for iterative deepening search.

/// Simple time manager: stops search when allocated time is exhausted.
pub struct TimeManager {
    max_time_ms: u64,
    start: Option<std::time::Instant>,
    hard_stop: bool,
}

impl TimeManager {
    pub fn new(max_time_ms: u64) -> Self {
        TimeManager {
            max_time_ms,
            start: None,
            hard_stop: false,
        }
    }

    /// Call at the start of search.
    pub fn start(&mut self) {
        self.start = Some(std::time::Instant::now());
        self.hard_stop = false;
    }

    /// Check if time is up. Call periodically during search.
    pub fn should_stop(&self) -> bool {
        if self.hard_stop {
            return true;
        }
        if let Some(start) = self.start {
            start.elapsed().as_millis() as u64 >= self.max_time_ms
        } else {
            false
        }
    }

    /// Elapsed milliseconds since start.
    pub fn elapsed_ms(&self) -> u64 {
        self.start
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    /// Force stop.
    pub fn stop(&mut self) {
        self.hard_stop = true;
    }

    /// Check if we have enough time for another iteration.
    /// Heuristic: if we've used more than 60% of the time, don't start a new depth.
    pub fn can_start_new_depth(&self) -> bool {
        if let Some(start) = self.start {
            let elapsed = start.elapsed().as_millis() as u64;
            elapsed < self.max_time_ms * 60 / 100
        } else {
            true
        }
    }
}

/// WASM-compatible time manager using f64 millis (no std::time::Instant in WASM).
#[cfg(target_arch = "wasm32")]
pub struct WasmTimeManager {
    max_time_ms: f64,
    start_ms: f64,
    hard_stop: bool,
}

#[cfg(target_arch = "wasm32")]
impl WasmTimeManager {
    pub fn new(max_time_ms: f64, now_ms: f64) -> Self {
        WasmTimeManager {
            max_time_ms,
            start_ms: now_ms,
            hard_stop: false,
        }
    }

    pub fn should_stop(&self, now_ms: f64) -> bool {
        self.hard_stop || (now_ms - self.start_ms) >= self.max_time_ms
    }

    pub fn stop(&mut self) {
        self.hard_stop = true;
    }
}
