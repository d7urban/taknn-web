//! Time management for iterative deepening search.
//!
//! On native targets, uses `std::time::Instant`.
//! On WASM (`wasm32`), uses `js_sys::Date::now()` since `Instant` is
//! unsupported on `wasm32-unknown-unknown`.

/// Simple time manager: stops search when allocated time is exhausted.
pub struct TimeManager {
    max_time_ms: u64,
    #[cfg(not(target_arch = "wasm32"))]
    start: Option<std::time::Instant>,
    #[cfg(target_arch = "wasm32")]
    start_ms: Option<f64>,
    hard_stop: bool,
}

impl TimeManager {
    pub fn new(max_time_ms: u64) -> Self {
        TimeManager {
            max_time_ms,
            #[cfg(not(target_arch = "wasm32"))]
            start: None,
            #[cfg(target_arch = "wasm32")]
            start_ms: None,
            hard_stop: false,
        }
    }

    /// Call at the start of search.
    pub fn start(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start = Some(std::time::Instant::now());
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.start_ms = Some(js_sys::Date::now());
        }
        self.hard_stop = false;
    }

    /// Elapsed milliseconds since start.
    pub fn elapsed_ms(&self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start
                .map(|s| s.elapsed().as_millis() as u64)
                .unwrap_or(0)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.start_ms
                .map(|s| (js_sys::Date::now() - s) as u64)
                .unwrap_or(0)
        }
    }

    /// Check if time is up. Call periodically during search.
    pub fn should_stop(&self) -> bool {
        if self.hard_stop {
            return true;
        }
        self.elapsed_ms() >= self.max_time_ms
    }

    /// Force stop.
    pub fn stop(&mut self) {
        self.hard_stop = true;
    }

    /// Check if we have enough time for another iteration.
    /// Heuristic: reserve roughly one third of the budget for the next depth.
    /// This is still conservative, but less prone to stalling at shallow depth
    /// when the process is under transient CPU load from neighboring tests.
    pub fn can_start_new_depth(&self) -> bool {
        let elapsed = self.elapsed_ms();
        elapsed * 3 < self.max_time_ms * 2
    }
}
