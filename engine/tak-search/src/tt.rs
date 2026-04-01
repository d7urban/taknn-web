//! Transposition table for alpha-beta search.

use tak_core::moves::Move;

use crate::eval::Score;

/// TT entry flag indicating the type of bound stored.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum TTFlag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
}

/// A single transposition table entry.
#[derive(Copy, Clone)]
pub struct TTEntry {
    /// Upper 32 bits of zobrist for verification.
    pub key: u32,
    /// Best move found (or Move from a previous search).
    pub best_move: Option<Move>,
    /// Stored score.
    pub score: i16,
    /// Search depth at which this entry was produced.
    pub depth: u8,
    /// Bound type.
    pub flag: TTFlag,
    /// Generation counter to detect stale entries.
    pub generation: u8,
}

impl TTEntry {
    fn empty() -> Self {
        TTEntry {
            key: 0,
            best_move: None,
            score: 0,
            depth: 0,
            flag: TTFlag::Exact,
            generation: 0,
        }
    }
}

/// Power-of-two sized transposition table.
pub struct TranspositionTable {
    entries: Vec<TTEntry>,
    mask: usize,
    generation: u8,
}

impl TranspositionTable {
    /// Create a new TT with the given size in megabytes.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let count = ((size_mb * 1024 * 1024) / entry_size).next_power_of_two();
        let count = count.max(1024); // minimum size
        TranspositionTable {
            entries: vec![TTEntry::empty(); count],
            mask: count - 1,
            generation: 0,
        }
    }

    /// Index into the table from a zobrist hash.
    #[inline]
    fn index(&self, zobrist: u64) -> usize {
        (zobrist as usize) & self.mask
    }

    /// Verification key from a zobrist hash.
    #[inline]
    fn verify_key(zobrist: u64) -> u32 {
        (zobrist >> 32) as u32
    }

    /// Probe the table for a matching entry.
    pub fn probe(&self, zobrist: u64) -> Option<&TTEntry> {
        let idx = self.index(zobrist);
        let entry = &self.entries[idx];
        if entry.key == Self::verify_key(zobrist) && entry.depth > 0 {
            Some(entry)
        } else {
            None
        }
    }

    /// Store an entry in the table. Uses replace-by-depth with generation preference.
    pub fn store(
        &mut self,
        zobrist: u64,
        best_move: Option<Move>,
        score: Score,
        depth: u8,
        flag: TTFlag,
    ) {
        let idx = self.index(zobrist);
        let existing = &self.entries[idx];

        // Replace if: new entry is from current generation and existing is old,
        // or new depth >= existing depth, or existing is empty.
        let should_replace = existing.depth == 0
            || existing.generation != self.generation
            || depth >= existing.depth;

        if should_replace {
            self.entries[idx] = TTEntry {
                key: Self::verify_key(zobrist),
                best_move,
                score: score.clamp(i16::MIN as Score, i16::MAX as Score) as i16,
                depth,
                flag,
                generation: self.generation,
            };
        }
    }

    /// Advance the generation counter (call at the start of each new search).
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.fill(TTEntry::empty());
        self.generation = 0;
    }

    /// Get the current fill rate (fraction of non-empty entries, sampled).
    pub fn fill_rate(&self) -> f64 {
        let sample = self.entries.len().min(1000);
        let filled = self.entries[..sample]
            .iter()
            .filter(|e| e.depth > 0)
            .count();
        filled as f64 / sample as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tak_core::board::Square;
    use tak_core::piece::PieceType;

    #[test]
    fn store_and_probe() {
        let mut tt = TranspositionTable::new(1);
        let zobrist = 0xDEAD_BEEF_CAFE_BABEu64;
        let mv = Move::Place {
            square: Square::from_rc(2, 3),
            piece_type: PieceType::Flat,
        };

        tt.store(zobrist, Some(mv), 150, 5, TTFlag::Exact);

        let entry = tt.probe(zobrist).expect("should find entry");
        assert_eq!(entry.score, 150);
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.flag, TTFlag::Exact);
        assert_eq!(entry.best_move, Some(mv));
    }

    #[test]
    fn probe_miss() {
        let tt = TranspositionTable::new(1);
        assert!(tt.probe(0x12345678_9ABCDEF0).is_none());
    }

    #[test]
    fn generation_replacement() {
        let mut tt = TranspositionTable::new(1);
        let z = 0xAAAA_BBBB_CCCC_DDDDu64;

        tt.store(z, None, 100, 8, TTFlag::Exact);
        tt.new_search();

        // New search with lower depth should still replace stale entry.
        tt.store(z, None, 200, 3, TTFlag::LowerBound);
        let entry = tt.probe(z).unwrap();
        assert_eq!(entry.score, 200);
    }
}
