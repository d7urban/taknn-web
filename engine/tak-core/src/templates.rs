use arrayvec::ArrayVec;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a drop template. ID 0 is reserved for "not a spread
/// move" (i.e. placements).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct DropTemplateId(pub u16);

/// A contiguous range of template IDs for a given (pickup_count, travel_length)
/// pair. The range is `base_id ..  base_id + count`.
#[derive(Clone, Debug)]
pub struct TemplateRange {
    pub base_id: u16,
    pub count: u16,
}

/// A drop sequence: how many pieces to drop at each step along the ray.
/// Length equals `travel_length`, sum equals `pickup_count`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DropSequence {
    pub drops: ArrayVec<u8, 8>,
}

/// Precomputed table of all valid drop templates for a given carry limit
/// (which equals the board size in Tak).
///
/// For a given carry_limit, precompute all valid (pickup_count, travel_length,
/// template) combinations. A drop template for pickup k over distance t is an
/// ordered partition of k into t parts, each >= 1.
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
#[derive(Clone)]
pub struct TemplateTable {
    /// For each (k, t): base template_id and count.
    /// Indexed as `entries[k - 1][t - 1]` for k in `1..=carry_limit`,
    /// t in `1..=carry_limit`.
    pub entries: Vec<Vec<TemplateRange>>,
    /// The actual drop sequences, indexed by `DropTemplateId`.
    /// `sequences[0]` is a dummy entry for ID 0 (the "not a spread" sentinel).
    pub sequences: Vec<DropSequence>,
}

// ---------------------------------------------------------------------------
// Combinatorics helpers
// ---------------------------------------------------------------------------

/// Binomial coefficient C(n, k). Returns 0 when k > n.
fn binom(n: u32, k: u32) -> u32 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    // Use the smaller of k and n-k to minimise iterations.
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result as u32
}

/// Generate all ordered partitions of `total` into `parts` parts (each >= 1),
/// in lexicographic order of [d_1, d_2, ..., d_parts].
///
/// An ordered partition of k into t parts with each part >= 1 is equivalent to
/// placing t-1 dividers among k-1 gaps. We enumerate them lexicographically.
fn generate_partitions(total: u8, parts: u8) -> Vec<ArrayVec<u8, 8>> {
    let mut results = Vec::new();
    let mut current: ArrayVec<u8, 8> = ArrayVec::new();
    generate_partitions_rec(total, parts, &mut current, &mut results);
    results
}

fn generate_partitions_rec(
    remaining: u8,
    parts_left: u8,
    current: &mut ArrayVec<u8, 8>,
    results: &mut Vec<ArrayVec<u8, 8>>,
) {
    if parts_left == 1 {
        current.push(remaining);
        results.push(current.clone());
        current.pop();
        return;
    }
    // The current part can be 1..=(remaining - parts_left + 1) so that the
    // remaining parts can each get at least 1.
    let max_here = remaining - (parts_left - 1);
    for v in 1..=max_here {
        current.push(v);
        generate_partitions_rec(remaining - v, parts_left - 1, current, results);
        current.pop();
    }
}

// ---------------------------------------------------------------------------
// TemplateTable implementation
// ---------------------------------------------------------------------------

impl TemplateTable {
    /// Build the template table for a given carry limit (== board size).
    pub fn build(carry_limit: u8) -> Self {
        let l = carry_limit as usize;

        // Sentinel entry at ID 0: "not a spread move".
        let mut sequences: Vec<DropSequence> = vec![DropSequence {
            drops: ArrayVec::new(),
        }];

        // entries[k-1][t-1] for k in 1..=L, t in 1..=L
        let mut entries: Vec<Vec<TemplateRange>> = Vec::with_capacity(l);

        let mut next_id: u16 = 1; // first real template ID

        for k in 1..=carry_limit {
            let mut row: Vec<TemplateRange> = Vec::with_capacity(l);
            for t in 1..=carry_limit {
                if t > k {
                    // Cannot partition k into t parts with each >= 1 when t > k.
                    row.push(TemplateRange {
                        base_id: next_id,
                        count: 0,
                    });
                } else {
                    let partitions = generate_partitions(k, t);
                    let count = partitions.len() as u16;
                    debug_assert_eq!(count, binom(k as u32 - 1, t as u32 - 1) as u16);

                    row.push(TemplateRange {
                        base_id: next_id,
                        count,
                    });

                    for p in partitions {
                        sequences.push(DropSequence { drops: p });
                    }
                    next_id += count;
                }
            }
            entries.push(row);
        }

        TemplateTable { entries, sequences }
    }

    /// Total number of real templates (excluding the sentinel ID 0).
    pub fn total_templates(&self) -> u16 {
        // sequences[0] is the sentinel, so real count is len - 1.
        (self.sequences.len() - 1) as u16
    }

    /// Look up the drop sequence for a given template ID.
    pub fn get_sequence(&self, id: DropTemplateId) -> &DropSequence {
        &self.sequences[id.0 as usize]
    }

    /// Look up the template range for a given (pickup, travel) pair.
    /// `pickup` is in `1..=carry_limit`, `travel` is in `1..=carry_limit`.
    pub fn lookup_range(&self, pickup: u8, travel: u8) -> &TemplateRange {
        &self.entries[pickup as usize - 1][travel as usize - 1]
    }
}

// ---------------------------------------------------------------------------
// Action-count functions
// ---------------------------------------------------------------------------

/// Returns the total number of movement actions for a single directed ray of
/// distance `d`, given a carry limit of `carry_limit`.
///
/// This is:  sum_{k=1}^{carry_limit} sum_{t=1}^{d} C(k-1, t-1)
pub fn actions_per_ray_distance(carry_limit: u8, d: u8) -> u32 {
    let mut total = 0u32;
    for k in 1..=carry_limit as u32 {
        for t in 1..=d as u32 {
            total += binom(k - 1, t - 1);
        }
    }
    total
}

/// Returns the grammar-maximum total number of movement actions for an NxN
/// board, by iterating over every square and every direction, computing the
/// ray distance to the board edge, and summing `actions_per_ray_distance`.
///
/// The carry limit equals the board size N.
pub fn total_movement_actions(size: u8) -> u32 {
    let n = size as u32;
    let carry_limit = size;

    // Precompute actions_per_ray for d in 1..=size-1 to avoid redundant work.
    let max_d = (size - 1) as usize;
    let mut ray_actions: Vec<u32> = vec![0; max_d + 1]; // index 0 unused
    for (d, slot) in ray_actions.iter_mut().enumerate().skip(1) {
        *slot = actions_per_ray_distance(carry_limit, d as u8);
    }

    let mut total = 0u32;
    for r in 0..n {
        for c in 0..n {
            // North: distance to top edge = r
            let dn = r;
            // South: distance to bottom edge = (N-1) - r
            let ds = n - 1 - r;
            // West: distance to left edge = c
            let dw = c;
            // East: distance to right edge = (N-1) - c
            let de = n - 1 - c;

            for &d in &[dn, ds, dw, de] {
                if d > 0 {
                    total += ray_actions[d as usize];
                }
            }
        }
    }
    total
}

/// Returns the grammar-maximum number of placement actions for an NxN board.
/// This is N*N*3 (flat, wall, cap) regardless of piece reserves.
pub fn total_placement_actions(size: u8) -> u32 {
    (size as u32) * (size as u32) * 3
}

/// Returns the total grammar-maximum action count (placements + movements).
pub fn total_actions(size: u8) -> u32 {
    total_placement_actions(size) + total_movement_actions(size)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- binom tests --

    #[test]
    fn binom_basic() {
        assert_eq!(binom(0, 0), 1);
        assert_eq!(binom(5, 0), 1);
        assert_eq!(binom(5, 5), 1);
        assert_eq!(binom(5, 2), 10);
        assert_eq!(binom(7, 3), 35);
        assert_eq!(binom(3, 4), 0); // k > n
    }

    // -- actions_per_ray_distance for carry_limit = 8 --

    #[test]
    fn actions_per_ray_8x8() {
        assert_eq!(actions_per_ray_distance(8, 1), 8);
        assert_eq!(actions_per_ray_distance(8, 2), 36);
        assert_eq!(actions_per_ray_distance(8, 3), 92);
        assert_eq!(actions_per_ray_distance(8, 4), 162);
        assert_eq!(actions_per_ray_distance(8, 5), 218);
        assert_eq!(actions_per_ray_distance(8, 6), 246);
        assert_eq!(actions_per_ray_distance(8, 7), 254);
    }

    // -- total_movement_actions for all sizes --

    // Movement counts are derived from the formula:
    //   total = sum over all (r,c) in 0..N, all 4 directions, of
    //           actions_per_ray_distance(carry_limit=N, d_to_edge)
    //
    // Due to the symmetry property APR(N, d) + APR(N, N-1-d) = APR(N, N-1),
    // this simplifies to N*N * 2 * APR(N, N-1) for an NxN board.
    //
    // Per-size expected values (spec §2.3):
    //   3x3:   108     4x4:   448     5x5:  1,500
    //   6x6: 4,464     7x7: 12,348    8x8: 32,512

    #[test]
    fn movement_actions_3x3() {
        assert_eq!(total_movement_actions(3), 108);
    }

    #[test]
    fn movement_actions_4x4() {
        assert_eq!(total_movement_actions(4), 448);
    }

    #[test]
    fn movement_actions_5x5() {
        assert_eq!(total_movement_actions(5), 1500);
    }

    #[test]
    fn movement_actions_6x6() {
        assert_eq!(total_movement_actions(6), 4464);
    }

    #[test]
    fn movement_actions_7x7() {
        assert_eq!(total_movement_actions(7), 12348);
    }

    #[test]
    fn movement_actions_8x8() {
        assert_eq!(total_movement_actions(8), 32512);
    }

    // -- total action counts (placements + movements) --

    #[test]
    fn total_actions_all_sizes() {
        // placements = N*N*3, movements from formula above
        assert_eq!(total_actions(3), 27 + 108);       // 135
        assert_eq!(total_actions(4), 48 + 448);       // 496
        assert_eq!(total_actions(5), 75 + 1500);      // 1575
        assert_eq!(total_actions(6), 108 + 4464);     // 4572
        assert_eq!(total_actions(7), 147 + 12348);    // 12495
        assert_eq!(total_actions(8), 192 + 32512);    // 32704
    }

    // -- drop sequence correctness --

    #[test]
    fn drop_sequences_k3_t2() {
        // Ordered partitions of 3 into 2 parts (each >= 1), lex order:
        // [1,2], [2,1]
        let table = TemplateTable::build(3);
        let range = table.lookup_range(3, 2);
        assert_eq!(range.count, 2);

        let seq0 = table.get_sequence(DropTemplateId(range.base_id));
        assert_eq!(seq0.drops.as_slice(), &[1, 2]);

        let seq1 = table.get_sequence(DropTemplateId(range.base_id + 1));
        assert_eq!(seq1.drops.as_slice(), &[2, 1]);
    }

    #[test]
    fn drop_sequences_k4_t3() {
        // Ordered partitions of 4 into 3 parts (each >= 1), lex order:
        // [1,1,2], [1,2,1], [2,1,1]
        let table = TemplateTable::build(4);
        let range = table.lookup_range(4, 3);
        assert_eq!(range.count, 3); // C(3,2) = 3

        let seq0 = table.get_sequence(DropTemplateId(range.base_id));
        assert_eq!(seq0.drops.as_slice(), &[1, 1, 2]);

        let seq1 = table.get_sequence(DropTemplateId(range.base_id + 1));
        assert_eq!(seq1.drops.as_slice(), &[1, 2, 1]);

        let seq2 = table.get_sequence(DropTemplateId(range.base_id + 2));
        assert_eq!(seq2.drops.as_slice(), &[2, 1, 1]);
    }

    #[test]
    fn drop_sequences_k1_t1() {
        // Trivial case: pick up 1, travel 1 => [1]
        let table = TemplateTable::build(1);
        let range = table.lookup_range(1, 1);
        assert_eq!(range.count, 1);

        let seq = table.get_sequence(DropTemplateId(range.base_id));
        assert_eq!(seq.drops.as_slice(), &[1]);
    }

    #[test]
    fn drop_sequences_k5_t5() {
        // Only one partition of 5 into 5 parts: [1,1,1,1,1]
        let table = TemplateTable::build(5);
        let range = table.lookup_range(5, 5);
        assert_eq!(range.count, 1);

        let seq = table.get_sequence(DropTemplateId(range.base_id));
        assert_eq!(seq.drops.as_slice(), &[1, 1, 1, 1, 1]);
    }

    #[test]
    fn drop_sequences_k_less_than_t_is_empty() {
        // Cannot partition 2 into 3 parts with each >= 1.
        let table = TemplateTable::build(3);
        let range = table.lookup_range(2, 3);
        assert_eq!(range.count, 0);
    }

    // -- template count for carry_limit = 8 --

    #[test]
    fn template_count_8x8() {
        let table = TemplateTable::build(8);
        // Total templates = sum_{k=1}^{8} sum_{t=1}^{8} C(k-1,t-1)
        // when t <= k, otherwise 0.
        //
        // k=1: C(0,0)=1                                                            = 1
        // k=2: C(1,0)+C(1,1)=1+1                                                   = 2
        // k=3: C(2,0)+C(2,1)+C(2,2)=1+2+1                                          = 4
        // k=4: C(3,0)+C(3,1)+C(3,2)+C(3,3)=1+3+3+1                                = 8
        // k=5: 1+4+6+4+1                                                           = 16
        // k=6: 1+5+10+10+5+1                                                       = 32
        // k=7: 1+6+15+20+15+6+1                                                    = 64
        // k=8: 1+7+21+35+35+21+7+1                                                 = 128
        // Total = 1+2+4+8+16+32+64+128 = 255
        assert_eq!(table.total_templates(), 255);
    }

    // -- sentinel at ID 0 --

    #[test]
    fn sentinel_id_zero() {
        let table = TemplateTable::build(4);
        let sentinel = table.get_sequence(DropTemplateId(0));
        assert!(sentinel.drops.is_empty());
    }

    // -- first real template starts at ID 1 --

    #[test]
    fn first_template_starts_at_1() {
        let table = TemplateTable::build(4);
        let range = table.lookup_range(1, 1);
        assert_eq!(range.base_id, 1);
        assert_eq!(range.count, 1);

        let seq = table.get_sequence(DropTemplateId(1));
        assert_eq!(seq.drops.as_slice(), &[1]);
    }

    // -- generate_partitions sanity --

    #[test]
    fn partitions_count_matches_binom() {
        for k in 1..=8u8 {
            for t in 1..=k {
                let partitions = generate_partitions(k, t);
                let expected = binom(k as u32 - 1, t as u32 - 1) as usize;
                assert_eq!(
                    partitions.len(),
                    expected,
                    "C({},{}) mismatch: got {}, expected {}",
                    k - 1,
                    t - 1,
                    partitions.len(),
                    expected
                );
                // Every partition should sum to k and have t parts.
                for p in &partitions {
                    assert_eq!(p.len(), t as usize);
                    assert_eq!(p.iter().copied().sum::<u8>(), k);
                    assert!(p.iter().all(|&v| v >= 1));
                }
            }
        }
    }
}
