# TakNN-Web: Implementation Plan

Train a neural network to play Tak (https://en.wikipedia.org/wiki/Tak_(game)) across all board sizes (3x3–8x8), then ship a browser-based player on Vercel using the trained net. Everything play-related must run client-side.

Primary optimization target: training speed on a single RTX 3090.
Secondary optimization target: browser inference/search speed.

### Rule Assumptions

Under standard Tak rules: the first two plies are placements of the opponent's flat stone, roads win by connecting opposite board edges, movement is orthogonal, and a capstone may flatten a standing stone only when the capstone itself makes the final step onto it. All design choices below rely only on these facts.

---

## Background

The fastest practical route is expert iteration with a search teacher, not a pure AlphaZero replica. The strongest current public consumer-hardware Tak result is Topaz, which uses an efficiently updatable neural network trained on roughly 1 billion self-play positions, focused on 6x6 Tak with 2 komi, and scores positively against the GPU-based TakZero at fast time controls.

Tak's rules favor a size-general design. The board sizes and piece counts:

| Size | Stones | Capstones | Carry Limit |
|------|--------|-----------|-------------|
| 3x3  | 10     | 0         | 3           |
| 4x4  | 15     | 0         | 4           |
| 5x5  | 21     | 1         | 5           |
| 6x6  | 30     | 1         | 6           |
| 7x7  | 40     | 2         | 7           |
| 8x8  | 50     | 2         | 8           |

Only the top N stones of a stack affect legal movement on an NxN board, so an 8x8-universal encoder needs only the top 8 layers plus aggregated buried-flat counts underneath.

The main competitive battleground is 6x6 with 2 komi. That is the primary strength target.

---

## Hard Requirements

- **One architecture family only**: size-conditioned residual network, no NNUE in mainline.
- **6x6 as the strength target**: peak-human-or-better.
- **3x3 as solver/regression only**: do not spend training budget here.
- **4x4 and 5x5 as curriculum and shipped support**.
- **7x7/8x8 as optional stretch goals**, not blockers.
- **Browser engine**: distilled small residual net + alpha-beta/PVS search, deployed on Vercel.

### Not Required for v1

- NNUE
- Pure neural MCTS
- Parity strength across all board sizes
- Giant self-play volume before shipping

---

## Frozen Design Decisions

These six choices are locked before implementation begins.

### Decision 1: Policy Head — Legal-Move Scorer, Not Additive Factorized

Do not use an additive semi-factorized policy head or sequential autoregressive decoding. Use a single-pass trunk + legal-move scorer head.

**Why not additive factorization:**

The 8x8 movement combinatorics are too large for an additive scheme to remain clean. For a directed ray with distance d to the edge, the legal `(pickup_count, drop_template)` combinations are:

| d | combinations |
|---|-------------|
| 1 | 8 |
| 2 | 36 |
| 3 | 92 |
| 4 | 162 |
| 5 | 218 |
| 6 | 246 |
| 7 | 254 |

With 32 directed rays of each distance on 8x8, total legal movement combinations = 32,512. Adding placements (64 × 3 = 192), the full 8x8 action universe is ~32,704. The `L_drop[src, dir, k, template]` tensor alone implies ~56,896 valid slots — the factorization is not buying enough simplicity to justify the additive-independence assumption between source, direction, pickup, and drop.

**Design:**

One trunk forward pass. Then:

1. The engine enumerates legal moves.
2. The network scores only those legal moves.
3. Softmax is taken over the legal-move list.

For each legal move `m`, build a compact descriptor and score it with a tiny shared MLP:

```
logit(m) = MLP([g, h_src, h_dst, path_pool, discrete_embs, flags])
```

**Move descriptor inputs:**

- `g` — global board embedding (pooled from trunk output)
- `h_src` — source-square embedding from trunk spatial output
- `h_dst` — destination / final-square embedding from trunk spatial output
- `path_pool` — pooled embedding over traversed squares (for stack moves)
- Learned embeddings for discrete features:
  - Move type (placement vs movement)
  - Placement piece type (flat, wall, capstone)
  - Direction
  - Pickup count
  - Drop template ID
  - Travel length
- Binary flags:
  - Capstone flatten
  - Enters occupied square
  - Opening-phase placement rule

For placements, `path_pool` is zero and `h_dst` carries the local board context.
For stack moves, the path features carry the interaction that the additive scheme was missing.

**Why this works:**

- No independence assumption between source, direction, pickup, and drop.
- No giant dense output tensor tied to the worst-case 8x8 grammar.
- Browser-compatible: actual legal move counts in real positions are much smaller than the 32,512 grammar maximum. The expensive part remains the trunk, not the tiny move-scoring MLP.

**Checkpoint 1 must produce:**

- Exact per-size move tables
- D4 transform tables for move IDs/descriptors
- Vectorized legal-move descriptor builder
- Regression test confirming the 8x8 counts above (32,512 movements + 192 placements = 32,704 max actions)

### Decision 2: Symmetry Augmentation — D4 Only, No Color Swap

Use all 8 geometric symmetries of the square board: the dihedral group D4.

- Identity
- Rotations 90 / 180 / 270
- Horizontal reflection
- Vertical reflection
- Main-diagonal reflection
- Anti-diagonal reflection

**No color-swap augmentation in v1.**

Rationale: D4 is always valid on square Tak boards. It preserves legal movement structure, road goals, and opening legality. Color swap adds bookkeeping complexity around side-to-move, reserves, opening-phase state, and teacher targets for relatively little gain.

**Application rule:**

- Training augmentation = D4 only, no color swap.
- Apply to both input and policy target:
  - Transform source/destination squares
  - Remap directions
  - Remap drop path orientation
  - Leave pickup count unchanged
- Implement once in the move codec and reuse everywhere.

### Decision 3: Position Thinning — Exact Rule-Based Tactical Detector First

Do not depend on the heuristic evaluator to define "tactical." Use an exact, move-generator-based tactical flag from day one.

**Initial tactical flag** — a position is tactical if any of these are true:

1. **Road-in-1 exists for either side**: generate legal moves for White and Black separately, test whether any move creates an immediate road win.
2. **Forced defense condition**: the side not to move has at least one road-in-1, and the side to move has at most B=2 blocking replies.
3. **Capstone tactical availability**: any legal capstone flatten move exists for either side.
4. **Endgame compression**: empty squares ≤ 2N (where N = board size), or either player has reserves ≤ N.

**Sampling policy:**

| Phase | Keep rate |
|-------|-----------|
| Tactical | every ply |
| Semi-tactical | every 2 plies |
| Quiet | every 4 plies |

**Later additions** (once search is competent):

- Root PV changed between depths
- Root score swing above threshold
- Fail-high / fail-low at root
- Large concentration of nodes in tactical extensions

The initial detector must be purely rule-based and exact.

### Decision 4: Checkpoint 7 After 6x6 Push, Not Overlapping

For one person, production deployment work happens after the 6x6 research push, not in parallel.

**Sequence:**

1. Checkpoints 1–2: dev browser UI + engine
2. Checkpoints 3–6: data, evaluation, training, 6x6 strength push
3. Checkpoint 7: production hardening and deployment

The only overlap allowed is lightweight profiling hooks and model-loading stubs, not full production integration.

### Decision 5: Opening Diversity — Forced Family Balancing + Explicit Book

Two mechanisms, not one.

**During training — forced opening diversity:**

- Define an opening family by the D4-canonicalized first 4 plies.
- Maintain per-family quotas in self-play.
- During plies 1–4: sample from underrepresented families first.
- During plies 1–6: add root noise.
- During plies 1–8: keep nonzero temperature.
- After ply 8: go near-greedy.

This prevents collapse into a few early 6x6 lines.

**Replay balancing:**

- Cap the fraction of any one opening family in the training window.
- Oversample rare but high-quality opening families when needed.

**For the shipped engine — explicit opening book:**

- Generate from high-search self-play plus gauntlet matches after the 6x6 teacher stabilizes.
- Canonicalize by D4.
- Keep positions up to ply 6 or 8.
- Store top 1–3 moves with support count and value estimate.
- Only retain entries above minimum support and stability thresholds.

**Browser use:** consult book first; if in book, play or shortlist from book; otherwise fall back to search.

### Decision 6: Size Handling — FiLM Conditioning, Not Heavy Per-Size Adapters

Do not build separate per-size adapter subnets. Use one shared trunk with lightweight size conditioning.

**Design:**

- Size embedding `e(s)` for `s ∈ {3, 4, 5, 6, 7, 8}`.
- Embedding dimension: 16 for student, 32 for teacher.
- Each residual block gets channel-wise FiLM parameters from `e(s)`:
  ```
  gamma_s = MLP_gamma(e(s))
  beta_s  = MLP_beta(e(s))
  y = gamma_s ⊙ norm(x) + beta_s
  ```
  Applied after normalization in each block.
- Heads also receive the same size embedding.

**Implications:**

- One weights file per model, not one per size.
- Negligible parameter overhead.
- Student and teacher share the same conditioning scheme.
- No separate convolutions.
- No explosion in browser model size.

If later calibration drift appears, add tiny per-size value-bias scalars at the very end. Nothing heavier unless there is clear evidence it is needed.

---

## 1) Single Mainline Architecture

### Encoder

One universal encoder for 3x3 through 8x8:

- Board padded to 8x8
- Per square: exact representation of the top 8 stack layers
- Buried-flat counts by color below the explicit top layers
- Side to move
- Board size (consumed by FiLM conditioning — see Decision 6)
- Reserves remaining
- Capstones remaining
- Opening-placement phase flag
- Carry limit
- Komi / rules variant flags
- Repetition / move-count state as needed

### Model Family

**Teacher** (native only): 10–12 residual blocks, 128 channels, FiLM size embedding dim 32.
**Student** (browser): 6 residual blocks, 64 channels, FiLM size embedding dim 16, same input schema and policy grammar, distilled from the teacher.

One shared trunk with FiLM size conditioning + shared heads. One weights file per model.

NNUE becomes a post-v1 research branch only if profiling later shows net eval — not move generation or transposition lookups — is the dominant browser bottleneck.

### Policy Head

Legal-move scorer head (see Decision 1). One trunk forward pass, then a shared MLP scores each legal move from its compact descriptor. Not autoregressive, not additive-factorized.

### Value Head

- WDL head
- Scalar score head for flat-margin / win-margin proxy
- Auxiliary tactical heads:
  - Immediate road threat
  - Immediate block threat
  - Capstone-flatten opportunity
  - Reserve exhaustion / endgame phase

These auxiliary heads help the network learn Tak-specific tactics earlier and reduce wasted self-play.

---

## 2) Search Design

Mainline search is alpha-beta / PVS, not MCTS.

Rationale: training speed is the primary objective; Tak has strong tactical structure and good pruning opportunities; browser search speed matters more than "simulation purity."

### Core Search Stack

- Iterative deepening PVS
- Transposition table
- Killer/history move ordering
- Aspiration windows
- Policy-prior move ordering
- Tak-specific extensions for immediate road threats and forced responses
- Late-move reductions only after tactical correctness is established
- No speculative null-move early on; Tak is too tactical to assume it is safe

The student net is used mainly for: move ordering, leaf evaluation, root policy prior.

---

## 3) Data Pipeline

### Storage Format

Store positions as compact binary shards, not JSON, not PTN-derived text.

Per record:
- Packed board tensor
- Board size
- Side to move
- Reserves / capstones left
- Sparse legal-move target distribution from search
- Teacher value
- Final game outcome
- Flat-margin target
- Metadata: source model, search depth/nodes, game id, ply index

Compress shards with zstd. Keep shard sizes large enough for sequential IO efficiency.

### Position Sampling

Use thinning with exact rule-based tactical detection (see Decision 3).

| Phase | Keep rate |
|-------|-----------|
| Tactical | every ply |
| Semi-tactical | every 2 plies |
| Quiet | every 4 plies |

Upweight decisive tactical turns and late endgames. That reduces correlation and keeps training throughput high.

### Replay Buffer

Windowed replay:

- 70% recent data
- 30% older stable data
- Size-balanced quotas so 6x6 dominates but smaller sizes do not disappear
- Optional quality weighting by search depth / node count

### Data Generation Streams

Three streams:

1. Heuristic-engine self-play for bootstrapping
2. Teacher-search labeling on sampled positions
3. Net-guided self-play after the first competent model exists

Do not wait for pure self-play to do everything.

### Training Input Path

- Manifest + shard index
- Async prefetch
- Memory-mapped reads where practical
- On-the-fly D4 symmetry augmentation (see Decision 2)
- Mixed precision training

Goal: keep the 3090 saturated and never blocked on decompression or Python overhead.

---

## 4) Evaluation Methodology

This must exist by Checkpoint 3.

### Rating Pools

Separate Elo pools for: 4x4, 5x5, 6x6. Treat 7x7 and 8x8 separately later.

### Anchors

- Anchor A: random/legal baseline
- Anchor B: shallow heuristic PVS
- Anchor C: stronger heuristic PVS
- Anchor D: current promoted model
- Anchor E: previous promoted model

If external engines can be integrated later, use them, but do not depend on them for the core loop.

### Promotion Rule

Fixed gating rule:

- Candidate must beat current promoted model by a target margin at fixed time controls
- Must not regress on tactical suites
- Must not exceed browser latency budget

### Tactical Test Suites

Curated suites for:

- One-move road wins
- Forced road blocks
- Capstone flattening
- Wall timing
- Stack-distribution tactics
- Reserve exhaustion
- Flat-count endgames

These are critical in Tak. Elo alone will hide tactical regressions.

### Browser-Performance Gates

For each promoted student model, record:

- Model size
- Eval latency in Chrome on a reference machine
- Nodes/sec with search enabled
- Median move time on 5x5 and 6x6 at fixed time budgets

A model that gains 25 Elo but doubles browser latency may not be promotable.

---

## 5) Training Recipe

### Phase A: Engine-First Bootstrapping

Start with a decent non-neural Tak engine before any RL:

- Alpha-beta / PVS
- Hand-coded static eval
- Move ordering
- Road-threat detection
- Flat-win heuristics
- Opening randomization

This gives a usable teacher immediately and makes every later phase faster.

### Phase B: Supervised Distillation from Search

Generate a large corpus of randomized positions on each size and label them with:

- Root move distribution from search
- Game result / deep-search value
- Flat-margin estimate
- Threat labels

This is the fastest path to first strength.

### Phase C: Expert Iteration

Replace static eval with the network, keep alpha-beta as the teacher, and iterate:

1. Search-guided self-play
2. Retrain net
3. Stronger search with updated net
4. Repeat

This is faster than waiting for pure self-play from scratch to discover Tak strategy.

### Phase D: Curriculum by Board Size

Train in this order:

1. **3x3**: regression tests only; verify perfect play behavior.
2. **4x4**: no-capstone wall tactics.
3. **5x5**: first capstone regime.
4. **6x6**: main strength target.
5. **7x7 / 8x8**: optional stretch, train later with transfer from 6x6.

Shared trunk with FiLM size conditioning (Decision 6). The trunk learns Tak-wide stack/road concepts; FiLM modulation absorbs reserve/capstone/carry-limit differences.

### Phase E: Browser Distillation

After the teacher is strong:

- Distill teacher search into the small browser student (6 blocks, 64 channels)
- Quantize to INT8
- Export to ONNX
- Run with ONNX Runtime Web: WebGPU when available, WASM as fallback

---

## 6) Opening Strategy

See Decision 5 for full specification.

**Training:** forced family-balanced diversity using D4-canonicalized opening families, per-family quotas, root noise through ply 6, nonzero temperature through ply 8, near-greedy after.

**Shipped engine:** explicit opening book generated from stabilized 6x6 teacher, canonicalized by D4, positions up to ply 6–8, top 1–3 moves per position.

---

## 7) Checkpoints

Each checkpoint must run something new.

### Checkpoint 1 — Full Legal Engine + Browser UI

**Runs:** human-vs-human Tak in browser for all sizes.

Deliverables:
- Rust rules engine
- WASM build
- Legal move generation
- PTN/TPS support, import/export
- Per-size reserves/capstone rules
- Exact per-size move tables and D4 transform tables for move IDs/descriptors
- Vectorized legal-move descriptor builder
- Regression tests, especially 3x3
- Regression test confirming 8x8 counts: 32,512 movements + 192 placements = 32,704 max actions

**Time:** 1–2 weeks

### Checkpoint 2 — Heuristic Search Bot

**Runs:** browser bot for all sizes using heuristic PVS.

Deliverables:
- Iterative deepening PVS
- Transposition table
- Move ordering
- Tak-specific tactical checks (road-in-1, forced defense, capstone flatten — see Decision 3)
- Simple heuristic evaluation
- Worker-thread search

This is the first actually playable bot and the first data generator.

**Time:** 2–4 weeks (this is the big engine block)

### Checkpoint 3 — Data Pipeline + Trainer + Evaluator

**Runs:** first trained model offline; browser can load student net for move ordering/eval on 4x4 and 5x5.

Deliverables:
- Self-play shard writer with zstd compression
- Dataset manifests
- Replay sampler with rule-based thinning (Decision 3)
- Training loop with D4 augmentation (Decision 2)
- Distillation targets
- Elo harness with anchors A–E
- Tactical test suites
- Promotion/gating script with browser-performance gates

**Time:** 1–2 weeks engineering, then immediate first training runs

### Checkpoint 4 — Strong 4x4 / Competent 5x5

**Runs:** first net-guided browser bot that is meaningfully better than pure heuristics.

Deliverables:
- Teacher residual net (10–12 blocks, 128 ch, FiLM dim 32)
- Distilled student residual net (6 blocks, 64 ch, FiLM dim 16)
- Legal-move scorer policy head (Decision 1)
- Policy-guided move ordering
- Value-guided leaf eval
- Stable training logs

**Training time on 3090:** 3–7 days once the pipeline is working

### Checkpoint 5 — Capstone-Competent 5x5

**Runs:** browser bot with reliable capstone tactics and nontrivial opening/midgame play.

Deliverables:
- Tactical auxiliary heads active
- Stronger search labeling
- Larger 5x5 training mix
- Improved student quantization/export

**Training time on 3090:** 1–2 additional weeks

### Checkpoint 6 — 6x6 Mainline Push

**Runs:** serious native 6x6 engine and distilled browser student.

Deliverables:
- 6x6-heavy curriculum
- Larger search budgets
- Stronger gating harness
- Opening diversity controls (Decision 5: family-balanced self-play)
- Opening book generation from stabilized teacher
- Value-target stabilization
- Versioned replay windows

This is the core research phase. "Peak human" becomes plausible here.

**Training time on 3090:** 6–10 weeks after Checkpoint 5 is stable

### Checkpoint 7 — Production Browser Player

**Runs:** deployable Vercel app with client-side search and inference.

**Sequenced after Checkpoint 6** (Decision 4). No parallel production work during the 6x6 push beyond lightweight profiling hooks and model-loading stubs.

Deliverables:
- Next.js frontend
- Model asset loading via CDN
- Web Worker search
- Quantized INT8 student model via ONNX Runtime Web
- WebGPU primary, WASM fallback
- Cross-origin isolation headers for SharedArrayBuffer / multithreaded WASM
- Opening book integration
- Telemetry/profiling hooks

At this point you ship the small residual student. No NNUE in the mainline.

**Time:** 1–2 weeks after Checkpoint 6

### Checkpoint 8 — Optional 7x7 / 8x8 Expansion (Stretch)

**Runs:** all-size trained models beyond "rules-complete."

Deliverables:
- Curriculum transfer from 6x6
- FiLM conditioning already handles size differences; fine-tune on 7x7/8x8 data
- Separate Elo pools
- Larger browser time budgets or reduced depth defaults

**Training time on 3090:** 8–16+ additional weeks; strength may still trail 6x6 materially.

This is clearly a stretch track.

---

## 8) Time Budget

Assuming one developer-researcher, one RTX 3090, and a decent multicore CPU:

| Phase | Time |
|-------|------|
| Engine + browser legality + heuristic PVS (CP 1–2) | 3–6 weeks |
| Data pipeline + trainer + eval harness (CP 3) | 1–2 weeks |
| Strong 4x4 / 5x5 (CP 4–5) | 1–3 weeks |
| 6x6 to peak-human target (CP 6) | 6–10 weeks |
| Production deployment (CP 7) | 1–2 weeks |
| **Total to shipped 6x6-focused product** | **12–23 weeks** |
| 7x7 / 8x8 stretch (CP 8) | 8–16+ more weeks |

Peak-human 6x6 on a single 3090: budget about 3–5 months total project time, with 6–10 weeks of that being the core 6x6 training/refinement phase after the infrastructure is mature.

All sizes at comparable strength: not a reasonable mainline promise on this hardware/schedule.

---

## Bottom Line

Build a good Tak engine first, build the data and evaluation system second, train one residual model family only, target 6x6 peak-human strength as the real milestone, ship a browser residual student on Vercel, treat 7x7/8x8 and NNUE as later research branches.
