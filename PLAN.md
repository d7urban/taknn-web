an implementation plan for teaching a neural net to play Tak (https://en.wikipedia.org/wiki/Tak_(game)) covering all board sizes and their respective piece and capstone differences. The final goal is to have a web based player using the trained neural net that can be deployed on Vercel.app, so everything play-related must be able to run in a browser.
Include net architecture, estimated training time on a single RTX 3090 to reach peak human skill level.
Split the plan into several checkpoints where every checkpoint can run something new. The primary consideration is the speed of training and the secondary consideration is the speed of simulations when playing.

I’m checking the current Tak ruleset details, board-size-specific piece counts, and the state of Tak engines so the plan is grounded rather than generic. Then I’ll convert that into staged checkpoints optimized first for training throughput and second for browser inference speed.

The rules details are straightforward; the harder part is sizing the learning system so one RTX 3090 is enough to get to strong human level without designing something too heavy for browser play. I’m checking existing Tak engine work now so the checkpoint sequence and training-time estimate are realistic.

Thought for 9m 39s

The fastest practical route is not a pure AlphaZero clone. For Tak, the current public evidence points toward strong search + fast neural evaluation + distillation: AlphaZero-style Tak engines exist and are strong, but the strongest current public consumer-hardware result I found is Topaz, which says it uses an efficiently updatable neural network trained on roughly 1 billion self-play positions, is focused on 6x6 Tak with 2 komi, and scores positively against the GPU-based TakZero at fast time controls. That strongly suggests your final system should be a browser-fast student model trained from a stronger offline teacher, not a heavy browser MCTS stack.

Tak’s rules also favor a size-general design. The board sizes relevant here are 3x3 through 8x8; the standard stone/capstone counts are 3x3: 10/0, 4x4: 15/0, 5x5: 21/1, 6x6: 30/1, 7x7: 40/2, 8x8: 50/2. Carry limit equals board size, capstones can flatten standing stones, and standing stones and capstones cannot normally be stacked upon. For training, that matters because only the top N stones of a stack can affect legal movement on an NxN board, so an 8x8-universal encoder only needs the top 8 layers exactly, plus aggregated buried-flat counts underneath.

If the target is “peak human” rather than “best public engine,” I would benchmark mainly on 6x6 with 2 komi, because that is the main competitive battleground in current Tak engine and tournament practice.

Recommended system architecture

Core game/search engine

Write the rules, move generator, PTN/TPS support, repetition/draw handling, zobrist hashing, and search in Rust, compiled both to native for training and WASM for browser play.
Parameterize board size, reserves, capstones, carry limit, and komi so one engine handles 3x3–8x8.

Teacher model for offline training

One shared 8x8-padded residual policy-value net.
Input: exact top-8 stack layers per square, buried flat counts by color, side-to-move, reserves left, capstones left, board-size one-hot, komi, and a few global counters.
Trunk: 10–12 residual blocks, 128 channels.
Heads:
Policy: factorized move grammar rather than a giant flat move softmax.
Value: WDL + scalar flat-margin prediction.
Auxiliary: road-threat map, connectivity/ownership hints, reserve exhaustion, immediate tak-threat flags.

Student model for browser play

Distill the teacher+search into either:
a small residual net (6 blocks, 64 channels) for the first shipped version, then
an incrementally updatable NNUE-style value/policy net for the final fast browser engine.
Final browser player uses iterative deepening PVS/alpha-beta, TT, killer/history ordering, policy-guided move ordering, and a small set of Tak-specific threat extensions.

That split is the right tradeoff: residual nets are easier to train well; incremental nets are much faster at search time.

Why not pure browser MCTS

Tak has a large and growing branching factor, and existing strong Tak results on consumer hardware are currently coming from search-heavy engines rather than pure client-side neural-MCTS. Pure AlphaZero-style self-play is also compute-hungry: AlphaZero reached superhuman chess/shogi levels quickly only with very large TPU resources, while KataGo’s main contribution was making AlphaZero-style self-play far more sample-efficient through architectural and training improvements. On a single 3090, you want every shortcut available.

Action representation

Do not use a single monolithic 60k+ action head first.

Use a factorized policy:

select source square or reserve placement mode,
select piece type or move direction,
select pickup count,
select drop partition.

That is much more sample-efficient for Tak because the move grammar is structured and board-size dependent. It also makes size-general training easier.

Training recipe
Phase A: engine-first bootstrapping

Start with a decent non-neural Tak engine before any RL:

alpha-beta / PVS
hand-coded static eval
move ordering
road-threat detection
flat-win heuristics
opening randomization

This gives you a usable teacher immediately and makes every later phase faster.

Phase B: supervised distillation from search

Generate a large corpus of randomized positions on each size and label them with:

root move distribution from search
game result / deep-search value
flat-margin estimate
threat labels

This is the fastest path to first strength.

Phase C: expert iteration

Replace static eval with the network, keep alpha-beta as the teacher, and iterate:

search-guided self-play,
retrain net,
stronger search with updated net,
repeat.

This is faster than waiting for pure self-play from scratch to discover Tak strategy.

Phase D: curriculum by board size

Train in this order:

3x3: regression tests only; use it to verify perfect play behavior.
4x4: no-capstone wall tactics.
5x5: first capstone regime.
6x6: main strength target.
7x7
8x8

Use shared trunk + per-size adapters/heads rather than six unrelated models. The trunk learns Tak-wide stack/road concepts; adapters absorb reserve/capstone/carry-limit differences.

Phase E: browser distillation

After the teacher is strong:

distill teacher search into the small browser student,
quantize to INT8,
export to ONNX,
run with ONNX Runtime Web using WebGPU when available and WASM as fallback. ORT’s docs explicitly position WASM for lightweight models and WebGPU for more compute-intensive ones.
Checkpoints
Checkpoint 1 — legal browser Tak on all sizes

Runs something new: a complete browser Tak client on 3x3–8x8 with human-vs-human, PGN/PTN/TPS import/export, random bot, and per-size rules.

Deliverables:

Rust rules engine compiled to WASM
React/Next.js UI
move generation/property tests
solved 3x3 regression tests

This is pure engineering, no GPU needed yet.

Checkpoint 2 — fast classical bot in browser

Runs something new: a browser alpha-beta bot that is weak but tactically coherent on every board size.

Deliverables:

iterative deepening PVS
TT
heuristic eval
move ordering
opening randomization

This becomes your data generator and sanity baseline.

Estimated effort: 1–2 weeks.

Checkpoint 3 — no-capstone learner

Runs something new: trained net for 3x3/4x4; browser can switch between heuristic and neural move ordering/eval.

Deliverables:

8x8-padded residual teacher
size-conditioned training
first distillation into small browser model
strong 4x4 play; near-perfect 3x3 behavior

Estimated 3090 time: 1–2 days of actual training.

Checkpoint 4 — capstone regime online

Runs something new: competent 5x5 neural bot in browser with capstone tactics.

Deliverables:

capstone-aware features
flattened-wall supervision
threat-map aux heads
expert-iteration loop active

Estimated 3090 time: 3–7 days beyond Checkpoint 3.

Checkpoint 5 — competitive 6x6 teacher

Runs something new: native offline teacher at serious 6x6, 2-komi strength; browser still uses compact student.

Deliverables:

6x6-heavy curriculum
larger search budget
opening book from randomized self-play
automated match harness vs existing engines/baselines

Estimated 3090 time: 2–4 weeks beyond Checkpoint 4, assuming a decent multicore CPU is generating search data in parallel.

This is where “top human” becomes plausible.

Checkpoint 6 — universal all-size model

Runs something new: one product that plays every size 3x3–8x8, with different adapters/weights loaded per size.

Deliverables:

shared trunk
per-size adapters/heads
mixed-size replay buffer
evaluation suite across all sizes

Estimated 3090 time: 2–4 additional weeks.

Checkpoint 7 — browser-fast final player

Runs something new: final Vercel-deployed client-side player with fast search in browser.

Deliverables:

distilled NNUE-style or tiny residual student
INT8 ONNX export
Web Worker search
WebGPU primary, WASM fallback
multithreaded WASM path where available

For multithreaded browser/WASM paths, set cross-origin isolation headers on Vercel; Vercel’s own guidance for SharedArrayBuffer uses Cross-Origin-Embedder-Policy: require-corp and Cross-Origin-Opener-Policy: same-origin. ORT Web also exposes WASM thread settings.

Estimated training time on one RTX 3090

This is the honest estimate:

Fastest practical route (engine distillation + self-play refinement):
3x3–5x5: about 1 week
6x6 to peak human level: about 3–6 weeks
7x7/8x8 to strong expert level: about another 4–8 weeks
Total for an all-size strong package: about 7–14 weeks
Pure AlphaZero-from-scratch route:
more like 3–6 months, with a real chance that 7x7/8x8 lag badly.

That estimate is an inference from the compute gap between large-scale AlphaZero, KataGo’s effort to recover efficiency, and the fact that the strongest current public Tak engine result I found is built around a fast updatable network trained on roughly 1 billion positions rather than a lightweight browser-MCTS setup.

Practical target strength by size
3x3: solved/test harness only; do not spend much training budget here. A 2025 paper reports 3x3 Tak is solved with Player 1 having a forced win from any starting position.
4x4: easy to get very strong.
5x5: strong in days, very strong in under two weeks.
6x6: main bottleneck; this is where most of the useful budget should go.
7x7 / 8x8: train later with transfer from 6x6, or they will absorb too much budget too early.
Deployment on Vercel

Use Vercel only as the delivery layer:

serve the Next.js app and static model assets,
cache model files on the CDN,
keep all move generation, search, and inference client-side,
use a Web Worker for search so the UI stays responsive. Vercel’s docs note static assets are cached on its CDN, which is exactly what you want for model delivery.
Bottom line

Build this as expert iteration with a search teacher, not as a pure AlphaZero replica.

If I were optimizing exactly for your stated priorities:

training speed first,
browser play speed second,

I would choose:

Rust/WASM engine
offline residual teacher
alpha-beta/PVS search teacher
distilled browser student
6x6-first curriculum, all-sizes by transfer

That is the highest-probability route to a deployable, browser-native Tak player that can plausibly reach top-human 6x6 strength on a single 3090 without turning into a multi-month compute sink.
