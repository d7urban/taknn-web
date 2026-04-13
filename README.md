# TakNN-Web

TakNN-Web is a Tak engine and browser UI monorepo.

It currently contains:

- A Rust workspace for Tak rules, search, self-play data generation, Python bindings, and training utilities
- A Next.js web app that runs the Tak engine in the browser through WebAssembly and a Web Worker
- A fully native Rust training pipeline leveraging `tch` (libtorch) for self-play, iteration, and distillation

The long-term design is documented in [PLAN.md](PLAN.md) and [BUILD_SPEC.md](BUILD_SPEC.md). This README describes what is actually in the repository today.

## Repo Layout

```text
.
├── engine/          Rust workspace
│   ├── tak-core/    Tak rules, board/state, PTN/TPS, tensor encoding
│   ├── tak-search/  Heuristic PVS search
│   ├── tak-data/    Self-play and shard I/O
│   ├── tak-wasm/    wasm-bindgen wrapper used by the web app
│   ├── tak-python/  PyO3 bindings used by Python training code (legacy)
│   └── tak-train/   Rust training/distillation/iteration/export binaries
├── train/           Legacy Python training scripts (superseded by tak-train)
└── web/             Next.js frontend
```

## Current Features

- Play Tak on board sizes `3x3` through `8x8`
- Full legal move generation, PTN/TPS support, and reserve tracking
- Browser-based bot move search using the Rust engine compiled to WASM
- Self-play shard generation in `.tknn` format for bootstrap and iterative training
- Fully native Rust teacher-net training loop (`tak-iterate`) featuring:
  - Multi-threaded, batched neural inference via `libtorch`
  - Dirichlet root noise for AlphaZero-style exploration
  - $D_4$ Canonical Opening Family tracking to force opening diversity
  - Automated candidate generation, match evaluation, and promotion

## Requirements

Common:

- `Rust` toolchain
- `Node.js` and `npm`

Optional, depending on what you want to run:

- `wasm-pack` for rebuilding the browser WASM bundle
- PyTorch-compatible CUDA setup (`libtorch` / `tch-rs`) if you want to run the neural training iterations

## Quick Start

### Run the web app

The repository already includes a generated WASM bundle under `web/public/wasm`, so you can usually start the UI directly:

```bash
cd web
npm install
npm run dev
```

Then open `http://localhost:3000`.

The app uses:

- `web/src/app/page.tsx` for the main UI and game state
- `web/src/components/Board.tsx` for board rendering
- `web/src/components/Controls.tsx` for move controls and search info
- `web/src/engine/worker.ts` for off-main-thread search

### Rebuild the WASM bundle after engine changes

If you change Rust code in the WASM-facing engine, rebuild and copy the generated artifacts:

```bash
cd engine/tak-wasm
wasm-pack build --target web --release

cd ../../web
npm run copy-wasm
```

The Next.js config enables cross-origin isolation headers needed for browser WASM execution in `web/next.config.js`.

## Rust Workspace

From the workspace root:

```bash
cd engine
cargo check
cargo test
```

Important crates:

- `engine/tak-core` implements board state, moves, rules, PTN/TPS, symmetries, and tensor encoding
- `engine/tak-search` implements heuristic PVS search
- `engine/tak-data` implements self-play and shard read/write
- `engine/tak-wasm` exports the browser-facing `TakGame` API

## Bootstrapping Data Generation

Generate `.tknn` shards with the Rust heuristic self-play binary:

```bash
cd engine
cargo run --release -p tak-data --bin selfplay -- --board-size 6 --num-games 100 --output-dir ../train/shards
```

There is also a quota-based helper script that will generate a robust dataset across all board sizes using Dirichlet root noise:

```bash
./engine/scripts/generate_shards.sh -n 10000
```

## Neural Training Pipeline (Rust / tch-rs)

The Python training pipeline has been superseded by a vastly more performant, fully native Rust implementation using `tch`. 

To iteratively improve the network using AlphaZero-style self-play, run the `tak-iterate` loop:

```bash
cd engine
cargo run --release -p tak-train --features nn --bin tak-iterate -- \
    --board-size 6 \
    --iterations 10 \
    --games-per-iteration 200 \
    --eval-games 20 \
    --search-depth 4 \
    --search-time 1000 \
    -j 20 \
    --nn-batch-size 64 \
    --nn-batch-wait-us 500
```

This single command will:
1. Load the bootstrap heuristic shards.
2. Train an initial teacher candidate and promote it.
3. Automatically spawn workers to play neural self-play games.
4. Distribute the `games-per-iteration` across all board sizes (3x3 to 8x8) based on predefined training quotas.
5. Retrain on the sliding replay window and evaluate candidate promotions.

Additional binaries are available in `tak-train` for distillation and export:
- `tak-train` (Raw supervised training)
- `tak-distill` (Knowledge Distillation)
- `tak-export` (ONNX / binary weight export)

## Notes

- The browser bot currently uses heuristic search through the WASM engine; there is no browser neural inference path wired into the UI yet.
- The repository may contain generated artifacts such as checkpoints and WASM files. Rebuild them when changing the underlying engine or training code.
