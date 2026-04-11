#!/usr/bin/env bash
#
# Generate self-play training shards for all board sizes,
# distributed according to the training size quotas.
#
# Counts existing shards per board size and only generates enough
# new games to bring each size up to its target proportion.
#
# Usage:
#   ./scripts/generate_shards.sh [TARGET_TOTAL_GAMES] [OUTPUT_DIR]
#   ./scripts/generate_shards.sh -n 20000 -d 6 -t 3000
#
# Options:
#   -n, --num-games       Target total games across all sizes (default: 10000)
#   -o, --output-dir      Output directory (default: train/shards)
#   -d, --depth           Search depth per move (default: 6)
#   -t, --time-ms         Max time per move in ms (default: 1000)
#   -j, --threads         Worker threads, 0 = all cores (default: 0)
#       --seed-base       Base RNG seed (default: random)
#       --games-per-shard Games per shard file (default: 50, must match existing shards)
#   -h, --help            Show this help text
#
# Defaults: 10000 target games → train/shards/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$ENGINE_DIR/.." && pwd)"

usage() {
    cat <<EOF
Usage:
  ./scripts/generate_shards.sh [TARGET_TOTAL_GAMES] [OUTPUT_DIR]
  ./scripts/generate_shards.sh -n 20000 -d 6 -t 3000

Options:
  -n, --num-games       Target total games across all sizes (default: 10000)
  -o, --output-dir      Output directory (default: train/shards)
  -d, --depth           Search depth per move (default: 6)
  -t, --time-ms         Max time per move in ms (default: 1000)
  -j, --threads         Worker threads, 0 = all cores (default: 0)
      --seed-base       Base RNG seed (default: random)
      --games-per-shard Games per shard file (default: 50, must match existing shards)
  -h, --help            Show this help text
EOF
}

TARGET_TOTAL="10000"
OUTPUT_DIR="$REPO_DIR/train/shards"
SEED_BASE="${SEED_BASE:-$RANDOM}"
THREADS="${THREADS:-0}"
DEPTH="${DEPTH:-6}"
TIME_MS="${TIME_MS:-1000}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-50}"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-games)
            TARGET_TOTAL="${2:?missing value for $1}"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="${2:?missing value for $1}"
            shift 2
            ;;
        -d|--depth)
            DEPTH="${2:?missing value for $1}"
            shift 2
            ;;
        -t|--time-ms)
            TIME_MS="${2:?missing value for $1}"
            shift 2
            ;;
        -j|--threads)
            THREADS="${2:?missing value for $1}"
            shift 2
            ;;
        --seed-base)
            SEED_BASE="${2:?missing value for $1}"
            shift 2
            ;;
        --games-per-shard)
            GAMES_PER_SHARD="${2:?missing value for $1}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if (( ${#POSITIONAL[@]} >= 1 )); then
    TARGET_TOTAL="${POSITIONAL[0]}"
fi
if (( ${#POSITIONAL[@]} >= 2 )); then
    OUTPUT_DIR="${POSITIONAL[1]}"
fi
if (( ${#POSITIONAL[@]} > 2 )); then
    echo "Too many positional arguments" >&2
    usage >&2
    exit 2
fi

for var_name in TARGET_TOTAL DEPTH TIME_MS THREADS GAMES_PER_SHARD; do
    value="${!var_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$var_name must be a non-negative integer, got: $value" >&2
        exit 2
    fi
done

# Size quotas (must match train/taknn/train/config.py)
#   3: 2%, 4: 10%, 5: 25%, 6: 55%, 7: 5%, 8: 3%
SIZES=(  3    4    5    6    7    8  )
PCTS=(   2   10   25   55    5    3  )

mkdir -p "$OUTPUT_DIR"

# Count existing games per size from shard filenames.
declare -A EXISTING
for size in "${SIZES[@]}"; do
    count=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*_${size}x${size}.tknn" | wc -l)
    EXISTING[$size]=$(( count * GAMES_PER_SHARD ))
done

existing_total=0
for size in "${SIZES[@]}"; do
    existing_total=$(( existing_total + EXISTING[$size] ))
done

echo "=== Self-play shard generation ==="
echo "Target total: $TARGET_TOTAL games"
echo "Existing:     $existing_total games"
echo "Output dir:   $OUTPUT_DIR"
echo "Depth:        $DEPTH"
echo "Time/move:    ${TIME_MS}ms"
echo "Threads:      $THREADS (0 = all cores)"
echo
echo "Current distribution:"
for i in "${!SIZES[@]}"; do
    size=${SIZES[$i]}
    pct=${PCTS[$i]}
    target=$(( TARGET_TOTAL * pct / 100 ))
    have=${EXISTING[$size]}
    if (( have >= target )); then
        status="OK"
    else
        status="need +$(( target - have ))"
    fi
    echo "  ${size}x${size}: ${have} / ${target} (quota ${pct}%) — ${status}"
done
echo

echo "Building selfplay binary..."
cargo build --release --manifest-path "$ENGINE_DIR/Cargo.toml" -p tak-data --bin selfplay
SELFPLAY="$ENGINE_DIR/target/release/selfplay"
echo

generated=0
run_start=$SECONDS
for i in "${!SIZES[@]}"; do
    size=${SIZES[$i]}
    pct=${PCTS[$i]}

    target=$(( TARGET_TOTAL * pct / 100 ))
    if (( target == 0 )); then
        target=1
    fi
    have=${EXISTING[$size]}

    if (( have >= target )); then
        echo "--- ${size}x${size}: already at quota (${have} >= ${target}), skipping ---"
        echo
        continue
    fi

    need=$(( target - have ))
    seed=$(( SEED_BASE + size * 1000 ))

    echo "--- ${size}x${size}: generating ${need} games to reach ${target} ---"
    size_start=$SECONDS
    "$SELFPLAY" \
        -s "$size" \
        -n "$need" \
        -d "$DEPTH" \
        -t "$TIME_MS" \
        -j "$THREADS" \
        --seed "$seed" \
        --games-per-shard "$GAMES_PER_SHARD" \
        --output-dir "$OUTPUT_DIR"
    elapsed=$(( SECONDS - size_start ))
    echo "    ${size}x${size}: finished in ${elapsed}s"
    generated=$(( generated + need ))
    echo
done

total_elapsed=$(( SECONDS - run_start ))
total_shards=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.tknn" | wc -l)
echo "=== Done: generated ${generated} new games, ${total_shards} total shard files in ${total_elapsed}s ==="
