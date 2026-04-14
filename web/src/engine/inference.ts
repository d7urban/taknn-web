import * as ort from "onnxruntime-web";

type MoveInfo = {
  index: number;
  ptn: string;
};

type GameInfo = {
  sideToMove: "white" | "black";
  result: string;
  ply: number;
};

export type SearchInfo = {
  bestMove: string;
  score: number;
  depth: number;
  nodes: number;
  pv: string[];
  ttHits: number;
  engineMode: "book" | "neural" | "heuristic";
  modelName?: string;
};

export type WasmGame = {
  applyMoveIndex(index: number): void;
  encodeBoard(): Float32Array;
  getInfo(): GameInfo;
  getTps(): string;
  legalMoves(): MoveInfo[];
  sizeId(): number;
  undo(): boolean;
};

type WasmNeuralPolicy = {
  scoreMoves(
    game: WasmGame,
    spatialPool: Float32Array,
    globalPool: Float32Array,
  ): Float32Array;
};

type WasmModule = {
  NeuralPolicy: new (weights: Uint8Array) => WasmNeuralPolicy;
};

type NeuralRuntime = {
  modelName: string;
  policy: WasmNeuralPolicy;
  session: ort.InferenceSession;
};

type PositionAnalysis = {
  score: number;
  orderedMoves: MoveInfo[];
};

type SearchNode = {
  score: number;
  pv: string[];
};

type SearchContext = {
  deadlineMs: number;
  nodes: number;
  cacheHits: number;
  runtime: NeuralRuntime;
  analysisCache: Map<string, PositionAnalysis>;
};

type ModelCandidate = {
  modelName: string;
  trunkPath: string;
  policyPath: string;
};

const MODEL_CANDIDATES: ModelCandidate[] = [
  {
    modelName: "student-int8",
    trunkPath: "/models/student_trunk_int8.onnx",
    policyPath: "/models/student_policy.bin",
  },
  {
    modelName: "student",
    trunkPath: "/models/student_trunk.onnx",
    policyPath: "/models/student_policy.bin",
  },
  {
    modelName: "teacher",
    trunkPath: "/models/teacher_trunk.onnx",
    policyPath: "/models/teacher_policy.bin",
  },
];

const SCORE_INF = 1_000_000;
const SCORE_MATE = 100_000;

let runtimePromise: Promise<NeuralRuntime | null> | null = null;

class SearchTimeout extends Error {
  constructor() {
    super("neural search timed out");
  }
}

export async function initNeuralRuntime(
  wasmModule: WasmModule,
): Promise<NeuralRuntime | null> {
  if (runtimePromise) {
    return runtimePromise;
  }

  runtimePromise = loadNeuralRuntime(wasmModule).catch((error: unknown) => {
    console.warn("Neural runtime unavailable, falling back to heuristic search", error);
    return null;
  });

  return runtimePromise;
}

export async function selectMoveWithNeural(
  wasmModule: WasmModule,
  game: WasmGame,
  maxDepth: number,
  timeMs: number,
): Promise<SearchInfo | null> {
  const runtime = await initNeuralRuntime(wasmModule);
  if (!runtime) {
    return null;
  }

  const legalMoves = game.legalMoves();
  if (legalMoves.length === 0) {
    throw new Error("No legal moves available for neural inference");
  }

  const context: SearchContext = {
    deadlineMs: performance.now() + Math.max(timeMs, 1),
    nodes: 0,
    cacheHits: 0,
    runtime,
    analysisCache: new Map<string, PositionAnalysis>(),
  };

  const rootAnalysis = await analyzePosition(game, context);
  const fallbackMove = rootAnalysis.orderedMoves[0] ?? legalMoves[0];
  let bestMove = fallbackMove;
  let bestScore = rootAnalysis.score;
  let bestPv = [fallbackMove.ptn];
  let completedDepth = 0;

  try {
    for (let depth = 1; depth <= Math.max(1, maxDepth); depth += 1) {
      const result = await searchRoot(game, depth, context);
      if (result.pv.length > 0) {
        bestMove = legalMoves.find((move) => move.ptn === result.pv[0]) ?? bestMove;
        bestPv = result.pv;
      }
      bestScore = result.score;
      completedDepth = depth;
    }
  } catch (error: unknown) {
    if (!(error instanceof SearchTimeout)) {
      throw error;
    }
  }

  return {
    bestMove: bestMove.ptn,
    score: bestScore,
    depth: completedDepth,
    nodes: context.nodes,
    pv: bestPv,
    ttHits: context.cacheHits,
    engineMode: "neural",
    modelName: runtime.modelName,
  };
}

async function searchRoot(
  game: WasmGame,
  depth: number,
  context: SearchContext,
): Promise<SearchNode> {
  ensureTimeRemaining(context);

  const analysis = await analyzePosition(game, context);
  let alpha = -SCORE_INF;
  const beta = SCORE_INF;
  let bestScore = -SCORE_INF;
  let bestPv: string[] = [];

  for (let index = 0; index < analysis.orderedMoves.length; index += 1) {
    ensureTimeRemaining(context);
    const move = analysis.orderedMoves[index];
    game.applyMoveIndex(move.index);

    let child: SearchNode;
    if (index === 0) {
      child = await pvs(game, depth - 1, -beta, -alpha, context, 1);
    } else {
      child = await pvs(game, depth - 1, -alpha - 1, -alpha, context, 1);
      let score = -child.score;
      if (score > alpha && score < beta) {
        child = await pvs(game, depth - 1, -beta, -alpha, context, 1);
        score = -child.score;
      }
    }

    if (!game.undo()) {
      throw new Error("Failed to undo move during root neural search");
    }

    const score = -child.score;
    if (score > bestScore) {
      bestScore = score;
      bestPv = [move.ptn, ...child.pv];
    }
    if (score > alpha) {
      alpha = score;
    }
  }

  return {
    score: bestScore,
    pv: bestPv,
  };
}

async function pvs(
  game: WasmGame,
  depth: number,
  alpha: number,
  beta: number,
  context: SearchContext,
  plyFromRoot: number,
): Promise<SearchNode> {
  ensureTimeRemaining(context);
  context.nodes += 1;

  const terminalScore = scoreTerminal(game.getInfo(), plyFromRoot);
  if (terminalScore !== null) {
    return { score: terminalScore, pv: [] };
  }

  if (depth <= 0) {
    const analysis = await analyzePosition(game, context);
    return { score: analysis.score, pv: [] };
  }

  const analysis = await analyzePosition(game, context);
  if (analysis.orderedMoves.length === 0) {
    return { score: analysis.score, pv: [] };
  }

  let bestScore = -SCORE_INF;
  let bestPv: string[] = [];
  let localAlpha = alpha;

  for (let index = 0; index < analysis.orderedMoves.length; index += 1) {
    ensureTimeRemaining(context);
    const move = analysis.orderedMoves[index];
    game.applyMoveIndex(move.index);

    let child: SearchNode;
    if (index === 0) {
      child = await pvs(game, depth - 1, -beta, -localAlpha, context, plyFromRoot + 1);
    } else {
      child = await pvs(game, depth - 1, -localAlpha - 1, -localAlpha, context, plyFromRoot + 1);
      let score = -child.score;
      if (score > localAlpha && score < beta) {
        child = await pvs(game, depth - 1, -beta, -localAlpha, context, plyFromRoot + 1);
        score = -child.score;
      }
    }

    if (!game.undo()) {
      throw new Error("Failed to undo move during neural search");
    }

    const score = -child.score;
    if (score > bestScore) {
      bestScore = score;
      bestPv = [move.ptn, ...child.pv];
    }
    if (score > localAlpha) {
      localAlpha = score;
    }
    if (localAlpha >= beta) {
      break;
    }
  }

  return {
    score: bestScore,
    pv: bestPv,
  };
}

async function analyzePosition(
  game: WasmGame,
  context: SearchContext,
): Promise<PositionAnalysis> {
  ensureTimeRemaining(context);

  const tps = game.getTps();
  const cached = context.analysisCache.get(tps);
  if (cached) {
    context.cacheHits += 1;
    return cached;
  }

  const legalMoves = game.legalMoves();
  if (legalMoves.length === 0) {
    const terminal = scoreTerminal(game.getInfo(), 0) ?? 0;
    const empty = { score: terminal, orderedMoves: [] };
    context.analysisCache.set(tps, empty);
    return empty;
  }

  const boardTensor = game.encodeBoard();
  const sizeId = game.sizeId();
  const feeds: Record<string, ort.Tensor> = {
    board_tensor: new ort.Tensor("float32", boardTensor, [1, 31, 8, 8]),
    size_id: new ort.Tensor("int64", BigInt64Array.of(BigInt(sizeId)), [1]),
  };

  const outputs = await context.runtime.session.run(feeds);
  const wdl = requireTensor(outputs, "wdl");
  const margin = requireTensor(outputs, "margin");
  const spatial = requireTensor(outputs, "spatial");
  const globalPool = requireTensor(outputs, "global_pool");

  const moveProbabilities = context.runtime.policy.scoreMoves(
    game,
    toFloat32Array(spatial.data),
    toFloat32Array(globalPool.data),
  );

  if (moveProbabilities.length !== legalMoves.length) {
    throw new Error(
      `Policy scorer returned ${moveProbabilities.length} probabilities for ${legalMoves.length} legal moves`,
    );
  }

  const wdlValues = toFloat32Array(wdl.data);
  const marginValue = toFloat32Array(margin.data)[0] ?? 0;
  const scalar = ((wdlValues[0] ?? 0) - (wdlValues[2] ?? 0)) + 0.5 * marginValue;
  const orderedMoves = legalMoves
    .map((move, index) => ({ move, probability: moveProbabilities[index] ?? 0 }))
    .sort((left, right) => right.probability - left.probability)
    .map(({ move }) => move);

  const analysis = {
    score: Math.round(scalar * 500),
    orderedMoves,
  };
  context.analysisCache.set(tps, analysis);
  return analysis;
}

async function loadNeuralRuntime(
  wasmModule: WasmModule,
): Promise<NeuralRuntime> {
  ort.env.wasm.proxy = false;

  for (const candidate of MODEL_CANDIDATES) {
    const loaded = await tryLoadCandidate(wasmModule, candidate);
    if (loaded) {
      return loaded;
    }
  }

  throw new Error(
    `No browser model artifacts found. Tried: ${MODEL_CANDIDATES.map((candidate) => candidate.trunkPath).join(
      ", ",
    )}`,
  );
}

async function tryLoadCandidate(
  wasmModule: WasmModule,
  candidate: ModelCandidate,
): Promise<NeuralRuntime | null> {
  const [trunkResponse, policyResponse] = await Promise.all([
    fetch(candidate.trunkPath),
    fetch(candidate.policyPath),
  ]);

  if (!trunkResponse.ok || !policyResponse.ok) {
    return null;
  }

  const [trunkBytes, policyBytes] = await Promise.all([
    trunkResponse.arrayBuffer(),
    policyResponse.arrayBuffer(),
  ]);

  const session = await ort.InferenceSession.create(trunkBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  return {
    modelName: candidate.modelName,
    policy: new wasmModule.NeuralPolicy(new Uint8Array(policyBytes)),
    session,
  };
}

function requireTensor(
  outputs: Record<string, ort.Tensor>,
  name: string,
): ort.Tensor {
  const tensor = outputs[name];
  if (!tensor) {
    throw new Error(`ORT output ${name} missing`);
  }
  return tensor;
}

function toFloat32Array(
  data: ort.Tensor["data"],
): Float32Array {
  if (data instanceof Float32Array) {
    return data;
  }
  if (Array.isArray(data)) {
    return Float32Array.from(data);
  }
  return Float32Array.from(data as ArrayLike<number>);
}

function ensureTimeRemaining(context: SearchContext): void {
  if (performance.now() >= context.deadlineMs) {
    throw new SearchTimeout();
  }
}

function scoreTerminal(info: GameInfo, plyFromRoot: number): number | null {
  if (info.result === "ongoing") {
    return null;
  }
  if (info.result === "draw") {
    return 0;
  }

  const winner = info.result.endsWith("white") ? "white" : "black";
  return winner === info.sideToMove
    ? SCORE_MATE - plyFromRoot
    : -SCORE_MATE + plyFromRoot;
}
