import * as ort from "onnxruntime-web";

type MoveInfo = {
  index: number;
  ptn: string;
};

export type SearchInfo = {
  bestMove: string;
  score: number;
  depth: number;
  nodes: number;
  pv: string[];
  ttHits: number;
  engineMode: "neural" | "heuristic";
  modelName?: string;
};

export type WasmGame = {
  encodeBoard(): Float32Array;
  sizeId(): number;
  legalMoves(): MoveInfo[];
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

let runtimePromise: Promise<NeuralRuntime | null> | null = null;

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
): Promise<SearchInfo | null> {
  const runtime = await initNeuralRuntime(wasmModule);
  if (!runtime) {
    return null;
  }

  const legalMoves = game.legalMoves();
  if (legalMoves.length === 0) {
    throw new Error("No legal moves available for neural inference");
  }

  const boardTensor = game.encodeBoard();
  const sizeId = game.sizeId();
  const feeds: Record<string, ort.Tensor> = {
    board_tensor: new ort.Tensor("float32", boardTensor, [1, 31, 8, 8]),
    size_id: new ort.Tensor("int64", BigInt64Array.of(BigInt(sizeId)), [1]),
  };

  const outputs = await runtime.session.run(feeds);
  const spatial = requireTensor(outputs, "spatial");
  const globalPool = requireTensor(outputs, "global_pool");
  const margin = requireTensor(outputs, "margin");

  const moveProbabilities = runtime.policy.scoreMoves(
    game,
    toFloat32Array(spatial.data),
    toFloat32Array(globalPool.data),
  );

  if (moveProbabilities.length !== legalMoves.length) {
    throw new Error(
      `Policy scorer returned ${moveProbabilities.length} probabilities for ${legalMoves.length} legal moves`,
    );
  }

  const bestIndex = argmax(moveProbabilities);
  const bestMove = legalMoves[bestIndex];
  const marginValue = toFloat32Array(margin.data)[0] ?? 0;

  return {
    bestMove: bestMove.ptn,
    score: Math.round(marginValue * 500),
    depth: 0,
    nodes: legalMoves.length,
    pv: [bestMove.ptn],
    ttHits: 0,
    engineMode: "neural",
    modelName: runtime.modelName,
  };
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

function argmax(values: ArrayLike<number>): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }

  return bestIndex;
}
