type MoveInfo = {
  index: number;
  ptn: string;
};

type SearchInfo = {
  bestMove: string;
  score: number;
  depth: number;
  nodes: number;
  pv: string[];
  ttHits: number;
  engineMode: "book";
  modelName?: string;
};

type BookMove = {
  moveKey: string;
  support: number;
  value: number;
};

type BookEntry = {
  moves: BookMove[];
};

type OpeningBook = {
  version: number;
  maxPly: number;
  entries: Record<string, BookEntry>;
};

type OpeningBookContext = {
  key: string;
  transform: number;
  ply: number;
  size: number;
  komi: number;
  halfKomi: boolean;
};

export type WasmBookGame = {
  legalMoves(): MoveInfo[];
  openingBookContext(): OpeningBookContext;
  resolveBookMove(moveKey: string, transform: number): number;
};

let openingBookPromise: Promise<OpeningBook | null> | null = null;

export async function lookupOpeningBookMove(
  game: WasmBookGame,
): Promise<SearchInfo | null> {
  const book = await loadOpeningBook();
  if (!book) {
    return null;
  }

  const context = game.openingBookContext();
  if (context.ply > book.maxPly) {
    return null;
  }

  const entry = book.entries[qualifyBookKey(context)] ?? book.entries[context.key];
  if (!entry || entry.moves.length === 0) {
    return null;
  }

  const legalMoves = game.legalMoves();
  for (const candidate of entry.moves) {
    const moveIndex = game.resolveBookMove(candidate.moveKey, context.transform);
    if (moveIndex < 0 || moveIndex >= legalMoves.length) {
      continue;
    }

    const move = legalMoves[moveIndex];
    return {
      bestMove: move.ptn,
      score: Math.round(candidate.value * 500),
      depth: 0,
      nodes: 0,
      pv: [move.ptn],
      ttHits: 0,
      engineMode: "book",
      modelName: `${candidate.support} games`,
    };
  }

  return null;
}

function qualifyBookKey(context: OpeningBookContext): string {
  return `${context.key}|k=${context.komi}${context.halfKomi ? ".5" : ""}`;
}

async function loadOpeningBook(): Promise<OpeningBook | null> {
  if (openingBookPromise) {
    return openingBookPromise;
  }

  openingBookPromise = fetch("/models/opening_book.json")
    .then(async (response) => {
      if (!response.ok) {
        return null;
      }
      return response.json() as Promise<OpeningBook>;
    })
    .catch((error: unknown) => {
      console.warn("Opening book unavailable, falling back to search", error);
      return null;
    });

  return openingBookPromise;
}
