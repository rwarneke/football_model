import type {
  CompactWinProbabilities,
  WinProbabilities,
  WinProbabilityEntry,
} from "@/lib/world-cup-predictor-types";

const ENTRY_INDEX = {
  HOME_ID: 0,
  AWAY_ID: 1,
  NEUTRAL: 2,
  NU: 3,
  LAM_HOME: 4,
  LAM_AWAY: 5,
  P_HOME: 6,
  P_DRAW: 7,
  P_AWAY: 8,
  P_HOME_PENS: 9,
  P_AWAY_PENS: 10,
} as const;

type CompactIndex = {
  teamIndex: Map<string, number>;
  entryIndex: Map<string, number[]>;
};

const compactIndexCache = new WeakMap<CompactWinProbabilities, CompactIndex>();

function buildEntryKey(homeId: number, awayId: number, neutralFlag: number) {
  return `${homeId}|${awayId}|${neutralFlag}`;
}

export function isCompactWinProbabilities(
  probabilities: WinProbabilities | null | undefined
): probabilities is CompactWinProbabilities {
  if (!probabilities || typeof probabilities !== "object") {
    return false;
  }
  const candidate = probabilities as CompactWinProbabilities;
  return (
    typeof candidate.version === "number" &&
    Array.isArray(candidate.teams) &&
    Array.isArray(candidate.entries)
  );
}

export function getCompactIndex(probabilities: CompactWinProbabilities): CompactIndex {
  const cached = compactIndexCache.get(probabilities);
  if (cached) {
    return cached;
  }
  const teamIndex = new Map<string, number>();
  probabilities.teams.forEach((team, idx) => {
    if (team) {
      teamIndex.set(team, idx);
    }
  });
  const entryIndex = new Map<string, number[]>();
  for (const entry of probabilities.entries) {
    if (!Array.isArray(entry)) {
      continue;
    }
    const homeId = entry[ENTRY_INDEX.HOME_ID];
    const awayId = entry[ENTRY_INDEX.AWAY_ID];
    const neutralFlag = entry[ENTRY_INDEX.NEUTRAL];
    if (
      Number.isFinite(homeId) &&
      Number.isFinite(awayId) &&
      Number.isFinite(neutralFlag)
    ) {
      entryIndex.set(buildEntryKey(homeId, awayId, neutralFlag), entry);
    }
  }
  const index = { teamIndex, entryIndex };
  compactIndexCache.set(probabilities, index);
  return index;
}

export function resolveCompactEntry(
  probabilities: CompactWinProbabilities,
  homeTeam: string,
  awayTeam: string,
  neutral: boolean
): number[] | null {
  const { teamIndex, entryIndex } = getCompactIndex(probabilities);
  const homeId = teamIndex.get(homeTeam);
  const awayId = teamIndex.get(awayTeam);
  if (homeId === undefined || awayId === undefined) {
    return null;
  }
  const key = buildEntryKey(homeId, awayId, neutral ? 1 : 0);
  return entryIndex.get(key) ?? null;
}

export function parseCompactEntry(entry: number[]): WinProbabilityEntry {
  return {
    nu: entry[ENTRY_INDEX.NU],
    lam_home: entry[ENTRY_INDEX.LAM_HOME],
    lam_away: entry[ENTRY_INDEX.LAM_AWAY],
    p_home: entry[ENTRY_INDEX.P_HOME],
    p_draw: entry[ENTRY_INDEX.P_DRAW],
    p_away: entry[ENTRY_INDEX.P_AWAY],
    p_home_pens: entry[ENTRY_INDEX.P_HOME_PENS],
    p_away_pens: entry[ENTRY_INDEX.P_AWAY_PENS],
  };
}
