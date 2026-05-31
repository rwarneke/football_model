import type {
  CompactWinProbabilities,
  WinProbabilities,
  WinProbabilityEntry,
} from "@/lib/world-cup-predictor-types";

const ENTRY_INDEX = {
  HOME_ID: 0,
  AWAY_ID: 1,
  NEUTRAL: 2,
  FRIENDLY: 3,
  NU: 4,
  LAM_HOME: 5,
  LAM_AWAY: 6,
  P_HOME: 7,
  P_DRAW: 8,
  P_AWAY: 9,
  P_HOME_PENS: 10,
  P_AWAY_PENS: 11,
} as const;

type CompactIndex = {
  teamIndex: Map<string, number>;
  entryIndex: Map<string, number[]>;
};

const compactIndexCache = new WeakMap<CompactWinProbabilities, CompactIndex>();

function buildEntryKey(
  homeId: number,
  awayId: number,
  neutralFlag: number,
  friendlyFlag: number
) {
  return `${homeId}|${awayId}|${neutralFlag}|${friendlyFlag}`;
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
    const friendlyFlag =
      entry.length > ENTRY_INDEX.P_AWAY_PENS ? entry[ENTRY_INDEX.FRIENDLY] : 0;
    if (
      Number.isFinite(homeId) &&
      Number.isFinite(awayId) &&
      Number.isFinite(neutralFlag) &&
      Number.isFinite(friendlyFlag)
    ) {
      entryIndex.set(buildEntryKey(homeId, awayId, neutralFlag, friendlyFlag), entry);
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
  neutral: boolean,
  isFriendly = false
): number[] | null {
  const { teamIndex, entryIndex } = getCompactIndex(probabilities);
  const homeId = teamIndex.get(homeTeam);
  const awayId = teamIndex.get(awayTeam);
  if (homeId === undefined || awayId === undefined) {
    return null;
  }
  const key = buildEntryKey(homeId, awayId, neutral ? 1 : 0, isFriendly ? 1 : 0);
  return entryIndex.get(key) ?? null;
}

export function parseCompactEntry(entry: number[]): WinProbabilityEntry {
  const shift = entry.length > ENTRY_INDEX.P_AWAY_PENS ? 0 : -1;
  return {
    nu: entry[ENTRY_INDEX.NU + shift],
    lam_home: entry[ENTRY_INDEX.LAM_HOME + shift],
    lam_away: entry[ENTRY_INDEX.LAM_AWAY + shift],
    p_home: entry[ENTRY_INDEX.P_HOME + shift],
    p_draw: entry[ENTRY_INDEX.P_DRAW + shift],
    p_away: entry[ENTRY_INDEX.P_AWAY + shift],
    p_home_pens: entry[ENTRY_INDEX.P_HOME_PENS + shift],
    p_away_pens: entry[ENTRY_INDEX.P_AWAY_PENS + shift],
  };
}
