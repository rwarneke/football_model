/**
 * Standalone tournament simulator that replicates the exact sampling logic
 * from the website's predictor page. Run with:
 *   npx ts-node scripts/simulate-tournament.ts
 */

import fs from "fs";
import path from "path";

// ============================================================================
// Types
// ============================================================================

type WinProbabilityEntry = {
  p_home?: number;
  p_draw?: number;
  p_away?: number;
  p_home_pens?: number;
  p_away_pens?: number;
  score_matrix?: number[][];
};

type WinProbabilities = Record<
  string,
  Record<string, { home?: WinProbabilityEntry; neutral?: WinProbabilityEntry }>
>;

type GroupDefinition = { id: string; teams: string[] };
type GroupMatch = {
  id: number;
  group: string;
  homeTeam: string;
  awayTeam: string;
  country: string;
};
type KnockoutMatch = {
  id: number;
  stage: string;
  homeLabel: string;
  awayLabel: string;
  country: string;
};
type QualifierMatch = {
  id: string;
  path: string;
  round: string;
  homeTeam: string;
  awayTeam: string;
  homeSource: string;
  awaySource: string;
  winnerSlot: string;
  neutral: boolean;
};
type MatchScore = { home: number; away: number };
type GroupTableRow = {
  team: string;
  group: string;
  played: number;
  wins: number;
  draws: number;
  losses: number;
  gf: number;
  ga: number;
  gd: number;
  points: number;
  position: number;
};

// ============================================================================
// Constants
// ============================================================================

const REFERENCE_DIR = path.resolve(__dirname, "..", "..", "reference_data");
const MODEL_OUTPUT_DIR = path.resolve(__dirname, "..", "..", "model_output");

const HOST_TEAMS = new Set(["USA", "Canada", "Mexico"]);
const HOST_TEAM_COUNTRIES: Record<string, string> = {
  USA: "United States",
  "United States": "United States",
  Canada: "Canada",
  Mexico: "Mexico",
};

// ============================================================================
// Data Loading
// ============================================================================

function readCsv(filePath: string): Record<string, string>[] {
  const contents = fs.readFileSync(filePath, "utf8").trim();
  const lines = contents ? contents.split(/\r?\n/) : [];
  if (lines.length === 0) return [];
  const headers = lines[0]?.split(",") ?? [];
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""])
    ) as Record<string, string>;
  });
}

function loadNameMap(): Map<string, string> {
  const map = new Map<string, string>();
  const files = [
    path.join(REFERENCE_DIR, "fifa_member_to_canonical_name_map.csv"),
    path.join(REFERENCE_DIR, "kaggle_team_to_canonical_name_map.csv"),
  ];
  for (const filePath of files) {
    if (!fs.existsSync(filePath)) continue;
    for (const row of readCsv(filePath)) {
      const original = row.original_name?.trim();
      const replacement = row.replacement_name?.trim();
      if (original && replacement) map.set(original, replacement);
    }
  }
  return map;
}

function normalizeName(raw: string, nameMap: Map<string, string>): string {
  const trimmed = raw?.trim() ?? "";
  if (!trimmed || trimmed.toLowerCase() === "nan") return "";
  if (/\bwinner$/i.test(trimmed)) return trimmed;
  return nameMap.get(trimmed) ?? trimmed;
}

function loadData() {
  const nameMap = loadNameMap();

  // Groups
  const groupRows = readCsv(path.join(REFERENCE_DIR, "world_cup_2026_groups.csv"));
  const groupsMap = new Map<string, string[]>();
  for (const row of groupRows) {
    const group = row.group?.trim();
    if (!group) continue;
    const team = normalizeName(row.team ?? "", nameMap);
    if (!groupsMap.has(group)) groupsMap.set(group, []);
    if (team) groupsMap.get(group)?.push(team);
  }
  const groups: GroupDefinition[] = Array.from(groupsMap.keys())
    .sort()
    .map((g) => ({ id: g, teams: groupsMap.get(g) ?? [] }));

  // Group matches
  const groupMatches: GroupMatch[] = readCsv(
    path.join(REFERENCE_DIR, "world_cup_2026_group_matches.csv")
  ).map((row) => ({
    id: Number(row.match_id),
    group: row.group?.trim() ?? "",
    homeTeam: normalizeName(row.home_team ?? "", nameMap),
    awayTeam: normalizeName(row.away_team ?? "", nameMap),
    country: row.country?.trim() ?? "",
  }));

  // Knockout matches
  const knockoutMatches: KnockoutMatch[] = readCsv(
    path.join(REFERENCE_DIR, "world_cup_2026_knockout_matches.csv")
  ).map((row) => ({
    id: Number(row.match_id),
    stage: row.stage?.trim() ?? "",
    homeLabel: row.home?.trim() ?? "",
    awayLabel: row.away?.trim() ?? "",
    country: row.country?.trim() ?? "",
  }));

  // Round of 32 combinations
  const combosRows = readCsv(
    path.join(REFERENCE_DIR, "world_cup_2026_round_of_32_combinations.csv")
  );
  const roundOf32Combos: Record<string, Record<string, string>> = {};
  for (const row of combosRows) {
    const combo = row.combo?.trim();
    if (!combo) continue;
    roundOf32Combos[combo] = {
      "1A": row["1A"]?.trim() ?? "",
      "1B": row["1B"]?.trim() ?? "",
      "1D": row["1D"]?.trim() ?? "",
      "1E": row["1E"]?.trim() ?? "",
      "1G": row["1G"]?.trim() ?? "",
      "1I": row["1I"]?.trim() ?? "",
      "1K": row["1K"]?.trim() ?? "",
      "1L": row["1L"]?.trim() ?? "",
    };
  }

  // Qualifiers
  const qualifierRows = readCsv(
    path.join(REFERENCE_DIR, "world_cup_2026_remaining_qualifiers.csv")
  );
  const qualifiers: QualifierMatch[] = qualifierRows.map((row) => ({
    id: `${row.path?.trim() ?? "path"}-${row.round?.trim() ?? "round"}`,
    path: row.path?.trim() ?? "",
    round: row.round?.trim() ?? "",
    homeTeam: normalizeName(row.home_team ?? "", nameMap),
    awayTeam: normalizeName(row.away_team ?? "", nameMap),
    homeSource: row.home_source?.trim() ?? "",
    awaySource: row.away_source?.trim() ?? "",
    winnerSlot: row.winner_slot?.trim() ?? "",
    neutral: String(row.neutral ?? "").trim().toLowerCase() === "true",
  }));

  // Win probabilities
  const probPath = path.join(MODEL_OUTPUT_DIR, "win_probabilities.json");
  const winProbabilities: WinProbabilities = fs.existsSync(probPath)
    ? JSON.parse(fs.readFileSync(probPath, "utf8"))
    : {};

  return { groups, groupMatches, knockoutMatches, roundOf32Combos, qualifiers, winProbabilities };
}

// ============================================================================
// Probability Resolution (exact copy from website)
// ============================================================================

function normalizeCountry(country: string | undefined): string {
  return (country ?? "").trim().toLowerCase();
}

function resolveMatchNeutrality({
  homeTeam,
  awayTeam,
  country,
  neutralOverride,
}: {
  homeTeam: string;
  awayTeam: string;
  country?: string | null;
  neutralOverride?: boolean | null;
}): { neutral: boolean; advantage: "home" | "away" | null } {
  if (neutralOverride === true) return { neutral: true, advantage: null };
  if (neutralOverride === false) return { neutral: false, advantage: "home" };

  let neutral = true;
  let advantage: "home" | "away" | null = null;
  if (country) {
    const matchCountry = normalizeCountry(country);
    const homeCountry = normalizeCountry(HOST_TEAM_COUNTRIES[homeTeam]);
    const awayCountry = normalizeCountry(HOST_TEAM_COUNTRIES[awayTeam]);
    const homeAdvantage = homeCountry && matchCountry && homeCountry === matchCountry;
    const awayAdvantage = awayCountry && matchCountry && awayCountry === matchCountry;
    if (homeAdvantage && awayAdvantage) {
      neutral = true;
    } else if (homeAdvantage || awayAdvantage) {
      neutral = false;
      advantage = homeAdvantage ? "home" : "away";
    }
  } else {
    const homeIsHost = HOST_TEAMS.has(homeTeam);
    const awayIsHost = HOST_TEAMS.has(awayTeam);
    if (homeIsHost !== awayIsHost) {
      neutral = false;
      advantage = homeIsHost ? "home" : "away";
    }
  }
  return { neutral, advantage };
}

function resolveProbabilityEntry({
  probabilities,
  homeTeam,
  awayTeam,
  country,
  neutralOverride,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  country?: string | null;
  neutralOverride?: boolean | null;
}) {
  const { neutral, advantage } = resolveMatchNeutrality({
    homeTeam,
    awayTeam,
    country,
    neutralOverride,
  });
  if (neutral) {
    const entry = probabilities[homeTeam]?.[awayTeam]?.neutral;
    return entry ? { entry, flipped: false } : null;
  }
  if (advantage === "home") {
    const entry = probabilities[homeTeam]?.[awayTeam]?.home;
    return entry ? { entry, flipped: false } : null;
  }
  if (advantage === "away") {
    const entry = probabilities[awayTeam]?.[homeTeam]?.home;
    return entry ? { entry, flipped: true } : null;
  }
  return null;
}

function transposeScoreMatrix(matrix: number[][]): number[][] {
  const rows = matrix.length;
  const cols = matrix.reduce((max, row) => Math.max(max, row.length), 0);
  const transposed = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      const value = matrix[i][j];
      transposed[j][i] = Number.isFinite(value) ? value : 0;
    }
  }
  return transposed;
}

function resolveMatchScoreMatrix({
  probabilities,
  homeTeam,
  awayTeam,
  country,
  neutralOverride,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  country?: string | null;
  neutralOverride?: boolean | null;
}): number[][] | null {
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam,
    awayTeam,
    country,
    neutralOverride,
  });
  if (!resolved?.entry?.score_matrix) return null;
  return resolved.flipped
    ? transposeScoreMatrix(resolved.entry.score_matrix)
    : resolved.entry.score_matrix;
}

function resolveMatchProbabilities({
  probabilities,
  homeTeam,
  awayTeam,
  allowDraw,
  country,
  neutralOverride,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  allowDraw: boolean;
  country?: string | null;
  neutralOverride?: boolean | null;
}): { home: number | null; draw: number | null; away: number | null } | null {
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam,
    awayTeam,
    country,
    neutralOverride,
  });
  if (!resolved) return null;
  const entry = resolved.entry;
  let values: { home: number | null; draw: number | null; away: number | null };
  if (allowDraw) {
    values = {
      home: entry.p_home ?? null,
      draw: entry.p_draw ?? null,
      away: entry.p_away ?? null,
    };
  } else {
    values = {
      home: entry.p_home_pens ?? null,
      draw: null,
      away: entry.p_away_pens ?? null,
    };
  }
  if (!resolved.flipped) return values;
  return { home: values.away, draw: values.draw, away: values.home };
}

// ============================================================================
// Sampling Functions (exact copy from website)
// ============================================================================

function sampleScoreMatrix(scoreMatrix: number[][]): MatchScore | null {
  let total = 0;
  for (const row of scoreMatrix) {
    for (const value of row) {
      if (Number.isFinite(value) && value > 0) total += value;
    }
  }
  if (total <= 0) return null;
  const target = Math.random() * total;
  let cumulative = 0;
  for (let i = 0; i < scoreMatrix.length; i++) {
    const row = scoreMatrix[i];
    for (let j = 0; j < row.length; j++) {
      const value = row[j];
      if (!Number.isFinite(value) || value <= 0) continue;
      cumulative += value;
      if (cumulative >= target) return { home: i, away: j };
    }
  }
  return { home: 0, away: 0 };
}

function sampleWinner(
  values: { home: number | null; away: number | null } | null
): "home" | "away" | null {
  if (!values || values.home === null || values.away === null) return null;
  const total = values.home + values.away;
  if (!Number.isFinite(total) || total <= 0) return null;
  const roll = Math.random() * total;
  return roll < values.home ? "home" : "away";
}

// ============================================================================
// Group Stage Logic
// ============================================================================

function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function headToHeadTable(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>
) {
  const table: Record<string, { points: number; gf: number; ga: number; gd: number }> = {};
  for (const team of teams) {
    table[team] = { points: 0, gf: 0, ga: 0, gd: 0 };
  }
  for (const match of matches) {
    if (!teams.includes(match.homeTeam) || !teams.includes(match.awayTeam)) continue;
    const home = table[match.homeTeam];
    const away = table[match.awayTeam];
    home.gf += match.homeScore;
    home.ga += match.awayScore;
    away.gf += match.awayScore;
    away.ga += match.homeScore;
    if (match.homeScore > match.awayScore) {
      home.points += 3;
    } else if (match.homeScore < match.awayScore) {
      away.points += 3;
    } else {
      home.points += 1;
      away.points += 1;
    }
  }
  for (const team of teams) {
    table[team].gd = table[team].gf - table[team].ga;
  }
  return table;
}

function rankOverall(teams: string[], table: Record<string, GroupTableRow>): string[] {
  const sorted = [...teams].sort((a, b) => {
    const rowA = table[a];
    const rowB = table[b];
    if (rowB.points !== rowA.points) return rowB.points - rowA.points;
    if (rowB.gd !== rowA.gd) return rowB.gd - rowA.gd;
    if (rowB.gf !== rowA.gf) return rowB.gf - rowA.gf;
    return 0;
  });

  const ordered: string[] = [];
  let i = 0;
  while (i < sorted.length) {
    const current = sorted[i];
    const tied = [current];
    i++;
    while (i < sorted.length) {
      const next = sorted[i];
      const rowA = table[current];
      const rowB = table[next];
      if (rowA.points === rowB.points && rowA.gd === rowB.gd && rowA.gf === rowB.gf) {
        tied.push(next);
        i++;
      } else break;
    }
    if (tied.length > 1) shuffleInPlace(tied);
    ordered.push(...tied);
  }
  return ordered;
}

function rankHeadToHead(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>,
  table: Record<string, GroupTableRow>
): string[] {
  if (teams.length <= 1) return teams;
  const h2h = headToHeadTable(teams, matches);
  const metrics = teams.map((team) => h2h[team]);
  const allEqual =
    metrics.every((m) => m.points === metrics[0].points) &&
    metrics.every((m) => m.gd === metrics[0].gd) &&
    metrics.every((m) => m.gf === metrics[0].gf);
  if (allEqual) return rankOverall(teams, table);

  const sorted = [...teams].sort((a, b) => {
    const rowA = h2h[a];
    const rowB = h2h[b];
    if (rowB.points !== rowA.points) return rowB.points - rowA.points;
    if (rowB.gd !== rowA.gd) return rowB.gd - rowA.gd;
    if (rowB.gf !== rowA.gf) return rowB.gf - rowA.gf;
    return 0;
  });

  const ordered: string[] = [];
  let i = 0;
  while (i < sorted.length) {
    const current = sorted[i];
    const tied = [current];
    i++;
    while (i < sorted.length) {
      const next = sorted[i];
      const rowA = h2h[current];
      const rowB = h2h[next];
      if (rowA.points === rowB.points && rowA.gd === rowB.gd && rowA.gf === rowB.gf) {
        tied.push(next);
        i++;
      } else break;
    }
    if (tied.length === 1) {
      ordered.push(tied[0]);
    } else {
      ordered.push(...rankHeadToHead(tied, matches, table));
    }
  }
  return ordered;
}

function rankGroup(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>,
  table: Record<string, GroupTableRow>
): string[] {
  const base = [...teams].sort((a, b) => {
    const rowA = table[a];
    const rowB = table[b];
    if (rowB.points !== rowA.points) return rowB.points - rowA.points;
    return 0;
  });

  const ranked: string[] = [];
  let i = 0;
  while (i < base.length) {
    const current = base[i];
    const tied = [current];
    i++;
    while (i < base.length) {
      const next = base[i];
      if (table[current].points === table[next].points) {
        tied.push(next);
        i++;
      } else break;
    }
    if (tied.length === 1) {
      ranked.push(tied[0]);
    } else {
      ranked.push(...rankHeadToHead(tied, matches, table));
    }
  }
  return ranked;
}

function buildGroupTable(
  group: GroupDefinition,
  matches: GroupMatch[],
  scores: Record<string, MatchScore>
): { table: Record<string, GroupTableRow>; ranking: string[] } {
  const table: Record<string, GroupTableRow> = {};
  for (const team of group.teams) {
    table[team] = {
      team,
      group: group.id,
      played: 0,
      wins: 0,
      draws: 0,
      losses: 0,
      gf: 0,
      ga: 0,
      gd: 0,
      points: 0,
      position: 0,
    };
  }

  const playedMatches: Array<{
    homeTeam: string;
    awayTeam: string;
    homeScore: number;
    awayScore: number;
  }> = [];

  for (const match of matches) {
    const score = scores[String(match.id)];
    if (!score) continue;
    const home = table[match.homeTeam];
    const away = table[match.awayTeam];
    if (!home || !away) continue;
    const homeScore = score.home;
    const awayScore = score.away;
    home.gf += homeScore;
    home.ga += awayScore;
    away.gf += awayScore;
    away.ga += homeScore;
    home.played += 1;
    away.played += 1;
    if (homeScore > awayScore) {
      home.wins += 1;
      away.losses += 1;
      home.points += 3;
    } else if (awayScore > homeScore) {
      away.wins += 1;
      home.losses += 1;
      away.points += 3;
    } else {
      home.draws += 1;
      away.draws += 1;
      home.points += 1;
      away.points += 1;
    }
    playedMatches.push({ homeTeam: match.homeTeam, awayTeam: match.awayTeam, homeScore, awayScore });
  }

  for (const team of Object.keys(table)) {
    table[team].gd = table[team].gf - table[team].ga;
  }

  const ranking = rankGroup(group.teams, playedMatches, table);
  ranking.forEach((team, index) => {
    if (table[team]) table[team].position = index + 1;
  });

  return { table, ranking };
}

function bestThirdPlace(
  groupTables: Array<{ ranking: string[]; table: Record<string, GroupTableRow> }>
): { team: string; group: string }[] {
  const entries: Array<{ team: string; group: string; points: number; gd: number; gf: number }> = [];
  for (const { ranking, table } of groupTables) {
    if (ranking.length < 3) continue;
    const team = ranking[2];
    const row = table[team];
    entries.push({ team, group: row.group, points: row.points, gd: row.gd, gf: row.gf });
  }

  entries.sort((a, b) => {
    if (b.points !== a.points) return b.points - a.points;
    if (b.gd !== a.gd) return b.gd - a.gd;
    if (b.gf !== a.gf) return b.gf - a.gf;
    return 0;
  });

  const ordered: typeof entries = [];
  let i = 0;
  while (i < entries.length) {
    const current = entries[i];
    const tied = [current];
    i++;
    while (i < entries.length) {
      const next = entries[i];
      if (current.points === next.points && current.gd === next.gd && current.gf === next.gf) {
        tied.push(next);
        i++;
      } else break;
    }
    if (tied.length > 1) shuffleInPlace(tied);
    ordered.push(...tied);
  }

  return ordered.slice(0, 8).map((e) => ({ team: e.team, group: e.group }));
}

// ============================================================================
// Tournament Simulation
// ============================================================================

function simulateTournament(data: ReturnType<typeof loadData>): string {
  const { groups, groupMatches, knockoutMatches, roundOf32Combos, qualifiers, winProbabilities } =
    data;

  // 1. Simulate remaining qualifiers
  const slotWinners: Record<string, string> = {};
  const qualifierWinners: Record<string, "home" | "away"> = {};

  // Sort qualifiers by round (semis before finals)
  const sortedQualifiers = [...qualifiers].sort((a, b) => {
    if (a.round === "semi" && b.round === "final") return -1;
    if (a.round === "final" && b.round === "semi") return 1;
    return 0;
  });

  for (const match of sortedQualifiers) {
    let homeTeam = match.homeTeam;
    let awayTeam = match.awayTeam;

    // Resolve from sources if needed
    if (match.homeSource) {
      const sourceId = `${match.path}-${match.homeSource}`;
      const sourceWinner = qualifierWinners[sourceId];
      const sourceMatch = qualifiers.find((m) => m.id === sourceId);
      if (sourceMatch && sourceWinner) {
        homeTeam = sourceWinner === "home" ? sourceMatch.homeTeam : sourceMatch.awayTeam;
        // Resolve nested sources
        if (!homeTeam && sourceMatch.homeSource) {
          const nestedSourceId = `${sourceMatch.path}-${sourceMatch.homeSource}`;
          const nestedSourceWinner = qualifierWinners[nestedSourceId];
          const nestedSourceMatch = qualifiers.find((m) => m.id === nestedSourceId);
          if (nestedSourceMatch && nestedSourceWinner) {
            homeTeam =
              nestedSourceWinner === "home"
                ? nestedSourceMatch.homeTeam
                : nestedSourceMatch.awayTeam;
          }
        }
      }
    }
    if (match.awaySource) {
      const sourceId = `${match.path}-${match.awaySource}`;
      const sourceWinner = qualifierWinners[sourceId];
      const sourceMatch = qualifiers.find((m) => m.id === sourceId);
      if (sourceMatch && sourceWinner) {
        awayTeam = sourceWinner === "home" ? sourceMatch.homeTeam : sourceMatch.awayTeam;
        if (!awayTeam && sourceMatch.awaySource) {
          const nestedSourceId = `${sourceMatch.path}-${sourceMatch.awaySource}`;
          const nestedSourceWinner = qualifierWinners[nestedSourceId];
          const nestedSourceMatch = qualifiers.find((m) => m.id === nestedSourceId);
          if (nestedSourceMatch && nestedSourceWinner) {
            awayTeam =
              nestedSourceWinner === "home"
                ? nestedSourceMatch.homeTeam
                : nestedSourceMatch.awayTeam;
          }
        }
      }
    }

    if (!homeTeam || !awayTeam) continue;

    const probs = resolveMatchProbabilities({
      probabilities: winProbabilities,
      homeTeam,
      awayTeam,
      allowDraw: false,
      neutralOverride: match.neutral ? true : undefined,
    });

    const winner = sampleWinner(probs);
    if (winner) {
      qualifierWinners[match.id] = winner;
      if (match.winnerSlot) {
        slotWinners[match.winnerSlot] = winner === "home" ? homeTeam : awayTeam;
      }
    }
  }

  // 2. Resolve group teams with qualifier winners
  const resolvedGroups = groups.map((g) => ({
    ...g,
    teams: g.teams.map((t) => slotWinners[t] ?? t),
  }));

  // 3. Simulate group stage
  const groupScores: Record<string, MatchScore> = {};
  for (const match of groupMatches) {
    const homeTeam = slotWinners[match.homeTeam] ?? match.homeTeam;
    const awayTeam = slotWinners[match.awayTeam] ?? match.awayTeam;

    const matrix = resolveMatchScoreMatrix({
      probabilities: winProbabilities,
      homeTeam,
      awayTeam,
      country: match.country,
    });

    if (matrix) {
      const score = sampleScoreMatrix(matrix);
      if (score) {
        groupScores[String(match.id)] = score;
      }
    }
  }

  // 4. Build group tables and rankings
  const groupResults: Array<{ ranking: string[]; table: Record<string, GroupTableRow> }> = [];
  const groupRankings: Record<string, string[]> = {};
  for (const group of resolvedGroups) {
    const matches = groupMatches
      .filter((m) => m.group === group.id)
      .map((m) => ({
        ...m,
        homeTeam: slotWinners[m.homeTeam] ?? m.homeTeam,
        awayTeam: slotWinners[m.awayTeam] ?? m.awayTeam,
      }));
    const result = buildGroupTable(
      { ...group, teams: group.teams },
      matches,
      groupScores
    );
    groupResults.push(result);
    groupRankings[group.id] = result.ranking;
  }

  // 5. Determine best third-place teams
  const bestThirds = bestThirdPlace(groupResults);
  const qualifiedThirdGroups = new Set(bestThirds.map((t) => t.group));
  const thirdPlaceByGroup: Record<string, string> = {};
  for (const { team, group } of bestThirds) {
    thirdPlaceByGroup[group] = team;
  }

  // 6. Determine third-place assignments using combo table
  const comboKey = Array.from(qualifiedThirdGroups).sort().join("");
  const thirdPlaceAssignments = roundOf32Combos[comboKey] ?? null;

  // 7. Simulate knockout stage
  const knockoutWinners: Map<number, string> = new Map();
  const knockoutLosers: Map<number, string> = new Map();

  // Sort knockout matches by stage order
  const stageOrder = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Third place", "Final"];
  const sortedKnockout = [...knockoutMatches].sort(
    (a, b) => stageOrder.indexOf(a.stage) - stageOrder.indexOf(b.stage)
  );

  for (const match of sortedKnockout) {
    // Resolve home team
    let homeTeam = resolveKnockoutTeam(
      match.homeLabel,
      match.awayLabel,
      groupRankings,
      thirdPlaceByGroup,
      thirdPlaceAssignments,
      knockoutWinners,
      knockoutLosers
    );

    // Resolve away team
    let awayTeam = resolveKnockoutTeam(
      match.awayLabel,
      match.homeLabel,
      groupRankings,
      thirdPlaceByGroup,
      thirdPlaceAssignments,
      knockoutWinners,
      knockoutLosers
    );

    if (!homeTeam || !awayTeam) continue;

    const probs = resolveMatchProbabilities({
      probabilities: winProbabilities,
      homeTeam,
      awayTeam,
      allowDraw: false,
      country: match.country,
    });

    const winner = sampleWinner(probs);
    if (winner) {
      const winningTeam = winner === "home" ? homeTeam : awayTeam;
      const losingTeam = winner === "home" ? awayTeam : homeTeam;
      knockoutWinners.set(match.id, winningTeam);
      knockoutLosers.set(match.id, losingTeam);
    }
  }

  // 8. Return champion (winner of final match)
  const finalMatch = knockoutMatches.find((m) => m.stage === "Final");
  if (finalMatch) {
    return knockoutWinners.get(finalMatch.id) ?? "Unknown";
  }
  return "Unknown";
}

function resolveKnockoutTeam(
  label: string,
  opponentLabel: string,
  groupRankings: Record<string, string[]>,
  thirdPlaceByGroup: Record<string, string>,
  thirdPlaceAssignments: Record<string, string> | null,
  knockoutWinners: Map<number, string>,
  knockoutLosers: Map<number, string>
): string | null {
  if (label.startsWith("Winner Group ")) {
    const group = label.replace("Winner Group ", "").trim();
    return groupRankings[group]?.[0] ?? null;
  }
  if (label.startsWith("Runner-up Group ")) {
    const group = label.replace("Runner-up Group ", "").trim();
    return groupRankings[group]?.[1] ?? null;
  }
  if (label.startsWith("3rd Group ") && opponentLabel.startsWith("Winner Group ")) {
    const winnerGroup = opponentLabel.replace("Winner Group ", "").trim();
    const key = `1${winnerGroup}`;
    const assignedGroup = thirdPlaceAssignments?.[key];
    if (assignedGroup) {
      return thirdPlaceByGroup[assignedGroup] ?? null;
    }
  }
  if (label.startsWith("Winner Match ")) {
    const matchId = Number(label.replace("Winner Match ", "").trim());
    return knockoutWinners.get(matchId) ?? null;
  }
  if (label.startsWith("Loser Match ")) {
    const matchId = Number(label.replace("Loser Match ", "").trim());
    return knockoutLosers.get(matchId) ?? null;
  }
  return null;
}

// ============================================================================
// Main
// ============================================================================

function main() {
  const N = 10000;
  console.log(`Loading data...`);
  const data = loadData();
  console.log(`Running ${N} simulations...`);

  const championCounts: Record<string, number> = {};
  const startTime = Date.now();

  for (let i = 0; i < N; i++) {
    const champion = simulateTournament(data);
    championCounts[champion] = (championCounts[champion] ?? 0) + 1;

    if ((i + 1) % 1000 === 0) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`  ${i + 1} / ${N} (${elapsed}s)`);
    }
  }

  // Sort by count descending
  const sorted = Object.entries(championCounts).sort((a, b) => b[1] - a[1]);

  // Write to CSV
  const outputPath = path.join(MODEL_OUTPUT_DIR, "web_simulation_results.csv");
  const csvLines = ["team,wins,probability"];
  for (const [team, count] of sorted) {
    const prob = (count / N).toFixed(4);
    csvLines.push(`${team},${count},${prob}`);
  }
  fs.writeFileSync(outputPath, csvLines.join("\n"));

  console.log(`\nResults written to: ${outputPath}`);
  console.log(`\nTop 10 Champions:`);
  for (const [team, count] of sorted.slice(0, 10)) {
    const pct = ((count / N) * 100).toFixed(2);
    console.log(`  ${team}: ${count} (${pct}%)`);
  }
}

main();
