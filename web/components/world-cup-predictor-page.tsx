"use client";

import * as React from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";
import type {
  GroupDefinition,
  GroupMatch,
  KnockoutMatch,
  QualifierMatch,
  RoundOf32Combos,
  WinProbabilities,
  WorldCupPredictorData,
} from "@/lib/world-cup-predictor";

type MatchScore = { home: number | null; away: number | null };
type WinnerSelection = "home" | "away" | null;

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
  randomTiebreak?: boolean;
};

type ResolvedQualifierMatch = QualifierMatch & {
  homeResolved: string;
  awayResolved: string;
  winner?: string;
};

type ResolvedKnockoutMatch = KnockoutMatch & {
  homeResolved: string;
  awayResolved: string;
  winner?: string;
};

const SKIP_INITIALS = new Set(["and", "of", "the"]);
const HOST_TEAM_COUNTRIES: Record<string, string> = {
  USA: "United States",
  "United States": "United States",
  Canada: "Canada",
  Mexico: "Mexico",
};
const HOST_TEAMS = new Set(["USA", "Canada", "Mexico"]);
const TIEBREAK_TOOLTIP =
  "Table order has been chosen randomly but would be determined by Fair Play Points in reality.";

type MatchProbabilityValues = {
  home: number | null;
  draw: number | null;
  away: number | null;
};

type MatchProbabilityLabels = {
  homeWinProb?: string;
  awayWinProb?: string;
  drawProb?: string | null;
};

function teamInitials(team: string) {
  const letters = team
    .split(/\s+/)
    .filter((word) => word && !SKIP_INITIALS.has(word.toLowerCase()))
    .map((word) => word[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();
  return letters || team.slice(0, 2).toUpperCase();
}

function isPlaceholderLabel(name: string) {
  if (!name) {
    return true;
  }
  return (
    name.includes("Winner Match") ||
    name.includes("Loser Match") ||
    name.includes("Winner Group") ||
    name.includes("Runner-up Group") ||
    name.includes("3rd Group") ||
    /^UEFA Path /i.test(name) ||
    /^IC Path /i.test(name) ||
    /^([123](st|nd|rd)) Group /i.test(name) ||
    /^([123](st|nd|rd)) Gr\. /i.test(name) ||
    /^Winner semi/i.test(name) ||
    /^Winner IC Path/i.test(name) ||
    /^Winner UEFA Path/i.test(name) ||
    /^Winner (R32|R16|QF|SF|Final)$/i.test(name) ||
    /^Loser (R32|R16|QF|SF|Final)$/i.test(name) ||
    /winner$/i.test(name)
  );
}

function formatDisplayLabel(label: string) {
  if (!label) {
    return label;
  }
  return label
    .replace(/^Winner\s+UEFA Path\s+/i, "UEFA Path ")
    .replace(/^Winner\s+IC Path\s+/i, "IC Path ")
    .replace(/^UEFA Path\s+(.+)\s+Winner$/i, "UEFA Path $1")
    .replace(/^IC Path\s+(.+)\s+Winner$/i, "IC Path $1")
    .replace(/^Winner\s+Semi(?:final)?\b/i, "Winner SF");
}

function formatQualifierSource(source: string | undefined) {
  if (!source) {
    return source;
  }
  if (source === "semi1" || source === "semi2") {
    return "Semifinal";
  }
  return source;
}

function hashString(value: string) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function createRng(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), 1 | x);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace<T>(items: T[], rng: () => number) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function seedFromGroupState(
  group: GroupDefinition,
  matches: GroupMatch[],
  scores: Record<string, MatchScore>
) {
  const parts = [group.id];
  const orderedMatches = [...matches].sort((a, b) => a.id - b.id);
  for (const match of orderedMatches) {
    const score = scores[String(match.id)];
    const home = score?.home ?? "x";
    const away = score?.away ?? "x";
    parts.push(`${match.id}:${home}-${away}`);
  }
  return hashString(parts.join("|"));
}

function seedFromThirdPlace(entries: Array<{ team: string; group: string; points: number; gd: number; gf: number }>) {
  const parts = entries.map(
    (entry) => `${entry.group}:${entry.team}:${entry.points}:${entry.gd}:${entry.gf}`
  );
  return hashString(parts.join("|"));
}

function extractGroupId(label: string) {
  const match = label.match(/Group\s+([A-Z])/i);
  return match?.[1] ? match[1].toUpperCase() : null;
}

function parseScore(value: string) {
  if (!value) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  const clamped = Math.max(0, Math.min(31, parsed));
  return clamped;
}

function formatProbability(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return undefined;
  }
  const percent = value * 100;
  if (percent < 0.95) {
    return `${percent.toFixed(1)}%`;
  }
  if (percent >= 99.05) {
    const rounded = Number(percent.toFixed(1));
    const capped = Math.min(rounded, 99.9);
    return `${capped.toFixed(1)}%`;
  }
  return `${Math.round(percent)}%`;
}

function normalizeCountry(value: string | null | undefined) {
  return value ? value.trim().toLowerCase() : "";
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
}) {
  if (neutralOverride !== undefined && neutralOverride !== null) {
    const neutral = Boolean(neutralOverride);
    return { neutral, advantage: neutral ? null : ("home" as const) };
  }
  let neutral = true;
  let advantage: "home" | "away" | null = null;
  if (country) {
    const matchCountry = normalizeCountry(country);
    const homeCountry = normalizeCountry(HOST_TEAM_COUNTRIES[homeTeam]);
    const awayCountry = normalizeCountry(HOST_TEAM_COUNTRIES[awayTeam]);
    const homeAdvantage =
      homeCountry && matchCountry && homeCountry === matchCountry;
    const awayAdvantage =
      awayCountry && matchCountry && awayCountry === matchCountry;
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

function selectProbabilityValues(
  entry: {
    p_home?: number;
    p_draw?: number;
    p_away?: number;
    p_home_pens?: number;
    p_away_pens?: number;
  } | undefined,
  allowDraw: boolean
): MatchProbabilityValues | null {
  if (!entry) {
    return null;
  }
  if (allowDraw) {
    return {
      home: entry.p_home ?? null,
      draw: entry.p_draw ?? null,
      away: entry.p_away ?? null,
    };
  }
  return {
    home: entry.p_home_pens ?? null,
    draw: null,
    away: entry.p_away_pens ?? null,
  };
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
}): MatchProbabilityValues | null {
  if (
    !probabilities ||
    isPlaceholderLabel(homeTeam) ||
    isPlaceholderLabel(awayTeam)
  ) {
    return null;
  }
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam,
    awayTeam,
    country,
    neutralOverride,
  });
  if (!resolved) {
    return null;
  }
  const values = selectProbabilityValues(resolved.entry, allowDraw);
  if (!values) {
    return null;
  }
  if (!resolved.flipped) {
    return values;
  }
  return {
    home: values.away,
    draw: values.draw,
    away: values.home,
  };
}

function transposeScoreMatrix(matrix: number[][]) {
  const rows = matrix.length;
  const cols = matrix.reduce((max, row) => Math.max(max, row.length), 0);
  const transposed = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < matrix[i].length; j += 1) {
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
  if (
    !probabilities ||
    isPlaceholderLabel(homeTeam) ||
    isPlaceholderLabel(awayTeam)
  ) {
    return null;
  }
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam,
    awayTeam,
    country,
    neutralOverride,
  });
  if (!resolved?.entry?.score_matrix) {
    return null;
  }
  return resolved.flipped
    ? transposeScoreMatrix(resolved.entry.score_matrix)
    : resolved.entry.score_matrix;
}

function sampleScoreMatrix(scoreMatrix: number[][]) {
  let total = 0;
  for (const row of scoreMatrix) {
    for (const value of row) {
      if (Number.isFinite(value) && value > 0) {
        total += value;
      }
    }
  }
  if (total <= 0) {
    return null;
  }
  const target = Math.random() * total;
  let cumulative = 0;
  for (let i = 0; i < scoreMatrix.length; i += 1) {
    const row = scoreMatrix[i];
    for (let j = 0; j < row.length; j += 1) {
      const value = row[j];
      if (!Number.isFinite(value) || value <= 0) {
        continue;
      }
      cumulative += value;
      if (cumulative >= target) {
        return { home: i, away: j };
      }
    }
  }
  return { home: 0, away: 0 };
}

function sampleWinner(values: MatchProbabilityValues | null): WinnerSelection {
  if (!values || values.home === null || values.away === null) {
    return null;
  }
  const total = values.home + values.away;
  if (!Number.isFinite(total) || total <= 0) {
    return null;
  }
  const roll = Math.random() * total;
  return roll < values.home ? "home" : "away";
}

function clearDependentScores(
  scores: Record<string, MatchScore>,
  matchId: string,
  dependents: Map<string, Set<string>>
) {
  const next = { ...scores };
  const visited = new Set<string>();
  const stack = [matchId];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) {
      continue;
    }
    const deps = dependents.get(current);
    if (!deps) {
      continue;
    }
    for (const dep of deps) {
      if (visited.has(dep)) {
        continue;
      }
      visited.add(dep);
      if (next[dep]) {
        next[dep] = { home: null, away: null };
      }
      stack.push(dep);
    }
  }
  return next;
}

function resolveQualifierState(
  qualifiers: QualifierMatch[],
  qualifierWinners: Record<string, WinnerSelection>
) {
  const sorted = sortQualifiers(qualifiers);
  let winnersByPathRound = new Map<string, string>();
  let slotWinners = new Map<string, string>();
  let resolvedMatches: ResolvedQualifierMatch[] = [];
  let changed = true;
  let iterations = 0;

  while (changed && iterations < 4) {
    iterations += 1;
    changed = false;
    const nextWinners = new Map<string, string>();
    const nextSlots = new Map<string, string>();
    const nextResolved: ResolvedQualifierMatch[] = [];

    for (const match of sorted) {
      const homeResolved =
        match.homeTeam ||
        (match.homeSource
          ? winnersByPathRound.get(`${match.path}:${match.homeSource}`) ??
            `Winner ${formatQualifierSource(match.homeSource)}`
          : "");
      const awayResolved =
        match.awayTeam ||
        (match.awaySource
          ? winnersByPathRound.get(`${match.path}:${match.awaySource}`) ??
            `Winner ${formatQualifierSource(match.awaySource)}`
          : "");
      const winner = resolveWinner(
        match.id,
        homeResolved,
        awayResolved,
        {},
        false,
        qualifierWinners
      );
      if (winner) {
        const key = `${match.path}:${match.round}`;
        if (winnersByPathRound.get(key) !== winner) {
          changed = true;
        }
        nextWinners.set(key, winner);
        if (match.winnerSlot) {
          if (slotWinners.get(match.winnerSlot) !== winner) {
            changed = true;
          }
          nextSlots.set(match.winnerSlot, winner);
        }
      }
      nextResolved.push({
        ...match,
        homeResolved,
        awayResolved,
        winner,
      });
    }

    winnersByPathRound = nextWinners;
    slotWinners = nextSlots;
    resolvedMatches = nextResolved;
  }

  return { matches: resolvedMatches, slotWinners };
}

function clearDependentSelections(
  selections: Record<string, WinnerSelection>,
  matchId: string,
  dependents: Map<string, Set<string>>
) {
  const next = { ...selections };
  const visited = new Set<string>();
  const stack = [matchId];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) {
      continue;
    }
    const deps = dependents.get(current);
    if (!deps) {
      continue;
    }
    for (const dep of deps) {
      if (visited.has(dep)) {
        continue;
      }
      visited.add(dep);
      if (next[dep] !== null) {
        next[dep] = null;
      }
      stack.push(dep);
    }
  }
  return next;
}

function useMediaQuery(query: string) {
  const [matches, setMatches] = React.useState(false);

  React.useEffect(() => {
    const media = window.matchMedia(query);
    const handler = () => setMatches(media.matches);
    handler();
    media.addEventListener("change", handler);
    return () => media.removeEventListener("change", handler);
  }, [query]);

  return matches;
}

function resolveWinner(
  matchId: string | number,
  homeTeam: string,
  awayTeam: string,
  scores: Record<string, MatchScore>,
  allowDraw: boolean,
  winnerSelections?: Record<string, WinnerSelection>
) {
  const selection = winnerSelections?.[String(matchId)] ?? null;
  if (selection) {
    return selection === "home" ? homeTeam : awayTeam;
  }
  const score = scores[String(matchId)];
  if (!score || score.home === null || score.away === null) {
    return undefined;
  }
  if (score.home === score.away) {
    return allowDraw ? undefined : undefined;
  }
  return score.home > score.away ? homeTeam : awayTeam;
}

function rankOverall(
  teams: string[],
  table: Record<string, GroupTableRow>,
  rng: () => number,
  randomTiebreakTeams: Set<string>
) {
  const sorted = [...teams].sort((a, b) => {
    const rowA = table[a];
    const rowB = table[b];
    if (rowB.points !== rowA.points) {
      return rowB.points - rowA.points;
    }
    if (rowB.gd !== rowA.gd) {
      return rowB.gd - rowA.gd;
    }
    if (rowB.gf !== rowA.gf) {
      return rowB.gf - rowA.gf;
    }
    return 0;
  });

  const ordered: string[] = [];
  let i = 0;
  while (i < sorted.length) {
    const current = sorted[i];
    const tied = [current];
    i += 1;
    while (i < sorted.length) {
      const next = sorted[i];
      const rowA = table[current];
      const rowB = table[next];
      if (
        rowA.points === rowB.points &&
        rowA.gd === rowB.gd &&
        rowA.gf === rowB.gf
      ) {
        tied.push(next);
        i += 1;
      } else {
        break;
      }
    }
    if (tied.length > 1) {
      shuffleInPlace(tied, rng);
      tied.forEach((team) => randomTiebreakTeams.add(team));
    }
    ordered.push(...tied);
  }
  return ordered;
}

function headToHeadTable(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>
) {
  const table: Record<string, { points: number; gf: number; ga: number; gd: number }> =
    {};
  for (const team of teams) {
    table[team] = { points: 0, gf: 0, ga: 0, gd: 0 };
  }
  for (const match of matches) {
    if (!teams.includes(match.homeTeam) || !teams.includes(match.awayTeam)) {
      continue;
    }
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
    const row = table[team];
    row.gd = row.gf - row.ga;
  }
  return table;
}

function rankHeadToHead(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>,
  table: Record<string, GroupTableRow>,
  rng: () => number,
  randomTiebreakTeams: Set<string>
): string[] {
  if (teams.length <= 1) {
    return teams;
  }
  const h2h = headToHeadTable(teams, matches);
  const metrics = teams.map((team) => h2h[team]);
  const allEqual =
    metrics.every((m) => m.points === metrics[0].points) &&
    metrics.every((m) => m.gd === metrics[0].gd) &&
    metrics.every((m) => m.gf === metrics[0].gf);
  if (allEqual) {
    return rankOverall(teams, table, rng, randomTiebreakTeams);
  }
  const sorted = [...teams].sort((a, b) => {
    const rowA = h2h[a];
    const rowB = h2h[b];
    if (rowB.points !== rowA.points) {
      return rowB.points - rowA.points;
    }
    if (rowB.gd !== rowA.gd) {
      return rowB.gd - rowA.gd;
    }
    if (rowB.gf !== rowA.gf) {
      return rowB.gf - rowA.gf;
    }
    return 0;
  });
  const ordered: string[] = [];
  let i = 0;
  while (i < sorted.length) {
    const current = sorted[i];
    const tied = [current];
    i += 1;
    while (i < sorted.length) {
      const next = sorted[i];
      const rowA = h2h[current];
      const rowB = h2h[next];
      if (
        rowA.points === rowB.points &&
        rowA.gd === rowB.gd &&
        rowA.gf === rowB.gf
      ) {
        tied.push(next);
        i += 1;
      } else {
        break;
      }
    }
    if (tied.length === 1) {
      ordered.push(tied[0]);
    } else {
      ordered.push(...rankHeadToHead(tied, matches, table, rng, randomTiebreakTeams));
    }
  }
  return ordered;
}

function rankGroup(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>,
  table: Record<string, GroupTableRow>,
  rng: () => number,
  randomTiebreakTeams: Set<string>
) {
  const base = [...teams].sort((a, b) => {
    const rowA = table[a];
    const rowB = table[b];
    if (rowB.points !== rowA.points) {
      return rowB.points - rowA.points;
    }
    return 0;
  });

  const ranked: string[] = [];
  let i = 0;
  while (i < base.length) {
    const current = base[i];
    const tied = [current];
    i += 1;
    while (i < base.length) {
      const next = base[i];
      if (table[current].points === table[next].points) {
        tied.push(next);
        i += 1;
      } else {
        break;
      }
    }
    if (tied.length === 1) {
      ranked.push(tied[0]);
    } else {
      ranked.push(...rankHeadToHead(tied, matches, table, rng, randomTiebreakTeams));
    }
  }
  return ranked;
}

function buildGroupTable(
  group: GroupDefinition,
  matches: GroupMatch[],
  scores: Record<string, MatchScore>
) {
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
    if (!score || score.home === null || score.away === null) {
      continue;
    }
    const home = table[match.homeTeam];
    const away = table[match.awayTeam];
    if (!home || !away) {
      continue;
    }
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
    playedMatches.push({
      homeTeam: match.homeTeam,
      awayTeam: match.awayTeam,
      homeScore,
      awayScore,
    });
  }

  for (const team of Object.keys(table)) {
    table[team].gd = table[team].gf - table[team].ga;
  }

  const randomTiebreakTeams = new Set<string>();
  const rng = createRng(seedFromGroupState(group, matches, scores));
  const ranking = rankGroup(group.teams, playedMatches, table, rng, randomTiebreakTeams);
  ranking.forEach((team, index) => {
    if (table[team]) {
      table[team].position = index + 1;
      table[team].randomTiebreak = randomTiebreakTeams.has(team);
    }
  });

  return { table, ranking, randomTiebreakTeams };
}

function bestThirdPlace(
  groupTables: Array<{ ranking: string[]; table: Record<string, GroupTableRow> }>
) {
  const entries: Array<{
    team: string;
    group: string;
    points: number;
    gd: number;
    gf: number;
  }> = [];
  for (const { ranking, table } of groupTables) {
    if (ranking.length < 3) {
      continue;
    }
    const team = ranking[2];
    const row = table[team];
    entries.push({
      team,
      group: row.group,
      points: row.points,
      gd: row.gd,
      gf: row.gf,
    });
  }
  const rng = createRng(seedFromThirdPlace(entries));
  const randomTiebreakTeams = new Set<string>();
  entries.sort((a, b) => {
    if (b.points !== a.points) {
      return b.points - a.points;
    }
    if (b.gd !== a.gd) {
      return b.gd - a.gd;
    }
    if (b.gf !== a.gf) {
      return b.gf - a.gf;
    }
    return 0;
  });
  const ordered: typeof entries = [];
  let i = 0;
  while (i < entries.length) {
    const current = entries[i];
    const tied = [current];
    i += 1;
    while (i < entries.length) {
      const next = entries[i];
      if (
        current.points === next.points &&
        current.gd === next.gd &&
        current.gf === next.gf
      ) {
        tied.push(next);
        i += 1;
      } else {
        break;
      }
    }
    if (tied.length > 1) {
      shuffleInPlace(tied, rng);
      tied.forEach((entry) => randomTiebreakTeams.add(entry.team));
    }
    ordered.push(...tied);
  }
  return { entries: ordered, randomTiebreakTeams };
}

function resolveGroupPlaceholder(
  label: string,
  groupRankings: Record<string, string[]>,
  thirdPlaceByGroup: Record<string, string>,
  groupCompletion: Record<string, boolean>,
  allowThirdPlaceResolve: boolean,
  qualifiedThirdGroups?: Set<string>
) {
  if (label.startsWith("Winner Group ")) {
    const group = label.replace("Winner Group ", "").trim();
    if (groupCompletion[group]) {
      return groupRankings[group]?.[0] ?? label;
    }
    return formatGroupPlaceholder(label);
  }
  if (label.startsWith("Runner-up Group ")) {
    const group = label.replace("Runner-up Group ", "").trim();
    if (groupCompletion[group]) {
      return groupRankings[group]?.[1] ?? label;
    }
    return formatGroupPlaceholder(label);
  }
  if (label.startsWith("3rd Group ")) {
    const group = label.replace("3rd Group ", "").trim();
    if (group.length === 1) {
      if (
        allowThirdPlaceResolve &&
        (!qualifiedThirdGroups || qualifiedThirdGroups.has(group))
      ) {
        return thirdPlaceByGroup[group] ?? label;
      }
      return formatGroupPlaceholder(label);
    }
  }
  return formatGroupPlaceholder(label);
}

function formatGroupPlaceholder(label: string) {
  if (label.startsWith("Winner Group ")) {
    return label.replace("Winner Group ", "1st Group ");
  }
  if (label.startsWith("Runner-up Group ")) {
    return label.replace("Runner-up Group ", "2nd Group ");
  }
  if (label.startsWith("3rd Group ")) {
    return label.replace("3rd Group ", "3rd Gr. ");
  }
  return label;
}

function formatStageShort(stage: string | undefined) {
  switch (stage) {
    case "Round of 32":
      return "R32";
    case "Round of 16":
      return "R16";
    case "Quarterfinal":
      return "QF";
    case "Semifinal":
      return "SF";
    case "Final":
      return "Final";
    default:
      return null;
  }
}

function resolveKnockoutLabel({
  label,
  opponentLabel,
  groupRankings,
  thirdPlaceByGroup,
  thirdPlaceAssignments,
  knockoutWinners,
  knockoutLosers,
  groupCompletion,
  allowThirdPlaceResolve,
  qualifiedThirdGroups,
  matchStageById,
}: {
  label: string;
  opponentLabel: string;
  groupRankings: Record<string, string[]>;
  thirdPlaceByGroup: Record<string, string>;
  thirdPlaceAssignments: Record<string, string> | null;
  knockoutWinners: Map<number, string>;
  knockoutLosers: Map<number, string>;
  groupCompletion: Record<string, boolean>;
  allowThirdPlaceResolve: boolean;
  qualifiedThirdGroups?: Set<string>;
  matchStageById: Record<number, string>;
}) {
  if (/^UEFA Path\s+.+\s+Winner$/i.test(label)) {
    return label.replace(/\s+Winner$/i, "");
  }
  if (/^IC Path\s+.+\s+Winner$/i.test(label)) {
    return label.replace(/\s+Winner$/i, "");
  }
  if (label.startsWith("Winner Match ")) {
    const matchId = Number(label.replace("Winner Match ", "").trim());
    const winner = knockoutWinners.get(matchId);
    if (winner) {
      return winner;
    }
    const stage = formatStageShort(matchStageById[matchId]);
    return stage ? `Winner ${stage}` : label;
  }
  if (label.startsWith("Winner UEFA Path ")) {
    return label.replace("Winner ", "");
  }
  if (label.startsWith("Winner IC Path ")) {
    return label.replace("Winner ", "");
  }
  if (label.startsWith("Loser Match ")) {
    const matchId = Number(label.replace("Loser Match ", "").trim());
    const loser = knockoutLosers.get(matchId);
    if (loser) {
      return loser;
    }
    const stage = formatStageShort(matchStageById[matchId]);
    return stage ? `Loser ${stage}` : label;
  }
  if (
    allowThirdPlaceResolve &&
    label.startsWith("3rd Group ") &&
    opponentLabel.startsWith("Winner Group ")
  ) {
    const winnerGroup = opponentLabel.replace("Winner Group ", "").trim();
    const key = `1${winnerGroup}`;
    const assignedGroup = thirdPlaceAssignments?.[key];
    if (assignedGroup) {
      return thirdPlaceByGroup[assignedGroup] ?? label;
    }
  }
  return resolveGroupPlaceholder(
    label,
    groupRankings,
    thirdPlaceByGroup,
    groupCompletion,
    allowThirdPlaceResolve,
    qualifiedThirdGroups
  );
}

function TeamFlag({
  team,
  flags,
}: {
  team: string;
  flags: Record<string, string | null>;
}) {
  const isPlaceholder = isPlaceholderLabel(team);
  const flagPath = flags[team];
  if (flagPath) {
    return (
      <div
        className={cn(
          "relative h-5 w-7 shrink-0 overflow-hidden rounded-[1px] border border-ink-900",
          isPlaceholder ? "bg-[#d9d9d9]" : "bg-ink-800"
        )}
      >
        <Image
          src={flagPath}
          alt={`${team} flag`}
          fill
          className="object-cover"
          sizes="24px"
        />
      </div>
    );
  }
  return (
    <div
      className={cn(
        "flex h-5 w-7 shrink-0 items-center justify-center rounded-[1px] border border-ink-900 text-[9px] font-semibold uppercase",
        isPlaceholder ? "bg-[#d9d9d9] text-transparent" : "bg-ink-800 text-ink-200"
      )}
    >
      {team && !isPlaceholder ? teamInitials(team) : ""}
    </div>
  );
}

function TeamBox({
  team,
  flags,
  score,
  onScoreChange,
  reverse,
  disabled,
  placeholder,
  onSelect,
  highlight,
  showScore = true,
  winProb,
  className,
}: {
  team: string;
  flags: Record<string, string | null>;
  score?: number | null;
  onScoreChange?: (value: number | null) => void;
  reverse?: boolean;
  disabled?: boolean;
  placeholder?: boolean;
  onSelect?: () => void;
  highlight?: boolean;
  showScore?: boolean;
  winProb?: string;
  className?: string;
}) {
  const formatted = formatDisplayLabel(team);
  const displayName =
    formatted === "Bosnia and Herzegovina" ? "Bosnia and Herz." : formatted;
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-[3px] border border-ink-900 bg-white px-2 py-1 text-xs lg:text-sm",
        showScore ? "w-[240px]" : "w-[200px]",
        reverse && "flex-row-reverse text-right",
        disabled && "bg-white text-ink-400",
        highlight && "border-ink-900 bg-[#f2e2e2] text-ebony",
        className
      )}
      onClick={disabled ? undefined : onSelect}
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
      onKeyDown={
        onSelect
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect();
              }
            }
          : undefined
      }
    >
      <TeamFlag team={team} flags={flags} />
      <span
        className={cn(
          "min-w-0 flex-1 truncate whitespace-nowrap text-xs font-medium text-ebony lg:text-sm"
        )}
      >
        {displayName || "TBD"}
      </span>
      {(winProb || showScore) && (
        <div
          className={cn(
            "flex shrink-0 items-center gap-1",
            !showScore && "ml-auto"
          )}
        >
          {reverse ? (
            <>
              {showScore && (
                <input
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={31}
                  value={score ?? ""}
                  onChange={(event) =>
                    onScoreChange?.(parseScore(event.target.value))
                  }
                  disabled={disabled}
                  onClick={(event) => event.stopPropagation()}
                  className="w-8 rounded border border-ink-900 bg-white text-right text-xs font-mono text-ink-200 focus:outline-none lg:text-sm"
                />
              )}
              {winProb && (
                <span className="text-[10px] font-semibold text-ink-400 lg:text-xs font-mono">
                  {winProb}
                </span>
              )}
            </>
          ) : (
            <>
              {winProb && (
                <span className="text-[10px] font-semibold text-ink-400 lg:text-xs font-mono">
                  {winProb}
                </span>
              )}
              {showScore && (
                <input
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={31}
                  value={score ?? ""}
                  onChange={(event) =>
                    onScoreChange?.(parseScore(event.target.value))
                  }
                  disabled={disabled}
                  onClick={(event) => event.stopPropagation()}
                  className="w-8 rounded border border-ink-900 bg-white text-right text-xs font-mono text-ink-200 focus:outline-none lg:text-sm"
                />
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function MatchCard({
  id,
  homeTeam,
  awayTeam,
  scores,
  onScoreChange,
  onScoreChangePair,
  allowDraw,
  orientation,
  flags,
  disabled,
  stackMode,
  fixedHeight,
  homeBoxRef,
  awayBoxRef,
  showScore = true,
  winnerSelection = null,
  onWinnerSelect,
  homeWinProb,
  awayWinProb,
  drawProb,
}: {
  id: string | number;
  homeTeam: string;
  awayTeam: string;
  scores?: Record<string, MatchScore>;
  onScoreChange?: (
    id: string | number,
    side: "home" | "away",
    value: number | null
  ) => void;
  onScoreChangePair?: (
    id: string | number,
    home: number | null,
    away: number | null
  ) => void;
  allowDraw: boolean;
  orientation: "horizontal" | "vertical";
  flags: Record<string, string | null>;
  disabled?: boolean;
  stackMode?: "centered";
  fixedHeight?: number;
  homeBoxRef?: React.Ref<HTMLDivElement>;
  awayBoxRef?: React.Ref<HTMLDivElement>;
  showScore?: boolean;
  winnerSelection?: WinnerSelection;
  onWinnerSelect?: (selection: WinnerSelection) => void;
  homeWinProb?: string;
  awayWinProb?: string;
  drawProb?: string | null;
}) {
  const score = showScore
    ? scores?.[String(id)] ?? { home: null, away: null }
    : { home: null, away: null };
  const hasScore = showScore && score.home !== null && score.away !== null;
  const isDraw = hasScore && score.home === score.away;
  const selection = showScore ? null : winnerSelection ?? null;
  const winner = showScore
    ? hasScore && score.home !== score.away
      ? score.home > score.away
        ? homeTeam
        : awayTeam
      : undefined
    : selection === "home"
      ? homeTeam
      : selection === "away"
        ? awayTeam
        : undefined;
  const highlightTeams = !showScore || !allowDraw || !isDraw;
  const placeholderHome = isPlaceholderLabel(homeTeam);
  const placeholderAway = isPlaceholderLabel(awayTeam);
  const homeProb = placeholderHome || placeholderAway ? undefined : homeWinProb;
  const awayProb = placeholderHome || placeholderAway ? undefined : awayWinProb;
  const drawLabel =
    showScore && allowDraw
      ? placeholderHome || placeholderAway
        ? ""
        : drawProb ?? "Draw"
      : null;
  const isDisabled = disabled || placeholderHome || placeholderAway;
  const setScores = (home: number | null, away: number | null) => {
    if (onScoreChangePair) {
      onScoreChangePair(id, home, away);
      return;
    }
    onScoreChange?.(id, "home", home);
    onScoreChange?.(id, "away", away);
  };
  const selectWinner = (side: "home" | "away") => {
    if (!onWinnerSelect) {
      return;
    }
    onWinnerSelect(selection === side ? null : side);
  };

  const drawButton =
    showScore && allowDraw ? (
    <button
      type="button"
      onClick={() => {
        if (score.home === 1 && score.away === 1) {
          setScores(null, null);
          return;
        }
        setScores(1, 1);
      }}
      disabled={isDisabled}
      className={cn(
        "border border-ink-900 w-[40px] px-2 py-1 text-[10px] font-semibold uppercase lg:text-xs font-mono",
        orientation === "horizontal" ? "rounded-none -mx-px" : "rounded-[3px]",
        isDraw ? "bg-[#f2e2e2] text-ebony" : "text-ink-400",
        isDisabled && "opacity-50"
      )}
    >
      {drawLabel}
    </button>
    ) : null;
  const highlightHome =
    !isDisabled && (highlightTeams ? winner === homeTeam : false);
  const highlightAway =
    !isDisabled && (highlightTeams ? winner === awayTeam : false);

  if (orientation === "vertical") {
    if (stackMode === "centered" && !allowDraw) {
      return (
        <div
          className="relative overflow-visible"
          style={fixedHeight ? { height: fixedHeight } : undefined}
        >
          <div
            className={cn(
              "grid",
              fixedHeight ? "h-full grid-rows-2" : "grid-rows-[auto_auto]"
            )}
          >
            <div className="flex items-end overflow-visible">
              <div ref={homeBoxRef}>
                <TeamBox
                  team={homeTeam}
                  flags={flags}
                  score={score.home}
                  onScoreChange={(value) => onScoreChange?.(id, "home", value)}
                  onSelect={() => {
                    if (showScore) {
                      if (score.home === 2 && score.away === 1) {
                        setScores(null, null);
                        return;
                      }
                      setScores(2, 1);
                    } else {
                      selectWinner("home");
                    }
                  }}
                  highlight={highlightHome}
                  disabled={isDisabled}
                  placeholder={placeholderHome}
                  showScore={showScore}
                  winProb={homeProb}
                  className="rounded-b-none"
                />
              </div>
            </div>
            <div className="flex items-start overflow-visible">
              <div ref={awayBoxRef}>
                <TeamBox
                  team={awayTeam}
                  flags={flags}
                  score={score.away}
                  onScoreChange={(value) => onScoreChange?.(id, "away", value)}
                  onSelect={() => {
                    if (showScore) {
                      if (score.home === 1 && score.away === 2) {
                        setScores(null, null);
                        return;
                      }
                      setScores(1, 2);
                    } else {
                      selectWinner("away");
                    }
                  }}
                  highlight={highlightAway}
                  disabled={isDisabled}
                  placeholder={placeholderAway}
                  showScore={showScore}
                  winProb={awayProb}
                  className="rounded-t-none border-t-0"
                />
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col gap-2">
        <TeamBox
          team={homeTeam}
          flags={flags}
          score={score.home}
          onScoreChange={(value) => onScoreChange?.(id, "home", value)}
          onSelect={() => {
            if (showScore) {
              if (score.home === 2 && score.away === 1) {
                setScores(null, null);
                return;
              }
              setScores(2, 1);
            } else {
              selectWinner("home");
            }
          }}
          highlight={highlightHome}
          disabled={isDisabled}
          placeholder={placeholderHome}
          showScore={showScore}
          winProb={homeProb}
        />
        {drawButton}
        <TeamBox
          team={awayTeam}
          flags={flags}
          score={score.away}
          onScoreChange={(value) => onScoreChange?.(id, "away", value)}
          onSelect={() => {
            if (showScore) {
              if (score.home === 1 && score.away === 2) {
                setScores(null, null);
                return;
              }
              setScores(1, 2);
            } else {
              selectWinner("away");
            }
          }}
          highlight={highlightAway}
          disabled={isDisabled}
          placeholder={placeholderAway}
          showScore={showScore}
          winProb={awayProb}
        />
      </div>
    );
  }

  const horizontalHomeClass = "rounded-r-none";
  const horizontalAwayClass = "rounded-l-none";

  return (
    <div className="flex items-stretch gap-0">
      <div ref={homeBoxRef}>
        <TeamBox
          team={homeTeam}
          flags={flags}
          score={score.home}
          onScoreChange={(value) => onScoreChange?.(id, "home", value)}
          onSelect={() => {
            if (showScore) {
              if (score.home === 2 && score.away === 1) {
                setScores(null, null);
                return;
              }
              setScores(2, 1);
            } else {
              selectWinner("home");
            }
          }}
          highlight={highlightHome}
          disabled={isDisabled}
          placeholder={placeholderHome}
          showScore={showScore}
          winProb={homeProb}
          className={horizontalHomeClass}
        />
      </div>
      {drawButton}
      <div ref={awayBoxRef}>
        <TeamBox
          team={awayTeam}
          flags={flags}
          score={score.away}
          onScoreChange={(value) => onScoreChange?.(id, "away", value)}
          reverse
          onSelect={() => {
            if (showScore) {
              if (score.home === 1 && score.away === 2) {
                setScores(null, null);
                return;
              }
              setScores(1, 2);
            } else {
              selectWinner("away");
            }
          }}
          highlight={highlightAway}
          disabled={isDisabled}
          placeholder={placeholderAway}
          showScore={showScore}
          winProb={awayProb}
          className={horizontalAwayClass}
        />
      </div>
    </div>
  );
}

function QualifierPathBracket({
  path,
  matches,
  winnerSelections,
  onWinnerSelect,
  flags,
  getMatchProbabilityLabels,
}: {
  path: string;
  matches: ResolvedQualifierMatch[];
  winnerSelections: Record<string, WinnerSelection>;
  onWinnerSelect: (id: string | number, selection: WinnerSelection) => void;
  flags: Record<string, string | null>;
  getMatchProbabilityLabels: (params: {
    homeTeam: string;
    awayTeam: string;
    allowDraw: boolean;
    country?: string | null;
    neutralOverride?: boolean | null;
  }) => MatchProbabilityLabels;
}) {
  const semis = matches.filter((match) => match.round.startsWith("semi"));
  const final = matches.find((match) => match.round === "final");
  const finalId = final?.id ?? null;
  const finalProbabilities = final
    ? getMatchProbabilityLabels({
        homeTeam: final.homeResolved ?? final.homeTeam,
        awayTeam: final.awayResolved ?? final.awayTeam,
        allowDraw: false,
        neutralOverride: final.neutral,
      })
    : null;
  const semisKey = React.useMemo(
    () => semis.map((match) => String(match.id)).join("|"),
    [semis]
  );
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const bracketRef = React.useRef<HTMLDivElement | null>(null);
  const matchRefs = React.useRef(new Map<string | number, HTMLDivElement>());
  const matchHomeRefs = React.useRef(new Map<string | number, HTMLDivElement>());
  const matchAwayRefs = React.useRef(new Map<string | number, HTMLDivElement>());
  const [paths, setPaths] = React.useState<string[]>([]);
  const [semisOffset, setSemisOffset] = React.useState(0);

  React.useLayoutEffect(() => {
    const container = containerRef.current;
    const bracket = bracketRef.current;
    if (!container || !bracket || !final) {
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const rect = bracket.getBoundingClientRect();
        const finalEl = matchRefs.current.get(final.id);
        if (!finalEl) {
          return;
        }
        const finalRect = finalEl.getBoundingClientRect();
        const nextPaths: string[] = [];
        const connectorInset = 12;
        const connectorStrokeWidth = 2;
        const finalTarget = path.startsWith("IC Path") ? 0.75 : 0.5;
        const endX = finalRect.left - rect.left + connectorInset;
        const finalHomeBox = matchHomeRefs.current.get(final.id);
        const finalAwayBox = matchAwayRefs.current.get(final.id);
        let endY = finalRect.top - rect.top + finalRect.height * finalTarget;
        if (path.startsWith("IC Path") && finalAwayBox) {
          const awayRect = finalAwayBox.getBoundingClientRect();
          endY = awayRect.top - rect.top + awayRect.height / 2;
        } else if (finalHomeBox && finalAwayBox) {
          const homeRect = finalHomeBox.getBoundingClientRect();
          const awayRect = finalAwayBox.getBoundingClientRect();
          endY = (homeRect.bottom + awayRect.top) / 2 - rect.top;
        }
        if (path.startsWith("IC Path") && semis.length === 1) {
          const semiEl = matchRefs.current.get(semis[0].id);
          if (semiEl) {
            const semiRect = semiEl.getBoundingClientRect();
            let startY = semiRect.top - rect.top + semiRect.height / 2;
            const semiHomeBox = matchHomeRefs.current.get(semis[0].id);
            const semiAwayBox = matchAwayRefs.current.get(semis[0].id);
            if (semiHomeBox && semiAwayBox) {
              const homeRect = semiHomeBox.getBoundingClientRect();
              const awayRect = semiAwayBox.getBoundingClientRect();
              const dividerY = (homeRect.bottom + awayRect.top) / 2;
              startY = dividerY - rect.top;
            }
            const baseCenter = startY - semisOffset;
            const desiredOffset =
              endY - baseCenter + connectorStrokeWidth / 4;
            setSemisOffset((prev) =>
              Math.abs(prev - desiredOffset) < 0.5 ? prev : desiredOffset
            );
          }
        } else if (semisOffset !== 0) {
          setSemisOffset(0);
        }
        semis.forEach((match) => {
          const semiEl = matchRefs.current.get(match.id);
          if (!semiEl) {
            return;
          }
          const semiRect = semiEl.getBoundingClientRect();
          const startX = semiRect.right - rect.left - connectorInset;
          let startY = semiRect.top - rect.top + semiRect.height / 2;
          const semiHomeBox = matchHomeRefs.current.get(match.id);
          const semiAwayBox = matchAwayRefs.current.get(match.id);
          if (semiHomeBox && semiAwayBox) {
            const homeRect = semiHomeBox.getBoundingClientRect();
            const awayRect = semiAwayBox.getBoundingClientRect();
            startY = (homeRect.bottom + awayRect.top) / 2 - rect.top;
          }
          const midX = startX + (endX - startX) * 0.5;
          nextPaths.push(
            `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`
          );
        });
        setPaths(nextPaths);
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(container);
    observer.observe(bracket);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [finalId, semisKey, path, semisOffset]);

  return (
    <div
      ref={containerRef}
      className="relative flex flex-col gap-4 overflow-hidden rounded-md border border-ink-900 bg-white/80 p-4 shadow-soft"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ebony">{path}</h3>
        <span className="text-xs text-ink-400">{matches[0]?.stage}</span>
      </div>
      <div ref={bracketRef} className="relative w-fit max-w-full">
        <svg
          className="absolute inset-0 z-0 h-full w-full pointer-events-none"
          aria-hidden="true"
        >
          {paths.map((pathDef, index) => (
            <path
              key={`${path}-${index}`}
              d={pathDef}
              fill="none"
              stroke="var(--color-primary-dark)"
              strokeWidth={2}
            />
          ))}
        </svg>
        <div className="relative z-10 flex max-w-full items-center gap-6">
          <div
            className="flex flex-col gap-3"
            style={semisOffset ? { marginTop: semisOffset } : undefined}
          >
            {semis.map((match) => {
              const probabilities = getMatchProbabilityLabels({
                homeTeam: match.homeResolved ?? match.homeTeam,
                awayTeam: match.awayResolved ?? match.awayTeam,
                allowDraw: false,
                neutralOverride: match.neutral,
              });
              return (
                <div
                  key={match.id}
                  ref={(el) => {
                    if (el) {
                      matchRefs.current.set(match.id, el);
                    } else {
                      matchRefs.current.delete(match.id);
                    }
                  }}
                >
                  <MatchCard
                    id={match.id}
                    homeTeam={match.homeResolved ?? match.homeTeam}
                    awayTeam={match.awayResolved ?? match.awayTeam}
                    showScore={false}
                    winnerSelection={winnerSelections[String(match.id)] ?? null}
                    onWinnerSelect={(selection) => onWinnerSelect(match.id, selection)}
                    allowDraw={false}
                    orientation="vertical"
                    stackMode="centered"
                    homeWinProb={probabilities.homeWinProb}
                    awayWinProb={probabilities.awayWinProb}
                    drawProb={probabilities.drawProb}
                    homeBoxRef={(el) => {
                      if (el) {
                        matchHomeRefs.current.set(match.id, el);
                      } else {
                        matchHomeRefs.current.delete(match.id);
                      }
                    }}
                    awayBoxRef={(el) => {
                      if (el) {
                        matchAwayRefs.current.set(match.id, el);
                      } else {
                        matchAwayRefs.current.delete(match.id);
                      }
                    }}
                    flags={flags}
                  />
                </div>
              );
            })}
          </div>
          {final && (
            <div
              ref={(el) => {
                if (el) {
                  matchRefs.current.set(final.id, el);
                } else {
                  matchRefs.current.delete(final.id);
                }
              }}
            >
              <MatchCard
                id={final.id}
                homeTeam={final.homeResolved ?? final.homeTeam}
                awayTeam={final.awayResolved ?? final.awayTeam}
                showScore={false}
                winnerSelection={winnerSelections[String(final.id)] ?? null}
                onWinnerSelect={(selection) => onWinnerSelect(final.id, selection)}
                allowDraw={false}
                orientation="vertical"
                stackMode="centered"
                homeWinProb={finalProbabilities?.homeWinProb}
                awayWinProb={finalProbabilities?.awayWinProb}
                drawProb={finalProbabilities?.drawProb ?? null}
                homeBoxRef={(el) => {
                  if (el) {
                    matchHomeRefs.current.set(final.id, el);
                  } else {
                    matchHomeRefs.current.delete(final.id);
                  }
                }}
                awayBoxRef={(el) => {
                  if (el) {
                    matchAwayRefs.current.set(final.id, el);
                  } else {
                    matchAwayRefs.current.delete(final.id);
                  }
                }}
                flags={flags}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function GroupTable({
  group,
  rows,
  highlightThird,
  highlightWeakThird,
  highlightTop = 2,
  flags,
  showTieInfo,
}: {
  group: GroupDefinition;
  rows: GroupTableRow[];
  highlightThird: boolean;
  highlightWeakThird: boolean;
  highlightTop?: number;
  flags: Record<string, string | null>;
  showTieInfo: boolean;
}) {
  return (
    <div className="min-w-0 w-[520px] box-border overflow-hidden rounded border border-ink-900 bg-white/80 p-0 text-xs shadow-soft lg:text-sm">
      <table className="w-full table-fixed border-collapse text-xs text-ink-200 lg:text-sm">
        <colgroup>
          <col style={{ width: "32px" }} />
          <col style={{ width: "270px" }} />
          <col style={{ width: "32px" }} />
          <col style={{ width: "24px" }} />
          <col style={{ width: "24px" }} />
          <col style={{ width: "24px" }} />
          <col style={{ width: "28px" }} />
          <col style={{ width: "28px" }} />
          <col style={{ width: "28px" }} />
          <col style={{ width: "30px" }} />
        </colgroup>
        <thead>
          <tr className="h-9 border-b-2 border-ink-900 text-xs uppercase tracking-wide text-ink-400">
            <th className="px-1 py-1 pl-3 text-left w-[32px] normal-case">Pos</th>
            <th className="box-border w-[270px] px-0 py-1 text-left normal-case">
              <div className="pl-10 pr-1">Team</div>
            </th>
            <th className="px-1 py-1 text-right w-[32px] normal-case">Pld</th>
            <th className="px-1 py-1 text-right w-[24px]">W</th>
            <th className="px-1 py-1 text-right w-[24px]">D</th>
            <th className="px-1 py-1 text-right w-[24px]">L</th>
            <th className="px-1 py-1 text-right w-[28px]">GF</th>
            <th className="px-1 py-1 text-right w-[28px]">GA</th>
            <th className="px-1 py-1 text-right w-[28px]">GD</th>
            <th className="px-1 py-1 pr-3 text-right w-[30px] normal-case">Pts</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const isTopTwo = row.position <= highlightTop;
            const isThird = row.position === 3;
            const highlight = isTopTwo || (highlightThird && isThird);
            const weakHighlight = !highlightThird && highlightWeakThird && isThird;
            const isLastRow = index === rows.length - 1;
            return (
              <tr
                key={row.team}
                className={cn(
                  "h-9 border-b border-ink-900",
                  isLastRow && "border-b-0",
                  highlight && "bg-[#f2e2e2] text-ebony",
                  weakHighlight && "bg-[#f8f1f0] text-ebony"
                )}
              >
                <td className="px-1 py-1 pl-3 text-left font-mono">
                  {row.position}
                </td>
                <td className="box-border w-[270px] px-0 py-1 min-w-0">
                  <div className="flex min-w-0 items-center gap-2 px-1">
                    <TeamFlag team={row.team} flags={flags} />
                    <span className="truncate">{formatDisplayLabel(row.team)}</span>
                    {showTieInfo && row.randomTiebreak && (
                      <span
                        className="inline-flex h-4 w-4 flex-none items-center justify-center rounded-full border border-ink-400 text-[10px] font-semibold text-ink-400"
                        title={TIEBREAK_TOOLTIP}
                        aria-label={TIEBREAK_TOOLTIP}
                      >
                        i
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-1 py-1 text-right font-mono">{row.played}</td>
                <td className="px-1 py-1 text-right font-mono">{row.wins}</td>
                <td className="px-1 py-1 text-right font-mono">{row.draws}</td>
                <td className="px-1 py-1 text-right font-mono">{row.losses}</td>
                <td className="px-1 py-1 text-right font-mono">{row.gf}</td>
                <td className="px-1 py-1 text-right font-mono">{row.ga}</td>
                <td className="px-1 py-1 text-right font-mono">{row.gd}</td>
                <td className="px-1 py-1 pr-3 text-right font-mono">{row.points}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function groupMatchesFor(
  groupId: string,
  matches: GroupMatch[]
) {
  return matches
    .filter((match) => match.group === groupId)
    .sort((a, b) => a.id - b.id);
}

function matchesByStage<T extends KnockoutMatch>(matches: T[]) {
  const map = new Map<string, T[]>();
  for (const match of matches) {
    if (!map.has(match.stage)) {
      map.set(match.stage, []);
    }
    map.get(match.stage)?.push(match);
  }
  for (const [stage, list] of map.entries()) {
    list.sort((a, b) => a.id - b.id);
    map.set(stage, list);
  }
  return map;
}

function sortQualifiers(matches: QualifierMatch[]) {
  const roundOrder: Record<string, number> = {
    semi1: 1,
    semi2: 2,
    semi: 1,
    final: 3,
  };
  return [...matches].sort((a, b) => {
    const dateA = Date.parse(a.date);
    const dateB = Date.parse(b.date);
    if (dateA !== dateB) {
      return dateA - dateB;
    }
    const orderA = roundOrder[a.round] ?? 99;
    const orderB = roundOrder[b.round] ?? 99;
    return orderA - orderB;
  });
}

export function WorldCupPredictorPage({ data }: { data: WorldCupPredictorData }) {
  const [groupScores, setGroupScores] = React.useState<
    Record<string, MatchScore>
  >({});
  const [autoGroupScores, setAutoGroupScores] = React.useState<
    Record<string, boolean>
  >({});
  const [qualifierWinners, setQualifierWinners] = React.useState<
    Record<string, WinnerSelection>
  >({});
  const [autoQualifierWinners, setAutoQualifierWinners] = React.useState<
    Record<string, boolean>
  >({});
  const [knockoutWinners, setKnockoutWinners] = React.useState<
    Record<string, WinnerSelection>
  >({});
  const [autoKnockoutWinners, setAutoKnockoutWinners] = React.useState<
    Record<string, boolean>
  >({});
  const isNarrow = false;
  const knockoutContainerRef = React.useRef<HTMLDivElement | null>(null);
  const knockoutRefs = React.useRef(new Map<number, HTMLDivElement>());
  const [knockoutPaths, setKnockoutPaths] = React.useState<string[]>([]);
  const roundOf32ListRef = React.useRef<HTMLDivElement | null>(null);
  const finalListRef = React.useRef<HTMLDivElement | null>(null);
  const [knockoutListHeight, setKnockoutListHeight] = React.useState<number | null>(null);
  const [knockoutCenters, setKnockoutCenters] = React.useState<Record<number, number>>(
    {}
  );
  const [knockoutCardHeight, setKnockoutCardHeight] = React.useState<number | null>(
    null
  );
  const [thirdPlaceOffset, setThirdPlaceOffset] = React.useState<number | null>(null);
  const [finalCenterOverride, setFinalCenterOverride] = React.useState<number | null>(
    null
  );
  const matchStageById = React.useMemo(() => {
    const mapping: Record<number, string> = {};
    for (const match of data.knockoutMatches) {
      mapping[match.id] = match.stage;
    }
    return mapping;
  }, [data.knockoutMatches]);

  const getMatchProbabilityLabels = React.useCallback(
    ({
      homeTeam,
      awayTeam,
      allowDraw,
      country,
      neutralOverride,
    }: {
      homeTeam: string;
      awayTeam: string;
      allowDraw: boolean;
      country?: string | null;
      neutralOverride?: boolean | null;
    }): MatchProbabilityLabels => {
      const values = resolveMatchProbabilities({
        probabilities: data.winProbabilities,
        homeTeam,
        awayTeam,
        allowDraw,
        country,
        neutralOverride,
      });
      if (!values) {
        return { homeWinProb: undefined, awayWinProb: undefined, drawProb: null };
      }
      return {
        homeWinProb: formatProbability(values.home),
        awayWinProb: formatProbability(values.away),
        drawProb: allowDraw ? formatProbability(values.draw) ?? null : null,
      };
    },
    [data.winProbabilities]
  );

  const qualifierDependents = React.useMemo(() => {
    const byPathRound = new Map<string, string>();
    data.qualifiers.forEach((match) => {
      byPathRound.set(`${match.path}:${match.round}`, String(match.id));
    });
    const deps = new Map<string, Set<string>>();
    data.qualifiers.forEach((match) => {
      [match.homeSource, match.awaySource]
        .filter(Boolean)
        .forEach((source) => {
          const sourceId = byPathRound.get(`${match.path}:${source}`);
          if (!sourceId) {
            return;
          }
          if (!deps.has(sourceId)) {
            deps.set(sourceId, new Set());
          }
          deps.get(sourceId)?.add(String(match.id));
        });
    });
    return deps;
  }, [data.qualifiers]);

  const qualifierSlotsByMatch = React.useMemo(() => {
    const slotsByMatch = new Map<string, Set<string>>();
    const matchById = new Map<string, QualifierMatch>();
    data.qualifiers.forEach((match) => {
      matchById.set(String(match.id), match);
    });

    const collectSlots = (matchId: string, visited: Set<string>) => {
      if (visited.has(matchId)) {
        return new Set<string>();
      }
      visited.add(matchId);
      const slots = new Set<string>();
      const match = matchById.get(matchId);
      if (match?.winnerSlot) {
        slots.add(match.winnerSlot);
      }
      const deps = qualifierDependents.get(matchId);
      if (deps) {
        deps.forEach((dep) => {
          collectSlots(dep, visited).forEach((slot) => slots.add(slot));
        });
      }
      return slots;
    };

    matchById.forEach((_, matchId) => {
      slotsByMatch.set(matchId, collectSlots(matchId, new Set()));
    });

    return slotsByMatch;
  }, [data.qualifiers, qualifierDependents]);

  const groupMatchIdsByTeam = React.useMemo(() => {
    const map = new Map<string, Set<string>>();
    data.groupMatches.forEach((match) => {
      const home = match.homeTeam;
      const away = match.awayTeam;
      if (!map.has(home)) {
        map.set(home, new Set());
      }
      if (!map.has(away)) {
        map.set(away, new Set());
      }
      map.get(home)?.add(String(match.id));
      map.get(away)?.add(String(match.id));
    });
    return map;
  }, [data.groupMatches]);

  const groupIdsBySlot = React.useMemo(() => {
    const map = new Map<string, Set<string>>();
    data.groups.forEach((group) => {
      group.teams.forEach((team) => {
        if (!map.has(team)) {
          map.set(team, new Set());
        }
        map.get(team)?.add(group.id);
      });
    });
    return map;
  }, [data.groups]);

  const knockoutRootsByGroup = React.useMemo(() => {
    const map = new Map<string, Set<string>>();
    data.knockoutMatches.forEach((match) => {
      [match.homeLabel, match.awayLabel].forEach((label) => {
        const groupId = extractGroupId(label);
        if (!groupId) {
          return;
        }
        if (!map.has(groupId)) {
          map.set(groupId, new Set());
        }
        map.get(groupId)?.add(String(match.id));
      });
    });
    return map;
  }, [data.knockoutMatches]);

  const knockoutDependents = React.useMemo(() => {
    const deps = new Map<string, Set<string>>();
    data.knockoutMatches.forEach((match) => {
      [match.homeLabel, match.awayLabel].forEach((label) => {
        if (label.startsWith("Winner Match ")) {
          const from = label.replace("Winner Match ", "").trim();
          if (!deps.has(from)) {
            deps.set(from, new Set());
          }
          deps.get(from)?.add(String(match.id));
        } else if (label.startsWith("Loser Match ")) {
          const from = label.replace("Loser Match ", "").trim();
          if (!deps.has(from)) {
            deps.set(from, new Set());
          }
          deps.get(from)?.add(String(match.id));
        }
      });
    });
    return deps;
  }, [data.knockoutMatches]);

  const updateQualifierWinner = React.useCallback(
    (id: string | number, selection: WinnerSelection) => {
      let changed = false;
      const key = String(id);
      setQualifierWinners((prev) => {
        if ((prev[key] ?? null) === selection) {
          return prev;
        }
        changed = true;
        const next = { ...prev, [key]: selection };
        return clearDependentSelections(next, key, qualifierDependents);
      });
      if (changed) {
        setAutoQualifierWinners((prev) => {
          if (!prev[key]) {
            return prev;
          }
          const next = { ...prev };
          delete next[key];
          return next;
        });
        const affectedSlots =
          qualifierSlotsByMatch.get(String(id)) ?? new Set<string>();
        const affectedGroups = new Set<string>();
        affectedSlots.forEach((slot) => {
          const groups = groupIdsBySlot.get(slot);
          if (groups) {
            groups.forEach((groupId) => affectedGroups.add(groupId));
          }
        });

        if (affectedGroups.size > 0) {
          setGroupScores((prev) => {
            if (!Object.keys(prev).length) {
              return prev;
            }
            const next = { ...prev };
            affectedSlots.forEach((slot) => {
              const matchIds = groupMatchIdsByTeam.get(slot);
              if (!matchIds) {
                return;
              }
              matchIds.forEach((matchId) => {
                delete next[matchId];
              });
            });
            return next;
          });
          setAutoGroupScores((prev) => {
            const next = { ...prev };
            affectedSlots.forEach((slot) => {
              const matchIds = groupMatchIdsByTeam.get(slot);
              if (!matchIds) {
                return;
              }
              matchIds.forEach((matchId) => {
                delete next[matchId];
              });
            });
            return next;
          });

          setKnockoutWinners((prev) => {
            if (!Object.keys(prev).length) {
              return prev;
            }
            let next = { ...prev };
            affectedGroups.forEach((groupId) => {
              const rootMatches = knockoutRootsByGroup.get(groupId);
              if (rootMatches) {
                rootMatches.forEach((matchId) => {
                  next[matchId] = null;
                  next = clearDependentSelections(
                    next,
                    matchId,
                    knockoutDependents
                  );
                });
              }
            });
            const clearedIds = Object.keys(prev).filter(
              (matchId) => prev[matchId] && next[matchId] === null
            );
            if (clearedIds.length > 0) {
              setAutoKnockoutWinners((currentAuto) => {
                const nextAuto = { ...currentAuto };
                clearedIds.forEach((matchId) => {
                  delete nextAuto[matchId];
                });
                return nextAuto;
              });
            }
            return next;
          });
        }
      }
    },
    [
      qualifierDependents,
      qualifierSlotsByMatch,
      groupIdsBySlot,
      groupMatchIdsByTeam,
      knockoutRootsByGroup,
      knockoutDependents,
    ]
  );

  const updateKnockoutWinner = React.useCallback(
    (id: string | number, selection: WinnerSelection) => {
      const key = String(id);
      setAutoKnockoutWinners((prev) => {
        if (!prev[key]) {
          return prev;
        }
        const next = { ...prev };
        delete next[key];
        return next;
      });
      setKnockoutWinners((prev) => {
        if ((prev[key] ?? null) === selection) {
          return prev;
        }
        const next = { ...prev, [key]: selection };
        return clearDependentSelections(next, key, knockoutDependents);
      });
    },
    [knockoutDependents]
  );

  const qualifierState = React.useMemo(
    () => resolveQualifierState(data.qualifiers, qualifierWinners),
    [data.qualifiers, qualifierWinners]
  );

  const slotWinners = qualifierState.slotWinners;

  const computeKnockoutContext = React.useCallback(
    (
      scores: Record<string, MatchScore>,
      slotWinnersOverride?: Map<string, string>
    ) => {
      const resolvedSlots = slotWinnersOverride ?? slotWinners;
      const resolvedMatches = data.groupMatches.map((match) => ({
        ...match,
        homeTeam: resolvedSlots.get(match.homeTeam) ?? match.homeTeam,
        awayTeam: resolvedSlots.get(match.awayTeam) ?? match.awayTeam,
      }));
      const resolvedGroupsLocal = data.groups.map((group) => ({
        ...group,
        teams: group.teams.map((team) => resolvedSlots.get(team) ?? team),
      }));
      const groupTablesLocal = resolvedGroupsLocal.map((group) => {
        const matches = groupMatchesFor(group.id, resolvedMatches);
        const { table, ranking } = buildGroupTable(group, matches, scores);
        const rows = ranking.map((team) => table[team]).filter(Boolean);
        return { group, ranking, table, rows };
      });
      const groupRankingsLocal: Record<string, string[]> = {};
      const groupCompletionLocal: Record<string, boolean> = {};
      groupTablesLocal.forEach((entry) => {
        groupRankingsLocal[entry.group.id] = entry.ranking;
        const matches = groupMatchesFor(entry.group.id, resolvedMatches);
        groupCompletionLocal[entry.group.id] = matches.every((match) => {
          const score = scores[String(match.id)];
          return score && score.home !== null && score.away !== null;
        });
      });
      const thirdPlaceEntries = bestThirdPlace(groupTablesLocal).entries;
      const thirdPlaceByGroupLocal: Record<string, string> = {};
      thirdPlaceEntries.forEach((entry) => {
        if (!thirdPlaceByGroupLocal[entry.group]) {
          thirdPlaceByGroupLocal[entry.group] = entry.team;
        }
      });
      const bestThirdGroups = thirdPlaceEntries.slice(0, 8);
      const qualifiedThirdGroupsLocal = new Set(
        bestThirdGroups.map((entry) => entry.group)
      );
      const groups = bestThirdGroups.map((entry) => entry.group).sort();
      const comboKey = groups.join("");
      const thirdPlaceAssignments =
        comboKey && data.roundOf32Combos[comboKey]
          ? data.roundOf32Combos[comboKey]
          : null;
      const allGroupMatchesComplete = resolvedMatches.every((match) => {
        const score = scores[String(match.id)];
        return score && score.home !== null && score.away !== null;
      });
      const labels = new Map<string, { home: string; away: string }>();
      data.knockoutMatches.forEach((match) => {
        const homeResolved = resolveKnockoutLabel({
          label: match.homeLabel,
          opponentLabel: match.awayLabel,
          groupRankings: groupRankingsLocal,
          thirdPlaceByGroup: thirdPlaceByGroupLocal,
          thirdPlaceAssignments,
          knockoutWinners: new Map(),
          knockoutLosers: new Map(),
          groupCompletion: groupCompletionLocal,
          allowThirdPlaceResolve: allGroupMatchesComplete,
          qualifiedThirdGroups: qualifiedThirdGroupsLocal,
          matchStageById,
        });
        const awayResolved = resolveKnockoutLabel({
          label: match.awayLabel,
          opponentLabel: match.homeLabel,
          groupRankings: groupRankingsLocal,
          thirdPlaceByGroup: thirdPlaceByGroupLocal,
          thirdPlaceAssignments,
          knockoutWinners: new Map(),
          knockoutLosers: new Map(),
          groupCompletion: groupCompletionLocal,
          allowThirdPlaceResolve: allGroupMatchesComplete,
          qualifiedThirdGroups: qualifiedThirdGroupsLocal,
          matchStageById,
        });
        labels.set(String(match.id), {
          home: homeResolved,
          away: awayResolved,
        });
      });
      return {
        labels,
        allGroupMatchesComplete,
        comboKey,
        bestThirdGroups: bestThirdGroups.map((entry) => entry.group),
        qualifiedThirdGroups: Array.from(qualifiedThirdGroupsLocal),
        thirdPlaceAssignments,
      thirdPlaceByGroup: thirdPlaceByGroupLocal,
      groupRankings: groupRankingsLocal,
      groupCompletion: groupCompletionLocal,
    };
  },
  [
    data.groupMatches,
    data.groups,
    data.knockoutMatches,
    data.roundOf32Combos,
    matchStageById,
    slotWinners,
  ]
);

  const clearKnockoutSelectionsByMatchIds = React.useCallback(
    (
      current: Record<string, WinnerSelection>,
      matchIds: Iterable<string>
    ) => {
      let next = { ...current };
      for (const matchId of matchIds) {
        next[matchId] = null;
        next = clearDependentSelections(next, matchId, knockoutDependents);
      }
      const clearedIds = Object.keys(current).filter(
        (id) => current[id] && next[id] === null
      );
      return { next, clearedIds };
    },
    [knockoutDependents]
  );

  const computeClearedKnockoutSelections = React.useCallback(
    (
      current: Record<string, WinnerSelection>,
      previousScores: Record<string, MatchScore>,
      nextScores: Record<string, MatchScore>,
      options?: {
        logChanges?: boolean;
        previousSlotWinners?: Map<string, string>;
        nextSlotWinners?: Map<string, string>;
      }
    ) => {
      if (!Object.keys(current).length) {
        return { nextWinners: current, clearedIds: [] as string[] };
      }
      const previousContext = computeKnockoutContext(
        previousScores,
        options?.previousSlotWinners
      );
      const nextContext = computeKnockoutContext(
        nextScores,
        options?.nextSlotWinners
      );
      const previousLabels = previousContext.labels;
      const nextLabels = nextContext.labels;
      const changedMatches = new Set<string>();
      data.knockoutMatches.forEach((match) => {
        const key = String(match.id);
        const before = previousLabels.get(key);
        const after = nextLabels.get(key);
        if (!before || !after) {
          return;
        }
        if (before.home !== after.home || before.away !== after.away) {
          changedMatches.add(key);
        }
      });
      if (changedMatches.size === 0) {
        return { nextWinners: current, clearedIds: [] as string[] };
      }
      let changedMatchDetails: Array<{
        matchId: string;
        stage: string;
        rawLabels: { homeLabel: string; awayLabel: string } | null;
        winnerGroupKey: string | null;
        assignedGroup: string | null;
        nextAssignedGroup: string | null;
        previousThirdTeam: string | null;
        nextThirdTeam: string | null;
        before: { home: string; away: string } | undefined;
        after: { home: string; away: string } | undefined;
      }> = [];
      if (options?.logChanges) {
        changedMatchDetails = Array.from(changedMatches).map((matchId) => {
          const before = previousLabels.get(matchId);
          const after = nextLabels.get(matchId);
          const match = data.knockoutMatches.find(
            (entry) => String(entry.id) === matchId
          );
          const winnerGroupFromLabel = (label?: string) => {
            if (!label || !label.startsWith("Winner Group ")) {
              return null;
            }
            return `1${label.replace("Winner Group ", "").trim()}`;
          };
          const winnerGroupKey =
            winnerGroupFromLabel(match?.homeLabel) ??
            winnerGroupFromLabel(match?.awayLabel);
          const assignedGroup =
            winnerGroupKey && previousContext.thirdPlaceAssignments
              ? previousContext.thirdPlaceAssignments[winnerGroupKey]
              : null;
          const nextAssignedGroup =
            winnerGroupKey && nextContext.thirdPlaceAssignments
              ? nextContext.thirdPlaceAssignments[winnerGroupKey]
              : null;
          const previousThirdTeam = assignedGroup
            ? previousContext.thirdPlaceByGroup[assignedGroup]
            : null;
          const nextThirdTeam = nextAssignedGroup
            ? nextContext.thirdPlaceByGroup[nextAssignedGroup]
            : null;
          return {
            matchId,
            stage: matchStageById[Number(matchId)],
            rawLabels: match
              ? { homeLabel: match.homeLabel, awayLabel: match.awayLabel }
              : null,
            winnerGroupKey,
            assignedGroup,
            nextAssignedGroup,
            previousThirdTeam,
            nextThirdTeam,
            before,
            after,
          };
        });
      }
      const { next, clearedIds } = clearKnockoutSelectionsByMatchIds(
        current,
        changedMatches
      );
      if (options?.logChanges) {
        const describeSideChange = (
          label: string | undefined,
          beforeTeam: string | undefined,
          afterTeam: string | undefined
        ) => {
          if (!beforeTeam || !afterTeam || beforeTeam === afterTeam) {
            return null;
          }
          if (label) {
            return `${label} changed (${beforeTeam} -> ${afterTeam})`;
          }
          return `participants changed (${beforeTeam} -> ${afterTeam})`;
        };
        const qualifiedNext = new Set(nextContext.qualifiedThirdGroups ?? []);
        const thirdPlaceNote = (groupId: string) =>
          qualifiedNext.has(groupId) ? ` and ${groupId} is in top-8 thirds` : "";
        changedMatchDetails.forEach((detail) => {
          if (!detail.before || !detail.after) {
            return;
          }
          let reason = "";
          if (
            detail.assignedGroup &&
            detail.nextAssignedGroup &&
            detail.assignedGroup === detail.nextAssignedGroup &&
            detail.previousThirdTeam !== detail.nextThirdTeam
          ) {
            reason = `Match ${detail.matchId} cleared because Group ${detail.assignedGroup} third-place team changed (${detail.previousThirdTeam} -> ${detail.nextThirdTeam})${thirdPlaceNote(
              detail.assignedGroup
            )}.`;
          } else if (
            detail.assignedGroup &&
            detail.nextAssignedGroup &&
            detail.assignedGroup !== detail.nextAssignedGroup
          ) {
            reason = `Match ${detail.matchId} cleared because third-place assignment changed from Group ${detail.assignedGroup} to Group ${detail.nextAssignedGroup} (combo ${previousContext.comboKey} -> ${nextContext.comboKey}).`;
          } else {
            const parts = [
              describeSideChange(
                detail.rawLabels?.homeLabel,
                detail.before.home,
                detail.after.home
              ),
              describeSideChange(
                detail.rawLabels?.awayLabel,
                detail.before.away,
                detail.after.away
              ),
            ].filter(Boolean);
            reason = `Match ${detail.matchId} cleared because ${parts.length ? parts.join(" and ") : "participants changed"}.`;
          }
          console.log(`[predictor] ${reason}`);
        });
      }
      return { nextWinners: next, clearedIds };
    },
    [clearKnockoutSelectionsByMatchIds, computeKnockoutContext, data.knockoutMatches, matchStageById]
  );

  const clearKnockoutOnGroupChange = React.useCallback(
    (nextScores: Record<string, MatchScore>) => {
      setKnockoutWinners((prev) => {
        const { nextWinners, clearedIds } = computeClearedKnockoutSelections(
          prev,
          groupScores,
          nextScores,
          { logChanges: true }
        );
        if (clearedIds.length > 0) {
          setAutoKnockoutWinners((currentAuto) => {
            const nextAuto = { ...currentAuto };
            clearedIds.forEach((matchId) => {
              delete nextAuto[matchId];
            });
            return nextAuto;
          });
        }
        return nextWinners;
      });
    },
    [computeClearedKnockoutSelections, groupScores]
  );

  const updateGroupScore = React.useCallback(
    (id: string | number, side: "home" | "away", value: number | null) => {
      let changed = false;
      let nextScores: Record<string, MatchScore> | null = null;
      const key = String(id);
      setGroupScores((prev) => {
        const prevScore = prev[key] ?? { home: null, away: null };
        const nextScore = { ...prevScore, [side]: value };
        if (
          prevScore.home === nextScore.home &&
          prevScore.away === nextScore.away
        ) {
          return prev;
        }
        changed = true;
        nextScores = { ...prev, [key]: nextScore };
        return nextScores;
      });
      if (changed) {
        setAutoGroupScores((prev) => {
          if (!prev[key]) {
            return prev;
          }
          const next = { ...prev };
          delete next[key];
          return next;
        });
        if (nextScores) {
          clearKnockoutOnGroupChange(nextScores);
        }
      }
    },
    [clearKnockoutOnGroupChange]
  );

  const updateGroupScorePair = React.useCallback(
    (id: string | number, home: number | null, away: number | null) => {
      let changed = false;
      let nextScores: Record<string, MatchScore> | null = null;
      const key = String(id);
      setGroupScores((prev) => {
        const prevScore = prev[key] ?? { home: null, away: null };
        const nextScore = { home, away };
        if (
          prevScore.home === nextScore.home &&
          prevScore.away === nextScore.away
        ) {
          return prev;
        }
        changed = true;
        nextScores = { ...prev, [key]: nextScore };
        return nextScores;
      });
      if (changed && nextScores) {
        setAutoGroupScores((prev) => {
          if (!prev[key]) {
            return prev;
          }
          const next = { ...prev };
          delete next[key];
          return next;
        });
        clearKnockoutOnGroupChange(nextScores);
      }
    },
    [clearKnockoutOnGroupChange]
  );

  const resolvedGroups = React.useMemo(() => {
    return data.groups.map((group) => ({
      ...group,
      teams: group.teams.map((team) => slotWinners.get(team) ?? team),
    }));
  }, [data.groups, slotWinners]);

  const resolvedGroupMatches = React.useMemo(() => {
    return data.groupMatches.map((match) => ({
      ...match,
      homeTeam: slotWinners.get(match.homeTeam) ?? match.homeTeam,
      awayTeam: slotWinners.get(match.awayTeam) ?? match.awayTeam,
    }));
  }, [data.groupMatches, slotWinners]);

  const groupTables = React.useMemo(() => {
    return resolvedGroups.map((group) => {
      const matches = groupMatchesFor(group.id, resolvedGroupMatches);
      const { table, ranking } = buildGroupTable(group, matches, groupScores);
      const rows = ranking.map((team) => table[team]).filter(Boolean);
      return { group, ranking, table, rows };
    });
  }, [resolvedGroups, resolvedGroupMatches, groupScores]);

  const groupCompletion = React.useMemo(() => {
    const completion: Record<string, boolean> = {};
    data.groups.forEach((group) => {
      const matches = groupMatchesFor(group.id, resolvedGroupMatches);
      completion[group.id] = matches.every((match) => {
        const score = groupScores[String(match.id)];
        return score && score.home !== null && score.away !== null;
      });
    });
    return completion;
  }, [data.groups, resolvedGroupMatches, groupScores]);

  const groupRankings = React.useMemo(() => {
    const rankings: Record<string, string[]> = {};
    for (const entry of groupTables) {
      rankings[entry.group.id] = entry.ranking;
    }
    return rankings;
  }, [groupTables]);

  const thirdPlaceResults = React.useMemo(
    () => bestThirdPlace(groupTables),
    [groupTables]
  );
  const thirdPlaceEntries = thirdPlaceResults.entries;
  const thirdPlaceRandomTiebreaks = thirdPlaceResults.randomTiebreakTeams;
  const thirdPlaceRankingRows = React.useMemo(() => {
    const rowByTeam = new Map<string, GroupTableRow>();
    groupTables.forEach((entry) => {
      Object.values(entry.table).forEach((row) => {
        rowByTeam.set(row.team, row);
      });
    });
    return thirdPlaceEntries
      .map((entry, index) => {
        const row = rowByTeam.get(entry.team);
        if (!row) {
          return null;
        }
        return {
          ...row,
          position: index + 1,
          randomTiebreak:
            row.randomTiebreak || thirdPlaceRandomTiebreaks.has(entry.team),
        };
      })
      .filter((row): row is GroupTableRow => Boolean(row));
  }, [groupTables, thirdPlaceEntries, thirdPlaceRandomTiebreaks]);
  const bestThirdGroups = thirdPlaceEntries.slice(0, 8);
  const qualifiedThirdGroups = React.useMemo(
    () => new Set(bestThirdGroups.map((entry) => entry.group)),
    [bestThirdGroups]
  );
  const thirdPlaceByGroup = React.useMemo(() => {
    const mapping: Record<string, string> = {};
    for (const entry of thirdPlaceEntries) {
      if (!mapping[entry.group]) {
        mapping[entry.group] = entry.team;
      }
    }
    return mapping;
  }, [thirdPlaceEntries]);

  const allGroupMatchesComplete = React.useMemo(() => {
    return resolvedGroupMatches.every((match) => {
      const score = groupScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
  }, [resolvedGroupMatches, groupScores]);

  const thirdPlaceAssignments = React.useMemo(() => {
    const groups = bestThirdGroups.map((entry) => entry.group).sort();
    const comboKey = groups.join("");
    if (!comboKey) {
      return null;
    }
    return data.roundOf32Combos[comboKey] ?? null;
  }, [bestThirdGroups, data.roundOf32Combos]);

  const logRoundOf32Match = React.useCallback(
    (_match: ResolvedKnockoutMatch) => {},
    []
  );

  const knockoutState = React.useMemo(() => {
    const winners = new Map<number, string>();
    const losers = new Map<number, string>();
    const resolvedMatches: ResolvedKnockoutMatch[] = [];
    const sorted = [...data.knockoutMatches].sort((a, b) => a.id - b.id);

    for (const match of sorted) {
      const homeResolved = resolveKnockoutLabel({
        label: match.homeLabel,
        opponentLabel: match.awayLabel,
        groupRankings,
        thirdPlaceByGroup,
        thirdPlaceAssignments,
        knockoutWinners: winners,
        knockoutLosers: losers,
        groupCompletion,
        allowThirdPlaceResolve: allGroupMatchesComplete,
        qualifiedThirdGroups,
        matchStageById,
      });
      const awayResolved = resolveKnockoutLabel({
        label: match.awayLabel,
        opponentLabel: match.homeLabel,
        groupRankings,
        thirdPlaceByGroup,
        thirdPlaceAssignments,
        knockoutWinners: winners,
        knockoutLosers: losers,
        groupCompletion,
        allowThirdPlaceResolve: allGroupMatchesComplete,
        qualifiedThirdGroups,
        matchStageById,
      });
      const winner = resolveWinner(
        match.id,
        homeResolved,
        awayResolved,
        {},
        false,
        knockoutWinners
      );
      if (winner) {
        winners.set(match.id, winner);
        const loser = winner === homeResolved ? awayResolved : homeResolved;
        losers.set(match.id, loser);
      }
      resolvedMatches.push({
        ...match,
        homeResolved,
        awayResolved,
        winner,
      });
    }
    return { winners, losers, matches: resolvedMatches };
  }, [
    data.knockoutMatches,
    groupRankings,
    thirdPlaceByGroup,
    thirdPlaceAssignments,
    knockoutWinners,
    allGroupMatchesComplete,
    matchStageById,
  ]);

  React.useEffect(() => {
    if (process.env.NODE_ENV === "production") {
      return;
    }
    const context = computeKnockoutContext(groupScores);
    const engineRoundOf32 = new Map<string, { home: string; away: string }>();
    data.knockoutMatches.forEach((match) => {
      if (match.stage !== "Round of 32") {
        return;
      }
      const label = context.labels.get(String(match.id));
      if (!label) {
        return;
      }
      engineRoundOf32.set(String(match.id), label);
    });
    const uiRoundOf32 = new Map<string, { home: string; away: string }>();
    knockoutState.matches.forEach((match) => {
      if (match.stage !== "Round of 32") {
        return;
      }
      uiRoundOf32.set(String(match.id), {
        home: match.homeResolved ?? match.homeLabel,
        away: match.awayResolved ?? match.awayLabel,
      });
    });
    const mismatches: Array<{
      matchId: string;
      engine: { home: string; away: string };
      ui: { home: string; away: string } | null;
    }> = [];
    engineRoundOf32.forEach((engine, matchId) => {
      const ui = uiRoundOf32.get(matchId);
      if (!ui) {
        mismatches.push({ matchId, engine, ui: null });
        return;
      }
      if (engine.home !== ui.home || engine.away !== ui.away) {
        mismatches.push({ matchId, engine, ui });
      }
    });
    if (mismatches.length > 0) {
      console.warn(
        `[predictor] reset label mismatch ${JSON.stringify({ mismatches })}`
      );
    }
  }, [
    computeKnockoutContext,
    data.knockoutMatches,
    groupScores,
    knockoutState.matches,
  ]);

  const knockoutMatchesByStage = React.useMemo(
    () => matchesByStage(knockoutState.matches),
    [knockoutState.matches]
  );

  const stageOrder = [
    "Round of 32",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Final",
  ];

  const thirdPlaceMatches = knockoutMatchesByStage.get("Third place") ?? [];

  const roundOf32Order = React.useMemo(() => {
    const matchById = new Map<number, ResolvedKnockoutMatch>();
    for (const matches of knockoutMatchesByStage.values()) {
      for (const match of matches) {
        matchById.set(match.id, match);
      }
    }

    const extractSource = (label: string) => {
      if (label.startsWith("Winner Match ")) {
        return Number(label.replace("Winner Match ", "").trim());
      }
      if (label.startsWith("Loser Match ")) {
        return Number(label.replace("Loser Match ", "").trim());
      }
      return null;
    };

    const stageMatches = (stage: string) =>
      knockoutMatchesByStage.get(stage) ?? [];

    const orderFromParent = (
      parents: ResolvedKnockoutMatch[],
      childStage: string
    ) => {
      const children = stageMatches(childStage);
      const childIds = new Set(children.map((match) => match.id));
      const used = new Set<number>();
      const order: number[] = [];
      for (const parent of parents) {
        const sources = [
          extractSource(parent.homeLabel),
          extractSource(parent.awayLabel),
        ];
        for (const source of sources) {
          if (source && childIds.has(source) && !used.has(source)) {
            order.push(source);
            used.add(source);
          }
        }
      }
      const remaining = children
        .map((match) => match.id)
        .filter((id) => !used.has(id))
        .sort((a, b) => a - b);
      return [...order, ...remaining];
    };

    const finalMatches = [...stageMatches("Final")].sort((a, b) => a.id - b.id);
    const semifinalOrder = orderFromParent(finalMatches, "Semifinal");
    const semifinalMatches = semifinalOrder
      .map((id) => matchById.get(id))
      .filter(Boolean) as ResolvedKnockoutMatch[];
    const quarterfinalOrder = orderFromParent(semifinalMatches, "Quarterfinal");
    const quarterfinalMatches = quarterfinalOrder
      .map((id) => matchById.get(id))
      .filter(Boolean) as ResolvedKnockoutMatch[];
    const round16Order = orderFromParent(quarterfinalMatches, "Round of 16");
    const round16Matches = round16Order
      .map((id) => matchById.get(id))
      .filter(Boolean) as ResolvedKnockoutMatch[];
    const round32Order = orderFromParent(round16Matches, "Round of 32");

    if (round32Order.length === 0) {
      return stageMatches("Round of 32")
        .map((match) => match.id)
        .sort((a, b) => a - b);
    }
    return round32Order;
  }, [knockoutMatchesByStage]);

  const knockoutEdges = React.useMemo(() => {
    const edges: Array<{ from: number; to: number }> = [];
    for (const match of data.knockoutMatches) {
      const labels = [match.homeLabel, match.awayLabel];
      for (const label of labels) {
        if (label.startsWith("Winner Match ")) {
          const from = Number(label.replace("Winner Match ", "").trim());
          if (Number.isFinite(from)) {
            edges.push({ from, to: match.id });
          }
        } else if (label.startsWith("Loser Match ")) {
          const from = Number(label.replace("Loser Match ", "").trim());
          if (Number.isFinite(from)) {
            edges.push({ from, to: match.id });
          }
        }
      }
    }
    return edges;
  }, [data.knockoutMatches]);

  React.useEffect(() => {
    const container = knockoutContainerRef.current;
    if (!container) {
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const rect = container.getBoundingClientRect();
        const connectorInset = 8;
        const paths: string[] = [];
        for (const edge of knockoutEdges) {
          const fromEl = knockoutRefs.current.get(edge.from);
          const toEl = knockoutRefs.current.get(edge.to);
          if (!fromEl || !toEl) {
            continue;
          }
          const fromRect = fromEl.getBoundingClientRect();
          const toRect = toEl.getBoundingClientRect();
          const startX = fromRect.right - rect.left - connectorInset;
          const startY = fromRect.top - rect.top + fromRect.height / 2;
          const endX = toRect.left - rect.left + connectorInset;
          const endY = toRect.top - rect.top + toRect.height / 2;
          const midX = startX + (endX - startX) * 0.5;
          paths.push(
            `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`
          );
        }
        setKnockoutPaths(paths);
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(container);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [knockoutEdges, thirdPlaceOffset, finalCenterOverride, knockoutListHeight]);

  React.useLayoutEffect(() => {
    const list = roundOf32ListRef.current;
    const container = knockoutContainerRef.current;
    if (!list || !container) {
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const listRect = list.getBoundingClientRect();
        const centers = new Map<number, number>();
        const round32Matches = roundOf32Order
          .map((id) =>
            (knockoutMatchesByStage.get("Round of 32") ?? []).find(
              (match) => match.id === id
            )
          )
          .filter(Boolean) as ResolvedKnockoutMatch[];
        for (const match of round32Matches) {
          const el = knockoutRefs.current.get(match.id);
          if (!el) {
            continue;
          }
          const rect = el.getBoundingClientRect();
          centers.set(
            match.id,
            rect.top - listRect.top + rect.height / 2
          );
          setKnockoutCardHeight((prev) =>
            prev && Math.abs(prev - rect.height) < 0.5 ? prev : rect.height
          );
        }
        const computed = new Map(centers);
        const extractSource = (label: string) => {
          if (label.startsWith("Winner Match ")) {
            return Number(label.replace("Winner Match ", "").trim());
          }
          if (label.startsWith("Loser Match ")) {
            return Number(label.replace("Loser Match ", "").trim());
          }
          return null;
        };
        const stageSequence = [
          "Round of 16",
          "Quarterfinal",
          "Semifinal",
          "Final",
          "Third place",
        ];
        for (const stage of stageSequence) {
          const matches = knockoutMatchesByStage.get(stage) ?? [];
          for (const match of matches) {
            const sources = [
              extractSource(match.homeLabel),
              extractSource(match.awayLabel),
            ].filter((id): id is number => Boolean(id));
            const sourceCenters = sources
              .map((id) => computed.get(id))
              .filter((value): value is number => typeof value === "number");
            if (sourceCenters.length === 0) {
              continue;
            }
            const avg =
              sourceCenters.reduce((sum, value) => sum + value, 0) /
              sourceCenters.length;
            computed.set(match.id, avg);
          }
        }
        const height = list.scrollHeight;
        if (height > 0) {
          setKnockoutListHeight((prev) => (prev === height ? prev : height));
        }
        const nextCenters: Record<number, number> = {};
        for (const [id, center] of computed.entries()) {
          nextCenters[id] = center;
        }
        setKnockoutCenters(nextCenters);
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(list);
    observer.observe(container);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [knockoutMatchesByStage, roundOf32Order, knockoutCardHeight]);

  React.useLayoutEffect(() => {
    const finalMatch = (knockoutMatchesByStage.get("Final") ?? [])[0];
    const finalList = finalListRef.current;
    if (!finalMatch || !finalList) {
      setThirdPlaceOffset(null);
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const finalEl = knockoutRefs.current.get(finalMatch.id);
        if (!finalEl) {
          return;
        }
        const listRect = finalList.getBoundingClientRect();
        const finalRect = finalEl.getBoundingClientRect();
        const baseGap = 72;
        const nextTop = finalRect.bottom - listRect.top + baseGap;
        setThirdPlaceOffset((prev) => (prev === nextTop ? prev : nextTop));
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(finalList);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [knockoutMatchesByStage, knockoutCenters]);

  React.useLayoutEffect(() => {
    const container = knockoutContainerRef.current;
    const finalList = finalListRef.current;
    const semifinalMatches = knockoutMatchesByStage.get("Semifinal") ?? [];
    if (!container || !finalList || semifinalMatches.length === 0) {
      setFinalCenterOverride(null);
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const containerRect = container.getBoundingClientRect();
        const finalListRect = finalList.getBoundingClientRect();
        const centers = semifinalMatches
          .map((match) => {
            const el = knockoutRefs.current.get(match.id);
            if (!el) {
              return null;
            }
            const rect = el.getBoundingClientRect();
            return rect.top - containerRect.top + rect.height / 2;
          })
          .filter((value): value is number => typeof value === "number");
        if (centers.length === 0) {
          return;
        }
        const avg =
          centers.reduce((sum, value) => sum + value, 0) / centers.length;
        const listOffset = finalListRect.top - containerRect.top;
        const nextCenter = avg - listOffset;
        setFinalCenterOverride((prev) => (prev === nextCenter ? prev : nextCenter));
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(container);
    observer.observe(finalList);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [knockoutMatchesByStage, knockoutCenters]);

  const handleAutopredict = React.useCallback(() => {
    let nextQualifierWinners = { ...qualifierWinners };
    let nextAutoQualifierWinners = { ...autoQualifierWinners };
    let nextGroupScores = { ...groupScores };
    let nextAutoGroupScores = { ...autoGroupScores };
    let nextKnockoutWinners = { ...knockoutWinners };
    let nextAutoKnockoutWinners = { ...autoKnockoutWinners };

    const applyQualifierSelection = (matchId: string, selection: WinnerSelection) => {
      const prevSelection = nextQualifierWinners[matchId] ?? null;
      if (prevSelection === selection) {
        return { changed: false, clearedIds: [] as string[] };
      }
      const updated = { ...nextQualifierWinners, [matchId]: selection };
      const cleared = clearDependentSelections(updated, matchId, qualifierDependents);
      const clearedIds = Object.keys(updated).filter(
        (id) => updated[id] && cleared[id] === null
      );
      nextQualifierWinners = cleared;
      if (selection) {
        nextAutoQualifierWinners[matchId] = true;
      }
      clearedIds.forEach((id) => {
        delete nextAutoQualifierWinners[id];
      });
      return { changed: true, clearedIds };
    };

    const applyKnockoutSelection = (matchId: string, selection: WinnerSelection) => {
      const prevSelection = nextKnockoutWinners[matchId] ?? null;
      if (prevSelection === selection) {
        return { changed: false, clearedIds: [] as string[] };
      }
      const updated = { ...nextKnockoutWinners, [matchId]: selection };
      const cleared = clearDependentSelections(updated, matchId, knockoutDependents);
      const clearedIds = Object.keys(updated).filter(
        (id) => updated[id] && cleared[id] === null
      );
      nextKnockoutWinners = cleared;
      if (selection) {
        nextAutoKnockoutWinners[matchId] = true;
      }
      clearedIds.forEach((id) => {
        delete nextAutoKnockoutWinners[id];
      });
      return { changed: true, clearedIds };
    };

    const maxIterations = 10;
    let iteration = 0;
    let changed = true;

    while (changed && iteration < maxIterations) {
      iteration += 1;
      changed = false;

      const previousSlotWinners = resolveQualifierState(
        data.qualifiers,
        nextQualifierWinners
      ).slotWinners;
      const previousGroupScores = { ...nextGroupScores };

      let qualifierState = resolveQualifierState(
        data.qualifiers,
        nextQualifierWinners
      );
      const changedQualifiers = new Set<string>();
      let qualifierProgress = true;
      let qualifierIterations = 0;

      while (qualifierProgress && qualifierIterations < maxIterations) {
        qualifierProgress = false;
        qualifierIterations += 1;
        qualifierState = resolveQualifierState(
          data.qualifiers,
          nextQualifierWinners
        );
        qualifierState.matches.forEach((match) => {
          const key = String(match.id);
          const isManual =
            nextQualifierWinners[key] && !nextAutoQualifierWinners[key];
          const existingSelection = nextQualifierWinners[key] ?? null;
          if (isManual || existingSelection) {
            return;
          }
          if (
            isPlaceholderLabel(match.homeResolved) ||
            isPlaceholderLabel(match.awayResolved)
          ) {
            return;
          }
          const values = resolveMatchProbabilities({
            probabilities: data.winProbabilities,
            homeTeam: match.homeResolved,
            awayTeam: match.awayResolved,
            allowDraw: false,
            neutralOverride: match.neutral,
          });
          const selection = sampleWinner(values);
          if (!selection) {
            return;
          }
          const result = applyQualifierSelection(key, selection);
          if (result.changed) {
            changed = true;
            qualifierProgress = true;
            changedQualifiers.add(key);
          }
        });
      }

      if (changedQualifiers.size > 0) {
        const affectedSlots = new Set<string>();
        changedQualifiers.forEach((matchId) => {
          const slots = qualifierSlotsByMatch.get(matchId);
          if (!slots) {
            return;
          }
          slots.forEach((slot) => affectedSlots.add(slot));
        });
        const affectedGroups = new Set<string>();
        affectedSlots.forEach((slot) => {
          const groups = groupIdsBySlot.get(slot);
          if (groups) {
            groups.forEach((groupId) => affectedGroups.add(groupId));
          }
        });

        affectedSlots.forEach((slot) => {
          const matchIds = groupMatchIdsByTeam.get(slot);
          if (!matchIds) {
            return;
          }
          matchIds.forEach((matchId) => {
            delete nextGroupScores[matchId];
            delete nextAutoGroupScores[matchId];
          });
        });

        if (affectedGroups.size > 0) {
          const rootsToClear = new Set<string>();
          affectedGroups.forEach((groupId) => {
            const rootMatches = knockoutRootsByGroup.get(groupId);
            if (rootMatches) {
              rootMatches.forEach((matchId) => rootsToClear.add(matchId));
            }
          });
          if (rootsToClear.size > 0) {
            const cleared = clearKnockoutSelectionsByMatchIds(
              nextKnockoutWinners,
              rootsToClear
            );
            nextKnockoutWinners = cleared.next;
            cleared.clearedIds.forEach((matchId) => {
              delete nextAutoKnockoutWinners[matchId];
            });
            if (cleared.clearedIds.length > 0) {
              changed = true;
            }
          }
        }

        qualifierState = resolveQualifierState(data.qualifiers, nextQualifierWinners);
      }

      const nextSlotWinners = qualifierState.slotWinners;
      const resolvedGroupMatches = data.groupMatches.map((match) => ({
        ...match,
        homeTeam: nextSlotWinners.get(match.homeTeam) ?? match.homeTeam,
        awayTeam: nextSlotWinners.get(match.awayTeam) ?? match.awayTeam,
      }));

      let groupScoresChanged = false;
      resolvedGroupMatches.forEach((match) => {
        const key = String(match.id);
        const existing = nextGroupScores[key];
        const hasScore =
          existing && existing.home !== null && existing.away !== null;
        const isManual = hasScore && !nextAutoGroupScores[key];
        if (isManual || hasScore) {
          return;
        }
        const matrix = resolveMatchScoreMatrix({
          probabilities: data.winProbabilities,
          homeTeam: match.homeTeam,
          awayTeam: match.awayTeam,
          country: match.country,
        });
        if (!matrix) {
          return;
        }
        const sample = sampleScoreMatrix(matrix);
        if (!sample) {
          return;
        }
        nextGroupScores[key] = { home: sample.home, away: sample.away };
        nextAutoGroupScores[key] = true;
        groupScoresChanged = true;
      });

      if (groupScoresChanged || changedQualifiers.size > 0) {
        const clearedForGroups = computeClearedKnockoutSelections(
          nextKnockoutWinners,
          previousGroupScores,
          nextGroupScores,
          {
            previousSlotWinners,
            nextSlotWinners,
          }
        );
        nextKnockoutWinners = clearedForGroups.nextWinners;
        clearedForGroups.clearedIds.forEach((matchId) => {
          delete nextAutoKnockoutWinners[matchId];
        });
        if (clearedForGroups.clearedIds.length > 0) {
          changed = true;
        }
      }

      const nextContext = computeKnockoutContext(
        nextGroupScores,
        nextSlotWinners
      );
      const winners = new Map<number, string>();
      const losers = new Map<number, string>();
      const sorted = [...data.knockoutMatches].sort((a, b) => a.id - b.id);

      for (const match of sorted) {
        const key = String(match.id);
        const homeResolved = resolveKnockoutLabel({
          label: match.homeLabel,
          opponentLabel: match.awayLabel,
          groupRankings: nextContext.groupRankings,
          thirdPlaceByGroup: nextContext.thirdPlaceByGroup,
          thirdPlaceAssignments: nextContext.thirdPlaceAssignments,
          knockoutWinners: winners,
          knockoutLosers: losers,
          groupCompletion: nextContext.groupCompletion,
          allowThirdPlaceResolve: nextContext.allGroupMatchesComplete,
          qualifiedThirdGroups: new Set(nextContext.qualifiedThirdGroups),
          matchStageById,
        });
        const awayResolved = resolveKnockoutLabel({
          label: match.awayLabel,
          opponentLabel: match.homeLabel,
          groupRankings: nextContext.groupRankings,
          thirdPlaceByGroup: nextContext.thirdPlaceByGroup,
          thirdPlaceAssignments: nextContext.thirdPlaceAssignments,
          knockoutWinners: winners,
          knockoutLosers: losers,
          groupCompletion: nextContext.groupCompletion,
          allowThirdPlaceResolve: nextContext.allGroupMatchesComplete,
          qualifiedThirdGroups: new Set(nextContext.qualifiedThirdGroups),
          matchStageById,
        });
        const existingSelection = nextKnockoutWinners[key] ?? null;
        const isManual = existingSelection && !nextAutoKnockoutWinners[key];
        if (!isManual && !existingSelection) {
          if (
            !isPlaceholderLabel(homeResolved) &&
            !isPlaceholderLabel(awayResolved)
          ) {
            const values = resolveMatchProbabilities({
              probabilities: data.winProbabilities,
              homeTeam: homeResolved,
              awayTeam: awayResolved,
              allowDraw: false,
              country: match.country,
            });
            const selection = sampleWinner(values);
            if (selection) {
              const result = applyKnockoutSelection(key, selection);
              if (result.changed) {
                changed = true;
              }
            }
          }
        }

        const winner = resolveWinner(
          match.id,
          homeResolved,
          awayResolved,
          {},
          false,
          nextKnockoutWinners
        );
        if (winner) {
          winners.set(match.id, winner);
          const loser = winner === homeResolved ? awayResolved : homeResolved;
          losers.set(match.id, loser);
        }
      }
    }

    setQualifierWinners(nextQualifierWinners);
    setAutoQualifierWinners(nextAutoQualifierWinners);
    setGroupScores(nextGroupScores);
    setAutoGroupScores(nextAutoGroupScores);
    setKnockoutWinners(nextKnockoutWinners);
    setAutoKnockoutWinners(nextAutoKnockoutWinners);
  }, [
    autoGroupScores,
    autoKnockoutWinners,
    autoQualifierWinners,
    clearKnockoutSelectionsByMatchIds,
    computeClearedKnockoutSelections,
    computeKnockoutContext,
    data.groupMatches,
    data.knockoutMatches,
    data.qualifiers,
    data.winProbabilities,
    groupIdsBySlot,
    groupMatchIdsByTeam,
    groupScores,
    knockoutDependents,
    knockoutRootsByGroup,
    knockoutWinners,
    matchStageById,
    qualifierDependents,
    qualifierSlotsByMatch,
    qualifierWinners,
  ]);

  const handleResetAll = React.useCallback(() => {
    setQualifierWinners({});
    setGroupScores({});
    setKnockoutWinners({});
    setAutoQualifierWinners({});
    setAutoGroupScores({});
    setAutoKnockoutWinners({});
  }, []);

  const handleResetAutopredictions = React.useCallback(() => {
    const autoQualifierIds = Object.keys(autoQualifierWinners);
    let nextQualifierWinners = { ...qualifierWinners };
    autoQualifierIds.forEach((matchId) => {
      const updated = { ...nextQualifierWinners, [matchId]: null };
      nextQualifierWinners = clearDependentSelections(
        updated,
        matchId,
        qualifierDependents
      );
    });

    const affectedSlots = new Set<string>();
    autoQualifierIds.forEach((matchId) => {
      const slots = qualifierSlotsByMatch.get(matchId);
      if (!slots) {
        return;
      }
      slots.forEach((slot) => affectedSlots.add(slot));
    });
    const affectedGroups = new Set<string>();
    affectedSlots.forEach((slot) => {
      const groups = groupIdsBySlot.get(slot);
      if (groups) {
        groups.forEach((groupId) => affectedGroups.add(groupId));
      }
    });

    const nextGroupScores = { ...groupScores };
    Object.keys(autoGroupScores).forEach((matchId) => {
      delete nextGroupScores[matchId];
    });
    affectedSlots.forEach((slot) => {
      const matchIds = groupMatchIdsByTeam.get(slot);
      if (!matchIds) {
        return;
      }
      matchIds.forEach((matchId) => {
        delete nextGroupScores[matchId];
      });
    });

    let nextKnockoutWinners = { ...knockoutWinners };
    let nextAutoKnockoutWinners = { ...autoKnockoutWinners };
    const autoKnockoutIds = Object.keys(autoKnockoutWinners);
    if (autoKnockoutIds.length > 0) {
      const cleared = clearKnockoutSelectionsByMatchIds(
        nextKnockoutWinners,
        autoKnockoutIds
      );
      nextKnockoutWinners = cleared.next;
      nextAutoKnockoutWinners = {};
    }

    if (affectedGroups.size > 0) {
      const rootsToClear = new Set<string>();
      affectedGroups.forEach((groupId) => {
        const rootMatches = knockoutRootsByGroup.get(groupId);
        if (rootMatches) {
          rootMatches.forEach((matchId) => rootsToClear.add(matchId));
        }
      });
      if (rootsToClear.size > 0) {
        const cleared = clearKnockoutSelectionsByMatchIds(
          nextKnockoutWinners,
          rootsToClear
        );
        nextKnockoutWinners = cleared.next;
        cleared.clearedIds.forEach((matchId) => {
          delete nextAutoKnockoutWinners[matchId];
        });
      }
    }

    const nextSlotWinners = resolveQualifierState(
      data.qualifiers,
      nextQualifierWinners
    ).slotWinners;
    const clearedForGroups = computeClearedKnockoutSelections(
      nextKnockoutWinners,
      groupScores,
      nextGroupScores,
      {
        previousSlotWinners: slotWinners,
        nextSlotWinners,
      }
    );
    nextKnockoutWinners = clearedForGroups.nextWinners;
    clearedForGroups.clearedIds.forEach((matchId) => {
      delete nextAutoKnockoutWinners[matchId];
    });

    setQualifierWinners(nextQualifierWinners);
    setAutoQualifierWinners({});
    setGroupScores(nextGroupScores);
    setAutoGroupScores({});
    setKnockoutWinners(nextKnockoutWinners);
    setAutoKnockoutWinners(nextAutoKnockoutWinners);
  }, [
    autoGroupScores,
    autoKnockoutWinners,
    autoQualifierWinners,
    clearKnockoutSelectionsByMatchIds,
    computeClearedKnockoutSelections,
    data.qualifiers,
    groupIdsBySlot,
    groupMatchIdsByTeam,
    groupScores,
    knockoutRootsByGroup,
    knockoutWinners,
    qualifierDependents,
    qualifierSlotsByMatch,
    qualifierWinners,
    slotWinners,
  ]);

  return (
    <div className="flex flex-col gap-12">
      <div className="flex flex-wrap items-center justify-end gap-3">
        <button
          type="button"
          onClick={handleAutopredict}
          className="rounded-[3px] border border-ink-900 px-3 py-1 text-xs font-semibold uppercase text-ink-400 hover:text-ebony"
        >
          Autopredict
        </button>
        <button
          type="button"
          onClick={handleResetAutopredictions}
          className="rounded-[3px] border border-ink-900 px-3 py-1 text-xs font-semibold uppercase text-ink-400 hover:text-ebony"
        >
          Reset autopredictions
        </button>
        <button
          type="button"
          onClick={handleResetAll}
          className="rounded-[3px] border border-ink-900 px-3 py-1 text-xs font-semibold uppercase text-ink-400 hover:text-ebony"
        >
          Reset all
        </button>
      </div>
      <section className="space-y-6">
        <div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-2xl font-semibold text-ebony">
              Qualifier playoffs
            </h2>
          </div>
          <p className="text-sm text-ink-400">
            Resolve UEFA and intercontinental playoff paths to fill the final
            group stage slots.
          </p>
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          {Array.from(
            qualifierState.matches.reduce((map, match) => {
              if (!map.has(match.path)) {
                map.set(match.path, []);
              }
              map.get(match.path)?.push(match);
              return map;
            }, new Map<string, ResolvedQualifierMatch[]>())
          )
            .sort(([a], [b]) => {
              const order = [
                "IC Path 1",
                "IC Path 2",
                "UEFA Path A",
                "UEFA Path B",
                "UEFA Path C",
                "UEFA Path D",
              ];
              const indexA = order.indexOf(a);
              const indexB = order.indexOf(b);
              if (indexA !== -1 || indexB !== -1) {
                return (indexA === -1 ? 99 : indexA) - (indexB === -1 ? 99 : indexB);
              }
              return a.localeCompare(b);
            })
            .map(([path, matches]) => {
            return (
              <QualifierPathBracket
                key={path}
                path={path}
                matches={matches}
                winnerSelections={qualifierWinners}
                onWinnerSelect={updateQualifierWinner}
                flags={data.flags}
                getMatchProbabilityLabels={getMatchProbabilityLabels}
              />
            );
          })}
        </div>
      </section>

      <section className="space-y-6">
        <div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-2xl font-semibold text-ebony">Group stage</h2>
          </div>
          <p className="text-sm text-ink-400">
            Select group match outcomes and see standings update instantly.
          </p>
        </div>
        <div className="flex flex-col gap-6">
          {groupTables.map(({ group, rows }) => {
            const matches = groupMatchesFor(group.id, resolvedGroupMatches);
            return (
              <div
                key={group.id}
                className="rounded-md border border-ink-900 bg-white/80 p-4 shadow-soft"
              >
                <div className="mb-3 flex items-center justify-between text-sm font-semibold text-ebony">
                  <span>Group {group.id}</span>
                  <span className="text-xs text-ink-400">Group stage</span>
                </div>
                <div className="flex flex-wrap items-start gap-6">
                  <div className="flex min-w-[520px] flex-1 flex-col gap-3">
                    {matches.map((match) => {
                      const probabilities = getMatchProbabilityLabels({
                        homeTeam: match.homeTeam,
                        awayTeam: match.awayTeam,
                        allowDraw: true,
                        country: match.country,
                      });
                      return (
                        <MatchCard
                          key={match.id}
                          id={match.id}
                          homeTeam={match.homeTeam}
                          awayTeam={match.awayTeam}
                          scores={groupScores}
                          onScoreChange={updateGroupScore}
                          onScoreChangePair={updateGroupScorePair}
                          allowDraw
                          orientation="horizontal"
                          flags={data.flags}
                          homeWinProb={probabilities.homeWinProb}
                          awayWinProb={probabilities.awayWinProb}
                          drawProb={probabilities.drawProb}
                        />
                      );
                    })}
                  </div>
                  <div className="w-fit min-w-[380px] flex-none">
                    <GroupTable
                      group={group}
                      rows={rows}
                      highlightThird={allGroupMatchesComplete && qualifiedThirdGroups.has(group.id)}
                      highlightWeakThird={!allGroupMatchesComplete}
                      showTieInfo={groupCompletion[group.id]}
                      flags={data.flags}
                    />
                  </div>
                </div>
              </div>
            );
          })}
          {thirdPlaceRankingRows.length > 0 && (
            <div className="rounded-md border border-ink-900 bg-white/80 p-4 shadow-soft">
              <div className="mb-3 flex items-center justify-between text-sm font-semibold text-ebony">
                <span>Ranking of 3rd place teams</span>
                <span className="text-xs text-ink-400">Group stage</span>
              </div>
              <div className="flex flex-wrap items-start gap-6">
                <div className="w-fit min-w-[380px] flex-none">
                  <GroupTable
                    group={{ id: "Third place", teams: [] }}
                    rows={thirdPlaceRankingRows}
                    highlightThird={false}
                    highlightWeakThird={false}
                    highlightTop={8}
                    showTieInfo={allGroupMatchesComplete}
                    flags={data.flags}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="space-y-6">
        <div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-2xl font-semibold text-ebony">Knockout stage</h2>
          </div>
          <p className="text-sm text-ink-400">
            Winners advance automatically through the bracket.
          </p>
        </div>
        <div className="overflow-x-auto overflow-y-visible pb-2">
          <div
            ref={knockoutContainerRef}
            className="relative min-w-[900px]"
          >
            <svg
              className="absolute inset-0 z-0 h-full w-full pointer-events-none"
              aria-hidden="true"
            >
              {knockoutPaths.map((path, index) => (
                <path
                  key={`${path}-${index}`}
                  d={path}
                  fill="none"
                  stroke="var(--color-primary-dark)"
                  strokeWidth={2}
                />
              ))}
            </svg>
            <div className="relative z-10 flex gap-6">
              {stageOrder.map((stage) => {
                const matches = knockoutMatchesByStage.get(stage) ?? [];
                const isRoundOf32 = stage === "Round of 32";
                const orderedMatches = isRoundOf32
                  ? roundOf32Order
                      .map((id) => matches.find((m) => m.id === id))
                      .filter(Boolean)
                  : matches;
                const cardHeight = knockoutCardHeight ?? 64;
                const headerOffset = 20;
                const thirdPlaceMatchTop = stage === "Final" ? thirdPlaceOffset : null;
                const labelGap = 28;
                const finalStageHeight =
                  thirdPlaceMatchTop !== null
                    ? Math.max(
                        knockoutListHeight ?? 0,
                        thirdPlaceMatchTop + cardHeight
                      )
                    : knockoutListHeight ?? null;
                const columnHeight =
                  knockoutListHeight && stage === "Final" && finalStageHeight
                    ? finalStageHeight + headerOffset
                    : knockoutListHeight
                      ? knockoutListHeight + headerOffset
                      : undefined;
                return (
                  <div
                    key={stage}
                    className="relative min-w-[200px]"
                    style={columnHeight ? { height: columnHeight } : undefined}
                  >
                    <div className="flex justify-center pb-2 text-xs font-semibold uppercase tracking-wide text-ink-400">
                      <span className="w-[200px] text-center">
                        {stage === "Final" ? "Final / Third place" : stage}
                      </span>
                    </div>
                    <div
                      ref={(el) => {
                        if (isRoundOf32) {
                          roundOf32ListRef.current = el;
                        }
                        if (stage === "Final") {
                          finalListRef.current = el;
                        }
                      }}
                      className={cn(
                        "relative",
                        isRoundOf32 ? "flex flex-col gap-[9px] pt-0" : "pt-4"
                      )}
                      style={
                        !isRoundOf32 && knockoutListHeight
                          ? {
                              minHeight: `${
                                stage === "Final" && finalStageHeight
                                  ? finalStageHeight
                                  : knockoutListHeight
                              }px`,
                            }
                          : undefined
                      }
                    >
                      {orderedMatches.map((match) => {
                        if (!match) {
                          return null;
                        }
                        const handleRoundOf32Click = isRoundOf32
                          ? () => logRoundOf32Match(match)
                          : undefined;
                        const center =
                          stage === "Final" && finalCenterOverride !== null
                            ? finalCenterOverride
                            : knockoutCenters[match.id] ?? 0;
                        const top = isRoundOf32 ? undefined : center - cardHeight / 2;
                        const probabilities = getMatchProbabilityLabels({
                          homeTeam: match.homeResolved ?? match.homeLabel,
                          awayTeam: match.awayResolved ?? match.awayLabel,
                          allowDraw: false,
                          country: match.country,
                        });
                        return (
                          <div
                            key={match.id}
                            ref={(el) => {
                              if (el) {
                                knockoutRefs.current.set(match.id, el);
                              } else {
                                knockoutRefs.current.delete(match.id);
                              }
                            }}
                            className={cn(
                              isRoundOf32 && "relative",
                              !isRoundOf32 && "absolute left-0",
                              stage === "Final" && "relative"
                            )}
                            style={top !== undefined ? { top } : undefined}
                            onClick={handleRoundOf32Click}
                          >
                            {stage === "Final" && (
                              <div
                                className="absolute left-0 w-full text-center text-xs font-semibold uppercase tracking-wide text-ink-400"
                                style={{ top: -labelGap }}
                              >
                                Final
                              </div>
                            )}
                            <MatchCard
                              id={match.id}
                              homeTeam={match.homeResolved ?? match.homeLabel}
                              awayTeam={match.awayResolved ?? match.awayLabel}
                              showScore={false}
                              winnerSelection={knockoutWinners[String(match.id)] ?? null}
                              onWinnerSelect={(selection) =>
                                updateKnockoutWinner(match.id, selection)
                              }
                              allowDraw={false}
                              orientation="vertical"
                              stackMode="centered"
                              flags={data.flags}
                              homeWinProb={probabilities.homeWinProb}
                              awayWinProb={probabilities.awayWinProb}
                              drawProb={probabilities.drawProb}
                            />
                          </div>
                        );
                      })}
                      {stage === "Final" &&
                        thirdPlaceMatches.length > 0 &&
                        thirdPlaceMatchTop !== null && (
                          <>
                            <div
                              className="absolute left-0 w-full text-center text-xs font-semibold uppercase tracking-wide text-ink-400"
                              style={{ top: thirdPlaceMatchTop - labelGap }}
                            >
                              Third place
                            </div>
                            <div
                              className="absolute left-0"
                              style={{ top: thirdPlaceMatchTop }}
                            >
                              {thirdPlaceMatches.map((match) => {
                                const probabilities = getMatchProbabilityLabels({
                                  homeTeam: match.homeResolved ?? match.homeLabel,
                                  awayTeam: match.awayResolved ?? match.awayLabel,
                                  allowDraw: false,
                                  country: match.country,
                                });
                                return (
                                  <div
                                    key={match.id}
                                    ref={(el) => {
                                      if (el) {
                                        knockoutRefs.current.set(match.id, el);
                                      } else {
                                        knockoutRefs.current.delete(match.id);
                                      }
                                    }}
                                  >
                                    <MatchCard
                                      id={match.id}
                                      homeTeam={match.homeResolved ?? match.homeLabel}
                                      awayTeam={match.awayResolved ?? match.awayLabel}
                                      showScore={false}
                                      winnerSelection={knockoutWinners[String(match.id)] ?? null}
                                      onWinnerSelect={(selection) =>
                                        updateKnockoutWinner(match.id, selection)
                                      }
                                      allowDraw={false}
                                      orientation="vertical"
                                      stackMode="centered"
                                      flags={data.flags}
                                      homeWinProb={probabilities.homeWinProb}
                                      awayWinProb={probabilities.awayWinProb}
                                      drawProb={probabilities.drawProb}
                                    />
                                  </div>
                                );
                              })}
                            </div>
                          </>
                        )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      <div className="flex flex-wrap items-center justify-end gap-3">
        <button
          type="button"
          onClick={handleAutopredict}
          className="rounded-[3px] border border-ink-900 px-3 py-1 text-xs font-semibold uppercase text-ink-400 hover:text-ebony"
        >
          Autopredict
        </button>
        <button
          type="button"
          onClick={handleResetAutopredictions}
          className="rounded-[3px] border border-ink-900 px-3 py-1 text-xs font-semibold uppercase text-ink-400 hover:text-ebony"
        >
          Reset autopredictions
        </button>
        <button
          type="button"
          onClick={handleResetAll}
          className="rounded-[3px] border border-ink-900 px-3 py-1 text-xs font-semibold uppercase text-ink-400 hover:text-ebony"
        >
          Reset all
        </button>
      </div>

      <div className="text-xs text-ink-400">
        Tie-breakers follow the tournament model (points, goal difference, goals
        for, head-to-head). Exact ties are randomized here; FIFA would use Fair
        Play Points.
      </div>
    </div>
  );
}
