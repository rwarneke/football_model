import { readFile } from "node:fs/promises";
import path from "node:path";
import { loadCompletedWorldCupMatches } from "@/lib/world-cup-results";

export type WorldCupMatch = {
  id: string;
  date: string;
  kickoffUtc?: string | null;
  stage: string;
  home: string;
  away: string;
  stadium: string;
  city: string;
  country: string;
  group?: string | null;
  neutral?: boolean | null;
};

const PUBLIC_DIR = path.join(process.cwd(), "public");
const REFERENCE_DIR = "/reference_data";
const GROUPS_FILE = `${REFERENCE_DIR}/world_cup_2026_groups.csv`;
const GROUP_MATCHES_FILE = `${REFERENCE_DIR}/world_cup_2026_group_matches.csv`;
const KNOCKOUT_MATCHES_FILE = `${REFERENCE_DIR}/world_cup_2026_knockout_matches.csv`;
const QUALIFIERS_FILE = `${REFERENCE_DIR}/world_cup_2026_remaining_qualifiers.csv`;
const KICKOFF_UTC_FILE = `${REFERENCE_DIR}/world_cup_2026_match_kickoff_utc.json`;
const RESULTS_ORDER_FILE = "/model_output/results_wc2026.csv";

type GroupDefinition = {
  id: string;
  teams: string[];
};

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

async function readPublicText(filePath: string) {
  const normalized = filePath.replace(/^\/+/, "");
  const fullPath = path.join(PUBLIC_DIR, normalized);
  return readFile(fullPath, "utf8");
}

async function readPublicJson<T>(filePath: string, fallback: T): Promise<T> {
  try {
    const contents = await readPublicText(filePath);
    return JSON.parse(contents) as T;
  } catch {
    return fallback;
  }
}

async function loadWorldCupOrderMap() {
  try {
    const contents = await readPublicText(RESULTS_ORDER_FILE);
    const rows = parseCsv(contents).rows;
    return new Map(
      rows
        .map((row, index) => {
          const matchId = row.match_id?.trim() ?? "";
          return matchId ? [matchId, index] : null;
        })
        .filter((entry): entry is [string, number] => Boolean(entry))
    );
  } catch {
    return null;
  }
}

function parseCsv(contents: string) {
  const trimmed = contents.trim();
  if (!trimmed) {
    return { headers: [] as string[], rows: [] as Record<string, string>[] };
  }
  const lines = trimmed.split(/\r?\n/);
  const headers = lines[0]?.split(",") ?? [];
  const rows = lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""])
    ) as Record<string, string>;
  });
  return { headers, rows };
}

function normalizeDate(value: string) {
  return value?.trim();
}

function isPlaceholderLabel(name: string) {
  const trimmed = name.trim();
  if (!trimmed) {
    return true;
  }
  return (
    /^Winner Match /i.test(trimmed) ||
    /^Loser Match /i.test(trimmed) ||
    /^Winner Group /i.test(trimmed) ||
    /^Runner-up Group /i.test(trimmed) ||
    /^3rd Group /i.test(trimmed) ||
    /^TBD$/i.test(trimmed) ||
    /^UEFA Path /i.test(trimmed) ||
    /^IC Path /i.test(trimmed) ||
    /winner$/i.test(trimmed)
  );
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
  matches: WorldCupMatch[],
  completedById: Map<string, { homeScore: number; awayScore: number }>
) {
  const parts = [group.id];
  const orderedMatches = [...matches].sort((a, b) => Number(a.id) - Number(b.id));
  for (const match of orderedMatches) {
    const score = completedById.get(String(match.id));
    const home = score?.homeScore ?? "x";
    const away = score?.awayScore ?? "x";
    parts.push(`${match.id}:${home}-${away}`);
  }
  return hashString(parts.join("|"));
}

function seedFromThirdPlace(
  entries: Array<{ team: string; group: string; points: number; gd: number; gf: number }>
) {
  const parts = entries.map(
    (entry) => `${entry.group}:${entry.team}:${entry.points}:${entry.gd}:${entry.gf}`
  );
  return hashString(parts.join("|"));
}

function rankOverall(
  teams: string[],
  table: Record<string, GroupTableRow>,
  rng: () => number
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
    }
    ordered.push(...tied);
  }
  return ordered;
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
    table[team].gd = table[team].gf - table[team].ga;
  }
  return table;
}

function rankHeadToHead(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>,
  table: Record<string, GroupTableRow>,
  rng: () => number
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
    return rankOverall(teams, table, rng);
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
      ordered.push(...rankHeadToHead(tied, matches, table, rng));
    }
  }
  return ordered;
}

function rankGroup(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>,
  table: Record<string, GroupTableRow>,
  rng: () => number
) {
  const base = [...teams].sort((a, b) => table[b].points - table[a].points);
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
      ranked.push(...rankHeadToHead(tied, matches, table, rng));
    }
  }
  return ranked;
}

function buildGroupTable(
  group: GroupDefinition,
  matches: WorldCupMatch[],
  completedById: Map<string, { homeScore: number; awayScore: number }>
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
    const score = completedById.get(String(match.id));
    if (!score) {
      continue;
    }
    const home = table[match.home];
    const away = table[match.away];
    if (!home || !away) {
      continue;
    }
    const { homeScore, awayScore } = score;
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
      homeTeam: match.home,
      awayTeam: match.away,
      homeScore,
      awayScore,
    });
  }
  for (const team of Object.keys(table)) {
    table[team].gd = table[team].gf - table[team].ga;
  }
  const rng = createRng(seedFromGroupState(group, matches, completedById));
  const ranking = rankGroup(group.teams, playedMatches, table, rng);
  ranking.forEach((team, index) => {
    table[team].position = index + 1;
  });
  return { table, ranking };
}

function bestThirdPlace(
  groupTables: Array<{ ranking: string[]; table: Record<string, GroupTableRow> }>
) {
  const entries: Array<{ team: string; group: string; points: number; gd: number; gf: number }> = [];
  for (const { ranking, table } of groupTables) {
    if (ranking.length < 3) {
      continue;
    }
    const team = ranking[2];
    const row = table[team];
    entries.push({ team, group: row.group, points: row.points, gd: row.gd, gf: row.gf });
  }
  const rng = createRng(seedFromThirdPlace(entries));
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
    }
    ordered.push(...tied);
  }
  return ordered;
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
    return groupCompletion[group] ? groupRankings[group]?.[0] ?? label : label;
  }
  if (label.startsWith("Runner-up Group ")) {
    const group = label.replace("Runner-up Group ", "").trim();
    return groupCompletion[group] ? groupRankings[group]?.[1] ?? label : label;
  }
  if (label.startsWith("3rd Group ")) {
    const group = label.replace("3rd Group ", "").trim();
    if (group.length === 1 && allowThirdPlaceResolve && (!qualifiedThirdGroups || qualifiedThirdGroups.has(group))) {
      return thirdPlaceByGroup[group] ?? label;
    }
  }
  return label;
}

function resolveKnockoutLabel(
  label: string,
  opponentLabel: string,
  stage: string,
  groupRankings: Record<string, string[]>,
  thirdPlaceByGroup: Record<string, string>,
  thirdPlaceAssignments: Record<string, string> | null,
  knockoutWinners: Map<string, string>,
  knockoutLosers: Map<string, string>,
  groupCompletion: Record<string, boolean>,
  allowThirdPlaceResolve: boolean,
  qualifiedThirdGroups: Set<string>,
  teamGroups: Record<string, string>
) {
  if (label.startsWith("Winner Match ")) {
    const priorId = label.replace("Winner Match ", "").trim();
    return knockoutWinners.get(priorId) ?? label;
  }
  if (label.startsWith("Loser Match ")) {
    const priorId = label.replace("Loser Match ", "").trim();
    return knockoutLosers.get(priorId) ?? label;
  }
  if (
    allowThirdPlaceResolve &&
    label.startsWith("3rd Group ") &&
    opponentLabel.startsWith("Winner Group ")
  ) {
    const winnerGroup = opponentLabel.replace("Winner Group ", "").trim();
    const assignedGroup = thirdPlaceAssignments?.[`1${winnerGroup}`];
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

function qualifierTeamLabel(
  team: string,
  source: string,
  path: string
): string {
  const trimmed = team?.trim();
  if (trimmed) {
    return trimmed;
  }
  const sourceTrimmed = source?.trim();
  if (sourceTrimmed) {
    return `Winner ${sourceTrimmed.toUpperCase()}`;
  }
  return path?.trim() ? `${path} TBD` : "TBD";
}

function formatQualifierStage(stage: string, path: string) {
  const trimmedStage = stage.trim();
  const trimmedPath = path.trim();
  if (!trimmedStage) {
    return trimmedPath;
  }
  if (!trimmedPath) {
    return trimmedStage;
  }
  if (trimmedStage.toLowerCase().includes("uefa") && trimmedPath.startsWith("UEFA ")) {
    return `${trimmedStage} ${trimmedPath.replace(/^UEFA\s+/, "")}`;
  }
  if (trimmedStage.toLowerCase().includes("inter-confederation") && trimmedPath.startsWith("IC ")) {
    return `IC Playoff ${trimmedPath.replace(/^IC\s+/, "")}`;
  }
  return `${trimmedStage} ${trimmedPath}`;
}

export async function loadWorldCupMatches(): Promise<WorldCupMatch[]> {
  const [groupDefinitionsContents, groupContents, knockoutContents, qualifierContents, worldCupOrderMap, completedMatches, kickoffUtcById] = await Promise.all([
    readPublicText(GROUPS_FILE),
    readPublicText(GROUP_MATCHES_FILE),
    readPublicText(KNOCKOUT_MATCHES_FILE),
    readPublicText(QUALIFIERS_FILE),
    loadWorldCupOrderMap(),
    loadCompletedWorldCupMatches(),
    readPublicJson<Record<string, string>>(KICKOFF_UTC_FILE, {}),
  ]);

  const groupDefinitionsRows = parseCsv(groupDefinitionsContents).rows;
  const groupRows = parseCsv(groupContents).rows;
  const knockoutRows = parseCsv(knockoutContents).rows;
  const qualifierRows = parseCsv(qualifierContents).rows;
  const completedById = new Map(
    completedMatches.map((match) => [
      String(match.matchId),
      { homeScore: match.homeScore, awayScore: match.awayScore, winner: match.winner },
    ])
  );

  const groupsMap = new Map<string, string[]>();
  for (const row of groupDefinitionsRows) {
    const group = row.group?.trim();
    const team = row.team?.trim();
    if (!group || !team) continue;
    if (!groupsMap.has(group)) groupsMap.set(group, []);
    groupsMap.get(group)?.push(team);
  }
  const groups: GroupDefinition[] = Array.from(groupsMap.keys())
    .sort()
    .map((id) => ({ id, teams: groupsMap.get(id) ?? [] }));
  const teamGroups: Record<string, string> = {};
  groups.forEach((group) => group.teams.forEach((team) => { teamGroups[team] = group.id; }));

  const groupMatches: WorldCupMatch[] = groupRows.map((row) => ({
    id: row.match_id ?? "",
    date: normalizeDate(row.date ?? ""),
    kickoffUtc: kickoffUtcById[row.match_id ?? ""] ?? null,
    stage: `Group ${row.group ?? ""}`.trim(),
    home: row.home_team ?? "",
    away: row.away_team ?? "",
    stadium: row.stadium ?? "",
    city: row.city ?? "",
    country: row.country ?? "",
    group: row.group ?? null,
    neutral: null,
  }));

  const qualifierMatches: WorldCupMatch[] = qualifierRows.map((row, index) => ({
    id: `Q-${index + 1}`,
    date: normalizeDate(row.date ?? ""),
    kickoffUtc: null,
    stage: formatQualifierStage(row.stage ?? "", row.path ?? ""),
    home: qualifierTeamLabel(row.home_team ?? "", row.home_source ?? "", row.path ?? ""),
    away: qualifierTeamLabel(row.away_team ?? "", row.away_source ?? "", row.path ?? ""),
    stadium: row.stadium ?? "",
    city: row.city ?? "",
    country: row.country ?? "",
    group: null,
    neutral:
      row.neutral?.trim().toLowerCase() === "true"
        ? true
        : row.neutral?.trim().toLowerCase() === "false"
        ? false
        : null,
  }));

  const groupTables = groups.map((group) => {
    const matches = groupMatches.filter((match) => match.group === group.id);
    const { table, ranking } = buildGroupTable(group, matches, completedById);
    return { group, table, ranking, matches };
  });
  const groupRankings: Record<string, string[]> = {};
  const groupCompletion: Record<string, boolean> = {};
  groupTables.forEach((entry) => {
    groupRankings[entry.group.id] = entry.ranking;
    groupCompletion[entry.group.id] = entry.matches.every((match) => completedById.has(String(match.id)));
  });
  const thirdPlaceEntries = bestThirdPlace(groupTables);
  const thirdPlaceByGroup: Record<string, string> = {};
  thirdPlaceEntries.forEach((entry) => {
    if (!thirdPlaceByGroup[entry.group]) {
      thirdPlaceByGroup[entry.group] = entry.team;
    }
  });
  const bestThirdGroups = thirdPlaceEntries.slice(0, 8);
  const qualifiedThirdGroups = new Set(bestThirdGroups.map((entry) => entry.group));
  const comboKey = bestThirdGroups.map((entry) => entry.group).sort().join("");
  let thirdPlaceAssignments: Record<string, string> | null = null;
  if (comboKey) {
    const combosRows = parseCsv(await readPublicText("/reference_data/world_cup_2026_round_of_32_combinations.csv")).rows;
    const comboRow = combosRows.find((row) => (row.combo?.trim() ?? "") === comboKey);
    if (comboRow) {
      thirdPlaceAssignments = {
        "1A": comboRow["1A"]?.trim() ?? "",
        "1B": comboRow["1B"]?.trim() ?? "",
        "1D": comboRow["1D"]?.trim() ?? "",
        "1E": comboRow["1E"]?.trim() ?? "",
        "1G": comboRow["1G"]?.trim() ?? "",
        "1I": comboRow["1I"]?.trim() ?? "",
        "1K": comboRow["1K"]?.trim() ?? "",
        "1L": comboRow["1L"]?.trim() ?? "",
      };
    }
  }
  const allGroupMatchesComplete = groupMatches.every((match) => completedById.has(String(match.id)));
  const knockoutWinners = new Map<string, string>();
  const knockoutLosers = new Map<string, string>();
  const resolvedKnockoutMatches: WorldCupMatch[] = knockoutRows
    .map((row) => ({
      id: row.match_id ?? "",
      date: normalizeDate(row.date ?? ""),
      kickoffUtc: kickoffUtcById[row.match_id ?? ""] ?? null,
      stage: row.stage ?? "",
      home: row.home ?? "",
      away: row.away ?? "",
      stadium: row.stadium ?? "",
      city: row.city ?? "",
      country: row.country ?? "",
      group: null,
      neutral: null,
    }))
    .sort((a, b) => Number(a.id) - Number(b.id))
    .map((match) => {
      const home = resolveKnockoutLabel(
        match.home,
        match.away,
        match.stage,
        groupRankings,
        thirdPlaceByGroup,
        thirdPlaceAssignments,
        knockoutWinners,
        knockoutLosers,
        groupCompletion,
        allGroupMatchesComplete,
        qualifiedThirdGroups,
        teamGroups
      );
      const away = resolveKnockoutLabel(
        match.away,
        match.home,
        match.stage,
        groupRankings,
        thirdPlaceByGroup,
        thirdPlaceAssignments,
        knockoutWinners,
        knockoutLosers,
        groupCompletion,
        allGroupMatchesComplete,
        qualifiedThirdGroups,
        teamGroups
      );
      const completed = completedById.get(match.id);
      const winner = completed?.winner ?? null;
      if (winner && !isPlaceholderLabel(home) && !isPlaceholderLabel(away)) {
        knockoutWinners.set(match.id, winner);
        knockoutLosers.set(match.id, winner === home ? away : home);
      }
      return { ...match, home, away };
    });

  return [...qualifierMatches, ...groupMatches, ...resolvedKnockoutMatches]
    .filter((match) => match.date)
    .sort((a, b) => {
      const aOrder = worldCupOrderMap?.get(a.id);
      const bOrder = worldCupOrderMap?.get(b.id);
      if (aOrder !== undefined && bOrder !== undefined) {
        return aOrder - bOrder;
      }
      if (aOrder !== undefined) {
        return 1;
      }
      if (bOrder !== undefined) {
        return -1;
      }
      const dateCompare = a.date.localeCompare(b.date);
      if (dateCompare !== 0) {
        return dateCompare;
      }
      const kickoffCompare = (a.kickoffUtc ?? "").localeCompare(b.kickoffUtc ?? "");
      if (kickoffCompare !== 0) {
        return kickoffCompare;
      }
      return a.id.localeCompare(b.id);
    });
}
