import fs from "fs";
import path from "path";

export type GroupDefinition = {
  id: string;
  teams: string[];
};

export type GroupMatch = {
  id: number;
  date: string;
  group: string;
  homeTeam: string;
  awayTeam: string;
  stadium: string;
  city: string;
  country: string;
};

export type KnockoutMatch = {
  id: number;
  stage: string;
  date: string;
  homeLabel: string;
  awayLabel: string;
  stadium: string;
  city: string;
  country: string;
};

export type QualifierMatch = {
  id: string;
  date: string;
  stage: string;
  path: string;
  round: string;
  homeTeam: string;
  awayTeam: string;
  homeSource: string;
  awaySource: string;
  winnerSlot: string;
  neutral: boolean;
};

export type RoundOf32Combos = Record<string, Record<string, string>>;

export type WinProbabilityEntry = {
  p_home?: number;
  p_draw?: number;
  p_away?: number;
  p_home_pens?: number;
  p_away_pens?: number;
};

export type WinProbabilities = Record<
  string,
  Record<string, { home?: WinProbabilityEntry; neutral?: WinProbabilityEntry }>
>;

export type WorldCupPredictorData = {
  groups: GroupDefinition[];
  groupMatches: GroupMatch[];
  knockoutMatches: KnockoutMatch[];
  roundOf32Combos: RoundOf32Combos;
  qualifiers: QualifierMatch[];
  flags: Record<string, string | null>;
  winProbabilities: WinProbabilities;
};

const REFERENCE_DIR = path.resolve(process.cwd(), "..", "reference_data");
const GROUPS_FILE = path.join(REFERENCE_DIR, "world_cup_2026_groups.csv");
const GROUP_MATCHES_FILE = path.join(
  REFERENCE_DIR,
  "world_cup_2026_group_matches.csv"
);
const KNOCKOUT_MATCHES_FILE = path.join(
  REFERENCE_DIR,
  "world_cup_2026_knockout_matches.csv"
);
const ROUND_OF_32_FILE = path.join(
  REFERENCE_DIR,
  "world_cup_2026_round_of_32_combinations.csv"
);
const WIN_PROBABILITIES_FILE = path.join(
  REFERENCE_DIR,
  "..",
  "model_output",
  "win_probabilities.json"
);
const QUALIFIERS_FILE = path.join(
  REFERENCE_DIR,
  "world_cup_2026_remaining_qualifiers.csv"
);
const NAME_MAP_FILES = [
  path.join(REFERENCE_DIR, "fifa_member_to_canonical_name_map.csv"),
  path.join(REFERENCE_DIR, "kaggle_team_to_canonical_name_map.csv"),
];
const FLAGS_DIR = path.resolve(process.cwd(), "public", "flags");

function readCsv(filePath: string) {
  const contents = fs.readFileSync(filePath, "utf8").trim();
  const lines = contents ? contents.split(/\r?\n/) : [];
  if (lines.length === 0) {
    return { headers: [] as string[], rows: [] as Record<string, string>[] };
  }
  const headers = lines[0]?.split(",") ?? [];
  const rows = lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""])
    ) as Record<string, string>;
  });
  return { headers, rows };
}

function buildNameMap() {
  const map = new Map<string, string>();
  for (const filePath of NAME_MAP_FILES) {
    const { rows } = readCsv(filePath);
    for (const row of rows) {
      const original = row.original_name?.trim();
      const replacement = row.replacement_name?.trim();
      if (original && replacement) {
        map.set(original, replacement);
      }
    }
  }
  return map;
}

function isSlotPlaceholder(name: string) {
  return /\bwinner$/i.test(name.trim());
}

function normalizeName(
  raw: string,
  nameMap: Map<string, string>
): string {
  const trimmed = raw?.trim() ?? "";
  if (!trimmed || trimmed.toLowerCase() === "nan") {
    return "";
  }
  if (isSlotPlaceholder(trimmed)) {
    return trimmed;
  }
  return nameMap.get(trimmed) ?? trimmed;
}

function resolveFlagPath(team: string) {
  if (!team || isSlotPlaceholder(team)) {
    return null;
  }
  const fileName = `${team.replace(/ /g, "_")}.png`;
  const filePath = path.join(FLAGS_DIR, fileName);
  if (fs.existsSync(filePath)) {
    return `/flags/${fileName}`;
  }
  return null;
}

export function loadWorldCupPredictorData(): WorldCupPredictorData {
  const nameMap = buildNameMap();

  const groupRows = readCsv(GROUPS_FILE).rows;
  const groupsMap = new Map<string, string[]>();
  for (const row of groupRows) {
    const group = row.group?.trim();
    if (!group) {
      continue;
    }
    const team = normalizeName(row.team ?? "", nameMap);
    if (!groupsMap.has(group)) {
      groupsMap.set(group, []);
    }
    if (team) {
      groupsMap.get(group)?.push(team);
    }
  }
  const groups = Array.from(groupsMap.keys())
    .sort()
    .map((group) => ({
      id: group,
      teams: groupsMap.get(group) ?? [],
    }));

  const groupMatches = readCsv(GROUP_MATCHES_FILE).rows.map((row) => ({
    id: Number(row.match_id),
    date: row.date,
    group: row.group?.trim() ?? "",
    homeTeam: normalizeName(row.home_team ?? "", nameMap),
    awayTeam: normalizeName(row.away_team ?? "", nameMap),
    stadium: row.stadium?.trim() ?? "",
    city: row.city?.trim() ?? "",
    country: row.country?.trim() ?? "",
  }));

  const knockoutMatches = readCsv(KNOCKOUT_MATCHES_FILE).rows.map((row) => ({
    id: Number(row.match_id),
    stage: row.stage?.trim() ?? "",
    date: row.date,
    homeLabel: row.home?.trim() ?? "",
    awayLabel: row.away?.trim() ?? "",
    stadium: row.stadium?.trim() ?? "",
    city: row.city?.trim() ?? "",
    country: row.country?.trim() ?? "",
  }));

  const combosRows = readCsv(ROUND_OF_32_FILE).rows;
  const roundOf32Combos: RoundOf32Combos = {};
  for (const row of combosRows) {
    const combo = row.combo?.trim();
    if (!combo) {
      continue;
    }
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

  const qualifierRows = readCsv(QUALIFIERS_FILE).rows;
  const qualifiers: QualifierMatch[] = qualifierRows.map((row) => ({
    id: `${row.path?.trim() ?? "path"}-${row.round?.trim() ?? "round"}`,
    date: row.date,
    stage: row.stage?.trim() ?? "",
    path: row.path?.trim() ?? "",
    round: row.round?.trim() ?? "",
    homeTeam: normalizeName(row.home_team ?? "", nameMap),
    awayTeam: normalizeName(row.away_team ?? "", nameMap),
    homeSource: row.home_source?.trim() ?? "",
    awaySource: row.away_source?.trim() ?? "",
    winnerSlot: row.winner_slot?.trim() ?? "",
    neutral: String(row.neutral ?? "").trim().toLowerCase() === "true",
  }));

  const allTeams = new Set<string>();
  for (const group of groups) {
    for (const team of group.teams) {
      if (team && !isSlotPlaceholder(team)) {
        allTeams.add(team);
      }
    }
  }
  for (const match of qualifiers) {
    if (match.homeTeam && !isSlotPlaceholder(match.homeTeam)) {
      allTeams.add(match.homeTeam);
    }
    if (match.awayTeam && !isSlotPlaceholder(match.awayTeam)) {
      allTeams.add(match.awayTeam);
    }
  }

  const flags: Record<string, string | null> = {};
  for (const team of allTeams) {
    flags[team] = resolveFlagPath(team);
  }

  const winProbabilities: WinProbabilities = fs.existsSync(WIN_PROBABILITIES_FILE)
    ? JSON.parse(fs.readFileSync(WIN_PROBABILITIES_FILE, "utf8"))
    : {};

  return {
    groups,
    groupMatches,
    knockoutMatches,
    roundOf32Combos,
    qualifiers,
    flags,
    winProbabilities,
  };
}
