import { readFile } from "node:fs/promises";
import path from "node:path";

export type WorldCupMatch = {
  id: string;
  date: string;
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
const GROUP_MATCHES_FILE = `${REFERENCE_DIR}/world_cup_2026_group_matches.csv`;
const KNOCKOUT_MATCHES_FILE = `${REFERENCE_DIR}/world_cup_2026_knockout_matches.csv`;
const QUALIFIERS_FILE = `${REFERENCE_DIR}/world_cup_2026_remaining_qualifiers.csv`;

async function readPublicText(filePath: string) {
  const normalized = filePath.replace(/^\/+/, "");
  const fullPath = path.join(PUBLIC_DIR, normalized);
  return readFile(fullPath, "utf8");
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
  const [groupContents, knockoutContents, qualifierContents] = await Promise.all([
    readPublicText(GROUP_MATCHES_FILE),
    readPublicText(KNOCKOUT_MATCHES_FILE),
    readPublicText(QUALIFIERS_FILE),
  ]);

  const groupRows = parseCsv(groupContents).rows;
  const knockoutRows = parseCsv(knockoutContents).rows;
  const qualifierRows = parseCsv(qualifierContents).rows;

  const groupMatches: WorldCupMatch[] = groupRows.map((row) => ({
    id: row.match_id ?? "",
    date: normalizeDate(row.date ?? ""),
    stage: `Group ${row.group ?? ""}`.trim(),
    home: row.home_team ?? "",
    away: row.away_team ?? "",
    stadium: row.stadium ?? "",
    city: row.city ?? "",
    country: row.country ?? "",
    group: row.group ?? null,
    neutral: null,
  }));

  const knockoutMatches: WorldCupMatch[] = knockoutRows.map((row) => ({
    id: row.match_id ?? "",
    date: normalizeDate(row.date ?? ""),
    stage: row.stage ?? "",
    home: row.home ?? "",
    away: row.away ?? "",
    stadium: row.stadium ?? "",
    city: row.city ?? "",
    country: row.country ?? "",
    group: null,
    neutral: null,
  }));

  const qualifierMatches: WorldCupMatch[] = qualifierRows.map((row, index) => ({
    id: `Q-${index + 1}`,
    date: normalizeDate(row.date ?? ""),
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

  return [...qualifierMatches, ...groupMatches, ...knockoutMatches]
    .filter((match) => match.date)
    .sort((a, b) => a.date.localeCompare(b.date));
}
