import { readFile } from "node:fs/promises";
import path from "node:path";

export type CompletedWorldCupMatch = {
  matchId: number;
  date: string;
  stage: string;
  group: string | null;
  homeTeam: string;
  awayTeam: string;
  stadium: string;
  city: string;
  country: string;
  neutral: boolean | null;
  homeScore: number;
  awayScore: number;
  wentExtraTime: boolean;
  wentPenalties: boolean;
  penaltyWinner: string | null;
  winner: string | null;
};

const PUBLIC_DIR = path.join(process.cwd(), "public");

function isErrnoException(error: unknown): error is NodeJS.ErrnoException {
  return typeof error === "object" && error !== null && "code" in error;
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

function normalizeNullableString(value: string | undefined) {
  const trimmed = value?.trim() ?? "";
  if (!trimmed || trimmed.toLowerCase() === "na" || trimmed.toLowerCase() === "nan") {
    return null;
  }
  return trimmed;
}

function parseNullableInt(value: string | undefined) {
  const normalized = normalizeNullableString(value);
  if (!normalized) {
    return null;
  }
  const parsed = Number(normalized);
  return Number.isInteger(parsed) ? parsed : null;
}

function parseNullableBool(value: string | undefined) {
  const normalized = normalizeNullableString(value)?.toLowerCase();
  if (!normalized) {
    return null;
  }
  if (["true", "t", "1", "yes"].includes(normalized)) {
    return true;
  }
  if (["false", "f", "0", "no"].includes(normalized)) {
    return false;
  }
  return null;
}

async function readPublicText(filePath: string) {
  const normalized = filePath.replace(/^\/+/, "");
  const fullPath = path.join(PUBLIC_DIR, normalized);
  return readFile(fullPath, "utf8");
}

export async function loadCompletedWorldCupMatches(
  modelOutputDir = "/model_output"
): Promise<CompletedWorldCupMatch[]> {
  let contents: string;
  try {
    contents = await readPublicText(`${modelOutputDir}/results_wc2026.csv`);
  } catch (error) {
    if (isErrnoException(error) && error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
  const rows = parseCsv(contents).rows;
  return rows
    .map((row) => {
      const matchId = parseNullableInt(row.match_id);
      const homeScore = parseNullableInt(row.home_score);
      const awayScore = parseNullableInt(row.away_score);
      if (matchId === null || homeScore === null || awayScore === null) {
        return null;
      }
      return {
        matchId,
        date: row.date?.trim() ?? "",
        stage: row.stage?.trim() ?? "",
        group: normalizeNullableString(row.group),
        homeTeam: row.home_team?.trim() ?? "",
        awayTeam: row.away_team?.trim() ?? "",
        stadium: row.stadium?.trim() ?? "",
        city: row.city?.trim() ?? "",
        country: row.country?.trim() ?? "",
        neutral: parseNullableBool(row.neutral),
        homeScore,
        awayScore,
        wentExtraTime: parseNullableBool(row.went_extra_time) ?? false,
        wentPenalties: parseNullableBool(row.went_penalties) ?? false,
        penaltyWinner: normalizeNullableString(row.penalty_winner),
        winner: normalizeNullableString(row.penalty_winner)
          ?? (homeScore > awayScore
            ? row.home_team?.trim() ?? ""
            : awayScore > homeScore
            ? row.away_team?.trim() ?? ""
            : null),
      } satisfies CompletedWorldCupMatch;
    })
    .filter((row): row is CompletedWorldCupMatch => Boolean(row))
    .sort((a, b) => a.matchId - b.matchId);
}
