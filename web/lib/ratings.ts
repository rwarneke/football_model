import { z } from "zod";
import { readFile } from "node:fs/promises";
import path from "node:path";

const ratingRowSchema = z.object({
  team: z.string().min(1),
  rating: z.number().finite(),
  rating_attack: z.number().finite(),
  rating_defense: z.number().finite(),
  year: z.number().finite(),
});

export type RatingRow = z.infer<typeof ratingRowSchema> & {
  flagPath: string | null;
};

const DATA_DIR = "/model_output";
const DATA_FILE = `${DATA_DIR}/ratings_current.csv`;
const HISTORY_DATA_FILE = `${DATA_DIR}/ratings_history_yearly.csv`;

function toNumber(value: string | undefined) {
  if (!value) {
    return Number.NaN;
  }
  return Number(value);
}

function toNumberOrNull(value: string | undefined) {
  if (!value) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function flagFileName(team: string) {
  return `${team.replace(/ /g, "_")}.png`;
}

function resolveFlagPath(team: string) {
  if (!team) {
    return null;
  }
  const fileName = flagFileName(team);
  return `/flags/${fileName}`;
}

const PUBLIC_DIR = path.join(process.cwd(), "public");

async function readPublicText(filePath: string) {
  const normalized = filePath.replace(/^\/+/, "");
  const fullPath = path.join(PUBLIC_DIR, normalized);
  return readFile(fullPath, "utf8");
}

export async function loadRatings(): Promise<RatingRow[]> {
  const contents = await readPublicText(DATA_FILE);
  const lines = contents.trim().split(/\r?\n/);
  if (lines.length <= 1) {
    return [];
  }
  const headers = lines[0]?.split(",") ?? [];
  const rows = lines.slice(1).map((line) => {
    const values = line.split(",");
    const record = Object.fromEntries(
      headers.map((header, index) => [header, values[index]])
    ) as Record<string, string | undefined>;

    const parsed = ratingRowSchema.safeParse({
      team: record.team ?? "",
      rating: toNumber(record.rating),
      rating_attack: toNumber(record.rating_attack),
      rating_defense: toNumber(record.rating_defense),
      year: toNumber(record.year),
    });

    if (!parsed.success) {
      return null;
    }

    return {
      ...parsed.data,
      flagPath: resolveFlagPath(parsed.data.team),
    } satisfies RatingRow;
  });

  return rows
    .filter((row): row is RatingRow => Boolean(row))
    .sort((a, b) => b.rating - a.rating);
}

export type RatingsHistoryPoint = {
  date: number;
  [team: string]: number | null;
};

export async function loadRatingsHistory() {
  const contents = await readPublicText(HISTORY_DATA_FILE);
  const lines = contents.trim().split(/\r?\n/);
  if (lines.length <= 1) {
    return { data: [], teams: [] };
  }
  const headers = lines[0]?.split(",") ?? [];
  const teams = headers.slice(1);

  const data = lines
    .slice(1)
    .map((line) => {
      const values = line.split(",");
      const record = Object.fromEntries(
        headers.map((header, index) => [header, values[index]])
      ) as Record<string, string | undefined>;

      const dateValue = record.date ? Date.parse(record.date) : Number.NaN;
      if (!Number.isFinite(dateValue)) {
        return null;
      }

      const entry: RatingsHistoryPoint = { date: dateValue };
      for (const team of teams) {
        entry[team] = toNumberOrNull(record[team]);
      }
      return entry;
    })
    .filter((row): row is RatingsHistoryPoint => Boolean(row));

  return { data, teams };
}
