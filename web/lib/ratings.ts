import fs from "fs";
import path from "path";
import { z } from "zod";

const ratingRowSchema = z.object({
  team: z.string().min(1),
  rating: z.number().finite(),
  rating_attack: z.number().finite(),
  rating_defense: z.number().finite(),
  quality: z.number().finite(),
  year: z.number().finite(),
});

export type RatingRow = z.infer<typeof ratingRowSchema> & {
  flagPath: string;
};

const DATA_FILE = path.resolve(
  process.cwd(),
  "..",
  "model_output",
  "ratings_current.csv"
);

function toNumber(value: string | undefined) {
  if (!value) {
    return Number.NaN;
  }
  return Number(value);
}

function flagFileName(team: string) {
  const normalized = team
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/&/g, "and")
    .replace(/'/g, "")
    .replace(/\./g, "")
    .replace(/-/g, "_")
    .replace(/\s+/g, "_")
    .replace(/[^A-Za-z0-9_]/g, "");
  return `${normalized}.png`;
}

export function loadRatings(): RatingRow[] {
  const contents = fs.readFileSync(DATA_FILE, "utf8");
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
      quality: toNumber(record.quality),
      year: toNumber(record.year),
    });

    if (!parsed.success) {
      return null;
    }

    return {
      ...parsed.data,
      flagPath: `/flags/${flagFileName(parsed.data.team)}`,
    } satisfies RatingRow;
  });

  return rows
    .filter((row): row is RatingRow => Boolean(row))
    .sort((a, b) => b.rating - a.rating);
}
