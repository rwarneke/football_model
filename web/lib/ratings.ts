import { z } from "zod";
import { readFile } from "node:fs/promises";
import path from "node:path";

const ratingRowSchema = z.object({
  team: z.string().min(1),
  rating: z.number().finite(),
  tilt: z.number().finite(),
  rating_attack: z.number().finite(),
  rating_defense: z.number().finite(),
  year: z.number().finite(),
});

export type RatingRow = z.infer<typeof ratingRowSchema> & {
  flagPath: string | null;
  confederation: string | null;
};

const DATA_DIR = "/model_output";
const DATA_FILE_NAME = "ratings_current.csv";
const HISTORY_DATA_FILE = `${DATA_DIR}/ratings_history_yearly.csv`;
const CONFEDERATIONS_FILE = "/reference_data/confederations.csv";

function toNumber(value: string | undefined) {
  if (!value) {
    return Number.NaN;
  }
  return Number(value);
}

function deriveTilt(record: Record<string, string | undefined>) {
  const explicitDisplayTilt = toNumber(record.display_tilt);
  if (Number.isFinite(explicitDisplayTilt)) {
    return explicitDisplayTilt;
  }
  const explicitTilt = toNumber(record.tilt);
  if (Number.isFinite(explicitTilt)) {
    return explicitTilt;
  }
  const muAttack = toNumber(record.mu_attack);
  const muDefense = toNumber(record.mu_defense);
  if (Number.isFinite(muAttack) && Number.isFinite(muDefense)) {
    return muAttack - muDefense;
  }
  return Number.NaN;
}

function toDisplayTilt(value: number) {
  if (!Number.isFinite(value)) {
    return Number.NaN;
  }
  return 10 * Math.tanh(value / 0.5);
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

type ConfederationEntry = {
  team: string;
  confederation: string;
  startYear: number | null;
  endYear: number | null;
};

const PUBLIC_DIR = path.join(process.cwd(), "public");

async function readPublicText(filePath: string) {
  const normalized = filePath.replace(/^\/+/, "");
  const fullPath = path.join(PUBLIC_DIR, normalized);
  return readFile(fullPath, "utf8");
}

function parseYear(value: string | undefined) {
  if (!value) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildConfederationsMap(contents: string) {
  const lines = contents.trim().split(/\r?\n/);
  if (lines.length <= 1) {
    return new Map<string, ConfederationEntry[]>();
  }
  const headers = lines[0]?.split(",") ?? [];
  const entries = lines.slice(1).flatMap((line) => {
    if (!line.trim()) {
      return [];
    }
    const values = line.split(",");
    const record = Object.fromEntries(
      headers.map((header, index) => [header, values[index]])
    ) as Record<string, string | undefined>;
    const team = record.team?.trim();
    const confederation = record.confederation?.trim();
    if (!team || !confederation) {
      return [];
    }
    return [
      {
        team,
        confederation,
        startYear: parseYear(record.start_year),
        endYear: parseYear(record.end_year),
      } satisfies ConfederationEntry,
    ];
  });

  const map = new Map<string, ConfederationEntry[]>();
  for (const entry of entries) {
    const list = map.get(entry.team);
    if (list) {
      list.push(entry);
    } else {
      map.set(entry.team, [entry]);
    }
  }
  return map;
}

function entryStartValue(entry: ConfederationEntry) {
  return entry.startYear ?? Number.NEGATIVE_INFINITY;
}

function entryEndValue(entry: ConfederationEntry) {
  return entry.endYear ?? Number.POSITIVE_INFINITY;
}

function pickConfederation(
  entries: ConfederationEntry[],
  year: number
): ConfederationEntry | null {
  if (!entries.length) {
    return null;
  }
  const hasYear = Number.isFinite(year);
  const inRange = hasYear
    ? entries.filter(
        (entry) =>
          entryStartValue(entry) <= year && entryEndValue(entry) >= year
      )
    : [];
  const candidates = inRange.length ? inRange : entries;

  return (
    candidates.reduce<ConfederationEntry | null>((best, entry) => {
      if (!best) {
        return entry;
      }
      const startDelta = entryStartValue(entry) - entryStartValue(best);
      if (startDelta !== 0) {
        return startDelta > 0 ? entry : best;
      }
      const endDelta = entryEndValue(entry) - entryEndValue(best);
      return endDelta > 0 ? entry : best;
    }, null) ?? null
  );
}

export async function loadRatings(modelOutputDir = DATA_DIR): Promise<RatingRow[]> {
  const [contents, confederationsContents] = await Promise.all([
    readPublicText(`${modelOutputDir}/${DATA_FILE_NAME}`),
    readPublicText(CONFEDERATIONS_FILE),
  ]);
  const confederationsMap = buildConfederationsMap(confederationsContents);
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
      tilt: record.display_tilt ? deriveTilt(record) : toDisplayTilt(deriveTilt(record)),
      rating_attack: toNumber(record.rating_attack),
      rating_defense: toNumber(record.rating_defense),
      year: toNumber(record.year),
    });

    if (!parsed.success) {
      return null;
    }

    const confederation =
      confederationsMap.size > 0
        ? pickConfederation(
            confederationsMap.get(parsed.data.team) ?? [],
            parsed.data.year
          )?.confederation ?? null
        : null;

    return {
      ...parsed.data,
      flagPath: resolveFlagPath(parsed.data.team),
      confederation,
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
