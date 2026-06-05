import { readFile } from "node:fs/promises";
import path from "node:path";

export type WorldCupProbabilities = {
  columns: string[];
  rows: Array<{
    team: string;
    flagPath: string;
    group: string | null;
    opponentProbabilities: OpponentProbabilities;
    groupRankProbabilities: GroupRankProbabilities;
    values: Record<string, number>;
    statuses: Record<string, ProbabilityStatus>;
  }>;
};

export type WorldCupOptionPricing = {
  strikes: number[];
  rows: Array<{
    team: string;
    flagPath: string;
    group: string | null;
    progressionFairValue: number;
    winFairValue: number;
    totalFairValue: number;
    calls: Record<string, number>;
    puts: Record<string, number>;
  }>;
};

export type ProbabilityStatus = "G" | "U" | "I";

export type OpponentProbabilities = {
  R32: Record<string, number>;
  R16: Record<string, number>;
  QF: Record<string, number>;
  SF: Record<string, number>;
  Final: Record<string, number>;
};

export type GroupRankProbabilities = Record<string, number>;

const DATA_FILE = "/model_output/simulation_results.csv";
const STATUS_FILE = "/model_output/simulation_results_status.csv";
const TEAM_PROB_FILE = "/model_output/simulation_team_probabilities.json";
const TEAM_VALUE_FILE = "/model_output/simulation_team_value_pricing.json";

function toNumber(value: string | undefined) {
  if (!value) {
    return Number.NaN;
  }
  return Number(value);
}

function toStatus(value: string | undefined): ProbabilityStatus {
  if (value === "G" || value === "U" || value === "I") {
    return value;
  }
  return "U";
}

function flagFileName(team: string) {
  return `${team.replace(/ /g, "_")}.png`;
}

const PUBLIC_DIR = path.join(process.cwd(), "public");

function isErrnoException(error: unknown): error is NodeJS.ErrnoException {
  return typeof error === "object" && error !== null && "code" in error;
}

async function readPublicText(filePath: string) {
  const normalized = filePath.replace(/^\/+/, "");
  const fullPath = path.join(PUBLIC_DIR, normalized);
  return readFile(fullPath, "utf8");
}

async function readOptionalPublicText(filePath: string) {
  try {
    return await readPublicText(filePath);
  } catch (error) {
    if (isErrnoException(error) && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

function emptyOpponentProbabilities(): OpponentProbabilities {
  return { R32: {}, R16: {}, QF: {}, SF: {}, Final: {} };
}

export async function loadWorldCupProbabilities(): Promise<WorldCupProbabilities> {
  const contents = await readPublicText(DATA_FILE);
  const lines = contents.trim().split(/\r?\n/);
  if (lines.length <= 1) {
    return { columns: [], rows: [] };
  }
  const headers = lines[0]?.split(",") ?? [];
  const columnRenames = new Map<string, string>([
    ["Reach Ro32", "Reach R32"],
    ["Reach Ro16", "Reach R16"],
  ]);
  const columnDefs = headers
    .filter((header) => header !== "team")
    .map((header) => ({
      source: header,
      label: columnRenames.get(header) ?? header,
    }));
  const columns = columnDefs.map((column) => column.label);

  let statusHeaders: string[] = [];
  const statusMap = new Map<string, Record<string, string | undefined>>();
  const statusContents = await readOptionalPublicText(STATUS_FILE);
  if (statusContents) {
    const statusLines = statusContents.trim().split(/\r?\n/);
    statusHeaders = statusLines[0]?.split(",") ?? [];
    for (const line of statusLines.slice(1)) {
      const values = line.split(",");
      const record = Object.fromEntries(
        statusHeaders.map((header, index) => [header, values[index]])
      ) as Record<string, string | undefined>;
      const team = record.team;
      if (team) {
        statusMap.set(team, record);
      }
    }
  }

  const referenceDir = "/reference_data";
  const groupFile = `${referenceDir}/world_cup_2026_groups.csv`;
  const qualifiedFile = `${referenceDir}/world_cup_2026_qualified.csv`;
  const groupContents = await readPublicText(groupFile);
  const groupLines = groupContents.trim().split(/\r?\n/);
  const groupHeaders = groupLines[0]?.split(",") ?? [];
  const groupRows = groupLines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      groupHeaders.map((header, index) => [header, values[index]])
    ) as Record<string, string | undefined>;
  });
  const groupMap = new Map(
    groupRows
      .filter((row) => row.team && row.group)
      .map((row) => [row.team as string, row.group as string])
  );

  const qualifiedContents = await readPublicText(qualifiedFile);
  const qualifiedLines = qualifiedContents.trim().split(/\r?\n/);
  const qualifiedHeaders = qualifiedLines[0]?.split(",") ?? [];
  const qualifiedRows = qualifiedLines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      qualifiedHeaders.map((header, index) => [header, values[index]])
    ) as Record<string, string | undefined>;
  });
  const qualifiedSet = new Set(
    qualifiedRows
      .filter((row) => row.team)
      .map((row) => row.team as string)
  );

  const remainingFile = `${referenceDir}/world_cup_2026_remaining_qualifiers.csv`;
  const remainingContents = await readPublicText(remainingFile);
  const remainingLines = remainingContents.trim().split(/\r?\n/);
  const remainingHeaders = remainingLines[0]?.split(",") ?? [];
  const remainingRows = remainingLines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      remainingHeaders.map((header, index) => [header, values[index]])
    ) as Record<string, string | undefined>;
  });

  const mapFiles = [
    `${referenceDir}/fifa_member_to_canonical_name_map.csv`,
    `${referenceDir}/kaggle_team_to_canonical_name_map.csv`,
  ];
  const nameMap = new Map<string, string>();
  for (const mapFile of mapFiles) {
    const mapContents = await readPublicText(mapFile);
    const mapLines = mapContents.trim().split(/\r?\n/);
    const mapHeaders = mapLines[0]?.split(",") ?? [];
    const mapRows = mapLines.slice(1).map((line) => {
      const values = line.split(",");
      return Object.fromEntries(
        mapHeaders.map((header, index) => [header, values[index]])
      ) as Record<string, string | undefined>;
    });
    for (const row of mapRows) {
      if (row.original_name && row.replacement_name) {
        nameMap.set(row.original_name, row.replacement_name);
      }
    }
  }

  const opponentMap = new Map<string, OpponentProbabilities>();
  const groupRankMap = new Map<string, GroupRankProbabilities>();
  const teamProbContents = await readOptionalPublicText(TEAM_PROB_FILE);
  if (teamProbContents) {
    const parsed = JSON.parse(teamProbContents) as Record<
      string,
      Record<string, Record<string, number> | undefined>
    >;
    for (const [team, record] of Object.entries(parsed)) {
      const getMap = (key: string) => {
        const value = record?.[key];
        if (!value || typeof value !== "object") {
          return {};
        }
        return Object.fromEntries(
          Object.entries(value).filter(
            ([, probability]) => typeof probability === "number"
          )
        );
      };
      opponentMap.set(team, {
        R32: getMap("R32_opponent_probability"),
        R16: getMap("R16_opponent_probability"),
        QF: getMap("QF_opponent_probability"),
        SF: getMap("SF_opponent_probability"),
        Final: getMap("Final_opponent_probability"),
      });
      groupRankMap.set(team, getMap("group_stage_rank_probability"));
    }
  }

  const pathGroupMap = new Map<string, string>();
  for (const row of groupRows) {
    const team = row.team ?? "";
    const group = row.group ?? "";
    if (!team || !group) {
      continue;
    }
    if (team.includes("Path") && team.toLowerCase().includes("winner")) {
      const path = team.replace(" winner", "");
      pathGroupMap.set(path, group);
    }
  }

  const teamPathMap = new Map<string, string>();
  for (const row of remainingRows) {
    const path = row.path ?? "";
    const home = row.home_team ? nameMap.get(row.home_team) ?? row.home_team : "";
    const away = row.away_team ? nameMap.get(row.away_team) ?? row.away_team : "";
    if (!path) {
      continue;
    }
    if (home) {
      teamPathMap.set(home, path);
    }
    if (away) {
      teamPathMap.set(away, path);
    }
  }

  const rows = lines.slice(1).map((line) => {
    const values = line.split(",");
    const record = Object.fromEntries(
      headers.map((header, index) => [header, values[index]])
    ) as Record<string, string | undefined>;
    const team = record.team ?? "";
    const group = groupMap.get(team) ?? null;
    const isQualified = qualifiedSet.has(team);
    const path = teamPathMap.get(team) ?? null;
    const pathGroup = path ? pathGroupMap.get(path) ?? null : null;

    const columnValues: Record<string, number> = {};
    const columnStatuses: Record<string, ProbabilityStatus> = {};
    const statusRecord = statusMap.get(team) ?? {};
    for (const column of columnDefs) {
      columnValues[column.label] = toNumber(record[column.source]);
      columnStatuses[column.label] = statusHeaders.length
        ? toStatus(statusRecord[column.source])
        : "U";
    }

    const resolvedGroup = group ?? pathGroup;
    const groupLabel = resolvedGroup
      ? isQualified
        ? resolvedGroup
        : `${resolvedGroup}*`
      : null;

    return {
      team,
      flagPath: `/flags/${flagFileName(team)}`,
      group: groupLabel,
      opponentProbabilities: opponentMap.get(team) ?? emptyOpponentProbabilities(),
      groupRankProbabilities: groupRankMap.get(team) ?? {},
      values: columnValues,
      statuses: columnStatuses,
    };
  });

  return { columns, rows };
}

export async function loadWorldCupOptionPricing(): Promise<WorldCupOptionPricing> {
  const [probabilities, valueContents] = await Promise.all([
    loadWorldCupProbabilities(),
    readOptionalPublicText(TEAM_VALUE_FILE),
  ]);

  if (!valueContents) {
    return { strikes: [], rows: [] };
  }

  const parsed = JSON.parse(valueContents) as {
    value_definition?: { call_put_strikes?: number[] };
    teams?: Record<
      string,
      {
        progression_fair_value?: number;
        win_fair_value?: number;
        total_fair_value?: number;
        calls?: Record<string, number>;
        puts?: Record<string, number>;
      }
    >;
  };

  const strikes = Array.isArray(parsed.value_definition?.call_put_strikes)
    ? parsed.value_definition?.call_put_strikes.filter((value) => Number.isFinite(value))
    : [];
  const teamValues = parsed.teams ?? {};

  const rows = probabilities.rows
    .flatMap((row) => {
      const entry = teamValues[row.team];
      if (!entry) {
        return [];
      }
      return [
        {
          team: row.team,
          flagPath: row.flagPath,
          group: row.group,
          progressionFairValue: Number(entry.progression_fair_value ?? 0),
          winFairValue: Number(entry.win_fair_value ?? 0),
          totalFairValue: Number(entry.total_fair_value ?? 0),
          calls: Object.fromEntries(
            Object.entries(entry.calls ?? {}).filter(([, value]) =>
              Number.isFinite(value)
            )
          ),
          puts: Object.fromEntries(
            Object.entries(entry.puts ?? {}).filter(([, value]) =>
              Number.isFinite(value)
            )
          ),
        },
      ];
    })
    .sort((a, b) => b.totalFairValue - a.totalFairValue);

  return { strikes, rows };
}
