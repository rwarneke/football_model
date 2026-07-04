import { readFile } from "node:fs/promises";
import path from "node:path";
import { loadCompletedWorldCupMatches } from "@/lib/world-cup-results";

export type WorldCupProbabilities = {
  columns: string[];
  rows: Array<{
    team: string;
    flagPath: string;
    group: string | null;
    opponentProbabilities: OpponentProbabilities;
    opponentStatuses: OpponentProbabilityStatuses;
    groupRankProbabilities: GroupRankProbabilities;
    groupRankStatuses: ProbabilityStatusMap;
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
    minimumPossibleValue: number;
    maximumPossibleValue: number;
    progressionFairValue: number;
    winFairValue: number;
    totalFairValue: number;
    calls: Record<string, number>;
    puts: Record<string, number>;
  }>;
};

export type ProbabilityStatus = "G" | "U" | "I";
export type ProbabilityStatusMap = Record<string, ProbabilityStatus>;

export type OpponentProbabilities = {
  R32: Record<string, number>;
  R16: Record<string, number>;
  QF: Record<string, number>;
  SF: Record<string, number>;
  Final: Record<string, number>;
};

export type OpponentProbabilityStatuses = {
  R32: ProbabilityStatusMap;
  R16: ProbabilityStatusMap;
  QF: ProbabilityStatusMap;
  SF: ProbabilityStatusMap;
  Final: ProbabilityStatusMap;
};

export type GroupRankProbabilities = Record<string, number>;

const DATA_FILE_NAME = "simulation_results.csv";
const STATUS_FILE_NAME = "simulation_results_status.csv";
const TEAM_PROB_FILE_NAME = "simulation_team_probabilities.json";
const TEAM_VALUE_FILE_NAME = "simulation_team_value_pricing.json";
const DEFAULT_PROGRESSION_STAGE_VALUES: Record<string, number> = {
  "Round of 32": 5,
  "Round of 16": 10,
  Quarterfinal: 20,
  Semifinal: 40,
  Final: 60,
  Champion: 80,
};

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

function parseStatus(value: unknown): ProbabilityStatus | undefined {
  if (value === "G" || value === "U" || value === "I") {
    return value;
  }
  return undefined;
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

function emptyOpponentStatuses(): OpponentProbabilityStatuses {
  return { R32: {}, R16: {}, QF: {}, SF: {}, Final: {} };
}

function computeLockedWinValueByTeam(
  completedMatches: Awaited<ReturnType<typeof loadCompletedWorldCupMatches>>
) {
  const lockedWinValueByTeam = new Map<string, number>();
  for (const match of completedMatches) {
    const homeScore90 = match.homeScore90 ?? match.homeScore;
    const awayScore90 = match.awayScore90 ?? match.awayScore;
    if (homeScore90 > awayScore90) {
      lockedWinValueByTeam.set(
        match.homeTeam,
        (lockedWinValueByTeam.get(match.homeTeam) ?? 0) + 5
      );
    } else if (awayScore90 > homeScore90) {
      lockedWinValueByTeam.set(
        match.awayTeam,
        (lockedWinValueByTeam.get(match.awayTeam) ?? 0) + 5
      );
    }
  }
  return lockedWinValueByTeam;
}

function computeMinimumProgressionValue(
  statuses: Record<string, ProbabilityStatus>,
  progressionStageValues: Record<string, number>
) {
  if (statuses["Champion"] === "G") {
    return progressionStageValues["Champion"] ?? 80;
  }
  if (statuses["Reach Final"] === "G") {
    return progressionStageValues["Final"] ?? 60;
  }
  if (statuses["Reach SF"] === "G") {
    return progressionStageValues["Semifinal"] ?? 40;
  }
  if (statuses["Reach QF"] === "G") {
    return progressionStageValues["Quarterfinal"] ?? 20;
  }
  if (statuses["Reach R16"] === "G") {
    return progressionStageValues["Round of 16"] ?? 10;
  }
  if (statuses["Reach R32"] === "G") {
    return progressionStageValues["Round of 32"] ?? 5;
  }
  return 0;
}

function computeMaximumProgressionValue(
  statuses: Record<string, ProbabilityStatus>,
  progressionStageValues: Record<string, number>
) {
  if (statuses["Champion"] !== "I") {
    return progressionStageValues["Champion"] ?? 80;
  }
  if (statuses["Reach Final"] !== "I") {
    return progressionStageValues["Final"] ?? 60;
  }
  if (statuses["Reach SF"] !== "I") {
    return progressionStageValues["Semifinal"] ?? 40;
  }
  if (statuses["Reach QF"] !== "I") {
    return progressionStageValues["Quarterfinal"] ?? 20;
  }
  if (statuses["Reach R16"] !== "I") {
    return progressionStageValues["Round of 16"] ?? 10;
  }
  if (statuses["Reach R32"] !== "I") {
    return progressionStageValues["Round of 32"] ?? 5;
  }
  return 0;
}

function computeMaximumAdditionalWinValue(
  statuses: Record<string, ProbabilityStatus>
) {
  if (statuses["Champion"] === "G") {
    return 0;
  }
  if (statuses["Reach R16"] === "I") {
    return 0;
  }
  const groupStageWinsRemaining = statuses["Reach R32"] === "G" ? 0 : 15;
  if (statuses["Reach Final"] === "G" && statuses["Champion"] === "I") {
    return 0;
  }
  if (statuses["Reach SF"] === "G" && statuses["Reach Final"] === "I") {
    return 5;
  }
  if (statuses["Reach Final"] === "G") {
    return 5;
  }
  if (statuses["Reach SF"] === "G") {
    return 10;
  }
  if (statuses["Reach QF"] === "G") {
    return 15;
  }
  if (statuses["Reach R16"] === "G") {
    return 20;
  }
  if (statuses["Reach R32"] === "G") {
    return 25;
  }
  return groupStageWinsRemaining + 25;
}

export async function loadWorldCupProbabilities(
  modelOutputDir = "/model_output"
): Promise<WorldCupProbabilities> {
  const contents = await readPublicText(`${modelOutputDir}/${DATA_FILE_NAME}`);
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
  const statusContents = await readOptionalPublicText(
    `${modelOutputDir}/${STATUS_FILE_NAME}`
  );
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
  const opponentStatusMap = new Map<string, OpponentProbabilityStatuses>();
  const groupRankMap = new Map<string, GroupRankProbabilities>();
  const groupRankStatusMap = new Map<string, ProbabilityStatusMap>();
  const stageStatusMap = new Map<string, ProbabilityStatusMap>();
  const teamProbContents = await readOptionalPublicText(
    `${modelOutputDir}/${TEAM_PROB_FILE_NAME}`
  );
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
      const getStatusMap = (key: string) => {
        const value = record?.[key];
        if (!value || typeof value !== "object") {
          return {};
        }
        return Object.fromEntries(
          Object.entries(value).flatMap(([entryKey, status]) => {
            const parsedStatus = parseStatus(status);
            return parsedStatus ? [[entryKey, parsedStatus]] : [];
          })
        ) as ProbabilityStatusMap;
      };
      opponentMap.set(team, {
        R32: getMap("R32_opponent_probability"),
        R16: getMap("R16_opponent_probability"),
        QF: getMap("QF_opponent_probability"),
        SF: getMap("SF_opponent_probability"),
        Final: getMap("Final_opponent_probability"),
      });
      opponentStatusMap.set(team, {
        R32: getStatusMap("R32_opponent_status"),
        R16: getStatusMap("R16_opponent_status"),
        QF: getStatusMap("QF_opponent_status"),
        SF: getStatusMap("SF_opponent_status"),
        Final: getStatusMap("Final_opponent_status"),
      });
      groupRankMap.set(team, getMap("group_stage_rank_probability"));
      groupRankStatusMap.set(team, getStatusMap("group_stage_rank_status"));
      stageStatusMap.set(team, getStatusMap("stage_status"));
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
    const stageStatusRecord = stageStatusMap.get(team) ?? {};
    for (const column of columnDefs) {
      columnValues[column.label] = toNumber(record[column.source]);
      columnStatuses[column.label] =
        parseStatus(statusRecord[column.source]) ??
        parseStatus(stageStatusRecord[column.label]) ??
        (statusHeaders.length ? toStatus(statusRecord[column.source]) : "U");
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
      opponentStatuses: opponentStatusMap.get(team) ?? emptyOpponentStatuses(),
      groupRankProbabilities: groupRankMap.get(team) ?? {},
      groupRankStatuses: groupRankStatusMap.get(team) ?? {},
      values: columnValues,
      statuses: columnStatuses,
    };
  });

  return { columns, rows };
}

export async function loadWorldCupOptionPricing(
  modelOutputDir = "/model_output"
): Promise<WorldCupOptionPricing> {
  const [probabilities, valueContents, completedMatches] = await Promise.all([
    loadWorldCupProbabilities(modelOutputDir),
    readOptionalPublicText(`${modelOutputDir}/${TEAM_VALUE_FILE_NAME}`),
    loadCompletedWorldCupMatches(modelOutputDir),
  ]);

  if (!valueContents) {
    return { strikes: [], rows: [] };
  }

  const parsed = JSON.parse(valueContents) as {
    value_definition?: {
      call_put_strikes?: number[];
      progression_stage_values?: Record<string, number>;
    };
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
  const progressionStageValues = {
    ...DEFAULT_PROGRESSION_STAGE_VALUES,
    ...(parsed.value_definition?.progression_stage_values ?? {}),
  };
  const teamValues = parsed.teams ?? {};
  const lockedWinValueByTeam = computeLockedWinValueByTeam(completedMatches);

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
          minimumPossibleValue:
            (lockedWinValueByTeam.get(row.team) ?? 0) +
            computeMinimumProgressionValue(row.statuses, progressionStageValues),
          maximumPossibleValue:
            (lockedWinValueByTeam.get(row.team) ?? 0) +
            computeMaximumProgressionValue(row.statuses, progressionStageValues) +
            computeMaximumAdditionalWinValue(row.statuses),
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
