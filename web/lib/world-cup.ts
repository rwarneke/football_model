export type WorldCupProbabilities = {
  columns: string[];
  rows: Array<{
    team: string;
    flagPath: string;
    group: string | null;
    values: Record<string, number>;
    statuses: Record<string, ProbabilityStatus>;
  }>;
};

export type ProbabilityStatus = "G" | "U" | "I";

const DATA_FILE = "/model_output/simulation_results.csv";
const STATUS_FILE = "/model_output/simulation_results_status.csv";

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

async function fetchText(filePath: string) {
  const headerList = headers();
  const forwardedHost = headerList.get("x-forwarded-host");
  const hostValue = forwardedHost ?? headerList.get("host");
  const proto = headerList.get("x-forwarded-proto") ?? "https";
  if (!hostValue) {
    throw new Error("Missing host header for data fetch");
  }
  const host = hostValue.startsWith("0.0.0.0")
    ? hostValue.replace(/^0\.0\.0\.0/, "127.0.0.1")
    : hostValue;
  const res = await fetch(`${proto}://${host}${filePath}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load ${filePath}`);
  }
  return res.text();
}

async function fetchOptionalText(filePath: string) {
  const headerList = headers();
  const forwardedHost = headerList.get("x-forwarded-host");
  const hostValue = forwardedHost ?? headerList.get("host");
  const proto = headerList.get("x-forwarded-proto") ?? "https";
  if (!hostValue) {
    return null;
  }
  const host = hostValue.startsWith("0.0.0.0")
    ? hostValue.replace(/^0\.0\.0\.0/, "127.0.0.1")
    : hostValue;
  const res = await fetch(`${proto}://${host}${filePath}`, { cache: "no-store" });
  if (!res.ok) {
    return null;
  }
  return res.text();
}

export async function loadWorldCupProbabilities(): Promise<WorldCupProbabilities> {
  const contents = await fetchText(DATA_FILE);
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
  const statusContents = await fetchOptionalText(STATUS_FILE);
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
  const groupContents = await fetchText(groupFile);
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

  const qualifiedContents = await fetchText(qualifiedFile);
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
  const remainingContents = await fetchText(remainingFile);
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
    const mapContents = await fetchText(mapFile);
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
      values: columnValues,
      statuses: columnStatuses,
    };
  });

  return { columns, rows };
}
import { headers } from "next/headers";
