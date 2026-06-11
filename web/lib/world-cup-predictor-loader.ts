import type {
  CompletedWorldCupMatch,
  GroupDefinition,
  GroupMatch,
  KnockoutMatch,
  QualifierMatch,
  RoundOf32Combos,
  TeamStageProbabilities,
  WinProbabilities,
  WorldCupPredictorData,
} from "@/lib/world-cup-predictor-types";

const REFERENCE_DIR = "/reference_data";
const GROUPS_FILE = `${REFERENCE_DIR}/world_cup_2026_groups.csv`;
const GROUP_MATCHES_FILE = `${REFERENCE_DIR}/world_cup_2026_group_matches.csv`;
const KNOCKOUT_MATCHES_FILE = `${REFERENCE_DIR}/world_cup_2026_knockout_matches.csv`;
const ROUND_OF_32_FILE = `${REFERENCE_DIR}/world_cup_2026_round_of_32_combinations.csv`;
const QUALIFIERS_FILE = `${REFERENCE_DIR}/world_cup_2026_remaining_qualifiers.csv`;
const RESULTS_FILE_NAME = "results_wc2026.csv";

async function readCsv(
  filePath: string,
  fetchTextFn: (path: string) => Promise<string>
) {
  const contents = (await fetchTextFn(filePath)).trim();
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

async function readOptionalCsv(
  filePath: string,
  fetchTextFn: (path: string) => Promise<string>
) {
  try {
    return await readCsv(filePath, fetchTextFn);
  } catch {
    return { headers: [] as string[], rows: [] as Record<string, string>[] };
  }
}

function isSlotPlaceholder(name: string) {
  return /\bwinner$/i.test(name.trim());
}

function normalizeName(raw: string): string {
  const trimmed = raw?.trim() ?? "";
  if (!trimmed || trimmed.toLowerCase() === "nan" || trimmed.toLowerCase() === "na") {
    return "";
  }
  if (isSlotPlaceholder(trimmed)) {
    return trimmed;
  }
  return trimmed;
}

function resolveFlagPath(team: string) {
  if (!team || isSlotPlaceholder(team)) {
    return null;
  }
  const fileName = `${team.replace(/ /g, "_")}.png`;
  return `/flags/${fileName}`;
}

export async function loadWorldCupPredictorDataWithFetchers(
  fetchTextFn: (path: string) => Promise<string>,
  fetchJsonFn: (path: string) => Promise<unknown>,
  modelOutputDir = "/model_output"
): Promise<WorldCupPredictorData> {
  const winProbabilitiesFile = `${modelOutputDir}/win_probabilities.json`;
  const teamProbabilitiesFile = `${modelOutputDir}/simulation_team_probabilities.json`;
  const resultsFile = `${modelOutputDir}/${RESULTS_FILE_NAME}`;
  const groupRows = (await readCsv(GROUPS_FILE, fetchTextFn)).rows;
  const groupsMap = new Map<string, string[]>();
  for (const row of groupRows) {
    const group = row.group?.trim();
    if (!group) {
      continue;
    }
    const team = normalizeName(row.team ?? "");
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

  const groupMatches = (await readCsv(GROUP_MATCHES_FILE, fetchTextFn)).rows.map(
    (row) => ({
      id: Number(row.match_id),
      date: row.date,
      group: row.group?.trim() ?? "",
      homeTeam: normalizeName(row.home_team ?? ""),
      awayTeam: normalizeName(row.away_team ?? ""),
      stadium: row.stadium?.trim() ?? "",
      city: row.city?.trim() ?? "",
      country: row.country?.trim() ?? "",
    })
  );

  const knockoutMatches = (await readCsv(KNOCKOUT_MATCHES_FILE, fetchTextFn)).rows.map(
    (row) => ({
      id: Number(row.match_id),
      stage: row.stage?.trim() ?? "",
      date: row.date,
      homeLabel: row.home?.trim() ?? "",
      awayLabel: row.away?.trim() ?? "",
      stadium: row.stadium?.trim() ?? "",
      city: row.city?.trim() ?? "",
      country: row.country?.trim() ?? "",
    })
  );

  const combosRows = (await readCsv(ROUND_OF_32_FILE, fetchTextFn)).rows;
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

  const qualifierRows = (await readCsv(QUALIFIERS_FILE, fetchTextFn)).rows;
  const qualifiers: QualifierMatch[] = qualifierRows.map((row) => ({
    id: `${row.path?.trim() ?? "path"}-${row.round?.trim() ?? "round"}`,
    date: row.date,
    stage: row.stage?.trim() ?? "",
    path: row.path?.trim() ?? "",
    round: row.round?.trim() ?? "",
    homeTeam: normalizeName(row.home_team ?? ""),
    awayTeam: normalizeName(row.away_team ?? ""),
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

  const completedMatches: CompletedWorldCupMatch[] = (
    await readOptionalCsv(resultsFile, fetchTextFn)
  ).rows
    .map((row) => {
      const homeScore = Number(row.home_score);
      const awayScore = Number(row.away_score);
      const matchId = Number(row.match_id);
      if (
        !Number.isInteger(matchId) ||
        !Number.isInteger(homeScore) ||
        !Number.isInteger(awayScore)
      ) {
        return null;
      }
      const penaltyWinner = normalizeName(row.penalty_winner ?? "");
      const winner =
        penaltyWinner
          || (homeScore > awayScore
            ? normalizeName(row.home_team ?? "")
            : awayScore > homeScore
            ? normalizeName(row.away_team ?? "")
            : "");
      return {
        matchId,
        date: row.date,
        stage: row.stage?.trim() ?? "",
        group: normalizeName(row.group ?? "") || null,
        homeTeam: normalizeName(row.home_team ?? ""),
        awayTeam: normalizeName(row.away_team ?? ""),
        stadium: row.stadium?.trim() ?? "",
        city: row.city?.trim() ?? "",
        country: row.country?.trim() ?? "",
        neutral:
          String(row.neutral ?? "").trim().toLowerCase() === "true"
            ? true
            : String(row.neutral ?? "").trim().toLowerCase() === "false"
            ? false
            : null,
        homeScore,
        awayScore,
        wentExtraTime:
          String(row.went_extra_time ?? "").trim().toLowerCase() === "true",
        wentPenalties:
          String(row.went_penalties ?? "").trim().toLowerCase() === "true",
        penaltyWinner: penaltyWinner || null,
        winner: winner || null,
      } satisfies CompletedWorldCupMatch;
    })
    .filter((row): row is CompletedWorldCupMatch => Boolean(row));

  const winProbabilities =
    (await fetchJsonFn(winProbabilitiesFile)) ?? {};
  const simulationTeamProbabilities =
    (await fetchJsonFn(teamProbabilitiesFile)) ?? {};

  return {
    groups,
    groupMatches,
    knockoutMatches,
    roundOf32Combos,
    qualifiers,
    flags,
    winProbabilities: winProbabilities as WinProbabilities,
    simulationTeamProbabilities:
      simulationTeamProbabilities as Record<string, TeamStageProbabilities>,
    completedMatches,
  };
}
