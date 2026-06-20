import type { Metadata } from "next";
import { stat } from "node:fs/promises";
import path from "node:path";
import { WorldCupProbabilitiesPage } from "@/components/world-cup-probabilities-page";
import { loadRatings } from "@/lib/ratings";
import { loadCompletedWorldCupMatches } from "@/lib/world-cup-results";
import { loadWorldCupProbabilities } from "@/lib/world-cup";

export const metadata: Metadata = {
  title: "World Cup 2026 Progression Chances",
};

async function modelOutputUpdatedLabel(dirName: string) {
  const stats = await stat(
    path.join(process.cwd(), "public", dirName.replace(/^\/+/, ""), "simulation_results.csv")
  );
  return stats.mtime.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

async function loadPretournamentProbabilities() {
  try {
    return await loadWorldCupProbabilities("/model_output_pretournament");
  } catch {
    return await loadWorldCupProbabilities("/model_output");
  }
}

async function loadPretournamentRatings() {
  try {
    return await loadRatings("/model_output_pretournament");
  } catch {
    return await loadRatings("/model_output");
  }
}

async function pretournamentUpdatedLabel() {
  try {
    return await modelOutputUpdatedLabel("/model_output_pretournament");
  } catch {
    return await modelOutputUpdatedLabel("/model_output");
  }
}

function stripQualifyColumn<T extends { values: Record<string, number>; statuses: Record<string, "G" | "U" | "I"> }>(
  data: {
    columns: string[];
    rows: T[];
  }
) {
  const columns = data.columns.filter((column) => column !== "Qualify");
  const rows = data.rows.map((row) => {
    const { Qualify: _qualifyValue, ...values } = row.values;
    const { Qualify: _qualifyStatus, ...statuses } = row.statuses;
    return {
      ...row,
      values,
      statuses,
    };
  });
  return { columns, rows };
}

function buildGroupRecordMap(
  matches: Awaited<ReturnType<typeof loadCompletedWorldCupMatches>>
) {
  const records = new Map<string, { wins: number; draws: number; losses: number }>();
  const ensureRecord = (team: string) => {
    const existing = records.get(team);
    if (existing) {
      return existing;
    }
    const next = { wins: 0, draws: 0, losses: 0 };
    records.set(team, next);
    return next;
  };

  matches
    .filter((match) => match.stage === "Group")
    .forEach((match) => {
      const home = ensureRecord(match.homeTeam);
      const away = ensureRecord(match.awayTeam);
      if (match.homeScore > match.awayScore) {
        home.wins += 1;
        away.losses += 1;
        return;
      }
      if (match.awayScore > match.homeScore) {
        away.wins += 1;
        home.losses += 1;
        return;
      }
      home.draws += 1;
      away.draws += 1;
    });

  return new Map(
    Array.from(records.entries()).map(([team, record]) => [
      team,
      `${record.wins}-${record.draws}-${record.losses}`,
    ])
  );
}

export default async function WorldCupProbabilitiesRoute() {
  const [
    currentRaw,
    pretournamentRaw,
    currentRatings,
    pretournamentRatings,
    currentCompletedMatches,
    currentUpdatedLabel,
    pretournamentUpdated,
  ] =
    await Promise.all([
      loadWorldCupProbabilities("/model_output"),
      loadPretournamentProbabilities(),
      loadRatings("/model_output"),
      loadPretournamentRatings(),
      loadCompletedWorldCupMatches("/model_output"),
      modelOutputUpdatedLabel("/model_output"),
      pretournamentUpdatedLabel(),
    ]);
  const current = stripQualifyColumn(currentRaw);
  const pretournament = stripQualifyColumn(pretournamentRaw);
  const currentGroupRecordMap = buildGroupRecordMap(currentCompletedMatches);
  const buildRatingsMap = (ratings: Awaited<ReturnType<typeof loadRatings>>) =>
    new Map(
      ratings.map((row) => [
        row.team,
        {
          rating: row.rating,
          tilt: row.tilt,
        },
      ])
    );
  const currentRatingsMap = buildRatingsMap(currentRatings);
  const pretournamentRatingsMap = buildRatingsMap(pretournamentRatings);
  const attachRatings = (
    rows: typeof current.rows,
    ratingsMap: Map<
      string,
      {
        rating: number;
        tilt: number;
      }
    >,
    groupRecordMap?: Map<string, string>
  ) =>
    rows.map((row) => {
      const rating = ratingsMap.get(row.team);
      return {
        ...row,
        ratingOverall: rating?.rating,
        tilt: rating?.tilt,
        groupRecord: groupRecordMap?.get(row.team),
      };
    });

  return (
    <main className="px-2 pb-16 pt-8 lg:px-6">
      <div className="flex w-full flex-col gap-10">
        <WorldCupProbabilitiesPage
          current={{
            columns: current.columns,
            rows: attachRatings(current.rows, currentRatingsMap, currentGroupRecordMap),
          }}
          pretournament={{
            columns: pretournament.columns,
            rows: attachRatings(pretournament.rows, pretournamentRatingsMap),
          }}
          currentUpdatedLabel={currentUpdatedLabel}
          pretournamentUpdatedLabel={pretournamentUpdated}
        />
      </div>
    </main>
  );
}
