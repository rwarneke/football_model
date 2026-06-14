import type { Metadata } from "next";
import { stat } from "node:fs/promises";
import path from "node:path";
import { WorldCupProbabilitiesPage } from "@/components/world-cup-probabilities-page";
import { loadRatings } from "@/lib/ratings";
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

export default async function WorldCupProbabilitiesRoute() {
  const [
    current,
    pretournament,
    currentRatings,
    pretournamentRatings,
    currentUpdatedLabel,
    pretournamentUpdated,
  ] =
    await Promise.all([
      loadWorldCupProbabilities("/model_output"),
      loadPretournamentProbabilities(),
      loadRatings("/model_output"),
      loadPretournamentRatings(),
      modelOutputUpdatedLabel("/model_output"),
      pretournamentUpdatedLabel(),
    ]);
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
    >
  ) =>
    rows.map((row) => {
      const rating = ratingsMap.get(row.team);
      return {
        ...row,
        ratingOverall: rating?.rating,
        tilt: rating?.tilt,
      };
    });

  return (
    <main className="px-2 pb-16 pt-8 lg:px-6">
      <div className="flex w-full flex-col gap-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.3em] text-ink-400">
            FIFA WORLD CUP 2026
          </p>
          <h1 className="text-3xl font-semibold text-ebony md:text-4xl">
            Progression Chances
          </h1>
          <p className="text-base text-ink-200">
            Each team's probability of reaching each stage of the 2026 FIFA
            World Cup, based on 10,000 simulations.
          </p>
        </header>

        <WorldCupProbabilitiesPage
          current={{
            columns: current.columns,
            rows: attachRatings(current.rows, currentRatingsMap),
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
