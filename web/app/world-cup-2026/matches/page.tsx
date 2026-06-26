import { readFile } from "node:fs/promises";
import path from "node:path";
import { loadWorldCupMatches } from "@/lib/world-cup-matches";
import { loadCompletedWorldCupMatches } from "@/lib/world-cup-results";
import { loadRatings } from "@/lib/ratings";
import { loadWorldCupMatchConditionalAdvancement } from "@/lib/world-cup-match-conditional-advancement";
import type { WinProbabilities } from "@/lib/world-cup-predictor-types";
import { WorldCupMatchesPageClient } from "@/components/world-cup-matches-page";

export const dynamic = "force-dynamic";
export const revalidate = 0;

async function loadWinProbabilities(): Promise<WinProbabilities> {
  const filePath = path.join(process.cwd(), "public", "model_output", "win_probabilities.json");
  const contents = await readFile(filePath, "utf8");
  return JSON.parse(contents) as WinProbabilities;
}

export default async function WorldCupMatchesPage() {
  const [
    matches,
    completedMatches,
    ratings,
    winProbabilities,
    conditionalAdvancement,
  ] = await Promise.all([
    loadWorldCupMatches(),
    loadCompletedWorldCupMatches(),
    loadRatings(),
    loadWinProbabilities(),
    loadWorldCupMatchConditionalAdvancement(),
  ]);
  const lastUpdated = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  return (
    <main className="px-2 pb-16 pt-8 lg:px-6">
      <div className="flex w-full flex-col gap-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.3em] text-ink-400">
            FIFA WORLD CUP 2026
          </p>
          <h1 className="text-3xl font-semibold text-ebony md:text-4xl">
            Match Predictions
          </h1>
          <div className="flex items-center gap-4 text-sm text-ink-400">
            <span>Updated {lastUpdated}</span>
          </div>
        </header>

        <WorldCupMatchesPageClient
          matches={matches}
          completedMatches={completedMatches}
          ratings={ratings}
          winProbabilities={winProbabilities}
          conditionalAdvancement={conditionalAdvancement}
        />
      </div>
    </main>
  );
}
