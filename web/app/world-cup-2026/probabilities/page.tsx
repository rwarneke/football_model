import { WorldCupProbabilitiesPage } from "@/components/world-cup-probabilities-page";
import { loadRatings } from "@/lib/ratings";
import { loadWorldCupProbabilities } from "@/lib/world-cup";

export const dynamic = "force-dynamic";
export const runtime = "edge";

export default async function WorldCupProbabilitiesRoute() {
  const { columns, rows } = await loadWorldCupProbabilities();
  const ratings = await loadRatings();
  const ratingsMap = new Map(
    ratings.map((row) => [
      row.team,
      {
        ratingOverall: row.rating,
        ratingAttack: row.rating_attack,
        ratingDefense: row.rating_defense,
      },
    ])
  );
  const rowsWithRatings = rows.map((row) => {
    const rating = ratingsMap.get(row.team);
    return {
      ...row,
      ratingOverall: rating?.ratingOverall,
      ratingAttack: rating?.ratingAttack,
      ratingDefense: rating?.ratingDefense,
    };
  });
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
          <h1 className="text-4xl font-semibold text-ebony md:text-5xl">
            Tournament Progression Chances
          </h1>
          <p className="text-base text-ink-200">
            Each team's probability of reaching each stage of the 2026 FIFA
            World Cup, based on 10,000 simulations.
          </p>
          <div className="flex items-center gap-4 text-sm text-ink-400">
            <span>Updated {lastUpdated}</span>
            <span className="h-1 w-1 rounded-lg bg-ink-600" />
            <span>{rows.length} teams</span>
          </div>
        </header>

        <WorldCupProbabilitiesPage columns={columns} rows={rowsWithRatings} />
      </div>
    </main>
  );
}
