import { RatingsHistoryChart } from "@/components/ratings-history-chart";
import { loadRatingsHistory } from "@/lib/ratings";

export const dynamic = "force-dynamic";
export const runtime = "edge";

export default async function HistoryPage() {
  const { data, teams } = await loadRatingsHistory();
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
            Historical Ratings
          </p>
          <h1 className="text-4xl font-semibold text-ebony md:text-5xl">
            Historical International Football Ratings
          </h1>
          <p className="text-base text-ink-200">
            Explore how each team has evolved since its first recorded match.
            Zoom across time and filter to isolate specific teams.
          </p>
          <div className="flex items-center gap-4 text-sm text-ink-400">
            <span>Updated {lastUpdated}</span>
            <span className="h-1 w-1 rounded-lg bg-ink-600" />
            <span>{teams.length} teams</span>
          </div>
        </header>

        <RatingsHistoryChart data={data} teams={teams} />
      </div>
    </main>
  );
}
