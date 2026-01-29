import { RatingsPage } from "@/components/ratings-page";
import { loadRatings } from "@/lib/ratings";

export default function CurrentRatingsPage() {
  const ratings = loadRatings();
  const lastUpdated = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  return (
    <main className="px-2 pb-16 pt-12 lg:px-6">
      <div className="flex w-full flex-col gap-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.3em] text-ink-400">
            Current Ratings
          </p>
          <h1 className="text-4xl font-semibold text-ebony md:text-5xl">
            International Football Team Ratings
          </h1>
          <p className="text-base text-ink-200">
            All current teams who have ever participated in FIFA World Cup
            qualification are included (211 FIFA members plus 6 confederation
            members).
          </p>
          <div className="flex items-center gap-4 text-sm text-ink-400">
            <span>Updated {lastUpdated}</span>
            <span className="h-1 w-1 rounded-lg bg-ink-600" />
            <span>{ratings.length} teams</span>
          </div>
        </header>

        <RatingsPage data={ratings} />
      </div>
    </main>
  );
}
