import { RatingsTable } from "@/components/ratings-table";
import { loadRatings } from "@/lib/ratings";

export default function HomePage() {
  const ratings = loadRatings();
  const lastUpdated = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  return (
    <main className="px-6 pb-16 pt-12 md:px-12">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.3em] text-ink-400">
            Global Ratings
          </p>
          <h1 className="text-4xl font-semibold text-white md:text-5xl">
            International Soccer Power Table
          </h1>
          <p className="max-w-2xl text-base text-ink-200">
            A lightweight snapshot of current national-team strength derived from
            our model output. Ratings are updated from the latest data pull and
            sortable across overall strength, attack, defense, and short-term
            form.
          </p>
          <div className="flex items-center gap-4 text-sm text-ink-400">
            <span>Updated {lastUpdated}</span>
            <span className="h-1 w-1 rounded-full bg-ink-600" />
            <span>{ratings.length} teams</span>
          </div>
        </header>

        <RatingsTable data={ratings} />
      </div>
    </main>
  );
}
