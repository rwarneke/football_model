import { WorldCupPredictorPage } from "@/components/world-cup-predictor-page";
import { loadWorldCupPredictorData } from "@/lib/world-cup-predictor";

export default function WorldCupPredictorRoute() {
  const data = loadWorldCupPredictorData();
  const lastUpdated = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  return (
    <main className="px-3 pb-16 pt-12 md:px-12">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.3em] text-ink-400">
            FIFA WORLD CUP 2026
          </p>
          <h1 className="text-4xl font-semibold text-ebony md:text-5xl">
            Tournament Predictor
          </h1>
          <p className="text-base text-ink-200">
            Select match results to watch group standings, qualification slots, and
            the knockout bracket update in real time.
          </p>
          <div className="flex items-center gap-4 text-sm text-ink-400">
            <span>Updated {lastUpdated}</span>
          </div>
        </header>

        <WorldCupPredictorPage data={data} />
      </div>
    </main>
  );
}
