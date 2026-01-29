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
    <main className="px-1.5 pb-16 pt-6 sm:px-2 md:px-6">
      <div className="flex w-full flex-col gap-6 sm:gap-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.3em] text-ink-400">
            FIFA WORLD CUP 2026
          </p>
          <h1 className="text-3xl sm:text-4xl font-semibold text-ebony md:text-5xl">
            Tournament Predictor
          </h1>
          <p className="text-base text-ink-200">
            Choose your own results or click "Auto-predict" to simulate stages or groups.
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
