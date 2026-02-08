import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { loadRatings } from "@/lib/ratings";
import { loadWorldCupProbabilities } from "@/lib/world-cup";

export const metadata: Metadata = {
  title: "TheBackPost Football Analytics",
};

const ratingFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const percentFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const ACCENT_DARK_RGB = "16, 185, 129";
const ACCENT_LIGHT_RGB = "147, 197, 253";

function ratingBackground(value: number, minValue: number) {
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(minValue) ||
    minValue >= 100
  ) {
    return undefined;
  }
  const clamped = Math.max(minValue, Math.min(value, 100));
  const scaled = (clamped - minValue) / (100 - minValue);
  const alpha = 0.3 + scaled * 0.7;
  return { backgroundColor: `rgba(${ACCENT_DARK_RGB}, ${alpha})` };
}

function probabilityBackground(value: number, maxValue: number) {
  if (!Number.isFinite(value) || !Number.isFinite(maxValue) || maxValue <= 0) {
    return undefined;
  }
  const clamped = Math.max(0, Math.min(value, maxValue));
  const alpha = clamped / maxValue;
  return { backgroundColor: `rgba(${ACCENT_LIGHT_RGB}, ${alpha})` };
}

function formatProbability(value: number, status: "G" | "U" | "I") {
  if (status === "G") {
    return "✓";
  }
  if (status === "I") {
    return "✕";
  }
  if (!Number.isFinite(value)) {
    return "";
  }
  if (value < 0.001) {
    return "<0.1%";
  }
  if (value >= 0.999) {
    return ">99.9";
  }
  return `${percentFormatter.format(value * 100)}%`;
}

export default async function HomePage() {
  const topRatings = (await loadRatings()).slice(0, 5);
  const ratingValues = topRatings.flatMap((row) => [
    row.rating,
    row.rating_attack,
    row.rating_defense,
  ]);
  const minRating = ratingValues.reduce(
    (min, value) => (Number.isFinite(value) ? Math.min(min, value) : min),
    Number.POSITIVE_INFINITY
  );
  const worldCup = await loadWorldCupProbabilities();
  const stageColumns = ["Reach SF", "Reach Final", "Champion"];
  const topStageRows = [...worldCup.rows]
    .sort(
      (a, b) =>
        Number(b.values.Champion ?? 0) - Number(a.values.Champion ?? 0)
    )
    .slice(0, 5);
  const maxStageProbability = topStageRows.reduce((max, row) => {
    return Math.max(
      max,
      ...stageColumns.map((col) => Number(row.values[col] ?? 0))
    );
  }, 0);
  return (
    <main className="px-2 pb-16 pt-8 lg:px-6">
      <div className="flex w-full flex-col gap-10">
        <section className="grid gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200">
            <div className="flex flex-col gap-4">
              <h2 className="text-lg font-semibold">International football team ratings</h2>
              <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
                <div className="grid grid-cols-[1.5rem_2rem_1fr_repeat(3,3.25rem)] sm:grid-cols-[3rem_2rem_1fr_repeat(3,4rem)] bg-slate-50 pl-1.5 sm:pl-3 pr-0 py-2 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                  <span>#</span>
                  <span />
                  <span>Team</span>
                  <span className="block w-full text-right pr-2">OVR</span>
                  <span className="block w-full text-right pr-2">ATT</span>
                  <span className="block w-full text-right pr-2">DEF</span>
                </div>
                <div className="divide-y divide-slate-100">
                  {topRatings.map((row, index) => (
                    <div
                      key={`${row.team}-${row.year}`}
                      className="grid grid-cols-[1.5rem_2rem_1fr_repeat(3,3.25rem)] sm:grid-cols-[3rem_2rem_1fr_repeat(3,4rem)] items-stretch pl-1.5 sm:pl-3 pr-0 text-xs text-slate-700"
                    >
                      <div className="flex items-center py-1.5 font-mono tabular-nums text-slate-500">
                        {index + 1}
                      </div>
                      <div className="flex items-center py-1.5">
                        {row.flagPath ? (
                          <span className="relative h-4 w-6 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
                            <Image
                              src={row.flagPath}
                              alt={`${row.team} flag`}
                              fill
                              sizes="24px"
                              className="object-cover"
                            />
                          </span>
                        ) : (
                          <span className="h-4 w-6 rounded-sm bg-slate-100" />
                        )}
                      </div>
                      <div className="flex items-center py-1.5 truncate text-xs font-medium text-slate-900">
                        {row.team}
                      </div>
                      <div
                        className="flex items-center justify-end py-1.5 pr-2 font-mono tabular-nums text-slate-700"
                        style={ratingBackground(row.rating, minRating)}
                      >
                        {ratingFormatter.format(row.rating)}
                      </div>
                      <div
                        className="flex items-center justify-end py-1.5 pr-2 font-mono tabular-nums text-slate-700"
                        style={ratingBackground(row.rating_attack, minRating)}
                      >
                        {ratingFormatter.format(row.rating_attack)}
                      </div>
                      <div
                        className="flex items-center justify-end py-1.5 pr-2 font-mono tabular-nums text-slate-700"
                        style={ratingBackground(row.rating_defense, minRating)}
                      >
                        {ratingFormatter.format(row.rating_defense)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex justify-center">
                <Link
                  className="inline-flex items-center justify-center rounded-md border border-slate-200 bg-white px-6 py-2 text-xs font-semibold uppercase tracking-wide text-slate-600 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 hover:text-slate-700"
                  href="/current-ratings"
                >
                  Show all
                </Link>
              </div>
            </div>
          </div>
          <div className="flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200">
            <div className="flex flex-col gap-4">
              <h2 className="text-lg font-semibold">2026 World Cup Progression Chances</h2>
              <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
                <div className="grid grid-cols-[1.5rem_2rem_1fr_repeat(3,3.5rem)] sm:grid-cols-[3rem_2rem_1fr_repeat(3,4.25rem)] bg-slate-50 pl-1.5 sm:pl-3 pr-0 py-2 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                  <span>#</span>
                  <span />
                  <span>Team</span>
                  {stageColumns.map((col) => (
                    <span key={col} className="block w-full text-right pr-2">
                      {col === "Champion" ? (
                        <>
                          <span className="sm:hidden">Champ.</span>
                          <span className="hidden sm:inline">Champion</span>
                        </>
                      ) : (
                        col.replace("Reach ", "")
                      )}
                    </span>
                  ))}
                </div>
                <div className="divide-y divide-slate-100">
                  {topStageRows.map((row, index) => (
                    <div
                      key={`${row.team}-${index}`}
                      className="grid grid-cols-[1.5rem_2rem_1fr_repeat(3,3.5rem)] sm:grid-cols-[3rem_2rem_1fr_repeat(3,4.25rem)] items-stretch pl-1.5 sm:pl-3 pr-0 text-xs text-slate-700"
                    >
                      <div className="flex items-center py-1.5 font-mono tabular-nums text-slate-500">
                        {index + 1}
                      </div>
                      <div className="flex items-center py-1.5">
                        {row.flagPath ? (
                          <span className="relative h-4 w-6 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
                            <Image
                              src={row.flagPath}
                              alt={`${row.team} flag`}
                              fill
                              sizes="24px"
                              className="object-cover"
                            />
                          </span>
                        ) : (
                          <span className="h-4 w-6 rounded-sm bg-slate-100" />
                        )}
                      </div>
                      <div className="flex items-center py-1.5 truncate text-xs font-medium text-slate-900">
                        {row.team}
                      </div>
                      {stageColumns.map((col) => {
                        const value = Number(row.values[col] ?? 0);
                        const status = row.statuses[col] ?? "U";
                        return (
                          <div
                            key={`${row.team}-${col}`}
                            className="flex items-center justify-end py-1.5 pr-2 font-mono tabular-nums text-slate-700"
                            style={probabilityBackground(value, maxStageProbability)}
                          >
                            {formatProbability(value, status)}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex justify-center">
                <Link
                  className="inline-flex items-center justify-center rounded-md border border-slate-200 bg-white px-6 py-2 text-xs font-semibold uppercase tracking-wide text-slate-600 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 hover:text-slate-700"
                  href="/world-cup-2026/probabilities"
                >
                  Show all
                </Link>
              </div>
            </div>
          </div>
          <Link
            className="group flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200 transition hover:bg-slate-50 sm:flex-row sm:flex-wrap sm:gap-0 sm:items-stretch"
            href="/world-cup-2026/predictor"
          >
            <div className="flex-1 sm:basis-1/2 sm:shrink-0 sm:pr-3">
              <h2 className="text-lg font-semibold">World Cup Bracket Predictor</h2>
              <p className="mt-2 text-sm text-ink-300">
                Simulate paths and pick outcomes.
              </p>
            </div>
            <div className="relative aspect-[21/9] w-full max-h-48 overflow-hidden rounded-lg bg-slate-100 opacity-80 sm:basis-1/2 sm:min-w-[240px] sm:flex-1 sm:self-center">
              <Image
                src="/img/preview-predictor.png"
                alt="Preview of tournament predictor"
                fill
                sizes="(min-width: 768px) 18rem, 100vw"
                className="object-cover object-left-top"
              />
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.45)_0%,rgba(255,255,255,0)_18%,rgba(255,255,255,0)_82%,rgba(255,255,255,0.45)_100%)]" />
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.45)_0%,rgba(255,255,255,0)_18%,rgba(255,255,255,0)_82%,rgba(255,255,255,0.45)_100%)]" />
            </div>
          </Link>
        </section>
      </div>
    </main>
  );
}
