import Link from "next/link";

export default function WorldCupHomePage() {
  return (
    <main className="px-3 pb-16 pt-12 md:px-12">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.3em] text-ink-400">
            World Cup 2026
          </p>
          <h1 className="text-4xl font-semibold text-ebony md:text-5xl">
            World Cup 2026 Simulator
          </h1>
          <p className="text-base text-ink-200">
            Explore simulated qualification outcomes, stage odds, and scenario
            summaries for the 2026 tournament.
          </p>
        </header>

        <section className="rounded-md border border-ink-700 bg-white/80 p-6 shadow-soft">
          <h2 className="text-lg font-semibold text-ebony">Pages</h2>
          <div className="mt-4 flex flex-col gap-3 text-sm text-ink-400">
            <Link
              className="rounded border border-ink-700 px-4 py-3 text-ebony transition hover:bg-ink-800/60"
              href="/world-cup-2026/probabilities"
            >
              Team probabilities by stage
            </Link>
            <Link
              className="rounded border border-ink-700 px-4 py-3 text-ebony transition hover:bg-ink-800/60"
              href="/world-cup-2026/predictor"
            >
              Scenario predictor
            </Link>
          </div>
        </section>
      </div>
    </main>
  );
}
