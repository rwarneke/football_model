import Image from "next/image";
import Link from "next/link";

export default function HomePage() {
  return (
    <main className="px-2 pb-16 pt-8 lg:px-6">
      <div className="flex w-full flex-col gap-10">
        <header className="text-center space-y-4">
          <h1 className="text-4xl font-semibold text-ebony md:text-5xl">
            TheBackPost
          </h1>
          <hr className="mx-auto h-px w-24 bg-slate-200" />
        </header>

        <section className="grid gap-4 md:grid-cols-2">
          <Link
            className="group flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200 transition hover:bg-slate-50 sm:flex-row sm:flex-wrap sm:gap-0 sm:items-stretch"
            href="/current-ratings"
          >
            <div className="flex-1 sm:basis-1/2 sm:shrink-0 sm:pr-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-ink-400">
                Ratings
              </p>
              <h2 className="mt-3 text-lg font-semibold">International football team ratings</h2>
              <p className="mt-2 text-sm text-ink-300">
                Snapshot of the latest team strength rankings.
              </p>
            </div>
            <div className="relative aspect-[21/9] w-full max-h-48 overflow-hidden rounded-lg bg-slate-100 opacity-80 sm:basis-1/2 sm:min-w-[240px] sm:flex-1 sm:self-center">
              <Image
                src="/img/preview-current-ratings.png"
                alt="Preview of current ratings"
                fill
                sizes="(min-width: 768px) 18rem, 100vw"
                className="object-cover object-left-top"
              />
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.45)_0%,rgba(255,255,255,0)_18%,rgba(255,255,255,0)_82%,rgba(255,255,255,0.45)_100%)]" />
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.45)_0%,rgba(255,255,255,0)_18%,rgba(255,255,255,0)_82%,rgba(255,255,255,0.45)_100%)]" />
            </div>
          </Link>
          <Link
            className="group flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200 transition hover:bg-slate-50 sm:flex-row sm:flex-wrap sm:gap-0 sm:items-stretch"
            href="/history"
          >
            <div className="flex-1 sm:basis-1/2 sm:shrink-0 sm:pr-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-ink-400">
                Trends
              </p>
              <h2 className="mt-3 text-lg font-semibold">Ratings history</h2>
              <p className="mt-2 text-sm text-ink-300">
                Explore historical ratings since the first ever international match in 1872.
              </p>
            </div>
            <div className="relative aspect-[21/9] w-full max-h-48 overflow-hidden rounded-lg bg-slate-100 opacity-80 sm:basis-1/2 sm:min-w-[240px] sm:flex-1 sm:self-center">
              <Image
                src="/img/preview-historical-ratings.png"
                alt="Preview of ratings history"
                fill
                sizes="(min-width: 768px) 18rem, 100vw"
                className="object-cover object-left-top"
              />
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.45)_0%,rgba(255,255,255,0)_18%,rgba(255,255,255,0)_82%,rgba(255,255,255,0.45)_100%)]" />
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.45)_0%,rgba(255,255,255,0)_18%,rgba(255,255,255,0)_82%,rgba(255,255,255,0.45)_100%)]" />
            </div>
          </Link>
          <Link
            className="group flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200 transition hover:bg-slate-50 sm:flex-row sm:flex-wrap sm:gap-0 sm:items-stretch"
            href="/world-cup-2026/probabilities"
          >
            <div className="flex-1 sm:basis-1/2 sm:shrink-0 sm:pr-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-ink-400">
                World Cup 2026
              </p>
              <h2 className="mt-3 text-lg font-semibold">Stage probabilities</h2>
              <p className="mt-2 text-sm text-ink-300">
                Chances of progressing through each round.
              </p>
            </div>
            <div className="relative aspect-[21/9] w-full max-h-48 overflow-hidden rounded-lg bg-slate-100 opacity-80 sm:basis-1/2 sm:min-w-[240px] sm:flex-1 sm:self-center">
              <Image
                src="/img/preview-probabilities.png"
                alt="Preview of stage probabilities"
                fill
                sizes="(min-width: 768px) 18rem, 100vw"
                className="object-cover object-left-top"
              />
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.45)_0%,rgba(255,255,255,0)_18%,rgba(255,255,255,0)_82%,rgba(255,255,255,0.45)_100%)]" />
              <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.45)_0%,rgba(255,255,255,0)_18%,rgba(255,255,255,0)_82%,rgba(255,255,255,0.45)_100%)]" />
            </div>
          </Link>
          <Link
            className="group flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200 transition hover:bg-slate-50 sm:flex-row sm:flex-wrap sm:gap-0 sm:items-stretch"
            href="/world-cup-2026/predictor"
          >
            <div className="flex-1 sm:basis-1/2 sm:shrink-0 sm:pr-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-ink-400">
                World Cup 2026
              </p>
              <h2 className="mt-3 text-lg font-semibold">Tournament predictor</h2>
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
