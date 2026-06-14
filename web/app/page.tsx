import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { loadRatings } from "@/lib/ratings";
import {
  formatRatingValue,
  formatTiltValue,
  ratingPillStyle,
  tiltPillStyle,
} from "@/lib/rating-display";
import { loadWorldCupProbabilities } from "@/lib/world-cup";
import { loadWorldCupMatches } from "@/lib/world-cup-matches";
import { loadCompletedWorldCupMatches } from "@/lib/world-cup-results";
import type { WinProbabilities, WinProbabilityEntry } from "@/lib/world-cup-predictor-types";
import {
  isCompactWinProbabilities,
  parseCompactEntry,
  resolveCompactEntry,
} from "@/lib/win-probabilities";

export const metadata: Metadata = {
  title: "TheBackPost Football Analytics",
};

const percentFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const ACCENT_LIGHT_RGB = "147, 197, 253";
const HOST_TEAM_COUNTRIES: Record<string, string> = {
  USA: "USA",
  "United States": "USA",
  Canada: "Canada",
  Mexico: "Mexico",
};
const HOST_TEAMS = new Set(["USA", "Canada", "Mexico"]);

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

function shouldUseDecimalPrecision(values: (number | null | undefined)[]) {
  return values.some((v) => v !== null && v !== undefined && Number.isFinite(v) && v * 100 < 0.5);
}

function formatPercent(value: number | null | undefined, forceDecimal = false) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  const percent = value * 100;
  if (forceDecimal || percent < 0.5 || percent >= 99.5) {
    if (percent > 0 && percent < 0.05) {
      return "<0.1%";
    }
    const rounded = Number(percent.toFixed(1));
    const capped = Math.min(rounded, 99.9);
    return `${capped.toFixed(1)}%`;
  }
  return `${Math.round(percent)}%`;
}

function formatMatchProbability(value: number | null, forceDecimal = false) {
  return formatPercent(value, forceDecimal);
}

function normalizeCountry(value: string | null | undefined) {
  return value ? value.trim().toLowerCase() : "";
}

function isPlaceholderLabel(name: string) {
  const trimmed = name.trim();
  if (!trimmed) {
    return true;
  }
  return (
    /^Winner\b/i.test(trimmed) ||
    /^Runner-up\b/i.test(trimmed) ||
    /^3rd\b/i.test(trimmed) ||
    /^3rd Group\b/i.test(trimmed) ||
    /^Loser\b/i.test(trimmed) ||
    /winner$/i.test(trimmed)
  );
}

function resolveMatchNeutrality({
  homeTeam,
  awayTeam,
  country,
  neutralOverride,
}: {
  homeTeam: string;
  awayTeam: string;
  country?: string | null;
  neutralOverride?: boolean | null;
}) {
  if (neutralOverride !== undefined && neutralOverride !== null) {
    const neutral = Boolean(neutralOverride);
    return { neutral, advantage: neutral ? null : ("home" as const) };
  }
  let neutral = true;
  let advantage: "home" | "away" | null = null;
  if (country) {
    const matchCountry = normalizeCountry(country);
    const homeCountry = normalizeCountry(HOST_TEAM_COUNTRIES[homeTeam]);
    const awayCountry = normalizeCountry(HOST_TEAM_COUNTRIES[awayTeam]);
    const homeAdvantage = homeCountry && matchCountry && homeCountry === matchCountry;
    const awayAdvantage = awayCountry && matchCountry && awayCountry === matchCountry;
    if (homeAdvantage && awayAdvantage) {
      neutral = true;
    } else if (homeAdvantage || awayAdvantage) {
      neutral = false;
      advantage = homeAdvantage ? "home" : "away";
    }
  } else {
    const homeIsHost = HOST_TEAMS.has(homeTeam);
    const awayIsHost = HOST_TEAMS.has(awayTeam);
    if (homeIsHost !== awayIsHost) {
      neutral = false;
      advantage = homeIsHost ? "home" : "away";
    }
  }
  return { neutral, advantage };
}

function resolveProbabilityEntry({
  probabilities,
  homeTeam,
  awayTeam,
  country,
  neutralOverride,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  country?: string | null;
  neutralOverride?: boolean | null;
}): { entry: WinProbabilityEntry; flipped: boolean } | null {
  const { neutral, advantage } = resolveMatchNeutrality({
    homeTeam,
    awayTeam,
    country,
    neutralOverride,
  });
  if (isCompactWinProbabilities(probabilities)) {
    if (neutral) {
      const entry = resolveCompactEntry(probabilities, homeTeam, awayTeam, true);
      return entry ? { entry: parseCompactEntry(entry), flipped: false } : null;
    }
    if (advantage === "home") {
      const entry = resolveCompactEntry(probabilities, homeTeam, awayTeam, false);
      return entry ? { entry: parseCompactEntry(entry), flipped: false } : null;
    }
    if (advantage === "away") {
      const entry = resolveCompactEntry(probabilities, awayTeam, homeTeam, false);
      return entry ? { entry: parseCompactEntry(entry), flipped: true } : null;
    }
    return null;
  }

  if (neutral) {
    const entry = probabilities[homeTeam]?.[awayTeam]?.neutral;
    return entry ? { entry, flipped: false } : null;
  }
  if (advantage === "home") {
    const entry = probabilities[homeTeam]?.[awayTeam]?.home;
    return entry ? { entry, flipped: false } : null;
  }
  if (advantage === "away") {
    const entry = probabilities[awayTeam]?.[homeTeam]?.home;
    return entry ? { entry, flipped: true } : null;
  }
  return null;
}

function resolveMatchProbabilities({
  probabilities,
  homeTeam,
  awayTeam,
  country,
  neutralOverride,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  country?: string | null;
  neutralOverride?: boolean | null;
}) {
  if (!probabilities || isPlaceholderLabel(homeTeam) || isPlaceholderLabel(awayTeam)) {
    return null;
  }
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam,
    awayTeam,
    country,
    neutralOverride,
  });
  if (!resolved) {
    return null;
  }
  const values = {
    home: resolved.entry.p_home ?? null,
    draw: resolved.entry.p_draw ?? null,
    away: resolved.entry.p_away ?? null,
  };
  if (!resolved.flipped) {
    return values;
  }
  return {
    home: values.away,
    draw: values.draw,
    away: values.home,
  };
}

async function loadWinProbabilities(): Promise<WinProbabilities> {
  const filePath = path.join(process.cwd(), "public", "model_output", "win_probabilities.json");
  const contents = await readFile(filePath, "utf8");
  return JSON.parse(contents) as WinProbabilities;
}

export default async function HomePage() {
  const topRatings = (await loadRatings()).slice(0, 5);
  const worldCup = await loadWorldCupProbabilities();
  const matches = await loadWorldCupMatches();
  const completedMatches = await loadCompletedWorldCupMatches();
  const winProbabilities = await loadWinProbabilities();
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

  const completedMatchIds = new Set(completedMatches.map((match) => String(match.matchId)));
  const upcomingMatches = matches
    .filter((match) => {
      if (completedMatchIds.has(String(match.id))) {
        return false;
      }
      return !isPlaceholderLabel(match.home) && !isPlaceholderLabel(match.away);
    })
    .slice(0, 20);
  return (
    <main className="px-2 pb-16 pt-0 lg:px-6">
      <div className="flex w-full flex-col gap-10">
        <section className="flex flex-col gap-4">
          <div className="flex items-center justify-end" />
          <div className="flex gap-4 overflow-x-auto pb-2 pt-1">
            {upcomingMatches.map((match) => {
              const homePlaceholder = isPlaceholderLabel(match.home);
              const awayPlaceholder = isPlaceholderLabel(match.away);
              const displayHome = homePlaceholder ? "TBD" : match.home;
              const displayAway = awayPlaceholder ? "TBD" : match.away;
              const values = resolveMatchProbabilities({
                probabilities: winProbabilities,
                homeTeam: match.home,
                awayTeam: match.away,
                country: match.country,
                neutralOverride: match.neutral ?? null,
              });
              const useDecimal = shouldUseDecimalPrecision([
                values?.home,
                values?.draw,
                values?.away,
              ]);
              const homePercent = Math.max(0, Math.min(100, (values?.home ?? 0) * 100));
              const drawPercent = Math.max(0, Math.min(100, (values?.draw ?? 0) * 100));
              const awayPercent = Math.max(0, Math.min(100, (values?.away ?? 0) * 100));
              const matchDate = new Date(`${match.date}T00:00:00`);
              const dateLabel = Number.isNaN(matchDate.getTime())
                ? match.date
                : matchDate.toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                  });
              return (
                <Link
                  key={`${match.id}-${match.home}-${match.away}`}
                  href={`/world-cup-2026/matches#match-${match.id}`}
                  className="min-w-[200px] max-w-[220px] shrink-0 rounded-xl bg-white px-2.5 py-2 shadow-sm ring-1 ring-slate-200 transition hover:ring-slate-300 first:ml-1"
                >
                  <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-slate-500">
                    <span>{dateLabel}</span>
                    <span>{match.stage.toUpperCase()}</span>
                  </div>
                  <div className="mt-1 space-y-1 text-xs font-medium text-slate-900">
                    <div className="flex items-center gap-1">
                      {homePlaceholder ? (
                        <span className="flex h-4 w-6 shrink-0 items-center justify-center rounded-[1px] border border-slate-300 bg-slate-200" />
                      ) : (
                        <span className="relative h-3.5 w-5 shrink-0 overflow-hidden rounded-[2px] shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
                          <Image
                            src={`/flags/${match.home.replace(/ /g, "_")}.png`}
                            alt={`${match.home} flag`}
                            fill
                            className="object-cover"
                            sizes="20px"
                          />
                        </span>
                      )}
                      <span className="truncate">{displayHome}</span>
                    </div>
                    <div className="flex items-center gap-1 justify-end text-right">
                      <span className="truncate">{displayAway}</span>
                      {awayPlaceholder ? (
                        <span className="flex h-4 w-6 shrink-0 items-center justify-center rounded-[1px] border border-slate-300 bg-slate-200" />
                      ) : (
                        <span className="relative h-3.5 w-5 shrink-0 overflow-hidden rounded-[2px] shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
                          <Image
                            src={`/flags/${match.away.replace(/ /g, "_")}.png`}
                            alt={`${match.away} flag`}
                            fill
                            className="object-cover"
                            sizes="20px"
                          />
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="mt-1.5 space-y-1">
                    <div className="flex items-center justify-between text-[10px] text-slate-600 tabular-nums">
                      <span>{formatMatchProbability(values?.home ?? null, useDecimal)}</span>
                      <span>{formatMatchProbability(values?.draw ?? null, useDecimal)}</span>
                      <span>{formatMatchProbability(values?.away ?? null, useDecimal)}</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                      <div className="flex h-full">
                        <div
                          className="h-full bg-emerald-300/70"
                          style={{ width: `${homePercent}%` }}
                        />
                        <div
                          className="h-full bg-slate-300/70"
                          style={{ width: `${drawPercent}%` }}
                        />
                        <div
                          className="h-full bg-rose-300/70"
                          style={{ width: `${awayPercent}%` }}
                        />
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-[9px] uppercase tracking-wide text-slate-500">
                      <span>Win</span>
                      <span>Draw</span>
                      <span>Win</span>
                    </div>
                  </div>
                </Link>
              );
            })}
            <div className="flex shrink-0 items-center pr-1">
              <Link
                className="inline-flex items-center justify-center rounded-md border border-slate-200 bg-white px-5 py-2 text-xs font-semibold uppercase tracking-wide text-slate-600 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 hover:text-slate-700"
                href="/world-cup-2026/matches"
              >
                View all matches
              </Link>
            </div>
          </div>
        </section>
        <section className="grid gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200">
            <div className="flex flex-col gap-4">
              <h2 className="text-lg font-semibold">International football team ratings</h2>
              <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
                <div className="grid grid-cols-[1.5rem_2rem_1fr_repeat(2,4.5rem)] lg:grid-cols-[3rem_2rem_1fr_repeat(2,5.25rem)] bg-slate-50 pl-1.5 lg:pl-3 pr-0 py-2 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                  <span>#</span>
                  <span />
                  <span>Team</span>
                  <span className="block w-full text-center">Rating</span>
                  <span className="block w-full text-center">Tilt</span>
                </div>
                <div className="divide-y divide-slate-100">
                  {topRatings.map((row, index) => (
                    <div
                      key={`${row.team}-${row.year}`}
                      className="grid grid-cols-[1.5rem_2rem_1fr_repeat(2,4.5rem)] lg:grid-cols-[3rem_2rem_1fr_repeat(2,5.25rem)] items-stretch pl-1.5 lg:pl-3 pr-0 text-xs text-slate-700"
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
                      <div className="flex items-center justify-center py-1.5">
                        <span
                          className="inline-flex min-w-[3.95rem] items-center justify-center rounded-full border px-1.5 py-1 text-[10px] font-mono font-semibold leading-none tabular-nums text-slate-700"
                          style={ratingPillStyle(row.rating)}
                        >
                          {formatRatingValue(row.rating)}
                        </span>
                      </div>
                      <div className="flex items-center justify-center py-1.5">
                        <span
                          className="inline-flex min-w-[3.95rem] items-center justify-center rounded-full border px-1.5 py-1 text-[10px] font-mono font-semibold leading-none tabular-nums text-slate-700"
                          style={tiltPillStyle(row.tilt)}
                        >
                          {formatTiltValue(row.tilt)}
                        </span>
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
                <div className="grid grid-cols-[1.5rem_2rem_1fr_repeat(3,3.5rem)] lg:grid-cols-[3rem_2rem_1fr_repeat(3,4.25rem)] bg-slate-50 pl-1.5 lg:pl-3 pr-0 py-2 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
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
                      className="grid grid-cols-[1.5rem_2rem_1fr_repeat(3,3.5rem)] lg:grid-cols-[3rem_2rem_1fr_repeat(3,4.25rem)] items-stretch pl-1.5 lg:pl-3 pr-0 text-xs text-slate-700"
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
          <Link
            className="group flex flex-col gap-4 rounded-xl bg-white p-5 text-ebony shadow-sm ring-1 ring-slate-200 transition hover:bg-slate-50 sm:flex-row sm:flex-wrap sm:gap-0 sm:items-stretch"
            href="/matchup"
          >
            <div className="flex-1 sm:basis-1/2 sm:shrink-0 sm:pr-3">
              <h2 className="text-lg font-semibold">Compare Teams</h2>
            </div>
            <div className="relative aspect-[21/9] w-full max-h-48 overflow-hidden rounded-lg bg-slate-100 opacity-80 sm:basis-1/2 sm:min-w-[240px] sm:flex-1 sm:self-center">
              <Image
                src="/img/preview-matchup.png"
                alt="Preview of arbitrary matchup finder"
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
