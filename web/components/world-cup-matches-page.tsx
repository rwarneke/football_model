"use client";

import * as React from "react";
import Image from "next/image";
import type { WorldCupMatch } from "@/lib/world-cup-matches";
import type { WinProbabilities, WinProbabilityEntry } from "@/lib/world-cup-predictor-types";
import { buildScoreMatrix } from "@/lib/score-matrix";
import {
  isCompactWinProbabilities,
  parseCompactEntry,
  resolveCompactEntry,
} from "@/lib/win-probabilities";

type MatchProbabilityValues = {
  home: number | null;
  draw: number | null;
  away: number | null;
};

const HOST_TEAM_COUNTRIES: Record<string, string> = {
  USA: "USA",
  "United States": "USA",
  Canada: "Canada",
  Mexico: "Mexico",
};
const HOST_TEAMS = new Set(["USA", "Canada", "Mexico"]);

const SCORE_LABELS = ["0", "1", "2", "3", "4", "5+"];
const PROBABILITY_HIGHLIGHT_RGB = "147, 197, 253";
const PROBABILITY_HIGHLIGHT_MAX_ALPHA = 0.98;

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

function selectProbabilityValues(
  entry: WinProbabilityEntry | undefined,
  allowDraw: boolean
): MatchProbabilityValues | null {
  if (!entry) {
    return null;
  }
  if (allowDraw) {
    return {
      home: entry.p_home ?? null,
      draw: entry.p_draw ?? null,
      away: entry.p_away ?? null,
    };
  }
  return {
    home: entry.p_home_pens ?? null,
    draw: null,
    away: entry.p_away_pens ?? null,
  };
}

function resolveMatchProbabilities({
  probabilities,
  homeTeam,
  awayTeam,
  allowDraw,
  country,
  neutralOverride,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  allowDraw: boolean;
  country?: string | null;
  neutralOverride?: boolean | null;
}): MatchProbabilityValues | null {
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
  const values = selectProbabilityValues(resolved.entry, allowDraw);
  if (!values) {
    return null;
  }
  if (!resolved.flipped) {
    return values;
  }
  return {
    home: values.away,
    draw: values.draw,
    away: values.home,
  };
}

function transposeScoreMatrix(matrix: number[][]) {
  const rows = matrix.length;
  const cols = matrix.reduce((max, row) => Math.max(max, row.length), 0);
  const transposed = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < matrix[i].length; j += 1) {
      const value = matrix[i][j];
      transposed[j][i] = Number.isFinite(value) ? value : 0;
    }
  }
  return transposed;
}

function resolveMatchScoreMatrix({
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
}): number[][] | null {
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
  if (resolved.entry.score_matrix) {
    return resolved.flipped
      ? transposeScoreMatrix(resolved.entry.score_matrix)
      : resolved.entry.score_matrix;
  }
  if (
    resolved.entry.nu === undefined ||
    resolved.entry.lam_home === undefined ||
    resolved.entry.lam_away === undefined
  ) {
    return null;
  }
  const maxGoals = isCompactWinProbabilities(probabilities)
    ? probabilities.max_goals ?? 8
    : 8;
  const matrix = buildScoreMatrix({
    nu: resolved.entry.nu,
    lamH: resolved.entry.lam_home,
    lamA: resolved.entry.lam_away,
    maxGoals,
  });
  return resolved.flipped ? transposeScoreMatrix(matrix) : matrix;
}

function sumScoreMatrix(matrix: number[][]) {
  let total = 0;
  for (const row of matrix) {
    for (const value of row) {
      total += Number(value ?? 0);
    }
  }
  return total;
}

function normalizeScoreMatrix(matrix: number[][]) {
  const total = sumScoreMatrix(matrix);
  if (total <= 0) {
    return matrix.map((row) => row.map(() => 0));
  }
  return matrix.map((row) => row.map((value) => Number(value ?? 0) / total));
}

function buildScoreGrid(matrix: number[][]) {
  const normalized = normalizeScoreMatrix(matrix);
  const rowCount = normalized.length;
  const colCount = normalized.reduce((max, row) => Math.max(max, row.length), 0);
  const maxIndex = Math.max(rowCount, colCount) - 1;

  const rows = 6;
  const cols = 6;
  const values: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const rowRange = r === 5 ? [5, maxIndex] : [r, r];
      const colRange = c === 5 ? [5, maxIndex] : [c, c];
      let sum = 0;
      for (let i = rowRange[0]; i <= rowRange[1]; i += 1) {
        const row = normalized[i];
        if (!row) {
          continue;
        }
        for (let j = colRange[0]; j <= colRange[1]; j += 1) {
          sum += Number(row[j] ?? 0);
        }
      }
      values[r][c] = sum;
    }
  }

  return values;
}

function buildMarginRow(matrix: number[][]) {
  const normalized = normalizeScoreMatrix(matrix);
  const buckets = {
    home3: 0,
    home2: 0,
    home1: 0,
    draw: 0,
    away1: 0,
    away2: 0,
    away3: 0,
  };

  for (let homeGoals = 0; homeGoals < normalized.length; homeGoals += 1) {
    const row = normalized[homeGoals] ?? [];
    for (let awayGoals = 0; awayGoals < row.length; awayGoals += 1) {
      const value = Number(row[awayGoals] ?? 0);
      const diff = homeGoals - awayGoals;
      if (diff >= 3) {
        buckets.home3 += value;
      } else if (diff === 2) {
        buckets.home2 += value;
      } else if (diff === 1) {
        buckets.home1 += value;
      } else if (diff === 0) {
        buckets.draw += value;
      } else if (diff === -1) {
        buckets.away1 += value;
      } else if (diff === -2) {
        buckets.away2 += value;
      } else if (diff <= -3) {
        buckets.away3 += value;
      }
    }
  }

  return [
    { label: "3+", value: buckets.home3 },
    { label: "2", value: buckets.home2 },
    { label: "1", value: buckets.home1 },
    { label: "0", value: buckets.draw },
    { label: "1", value: buckets.away1 },
    { label: "2", value: buckets.away2 },
    { label: "3+", value: buckets.away3 },
  ];
}

function scoreMatrixHighlight(value: number) {
  if (!Number.isFinite(value)) {
    return undefined;
  }
  const clamped = Math.max(0, Math.min(value, 1));
  let alpha = 0;
  if (clamped <= 0.01) {
    const scaled = clamped / 0.01;
    alpha = scaled * (PROBABILITY_HIGHLIGHT_MAX_ALPHA * 0.5);
  } else if (clamped <= 0.2) {
    const scaled = (clamped - 0.01) / 0.19;
    alpha =
      PROBABILITY_HIGHLIGHT_MAX_ALPHA * 0.5 +
      scaled * (PROBABILITY_HIGHLIGHT_MAX_ALPHA * 0.5);
  } else {
    alpha = PROBABILITY_HIGHLIGHT_MAX_ALPHA;
  }
  return { backgroundColor: `rgba(${PROBABILITY_HIGHLIGHT_RGB}, ${alpha})` };
}

function shouldUseDecimalPrecision(values: (number | null | undefined)[]) {
  return values.some((v) => v !== null && v !== undefined && Number.isFinite(v) && v * 100 < 0.5);
}

function formatPercent(value: number | null | undefined, forceDecimal = false) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  const percent = value * 100;
  if (percent < 0.1) {
    return "<0.1%";
  }
  if (percent > 99.9) {
    return ">99.9%";
  }
  if (forceDecimal || percent < 0.5 || percent >= 99.5) {
    const rounded = Number(percent.toFixed(1));
    const capped = Math.min(rounded, 99.9);
    return `${capped.toFixed(1)}%`;
  }
  return `${Math.round(percent)}%`;
}

function formatDecimalOdds(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  if (value <= 0) {
    return ">$1000";
  }
  const odds = 1 / value;
  if (odds > 1000) {
    return ">1000";
  }
  if (odds < 1.001) {
    return "<1.001";
  }
  let fractionDigits = 0;
  if (odds < 1.0095) {
    fractionDigits = 3;
  } else if (odds < 10) {
    fractionDigits = 2;
  } else if (odds < 100) {
    fractionDigits = 1;
  }
  return odds.toFixed(fractionDigits);
}

function formatProbabilityLabel(
  value: number | null | undefined,
  mode: "percent" | "decimal",
  forceDecimal = false
) {
  return mode === "decimal" ? formatDecimalOdds(value) : formatPercent(value, forceDecimal);
}

function parseProbabilityLabel(label?: string | null) {
  if (!label) {
    return null;
  }
  if (label === "<0.1%") {
    return 0.05;
  }
  const match = label.match(/(\d+(?:\.\d+)?)%/);
  if (!match) {
    return null;
  }
  const value = Number(match[1]);
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.max(0, Math.min(100, value));
}

function formatNormalizedPercent(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  if (value < 0.1) {
    return "<0.1%";
  }
  if (value > 99.9) {
    return ">99.9%";
  }
  if (value !== Math.round(value)) {
    return `${value.toFixed(1)}%`;
  }
  return `${Math.round(value)}%`;
}

function normalizeProbabilitySegments(values: {
  home: number | null;
  draw: number | null;
  away: number | null;
}) {
  const { home, draw, away } = values;
  if (home === null || draw === null || away === null) {
    return null;
  }
  const raw = [home, draw, away];
  const hasDecimal = raw.some((v) => v !== Math.round(v));
  if (hasDecimal) {
    const rounded = raw.map((value) => Number(value.toFixed(1)));
    const total = rounded.reduce((sum, value) => sum + value, 0);
    const remainder = Number((100 - total).toFixed(1));
    if (Math.abs(remainder) >= 0.05) {
      rounded[1] = Math.max(0, Number((rounded[1] + remainder).toFixed(1)));
    }
    return { home: rounded[0], draw: rounded[1], away: rounded[2] };
  }
  const rounded = raw.map((value) => Math.round(value));
  const total = rounded.reduce((sum, value) => sum + value, 0);
  const remainder = 100 - total;
  if (remainder !== 0) {
    rounded[1] = Math.max(0, rounded[1] + remainder);
  }
  return { home: rounded[0], draw: rounded[1], away: rounded[2] };
}

function normalizeTwoSegments(values: { home: number | null; away: number | null }) {
  const { home, away } = values;
  if (home === null || away === null) {
    return null;
  }
  const hasDecimal = home !== Math.round(home) || away !== Math.round(away);
  if (hasDecimal) {
    const roundedHome = Number(home.toFixed(1));
    const roundedAway = Number(away.toFixed(1));
    const remainder = Number((100 - roundedHome - roundedAway).toFixed(1));
    return {
      home: Math.max(0, Number((roundedHome + remainder).toFixed(1))),
      away: roundedAway,
    };
  }
  const roundedHome = Math.round(home);
  const roundedAway = Math.round(away);
  const remainder = 100 - (roundedHome + roundedAway);
  return {
    home: Math.max(0, roundedHome + remainder),
    away: roundedAway,
  };
}

function formatDateHeading(date: string) {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) {
    return date;
  }
  return parsed.toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

export function WorldCupMatchesPageClient({
  matches,
  winProbabilities,
}: {
  matches: WorldCupMatch[];
  winProbabilities: WinProbabilities;
}) {
  const [probabilityMode, setProbabilityMode] = React.useState<"percent" | "decimal">("percent");
  const [query, setQuery] = React.useState("");

  const filteredMatches = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return matches;
    }
    return matches.filter((match) => {
      const home = match.home.toLowerCase();
      const away = match.away.toLowerCase();
      return home.includes(normalized) || away.includes(normalized);
    });
  }, [matches, query]);

  const grouped = React.useMemo(() => {
    return filteredMatches.reduce<Record<string, WorldCupMatch[]>>((acc, match) => {
      acc[match.date] = acc[match.date] ?? [];
      acc[match.date].push(match);
      return acc;
    }, {});
  }, [filteredMatches]);

  const dates = React.useMemo(
    () => Object.keys(grouped).sort((a, b) => a.localeCompare(b)),
    [grouped]
  );

  return (
    <div className="space-y-6">
      <div className="flex w-full items-center gap-3">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search teams"
          className="min-w-0 w-full max-w-[25rem] flex-1 rounded-md bg-white px-3 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 md:w-64"
        />
        <div className="ml-auto flex w-40 shrink-0 items-center gap-2">
          <select
            value={probabilityMode}
            onChange={(event) =>
              setProbabilityMode(event.target.value as "percent" | "decimal")
            }
            className="w-full rounded-md bg-white px-2.5 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
          >
            <option value="percent">% Chance</option>
            <option value="decimal">Decimal Odds</option>
          </select>
        </div>
      </div>

      <div className="space-y-6">
        <div className="text-2xl font-semibold text-ebony md:text-3xl">
          Upcoming matches
        </div>
        {dates.map((date) => (
          <section key={date} className="space-y-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              {formatDateHeading(date)}
            </h2>
            <div className="grid gap-4 grid-cols-1 sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
              {grouped[date].map((match) => {
                const isQualifier = match.id.startsWith("Q-");
                const locationWidth = isQualifier ? "w-[45%]" : "w-[60%]";
                const stageWidth = isQualifier ? "w-[45%]" : "w-[40%]";
                const allowDraw = Boolean(match.group);
                const requiresResult = !allowDraw;
                const homePlaceholder = isPlaceholderLabel(match.home);
                const awayPlaceholder = isPlaceholderLabel(match.away);
                const displayHome = homePlaceholder ? "TBD" : match.home;
                const displayAway = awayPlaceholder ? "TBD" : match.away;
                const ninetyValues = resolveMatchProbabilities({
                  probabilities: winProbabilities,
                  homeTeam: match.home,
                  awayTeam: match.away,
                  allowDraw: true,
                  country: match.country,
                  neutralOverride: match.neutral ?? null,
                });
                const fullTimeValues = resolveMatchProbabilities({
                  probabilities: winProbabilities,
                  homeTeam: match.home,
                  awayTeam: match.away,
                  allowDraw: false,
                  country: match.country,
                  neutralOverride: match.neutral ?? null,
                });
                const shownValues = requiresResult ? fullTimeValues : ninetyValues;
                const useDecimal = shouldUseDecimalPrecision([
                  shownValues?.home,
                  shownValues?.draw,
                  shownValues?.away,
                ]);
                const homeLabelRaw = formatProbabilityLabel(
                  shownValues?.home ?? null,
                  probabilityMode,
                  useDecimal
                );
                const drawLabelRaw = allowDraw
                  ? formatProbabilityLabel(shownValues?.draw ?? null, probabilityMode, useDecimal)
                  : null;
                const awayLabelRaw = formatProbabilityLabel(
                  shownValues?.away ?? null,
                  probabilityMode,
                  useDecimal
                );
                const normalizedShown = probabilityMode === "percent" && allowDraw
                  ? normalizeProbabilitySegments({
                      home:
                        shownValues?.home !== null && shownValues?.home !== undefined
                          ? shownValues.home * 100
                          : null,
                      draw:
                        shownValues?.draw !== null && shownValues?.draw !== undefined
                          ? shownValues.draw * 100
                          : null,
                      away:
                        shownValues?.away !== null && shownValues?.away !== undefined
                          ? shownValues.away * 100
                          : null,
                    })
                  : null;
                const homeLabel =
                  probabilityMode === "percent" && normalizedShown
                    ? formatNormalizedPercent(normalizedShown.home)
                    : homeLabelRaw;
                const drawLabel =
                  allowDraw && probabilityMode === "percent" && normalizedShown
                    ? formatNormalizedPercent(normalizedShown.draw)
                    : allowDraw
                      ? drawLabelRaw
                      : null;
                const awayLabel =
                  probabilityMode === "percent" && normalizedShown
                    ? formatNormalizedPercent(normalizedShown.away)
                    : awayLabelRaw;
                const homePercent =
                  probabilityMode === "percent" && normalizedShown
                    ? Math.max(0, Math.min(100, normalizedShown.home))
                    : Math.max(0, Math.min(100, (shownValues?.home ?? 0) * 100));
                const drawPercent = allowDraw
                  ? probabilityMode === "percent" && normalizedShown
                    ? Math.max(0, Math.min(100, normalizedShown.draw))
                    : Math.max(0, Math.min(100, (shownValues?.draw ?? 0) * 100))
                  : 0;
                const awayPercent =
                  probabilityMode === "percent" && normalizedShown
                    ? Math.max(0, Math.min(100, normalizedShown.away))
                    : Math.max(0, Math.min(100, (shownValues?.away ?? 0) * 100));
                const ninetyUseDecimal = shouldUseDecimalPrecision([
                  ninetyValues?.home,
                  ninetyValues?.draw,
                  ninetyValues?.away,
                ]);
                const fullTimeUseDecimal = shouldUseDecimalPrecision([
                  fullTimeValues?.home,
                  fullTimeValues?.away,
                ]);
                const ninetyHomeLabelRaw = formatProbabilityLabel(
                  ninetyValues?.home ?? null,
                  probabilityMode,
                  ninetyUseDecimal
                );
                const ninetyDrawLabelRaw = formatProbabilityLabel(
                  ninetyValues?.draw ?? null,
                  probabilityMode,
                  ninetyUseDecimal
                );
                const ninetyAwayLabelRaw = formatProbabilityLabel(
                  ninetyValues?.away ?? null,
                  probabilityMode,
                  ninetyUseDecimal
                );
                const fullTimeHomeLabelRaw = formatProbabilityLabel(
                  fullTimeValues?.home ?? null,
                  probabilityMode,
                  fullTimeUseDecimal
                );
                const fullTimeAwayLabelRaw = formatProbabilityLabel(
                  fullTimeValues?.away ?? null,
                  probabilityMode,
                  fullTimeUseDecimal
                );
                const normalizedNinety = probabilityMode === "percent"
                  ? normalizeProbabilitySegments({
                      home:
                        ninetyValues?.home !== null && ninetyValues?.home !== undefined
                          ? ninetyValues.home * 100
                          : null,
                      draw:
                        ninetyValues?.draw !== null && ninetyValues?.draw !== undefined
                          ? ninetyValues.draw * 100
                          : null,
                      away:
                        ninetyValues?.away !== null && ninetyValues?.away !== undefined
                          ? ninetyValues.away * 100
                          : null,
                    })
                  : null;
                const normalizedFullTime = probabilityMode === "percent"
                  ? normalizeTwoSegments({
                      home:
                        fullTimeValues?.home !== null && fullTimeValues?.home !== undefined
                          ? fullTimeValues.home * 100
                          : null,
                      away:
                        fullTimeValues?.away !== null && fullTimeValues?.away !== undefined
                          ? fullTimeValues.away * 100
                          : null,
                    })
                  : null;
                const ninetyHomeLabel =
                  probabilityMode === "percent" && normalizedNinety
                    ? formatNormalizedPercent(normalizedNinety.home)
                    : ninetyHomeLabelRaw;
                const ninetyDrawLabel =
                  probabilityMode === "percent" && normalizedNinety
                    ? formatNormalizedPercent(normalizedNinety.draw)
                    : ninetyDrawLabelRaw;
                const ninetyAwayLabel =
                  probabilityMode === "percent" && normalizedNinety
                    ? formatNormalizedPercent(normalizedNinety.away)
                    : ninetyAwayLabelRaw;
                const fullTimeHomeLabel =
                  probabilityMode === "percent" && normalizedFullTime
                    ? formatNormalizedPercent(normalizedFullTime.home)
                    : fullTimeHomeLabelRaw;
                const fullTimeAwayLabel =
                  probabilityMode === "percent" && normalizedFullTime
                    ? formatNormalizedPercent(normalizedFullTime.away)
                    : fullTimeAwayLabelRaw;
                const ninetyHomePercent = probabilityMode === "percent" && normalizedNinety
                  ? Math.max(0, Math.min(100, normalizedNinety.home))
                  : Math.max(0, Math.min(100, (ninetyValues?.home ?? 0) * 100));
                const ninetyDrawPercent = probabilityMode === "percent" && normalizedNinety
                  ? Math.max(0, Math.min(100, normalizedNinety.draw))
                  : Math.max(0, Math.min(100, (ninetyValues?.draw ?? 0) * 100));
                const ninetyAwayPercent = probabilityMode === "percent" && normalizedNinety
                  ? Math.max(0, Math.min(100, normalizedNinety.away))
                  : Math.max(0, Math.min(100, (ninetyValues?.away ?? 0) * 100));
                const fullTimeHomePercent = probabilityMode === "percent" && normalizedFullTime
                  ? Math.max(0, Math.min(100, normalizedFullTime.home))
                  : Math.max(0, Math.min(100, (fullTimeValues?.home ?? 0) * 100));
                const fullTimeAwayPercent = probabilityMode === "percent" && normalizedFullTime
                  ? Math.max(0, Math.min(100, normalizedFullTime.away))
                  : Math.max(0, Math.min(100, (fullTimeValues?.away ?? 0) * 100));
                const scoreMatrix = resolveMatchScoreMatrix({
                  probabilities: winProbabilities,
                  homeTeam: match.home,
                  awayTeam: match.away,
                  country: match.country,
                  neutralOverride: match.neutral ?? null,
                });
                const scoreGrid = scoreMatrix ? buildScoreGrid(scoreMatrix) : null;
                const marginRow = scoreMatrix ? buildMarginRow(scoreMatrix) : null;

                return (
                  <div
                    key={`${match.id}-${match.home}-${match.away}`}
                    id={`match-${match.id}`}
                    className="min-w-0 scroll-mt-24"
                  >
                    <div className="h-full rounded-xl bg-white ring-1 ring-slate-200 shadow-sm px-4 py-3 flex flex-col">
                      <div className="space-y-1">
                        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 text-base font-semibold text-slate-900">
                          <span className="flex items-center gap-2 min-w-0">
                            {homePlaceholder ? (
                              <span className="flex h-4 w-6 shrink-0 items-center justify-center rounded-[1px] border border-slate-300 bg-slate-200" />
                            ) : (
                              <span className="relative h-4 w-6 shrink-0 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
                                <Image
                                  src={`/flags/${match.home.replace(/ /g, "_")}.png`}
                                  alt={`${match.home} flag`}
                                  fill
                                  className="object-cover"
                                  sizes="24px"
                                />
                              </span>
                            )}
                            <span className="whitespace-normal break-words">{displayHome}</span>
                          </span>
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                            vs.
                          </span>
                          <span className="flex items-center gap-2 min-w-0 justify-end text-right">
                            <span className="whitespace-normal break-words">{displayAway}</span>
                            {awayPlaceholder ? (
                              <span className="flex h-4 w-6 shrink-0 items-center justify-center rounded-[1px] border border-slate-300 bg-slate-200" />
                            ) : (
                              <span className="relative h-4 w-6 shrink-0 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
                                <Image
                                  src={`/flags/${match.away.replace(/ /g, "_")}.png`}
                                  alt={`${match.away} flag`}
                                  fill
                                  className="object-cover"
                                  sizes="24px"
                                />
                              </span>
                            )}
                          </span>
                        </div>
                      </div>
                      {requiresResult ? (
                        <div className="mt-3 space-y-3">
                          <div>
                            <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-slate-500">
                              <span>After 90</span>
                              <span>Win / Draw / Win</span>
                            </div>
                            <div className="mt-1 flex items-center justify-between text-[11px] text-slate-600 tabular-nums">
                              <span>{ninetyHomeLabel}</span>
                              <span>{ninetyDrawLabel}</span>
                              <span>{ninetyAwayLabel}</span>
                            </div>
                            <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                              <div className="flex h-full">
                                <div
                                  className="h-full bg-emerald-300/70"
                                  style={{ width: `${ninetyHomePercent}%` }}
                                />
                                <div
                                  className="h-full bg-slate-300/70"
                                  style={{ width: `${ninetyDrawPercent}%` }}
                                />
                                <div
                                  className="h-full bg-rose-300/70"
                                  style={{ width: `${ninetyAwayPercent}%` }}
                                />
                              </div>
                            </div>
                          </div>
                          <div>
                            <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-slate-500">
                              <span>Full Time</span>
                              <span>Win / Win</span>
                            </div>
                            <div className="mt-1 flex items-center justify-between text-[11px] text-slate-600 tabular-nums">
                              <span>{fullTimeHomeLabel}</span>
                              <span>{fullTimeAwayLabel}</span>
                            </div>
                            <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                              <div className="flex h-full">
                                <div
                                  className="h-full bg-emerald-300/70"
                                  style={{ width: `${fullTimeHomePercent}%` }}
                                />
                                <div
                                  className="h-full bg-rose-300/70"
                                  style={{ width: `${fullTimeAwayPercent}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="mt-3 space-y-1">
                          <div className="flex items-center justify-between text-[11px] text-slate-600 tabular-nums">
                            <span>{homeLabel}</span>
                            {allowDraw ? <span>{drawLabel ?? "--"}</span> : <span />}
                            <span>{awayLabel}</span>
                          </div>
                          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                            <div className="flex h-full">
                              <div
                                className="h-full bg-emerald-300/70"
                                style={{ width: `${homePercent}%` }}
                              />
                              {allowDraw ? (
                                <div
                                  className="h-full bg-slate-300/70"
                                  style={{ width: `${drawPercent}%` }}
                                />
                              ) : null}
                              <div
                                className="h-full bg-rose-300/70"
                                style={{ width: `${awayPercent}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      )}
                      {marginRow ? (
                        <div className="mt-3">
                          <div className="text-[10px] uppercase tracking-wide text-slate-500">
                            Margin
                          </div>
                          <div className="mt-2 grid w-full grid-cols-7 gap-px overflow-hidden rounded-md border border-slate-200 text-[10px] text-slate-600">
                            {marginRow.map((cell, index) => (
                              <div key={`margin-${cell.label}-${index}`} className="contents">
                                <div className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500">
                                  {cell.label}
                                </div>
                              </div>
                            ))}
                            {marginRow.map((cell, index) => (
                              <div
                                key={`margin-value-${cell.label}-${index}`}
                                className="bg-white px-1 py-1 text-center tabular-nums"
                                style={scoreMatrixHighlight(cell.value)}
                              >
                                {formatProbabilityLabel(cell.value, probabilityMode, true).replace("%", "")}
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                      {scoreGrid ? (
                        <div className="mt-3">
                          <div className="text-[10px] uppercase tracking-wide text-slate-500">
                            Score Matrix
                          </div>
                          <div className="mt-2 w-full overflow-x-auto">
                            <div className="min-w-full w-full">
                              <div className="grid w-full min-w-[22rem] grid-cols-[1.5rem_repeat(6,minmax(0,1fr))] gap-px overflow-hidden rounded-md border border-slate-200 text-[10px] text-slate-600">
                                <div className="bg-slate-50 px-1 py-1 text-center font-semibold uppercase text-slate-500">
                                  H/A
                                </div>
                                {SCORE_LABELS.map((label) => (
                                  <div
                                    key={`col-${label}`}
                                    className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500"
                                  >
                                    {label}
                                  </div>
                                ))}
                                {scoreGrid.map((row, rowIndex) => (
                                  <React.Fragment key={`row-${rowIndex}`}>
                                    <div className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500">
                                      {SCORE_LABELS[rowIndex]}
                                    </div>
                                    {row.map((value, colIndex) => (
                                      <div
                                        key={`cell-${rowIndex}-${colIndex}`}
                                        className="bg-white px-1 py-1 text-center tabular-nums"
                                        style={scoreMatrixHighlight(value)}
                                      >
                                        {formatProbabilityLabel(
                                          value,
                                          probabilityMode,
                                          true
                                        ).replace("%", "")}
                                      </div>
                                    ))}
                                  </React.Fragment>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : null}
                      <div className="mt-auto pt-3 flex items-end justify-between gap-3">
                        <div
                          className={`text-sm text-slate-600 ${locationWidth} whitespace-normal break-words`}
                        >
                          {match.city || match.country
                            ? `${match.city}${match.city && match.country ? ", " : ""}${match.country}`
                            : "TBD"}
                        </div>
                        <div
                          className={`text-xs uppercase tracking-wide text-slate-500 text-right ${stageWidth} whitespace-normal break-words`}
                        >
                          {match.stage}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        ))}
      </div>

      <div className="space-y-3">
        <div className="text-2xl font-semibold text-ebony md:text-3xl">
          Past matches
        </div>
        <div className="text-sm text-slate-500">No past matches yet.</div>
      </div>
    </div>
  );
}
