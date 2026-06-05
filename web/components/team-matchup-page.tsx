"use client";

import * as React from "react";
import Image from "next/image";
import type { RatingRow } from "@/lib/ratings";
import { fetchJsonWithGzipFallback } from "@/lib/fetch-json-gzip-client";
import type { WinProbabilities, WinProbabilityEntry } from "@/lib/world-cup-predictor-types";
import { buildScoreMatrix } from "@/lib/score-matrix";
import {
  isCompactWinProbabilities,
  parseCompactEntry,
  resolveCompactEntry,
} from "@/lib/win-probabilities";
import { ArrowLeftRight } from "lucide-react";

type MatchProbabilityValues = {
  home: number | null;
  draw: number | null;
  away: number | null;
};

type TeamOption = {
  team: string;
  flagPath: string | null;
  confederation: string | null;
  rating: number;
  ratingAttack: number;
  ratingDefense: number;
  worldRank: number;
  confederationRank: number | null;
};

const SCORE_LABELS = ["0", "1", "2", "3", "4", "5+"];
const PROBABILITY_HIGHLIGHT_RGB = "147, 197, 253";
const PROBABILITY_HIGHLIGHT_MAX_ALPHA = 0.98;

function resolveProbabilityEntry({
  probabilities,
  homeTeam,
  awayTeam,
  neutral,
  isFriendly,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  neutral: boolean;
  isFriendly: boolean;
}): { entry: WinProbabilityEntry; flipped: boolean } | null {
  if (isCompactWinProbabilities(probabilities)) {
    const entry = resolveCompactEntry(
      probabilities,
      homeTeam,
      awayTeam,
      neutral,
      isFriendly
    );
    return entry ? { entry: parseCompactEntry(entry), flipped: false } : null;
  }
  if (neutral) {
    const entry = probabilities[homeTeam]?.[awayTeam]?.neutral;
    return entry ? { entry, flipped: false } : null;
  }
  const entry = probabilities[homeTeam]?.[awayTeam]?.home;
  return entry ? { entry, flipped: false } : null;
}

function resolveMatchProbabilities({
  probabilities,
  homeTeam,
  awayTeam,
  allowDraw = true,
  neutral,
  isFriendly,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  allowDraw?: boolean;
  neutral: boolean;
  isFriendly: boolean;
}): MatchProbabilityValues | null {
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam,
    awayTeam,
    neutral,
    isFriendly,
  });
  if (!resolved) {
    return null;
  }
  const entry = resolved.entry;
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

function resolveMatchScoreMatrix({
  probabilities,
  homeTeam,
  awayTeam,
  neutral,
  isFriendly,
}: {
  probabilities: WinProbabilities;
  homeTeam: string;
  awayTeam: string;
  neutral: boolean;
  isFriendly: boolean;
}): number[][] | null {
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam,
    awayTeam,
    neutral,
    isFriendly,
  });
  if (!resolved) {
    return null;
  }
  if (resolved.entry.score_matrix) {
    return resolved.entry.score_matrix;
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
  return buildScoreMatrix({
    nu: resolved.entry.nu,
    lamH: resolved.entry.lam_home,
    lamA: resolved.entry.lam_away,
    maxGoals,
  });
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
  return `${value.toFixed(1)}%`;
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

function confederationLabel(confederation: string | null) {
  return confederation ?? "No confederation";
}

function TeamFlag({
  team,
  flagPath,
}: {
  team: string;
  flagPath: string | null;
}) {
  if (!flagPath) {
    return (
      <span className="flex h-4 w-6 shrink-0 items-center justify-center rounded-[1px] border border-slate-300 bg-slate-200" />
    );
  }
  return (
    <span className="relative h-4 w-6 shrink-0 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
      <Image src={flagPath} alt={`${team} flag`} fill className="object-cover" sizes="24px" />
    </span>
  );
}

export function TeamMatchupPage({
  ratings,
  winProbabilities,
  winProbabilitiesPath,
}: {
  ratings: RatingRow[];
  winProbabilities?: WinProbabilities | null;
  winProbabilitiesPath?: string;
}) {
  const [loadedWinProbabilities, setLoadedWinProbabilities] =
    React.useState<WinProbabilities | null>(winProbabilities ?? null);

  React.useEffect(() => {
    if (loadedWinProbabilities || !winProbabilitiesPath) {
      return;
    }
    let cancelled = false;
    void (async () => {
      const data = await fetchJsonWithGzipFallback(winProbabilitiesPath);
      if (!cancelled && data) {
        setLoadedWinProbabilities(data as WinProbabilities);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadedWinProbabilities, winProbabilitiesPath]);

  const resolvedWinProbabilities = loadedWinProbabilities;

  const teams = React.useMemo<TeamOption[]>(() => {
    const confedCounts = new Map<string, number>();
    return ratings.map((row, index) => {
      const confed = confederationLabel(row.confederation);
      const nextConfedRank = (confedCounts.get(confed) ?? 0) + 1;
      confedCounts.set(confed, nextConfedRank);
      return {
        team: row.team,
        flagPath: row.flagPath,
        confederation: row.confederation,
        rating: row.rating,
        ratingAttack: row.rating_attack,
        ratingDefense: row.rating_defense,
        worldRank: index + 1,
        confederationRank: row.confederation ? nextConfedRank : null,
      };
    });
  }, [ratings]);
  const teamMap = React.useMemo(
    () => new Map(teams.map((team) => [team.team.toLowerCase(), team])),
    [teams]
  );
  const sortedTeams = React.useMemo(
    () => [...teams].sort((a, b) => a.team.localeCompare(b.team)),
    [teams]
  );

  const [teamAInput, setTeamAInput] = React.useState(teams[0]?.team ?? "");
  const [teamBInput, setTeamBInput] = React.useState(teams[1]?.team ?? "");
  const [neutral, setNeutral] = React.useState(false);
  const [isFriendly, setIsFriendly] = React.useState(false);
  const [requiresResult, setRequiresResult] = React.useState(false);
  const [probabilityMode, setProbabilityMode] = React.useState<"percent" | "decimal">("percent");
  const teamA = teamMap.get(teamAInput.trim().toLowerCase()) ?? null;
  const teamB = teamMap.get(teamBInput.trim().toLowerCase()) ?? null;

  const probabilityValues = React.useMemo(
    () =>
      resolvedWinProbabilities && teamA && teamB
        ? resolveMatchProbabilities({
            probabilities: resolvedWinProbabilities,
            homeTeam: teamA.team,
            awayTeam: teamB.team,
            neutral,
            isFriendly,
          })
        : null,
    [isFriendly, neutral, teamA, teamB, resolvedWinProbabilities]
  );

  const scoreMatrix = React.useMemo(
    () =>
      resolvedWinProbabilities && teamA && teamB
        ? resolveMatchScoreMatrix({
            probabilities: resolvedWinProbabilities,
            homeTeam: teamA.team,
            awayTeam: teamB.team,
            neutral,
            isFriendly,
          })
        : null,
    [isFriendly, neutral, teamA, teamB, resolvedWinProbabilities]
  );

  const scoreGrid = React.useMemo(
    () => (scoreMatrix ? buildScoreGrid(scoreMatrix) : null),
    [scoreMatrix]
  );
  const marginRow = React.useMemo(
    () => (scoreMatrix ? buildMarginRow(scoreMatrix) : null),
    [scoreMatrix]
  );

  const useDecimal = shouldUseDecimalPrecision([
    probabilityValues?.home,
    probabilityValues?.draw,
    probabilityValues?.away,
  ]);

  const homeLabelRaw = formatProbabilityLabel(
    probabilityValues?.home ?? null,
    probabilityMode,
    useDecimal
  );
  const drawLabelRaw = formatProbabilityLabel(
    probabilityValues?.draw ?? null,
    probabilityMode,
    useDecimal
  );
  const awayLabelRaw = formatProbabilityLabel(
    probabilityValues?.away ?? null,
    probabilityMode,
    useDecimal
  );
  const normalizedShown =
    probabilityMode === "percent"
      ? normalizeProbabilitySegments({
          home:
            probabilityValues?.home !== null && probabilityValues?.home !== undefined
              ? probabilityValues.home * 100
              : null,
          draw:
            probabilityValues?.draw !== null && probabilityValues?.draw !== undefined
              ? probabilityValues.draw * 100
              : null,
          away:
            probabilityValues?.away !== null && probabilityValues?.away !== undefined
              ? probabilityValues.away * 100
              : null,
        })
      : null;

  const homeLabel =
    probabilityMode === "percent" && normalizedShown
      ? formatNormalizedPercent(normalizedShown.home)
      : homeLabelRaw;
  const drawLabel =
    probabilityMode === "percent" && normalizedShown
      ? formatNormalizedPercent(normalizedShown.draw)
      : drawLabelRaw;
  const awayLabel =
    probabilityMode === "percent" && normalizedShown
      ? formatNormalizedPercent(normalizedShown.away)
      : awayLabelRaw;

  const homePercent =
    probabilityMode === "percent" && normalizedShown
      ? Math.max(0, Math.min(100, normalizedShown.home))
      : Math.max(0, Math.min(100, (probabilityValues?.home ?? 0) * 100));
  const drawPercent =
    probabilityMode === "percent" && normalizedShown
      ? Math.max(0, Math.min(100, normalizedShown.draw))
      : Math.max(0, Math.min(100, (probabilityValues?.draw ?? 0) * 100));
  const awayPercent =
    probabilityMode === "percent" && normalizedShown
      ? Math.max(0, Math.min(100, normalizedShown.away))
      : Math.max(0, Math.min(100, (probabilityValues?.away ?? 0) * 100));

  const ninetyValues = probabilityValues;
  const fullTimeValues = React.useMemo(
    () =>
      resolvedWinProbabilities && teamA && teamB && requiresResult
        ? resolveMatchProbabilities({
            probabilities: resolvedWinProbabilities,
            homeTeam: teamA.team,
            awayTeam: teamB.team,
            allowDraw: false,
            neutral,
            isFriendly,
          })
        : null,
    [isFriendly, neutral, requiresResult, teamA, teamB, resolvedWinProbabilities]
  );

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

  const handleSwap = React.useCallback(() => {
    setTeamAInput(teamBInput);
    setTeamBInput(teamAInput);
  }, [teamAInput, teamBInput]);

  const matchupReady = Boolean(
    resolvedWinProbabilities && teamA && teamB && teamA.team !== teamB.team && probabilityValues
  );

  return (
    <div className="space-y-6">
      {!resolvedWinProbabilities ? (
        <div className="rounded-xl border border-dashed border-slate-300 bg-white/70 px-4 py-8 text-sm text-slate-600">
          Loading matchup probabilities...
        </div>
      ) : matchupReady && teamA && teamB ? (
        <div className="space-y-6">
          <div className="lg:grid lg:grid-cols-[minmax(0,1fr)_minmax(24rem,28rem)_minmax(0,1fr)] lg:items-start lg:justify-center lg:gap-4">
            <div className="relative hidden pt-5 lg:block lg:justify-self-end">
              <span className="absolute left-0 top-0 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Team A
              </span>
              <label className="block">
                <select
                  value={teamAInput}
                  onChange={(event) => setTeamAInput(event.target.value)}
                  className="w-full rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  {sortedTeams.map((team) => (
                    <option key={team.team} value={team.team}>
                      {team.team}
                    </option>
                  ))}
                </select>
              </label>
              <div className="mt-3">
                <TeamSummaryCard team={teamA} label="Team A" />
              </div>
            </div>
            <div className="min-w-0 lg:w-full lg:max-w-[28rem] lg:justify-self-center lg:pt-5">
            <div className="mb-3 space-y-3">
              <div className="grid gap-3 sm:grid-cols-2">
                <button
                  type="button"
                  onClick={handleSwap}
                  className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-900"
                >
                  <ArrowLeftRight className="h-4 w-4" />
                  Swap
                </button>
                <label>
                  <span className="sr-only">Display</span>
                  <select
                    value={probabilityMode}
                    onChange={(event) =>
                      setProbabilityMode(event.target.value as "percent" | "decimal")
                    }
                    aria-label="Display"
                    className="h-10 w-full rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  >
                    <option value="percent">% Chance</option>
                    <option value="decimal">Decimal Odds</option>
                  </select>
                </label>
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                <label className="inline-flex h-10 w-full items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-[11px] font-semibold uppercase tracking-wide text-slate-700">
                  <input
                    type="checkbox"
                    checked={neutral}
                    onChange={(event) => setNeutral(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                  />
                  Neutral
                </label>
                <label className="inline-flex h-10 w-full items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-[11px] font-semibold uppercase tracking-wide text-slate-700">
                  <input
                    type="checkbox"
                    checked={isFriendly}
                    onChange={(event) => setIsFriendly(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                  />
                  Friendly
                </label>
                <label className="inline-flex h-10 w-full items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-[11px] font-semibold uppercase tracking-wide text-slate-700">
                  <input
                    type="checkbox"
                    checked={requiresResult}
                    onChange={(event) => setRequiresResult(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                  />
                  Requires Result
                </label>
              </div>
            </div>
            <div className="rounded-xl bg-white px-4 py-3 shadow-sm ring-1 ring-slate-200">
              <div className="space-y-1">
                <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 text-base font-semibold text-slate-900">
                  <span className="flex min-w-0 items-center gap-2">
                    <TeamFlag team={teamA.team} flagPath={teamA.flagPath} />
                    <span className="whitespace-normal break-words">{teamA.team}</span>
                  </span>
                  <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    vs.
                  </span>
                  <span className="flex min-w-0 items-center justify-end gap-2 text-right">
                    <span className="whitespace-normal break-words">{teamB.team}</span>
                    <TeamFlag team={teamB.team} flagPath={teamB.flagPath} />
                  </span>
                </div>
                <div className="text-[11px] text-slate-600">
                  {[neutral ? "Neutral venue" : `${teamA.team} home advantage`, isFriendly ? "Friendly" : "Competitive"]
                    .join(" • ")}
                </div>
              </div>

              {!requiresResult ? (
                <div className="mt-3 space-y-1">
                  <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-slate-500">
                    <span>Win / Draw / Win</span>
                    <span>{probabilityMode === "percent" ? "% chance" : "decimal odds"}</span>
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-slate-600 tabular-nums">
                    <span>{homeLabel}</span>
                    <span>{drawLabel}</span>
                    <span>{awayLabel}</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                    <div className="flex h-full">
                      <div className="h-full bg-emerald-300/70" style={{ width: `${homePercent}%` }} />
                      <div className="h-full bg-slate-300/70" style={{ width: `${drawPercent}%` }} />
                      <div className="h-full bg-rose-300/70" style={{ width: `${awayPercent}%` }} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-3 space-y-3">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-slate-500">
                      <span>After 90&apos;</span>
                      <span>{probabilityMode === "percent" ? "% chance" : "decimal odds"}</span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-[11px] text-slate-600 tabular-nums">
                      <span>{ninetyHomeLabel}</span>
                      <span>{ninetyDrawLabel}</span>
                      <span>{ninetyAwayLabel}</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                      <div className="flex h-full">
                        <div className="h-full bg-emerald-300/70" style={{ width: `${ninetyHomePercent}%` }} />
                        <div className="h-full bg-slate-300/70" style={{ width: `${ninetyDrawPercent}%` }} />
                        <div className="h-full bg-rose-300/70" style={{ width: `${ninetyAwayPercent}%` }} />
                      </div>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-slate-500">
                      <span>Full Time</span>
                      <span>{probabilityMode === "percent" ? "% chance" : "decimal odds"}</span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-[11px] text-slate-600 tabular-nums">
                      <span>{fullTimeHomeLabel}</span>
                      <span>{fullTimeAwayLabel}</span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                      <div className="flex h-full">
                        <div className="h-full bg-emerald-300/70" style={{ width: `${fullTimeHomePercent}%` }} />
                        <div className="h-full bg-rose-300/70" style={{ width: `${fullTimeAwayPercent}%` }} />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {marginRow ? (
                <div className="mt-3">
                  <div className="text-[10px] uppercase tracking-wide text-slate-500">Margin</div>
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
                    {requiresResult ? "Score Matrix (90')" : "Score Matrix"}
                  </div>
                  <div className="mt-2 w-full overflow-x-auto">
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
                        <div key={`row-wrap-${rowIndex}`} className="contents">
                          <div className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500">
                            {SCORE_LABELS[rowIndex]}
                          </div>
                          {row.map((value, colIndex) => (
                            <div
                              key={`cell-${rowIndex}-${colIndex}`}
                              className="bg-white px-1 py-1 text-center tabular-nums"
                              style={scoreMatrixHighlight(value)}
                            >
                              {formatProbabilityLabel(value, probabilityMode, true).replace("%", "")}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
            <div className="relative hidden pt-5 lg:block lg:justify-self-start">
              <span className="absolute left-0 top-0 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Team B
              </span>
              <label className="block">
                <select
                  value={teamBInput}
                  onChange={(event) => setTeamBInput(event.target.value)}
                  className="w-full rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  {sortedTeams.map((team) => (
                    <option key={team.team} value={team.team}>
                      {team.team}
                    </option>
                  ))}
                </select>
              </label>
              <div className="mt-3">
                <TeamSummaryCard team={teamB} label="Team B" align="right" />
              </div>
            </div>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:hidden">
            <div className="space-y-3">
              <label className="block space-y-1">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Team A
                </span>
                <select
                  value={teamAInput}
                  onChange={(event) => setTeamAInput(event.target.value)}
                  className="w-full rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  {sortedTeams.map((team) => (
                    <option key={team.team} value={team.team}>
                      {team.team}
                    </option>
                  ))}
                </select>
              </label>
              <TeamSummaryCard team={teamA} label="Team A" />
            </div>
            <div className="space-y-3">
              <label className="block space-y-1">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Team B
                </span>
                <select
                  value={teamBInput}
                  onChange={(event) => setTeamBInput(event.target.value)}
                  className="w-full rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  {sortedTeams.map((team) => (
                    <option key={team.team} value={team.team}>
                      {team.team}
                    </option>
                  ))}
                </select>
              </label>
              <TeamSummaryCard team={teamB} label="Team B" align="right" />
            </div>
          </div>
        </div>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,15rem)_minmax(0,30rem)_minmax(0,15rem)] xl:items-start xl:justify-center">
          <div className="order-2 space-y-3 xl:order-1 xl:justify-self-end">
            <label className="block space-y-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Team A
              </span>
              <select
                value={teamAInput}
                onChange={(event) => setTeamAInput(event.target.value)}
                className="w-full rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
              >
                {sortedTeams.map((team) => (
                  <option key={team.team} value={team.team}>
                    {team.team}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="order-1 min-w-0 xl:order-2 xl:justify-self-center">
            <div className="mb-3 space-y-3">
              <div className="grid gap-3 sm:grid-cols-2">
                <button
                  type="button"
                  onClick={handleSwap}
                  className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:text-slate-900"
                >
                  <ArrowLeftRight className="h-4 w-4" />
                  Swap
                </button>
                <label>
                  <span className="sr-only">Display</span>
                  <select
                    value={probabilityMode}
                    onChange={(event) =>
                      setProbabilityMode(event.target.value as "percent" | "decimal")
                    }
                    aria-label="Display"
                    className="h-10 w-full rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  >
                    <option value="percent">% Chance</option>
                    <option value="decimal">Decimal Odds</option>
                  </select>
                </label>
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                <label className="inline-flex h-10 w-full items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-[11px] font-semibold uppercase tracking-wide text-slate-700">
                  <input
                    type="checkbox"
                    checked={neutral}
                    onChange={(event) => setNeutral(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                  />
                  Neutral
                </label>
                <label className="inline-flex h-10 w-full items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-[11px] font-semibold uppercase tracking-wide text-slate-700">
                  <input
                    type="checkbox"
                    checked={isFriendly}
                    onChange={(event) => setIsFriendly(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                  />
                  Friendly
                </label>
                <label className="inline-flex h-10 w-full items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-[11px] font-semibold uppercase tracking-wide text-slate-700">
                  <input
                    type="checkbox"
                    checked={requiresResult}
                    onChange={(event) => setRequiresResult(event.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                  />
                  Requires Result
                </label>
              </div>
            </div>
            <div className="rounded-xl border border-dashed border-slate-300 bg-white/70 px-4 py-8 text-sm text-slate-600">
              {teamA && teamB && teamA.team === teamB.team
                ? "Choose two different teams to view the matchup card."
                : "Pick two valid teams from the current ratings universe to view the matchup card."}
            </div>
          </div>
          <div className="order-3 space-y-3 xl:justify-self-start">
            <label className="block space-y-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Team B
              </span>
              <select
                value={teamBInput}
                onChange={(event) => setTeamBInput(event.target.value)}
                className="w-full rounded-md bg-white px-3 py-2 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
              >
                {sortedTeams.map((team) => (
                  <option key={team.team} value={team.team}>
                    {team.team}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}

function TeamSummaryCard({
  team,
  label,
  align = "left",
}: {
  team: TeamOption;
  label: string;
  align?: "left" | "right";
}) {
  const isRight = align === "right";
  return (
    <div className="w-full rounded-xl bg-white px-4 py-4 shadow-sm ring-1 ring-slate-200 md:max-w-none xl:max-w-[15rem]">
      <div className={`flex items-center gap-3 ${isRight ? "justify-between" : ""}`}>
        {!isRight ? <TeamFlag team={team.team} flagPath={team.flagPath} /> : null}
        <div className={`min-w-0 ${isRight ? "text-right" : ""}`}>
          <div className="line-clamp-2 text-base font-semibold leading-tight text-slate-900">
            {team.team}
          </div>
        </div>
        {isRight ? <TeamFlag team={team.team} flagPath={team.flagPath} /> : null}
      </div>

      <dl className="mt-4 space-y-2.5 text-sm">
        <MetricRow label="Overall" value={team.rating.toFixed(1)} />
        <MetricRow label="Offense" value={team.ratingAttack.toFixed(1)} />
        <MetricRow label="Defense" value={team.ratingDefense.toFixed(1)} />
        <MetricRow label="World rank" value={`#${team.worldRank}`} />
        <MetricRow
          label="Confed. rank"
          value={team.confederationRank ? `#${team.confederationRank}` : "--"}
        />
        <MetricRow
          label="Confederation"
          value={confederationLabel(team.confederation)}
        />
      </dl>
    </div>
  );
}

function MetricRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right font-medium tabular-nums text-slate-900">{value}</dd>
    </div>
  );
}
