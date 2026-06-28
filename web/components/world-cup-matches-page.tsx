"use client";

import * as React from "react";
import Image from "next/image";
import type { WorldCupMatch } from "@/lib/world-cup-matches";
import type { CompletedWorldCupMatch } from "@/lib/world-cup-results";
import type { MatchConditionalAdvancement } from "@/lib/world-cup-match-conditional-advancement";
import type { RatingRow } from "@/lib/ratings";
import {
  formatRatingValue,
  formatTiltValue,
  ratingPillStyle,
  tiltPillStyle,
} from "@/lib/rating-display";
import type { WinProbabilities, WinProbabilityEntry } from "@/lib/world-cup-predictor-types";
import { buildScoreMatrix } from "@/lib/score-matrix";
import { ChevronDown } from "lucide-react";
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

type MatchConditionalAdvancementMap = Map<string, MatchConditionalAdvancement>;

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
const DEFAULT_TIME_ZONE = "Australia/Sydney";
const DEFAULT_TIME_ZONE_OFFSET_VALUE = "offset:+10:00";
const PREFERRED_REPRESENTATIVE_TIME_ZONES = [
  "Pacific/Pago_Pago",
  "Pacific/Honolulu",
  "America/Anchorage",
  "America/Los_Angeles",
  "America/Denver",
  "America/Chicago",
  "America/New_York",
  "America/Halifax",
  "America/St_Johns",
  "America/Sao_Paulo",
  "Atlantic/South_Georgia",
  "Atlantic/Azores",
  "UTC",
  "Europe/London",
  "Europe/Paris",
  "Europe/Athens",
  "Europe/Moscow",
  "Asia/Tehran",
  "Asia/Dubai",
  "Asia/Kabul",
  "Asia/Karachi",
  "Asia/Kolkata",
  "Asia/Dhaka",
  "Asia/Bangkok",
  "Asia/Singapore",
  "Asia/Shanghai",
  "Asia/Tokyo",
  "Australia/Darwin",
  "Australia/Sydney",
  "Pacific/Noumea",
  "Pacific/Auckland",
  "Pacific/Chatham",
  "Pacific/Apia",
  "Pacific/Kiritimati",
] as const;

function normalizeCountry(value: string | null | undefined) {
  return value ? value.trim().toLowerCase() : "";
}

function isValidTimeZone(value: string) {
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: value }).format(new Date());
    return true;
  } catch {
    return false;
  }
}

function getTimeZoneOffsetMinutes(timeZone: string, date = new Date()) {
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone,
      timeZoneName: "shortOffset",
    }).formatToParts(date);
    const offsetPart = parts.find((part) => part.type === "timeZoneName")?.value ?? "GMT";
    if (offsetPart === "GMT" || offsetPart === "UTC") {
      return 0;
    }
    const match = offsetPart.match(/^GMT([+-])(\d{1,2})(?::?(\d{2}))?$/);
    if (!match) {
      return null;
    }
    const sign = match[1] === "-" ? -1 : 1;
    const hours = Number(match[2]);
    const minutes = Number(match[3] ?? "0");
    return sign * (hours * 60 + minutes);
  } catch {
    return null;
  }
}

function formatOffsetLabel(offsetMinutes: number) {
  const sign = offsetMinutes < 0 ? "-" : "+";
  const absoluteMinutes = Math.abs(offsetMinutes);
  const hours = Math.floor(absoluteMinutes / 60);
  const minutes = absoluteMinutes % 60;
  if (minutes === 0) {
    return `UTC${sign}${hours}`;
  }
  return `UTC${sign}${hours}:${String(minutes).padStart(2, "0")}`;
}

function formatRepresentativeCity(timeZone: string) {
  if (timeZone === "UTC") {
    return "UTC";
  }
  const parts = timeZone.split("/");
  const mostSpecific = parts[parts.length - 1] ?? timeZone;
  return mostSpecific.replace(/_/g, " ");
}

function offsetValueFromMinutes(offsetMinutes: number) {
  const sign = offsetMinutes < 0 ? "-" : "+";
  const absoluteMinutes = Math.abs(offsetMinutes);
  const hours = String(Math.floor(absoluteMinutes / 60)).padStart(2, "0");
  const minutes = String(absoluteMinutes % 60).padStart(2, "0");
  return `offset:${sign}${hours}:${minutes}`;
}

function parseOffsetValue(value: string) {
  const match = value.match(/^offset:([+-])(\d{2}):(\d{2})$/);
  if (!match) {
    return null;
  }
  const sign = match[1] === "-" ? -1 : 1;
  const hours = Number(match[2]);
  const minutes = Number(match[3]);
  return sign * (hours * 60 + minutes);
}

function shiftDateByOffset(date: Date, offsetMinutes: number) {
  return new Date(date.getTime() + offsetMinutes * 60_000);
}

function formatDatePartsAtOffset(date: Date, offsetMinutes: number) {
  const shifted = shiftDateByOffset(date, offsetMinutes);
  return {
    year: String(shifted.getUTCFullYear()),
    month: String(shifted.getUTCMonth() + 1).padStart(2, "0"),
    day: String(shifted.getUTCDate()).padStart(2, "0"),
  };
}

function buildOffsetOptions() {
  const supportedValuesOf = (Intl as typeof Intl & {
    supportedValuesOf?: (key: string) => string[];
  }).supportedValuesOf;
  const timeZones =
    typeof supportedValuesOf === "function"
      ? supportedValuesOf("timeZone")
      : [DEFAULT_TIME_ZONE, "UTC"];
  const offsetToRepresentativeCity = new Map<number, string>();
  for (const timeZone of PREFERRED_REPRESENTATIVE_TIME_ZONES) {
    const offsetMinutes = getTimeZoneOffsetMinutes(timeZone);
    if (offsetMinutes === null) {
      continue;
    }
    if (!offsetToRepresentativeCity.has(offsetMinutes)) {
      offsetToRepresentativeCity.set(offsetMinutes, formatRepresentativeCity(timeZone));
    }
  }
  for (const timeZone of timeZones) {
    const offsetMinutes = getTimeZoneOffsetMinutes(timeZone);
    if (offsetMinutes !== null) {
      const representativeCity = formatRepresentativeCity(timeZone);
      if (!offsetToRepresentativeCity.has(offsetMinutes)) {
        offsetToRepresentativeCity.set(offsetMinutes, representativeCity);
      }
    }
  }
  return [...offsetToRepresentativeCity.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([offsetMinutes, representativeCity]) => ({
      value: offsetValueFromMinutes(offsetMinutes),
      label: `${formatOffsetLabel(offsetMinutes)} (${representativeCity})`,
    }));
}

const OFFSET_OPTIONS: Array<{ value: string; label: string }> = buildOffsetOptions();

function formatLocalDateKey(dateInput: string, timeZone: string) {
  const date = new Date(dateInput);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  const offsetMinutes = parseOffsetValue(timeZone);
  if (offsetMinutes === null) {
    return null;
  }
  const parts = formatDatePartsAtOffset(date, offsetMinutes);
  return `${parts.year}-${parts.month}-${parts.day}`;
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

function buildExpectedGoals(matrix: number[][]) {
  const normalized = normalizeScoreMatrix(matrix);
  let home = 0;
  let away = 0;
  for (let homeGoals = 0; homeGoals < normalized.length; homeGoals += 1) {
    const row = normalized[homeGoals] ?? [];
    for (let awayGoals = 0; awayGoals < row.length; awayGoals += 1) {
      const value = Number(row[awayGoals] ?? 0);
      home += homeGoals * value;
      away += awayGoals * value;
    }
  }
  return { home, away };
}

function buildTotalGoalsRow(matrix: number[][]) {
  const normalized = normalizeScoreMatrix(matrix);
  const buckets = Array(7).fill(0);
  for (let homeGoals = 0; homeGoals < normalized.length; homeGoals += 1) {
    const row = normalized[homeGoals] ?? [];
    for (let awayGoals = 0; awayGoals < row.length; awayGoals += 1) {
      const value = Number(row[awayGoals] ?? 0);
      const totalGoals = homeGoals + awayGoals;
      buckets[Math.min(totalGoals, 6)] += value;
    }
  }
  return buckets.map((value, index) => ({
    label: index === 6 ? "6+" : String(index),
    value,
  }));
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

function TeamOptionFlags({
  teams,
  align = "left",
}: {
  teams: string[];
  align?: "left" | "right";
}) {
  return (
    <span
      className={`flex min-w-0 items-center gap-1 ${align === "right" ? "justify-end" : ""}`}
    >
      {teams.map((team, index) => (
        <React.Fragment key={team}>
          <span className="relative h-4 w-6 shrink-0 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
            <Image
              src={`/flags/${team.replace(/ /g, "_")}.png`}
              alt={`${team} flag`}
              fill
              className="object-cover"
              sizes="24px"
            />
          </span>
          {index < teams.length - 1 ? (
            <span className="shrink-0 text-xs font-semibold text-slate-400">/</span>
          ) : null}
        </React.Fragment>
      ))}
    </span>
  );
}

type TeamRatingsSummary = {
  overall: number;
  tilt: number;
}

function RatingsBadge({
  label,
  value,
  variant,
}: {
  label: string;
  value: number | null | undefined;
  variant: "rating" | "tilt";
}) {
  return (
    <span
      className="inline-flex items-center justify-center gap-1 whitespace-nowrap rounded-full border px-1.5 py-1 text-[11px] font-mono font-semibold leading-none tabular-nums text-slate-700"
      style={variant === "rating" ? ratingPillStyle(value) : tiltPillStyle(value)}
    >
      <span className="font-sans text-[9px] font-semibold uppercase tracking-wide text-slate-500">
        {label}
      </span>
      <span>{variant === "rating" ? formatRatingValue(value) : formatTiltValue(value)}</span>
    </span>
  );
}

function TeamRatingsPanel({
  team,
  ratings,
  align = "left",
  reverse = false,
}: {
  team: string;
  ratings: TeamRatingsSummary | null;
  align?: "left" | "right";
  reverse?: boolean;
}) {
  const isPlaceholder = isPlaceholderLabel(team);
  const badges = [
    <RatingsBadge key="rating" label="Rating" value={ratings?.overall ?? null} variant="rating" />,
    <RatingsBadge key="tilt" label="TILT" value={ratings?.tilt ?? null} variant="tilt" />,
  ];
  return (
    <div className={`${align === "right" ? "text-right" : "text-left"}`}>
      <div className={`flex gap-1.5 ${align === "right" ? "justify-end" : "justify-start"}`}>
        {reverse ? [...badges].reverse() : badges}
      </div>
    </div>
  );
}

function ExpandToggle({
  expanded,
  onToggle,
}: {
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={expanded ? "Collapse details" : "Expand details"}
      className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-slate-50 text-slate-500 ring-1 ring-slate-200 transition hover:bg-slate-100 hover:text-slate-700"
    >
      <ChevronDown
        className={`h-4 w-4 transition-transform ${expanded ? "rotate-180" : ""}`}
        strokeWidth={2.25}
      />
    </button>
  );
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
    timeZone: "UTC",
  });
}

function formatMatchDateTime(dateInput: string | null | undefined, timeZone: string) {
  if (!dateInput) {
    return null;
  }
  const date = new Date(dateInput);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  const offsetMinutes = parseOffsetValue(timeZone);
  if (offsetMinutes === null) {
    return null;
  }
  const shifted = shiftDateByOffset(date, offsetMinutes);
  const dateLabel = new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(shifted);
  return `${dateLabel} ${formatOffsetLabel(offsetMinutes)}`;
}

function resolveInitialOffsetValue() {
  const offsetMinutes = getTimeZoneOffsetMinutes(DEFAULT_TIME_ZONE);
  if (offsetMinutes === null) {
    return DEFAULT_TIME_ZONE_OFFSET_VALUE;
  }
  return offsetValueFromMinutes(offsetMinutes);
}

function resolveBrowserOffsetValue() {
  const browserTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  if (!browserTimeZone || !isValidTimeZone(browserTimeZone)) {
    return null;
  }
  const offsetMinutes = getTimeZoneOffsetMinutes(browserTimeZone);
  if (offsetMinutes === null) {
    return null;
  }
  return offsetValueFromMinutes(offsetMinutes);
}

function ConditionalAdvancementPanel({
  conditional,
  probabilityMode,
  homeTeam,
  awayTeam,
}: {
  conditional: MatchConditionalAdvancement;
  probabilityMode: "percent" | "decimal";
  homeTeam: string;
  awayTeam: string;
}) {
  const forceDecimal = shouldUseDecimalPrecision([
    conditional.outcomes.home_win.homeTeamProbability,
    conditional.outcomes.home_win.awayTeamProbability,
    conditional.outcomes.draw.homeTeamProbability,
    conditional.outcomes.draw.awayTeamProbability,
    conditional.outcomes.away_win.homeTeamProbability,
    conditional.outcomes.away_win.awayTeamProbability,
  ]);
  const outcomeLabel =
    conditional.basis === "after_90" ? "conditional on 90'" : "conditional on result";
  const nextStageLabel =
    conditional.nextStage === "Final"
      ? "Advance to Final"
      : `Advance to ${conditional.nextStage}`;
  const columns = [
    {
      key: "home_win",
      label: "Home win",
      homeValue: conditional.outcomes.home_win.homeTeamProbability,
      awayValue: conditional.outcomes.home_win.awayTeamProbability,
    },
    {
      key: "draw",
      label: "Draw",
      homeValue: conditional.outcomes.draw.homeTeamProbability,
      awayValue: conditional.outcomes.draw.awayTeamProbability,
    },
    {
      key: "away_win",
      label: "Away win",
      homeValue: conditional.outcomes.away_win.homeTeamProbability,
      awayValue: conditional.outcomes.away_win.awayTeamProbability,
    },
  ] as const;

  return (
    <div>
      <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
        <div className="text-[10px] uppercase tracking-wide text-slate-500">
          {nextStageLabel}
        </div>
        <div className="text-[10px] text-slate-400">{outcomeLabel}</div>
      </div>
      <div className="mt-2 overflow-hidden rounded-md border border-slate-200">
        <div className="grid grid-cols-[minmax(0,1fr)_repeat(3,minmax(0,1fr))] bg-slate-50 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
          <div className="px-2 py-1.5">Team</div>
          {columns.map((column) => (
            <div key={column.key} className="px-2 py-1.5 text-center normal-case tracking-normal">
              {column.label}
            </div>
          ))}
        </div>
        {[
          {
            team: homeTeam,
            values: columns.map((column) => column.homeValue),
          },
          {
            team: awayTeam,
            values: columns.map((column) => column.awayValue),
          },
        ].map((row) => (
          <div
            key={row.team}
            className="grid grid-cols-[minmax(0,1fr)_repeat(3,minmax(0,1fr))] border-t border-slate-200 bg-white text-[11px] text-slate-700"
          >
            <div className="px-2 py-1.5 font-medium">{row.team}</div>
            {row.values.map((value, index) => (
              <div key={`${row.team}-${columns[index].key}`} className="px-2 py-1.5 text-center font-mono tabular-nums">
                {formatProbabilityLabel(value, probabilityMode, forceDecimal)}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export function WorldCupMatchesPageClient({
  matches,
  completedMatches,
  ratings,
  winProbabilities,
  conditionalAdvancement,
}: {
  matches: WorldCupMatch[];
  completedMatches: CompletedWorldCupMatch[];
  ratings: RatingRow[];
  winProbabilities: WinProbabilities;
  conditionalAdvancement: MatchConditionalAdvancementMap;
}) {
  const [probabilityMode, setProbabilityMode] = React.useState<"percent" | "decimal">("percent");
  const [selectedTimeZone, setSelectedTimeZone] = React.useState(resolveInitialOffsetValue);
  const [query, setQuery] = React.useState("");
  const [expandedMatches, setExpandedMatches] = React.useState<Record<string, boolean>>({});
  React.useEffect(() => {
    const browserOffsetValue = resolveBrowserOffsetValue();
    if (browserOffsetValue) {
      setSelectedTimeZone(browserOffsetValue);
    }
  }, []);
  const completedById = React.useMemo(
    () => new Map(completedMatches.map((match) => [String(match.matchId), match])),
    [completedMatches]
  );
  const matchesById = React.useMemo(
    () => new Map(matches.map((match) => [String(match.id), match])),
    [matches]
  );
  const ratingsByTeam = React.useMemo(
    () =>
      new Map(
        ratings.map((row) => [
          row.team,
          {
            overall: row.rating,
            tilt: row.tilt,
          } satisfies TeamRatingsSummary,
        ])
      ),
    [ratings]
  );
  const toggleExpanded = React.useCallback((matchKey: string) => {
    setExpandedMatches((current) => ({ ...current, [matchKey]: !current[matchKey] }));
  }, []);

  const filteredUpcomingMatches = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return matches.filter((match) => {
      if (completedById.has(String(match.id))) {
        return false;
      }
      const home = match.home.toLowerCase();
      const away = match.away.toLowerCase();
      return !normalized || home.includes(normalized) || away.includes(normalized);
    });
  }, [completedById, matches, query]);

  const filteredCompletedMatches = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return completedMatches.filter((match) => {
      if (!normalized) {
        return true;
      }
      return (
        match.homeTeam.toLowerCase().includes(normalized) ||
        match.awayTeam.toLowerCase().includes(normalized)
      );
    });
  }, [completedMatches, query]);

  const timeZoneOptions = React.useMemo(() => {
    return OFFSET_OPTIONS;
  }, [selectedTimeZone]);

  const upcomingSections = React.useMemo(
    () => {
      const grouped = new Map<string, { label: string; matches: WorldCupMatch[] }>();
      for (const match of filteredUpcomingMatches) {
        const dateKey = match.kickoffUtc
          ? formatLocalDateKey(match.kickoffUtc, selectedTimeZone) ?? match.date
          : match.date;
        if (!grouped.has(dateKey)) {
          grouped.set(dateKey, { label: formatDateHeading(dateKey), matches: [] });
        }
        grouped.get(dateKey)?.matches.push(match);
      }
      return Array.from(grouped.entries())
        .sort((a, b) => a[0].localeCompare(b[0]))
        .map(([key, value]) => ({ key, ...value }));
    },
    [filteredUpcomingMatches, selectedTimeZone]
  );

  return (
    <div className="space-y-6">
      <div className="flex w-full flex-col gap-3 lg:flex-row lg:items-center">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search teams"
          className="min-w-0 w-full max-w-[25rem] flex-1 rounded-md bg-white px-3 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 md:w-64"
        />
        <div className="flex w-full flex-col gap-3 sm:flex-row lg:ml-auto lg:w-auto">
          <div className="w-full sm:w-52 lg:w-56">
            <select
              value={selectedTimeZone}
              onChange={(event) => setSelectedTimeZone(event.target.value)}
              aria-label="Timezone"
              className="w-full rounded-md bg-white px-2.5 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
            >
              {timeZoneOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="w-full sm:w-40">
            <select
              value={probabilityMode}
              onChange={(event) =>
                setProbabilityMode(event.target.value as "percent" | "decimal")
              }
              aria-label="Probability display mode"
              className="w-full rounded-md bg-white px-2.5 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
            >
              <option value="percent">% Chance</option>
              <option value="decimal">Decimal Odds</option>
            </select>
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <div className="text-2xl font-semibold text-ebony md:text-3xl">
          Upcoming matches
        </div>
        {upcomingSections.length === 0 ? (
          <div className="text-sm text-slate-500">No upcoming matches.</div>
        ) : upcomingSections.map((section) => (
          <section key={section.key} className="space-y-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              {section.label}
            </h2>
            <div className="grid items-start gap-4 grid-cols-1 sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
              {section.matches.map((match) => {
                const isQualifier = match.id.startsWith("Q-");
                const locationWidth = isQualifier ? "w-[45%]" : "w-[60%]";
                const stageWidth = isQualifier ? "w-[45%]" : "w-[40%]";
                const allowDraw = Boolean(match.group);
                const requiresResult = !allowDraw;
                const homePlaceholder = isPlaceholderLabel(match.home);
                const awayPlaceholder = isPlaceholderLabel(match.away);
                const homeOptionFlags =
                  homePlaceholder && match.homeOptions && match.homeOptions.length >= 2
                    ? match.homeOptions
                    : null;
                const awayOptionFlags =
                  awayPlaceholder && match.awayOptions && match.awayOptions.length >= 2
                    ? match.awayOptions
                    : null;
                const displayHome = homePlaceholder ? "TBD" : match.home;
                const displayAway = awayPlaceholder ? "TBD" : match.away;
                const kickoffLabel =
                  formatMatchDateTime(match.kickoffUtc, selectedTimeZone) ??
                  formatDateHeading(match.date);
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
                const expectedGoals = scoreMatrix ? buildExpectedGoals(scoreMatrix) : null;
                const totalGoalsRow = scoreMatrix ? buildTotalGoalsRow(scoreMatrix) : null;
                const matchKey = `upcoming-${match.id}-${match.home}-${match.away}`;
                const expanded = Boolean(expandedMatches[matchKey]);
                const homeRatings = ratingsByTeam.get(match.home) ?? null;
                const awayRatings = ratingsByTeam.get(match.away) ?? null;
                const conditional = conditionalAdvancement.get(String(match.id));
                const hasExpandableDetails = Boolean(
                  expectedGoals || totalGoalsRow || marginRow || scoreGrid || conditional
                );

                return (
                  <div
                    key={`${match.id}-${match.home}-${match.away}`}
                    id={`match-${match.id}`}
                    className="mb-3 min-w-0 scroll-mt-24"
                  >
                    <div className="relative rounded-xl bg-white ring-1 ring-slate-200 shadow-sm px-4 py-3 flex flex-col">
                      <div className="space-y-1">
                        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 text-base font-semibold text-slate-900">
                          <span className="flex items-center gap-2 min-w-0">
                            {homeOptionFlags ? (
                              <TeamOptionFlags teams={homeOptionFlags} />
                            ) : homePlaceholder ? (
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
                            <span className="whitespace-normal break-words">
                              {homeOptionFlags ? "" : displayHome}
                            </span>
                          </span>
                          <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                            vs.
                          </span>
                          <span className="flex items-center gap-2 min-w-0 justify-end text-right">
                            <span className="whitespace-normal break-words">
                              {awayOptionFlags ? "" : displayAway}
                            </span>
                            {awayOptionFlags ? (
                              <TeamOptionFlags teams={awayOptionFlags} align="right" />
                            ) : awayPlaceholder ? (
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
                        <div className="text-[11px] text-slate-500">{kickoffLabel}</div>
                      </div>
                      <div className="mt-3 grid grid-cols-1 gap-3 border-t border-slate-100 pt-3 min-[370px]:grid-cols-2">
                        <TeamRatingsPanel team={match.home} ratings={homeRatings} />
                        <TeamRatingsPanel
                          team={match.away}
                          ratings={awayRatings}
                          align="right"
                          reverse
                        />
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
                      {hasExpandableDetails && expanded ? (
                        <div className="mt-3 border-t border-slate-100 pt-3">
                          <div className="space-y-3">
                              {conditional ? (
                                <ConditionalAdvancementPanel
                                  conditional={conditional}
                                  probabilityMode={probabilityMode}
                                  homeTeam={displayHome}
                                  awayTeam={displayAway}
                                />
                              ) : null}
                              {expectedGoals ? (
                                <div>
                                  <div className="text-[10px] uppercase tracking-wide text-slate-500">
                                    {requiresResult ? "Expected Goals (90')" : "Expected Goals"}
                                  </div>
                                  <div className="mt-2 grid grid-cols-[minmax(0,1fr)_auto_auto_minmax(0,1fr)] gap-px overflow-hidden rounded-md border border-slate-200 text-[11px] text-slate-700">
                                    <div className="bg-slate-50 px-2 py-1 text-left font-semibold text-slate-500">
                                      {displayHome}
                                    </div>
                                    <div className="bg-white px-2 py-1 text-center font-mono tabular-nums">
                                      {expectedGoals.home.toFixed(2)}
                                    </div>
                                    <div className="bg-white px-2 py-1 text-center font-mono tabular-nums">
                                      {expectedGoals.away.toFixed(2)}
                                    </div>
                                    <div className="bg-slate-50 px-2 py-1 text-right font-semibold text-slate-500">
                                      {displayAway}
                                    </div>
                                  </div>
                                </div>
                              ) : null}
                              {totalGoalsRow ? (
                                <div>
                                  <div className="text-[10px] uppercase tracking-wide text-slate-500">
                                    {requiresResult ? "Total Goals (90')" : "Total Goals"}
                                  </div>
                                  <div className="mt-2 grid w-full grid-cols-7 gap-px overflow-hidden rounded-md border border-slate-200 text-[10px] text-slate-600">
                                    {totalGoalsRow.map((cell, index) => (
                                      <div key={`total-goals-label-${match.id}-${index}`} className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500">
                                        {cell.label}
                                      </div>
                                    ))}
                                    {totalGoalsRow.map((cell, index) => (
                                      <div
                                        key={`total-goals-value-${match.id}-${index}`}
                                        className="bg-white px-1 py-1 text-center tabular-nums"
                                        style={scoreMatrixHighlight(cell.value)}
                                      >
                                        {formatProbabilityLabel(cell.value, probabilityMode, true).replace("%", "")}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              ) : null}
                              {marginRow ? (
                                <div>
                                  <div className="text-[10px] uppercase tracking-wide text-slate-500">
                                    {requiresResult ? "Margin (90')" : "Margin"}
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
                                <div>
                                  <div className="text-[10px] uppercase tracking-wide text-slate-500">
                                    {requiresResult ? "Score Matrix (90')" : "Score Matrix"}
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
                      {hasExpandableDetails ? (
                        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex translate-y-1/2 justify-center">
                          <div className="pointer-events-auto bg-white px-1">
                            <ExpandToggle
                              expanded={expanded}
                              onToggle={() => toggleExpanded(matchKey)}
                            />
                          </div>
                        </div>
                      ) : null}
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
        {filteredCompletedMatches.length === 0 ? (
          <div className="text-sm text-slate-500">No past matches yet.</div>
        ) : (
          <div className="grid items-start gap-4 grid-cols-1 sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {filteredCompletedMatches.map((match) => {
              const scheduledMatch = matchesById.get(String(match.matchId));
              const kickoffLabel =
                formatMatchDateTime(scheduledMatch?.kickoffUtc, selectedTimeZone) ??
                formatDateHeading(match.date);
              const margin = match.homeScore - match.awayScore;
              const homeWon = margin > 0 || match.winner === match.homeTeam;
              const awayWon = margin < 0 || match.winner === match.awayTeam;
              const isDraw = !homeWon && !awayWon;
              const matrix = resolveMatchScoreMatrix({
                probabilities: winProbabilities,
                homeTeam: match.homeTeam,
                awayTeam: match.awayTeam,
                country: match.country,
                neutralOverride: match.neutral ?? null,
              });
              const scoreGrid = matrix ? buildScoreGrid(matrix) : null;
              const marginRow = matrix ? buildMarginRow(matrix) : null;
              const expectedGoals = matrix ? buildExpectedGoals(matrix) : null;
              const totalGoalsRow = matrix ? buildTotalGoalsRow(matrix) : null;
              const actualMarginIndex =
                margin >= 3 ? 0 : margin === 2 ? 1 : margin === 1 ? 2 : margin === 0 ? 3 : margin === -1 ? 4 : margin === -2 ? 5 : 6;
              const actualHomeBucket = Math.min(match.homeScore, 5);
              const actualAwayBucket = Math.min(match.awayScore, 5);
              const matchKey = `past-${match.matchId}`;
              const expanded = Boolean(expandedMatches[matchKey]);
              const homeRatings = ratingsByTeam.get(match.homeTeam) ?? null;
              const awayRatings = ratingsByTeam.get(match.awayTeam) ?? null;
              const conditional = conditionalAdvancement.get(String(match.matchId));
              const allowDraw = match.stage === "Group";
              const hasExpandableDetails = Boolean(
                expectedGoals || totalGoalsRow || marginRow || scoreGrid || conditional
              );
              const shownValues = resolveMatchProbabilities({
                probabilities: winProbabilities,
                homeTeam: match.homeTeam,
                awayTeam: match.awayTeam,
                allowDraw,
                country: match.country,
                neutralOverride: match.neutral ?? null,
              });
              const shownLabelValues =
                probabilityMode === "percent"
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
              const normalizedShown = probabilityMode === "percent" ? shownLabelValues : null;
              const homeLabelRaw = formatProbabilityLabel(
                shownValues?.home ?? null,
                probabilityMode
              );
              const drawLabelRaw = allowDraw
                ? formatProbabilityLabel(shownValues?.draw ?? null, probabilityMode)
                : null;
              const awayLabelRaw = formatProbabilityLabel(
                shownValues?.away ?? null,
                probabilityMode
              );
              const homeLabel =
                probabilityMode === "percent" && normalizedShown
                  ? formatNormalizedPercent(normalizedShown.home)
                  : homeLabelRaw;
              const drawLabel =
                allowDraw && probabilityMode === "percent" && normalizedShown
                  ? formatNormalizedPercent(normalizedShown.draw)
                  : drawLabelRaw;
              const awayLabel =
                probabilityMode === "percent" && normalizedShown
                  ? formatNormalizedPercent(normalizedShown.away)
                  : awayLabelRaw;
              const homePercent =
                probabilityMode === "percent" && normalizedShown
                  ? Math.max(0, Math.min(100, normalizedShown.home))
                  : Math.max(0, Math.min(100, (shownValues?.home ?? 0) * 100));
              const drawPercent =
                allowDraw
                  ? probabilityMode === "percent" && normalizedShown
                    ? Math.max(0, Math.min(100, normalizedShown.draw))
                    : Math.max(0, Math.min(100, (shownValues?.draw ?? 0) * 100))
                  : 0;
              const awayPercent =
                probabilityMode === "percent" && normalizedShown
                  ? Math.max(0, Math.min(100, normalizedShown.away))
                  : Math.max(0, Math.min(100, (shownValues?.away ?? 0) * 100));
              const actualOutcome =
                homeWon ? "home" : awayWon ? "away" : "draw";
              return (
                <div key={`past-${match.matchId}`} className="mb-3 min-w-0">
                  <div className="relative rounded-xl bg-white ring-1 ring-slate-200 shadow-sm px-4 py-3 flex flex-col">
                    <div className="space-y-1">
                      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 text-base font-semibold text-slate-900">
                        <span className={`flex items-center gap-2 min-w-0 ${awayWon ? "opacity-40" : "opacity-100"}`}>
                          <span className="relative h-4 w-6 shrink-0 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
                            <Image
                              src={`/flags/${match.homeTeam.replace(/ /g, "_")}.png`}
                              alt={`${match.homeTeam} flag`}
                              fill
                              className="object-cover"
                              sizes="24px"
                            />
                          </span>
                          <span
                            className={`whitespace-normal break-words ${homeWon ? "font-bold text-slate-900" : isDraw ? "text-slate-900" : ""}`}
                          >
                            {match.homeTeam}
                          </span>
                        </span>
                        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                          {match.homeScore}-{match.awayScore}
                        </span>
                        <span className={`flex items-center gap-2 min-w-0 justify-end text-right ${homeWon ? "opacity-40" : "opacity-100"}`}>
                          <span
                            className={`whitespace-normal break-words ${awayWon ? "font-bold text-slate-900" : isDraw ? "text-slate-900" : ""}`}
                          >
                            {match.awayTeam}
                          </span>
                          <span className="relative h-4 w-6 shrink-0 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
                            <Image
                              src={`/flags/${match.awayTeam.replace(/ /g, "_")}.png`}
                              alt={`${match.awayTeam} flag`}
                              fill
                              className="object-cover"
                              sizes="24px"
                            />
                          </span>
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-500">{kickoffLabel}</div>
                    </div>
                    <div className="mt-3 grid grid-cols-1 gap-3 border-t border-slate-100 pt-3 min-[370px]:grid-cols-2">
                      <TeamRatingsPanel team={match.homeTeam} ratings={homeRatings} />
                      <TeamRatingsPanel
                        team={match.awayTeam}
                        ratings={awayRatings}
                        align="right"
                        reverse
                      />
                    </div>
                    <div className="mt-3 space-y-1">
                      <div className="flex items-center justify-between text-[11px] tabular-nums">
                        <span className={actualOutcome === "home" ? "font-semibold text-slate-900" : "text-slate-400"}>
                          {homeLabel}
                        </span>
                        {allowDraw ? (
                          <span className={actualOutcome === "draw" ? "font-semibold text-slate-900" : "text-slate-400"}>
                            {drawLabel ?? "--"}
                          </span>
                        ) : (
                          <span />
                        )}
                        <span className={actualOutcome === "away" ? "font-semibold text-slate-900" : "text-slate-400"}>
                          {awayLabel}
                        </span>
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                        <div className="flex h-full">
                          <div
                            className={actualOutcome === "home" ? "h-full bg-emerald-400/95" : "h-full bg-emerald-200/35"}
                            style={{ width: `${homePercent}%` }}
                          />
                          {allowDraw ? (
                            <div
                              className={actualOutcome === "draw" ? "h-full bg-slate-500/80" : "h-full bg-slate-300/35"}
                              style={{ width: `${drawPercent}%` }}
                            />
                          ) : null}
                          <div
                            className={actualOutcome === "away" ? "h-full bg-rose-400/95" : "h-full bg-rose-200/35"}
                            style={{ width: `${awayPercent}%` }}
                          />
                        </div>
                      </div>
                    </div>
                    {hasExpandableDetails && expanded ? (
                      <div className="mt-3 border-t border-slate-100 pt-3">
                        <div className="space-y-3">
                            {conditional ? (
                              <ConditionalAdvancementPanel
                                conditional={conditional}
                                probabilityMode={probabilityMode}
                                homeTeam={match.homeTeam}
                                awayTeam={match.awayTeam}
                              />
                            ) : null}
                            {expectedGoals ? (
                              <div>
                                <div className="text-[10px] uppercase tracking-wide text-slate-500">
                                  Expected Goals
                                </div>
                                <div className="mt-2 grid grid-cols-[minmax(0,1fr)_auto_auto_minmax(0,1fr)] gap-px overflow-hidden rounded-md border border-slate-200 text-[11px] text-slate-700">
                                  <div className="bg-slate-50 px-2 py-1 text-left font-semibold text-slate-500">
                                    {match.homeTeam}
                                  </div>
                                  <div className="bg-white px-2 py-1 text-center font-mono tabular-nums">
                                    {expectedGoals.home.toFixed(2)}
                                  </div>
                                  <div className="bg-white px-2 py-1 text-center font-mono tabular-nums">
                                    {expectedGoals.away.toFixed(2)}
                                  </div>
                                  <div className="bg-slate-50 px-2 py-1 text-right font-semibold text-slate-500">
                                    {match.awayTeam}
                                  </div>
                                </div>
                              </div>
                            ) : null}
                            {totalGoalsRow ? (
                              <div>
                                <div className="text-[10px] uppercase tracking-wide text-slate-500">
                                  Total Goals
                                </div>
                                <div className="mt-2 grid w-full grid-cols-7 gap-px overflow-hidden rounded-md border border-slate-200 text-[10px] text-slate-600">
                                  {totalGoalsRow.map((cell, index) => (
                                    <div key={`past-total-goals-label-${match.matchId}-${index}`} className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500">
                                      {cell.label}
                                    </div>
                                  ))}
                                  {totalGoalsRow.map((cell, index) => (
                                    <div
                                      key={`past-total-goals-value-${match.matchId}-${index}`}
                                      className="bg-white px-1 py-1 text-center tabular-nums"
                                      style={scoreMatrixHighlight(cell.value)}
                                    >
                                      {formatProbabilityLabel(cell.value, probabilityMode, true).replace("%", "")}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                            {marginRow ? (
                              <div>
                                <div className="text-[10px] uppercase tracking-wide text-slate-500">
                                  Margin
                                </div>
                                <div className="mt-2 grid w-full grid-cols-7 gap-px overflow-hidden rounded-md border border-slate-200 text-[10px] text-slate-600">
                                  {marginRow.map((cell, index) => (
                                    <div key={`past-margin-label-${match.matchId}-${index}`} className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500">
                                      {cell.label}
                                    </div>
                                  ))}
                                  {marginRow.map((cell, index) => (
                                    <div
                                      key={`past-margin-value-${match.matchId}-${index}`}
                                      className={`bg-white px-1 py-1 text-center tabular-nums ${index === actualMarginIndex ? "text-blue-800" : "opacity-30"}`}
                                      style={index === actualMarginIndex ? scoreMatrixHighlight(cell.value) : undefined}
                                    >
                                      {formatProbabilityLabel(cell.value, probabilityMode, true).replace("%", "")}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                            {scoreGrid ? (
                              <div>
                                <div className="text-[10px] uppercase tracking-wide text-slate-500">
                                  Exact score
                                </div>
                                <div className="mt-2 w-full overflow-x-auto">
                                  <div className="min-w-full w-full">
                                    <div className="grid w-full min-w-[22rem] grid-cols-[1.5rem_repeat(6,minmax(0,1fr))] gap-px overflow-hidden rounded-md border border-slate-200 text-[10px] text-slate-600">
                                      <div className="bg-slate-50 px-1 py-1 text-center font-semibold uppercase text-slate-500">
                                        H/A
                                      </div>
                                      {SCORE_LABELS.map((label) => (
                                        <div
                                          key={`past-col-${match.matchId}-${label}`}
                                          className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500"
                                        >
                                          {label}
                                        </div>
                                      ))}
                                      {scoreGrid.map((row, rowIndex) => (
                                        <React.Fragment key={`past-row-${match.matchId}-${rowIndex}`}>
                                          <div className="bg-slate-50 px-1 py-1 text-center font-semibold text-slate-500">
                                            {SCORE_LABELS[rowIndex]}
                                          </div>
                                          {row.map((value, colIndex) => {
                                            const isActual =
                                              rowIndex === actualHomeBucket &&
                                              colIndex === actualAwayBucket;
                                            return (
                                              <div
                                                key={`past-cell-${match.matchId}-${rowIndex}-${colIndex}`}
                                                className={`bg-white px-1 py-1 text-center tabular-nums ${isActual ? "text-blue-800" : "opacity-25"}`}
                                                style={isActual ? scoreMatrixHighlight(value) : undefined}
                                              >
                                                {formatProbabilityLabel(
                                                  value,
                                                  probabilityMode,
                                                  true
                                                ).replace("%", "")}
                                              </div>
                                            );
                                          })}
                                        </React.Fragment>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ) : null}
                          </div>
                      </div>
                    ) : null}
                      <div className="mt-auto pt-3 flex items-end justify-between gap-3">
                        <div className="text-sm text-slate-600 w-[60%] whitespace-normal break-words">
                          {match.city || match.country
                            ? `${match.city}${match.city && match.country ? ", " : ""}${match.country}`
                          : "TBD"}
                      </div>
                      <div className="text-xs uppercase tracking-wide text-slate-500 text-right w-[40%] whitespace-normal break-words">
                          {match.stage}
                        </div>
                      </div>
                      {hasExpandableDetails ? (
                        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex translate-y-1/2 justify-center">
                          <div className="pointer-events-auto bg-white px-1">
                            <ExpandToggle
                              expanded={expanded}
                              onToggle={() => toggleExpanded(matchKey)}
                            />
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </div>
                );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
