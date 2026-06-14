"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { loadWorldCupPredictorDataClient } from "@/lib/world-cup-predictor-client";
import { FLAG_COLORS } from "@/lib/flag-colors";
import type {
  GroupDefinition,
  GroupMatch,
  KnockoutMatch,
  QualifierMatch,
  RoundOf32Combos,
  TeamStageProbabilities,
  WinProbabilities,
  WinProbabilityEntry,
  WorldCupPredictorData,
} from "@/lib/world-cup-predictor-types";
import { buildScoreMatrix } from "@/lib/score-matrix";
import {
  isCompactWinProbabilities,
  parseCompactEntry,
  resolveCompactEntry,
} from "@/lib/win-probabilities";

type MatchScore = { home: number | null; away: number | null };
type WinnerSelection = "home" | "away" | null;

type GroupTableRow = {
  team: string;
  group: string;
  played: number;
  wins: number;
  draws: number;
  losses: number;
  gf: number;
  ga: number;
  gd: number;
  points: number;
  position: number;
  randomTiebreak?: boolean;
};

type ResolvedQualifierMatch = QualifierMatch & {
  homeResolved: string;
  awayResolved: string;
  winner?: string;
};

type ResolvedGroupMatch = GroupMatch & {
  homeTeam: string;
  awayTeam: string;
};

type ResolvedKnockoutMatch = KnockoutMatch & {
  homeResolved: string;
  awayResolved: string;
  winner?: string;
};

const SKIP_INITIALS = new Set(["and", "of", "the"]);

// Get flag colors for a team, with fallback to default colors
const getTeamFlagColors = (team: string): string[] => {
  // Try exact match first
  if (FLAG_COLORS[team]) {
    return FLAG_COLORS[team];
  }
  // Try with underscores instead of spaces
  const teamWithUnderscores = team.replace(/\s+/g, '_');
  if (FLAG_COLORS[teamWithUnderscores]) {
    return FLAG_COLORS[teamWithUnderscores];
  }
  // Fallback to default vibrant colors
  return ['#FF0000', '#0000FF', '#FFFF00', '#00FF00'];
};

// Confetti particle component - positioned relative to container
const ConfettiParticle: React.FC<{
  delay: number;
  duration: number;
  color: string;
  funny?: boolean;
}> = ({ delay, duration, color, funny = false }) => {
  const angle = React.useRef(funny ? 45 + Math.random() * 90 : Math.random() * 360);
  const distance = React.useRef(200 + Math.random() * 300);
  const rotation = React.useRef(Math.random() * 720 - 360);
  const size = React.useRef(8 + Math.random() * 6);
  const isCircle = React.useRef(Math.random() > 0.5);

  const x = Math.cos((angle.current * Math.PI) / 180) * distance.current;
  const y = Math.sin((angle.current * Math.PI) / 180) * distance.current;
  const dropletOrientation = angle.current - 90;
  const startRotation = funny ? dropletOrientation : 0;
  const finalRotation = funny ? dropletOrientation : rotation.current;

  // Create unique animation ID for this particle
  const animationId = React.useRef(`confetti-${Math.random().toString(36).substr(2, 9)}`);

  // Inject keyframe animation for this specific particle
  React.useEffect(() => {
    const styleId = `style-${animationId.current}`;
    let styleEl = document.getElementById(styleId) as HTMLStyleElement;
    
    if (!styleEl) {
      styleEl = document.createElement('style');
      styleEl.id = styleId;
      document.head.appendChild(styleEl);
    }
    
    styleEl.textContent = `
      @keyframes ${animationId.current} {
        0% {
          opacity: 1;
          transform: translate(-50%, -50%) translate(0, 0) rotate(${startRotation}deg);
        }
        100% {
          opacity: 0;
          transform: translate(-50%, -50%) translate(${x}px, ${y}px) rotate(${finalRotation}deg);
        }
      }
    `;

    return () => {
      const el = document.getElementById(styleId);
      if (el) {
        document.head.removeChild(el);
      }
    };
  }, [x, y, startRotation, finalRotation]);

  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: '50%',
        top: '50%',
        width: funny ? `${Math.max(6, size.current * 0.72)}px` : `${size.current}px`,
        height: funny ? `${Math.max(12, size.current * 1.8)}px` : `${size.current}px`,
        backgroundColor: color,
        borderRadius: funny
          ? '58% 58% 68% 68% / 14% 14% 88% 88%'
          : isCircle.current
            ? '50%'
            : '0%',
        clipPath: funny
          ? 'polygon(50% 0%, 61% 14%, 69% 28%, 77% 45%, 82% 63%, 80% 79%, 70% 91%, 58% 98%, 50% 100%, 42% 98%, 30% 91%, 20% 79%, 18% 63%, 23% 45%, 31% 28%, 39% 14%)'
          : undefined,
        animation: `${animationId.current} ${duration}ms ease-out ${delay}ms forwards`,
        opacity: 1,
      }}
    />
  );
};

// Confetti animation component - renders particles relative to its container
const ConfettiAnimation: React.FC<{ duration: number; champion: string; funny?: boolean }> = ({
  duration,
  champion,
  funny = false,
}) => {
  const [particles, setParticles] = React.useState<Array<{ id: number; delay: number; color: string }>>([]);
  const flagColors = React.useMemo(() => getTeamFlagColors(champion), [champion]);

  React.useEffect(() => {
    // Generate 50 confetti particles using team flag colors
    const newParticles = Array.from({ length: 50 }, (_, i) => ({
      id: i,
      delay: Math.random() * 300,
      color: flagColors[Math.floor(Math.random() * flagColors.length)],
    }));
    setParticles(newParticles);

    // Clean up after animation
    const timer = setTimeout(() => {
      setParticles([]);
    }, duration + 500);

    return () => {
      clearTimeout(timer);
    };
  }, [duration, champion, flagColors]);

  if (particles.length === 0) return null;

  return (
    <>
      {particles.map((particle) => (
        <ConfettiParticle
          key={particle.id}
          delay={particle.delay}
          duration={duration}
          color={particle.color}
          funny={funny}
        />
      ))}
    </>
  );
};

const HOST_TEAM_COUNTRIES: Record<string, string> = {
  USA: "USA",
  "United States": "USA",
  Canada: "Canada",
  Mexico: "Mexico",
};
const HOST_TEAMS = new Set(["USA", "Canada", "Mexico"]);
const TIEBREAK_TOOLTIP =
  "Tiebreakers have been chosen randomly but would be determined by Fair Play Points in reality.";

type MatchProbabilityValues = {
  home: number | null;
  draw: number | null;
  away: number | null;
};

type MatchProbabilityLabels = {
  homeWinProb?: string;
  awayWinProb?: string;
  drawProb?: string | null;
};

type ScoredWinnerSample = {
  selection: WinnerSelection;
  probability: number;
};

type ScoredScoreSample = {
  home: number;
  away: number;
  probability: number;
};

type AutopredictSnapshot = {
  qualifierWinners: Record<string, WinnerSelection>;
  autoQualifierWinners: Record<string, boolean>;
  groupScores: Record<string, MatchScore>;
  autoGroupScores: Record<string, boolean>;
  knockoutWinners: Record<string, WinnerSelection>;
  autoKnockoutWinners: Record<string, boolean>;
  funninessScore: number;
};

function parseFunnyRuns(value: string | null) {
  if (!value) {
    return null;
  }
  if (!/^\d+$/.test(value)) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    return null;
  }
  return parsed;
}

function funnyWeightForStage(stage: string | undefined) {
  switch (stage) {
    case "Round of 32":
      return 3;
    case "Round of 16":
      return 5;
    case "Quarterfinal":
      return 10;
    case "Semifinal":
      return 20;
    case "Final":
      return 50;
    default:
      return 1;
  }
}

function probabilityPenalty(probability: number, weight = 1) {
  const clamped = Math.max(probability, Number.MIN_VALUE);
  return Math.log(clamped) * weight;
}

type FinalProgressionStage =
  | "Group stage"
  | "Round of 32"
  | "Round of 16"
  | "Quarterfinal"
  | "Semifinal"
  | "Third place"
  | "Reach Final"
  | "Champion";

function exactProgressionProbability(
  probabilities: TeamStageProbabilities | undefined,
  stage: FinalProgressionStage
) {
  if (!probabilities) {
    return null;
  }
  const p = probabilities.stage_probability ?? {};
  const reachR32 = Number(p["Reach R32"] ?? 0);
  const reachR16 = Number(p["Reach R16"] ?? 0);
  const reachQF = Number(p["Reach QF"] ?? 0);
  const reachSF = Number(p["Reach SF"] ?? 0);
  const thirdPlace = Number(p["Third place"] ?? 0);
  const reachFinal = Number(p["Reach Final"] ?? 0);
  const champion = Number(p["Champion"] ?? 0);
  const exactByStage: Record<FinalProgressionStage, number> = {
    "Group stage": 1 - reachR32,
    "Round of 32": reachR32 - reachR16,
    "Round of 16": reachR16 - reachQF,
    Quarterfinal: reachQF - reachSF,
    Semifinal: reachSF - reachFinal - thirdPlace,
    "Third place": thirdPlace,
    "Reach Final": reachFinal - champion,
    Champion: champion,
  };
  return Math.max(exactByStage[stage] ?? 0, 0);
}

function scoreResultProbability(
  scoreMatrix: number[][],
  sample: Pick<ScoredScoreSample, "home" | "away">
) {
  let total = 0;
  let resultTotal = 0;
  for (let i = 0; i < scoreMatrix.length; i += 1) {
    const row = scoreMatrix[i];
    for (let j = 0; j < row.length; j += 1) {
      const value = row[j];
      if (!Number.isFinite(value) || value <= 0) {
        continue;
      }
      total += value;
      const matchesResult =
        sample.home === sample.away
          ? i === j
          : sample.home > sample.away
            ? i > j
            : i < j;
      if (matchesResult) {
        resultTotal += value;
      }
    }
  }
  if (total <= 0 || resultTotal <= 0) {
    return null;
  }
  return resultTotal / total;
}

function groupScorePenalty(scoreMatrix: number[][], sample: ScoredScoreSample) {
  const resultProbability = scoreResultProbability(scoreMatrix, sample);
  if (resultProbability === null) {
    return probabilityPenalty(sample.probability);
  }
  return (
    probabilityPenalty(resultProbability, 0.8) +
    probabilityPenalty(sample.probability, 0.2)
  );
}

function teamInitials(team: string) {
  const letters = team
    .split(/\s+/)
    .filter((word) => word && !SKIP_INITIALS.has(word.toLowerCase()))
    .map((word) => word[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();
  return letters || team.slice(0, 2).toUpperCase();
}

type LoadingButtonProps = {
  loading: boolean;
  onClick: () => void;
  disabled?: boolean;
  className?: string;
  children: React.ReactNode;
};

const LoadingButton: React.FC<LoadingButtonProps> = ({
  loading,
  onClick,
  disabled = false,
  className,
  children,
}) => {
  const isDisabled = loading || disabled;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={isDisabled}
      aria-busy={loading}
      className={cn(
        "relative overflow-hidden transition-colors",
        loading && "cursor-wait",
        isDisabled && !loading && "cursor-default",
        className
      )}
    >
      <span
        className="absolute inset-0 bg-slate-200/70 origin-left"
        style={{
          transform: loading ? "scaleX(1)" : "scaleX(0)",
          transition: loading ? "transform 300ms linear" : "none",
        }}
        aria-hidden="true"
      />
      <span className="relative z-10">{children}</span>
    </button>
  );
};

function isPlaceholderLabel(name: string) {
  if (!name) {
    return true;
  }
  return (
    name.includes("Winner Match") ||
    name.includes("Loser Match") ||
    name.includes("Winner Group") ||
    name.includes("Runner-up Group") ||
    name.includes("3rd Group") ||
    /^TBD$/i.test(name) ||
    /^UEFA Path /i.test(name) ||
    /^IC Path /i.test(name) ||
    /^([123](st|nd|rd)) Group /i.test(name) ||
    /^([123](st|nd|rd)) Gr\. /i.test(name) ||
    /^Winner semi/i.test(name) ||
    /^Winner IC Path/i.test(name) ||
    /^Winner UEFA Path/i.test(name) ||
    /^Winner (R32|R16|QF|SF|Final)$/i.test(name) ||
    /^Loser (R32|R16|QF|SF|Final)$/i.test(name) ||
    /winner$/i.test(name)
  );
}

function isConcreteTeam(name: string | null | undefined) {
  if (!name) {
    return false;
  }
  return !isPlaceholderLabel(name);
}

function formatDisplayLabel(label: string) {
  if (!label) {
    return label;
  }
  return label
    .replace(/^Winner\s+UEFA Path\s+/i, "UEFA Path ")
    .replace(/^Winner\s+IC Path\s+/i, "IC Path ")
    .replace(/^UEFA Path\s+(.+)\s+Winner$/i, "UEFA Path $1")
    .replace(/^IC Path\s+(.+)\s+Winner$/i, "IC Path $1")
    .replace(/^Winner\s+Semi(?:final)?\b/i, "Winner SF");
}

function formatQualifierSource(source: string | undefined) {
  if (!source) {
    return source;
  }
  if (source === "semi1" || source === "semi2") {
    return "Semifinal";
  }
  return source;
}

function hashString(value: string) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function createRng(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), 1 | x);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace<T>(items: T[], rng: () => number) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function seedFromGroupState(
  group: GroupDefinition,
  matches: GroupMatch[],
  scores: Record<string, MatchScore>
) {
  const parts = [group.id];
  const orderedMatches = [...matches].sort((a, b) => a.id - b.id);
  for (const match of orderedMatches) {
    const score = scores[String(match.id)];
    const home = score?.home ?? "x";
    const away = score?.away ?? "x";
    parts.push(`${match.id}:${home}-${away}`);
  }
  return hashString(parts.join("|"));
}

function seedFromThirdPlace(entries: Array<{ team: string; group: string; points: number; gd: number; gf: number }>) {
  const parts = entries.map(
    (entry) => `${entry.group}:${entry.team}:${entry.points}:${entry.gd}:${entry.gf}`
  );
  return hashString(parts.join("|"));
}

function extractGroupId(label: string) {
  const match = label.match(/Group\s+([A-Z])/i);
  return match?.[1] ? match[1].toUpperCase() : null;
}

function parseScore(value: string) {
  if (!value) {
    return null;
  }
  const parsed = parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  const clamped = Math.max(0, Math.min(31, parsed));
  return clamped;
}

function formatProbability(value: number | null | undefined, forceDecimal = false) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return undefined;
  }
  const percent = value * 100;
  if (percent < 0.1) {
    return "<0.1%";
  }
  if (percent > 99.9) {
    return ">99.9%";
  }
  if (forceDecimal || percent < 0.5 || percent >= 99.5) {
    return `${percent.toFixed(1)}%`;
  }
  return `${Math.round(percent)}%`;
}

function shouldUseDecimalPrecision(values: (number | null | undefined)[]) {
  // If any probability would round to 0 (< 0.5%), use decimal precision for all
  return values.some((v) => v !== null && v !== undefined && Number.isFinite(v) && v * 100 < 0.5);
}

function parseProbabilityLabel(label?: string | null) {
  if (!label) {
    return null;
  }
  // Handle "<0.1%" format
  if (label === "<0.1%") {
    return 0.05; // Use midpoint for bar width calculation
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

function formatSegmentDisplay(value: number): string {
  if (value < 0.1) {
    return "<0.1";
  }
  if (value > 99.9) {
    return ">99.9";
  }
  if (value !== Math.round(value)) {
    return value.toFixed(1);
  }
  return String(value);
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
  // Check if any value has a decimal (indicating precision mode)
  const hasDecimal = raw.some((v) => v !== Math.round(v));
  if (hasDecimal) {
    // Preserve 1dp precision, normalize to sum to 100
    const rounded = raw.map((value) => Number(value.toFixed(1)));
    const total = rounded.reduce((sum, value) => sum + value, 0);
    const remainder = Number((100 - total).toFixed(1));
    if (Math.abs(remainder) >= 0.05) {
      // Adjust draw to make sum 100
      rounded[1] = Math.max(0, Number((rounded[1] + remainder).toFixed(1)));
    }
    return { home: rounded[0], draw: rounded[1], away: rounded[2] };
  }
  // Integer mode
  const rounded = raw.map((value) => Math.round(value));
  const total = rounded.reduce((sum, value) => sum + value, 0);
  const remainder = 100 - total;
  if (remainder !== 0) {
    const targetIndex = 1;
    rounded[targetIndex] = Math.max(0, rounded[targetIndex] + remainder);
  }
  return { home: rounded[0], draw: rounded[1], away: rounded[2] };
}

function normalizeTwoSegments(values: { home: number | null; away: number | null }) {
  const { home, away } = values;
  if (home === null || away === null) {
    return null;
  }
  // Check if any value has a decimal (indicating precision mode)
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
  // Integer mode
  const roundedHome = Math.round(home);
  const roundedAway = Math.round(away);
  const remainder = 100 - (roundedHome + roundedAway);
  return {
    home: Math.max(0, roundedHome + remainder),
    away: roundedAway,
  };
}

function normalizeCountry(value: string | null | undefined) {
  return value ? value.trim().toLowerCase() : "";
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
    const homeAdvantage =
      homeCountry && matchCountry && homeCountry === matchCountry;
    const awayAdvantage =
      awayCountry && matchCountry && awayCountry === matchCountry;
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
  if (
    !probabilities ||
    isPlaceholderLabel(homeTeam) ||
    isPlaceholderLabel(awayTeam)
  ) {
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
  if (
    !probabilities ||
    isPlaceholderLabel(homeTeam) ||
    isPlaceholderLabel(awayTeam)
  ) {
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

function sampleScoreMatrix(scoreMatrix: number[][]) {
  let total = 0;
  for (const row of scoreMatrix) {
    for (const value of row) {
      if (Number.isFinite(value) && value > 0) {
        total += value;
      }
    }
  }
  if (total <= 0) {
    return null;
  }
  const target = Math.random() * total;
  let cumulative = 0;
  for (let i = 0; i < scoreMatrix.length; i += 1) {
    const row = scoreMatrix[i];
    for (let j = 0; j < row.length; j += 1) {
      const value = row[j];
      if (!Number.isFinite(value) || value <= 0) {
        continue;
      }
      cumulative += value;
      if (cumulative >= target) {
        return { home: Math.min(i, 31), away: Math.min(j, 31) };
      }
    }
  }
  return { home: 0, away: 0 };
}

function sampleScoreMatrixByResult(
  scoreMatrix: number[][],
  result: "home" | "away" | "draw"
) {
  let total = 0;
  for (let i = 0; i < scoreMatrix.length; i += 1) {
    const row = scoreMatrix[i];
    for (let j = 0; j < row.length; j += 1) {
      const value = row[j];
      if (!Number.isFinite(value) || value <= 0) {
        continue;
      }
      const matchesResult =
        result === "draw" ? i === j : result === "home" ? i > j : i < j;
      if (matchesResult) {
        total += value;
      }
    }
  }
  if (total <= 0) {
    return null;
  }
  const target = Math.random() * total;
  let cumulative = 0;
  for (let i = 0; i < scoreMatrix.length; i += 1) {
    const row = scoreMatrix[i];
    for (let j = 0; j < row.length; j += 1) {
      const value = row[j];
      if (!Number.isFinite(value) || value <= 0) {
        continue;
      }
      const matchesResult =
        result === "draw" ? i === j : result === "home" ? i > j : i < j;
      if (!matchesResult) {
        continue;
      }
      cumulative += value;
      if (cumulative >= target) {
        return { home: Math.min(i, 31), away: Math.min(j, 31) };
      }
    }
  }
  return null;
}

function sampleWinner(values: MatchProbabilityValues | null): WinnerSelection {
  if (!values || values.home === null || values.away === null) {
    return null;
  }
  const total = values.home + values.away;
  if (!Number.isFinite(total) || total <= 0) {
    return null;
  }
  const roll = Math.random() * total;
  return roll < values.home ? "home" : "away";
}

function sampleWinnerWithProbability(
  values: MatchProbabilityValues | null
): ScoredWinnerSample | null {
  if (!values || values.home === null || values.away === null) {
    return null;
  }
  const total = values.home + values.away;
  if (!Number.isFinite(total) || total <= 0) {
    return null;
  }
  const roll = Math.random() * total;
  if (roll < values.home) {
    return { selection: "home", probability: values.home / total };
  }
  return { selection: "away", probability: values.away / total };
}

function sampleScoreMatrixWithProbability(
  scoreMatrix: number[][]
): ScoredScoreSample | null {
  let total = 0;
  for (const row of scoreMatrix) {
    for (const value of row) {
      if (Number.isFinite(value) && value > 0) {
        total += value;
      }
    }
  }
  if (total <= 0) {
    return null;
  }
  const target = Math.random() * total;
  let cumulative = 0;
  for (let i = 0; i < scoreMatrix.length; i += 1) {
    const row = scoreMatrix[i];
    for (let j = 0; j < row.length; j += 1) {
      const value = row[j];
      if (!Number.isFinite(value) || value <= 0) {
        continue;
      }
      cumulative += value;
      if (cumulative >= target) {
        return {
          home: Math.min(i, 31),
          away: Math.min(j, 31),
          probability: value / total,
        };
      }
    }
  }
  return null;
}

const SHARE_VERSION = 1;
function progressionPlacementMultiplier(stage: FinalProgressionStage) {
  switch (stage) {
    case "Champion":
      return 30;
    case "Reach Final":
      return 10;
    case "Third place":
    case "Semifinal":
      return 5;
    default:
      return 1;
  }
}

class BitWriter {
  private bytes: number[] = [];
  private current = 0;
  private bitPos = 0;

  writeBits(value: number, count: number) {
    for (let i = count - 1; i >= 0; i -= 1) {
      const bit = (value >> i) & 1;
      this.current = (this.current << 1) | bit;
      this.bitPos += 1;
      if (this.bitPos === 8) {
        this.bytes.push(this.current);
        this.current = 0;
        this.bitPos = 0;
      }
    }
  }

  toUint8Array() {
    if (this.bitPos > 0) {
      this.bytes.push(this.current << (8 - this.bitPos));
    }
    return new Uint8Array(this.bytes);
  }
}

class BitReader {
  private bytes: Uint8Array;
  private index = 0;
  private bitPos = 0;

  constructor(bytes: Uint8Array) {
    this.bytes = bytes;
  }

  readBits(count: number) {
    let value = 0;
    for (let i = 0; i < count; i += 1) {
      if (this.index >= this.bytes.length) {
        return null;
      }
      const byte = this.bytes[this.index];
      const bit = (byte >> (7 - this.bitPos)) & 1;
      value = (value << 1) | bit;
      this.bitPos += 1;
      if (this.bitPos === 8) {
        this.bitPos = 0;
        this.index += 1;
      }
    }
    return value;
  }
}

function bytesToBase64Url(bytes: Uint8Array) {
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function base64UrlToBytes(token: string) {
  try {
    const padded = token.padEnd(Math.ceil(token.length / 4) * 4, "=");
    const base64 = padded.replace(/-/g, "+").replace(/_/g, "/");
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  } catch {
    return null;
  }
}

function encodeShareStateCompact(params: {
  qualifiers: QualifierMatch[];
  groupMatches: GroupMatch[];
  knockouts: KnockoutMatch[];
  qualifierWinners: Record<string, WinnerSelection>;
  groupScores: Record<string, MatchScore>;
  knockoutWinners: Record<string, WinnerSelection>;
}) {
  const writer = new BitWriter();
  writer.writeBits(SHARE_VERSION, 4);

  const qualifiers = [...params.qualifiers].sort((a, b) =>
    String(a.id).localeCompare(String(b.id))
  );
  qualifiers.forEach((match) => {
    const selection = params.qualifierWinners[String(match.id)] ?? null;
    const code = selection === "home" ? 1 : selection === "away" ? 2 : 0;
    writer.writeBits(code, 2);
  });

  const groups = [...params.groupMatches].sort((a, b) => a.id - b.id);
  groups.forEach((match) => {
    const score = params.groupScores[String(match.id)];
    const hasHome = score?.home !== null && score?.home !== undefined;
    const hasAway = score?.away !== null && score?.away !== undefined;
    writer.writeBits(hasHome ? 1 : 0, 1);
    if (hasHome) {
      writer.writeBits(score?.home ?? 0, 5);
    }
    writer.writeBits(hasAway ? 1 : 0, 1);
    if (hasAway) {
      writer.writeBits(score?.away ?? 0, 5);
    }
  });

  const knockouts = [...params.knockouts].sort((a, b) => a.id - b.id);
  knockouts.forEach((match) => {
    const selection = params.knockoutWinners[String(match.id)] ?? null;
    const code = selection === "home" ? 1 : selection === "away" ? 2 : 0;
    writer.writeBits(code, 2);
  });

  return bytesToBase64Url(writer.toUint8Array());
}

function decodeShareStateCompact(
  token: string,
  params: {
    qualifiers: QualifierMatch[];
    groupMatches: GroupMatch[];
    knockouts: KnockoutMatch[];
  }
) {
  const bytes = base64UrlToBytes(token);
  if (!bytes) {
    return null;
  }
  const reader = new BitReader(bytes);
  const version = reader.readBits(4);
  if (version !== SHARE_VERSION) {
    return null;
  }

  const qualifiers = [...params.qualifiers].sort((a, b) =>
    String(a.id).localeCompare(String(b.id))
  );
  const qualifierWinners: Record<string, WinnerSelection> = {};
  for (const match of qualifiers) {
    const code = reader.readBits(2);
    if (code === null) {
      return null;
    }
    qualifierWinners[String(match.id)] = code === 1 ? "home" : code === 2 ? "away" : null;
  }

  const groups = [...params.groupMatches].sort((a, b) => a.id - b.id);
  const groupScores: Record<string, MatchScore> = {};
  for (const match of groups) {
    const hasHome = reader.readBits(1);
    if (hasHome === null) {
      return null;
    }
    let home: number | null = null;
    if (hasHome === 1) {
      const value = reader.readBits(5);
      if (value === null) {
        return null;
      }
      home = value;
    }
    const hasAway = reader.readBits(1);
    if (hasAway === null) {
      return null;
    }
    let away: number | null = null;
    if (hasAway === 1) {
      const value = reader.readBits(5);
      if (value === null) {
        return null;
      }
      away = value;
    }
    if (home !== null || away !== null) {
      groupScores[String(match.id)] = { home, away };
    }
  }

  const knockouts = [...params.knockouts].sort((a, b) => a.id - b.id);
  const knockoutWinners: Record<string, WinnerSelection> = {};
  for (const match of knockouts) {
    const code = reader.readBits(2);
    if (code === null) {
      return null;
    }
    knockoutWinners[String(match.id)] = code === 1 ? "home" : code === 2 ? "away" : null;
  }

  return { qualifierWinners, groupScores, knockoutWinners };
}

function clearDependentScores(
  scores: Record<string, MatchScore>,
  matchId: string,
  dependents: Map<string, Set<string>>
) {
  const next = { ...scores };
  const visited = new Set<string>();
  const stack = [matchId];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) {
      continue;
    }
    const deps = dependents.get(current);
    if (!deps) {
      continue;
    }
    for (const dep of deps) {
      if (visited.has(dep)) {
        continue;
      }
      visited.add(dep);
      if (next[dep]) {
        next[dep] = { home: null, away: null };
      }
      stack.push(dep);
    }
  }
  return next;
}

function resolveQualifierState(
  qualifiers: QualifierMatch[],
  qualifierWinners: Record<string, WinnerSelection>
) {
  const sorted = sortQualifiers(qualifiers);
  let winnersByPathRound = new Map<string, string>();
  let slotWinners = new Map<string, string>();
  let resolvedMatches: ResolvedQualifierMatch[] = [];
  let changed = true;
  let iterations = 0;

  while (changed && iterations < 4) {
    iterations += 1;
    changed = false;
    const nextWinners = new Map<string, string>();
    const nextSlots = new Map<string, string>();
    const nextResolved: ResolvedQualifierMatch[] = [];

    for (const match of sorted) {
      const homeResolved =
        match.homeTeam ||
        (match.homeSource
          ? winnersByPathRound.get(`${match.path}:${match.homeSource}`) ??
            `Winner ${formatQualifierSource(match.homeSource)}`
          : "");
      const awayResolved =
        match.awayTeam ||
        (match.awaySource
          ? winnersByPathRound.get(`${match.path}:${match.awaySource}`) ??
            `Winner ${formatQualifierSource(match.awaySource)}`
          : "");
      const isPickableMatch =
        isConcreteTeam(homeResolved) && isConcreteTeam(awayResolved);
      const winner = isPickableMatch
        ? resolveWinner(
            match.id,
            homeResolved,
            awayResolved,
            {},
            false,
            qualifierWinners
          )
        : undefined;
      if (winner) {
        const key = `${match.path}:${match.round}`;
        if (winnersByPathRound.get(key) !== winner) {
          changed = true;
        }
        nextWinners.set(key, winner);
        if (match.winnerSlot) {
          if (slotWinners.get(match.winnerSlot) !== winner) {
            changed = true;
          }
          nextSlots.set(match.winnerSlot, winner);
        }
      }
      nextResolved.push({
        ...match,
        homeResolved,
        awayResolved,
        winner,
      });
    }

    winnersByPathRound = nextWinners;
    slotWinners = nextSlots;
    resolvedMatches = nextResolved;
  }

  return { matches: resolvedMatches, slotWinners };
}

function clearDependentSelections(
  selections: Record<string, WinnerSelection>,
  matchId: string,
  dependents: Map<string, Set<string>>
) {
  const next = { ...selections };
  const visited = new Set<string>();
  const stack = [matchId];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) {
      continue;
    }
    const deps = dependents.get(current);
    if (!deps) {
      continue;
    }
    for (const dep of deps) {
      if (visited.has(dep)) {
        continue;
      }
      visited.add(dep);
      if (next[dep] !== null) {
        next[dep] = null;
      }
      stack.push(dep);
    }
  }
  return next;
}

function useMediaQuery(query: string) {
  const [matches, setMatches] = React.useState(false);

  React.useEffect(() => {
    const media = window.matchMedia(query);
    const handler = () => setMatches(media.matches);
    handler();
    media.addEventListener("change", handler);
    return () => media.removeEventListener("change", handler);
  }, [query]);

  return matches;
}

// FIFA three-letter country codes mapping
const FIFA_CODES: Record<string, string> = {
  "Algeria": "ALG",
  "Argentina": "ARG",
  "Australia": "AUS",
  "Austria": "AUT",
  "Belgium": "BEL",
  "Bolivia": "BOL",
  "Bosnia and Herzegovina": "BIH",
  "Brazil": "BRA",
  "Canada": "CAN",
  "Cape Verde": "CPV",
  "Colombia": "COL",
  "Croatia": "CRO",
  "Curacao": "CUW",
  "Czechia": "CZE",
  "Denmark": "DEN",
  "DR Congo": "COD",
  "Ecuador": "ECU",
  "Egypt": "EGY",
  "England": "ENG",
  "France": "FRA",
  "Germany": "GER",
  "Ghana": "GHA",
  "Haiti": "HTI",
  "Iran": "IRN",
  "Iraq": "IRQ",
  "Italy": "ITA",
  "Ivory Coast": "CIV",
  "Jamaica": "JAM",
  "Japan": "JPN",
  "Jordan": "JOR",
  "Kosovo": "KOS",
  "Mexico": "MEX",
  "Morocco": "MAR",
  "Netherlands": "NED",
  "New Caledonia": "NCL",
  "New Zealand": "NZL",
  "North Macedonia": "MKD",
  "Northern Ireland": "NIR",
  "Norway": "NOR",
  "Panama": "PAN",
  "Paraguay": "PAR",
  "Poland": "POL",
  "Portugal": "POR",
  "Qatar": "QAT",
  "Republic of Ireland": "IRL",
  "Romania": "ROU",
  "Saudi Arabia": "KSA",
  "Scotland": "SCO",
  "Senegal": "SEN",
  "Slovakia": "SVK",
  "South Africa": "RSA",
  "South Korea": "KOR",
  "Spain": "ESP",
  "Suriname": "SUR",
  "Sweden": "SWE",
  "Switzerland": "SUI",
  "Tunisia": "TUN",
  "Turkey": "TUR",
  "USA": "USA",
  "Uruguay": "URU",
  "Uzbekistan": "UZB",
  "Wales": "WAL",
};

function getFifaCode(team: string): string | null {
  return FIFA_CODES[team] ?? null;
}

function resolveWinner(
  matchId: string | number,
  homeTeam: string,
  awayTeam: string,
  scores: Record<string, MatchScore>,
  allowDraw: boolean,
  winnerSelections?: Record<string, WinnerSelection>
) {
  const selection = winnerSelections?.[String(matchId)] ?? null;
  if (selection) {
    return selection === "home" ? homeTeam : awayTeam;
  }
  const score = scores[String(matchId)];
  if (!score || score.home === null || score.away === null) {
    return undefined;
  }
  if (score.home === score.away) {
    return allowDraw ? undefined : undefined;
  }
  return score.home > score.away ? homeTeam : awayTeam;
}

function rankOverall(
  teams: string[],
  table: Record<string, GroupTableRow>,
  rng: () => number,
  randomTiebreakTeams: Set<string>
) {
  const sorted = [...teams].sort((a, b) => {
    const rowA = table[a];
    const rowB = table[b];
    if (rowB.points !== rowA.points) {
      return rowB.points - rowA.points;
    }
    if (rowB.gd !== rowA.gd) {
      return rowB.gd - rowA.gd;
    }
    if (rowB.gf !== rowA.gf) {
      return rowB.gf - rowA.gf;
    }
    return 0;
  });

  const ordered: string[] = [];
  let i = 0;
  while (i < sorted.length) {
    const current = sorted[i];
    const tied = [current];
    i += 1;
    while (i < sorted.length) {
      const next = sorted[i];
      const rowA = table[current];
      const rowB = table[next];
      if (
        rowA.points === rowB.points &&
        rowA.gd === rowB.gd &&
        rowA.gf === rowB.gf
      ) {
        tied.push(next);
        i += 1;
      } else {
        break;
      }
    }
    if (tied.length > 1) {
      shuffleInPlace(tied, rng);
      tied.forEach((team) => randomTiebreakTeams.add(team));
    }
    ordered.push(...tied);
  }
  return ordered;
}

function headToHeadTable(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>
) {
  const table: Record<string, { points: number; gf: number; ga: number; gd: number }> =
    {};
  for (const team of teams) {
    table[team] = { points: 0, gf: 0, ga: 0, gd: 0 };
  }
  for (const match of matches) {
    if (!teams.includes(match.homeTeam) || !teams.includes(match.awayTeam)) {
      continue;
    }
    const home = table[match.homeTeam];
    const away = table[match.awayTeam];
    home.gf += match.homeScore;
    home.ga += match.awayScore;
    away.gf += match.awayScore;
    away.ga += match.homeScore;
    if (match.homeScore > match.awayScore) {
      home.points += 3;
    } else if (match.homeScore < match.awayScore) {
      away.points += 3;
    } else {
      home.points += 1;
      away.points += 1;
    }
  }
  for (const team of teams) {
    const row = table[team];
    row.gd = row.gf - row.ga;
  }
  return table;
}

function rankHeadToHead(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>,
  table: Record<string, GroupTableRow>,
  rng: () => number,
  randomTiebreakTeams: Set<string>
): string[] {
  if (teams.length <= 1) {
    return teams;
  }
  const h2h = headToHeadTable(teams, matches);
  const metrics = teams.map((team) => h2h[team]);
  const allEqual =
    metrics.every((m) => m.points === metrics[0].points) &&
    metrics.every((m) => m.gd === metrics[0].gd) &&
    metrics.every((m) => m.gf === metrics[0].gf);
  if (allEqual) {
    return rankOverall(teams, table, rng, randomTiebreakTeams);
  }
  const sorted = [...teams].sort((a, b) => {
    const rowA = h2h[a];
    const rowB = h2h[b];
    if (rowB.points !== rowA.points) {
      return rowB.points - rowA.points;
    }
    if (rowB.gd !== rowA.gd) {
      return rowB.gd - rowA.gd;
    }
    if (rowB.gf !== rowA.gf) {
      return rowB.gf - rowA.gf;
    }
    return 0;
  });
  const ordered: string[] = [];
  let i = 0;
  while (i < sorted.length) {
    const current = sorted[i];
    const tied = [current];
    i += 1;
    while (i < sorted.length) {
      const next = sorted[i];
      const rowA = h2h[current];
      const rowB = h2h[next];
      if (
        rowA.points === rowB.points &&
        rowA.gd === rowB.gd &&
        rowA.gf === rowB.gf
      ) {
        tied.push(next);
        i += 1;
      } else {
        break;
      }
    }
    if (tied.length === 1) {
      ordered.push(tied[0]);
    } else {
      ordered.push(...rankHeadToHead(tied, matches, table, rng, randomTiebreakTeams));
    }
  }
  return ordered;
}

function rankGroup(
  teams: string[],
  matches: Array<{ homeTeam: string; awayTeam: string; homeScore: number; awayScore: number }>,
  table: Record<string, GroupTableRow>,
  rng: () => number,
  randomTiebreakTeams: Set<string>
) {
  const base = [...teams].sort((a, b) => {
    const rowA = table[a];
    const rowB = table[b];
    if (rowB.points !== rowA.points) {
      return rowB.points - rowA.points;
    }
    return 0;
  });

  const ranked: string[] = [];
  let i = 0;
  while (i < base.length) {
    const current = base[i];
    const tied = [current];
    i += 1;
    while (i < base.length) {
      const next = base[i];
      if (table[current].points === table[next].points) {
        tied.push(next);
        i += 1;
      } else {
        break;
      }
    }
    if (tied.length === 1) {
      ranked.push(tied[0]);
    } else {
      ranked.push(...rankHeadToHead(tied, matches, table, rng, randomTiebreakTeams));
    }
  }
  return ranked;
}

function buildGroupTable(
  group: GroupDefinition,
  matches: GroupMatch[],
  scores: Record<string, MatchScore>
) {
  const table: Record<string, GroupTableRow> = {};
  for (const team of group.teams) {
    table[team] = {
      team,
      group: group.id,
      played: 0,
      wins: 0,
      draws: 0,
      losses: 0,
      gf: 0,
      ga: 0,
      gd: 0,
      points: 0,
      position: 0,
    };
  }
  const playedMatches: Array<{
    homeTeam: string;
    awayTeam: string;
    homeScore: number;
    awayScore: number;
  }> = [];

  for (const match of matches) {
    const score = scores[String(match.id)];
    if (!score || score.home === null || score.away === null) {
      continue;
    }
    const home = table[match.homeTeam];
    const away = table[match.awayTeam];
    if (!home || !away) {
      continue;
    }
    const homeScore = score.home;
    const awayScore = score.away;
    home.gf += homeScore;
    home.ga += awayScore;
    away.gf += awayScore;
    away.ga += homeScore;
    home.played += 1;
    away.played += 1;
    if (homeScore > awayScore) {
      home.wins += 1;
      away.losses += 1;
      home.points += 3;
    } else if (awayScore > homeScore) {
      away.wins += 1;
      home.losses += 1;
      away.points += 3;
    } else {
      home.draws += 1;
      away.draws += 1;
      home.points += 1;
      away.points += 1;
    }
    playedMatches.push({
      homeTeam: match.homeTeam,
      awayTeam: match.awayTeam,
      homeScore,
      awayScore,
    });
  }

  for (const team of Object.keys(table)) {
    table[team].gd = table[team].gf - table[team].ga;
  }

  const randomTiebreakTeams = new Set<string>();
  const rng = createRng(seedFromGroupState(group, matches, scores));
  const ranking = rankGroup(group.teams, playedMatches, table, rng, randomTiebreakTeams);
  ranking.forEach((team, index) => {
    if (table[team]) {
      table[team].position = index + 1;
      table[team].randomTiebreak = randomTiebreakTeams.has(team);
    }
  });

  return { table, ranking, randomTiebreakTeams };
}

function bestThirdPlace(
  groupTables: Array<{ ranking: string[]; table: Record<string, GroupTableRow> }>
) {
  const entries: Array<{
    team: string;
    group: string;
    points: number;
    gd: number;
    gf: number;
  }> = [];
  for (const { ranking, table } of groupTables) {
    if (ranking.length < 3) {
      continue;
    }
    const team = ranking[2];
    const row = table[team];
    entries.push({
      team,
      group: row.group,
      points: row.points,
      gd: row.gd,
      gf: row.gf,
    });
  }
  const rng = createRng(seedFromThirdPlace(entries));
  const randomTiebreakTeams = new Set<string>();
  entries.sort((a, b) => {
    if (b.points !== a.points) {
      return b.points - a.points;
    }
    if (b.gd !== a.gd) {
      return b.gd - a.gd;
    }
    if (b.gf !== a.gf) {
      return b.gf - a.gf;
    }
    return 0;
  });
  const ordered: typeof entries = [];
  let i = 0;
  while (i < entries.length) {
    const current = entries[i];
    const tied = [current];
    i += 1;
    while (i < entries.length) {
      const next = entries[i];
      if (
        current.points === next.points &&
        current.gd === next.gd &&
        current.gf === next.gf
      ) {
        tied.push(next);
        i += 1;
      } else {
        break;
      }
    }
    if (tied.length > 1) {
      shuffleInPlace(tied, rng);
      tied.forEach((entry) => randomTiebreakTeams.add(entry.team));
    }
    ordered.push(...tied);
  }
  return { entries: ordered, randomTiebreakTeams };
}

function resolveGroupPlaceholder(
  label: string,
  groupRankings: Record<string, string[]>,
  thirdPlaceByGroup: Record<string, string>,
  groupCompletion: Record<string, boolean>,
  allowThirdPlaceResolve: boolean,
  qualifiedThirdGroups?: Set<string>
) {
  if (label.startsWith("Winner Group ")) {
    const group = label.replace("Winner Group ", "").trim();
    if (groupCompletion[group]) {
      return groupRankings[group]?.[0] ?? label;
    }
    return formatGroupPlaceholder(label);
  }
  if (label.startsWith("Runner-up Group ")) {
    const group = label.replace("Runner-up Group ", "").trim();
    if (groupCompletion[group]) {
      return groupRankings[group]?.[1] ?? label;
    }
    return formatGroupPlaceholder(label);
  }
  if (label.startsWith("3rd Group ")) {
    const group = label.replace("3rd Group ", "").trim();
    if (group.length === 1) {
      if (
        allowThirdPlaceResolve &&
        (!qualifiedThirdGroups || qualifiedThirdGroups.has(group))
      ) {
        return thirdPlaceByGroup[group] ?? label;
      }
      return formatGroupPlaceholder(label);
    }
  }
  return formatGroupPlaceholder(label);
}

function formatGroupPlaceholder(label: string) {
  if (label.startsWith("Winner Group ")) {
    return label.replace("Winner Group ", "1st Group ");
  }
  if (label.startsWith("Runner-up Group ")) {
    return label.replace("Runner-up Group ", "2nd Group ");
  }
  if (label.startsWith("3rd Group ")) {
    return label.replace("3rd Group ", "3rd Gr. ");
  }
  return label;
}

function formatStageShort(stage: string | undefined) {
  switch (stage) {
    case "Round of 32":
      return "R32";
    case "Round of 16":
      return "R16";
    case "Quarterfinal":
      return "QF";
    case "Semifinal":
      return "SF";
    case "Final":
      return "Final";
    default:
      return null;
  }
}

function resolveKnockoutLabel({
  label,
  opponentLabel,
  groupRankings,
  thirdPlaceByGroup,
  thirdPlaceAssignments,
  knockoutWinners,
  knockoutLosers,
  groupCompletion,
  allowThirdPlaceResolve,
  qualifiedThirdGroups,
  matchStageById,
}: {
  label: string;
  opponentLabel: string;
  groupRankings: Record<string, string[]>;
  thirdPlaceByGroup: Record<string, string>;
  thirdPlaceAssignments: Record<string, string> | null;
  knockoutWinners: Map<number, string>;
  knockoutLosers: Map<number, string>;
  groupCompletion: Record<string, boolean>;
  allowThirdPlaceResolve: boolean;
  qualifiedThirdGroups?: Set<string>;
  matchStageById: Record<number, string>;
}) {
  if (/^UEFA Path\s+.+\s+Winner$/i.test(label)) {
    return label.replace(/\s+Winner$/i, "");
  }
  if (/^IC Path\s+.+\s+Winner$/i.test(label)) {
    return label.replace(/\s+Winner$/i, "");
  }
  if (label.startsWith("Winner Match ")) {
    const matchId = Number(label.replace("Winner Match ", "").trim());
    const winner = knockoutWinners.get(matchId);
    if (winner) {
      return winner;
    }
    const stage = formatStageShort(matchStageById[matchId]);
    return stage ? `Winner ${stage}` : label;
  }
  if (label.startsWith("Winner UEFA Path ")) {
    return label.replace("Winner ", "");
  }
  if (label.startsWith("Winner IC Path ")) {
    return label.replace("Winner ", "");
  }
  if (label.startsWith("Loser Match ")) {
    const matchId = Number(label.replace("Loser Match ", "").trim());
    const loser = knockoutLosers.get(matchId);
    if (loser) {
      return loser;
    }
    const stage = formatStageShort(matchStageById[matchId]);
    return stage ? `Loser ${stage}` : label;
  }
  if (
    allowThirdPlaceResolve &&
    label.startsWith("3rd Group ") &&
    opponentLabel.startsWith("Winner Group ")
  ) {
    const winnerGroup = opponentLabel.replace("Winner Group ", "").trim();
    const key = `1${winnerGroup}`;
    const assignedGroup = thirdPlaceAssignments?.[key];
    if (assignedGroup) {
      return thirdPlaceByGroup[assignedGroup] ?? label;
    }
  }
  return resolveGroupPlaceholder(
    label,
    groupRankings,
    thirdPlaceByGroup,
    groupCompletion,
    allowThirdPlaceResolve,
    qualifiedThirdGroups
  );
}

function TeamFlag({
  team,
  flags,
  className,
}: {
  team: string;
  flags: Record<string, string | null>;
  className?: string;
}) {
  const isPlaceholder = isPlaceholderLabel(team);
  const flagPath = flags[team];
  if (flagPath) {
    return (
      <div
        className={cn(
          "relative h-5 w-7 shrink-0 overflow-hidden rounded-[1px] border border-ink-900",
          className,
          isPlaceholder ? "bg-[#d9d9d9]" : "bg-ink-800"
        )}
      >
        <img
          src={flagPath}
          alt={`${team} flag`}
          className="h-full w-full object-cover"
          loading="eager"
          decoding="async"
        />
      </div>
    );
  }
  return (
    <div
      className={cn(
        "flex h-5 w-7 shrink-0 items-center justify-center rounded-[1px] border border-ink-900 text-[9px] font-semibold uppercase",
        className,
        isPlaceholder ? "bg-[#d9d9d9] text-transparent" : "bg-ink-800 text-ink-200"
      )}
    >
      {team && !isPlaceholder ? teamInitials(team) : ""}
    </div>
  );
}

function TeamBox({
  team,
  flags,
  score,
  onScoreChange,
  reverse,
  disabled,
  placeholder,
  onSelect,
  highlight,
  showScore = true,
  winProb,
  className,
}: {
  team: string;
  flags: Record<string, string | null>;
  score?: number | null;
  onScoreChange?: (value: number | null) => void;
  reverse?: boolean;
  disabled?: boolean;
  placeholder?: boolean;
  onSelect?: () => void;
  highlight?: boolean;
  showScore?: boolean;
  winProb?: string;
  className?: string;
}) {
  const formatted = formatDisplayLabel(team);
  const displayName = formatted;
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-[3px] border border-ink-900 bg-white px-2 py-1 text-xs lg:text-sm",
        showScore ? "w-[240px]" : "w-[200px]",
        reverse && "flex-row-reverse text-right",
        disabled && "bg-white text-ink-400",
        highlight && "border-ink-900 bg-[#f2e2e2] text-ebony",
        className
      )}
      onClick={disabled ? undefined : onSelect}
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
      onKeyDown={
        onSelect
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect();
              }
            }
          : undefined
      }
    >
      <TeamFlag team={team} flags={flags} />
      <span
        className={cn(
          "min-w-0 flex-1 truncate whitespace-nowrap text-xs font-medium text-ebony lg:text-sm"
        )}
      >
        {displayName || "TBD"}
      </span>
      {(winProb || showScore) && (
        <div
          className={cn(
            "flex shrink-0 items-center gap-1",
            !showScore && "ml-auto"
          )}
        >
          {reverse ? (
            <>
              {showScore && (
                <input
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={31}
                  value={score ?? ""}
                  onChange={(event) =>
                    onScoreChange?.(parseScore(event.target.value))
                  }
                  onKeyDown={(event) => {
                    if ([".", ",", "e", "E", "+", "-"].includes(event.key)) {
                      event.preventDefault();
                    }
                  }}
                  disabled={disabled}
                  onClick={(event) => event.stopPropagation()}
                  className="w-8 rounded border border-ink-900 bg-white text-right text-xs font-mono text-ink-200 focus:outline-none lg:text-sm"
                />
              )}
              {winProb && (
                <span className="text-[10px] font-semibold text-ink-400 lg:text-xs font-mono">
                  {winProb}
                </span>
              )}
            </>
          ) : (
            <>
              {winProb && (
                <span className="text-[10px] font-semibold text-ink-400 lg:text-xs font-mono">
                  {winProb}
                </span>
              )}
              {showScore && (
                <input
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={31}
                  value={score ?? ""}
                  onChange={(event) =>
                    onScoreChange?.(parseScore(event.target.value))
                  }
                  onKeyDown={(event) => {
                    if ([".", ",", "e", "E", "+", "-"].includes(event.key)) {
                      event.preventDefault();
                    }
                  }}
                  disabled={disabled}
                  onClick={(event) => event.stopPropagation()}
                  className="w-8 rounded border border-ink-900 bg-white text-right text-xs font-mono text-ink-200 focus:outline-none lg:text-sm"
                />
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function MatchCard({
  id,
  homeTeam,
  awayTeam,
  scores,
  onScoreChange,
  onScoreChangePair,
  allowDraw,
  orientation,
  flags,
  disabled,
  locked = false,
  stackMode,
  fixedHeight,
  homeBoxRef,
  awayBoxRef,
  showScore = true,
  winnerSelection = null,
  onWinnerSelect,
  homeWinProb,
  awayWinProb,
  drawProb,
  showDivider,
  scoreMatrix,
  showHintRow,
  onHintDismiss,
}: {
  id: string | number;
  homeTeam: string;
  awayTeam: string;
  scores?: Record<string, MatchScore>;
  onScoreChange?: (
    id: string | number,
    side: "home" | "away",
    value: number | null
  ) => void;
  onScoreChangePair?: (
    id: string | number,
    home: number | null,
    away: number | null
  ) => void;
  allowDraw: boolean;
  orientation: "horizontal" | "vertical";
  flags: Record<string, string | null>;
  disabled?: boolean;
  locked?: boolean;
  stackMode?: "centered";
  fixedHeight?: number;
  homeBoxRef?: React.Ref<HTMLDivElement>;
  awayBoxRef?: React.Ref<HTMLDivElement>;
  showScore?: boolean;
  winnerSelection?: WinnerSelection;
  onWinnerSelect?: (selection: WinnerSelection) => void;
  homeWinProb?: string;
  awayWinProb?: string;
  drawProb?: string | null;
  showDivider?: boolean;
  scoreMatrix?: number[][] | null;
  showHintRow?: boolean;
  onHintDismiss?: () => void;
}) {
  const homeInputRef = React.useRef<HTMLInputElement>(null);
  const awayInputRef = React.useRef<HTMLInputElement>(null);
  const score = showScore
    ? scores?.[String(id)] ?? { home: null, away: null }
    : { home: null, away: null };
  const hasScore = showScore && score.home !== null && score.away !== null;
  const isDraw = hasScore && score.home === score.away;
  const selection = showScore ? null : winnerSelection ?? null;
  const winner = showScore
    ? hasScore &&
      score.home !== null &&
      score.away !== null &&
      score.home !== score.away
      ? score.home > score.away
        ? homeTeam
        : awayTeam
      : undefined
    : selection === "home"
      ? homeTeam
      : selection === "away"
        ? awayTeam
        : undefined;
  const highlightTeams = !showScore || !allowDraw || !isDraw;
  const placeholderHome = isPlaceholderLabel(homeTeam);
  const placeholderAway = isPlaceholderLabel(awayTeam);
  const homeProb = placeholderHome || placeholderAway ? undefined : homeWinProb;
  const awayProb = placeholderHome || placeholderAway ? undefined : awayWinProb;
  const drawLabel =
    showScore && allowDraw
      ? placeholderHome || placeholderAway
        ? ""
        : drawProb ?? "Draw"
      : null;
  const isDisabled = disabled || placeholderHome || placeholderAway;
  const isPickableMatch = !isDisabled;
  const setScores = (home: number | null, away: number | null) => {
    if (onScoreChangePair) {
      onScoreChangePair(id, home, away);
      return;
    }
    onScoreChange?.(id, "home", home);
    onScoreChange?.(id, "away", away);
  };
  const selectWinner = (side: "home" | "away") => {
    if (!onWinnerSelect) {
      return;
    }
    onWinnerSelect(selection === side ? null : side);
  };
  const sampleScoreForResult = (result: "home" | "away" | "draw") => {
    if (!scoreMatrix) {
      return null;
    }
    return sampleScoreMatrixByResult(scoreMatrix, result);
  };

  const [isDrawHovered, setIsDrawHovered] = React.useState(false);
  const [hintRowVisible, setHintRowVisible] = React.useState(false);
  const dismissHint = React.useCallback(() => {
    if (showHintRow && onHintDismiss) {
      onHintDismiss();
    }
  }, [onHintDismiss, showHintRow]);

  React.useEffect(() => {
    if (!showHintRow) {
      setHintRowVisible(false);
      return;
    }
    setHintRowVisible(false);
    const frame = requestAnimationFrame(() => setHintRowVisible(true));
    return () => cancelAnimationFrame(frame);
  }, [showHintRow]);

  React.useEffect(() => {
    if (showHintRow && hasScore) {
      dismissHint();
    }
  }, [dismissHint, hasScore, showHintRow]);

  if (orientation === "horizontal" && showScore) {
    const isMobile = useMediaQuery("(max-width: 768px)");
    const isScoreSet = score.home !== null && score.away !== null;
    const segments = normalizeProbabilitySegments({
      home: parseProbabilityLabel(homeProb),
      draw: allowDraw ? parseProbabilityLabel(drawProb ?? undefined) : null,
      away: parseProbabilityLabel(awayProb),
    });
    const isDecimalProbabilities = Boolean(
      segments &&
        [segments.home, segments.draw, segments.away].some(
          (value) => value !== Math.round(value)
        )
    );
    const probabilityLabelWidthClass = isDecimalProbabilities
      ? "min-w-[56px] sm:min-w-[74px]"
      : "w-12 sm:w-16";
    const formattedHome = formatDisplayLabel(homeTeam);
    const formattedAway = formatDisplayLabel(awayTeam);
    // Use FIFA codes on mobile for group stage matches, or "Qualifier" for qualifier placeholders
    const displayHome = isMobile && placeholderHome
      ? "Qualifier"
      : isMobile && !placeholderHome
        ? (getFifaCode(homeTeam) ?? formattedHome)
        : formattedHome;
    const displayAway = isMobile && placeholderAway
      ? "Qualifier"
      : isMobile && !placeholderAway
        ? (getFifaCode(awayTeam) ?? formattedAway)
        : formattedAway;

    const updateSideScore = (side: "home" | "away", value: number | null) => {
      if (side === "home") {
        setScores(value, score.away);
      } else {
        setScores(score.home, value);
      }
    };

    const adjustScore = (side: "home" | "away", delta: number) => {
      if (!isPickableMatch) {
        return;
      }
      const current = side === "home" ? score.home : score.away;
      const base = current ?? 0;
      const next = parseScore(String(base + delta));
      updateSideScore(side, next);
    };

    const handleScoreKeyDown = (
      event: React.KeyboardEvent<HTMLInputElement>,
      side: "home" | "away"
    ) => {
      // Block non-integer characters
      if ([".", ",", "e", "E", "+", "-"].includes(event.key)) {
        event.preventDefault();
        return;
      }
      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
        event.preventDefault();
        adjustScore(side, event.shiftKey ? 5 : 1);
      }
      if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
        event.preventDefault();
        adjustScore(side, event.shiftKey ? -5 : -1);
      }
    };

    const handleTeamSelect = (side: "home" | "away") => {
      if (!isPickableMatch) {
        return;
      }
      if (hasScore && winner === (side === "home" ? homeTeam : awayTeam)) {
        setScores(null, null);
        return;
      }
      if (side === "home") {
        const sampled = sampleScoreForResult("home");
        setScores(sampled?.home ?? 2, sampled?.away ?? 1);
        dismissHint();
        return;
      }
      const sampled = sampleScoreForResult("away");
      setScores(sampled?.home ?? 1, sampled?.away ?? 2);
      dismissHint();
    };

    const handleDrawSelect = () => {
      if (!isPickableMatch || !allowDraw) {
        return;
      }
      if (hasScore && isDraw) {
        setScores(null, null);
        return;
      }
      const sampled = sampleScoreForResult("draw");
      setScores(sampled?.home ?? 1, sampled?.away ?? 1);
      dismissHint();
    };

    const homeIsWinner = isScoreSet && !isDraw && score.home !== null && score.away !== null && score.home > score.away;
    const awayIsWinner = isScoreSet && !isDraw && score.home !== null && score.away !== null && score.away > score.home;

    const renderScoreInput = (
      side: "home" | "away",
      inputRef: React.RefObject<HTMLInputElement>
    ) => {
      const value = side === "home" ? score.home : score.away;
      const isWin = side === "home" ? homeIsWinner : awayIsWinner;
      return (
        <div className="group relative flex items-center justify-center">
          <input
            ref={inputRef}
            type="number"
            inputMode="numeric"
            min={0}
            max={31}
            value={value ?? ""}
            placeholder="-"
            onChange={(event) =>
              updateSideScore(side, parseScore(event.target.value))
            }
            onKeyDown={(event) => handleScoreKeyDown(event, side)}
            onWheel={(event) => {
              if (document.activeElement !== inputRef.current) {
                return;
              }
              event.preventDefault();
              adjustScore(side, event.deltaY > 0 ? -1 : 1);
            }}
            disabled={!isPickableMatch}
            className={cn(
              "w-7 h-6 sm:w-8 sm:h-7 rounded-md text-center text-xs sm:text-sm font-semibold tabular-nums focus:outline-none focus:ring-2 focus:ring-blue-400 appearance-none [-moz-appearance:textfield] [-webkit-appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none transition-colors",
              !isScoreSet && "bg-slate-100 text-slate-400 placeholder:text-slate-400",
              isScoreSet && isWin && (locked ? "bg-transparent text-slate-700" : "bg-transparent text-blue-700"),
              isScoreSet && isDraw && (locked ? "bg-transparent text-slate-700" : "bg-transparent text-blue-700"),
              isScoreSet && !isWin && !isDraw && "bg-transparent text-slate-400",
              !isPickableMatch && "cursor-default opacity-60"
            )}
          />
        </div>
      );
    };

    const renderTeamRow = (
      side: "home" | "away",
      team: string,
      displayName: string,
      inputRef: React.RefObject<HTMLInputElement>
    ) => {
      const isWin = side === "home" ? homeIsWinner : awayIsWinner;
      const isLoser = side === "home" ? awayIsWinner : homeIsWinner;
      return (
        <button
          type="button"
          onClick={() => handleTeamSelect(side)}
          disabled={!isPickableMatch}
          className={cn(
            "group flex items-center gap-1.5 sm:gap-3 px-1 sm:px-1.5 py-1.5 sm:py-2 transition-all duration-200 w-full relative",
            side === "home" ? "justify-end" : "justify-start",
            isPickableMatch && "cursor-pointer",
            !isPickableMatch && "cursor-default"
          )}
        >
          {/* Hover gradient overlay - only shows when hovering this specific button */}
          {isPickableMatch && (
            <div
              className={cn(
                "absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-0",
                side === "home"
                  ? "bg-[linear-gradient(to_right,rgba(219,234,254,0.5)_0%,rgba(219,234,254,0.5)_50%,transparent_100%)]"
                  : "bg-[linear-gradient(to_left,rgba(219,234,254,0.5)_0%,rgba(219,234,254,0.5)_50%,transparent_100%)]"
              )}
            />
          )}
          {side === "away" && (
            <TeamFlag
              team={team}
              flags={flags}
              className="h-3.5 w-5 sm:h-4 sm:w-6 flex-shrink-0 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] relative z-10"
            />
          )}
          <span
            className={cn(
              "min-w-0 truncate text-xs sm:text-sm leading-5 relative z-10",
              side === "home" && "text-right",
              !isPickableMatch && "text-slate-400",
              !isScoreSet && isPickableMatch && "font-medium text-slate-900",
              isScoreSet && isWin && "font-bold text-slate-900",
              isScoreSet && isDraw && "font-medium text-slate-700",
              isScoreSet && isLoser && "font-medium text-slate-500"
            )}
          >
            {displayName || "TBD"}
          </span>
          {side === "home" && (
            <TeamFlag
              team={team}
              flags={flags}
              className="h-3.5 w-5 sm:h-4 sm:w-6 flex-shrink-0 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] relative z-10"
            />
          )}
        </button>
      );
    };

    // Calculate gradient position for blue highlight (aligned with knockouts: blue-100 at 50% opacity)
    const getGradientStyle = () => {
      if (!isScoreSet) return {};
      
      if (isDraw) {
        // For draws: gradient spans symmetrically around the center score area
        // Making it symmetric and wider to account for varying team name lengths
        return {
          background: `linear-gradient(to right, 
            transparent 0%, 
            transparent 35%, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 40%, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 50%, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 60%, 
            transparent 65%, 
            transparent 100%)`
        };
      } else if (homeIsWinner) {
        // For home wins: gradient starts from left edge, fades to white around home score, extending to cover score area
        return {
          background: `linear-gradient(to right, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 0%, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 35%, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 42%, 
            rgba(255, 255, 255, 0) 48%, 
            transparent 52%, 
            transparent 100%)`
        };
      } else if (awayIsWinner) {
        // For away wins: gradient starts from right edge, fades to white around away score, extending to cover score area
        return {
          background: `linear-gradient(to right, 
            transparent 0%, 
            transparent 48%, 
            rgba(255, 255, 255, 0) 52%, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 58%, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 65%, 
            ${locked ? "rgb(226, 232, 240)" : "rgb(219, 234, 254)"} 100%)`
        };
      }
      return {};
    };

    const hintTextHome = `Click to predict ${formatDisplayLabel(homeTeam)}`;
    const hintTextAway = `Click to predict ${formatDisplayLabel(awayTeam)}`;
    const hintRow = showHintRow ? (
      <div
        className={cn(
          "pointer-events-none absolute left-0 right-0 top-full -mt-2 z-20 transition-opacity duration-200 ease-out",
          hintRowVisible ? "opacity-100" : "opacity-0"
        )}
      >
        <div className="grid grid-cols-[1fr_auto_1fr] items-start gap-2 px-1">
          <div className="flex flex-col items-center gap-0">
            <svg className="h-2 w-4 -mb-px" viewBox="0 0 20 8" fill="none" aria-hidden="true">
              <path d="M0 8 L10 0 L20 8" fill="rgb(15 23 42)" />
            </svg>
            <div className="flex items-center justify-center gap-1 rounded-md bg-slate-900 px-1.5 sm:px-2 py-0.5 sm:py-1 text-[10px] sm:text-[11px] font-semibold text-white shadow-sm text-center">
              <span>{hintTextHome}</span>
            </div>
          </div>
          <div className="flex flex-col items-center gap-0">
            <svg className="h-2 w-4 -mb-px" viewBox="0 0 20 8" fill="none" aria-hidden="true">
              <path d="M0 8 L10 0 L20 8" fill="rgb(15 23 42)" />
            </svg>
            <div className="flex items-center justify-center gap-1 rounded-md bg-slate-900 px-1.5 sm:px-2 py-0.5 sm:py-1 text-[10px] sm:text-[11px] font-semibold text-white shadow-sm text-center">
              <span>Click the probability bar to predict a draw</span>
            </div>
          </div>
          <div className="flex flex-col items-center gap-0">
            <svg className="h-2 w-4 -mb-px" viewBox="0 0 20 8" fill="none" aria-hidden="true">
              <path d="M0 8 L10 0 L20 8" fill="rgb(15 23 42)" />
            </svg>
            <div className="flex items-center justify-center gap-1 rounded-md bg-slate-900 px-1.5 sm:px-2 py-0.5 sm:py-1 text-[10px] sm:text-[11px] font-semibold text-white shadow-sm text-center">
              <span>{hintTextAway}</span>
            </div>
          </div>
        </div>
      </div>
    ) : null;

    return (
      <div className="relative">
        <div
          className={cn(
            "relative overflow-hidden rounded-xl shadow-sm transition-shadow hover:shadow",
            isPickableMatch && !isScoreSet && "bg-white ring-2 ring-[color:var(--cta-color)]",
            isScoreSet && "bg-white ring-1 ring-slate-400",
            !isScoreSet && !isPickableMatch && "bg-white ring-1 ring-slate-200",
            showHintRow && "hint-pulse"
          )}
        >
        {/* Blue gradient highlight background */}
        {isScoreSet && (isDraw || homeIsWinner || awayIsWinner) && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={getGradientStyle()}
          />
        )}
        {/* Hover gradient for draw button - positioned relative to match card, extends full height */}
        {allowDraw && isPickableMatch && (
          <div
            className={cn(
              "absolute inset-0 pointer-events-none transition-opacity duration-200",
              isDrawHovered ? "opacity-100" : "opacity-0"
            )}
            style={{
              background: `linear-gradient(to right, 
                transparent 0%, 
                transparent 38%, 
                rgba(219, 234, 254, 0.5) 44%, 
                rgba(219, 234, 254, 0.5) 56%, 
                transparent 58%, 
                transparent 100%)`
            }}
          />
        )}
        <div className="relative flex items-center">
          {/* Home team */}
          <div className="flex-1 min-w-0">
            {renderTeamRow("home", homeTeam, displayHome, homeInputRef)}
          </div>

          {/* Score area */}
          <div className="relative flex items-center gap-0.5 sm:gap-1 px-0.5 sm:px-1.5">
            {renderScoreInput("home", homeInputRef)}
            <button
              type="button"
              onClick={handleDrawSelect}
              disabled={!allowDraw || !isPickableMatch}
              onMouseEnter={() => setIsDrawHovered(true)}
              onMouseLeave={() => setIsDrawHovered(false)}
              className={cn(
                "flex flex-col items-center justify-center gap-0.5 sm:gap-1 px-1 sm:px-2 py-0.5 sm:py-1 rounded-md transition-colors",
                allowDraw && isPickableMatch && "cursor-pointer",
                !isPickableMatch && "cursor-default"
              )}
            >
              <div className="flex h-1 w-12 sm:w-16 overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-full bg-emerald-400"
                  style={{ width: `${segments?.home ?? 0}%` }}
                />
                <div
                  className="h-full bg-slate-400"
                  style={{ width: `${segments?.draw ?? 0}%` }}
                />
                <div
                  className="h-full bg-rose-400"
                  style={{ width: `${segments?.away ?? 0}%` }}
                />
              </div>
              <div
                className={cn(
                  "flex justify-between text-[10px] sm:text-xs leading-none tabular-nums text-slate-600",
                  probabilityLabelWidthClass,
                  isDecimalProbabilities && "gap-0.5 sm:gap-1"
                )}
              >
                <span className={cn(isDecimalProbabilities && "min-w-[18px] text-center")}>
                  {segments ? formatSegmentDisplay(segments.home) : "--"}
                </span>
                <span className={cn(isDecimalProbabilities && "min-w-[18px] text-center")}>
                  {segments ? formatSegmentDisplay(segments.draw) : "--"}
                </span>
                <span className={cn(isDecimalProbabilities && "min-w-[18px] text-center")}>
                  {segments ? formatSegmentDisplay(segments.away) : "--"}
                </span>
              </div>
            </button>
            {renderScoreInput("away", awayInputRef)}
          </div>

          {/* Away team */}
          <div className="flex-1 min-w-0">
            {renderTeamRow("away", awayTeam, displayAway, awayInputRef)}
          </div>
        </div>
        </div>
        {hintRow}
      </div>
    );
  }

  const drawButton =
    showScore && allowDraw ? (
    <button
      type="button"
      onClick={() => {
        if (score.home === 1 && score.away === 1) {
          setScores(null, null);
          return;
        }
        const sampled = sampleScoreForResult("draw");
        setScores(sampled?.home ?? 1, sampled?.away ?? 1);
      }}
      disabled={isDisabled}
      className={cn(
        "border border-ink-900 w-[40px] bg-white px-2 py-1 text-[10px] font-semibold uppercase lg:text-xs font-mono",
        orientation === "horizontal" ? "rounded-none -mx-px" : "rounded-[3px]",
        isDraw ? "bg-[#f2e2e2] text-ebony" : "text-ink-400",
        isDisabled && "disabled:cursor-not-allowed"
      )}
    >
      {drawLabel}
    </button>
    ) : null;
  const highlightHome =
    !isDisabled && (highlightTeams ? winner === homeTeam : false);
  const highlightAway =
    !isDisabled && (highlightTeams ? winner === awayTeam : false);

  if (orientation === "vertical") {
    if (stackMode === "centered" && !allowDraw) {
      return (
        <div
          className="relative overflow-visible"
          style={fixedHeight ? { height: fixedHeight } : undefined}
        >
          <div
            className={cn(
              "grid",
              fixedHeight ? "h-full grid-rows-2" : "grid-rows-[auto_auto]"
            )}
          >
            <div className="flex items-end overflow-visible">
              <div ref={homeBoxRef}>
                <TeamBox
                  team={homeTeam}
                  flags={flags}
                  score={score.home}
                  onScoreChange={(value) => onScoreChange?.(id, "home", value)}
                  onSelect={() => {
                    if (showScore) {
                      if (score.home === 2 && score.away === 1) {
                        setScores(null, null);
                        return;
                      }
                      const sampled = sampleScoreForResult("home");
                      setScores(sampled?.home ?? 2, sampled?.away ?? 1);
                    } else {
                      selectWinner("home");
                    }
                  }}
                  highlight={highlightHome}
                  disabled={isDisabled}
                  placeholder={placeholderHome}
                  showScore={showScore}
                  winProb={homeProb}
                  className="rounded-b-none"
                />
              </div>
            </div>
            <div className="flex items-start overflow-visible">
              <div ref={awayBoxRef}>
                <TeamBox
                  team={awayTeam}
                  flags={flags}
                  score={score.away}
                  onScoreChange={(value) => onScoreChange?.(id, "away", value)}
                  onSelect={() => {
                    if (showScore) {
                      if (score.home === 1 && score.away === 2) {
                        setScores(null, null);
                        return;
                      }
                      const sampled = sampleScoreForResult("away");
                      setScores(sampled?.home ?? 1, sampled?.away ?? 2);
                    } else {
                      selectWinner("away");
                    }
                  }}
                  highlight={highlightAway}
                  disabled={isDisabled}
                  placeholder={placeholderAway}
                  showScore={showScore}
                  winProb={awayProb}
                  className="rounded-t-none border-t-0"
                />
              </div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col gap-2">
        <TeamBox
          team={homeTeam}
          flags={flags}
          score={score.home}
          onScoreChange={(value) => onScoreChange?.(id, "home", value)}
          onSelect={() => {
            if (showScore) {
              if (score.home === 2 && score.away === 1) {
                setScores(null, null);
                return;
              }
              const sampled = sampleScoreForResult("home");
              setScores(sampled?.home ?? 2, sampled?.away ?? 1);
            } else {
              selectWinner("home");
            }
          }}
          highlight={highlightHome}
          disabled={isDisabled}
          placeholder={placeholderHome}
          showScore={showScore}
          winProb={homeProb}
        />
        {drawButton}
        <TeamBox
          team={awayTeam}
          flags={flags}
          score={score.away}
          onScoreChange={(value) => onScoreChange?.(id, "away", value)}
          onSelect={() => {
            if (showScore) {
              if (score.home === 1 && score.away === 2) {
                setScores(null, null);
                return;
              }
              const sampled = sampleScoreForResult("away");
              setScores(sampled?.home ?? 1, sampled?.away ?? 2);
            } else {
              selectWinner("away");
            }
          }}
          highlight={highlightAway}
          disabled={isDisabled}
          placeholder={placeholderAway}
          showScore={showScore}
          winProb={awayProb}
        />
      </div>
    );
  }

  const horizontalHomeClass = "rounded-r-none";
  const horizontalAwayClass = "rounded-l-none";

  return (
    <div className="flex items-stretch gap-0">
      <div ref={homeBoxRef}>
        <TeamBox
          team={homeTeam}
          flags={flags}
          score={score.home}
          onScoreChange={(value) => onScoreChange?.(id, "home", value)}
          onSelect={() => {
            if (showScore) {
              if (score.home === 2 && score.away === 1) {
                setScores(null, null);
                return;
              }
              setScores(2, 1);
            } else {
              selectWinner("home");
            }
          }}
          highlight={highlightHome}
          disabled={isDisabled}
          placeholder={placeholderHome}
          showScore={showScore}
          winProb={homeProb}
          className={horizontalHomeClass}
        />
      </div>
      {drawButton}
      <div ref={awayBoxRef}>
        <TeamBox
          team={awayTeam}
          flags={flags}
          score={score.away}
          onScoreChange={(value) => onScoreChange?.(id, "away", value)}
          reverse
          onSelect={() => {
            if (showScore) {
              if (score.home === 1 && score.away === 2) {
                setScores(null, null);
                return;
              }
              setScores(1, 2);
            } else {
              selectWinner("away");
            }
          }}
          highlight={highlightAway}
          disabled={isDisabled}
          placeholder={placeholderAway}
          showScore={showScore}
          winProb={awayProb}
          className={horizontalAwayClass}
        />
      </div>
    </div>
  );
}

function KnockoutMatchCard({
  homeTeam,
  awayTeam,
  winnerSelection,
  onWinnerSelect,
  flags,
  homeWinProb,
  awayWinProb,
  drawProb,
  compact,
  isFinal,
  centerPlaceholders,
  cardWidthClass = "w-[152px] sm:w-[192px]",
  containerRef,
  homeRowRef,
  awayRowRef,
  mirrored,
  compactMode,
  locked = false,
  className,
}: {
  homeTeam: string;
  awayTeam: string;
  winnerSelection: WinnerSelection;
  onWinnerSelect: (selection: WinnerSelection) => void;
  flags: Record<string, string | null>;
  homeWinProb?: string;
  awayWinProb?: string;
  drawProb?: string | null;
  compact?: boolean;
  isFinal?: boolean;
  centerPlaceholders?: boolean;
  cardWidthClass?: string;
  containerRef?: React.Ref<HTMLDivElement>;
  homeRowRef?: React.Ref<HTMLButtonElement>;
  awayRowRef?: React.Ref<HTMLButtonElement>;
  mirrored?: boolean;
  compactMode?: boolean;
  locked?: boolean;
  className?: string;
}) {
  const placeholderHome = !isConcreteTeam(homeTeam);
  const placeholderAway = !isConcreteTeam(awayTeam);
  const isPickableMatch = !locked && !placeholderHome && !placeholderAway;
  const isPendingMatch =
    (placeholderHome && !placeholderAway) ||
    (!placeholderHome && placeholderAway);
  const isLockedMatch = placeholderHome && placeholderAway;
  const hideProbabilities = isPendingMatch || isLockedMatch;
  const winner = isPickableMatch
    ? winnerSelection === "home"
      ? homeTeam
      : winnerSelection === "away"
        ? awayTeam
        : null
    : null;
  const hasSelection = isPickableMatch && winnerSelection !== null;
  const needsPick = isPickableMatch && winner === null;
  const isFinalResolved = Boolean(isFinal && winner);
  const showDraw = Boolean(drawProb);
  const homeValue = parseProbabilityLabel(homeWinProb);
  const awayValue = parseProbabilityLabel(awayWinProb);
  const drawValue = showDraw ? parseProbabilityLabel(drawProb ?? undefined) : null;
  const segments = showDraw
    ? normalizeProbabilitySegments({
        home: homeValue,
        draw: drawValue,
        away: awayValue,
      })
    : normalizeTwoSegments({ home: homeValue, away: awayValue });
  const drawSegment = showDraw ? (segments as { draw?: number }).draw ?? 0 : 0;
  const paddedRow = compact ? "py-1 sm:py-1.5" : "py-0.5";
  const scoreSlot = <div className="flex w-0 flex-none" />;

  const renderTeamLabel = (
    team: string,
    isPlaceholder: boolean,
    isWinner: boolean,
    isLoser: boolean,
    isResolved: boolean
  ) => {
    // Center placeholders for Final/Third Place matches, otherwise use mirrored alignment
    const textAlign = isPlaceholder && centerPlaceholders ? "text-center" : mirrored ? "text-right" : "text-left";
    if (isPlaceholder) {
      return (
      <span className={cn("inline-flex h-[18px] sm:h-[20px] max-w-full items-center truncate rounded-md bg-slate-50 px-1.5 sm:px-2 text-[11px] sm:text-[12px] leading-[18px] sm:leading-[20px] text-slate-500 ring-1 ring-slate-200", textAlign, centerPlaceholders && "justify-center")}>
        {formatDisplayLabel(team)}
      </span>
    );
  }
  return (
      <span
        className={cn(
          "block min-w-0 truncate text-xs sm:text-sm leading-[20px]",
          textAlign,
          !isResolved && "font-medium text-slate-900",
          isResolved && isWinner && "font-bold text-slate-900",
          isResolved && isLoser && "font-medium text-slate-700"
        )}
      >
        {formatDisplayLabel(team)}
      </span>
    );
  };

  const renderRow = (
    team: string,
    side: "home" | "away",
    isPlaceholder: boolean,
    rowRef?: React.Ref<HTMLButtonElement>
  ) => {
    const isWinner = winner === team;
    const isResolved = winner !== null;
    const isLoser = isResolved && !isWinner;
    const isChampionRow = isFinalResolved && isWinner;
    
    // Gradient directions for winner highlight
    const normalGradient = locked
      ? "bg-[linear-gradient(90deg,transparent_0%,rgb(226,232,240)_10%,rgb(226,232,240)_100%)]"
      : "bg-[linear-gradient(90deg,transparent_0%,rgb(219,234,254)_10%,rgb(219,234,254)_100%)]";
    const mirroredGradient = locked
      ? "bg-[linear-gradient(270deg,transparent_0%,rgb(226,232,240)_10%,rgb(226,232,240)_100%)]"
      : "bg-[linear-gradient(270deg,transparent_0%,rgb(219,234,254)_10%,rgb(219,234,254)_100%)]";
    const normalChampionGradient = "bg-[linear-gradient(90deg,rgba(254,243,199,0)_0%,rgba(254,243,199,0.6)_10%,rgba(254,243,199,0.6)_100%)]";
    const mirroredChampionGradient = "bg-[linear-gradient(270deg,rgba(254,243,199,0)_0%,rgba(254,243,199,0.6)_10%,rgba(254,243,199,0.6)_100%)]";
    
    return (
      <button
        ref={rowRef}
        type="button"
        onClick={() => {
          if (!isPickableMatch) {
            return;
          }
          onWinnerSelect(winnerSelection === side ? null : side);
        }}
        disabled={!isPickableMatch}
        className={cn(
          "group flex w-full flex-1 items-center relative z-10",
          centerPlaceholders && isLockedMatch ? "px-1.5 sm:px-2 justify-center gap-0" : "gap-1.5 sm:gap-2",
          centerPlaceholders && isLockedMatch ? "" : mirrored ? "pl-0 pr-0" : "pl-0 pr-0",
          paddedRow,
          isResolved &&
            isWinner &&
            !isChampionRow &&
            (mirrored ? mirroredGradient : normalGradient),
          isChampionRow &&
            (mirrored ? mirroredChampionGradient : normalChampionGradient)
        )}
        style={{
          cursor: isPickableMatch ? 'pointer' : 'default',
          pointerEvents: 'auto'
        }}
      >
        {/* Hover gradient overlay */}
        {isPickableMatch && !isChampionRow && (
          <div
            className={cn(
              "absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200",
              mirrored 
                ? "bg-[linear-gradient(270deg,transparent_0%,rgba(219,234,254,0.5)_10%,rgba(219,234,254,0.5)_100%)]"
                : "bg-[linear-gradient(90deg,transparent_0%,rgba(219,234,254,0.5)_10%,rgba(219,234,254,0.5)_100%)]"
            )}
          />
        )}
        {mirrored ? (
          <>
            {scoreSlot}
            <div className="relative flex min-w-0 flex-1 items-center">
              <div
                className={cn(
                  "flex min-w-0 flex-1 items-center justify-end gap-2",
                  isPlaceholder ? "pr-0" : "pr-2"
                )}
              >
                <div className="flex min-w-0 flex-1 items-center justify-end">
                  {renderTeamLabel(team, isPlaceholder, isWinner, isLoser, isResolved)}
                </div>
              </div>
              <span
                className={cn(
                  "absolute right-0 top-0 h-full w-1 rounded-full",
                  isResolved && isWinner
                    ? isChampionRow
                      ? "bg-amber-300"
                      : locked
                      ? "bg-slate-400"
                      : "bg-blue-200"
                    : "bg-transparent"
                )}
                aria-hidden="true"
              />
            </div>
            {!isPlaceholder && (
              <TeamFlag
                team={team}
                flags={flags}
                className="h-3.5 w-5 sm:h-4 sm:w-6 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
              />
            )}
          </>
        ) : (
          <>
            {!isPlaceholder && (
              <TeamFlag
                team={team}
                flags={flags}
                className="h-3.5 w-5 sm:h-4 sm:w-6 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
              />
            )}
            <div className={cn("relative flex min-w-0 items-center", isPlaceholder && centerPlaceholders ? "flex-1 justify-center" : "flex-1")}>
              <span
                className={cn(
                  "absolute left-0 top-0 h-full w-1 rounded-full",
                  isResolved && isWinner
                    ? isChampionRow
                      ? "bg-amber-300"
                      : locked
                      ? "bg-slate-400"
                      : "bg-blue-200"
                    : "bg-transparent"
                )}
                aria-hidden="true"
              />
              <div
                className={cn(
                  "flex items-center",
                  isPlaceholder && centerPlaceholders ? "justify-center w-full" : "min-w-0 flex-1 gap-2",
                  isPlaceholder ? "pl-0" : "pl-2"
                )}
              >
                {renderTeamLabel(team, isPlaceholder, isWinner, isLoser, isResolved)}
              </div>
            </div>
            {!(isPlaceholder && centerPlaceholders) && scoreSlot}
          </>
        )}
      </button>
    );
  };

  const probabilityBar = (
    <div
      className={cn(
        "flex h-[56px] sm:h-[72px] w-5 sm:w-7 flex-col items-center justify-center px-2 sm:px-4 py-0.5 sm:py-1 pointer-events-none",
        hideProbabilities && "invisible",
        hasSelection && "opacity-55"
      )}
      aria-hidden={hideProbabilities}
    >
      <span className="text-[10px] sm:text-xs tabular-nums text-slate-600">
        {segments ? formatSegmentDisplay(segments.home) : "--"}
      </span>
      <div className="h-4 sm:h-6 w-1.5 sm:w-2 overflow-hidden rounded-full bg-slate-200/70">
        <div className="flex h-full flex-col">
          <div
            className="w-full bg-emerald-300/70"
            style={{ height: `${segments?.home ?? 0}%` }}
          />
          {showDraw && (
            <div
              className="w-full bg-slate-300/70"
              style={{ height: `${drawSegment}%` }}
            />
          )}
          <div
            className="w-full bg-rose-300/70"
            style={{ height: `${segments?.away ?? 0}%` }}
          />
        </div>
      </div>
      <span className="text-[10px] sm:text-xs tabular-nums text-slate-600">
        {segments ? formatSegmentDisplay(segments.away) : "--"}
      </span>
    </div>
  );

  const teamRows = (
    <div className="flex min-w-0 flex-1 flex-col relative z-10">
      {renderRow(homeTeam, "home", placeholderHome, homeRowRef)}
      {renderRow(awayTeam, "away", placeholderAway, awayRowRef)}
    </div>
  );

  // Compact mode: just flags with blue background for winner
  if (compactMode) {
    const renderCompactRow = (team: string, side: "home" | "away", isPlaceholder: boolean) => {
      const isWinner = winner === team;
      const isResolved = winner !== null;
      const isChampionRow = isFinalResolved && isWinner;
      
      return (
        <button
          type="button"
          onClick={() => {
            if (!isPickableMatch) return;
            onWinnerSelect(winnerSelection === side ? null : side);
          }}
          disabled={!isPickableMatch}
          className={cn(
            "relative z-10 flex w-full flex-1 items-center justify-center p-1",
            isResolved && isWinner && !isChampionRow && (locked ? "bg-slate-300" : "bg-blue-200"),
            isChampionRow && "bg-amber-200",
            isPickableMatch ? "cursor-pointer" : "cursor-default"
          )}
          style={{ pointerEvents: "auto" }}
        >
          {isPlaceholder ? (
            <div className="h-4 w-6 rounded-sm bg-slate-100 ring-1 ring-slate-200" />
          ) : (
            <TeamFlag
              team={team}
              flags={flags}
              className="h-4 w-6 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
            />
          )}
        </button>
      );
    };

    return (
      <div
        ref={containerRef}
        className={cn(
          "w-[40px] overflow-hidden rounded-lg shadow-sm",
          needsPick
            ? "bg-white ring-2 ring-[color:var(--cta-color)]"
            : hasSelection
              ? "bg-white ring-1 ring-slate-400"
              : "bg-white ring-1 ring-slate-200",
          className
        )}
      >
        <div className="flex flex-col">
          {renderCompactRow(homeTeam, "home", placeholderHome)}
          {renderCompactRow(awayTeam, "away", placeholderAway)}
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        cardWidthClass,
        "overflow-hidden rounded-xl shadow-sm transition-shadow hover:shadow",
        needsPick
          ? "bg-white ring-2 ring-[color:var(--cta-color)]"
          : hasSelection
            ? "bg-white ring-1 ring-slate-400"
            : "bg-white ring-1 ring-slate-200",
        className
      )}
    >
      <div className="flex h-[56px] sm:h-[72px]">
        {centerPlaceholders && isLockedMatch ? (
          // For Final/Third Place with both placeholders, hide probability bar for proper centering
          teamRows
        ) : mirrored ? (
          <>
            {teamRows}
            {probabilityBar}
          </>
        ) : (
          <>
            {probabilityBar}
            {teamRows}
          </>
        )}
      </div>
    </div>
  );
}

function QualifierPathBracket({
  path,
  matches,
  winnerSelections,
  onWinnerSelect,
  onAutoPredict,
  onReset,
  autoPredictLoading,
  flags,
  getMatchProbabilityLabels,
  showTitle = true,
  embedded = false,
  showHint = false,
}: {
  path: string;
  matches: ResolvedQualifierMatch[];
  winnerSelections: Record<string, WinnerSelection>;
  onWinnerSelect: (id: string | number, selection: WinnerSelection) => void;
  onAutoPredict: (path: string) => void;
  onReset: (path: string) => void;
  autoPredictLoading: boolean;
  flags: Record<string, string | null>;
  getMatchProbabilityLabels: (params: {
    homeTeam: string;
    awayTeam: string;
    allowDraw: boolean;
    country?: string | null;
    neutralOverride?: boolean | null;
  }) => MatchProbabilityLabels;
  onGroupHintDismiss?: () => void;
  showTitle?: boolean;
  embedded?: boolean;
  showHint?: boolean;
}) {
  const semis = matches.filter((match) => match.round.startsWith("semi"));
  const final = matches.find((match) => match.round === "final");
  const finalHome = final?.homeResolved ?? final?.homeTeam ?? "";
  const finalAway = final?.awayResolved ?? final?.awayTeam ?? "";
  const finalId = final?.id ?? null;
  const finalProbabilities = final
    ? getMatchProbabilityLabels({
        homeTeam: finalHome,
        awayTeam: finalAway,
        allowDraw: false,
        neutralOverride: final.neutral,
      })
    : null;
  const isFinalPickable =
    Boolean(final) && isConcreteTeam(finalHome) && isConcreteTeam(finalAway);
  const finalWinnerSelection = final ? winnerSelections[String(final.id)] ?? null : null;
  const qualifiedTeam = isFinalPickable
    ? finalWinnerSelection === "home"
      ? finalHome
      : finalWinnerSelection === "away"
        ? finalAway
        : null
    : null;
  const topSemi = semis.length > 1 ? semis[0] : null;
  const bottomSemi = semis.length > 1 ? semis[1] : semis[0] ?? null;
  const hintMatchId = showHint ? (topSemi ?? bottomSemi)?.id ?? null : null;
  const hintPulseClass =
    "ring-2 ring-[color:var(--cta-color)] shadow-[0_0_0_6px_rgb(var(--cta-color-rgb)/0.35)] hint-pulse";
  const hasUnpredictedPathMatches = matches.some((match) => {
    const key = String(match.id);
    const selection = winnerSelections[key] ?? null;
    if (selection !== null) {
      return false;
    }
    const home = match.homeResolved ?? match.homeTeam;
    const away = match.awayResolved ?? match.awayTeam;
    return isConcreteTeam(home) && isConcreteTeam(away);
  });
  const hasPredictedPathMatches = matches.some((match) => {
    const key = String(match.id);
    return (winnerSelections[key] ?? null) !== null;
  });
  const semisKey = React.useMemo(
    () => semis.map((match) => String(match.id)).join("|"),
    [semis]
  );
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const bracketRef = React.useRef<HTMLDivElement | null>(null);
  const matchRefs = React.useRef(new Map<string | number, HTMLDivElement>());
  const matchHomeRefs = React.useRef(new Map<string | number, HTMLButtonElement>());
  const matchAwayRefs = React.useRef(new Map<string | number, HTMLButtonElement>());
  const [paths, setPaths] = React.useState<string[]>([]);
  const [semisOffset, setSemisOffset] = React.useState(0);
  const hintTextRef = React.useRef<HTMLDivElement | null>(null);
  const [hintPosition, setHintPosition] = React.useState<{
    textX: number;
    textY: number;
    targetX: number;
    targetY: number;
  } | null>(null);
  const [hintBox, setHintBox] = React.useState<{ width: number; height: number } | null>(
    null
  );
  const [hintVisible, setHintVisible] = React.useState(false);

  React.useLayoutEffect(() => {
    const container = containerRef.current;
    const bracket = bracketRef.current;
    if (!container || !bracket || !final) {
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const rect = bracket.getBoundingClientRect();
        const finalEl = matchRefs.current.get(final.id);
        if (!finalEl) {
          return;
        }
        const finalRect = finalEl.getBoundingClientRect();
        const nextPaths: string[] = [];
        const connectorInset = 12;
        const connectorStrokeWidth = 2;
        const isIcPath = path.startsWith("IC Path");
        const endX = finalRect.left - rect.left + connectorInset;
        const finalHomeBox = matchHomeRefs.current.get(final.id);
        const finalAwayBox = matchAwayRefs.current.get(final.id);
        let endY = finalRect.top - rect.top + finalRect.height / 2;
        if (isIcPath && finalAwayBox) {
          // IC Path: connect to away slot (home is already determined)
          const awayRect = finalAwayBox.getBoundingClientRect();
          endY = awayRect.top - rect.top + awayRect.height / 2;
        } else if (!isIcPath && finalHomeBox && finalAwayBox) {
          const homeRect = finalHomeBox.getBoundingClientRect();
          const awayRect = finalAwayBox.getBoundingClientRect();
          endY = (homeRect.bottom + awayRect.top) / 2 - rect.top;
        }
        if (isIcPath && semisOffset !== 0) {
          setSemisOffset(0);
        }
        semis.forEach((match) => {
          const semiEl = matchRefs.current.get(match.id);
          if (!semiEl) {
            return;
          }
          const semiRect = semiEl.getBoundingClientRect();
          const startX = Math.round(semiRect.right - rect.left - connectorInset);
          let startY = semiRect.top - rect.top + semiRect.height / 2;
          const semiHomeBox = matchHomeRefs.current.get(match.id);
          const semiAwayBox = matchAwayRefs.current.get(match.id);
          if (semiHomeBox && semiAwayBox) {
            const homeRect = semiHomeBox.getBoundingClientRect();
            const awayRect = semiAwayBox.getBoundingClientRect();
            startY = (homeRect.bottom + awayRect.top) / 2 - rect.top;
          }
          const drawStartY = startY;
          const midX = Math.round(startX + (endX - startX) * 0.5);
          const roundedEndY = Math.round(endY);
          const roundedStartY = Math.round(drawStartY);
          nextPaths.push(
            `M ${startX} ${roundedStartY} L ${midX} ${roundedStartY} L ${midX} ${roundedEndY} L ${endX} ${roundedEndY}`
          );
        });
        setPaths(nextPaths);
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(container);
    observer.observe(bracket);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [finalId, semisKey, path, semisOffset]);

  React.useLayoutEffect(() => {
    if (!showHint) {
      setHintPosition(null);
      return;
    }
    const bracket = bracketRef.current;
    const targetMatch = topSemi ?? bottomSemi;
    if (!bracket || !targetMatch) {
      return;
    }
    let frame = 0;
    const update = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const bracketRect = bracket.getBoundingClientRect();
        const targetEl = matchRefs.current.get(targetMatch.id);
        if (!targetEl) {
          setHintPosition(null);
          return;
        }
        const targetRect = targetEl.getBoundingClientRect();
        const targetX = targetRect.left - bracketRect.left + targetRect.width / 2;
        const targetY = targetRect.top - bracketRect.top + 6;
        const hintBoxWidth = 176;
        const hintBoxHeight = hintBox?.height ?? 44;
        const arrowLength = 20;
        const arrowGap = 8;
        const maxTextX = Math.max(0, bracketRect.width - hintBoxWidth);
        const textX = Math.min(
          maxTextX,
          Math.max(0, targetX - hintBoxWidth / 2)
        );
        const textY = Math.max(8, targetY - arrowLength - hintBoxHeight - arrowGap);
        setHintPosition({ textX, textY, targetX, targetY });
      });
    };
    const observer = new ResizeObserver(update);
    observer.observe(bracket);
    update();
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("resize", update);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [showHint, topSemi, bottomSemi, hintBox?.height]);

  React.useLayoutEffect(() => {
    if (!showHint || !hintPosition) {
      return;
    }
    const el = hintTextRef.current;
    if (!el) {
      return;
    }
    const rect = el.getBoundingClientRect();
    setHintBox({ width: rect.width, height: rect.height });
  }, [hintPosition, showHint]);

  React.useEffect(() => {
    if (!showHint || !hintPosition) {
      return;
    }
    setHintVisible(false);
    const frame = requestAnimationFrame(() => setHintVisible(true));
    return () => cancelAnimationFrame(frame);
  }, [hintPosition, showHint]);

  // Auto-scroll to show final when a semi is selected
  React.useEffect(() => {
    const bracket = bracketRef.current;
    const finalEl = final ? matchRefs.current.get(final.id) : null;
    if (!bracket || !final || !finalEl) {
      return;
    }
    
    // Check if semis are selected
    // For UEFA (2 semis): both must be selected
    // For IC (1 semi): that one must be selected
    const allSemisSelected = semis.length > 0 && semis.every(
      (semi) => winnerSelections[String(semi.id)] !== null && winnerSelections[String(semi.id)] !== undefined
    );
    
    if (allSemisSelected) {
      // Check if final is visible in viewport
      const checkAndScroll = () => {
        const bracketRect = bracket.getBoundingClientRect();
        const finalRect = finalEl.getBoundingClientRect();
        
        // Check if final is outside the visible area (to the right)
        const isFinalVisible = 
          finalRect.left >= bracketRect.left && 
          finalRect.right <= bracketRect.right;
        
        if (!isFinalVisible) {
          // Scroll to show the final
          bracket.scrollTo({
            left: bracket.scrollWidth - bracket.clientWidth,
            behavior: 'smooth',
          });
        }
      };
      
      // Use requestAnimationFrame to ensure layout is complete
      const frame = requestAnimationFrame(() => {
        requestAnimationFrame(checkAndScroll);
      });
      
      return () => cancelAnimationFrame(frame);
    }
  }, [semis, final, winnerSelections]);

  const content = (
    <div className="flex w-full flex-col gap-2 sm:gap-4">
      <div
        className={cn(
          "flex items-center gap-2 sm:gap-3 min-h-[36px] sm:min-h-[44px]",
          showTitle ? "justify-between" : "justify-start flex-wrap mb-2 sm:mb-4"
        )}
      >
        {showTitle && (
          <h3 className="text-xs sm:text-sm font-semibold text-slate-900">{path}</h3>
        )}
        <div className="flex items-center gap-1.5 sm:gap-2 text-xs">
          <LoadingButton
            loading={autoPredictLoading}
            disabled={!hasUnpredictedPathMatches}
            onClick={() => onAutoPredict(path)}
            className={cn(
              "rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
              hasUnpredictedPathMatches
                ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                : "text-slate-500"
            )}
          >
            Auto-predict
          </LoadingButton>
          <button
            type="button"
            disabled={!hasPredictedPathMatches}
            onClick={() => onReset(path)}
            className={cn(
              "rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
              hasPredictedPathMatches
                ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                : "text-slate-500 cursor-default"
            )}
          >
            Reset path
          </button>
        </div>
      </div>
      <div ref={bracketRef} className="relative w-full overflow-x-auto pl-4 pr-8 py-2 sm:pr-6">
        {showHint && hintPosition && (
          <div
            className={cn(
              "pointer-events-none absolute inset-0 z-20 transition-opacity duration-200 ease-out",
              hintVisible ? "opacity-100" : "opacity-0"
            )}
          >
            <div
              ref={hintTextRef}
              className="absolute flex w-44 items-center justify-center gap-1 rounded-md bg-slate-900 px-1.5 sm:px-2 py-0.5 sm:py-1 text-[10px] sm:text-[11px] font-semibold text-white shadow-sm text-center"
              style={{ left: hintPosition.textX, top: hintPosition.textY }}
            >
              <span>Click to predict a winner</span>
            </div>
            <svg
              className="absolute h-2 w-4"
              style={{
                left: `${hintPosition.textX + 88}px`,
                top: `${hintPosition.textY + (hintBox?.height ?? 44) - 1}px`,
                transform: "translateX(-50%)",
              }}
              viewBox="0 0 20 8"
              fill="none"
              aria-hidden="true"
            >
              <path d="M0 0 L10 8 L20 0" fill="rgb(15 23 42)" />
            </svg>
          </div>
        )}
        <svg
          className="absolute inset-0 z-0 h-full w-full pointer-events-none"
          aria-hidden="true"
        >
          {paths.map((pathDef, index) => (
            <path
              key={`${path}-${index}`}
              d={pathDef}
              fill="none"
              stroke="rgb(203 213 225)"
              strokeWidth={1.5}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
        </svg>
        <div className="relative z-10">
          <div className="grid min-w-max grid-cols-[max-content_max-content] gap-3 sm:gap-4 pr-2 sm:pr-1">
          <div
            className="grid w-fit grid-rows-[56px_56px] sm:grid-rows-[72px_72px] gap-3 sm:gap-4"
            style={semisOffset ? { marginTop: semisOffset } : undefined}
          >
            {topSemi ? (
              (() => {
                const probabilities = getMatchProbabilityLabels({
                  homeTeam: topSemi.homeResolved ?? topSemi.homeTeam,
                  awayTeam: topSemi.awayResolved ?? topSemi.awayTeam,
                  allowDraw: false,
                  neutralOverride: topSemi.neutral,
                });
                return (
                  <KnockoutMatchCard
                    homeTeam={topSemi.homeResolved ?? topSemi.homeTeam}
                    awayTeam={topSemi.awayResolved ?? topSemi.awayTeam}
                    winnerSelection={winnerSelections[String(topSemi.id)] ?? null}
                    onWinnerSelect={(selection) => onWinnerSelect(topSemi.id, selection)}
                    flags={flags}
                    homeWinProb={probabilities.homeWinProb}
                    awayWinProb={probabilities.awayWinProb}
                    drawProb={probabilities.drawProb}
                    compact
                    className={topSemi.id === hintMatchId ? hintPulseClass : undefined}
                    containerRef={(el) => {
                      if (el) {
                        matchRefs.current.set(topSemi.id, el);
                      } else {
                        matchRefs.current.delete(topSemi.id);
                      }
                    }}
                    homeRowRef={(el) => {
                      if (el) {
                        matchHomeRefs.current.set(topSemi.id, el);
                      } else {
                        matchHomeRefs.current.delete(topSemi.id);
                      }
                    }}
                    awayRowRef={(el) => {
                      if (el) {
                        matchAwayRefs.current.set(topSemi.id, el);
                      } else {
                        matchAwayRefs.current.delete(topSemi.id);
                      }
                    }}
                  />
                );
              })()
            ) : (
              <div className="h-[56px] sm:h-[72px]" />
            )}
            {bottomSemi ? (
              (() => {
                const probabilities = getMatchProbabilityLabels({
                  homeTeam: bottomSemi.homeResolved ?? bottomSemi.homeTeam,
                  awayTeam: bottomSemi.awayResolved ?? bottomSemi.awayTeam,
                  allowDraw: false,
                  neutralOverride: bottomSemi.neutral,
                });
                return (
                  <KnockoutMatchCard
                    homeTeam={bottomSemi.homeResolved ?? bottomSemi.homeTeam}
                    awayTeam={bottomSemi.awayResolved ?? bottomSemi.awayTeam}
                    winnerSelection={winnerSelections[String(bottomSemi.id)] ?? null}
                    onWinnerSelect={(selection) =>
                      onWinnerSelect(bottomSemi.id, selection)
                    }
                    flags={flags}
                    homeWinProb={probabilities.homeWinProb}
                    awayWinProb={probabilities.awayWinProb}
                    drawProb={probabilities.drawProb}
                    compact
                    className={bottomSemi.id === hintMatchId ? hintPulseClass : undefined}
                    containerRef={(el) => {
                      if (el) {
                        matchRefs.current.set(bottomSemi.id, el);
                      } else {
                        matchRefs.current.delete(bottomSemi.id);
                      }
                    }}
                    homeRowRef={(el) => {
                      if (el) {
                        matchHomeRefs.current.set(bottomSemi.id, el);
                      } else {
                        matchHomeRefs.current.delete(bottomSemi.id);
                      }
                    }}
                    awayRowRef={(el) => {
                      if (el) {
                        matchAwayRefs.current.set(bottomSemi.id, el);
                      } else {
                        matchAwayRefs.current.delete(bottomSemi.id);
                      }
                    }}
                  />
                );
              })()
            ) : (
              <div className="h-[56px] sm:h-[72px]" />
            )}
          </div>
          <div className="grid w-fit grid-rows-[56px_86px] sm:grid-rows-[72px_96px] gap-3 sm:gap-4">
            {final && (
              <KnockoutMatchCard
                homeTeam={finalHome}
                awayTeam={finalAway}
                winnerSelection={finalWinnerSelection}
                onWinnerSelect={(selection) => onWinnerSelect(final.id, selection)}
                flags={flags}
                homeWinProb={finalProbabilities?.homeWinProb}
                awayWinProb={finalProbabilities?.awayWinProb}
                drawProb={finalProbabilities?.drawProb ?? null}
                compact={false}
                containerRef={(el) => {
                  if (el) {
                    matchRefs.current.set(final.id, el);
                  } else {
                    matchRefs.current.delete(final.id);
                  }
                }}
                homeRowRef={(el) => {
                  if (el) {
                    matchHomeRefs.current.set(final.id, el);
                  } else {
                    matchHomeRefs.current.delete(final.id);
                  }
                }}
                awayRowRef={(el) => {
                  if (el) {
                    matchAwayRefs.current.set(final.id, el);
                  } else {
                    matchAwayRefs.current.delete(final.id);
                  }
                }}
              />
            )}
            <div className="flex items-center justify-center">
              {qualifiedTeam && (
                <div className="max-w-full rounded-lg px-3 py-2 bg-[radial-gradient(ellipse_at_center,rgba(219,234,254,0.7)_0%,rgba(219,234,254,0.35)_45%,rgba(219,234,254,0.15)_65%,transparent_100%)]">
                  <div className="flex max-w-full flex-col items-center gap-2 text-slate-600">
                    <div className="text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-blue-700">
                      Qualified
                    </div>
                    <div className="flex w-full max-w-[220px] flex-wrap items-center justify-center gap-1.5 sm:gap-2">
                      <TeamFlag
                        team={qualifiedTeam}
                        flags={flags}
                        className="h-5 w-7 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
                      />
                      <span className="min-w-0 w-full text-[15px] font-semibold text-slate-900 text-center break-words whitespace-normal">
                        {formatDisplayLabel(qualifiedTeam)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        </div>
      </div>
    </div>
  );

  if (embedded) {
    return (
      <div ref={containerRef} className="w-full">
        {content}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative flex w-full min-w-0 flex-col overflow-hidden rounded-xl bg-slate-50 ring-1 ring-slate-200 p-4 h-full"
    >
      {content}
    </div>
  );
}

function GroupTable({
  group,
  rows,
  highlightThird,
  highlightWeakThird,
  highlightTop = 2,
  flags,
  showTieInfo,
}: {
  group: GroupDefinition;
  rows: GroupTableRow[];
  highlightThird: boolean;
  highlightWeakThird: boolean;
  highlightTop?: number;
  flags: Record<string, string | null>;
  showTieInfo: boolean;
}) {
  const tbodyRef = React.useRef<HTMLTableSectionElement>(null);
  const tableRef = React.useRef<HTMLTableElement>(null);
  const [rowPositions, setRowPositions] = React.useState<number[]>([]);
  const [headerHeight, setHeaderHeight] = React.useState(40);

  React.useEffect(() => {
    if (!tbodyRef.current || !tableRef.current) return;
    const rows = tbodyRef.current.querySelectorAll("tr");
    const positions: number[] = [];
    let currentTop = 0;
    rows.forEach((row) => {
      positions.push(currentTop);
      currentTop += row.getBoundingClientRect().height;
    });
    setRowPositions(positions);
    if (rows.length > 0) {
      const tableTop = tableRef.current.getBoundingClientRect().top;
      const firstRowTop = rows[0].getBoundingClientRect().top;
      const offset = Math.round(firstRowTop - tableTop);
      setHeaderHeight((prev) => (prev === offset ? prev : offset));
    }
  }, [rows]);


  return (
    <div className="w-full rounded-xl bg-white ring-1 ring-slate-200 shadow-sm overflow-hidden relative">
      <style dangerouslySetInnerHTML={{ __html: `
        @media (max-width: 639px) {
          .group-table-mobile col:nth-child(1) { width: 30px !important; }
          .group-table-mobile col:nth-child(2) { min-width: 80px !important; }
          .group-table-mobile col:nth-child(3) { width: 26px !important; }
          .group-table-mobile col:nth-child(4) { width: 22px !important; }
          .group-table-mobile col:nth-child(5) { width: 22px !important; }
          .group-table-mobile col:nth-child(6) { width: 22px !important; }
          .group-table-mobile col:nth-child(7) { width: 28px !important; }
          .group-table-mobile col:nth-child(8) { width: 28px !important; }
          .group-table-mobile col:nth-child(9) { width: 26px !important; }
          .group-table-mobile col:nth-child(10) { width: 32px !important; }
        }
        @media (max-width: 479px) {
          .group-table-mobile col:nth-child(1) { width: 28px !important; }
          .group-table-mobile col:nth-child(2) { min-width: 70px !important; }
          .group-table-mobile col:nth-child(3) { width: 24px !important; }
          .group-table-mobile col:nth-child(4) { width: 20px !important; }
          .group-table-mobile col:nth-child(5) { width: 20px !important; }
          .group-table-mobile col:nth-child(6) { width: 20px !important; }
          .group-table-mobile col:nth-child(7) { width: 24px !important; }
          .group-table-mobile col:nth-child(8) { width: 24px !important; }
          .group-table-mobile col:nth-child(9) { width: 28px !important; }
          .group-table-mobile col:nth-child(10) { width: 32px !important; }
        }
        @media (max-width: 359px) {
          .group-table-mobile col:nth-child(1) { width: 26px !important; }
          .group-table-mobile col:nth-child(2) { min-width: 65px !important; }
          .group-table-mobile col:nth-child(3) { width: 22px !important; }
          .group-table-mobile col:nth-child(4) { width: 18px !important; }
          .group-table-mobile col:nth-child(5) { width: 18px !important; }
          .group-table-mobile col:nth-child(6) { width: 18px !important; }
          .group-table-mobile col:nth-child(9) { width: 26px !important; }
          .group-table-mobile col:nth-child(10) { width: 30px !important; }
          .group-table-mobile .group-table-gfga { display: none !important; }
        }
      `}} />
      {/* Qualifier markers overlay - positioned relative to table container */}
      <div className="absolute left-0 top-0 bottom-0 w-1 pointer-events-none z-10">
        {rows.map((row, index) => {
          const isTopTwo = row.position <= highlightTop;
          const isThird = row.position === 3;
          const weakHighlight = !highlightThird && highlightWeakThird && isThird;
          const hasQualifier = isTopTwo || (highlightThird && isThird) || weakHighlight;
          if (!hasQualifier || rowPositions.length === 0) return null;
          
          const rowTop = headerHeight + rowPositions[index];
          const rowHeight = index < rowPositions.length - 1 
            ? rowPositions[index + 1] - rowPositions[index]
            : 40; // fallback
          
          return (
            <div
              key={row.team}
              className={cn(
                "absolute left-0 w-full",
                isTopTwo || (highlightThird && isThird)
                  ? "bg-blue-300"
                  : "bg-emerald-300"
              )}
              style={{
                top: `${rowTop}px`,
                height: `${rowHeight}px`,
              }}
            />
          );
        })}
      </div>
      <div className="overflow-x-auto lg:overflow-visible pb-px">
        <table ref={tableRef} className="w-full table-fixed text-sm group-table-mobile">
          <colgroup>
            <col style={{ width: "40px" }} />
            <col style={{ minWidth: "100px" }} />
            <col style={{ width: "36px" }} />
            <col style={{ width: "32px" }} />
            <col style={{ width: "32px" }} />
            <col style={{ width: "32px" }} />
            <col className="group-table-gfga" style={{ width: "36px" }} />
            <col className="group-table-gfga" style={{ width: "36px" }} />
            <col style={{ width: "36px" }} />
            <col style={{ width: "44px" }} />
          </colgroup>
          <thead className="bg-slate-200 border-b border-slate-200">
            <tr>
              <th className="px-0.5 sm:px-1.5 lg:px-2 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                <span className="sm:hidden">#</span>
                <span className="hidden sm:inline">Pos</span>
              </th>
              <th className="px-1 sm:px-2 py-1.5 sm:py-2.5 text-left text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                Team
              </th>
              <th className="px-0.5 sm:px-1 lg:px-2 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                Pld
              </th>
              <th className="px-0.5 sm:px-1 lg:px-2 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                W
              </th>
              <th className="px-0.5 sm:px-1 lg:px-2 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                D
              </th>
              <th className="px-0.5 sm:px-1 lg:px-2 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                L
              </th>
              <th className="group-table-gfga px-0.5 sm:px-1 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                GF
              </th>
              <th className="group-table-gfga px-0.5 sm:px-1 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                GA
              </th>
              <th className="px-0.5 sm:px-0.5 lg:px-1 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                GD
              </th>
              <th className="px-0.5 sm:px-1.5 lg:px-2 py-1.5 sm:py-2.5 text-center text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                Pts
              </th>
            </tr>
          </thead>
          <tbody ref={tbodyRef} className="divide-y divide-slate-100">
            {rows.map((row, index) => {
              const isTopTwo = row.position <= highlightTop;
              const isThird = row.position === 3;
              const weakHighlight = !highlightThird && highlightWeakThird && isThird;
              const isCutLine = row.position === highlightTop;
              const isLastRow = index === rows.length - 1;
              const hasQualifier = isTopTwo || (highlightThird && isThird) || weakHighlight;
              return (
                <tr
                  key={row.team}
                  className={cn(
                    "transition-colors hover:bg-slate-50/70",
                    isCutLine && "border-b border-slate-200",
                    isLastRow && "border-b-0"
                  )}
                >
                  <td className="px-0.5 sm:px-1.5 lg:px-2 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm tabular-nums text-slate-600">
                    {row.position}
                  </td>
                  <td className="px-1 sm:px-2 py-1.5 sm:py-2.5">
                    <div className="flex min-w-0 items-center gap-1 sm:gap-2">
                      <TeamFlag
                        team={row.team}
                        flags={flags}
                        className="h-3.5 w-5 sm:h-4 sm:w-6 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] flex-shrink-0"
                      />
                      <span className="min-w-0 truncate text-xs sm:text-sm font-medium text-slate-900">
                        {formatDisplayLabel(row.team)}
                      </span>
                      {showTieInfo && row.randomTiebreak && (
                        <span
                          className="group relative inline-flex h-4 w-4 flex-none items-center justify-center rounded-full border border-slate-300 bg-white text-[10px] font-semibold text-slate-500 cursor-help transition-colors hover:border-slate-400 hover:bg-slate-50 hover:text-slate-700 focus-visible:border-slate-400 focus-visible:bg-slate-50 focus-visible:text-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-300 focus-visible:ring-offset-1"
                          aria-label={TIEBREAK_TOOLTIP}
                          tabIndex={0}
                        >
                          i
                          <span className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 w-52 -translate-x-1/2 rounded-md bg-slate-900 px-2.5 py-1.5 text-[11px] font-medium text-white opacity-0 shadow-lg transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100">
                            {TIEBREAK_TOOLTIP}
                          </span>
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-0.5 sm:px-1 lg:px-2 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.played}
                  </td>
                  <td className="px-0.5 sm:px-1 lg:px-2 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.wins}
                  </td>
                  <td className="px-0.5 sm:px-1 lg:px-2 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.draws}
                  </td>
                  <td className="px-0.5 sm:px-1 lg:px-2 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.losses}
                  </td>
                  <td className="group-table-gfga px-0.5 sm:px-1 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.gf}
                  </td>
                  <td className="group-table-gfga px-0.5 sm:px-1 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.ga}
                  </td>
                  <td className="px-0.5 sm:px-0.5 lg:px-1 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm font-medium tabular-nums text-slate-700 whitespace-nowrap">
                    {row.gd > 0 ? `+${row.gd}` : row.gd}
                  </td>
                  <td className="px-0.5 sm:px-1.5 lg:px-2 py-1.5 sm:py-2.5 text-center text-xs sm:text-sm font-semibold tabular-nums text-slate-900 whitespace-nowrap">
                    {row.points}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function groupMatchesFor(
  groupId: string,
  matches: GroupMatch[]
) {
  return matches
    .filter((match) => match.group === groupId)
    .sort((a, b) => a.id - b.id);
}

type GroupStageCardsProps = {
  groupTables: Array<{ group: GroupDefinition; rows: GroupTableRow[] }>;
  resolvedGroupMatches: GroupMatch[];
  groupScores: Record<string, MatchScore>;
  lockedGroupMatchIds: Set<string>;
  updateGroupScore: (id: string | number, side: "home" | "away", value: number | null) => void;
  updateGroupScorePair: (id: string | number, home: number | null, away: number | null) => void;
  winProbabilities: WinProbabilities;
  groupsWithUnresolvedParticipants: Set<string>;
  groupQualifierPaths: Map<string, string[]>;
  showGroupHint: boolean;
  groupsWithCtaMatches: Set<string>;
  getMatchProbabilityLabels: (params: {
    homeTeam: string;
    awayTeam: string;
    allowDraw: boolean;
    country?: string | null;
    neutralOverride?: boolean | null;
  }) => MatchProbabilityLabels;
  onGroupHintDismiss?: () => void;
  loadingKeys: Record<string, boolean>;
  runAutopredictWithDelay: (key: string, action: () => void) => void;
  handleGroupAutopredict: (groupId: string) => void;
  handleGroupReset: (groupId: string) => void;
  handleQualifierAutopredict: (pathId: string) => void;
  qualifierPathPredictionStatus: Map<
    string,
    { hasUnpredicted: boolean; hasPredicted: boolean }
  >;
  groupCompletion: Record<string, boolean>;
  qualifiedThirdGroups: Set<string>;
  allGroupMatchesComplete: boolean;
  flags: Record<string, string | null>;
  isTabbed: boolean;
  lockResultsActive: boolean;
};

function GroupStageCards({
  groupTables,
  resolvedGroupMatches,
  groupScores,
  lockedGroupMatchIds,
  updateGroupScore,
  updateGroupScorePair,
  winProbabilities,
  groupsWithUnresolvedParticipants,
  groupQualifierPaths,
  showGroupHint,
  groupsWithCtaMatches,
  getMatchProbabilityLabels,
  onGroupHintDismiss,
  loadingKeys,
  runAutopredictWithDelay,
  handleGroupAutopredict,
  handleGroupReset,
  handleQualifierAutopredict,
  qualifierPathPredictionStatus,
  groupCompletion,
  qualifiedThirdGroups,
  allGroupMatchesComplete,
  flags,
  isTabbed,
  lockResultsActive,
}: GroupStageCardsProps) {
  const [activeGroupId, setActiveGroupId] = React.useState<string>(
    groupTables[0]?.group.id ?? ""
  );
  const firstGroupId = groupTables[0]?.group.id ?? null;
  const groupTabsRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (!groupTables.length) {
      return;
    }
    const hasActive = groupTables.some((entry) => entry.group.id === activeGroupId);
    if (!hasActive) {
      setActiveGroupId(groupTables[0].group.id);
    }
  }, [groupTables, activeGroupId]);

  const renderGroupContent = (
    entry: { group: GroupDefinition; rows: GroupTableRow[] },
    showTitle: boolean
  ) => {
    const matches = groupMatchesFor(entry.group.id, resolvedGroupMatches);
    const qualifierPaths = groupQualifierPaths.get(entry.group.id) ?? [];
    const showQualifierWarning = qualifierPaths.length > 0;
    const qualifierLoading = qualifierPaths.some(
      (path) => loadingKeys[`qual:${path}`]
    );
    const hasUnpredictedGroupMatches = matches.some((match) => {
      const score = groupScores[String(match.id)];
      return !score || score.home === null || score.away === null;
    });
    const hasPredictedGroupMatches = matches.some((match) => {
      if (lockResultsActive && lockedGroupMatchIds.has(String(match.id))) {
        return false;
      }
      const score = groupScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
    const canAutopredictQualifiersForGroup = qualifierPaths.some(
      (path) => qualifierPathPredictionStatus.get(path)?.hasUnpredicted
    );
    return (
      <>
        <div
          className={cn(
            "flex items-center gap-2 sm:gap-3 min-h-[36px] sm:min-h-[44px]",
            showTitle ? "justify-between" : "justify-start flex-wrap mb-2 sm:mb-4"
          )}
        >
          {showTitle && (
            <h3 className="text-base sm:text-lg font-semibold text-slate-900">
              Group {entry.group.id}
            </h3>
          )}
          <div className="flex min-h-[32px] items-center gap-1.5 text-xs sm:min-h-[36px] sm:gap-2">
            {!groupsWithUnresolvedParticipants.has(entry.group.id) && (
              <>
                <LoadingButton
                  loading={Boolean(loadingKeys[`group:${entry.group.id}`])}
                  disabled={!hasUnpredictedGroupMatches}
                  onClick={() =>
                    runAutopredictWithDelay(
                      `group:${entry.group.id}`,
                      () => handleGroupAutopredict(entry.group.id)
                    )
                  }
                  className={cn(
                    "rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                    hasUnpredictedGroupMatches
                      ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                      : "text-slate-500"
                  )}
                >
                  Auto-predict
                </LoadingButton>
                <button
                  type="button"
                  disabled={!hasPredictedGroupMatches}
                  onClick={() => handleGroupReset(entry.group.id)}
                  className={cn(
                    "rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                    hasPredictedGroupMatches
                      ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                      : "text-slate-500 cursor-default"
                  )}
                >
                  Reset group
                </button>
              </>
            )}
            {groupsWithUnresolvedParticipants.has(entry.group.id) && (
              <div className="inline-flex flex-wrap items-center gap-2 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-[10px] font-medium text-red-700">
                <span>
                  {qualifierPaths.length ? qualifierPaths.join(", ") : "Qualifier"} must be predicted.
                </span>
                {showQualifierWarning && (
                  <LoadingButton
                    loading={qualifierLoading}
                    disabled={!canAutopredictQualifiersForGroup}
                    onClick={() => {
                      qualifierPaths.forEach((pathId) =>
                        runAutopredictWithDelay(`qual:${pathId}`, () =>
                          handleQualifierAutopredict(pathId)
                        )
                      );
                    }}
                    className={cn(
                      "rounded-md bg-white px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide ring-1 ring-red-200",
                      canAutopredictQualifiersForGroup
                        ? "text-red-700 hover:bg-red-100"
                        : "text-red-300 cursor-default"
                    )}
                  >
                    Auto-predict qualifier
                  </LoadingButton>
                )}
              </div>
            )}
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:gap-4">
          <div className="flex w-full flex-col gap-2 sm:gap-3 px-0.5">
            {matches.map((match) => {
              const probabilities = getMatchProbabilityLabels({
                homeTeam: match.homeTeam,
                awayTeam: match.awayTeam,
                allowDraw: true,
                country: match.country,
              });
              const isHintMatch =
                showGroupHint &&
                firstGroupId !== null &&
                entry.group.id === firstGroupId &&
                match.id === matches[0]?.id;
              return (
                <MatchCard
                  key={match.id}
                  id={match.id}
                  homeTeam={match.homeTeam}
                  awayTeam={match.awayTeam}
                  scores={groupScores}
                  onScoreChange={updateGroupScore}
                  onScoreChangePair={updateGroupScorePair}
                  allowDraw
                  orientation="horizontal"
                  flags={flags}
                  disabled={lockResultsActive && lockedGroupMatchIds.has(String(match.id))}
                  locked={lockResultsActive && lockedGroupMatchIds.has(String(match.id))}
                  homeWinProb={probabilities.homeWinProb}
                  awayWinProb={probabilities.awayWinProb}
                  drawProb={probabilities.drawProb}
                  scoreMatrix={resolveMatchScoreMatrix({
                    probabilities: winProbabilities,
                    homeTeam: match.homeTeam,
                    awayTeam: match.awayTeam,
                    country: match.country,
                  })}
                  showDivider={false}
                  showHintRow={isHintMatch}
                  onHintDismiss={isHintMatch ? onGroupHintDismiss : undefined}
                />
              );
            })}
          </div>
          <div className="flex w-full px-0.5">
            <GroupTable
              group={entry.group}
              rows={entry.rows}
              highlightThird={
                allGroupMatchesComplete && qualifiedThirdGroups.has(entry.group.id)
              }
              highlightWeakThird={!allGroupMatchesComplete}
              showTieInfo={groupCompletion[entry.group.id]}
              flags={flags}
            />
          </div>
        </div>
      </>
    );
  };

  const renderGroupCard = (
    entry: { group: GroupDefinition; rows: GroupTableRow[] },
    showTitle: boolean
  ) => {
    return (
      <div
        key={entry.group.id}
        className="relative flex w-full min-w-0 flex-col gap-2 sm:gap-4 overflow-hidden rounded-xl bg-slate-50 ring-1 ring-slate-200 p-2 sm:p-4 h-full"
      >
        {renderGroupContent(entry, showTitle)}
      </div>
    );
  };

  if (isTabbed) {
    const activeEntry =
      groupTables.find((entry) => entry.group.id === activeGroupId) ?? groupTables[0];
    if (!activeEntry) {
      return null;
    }
    return (
      <div className="relative flex w-full min-w-0 flex-col overflow-visible rounded-xl bg-slate-50 ring-1 ring-slate-200 p-2 sm:p-4 flex-1">
        <div className="border-b border-slate-200 pb-3">
          <div className="overflow-visible pl-1 pr-2">
            <div
              ref={groupTabsRef}
              role="tablist"
              aria-label="Group tabs"
              className="flex w-full min-w-0 items-center gap-2 overflow-x-auto pb-2 pt-2 pl-1 pr-2"
            >
              {groupTables.map((entry, index) => {
                const isActive = entry.group.id === activeGroupId;
                const isHighlighted = groupsWithCtaMatches.has(entry.group.id);
                const hasLeftNeighbor = index > 0;
                const hasRightNeighbor = index < groupTables.length - 1;
                return (
                  <button
                    key={entry.group.id}
                    type="button"
                    role="tab"
                    aria-selected={isActive}
                    aria-controls={`group-panel-${entry.group.id}`}
                    className={cn(
                      "inline-flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full border text-xs font-semibold uppercase tracking-wide transition-colors",
                      isHighlighted && "ring-2 ring-[color:var(--cta-color)]",
                      isActive
                        ? "border-slate-900 bg-slate-900 text-white"
                        : cn(
                            "bg-white text-slate-600 hover:bg-slate-100",
                            isHighlighted ? "border-[color:var(--cta-color)]" : "border-slate-200"
                          )
                    )}
                    onClick={(e) => {
                      setActiveGroupId(entry.group.id);
                      // Scroll button into view, showing both neighbors if they exist
                      const container = groupTabsRef.current;
                      const button = e.currentTarget;
                      if (container) {
                        const containerRect = container.getBoundingClientRect();
                        const buttonRect = button.getBoundingClientRect();
                        
                        // Check if button is fully visible
                        const isFullyVisible = 
                          buttonRect.left >= containerRect.left &&
                          buttonRect.right <= containerRect.right;
                        
                        if (!isFullyVisible || (hasLeftNeighbor && hasRightNeighbor)) {
                          // If it has both neighbors, scroll to center it so both are visible
                          if (hasLeftNeighbor && hasRightNeighbor) {
                            // Center the button in the viewport
                            const scrollLeft = button.offsetLeft - (container.clientWidth / 2) + (button.clientWidth / 2);
                            container.scrollTo({
                              left: Math.max(0, scrollLeft),
                              behavior: 'smooth',
                            });
                          } else {
                            // Just scroll the button into view
                            button.scrollIntoView({
                              behavior: 'smooth',
                              block: 'nearest',
                              inline: 'center',
                            });
                          }
                        }
                      }
                    }}
                  >
                    {entry.group.id}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
        <div id={`group-panel-${activeEntry.group.id}`} role="tabpanel" className="pt-4 flex-1 flex flex-col min-h-0">
          {renderGroupContent(activeEntry, false)}
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-6 grid-cols-1 lg:grid-cols-2 items-stretch">
      {groupTables.map((entry) => renderGroupCard(entry, true))}
    </div>
  );
}

function matchesByStage<T extends KnockoutMatch>(matches: T[]) {
  const map = new Map<string, T[]>();
  for (const match of matches) {
    if (!map.has(match.stage)) {
      map.set(match.stage, []);
    }
    map.get(match.stage)?.push(match);
  }
  for (const [stage, list] of map.entries()) {
    list.sort((a, b) => a.id - b.id);
    map.set(stage, list);
  }
  return map;
}

function sortQualifiers(matches: QualifierMatch[]) {
  const roundOrder: Record<string, number> = {
    semi1: 1,
    semi2: 2,
    semi: 1,
    final: 3,
  };
  return [...matches].sort((a, b) => {
    const dateA = Date.parse(a.date);
    const dateB = Date.parse(b.date);
    if (dateA !== dateB) {
      return dateA - dateB;
    }
    const orderA = roundOrder[a.round] ?? 99;
    const orderB = roundOrder[b.round] ?? 99;
    return orderA - orderB;
  });
}

export function WorldCupPredictorPage({
  data,
}: {
  data?: WorldCupPredictorData;
}) {
  const [loadedData, setLoadedData] = React.useState<WorldCupPredictorData | null>(
    data ?? null
  );
  const [loadError, setLoadError] = React.useState<string | null>(null);

  React.useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, []);

  React.useEffect(() => {
    if (data) {
      setLoadedData(data);
      setLoadError(null);
      return;
    }
    let canceled = false;
    loadWorldCupPredictorDataClient()
      .then((nextData) => {
        if (canceled) {
          return;
        }
        setLoadedData(nextData);
        setLoadError(null);
      })
      .catch((error) => {
        if (canceled) {
          return;
        }
        setLoadError(
          error instanceof Error ? error.message : "Failed to load predictor data."
        );
      });
    return () => {
      canceled = true;
    };
  }, [data]);

  if (loadError) {
    return (
      <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
        {loadError}
      </div>
    );
  }

  if (!loadedData) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
        Loading predictor data...
      </div>
    );
  }

  return <WorldCupPredictorContent data={loadedData} />;
}

function WorldCupPredictorContent({ data }: { data: WorldCupPredictorData }) {
  const [funnyRuns, setFunnyRuns] = React.useState<number | null>(null);
  const [showPretournament, setShowPretournament] = React.useState(false);
  const [pretournamentData, setPretournamentData] = React.useState<WorldCupPredictorData | null>(
    null
  );
  const [pretournamentLoadError, setPretournamentLoadError] = React.useState<string | null>(
    null
  );
  const showingCurrent = !showPretournament;
  const [groupScores, setGroupScores] = React.useState<
    Record<string, MatchScore>
  >({});
  const [autoGroupScores, setAutoGroupScores] = React.useState<
    Record<string, boolean>
  >({});
  const [qualifierWinners, setQualifierWinners] = React.useState<
    Record<string, WinnerSelection>
  >({});
  const [autoQualifierWinners, setAutoQualifierWinners] = React.useState<
    Record<string, boolean>
  >({});
  const [knockoutWinners, setKnockoutWinners] = React.useState<
    Record<string, WinnerSelection>
  >({});
  const [autoKnockoutWinners, setAutoKnockoutWinners] = React.useState<
    Record<string, boolean>
  >({});
  const [loadingKeys, setLoadingKeys] = React.useState<Record<string, boolean>>({});
  const loadingTimers = React.useRef<Record<string, number>>({});
  const isNarrow = false;
  const knockoutContainerRef = React.useRef<HTMLDivElement | null>(null);
  const knockoutRefs = React.useRef(new Map<number, HTMLDivElement>());
  const [knockoutPaths, setKnockoutPaths] = React.useState<string[]>([]);
  const roundOf32ListRef = React.useRef<HTMLDivElement | null>(null);
  const finalListRef = React.useRef<HTMLDivElement | null>(null);
  const [knockoutListHeight, setKnockoutListHeight] = React.useState<number | null>(null);
  const [knockoutContainerWidth, setKnockoutContainerWidth] = React.useState<number | null>(null);
  const [knockoutCenters, setKnockoutCenters] = React.useState<Record<number, number>>(
    {}
  );
  const [knockoutCardHeight, setKnockoutCardHeight] = React.useState<number | null>(
    null
  );
  const [thirdPlaceOffset, setThirdPlaceOffset] = React.useState<number | null>(null);
  const [finalCenterOverride, setFinalCenterOverride] = React.useState<number | null>(
    null
  );
  const [compactKnockout, setCompactKnockout] = React.useState(false);
  const hasUserSetCompactKnockout = React.useRef(false);
  const isSmallScreen = useMediaQuery("(max-width: 639px)");
  const pendingGroupsAfterQualifiers = React.useRef(false);
  const groupCardsContainerRef = React.useRef<HTMLDivElement | null>(null);
  const qualifierPathTabsRef = React.useRef<HTMLDivElement | null>(null);
  const isGroupTabbed = true;
  const [activeQualifierPath, setActiveQualifierPath] = React.useState<string | null>(
    null
  );
  const [showQualifierHint, setShowQualifierHint] = React.useState(true);
  const [showGroupHint, setShowGroupHint] = React.useState(true);
  const [showCompactModeHint, setShowCompactModeHint] = React.useState(false);
  const [compactModeHintDismissed, setCompactModeHintDismissed] = React.useState(false);
  const [showGroupStageContent, setShowGroupStageContent] = React.useState(false);
  const [showKnockoutSection, setShowKnockoutSection] = React.useState(false);
  const [showKnockoutContent, setShowKnockoutContent] = React.useState(false);
  const compactModeToggleRef = React.useRef<HTMLButtonElement | null>(null);
  const compactModeHintBoxRef = React.useRef<HTMLDivElement | null>(null);
  const compactModeHintPositionAdjustedRef = React.useRef(false);
  const [compactModeHintVisible, setCompactModeHintVisible] = React.useState(false);
  const [compactModeHintPosition, setCompactModeHintPosition] = React.useState<{
    x: number;
    y: number;
  } | null>(null);
  const [compactModeHintArrowLeft, setCompactModeHintArrowLeft] = React.useState<number | null>(null);
  const [shareStatus, setShareStatus] = React.useState<"idle" | "copied" | "error">(
    "idle"
  );
  const shareStatusRef = React.useRef<"idle" | "copied" | "error">("idle");
  const hasLoadedShare = React.useRef(false);
  const pendingSharedKnockouts = React.useRef<Record<string, WinnerSelection> | null>(
    null
  );
  let resolvedGroupMatches: ResolvedGroupMatch[] = [];
  let knockoutState: {
    winners: Map<number, string>;
    losers: Map<number, string>;
    matches: ResolvedKnockoutMatch[];
  } = {
    winners: new Map<number, string>(),
    losers: new Map<number, string>(),
    matches: [],
  };
  const activeWinProbabilities =
    showPretournament && pretournamentData
      ? pretournamentData.winProbabilities
      : data.winProbabilities;
  const activeSimulationTeamProbabilities =
    showPretournament && pretournamentData
      ? pretournamentData.simulationTeamProbabilities
      : data.simulationTeamProbabilities;

  React.useEffect(() => {
    if (!showPretournament || pretournamentData) {
      return;
    }
    let canceled = false;
    loadWorldCupPredictorDataClient("/model_output_pretournament")
      .then((nextData) => {
        if (canceled) {
          return;
        }
        setPretournamentData(nextData);
        setPretournamentLoadError(null);
      })
      .catch((error) => {
        if (canceled) {
          return;
        }
        setPretournamentLoadError(
          error instanceof Error
            ? error.message
            : "Failed to load pre-tournament predictor data."
        );
      });
    return () => {
      canceled = true;
    };
  }, [pretournamentData, showPretournament]);

  React.useEffect(() => {
    shareStatusRef.current = shareStatus;
  }, [shareStatus]);

  React.useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const updateFunnyRuns = () => {
      const params = new URLSearchParams(window.location.search);
      setFunnyRuns(parseFunnyRuns(params.get("funny")));
    };
    updateFunnyRuns();
    window.addEventListener("popstate", updateFunnyRuns);
    return () => {
      window.removeEventListener("popstate", updateFunnyRuns);
    };
  }, []);

  React.useEffect(() => {
    if (shareStatusRef.current !== "idle") {
      setShareStatus("idle");
    }
  }, [qualifierWinners, groupScores, knockoutWinners]);

  React.useEffect(() => {
    if (hasUserSetCompactKnockout.current) {
      return;
    }
    if (typeof window === "undefined") {
      return;
    }
    const isMobile = window.matchMedia("(max-width: 768px)").matches;
    if (isMobile) {
      setCompactKnockout(true);
    }
  }, []);

  const dismissCompactModeHint = React.useCallback(() => {
    setShowCompactModeHint(false);
    setCompactModeHintDismissed(true);
  }, []);

  // Initialize compact mode hint - only triggered when the knockout bracket appears
  React.useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (!showKnockoutContent || compactModeHintDismissed || !compactKnockout) {
      if (process.env.NODE_ENV !== "production") {
        console.log("[predictor] compact hint: gated", {
          showKnockoutContent,
          compactModeHintDismissed,
          compactKnockout,
        });
      }
      if (!compactKnockout) {
        setShowCompactModeHint(false);
      }
      return;
    }
    if (process.env.NODE_ENV !== "production") {
      console.log("[predictor] compact hint: show (knockout bracket appeared)");
    }
    setShowCompactModeHint(true);
  }, [showKnockoutContent, compactModeHintDismissed, compactKnockout]);

  // Auto-dismiss compact mode hint when any knockout team is selected (manually or automatically)
  React.useEffect(() => {
    if (!showCompactModeHint) {
      return;
    }
    const hasAnySelection = Object.values(knockoutWinners).some(
      (selection) => selection !== null && selection !== undefined
    );
    if (hasAnySelection) {
      if (process.env.NODE_ENV !== "production") {
        console.log("[predictor] compact hint: dismiss (knockout pick made)");
      }
      dismissCompactModeHint();
    }
  }, [knockoutWinners, showCompactModeHint, dismissCompactModeHint]);

  const computeCompactModeHintPosition = React.useCallback(() => {
    if (!showCompactModeHint || !compactModeToggleRef.current) {
      setCompactModeHintPosition(null);
      return;
    }
    const toggle = compactModeToggleRef.current;
    const section = toggle.closest("section");
    if (!section) {
      return;
    }
    const toggleRect = toggle.getBoundingClientRect();
    const sectionRect = section.getBoundingClientRect();
    const toggleRightX = toggleRect.left - sectionRect.left + toggleRect.width;
    // Position box so its right edge aligns with toggle's right edge
    // We'll measure the actual box width after render, but estimate for initial positioning
    const estimatedBoxWidth = 280;
    const boxX = toggleRightX - estimatedBoxWidth;
    // Ensure it doesn't go off the left edge
    setCompactModeHintPosition({
      x: Math.max(16, boxX),
      y: toggleRect.top - sectionRect.top - 4,
    });
    if (process.env.NODE_ENV !== "production") {
      console.log("[predictor] compact hint: base position", {
        toggleRightX,
        boxX,
      });
    }
    compactModeHintPositionAdjustedRef.current = false;
  }, [showCompactModeHint]);

  // Calculate compact mode hint position
  React.useLayoutEffect(() => {
    computeCompactModeHintPosition();
  }, [computeCompactModeHintPosition, compactKnockout]);

  // Recalculate hint position on resize so it stays anchored to the toggle
  React.useEffect(() => {
    if (!showCompactModeHint) {
      return;
    }
    const handleResize = () => {
      compactModeHintPositionAdjustedRef.current = false;
      computeCompactModeHintPosition();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [showCompactModeHint, computeCompactModeHintPosition]);

  // Calculate arrow position and adjust box position based on actual box width
  React.useLayoutEffect(() => {
    if (!showCompactModeHint || !compactModeHintBoxRef.current || !compactModeToggleRef.current || !compactModeHintPosition) {
      setCompactModeHintArrowLeft(null);
      return;
    }
    // Only adjust position once to avoid infinite loop
    if (compactModeHintPositionAdjustedRef.current) {
      const boxWidth = compactModeHintBoxRef.current.offsetWidth;
      setCompactModeHintArrowLeft(boxWidth * 0.75);
      return;
    }
    const box = compactModeHintBoxRef.current;
    const toggle = compactModeToggleRef.current;
    const section = toggle.closest("section");
    if (!section) {
      return;
    }
    const boxWidth = box.offsetWidth;
    const toggleRect = toggle.getBoundingClientRect();
    const sectionRect = section.getBoundingClientRect();
    const toggleRightX = toggleRect.left - sectionRect.left + toggleRect.width;
    // Reposition box so its right edge aligns with toggle's right edge
    const boxX = toggleRightX - boxWidth;
    const newX = Math.max(16, boxX);
    setCompactModeHintPosition({
      x: newX,
      y: compactModeHintPosition.y,
    });
    compactModeHintPositionAdjustedRef.current = true;
    setCompactModeHintArrowLeft(boxWidth * 0.75);
    if (process.env.NODE_ENV !== "production") {
      console.log("[predictor] compact hint: adjusted position", {
        boxWidth,
        newX,
      });
    }
  }, [showCompactModeHint, compactModeHintPosition]);

  // Show/hide compact mode hint with animation
  React.useEffect(() => {
    if (!showCompactModeHint || !compactModeHintPosition) {
      setCompactModeHintVisible(false);
      return;
    }
    setCompactModeHintVisible(false);
    const frame = requestAnimationFrame(() => setCompactModeHintVisible(true));
    if (process.env.NODE_ENV !== "production") {
      console.log("[predictor] compact hint: visible", compactModeHintPosition);
    }
    return () => cancelAnimationFrame(frame);
  }, [compactModeHintPosition, showCompactModeHint]);

  React.useEffect(() => {
    if (hasLoadedShare.current) {
      return;
    }
    hasLoadedShare.current = true;
    if (typeof window === "undefined") {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const token = params.get("p");
    if (!token) {
      return;
    }
    const decoded = decodeShareStateCompact(token, {
      qualifiers: data.qualifiers,
      groupMatches: data.groupMatches,
      knockouts: data.knockoutMatches,
    });
    if (!decoded) {
      return;
    }
    setQualifierWinners(decoded.qualifierWinners);
    setAutoQualifierWinners({});
    setGroupScores(decoded.groupScores);
    setAutoGroupScores({});
    pendingSharedKnockouts.current = decoded.knockoutWinners;
    setKnockoutWinners({});
    setAutoKnockoutWinners({});
    setShowQualifierHint(false);
    setShowGroupHint(false);
  }, []);


  React.useEffect(() => {
    return () => {
      Object.values(loadingTimers.current).forEach((timerId) => {
        clearTimeout(timerId);
      });
      loadingTimers.current = {};
    };
  }, []);

  const runAutopredictWithDelay = React.useCallback(
    (key: string, action: () => void) => {
      setLoadingKeys((prev) => {
        if (prev[key]) {
          return prev;
        }
        return { ...prev, [key]: true };
      });
      if (loadingTimers.current[key]) {
        return;
      }
      loadingTimers.current[key] = window.setTimeout(() => {
        action();
        setLoadingKeys((prev) => {
          if (!prev[key]) {
            return prev;
          }
          const next = { ...prev };
          delete next[key];
          return next;
        });
        delete loadingTimers.current[key];
      }, 300);
    },
    []
  );
  const matchStageById = React.useMemo(() => {
    const mapping: Record<number, string> = {};
    for (const match of data.knockoutMatches) {
      mapping[match.id] = match.stage;
    }
    return mapping;
  }, [data.knockoutMatches]);
  const lockedGroupScores = React.useMemo(() => {
    const locked: Record<string, MatchScore> = {};
    data.completedMatches
      .filter((match) => match.stage === "Group")
      .forEach((match) => {
        locked[String(match.matchId)] = {
          home: match.homeScore,
          away: match.awayScore,
        };
      });
    return locked;
  }, [data.completedMatches]);
  const lockedKnockoutWinners = React.useMemo(() => {
    const locked: Record<string, WinnerSelection> = {};
    data.completedMatches
      .filter((match) => match.stage !== "Group")
      .forEach((match) => {
        if (match.winner === match.homeTeam) {
          locked[String(match.matchId)] = "home";
        } else if (match.winner === match.awayTeam) {
          locked[String(match.matchId)] = "away";
        }
      });
    return locked;
  }, [data.completedMatches]);
  const lockedGroupMatchIds = React.useMemo(
    () => new Set(Object.keys(lockedGroupScores)),
    [lockedGroupScores]
  );
  const lockedKnockoutMatchIds = React.useMemo(
    () => new Set(Object.keys(lockedKnockoutWinners)),
    [lockedKnockoutWinners]
  );
  const lockResultsActive = !showPretournament;

  const getMatchProbabilityLabels = React.useCallback(
    ({
      homeTeam,
      awayTeam,
      allowDraw,
      country,
      neutralOverride,
    }: {
      homeTeam: string;
      awayTeam: string;
      allowDraw: boolean;
      country?: string | null;
      neutralOverride?: boolean | null;
    }): MatchProbabilityLabels => {
      const values = resolveMatchProbabilities({
        probabilities: activeWinProbabilities,
        homeTeam,
        awayTeam,
        allowDraw,
        country,
        neutralOverride,
      });
      if (!values) {
        return { homeWinProb: undefined, awayWinProb: undefined, drawProb: null };
      }
      // Check if any probability would round to 0 - if so, use 1dp for all
      const allValues = allowDraw
        ? [values.home, values.draw, values.away]
        : [values.home, values.away];
      const useDecimal = shouldUseDecimalPrecision(allValues);
      return {
        homeWinProb: formatProbability(values.home, useDecimal),
        awayWinProb: formatProbability(values.away, useDecimal),
        drawProb: allowDraw ? formatProbability(values.draw, useDecimal) ?? null : null,
      };
    },
    [activeWinProbabilities]
  );

  const qualifierDependents = React.useMemo(() => {
    const byPathRound = new Map<string, string>();
    data.qualifiers.forEach((match) => {
      byPathRound.set(`${match.path}:${match.round}`, String(match.id));
    });
    const deps = new Map<string, Set<string>>();
    data.qualifiers.forEach((match) => {
      [match.homeSource, match.awaySource]
        .filter(Boolean)
        .forEach((source) => {
          const sourceId = byPathRound.get(`${match.path}:${source}`);
          if (!sourceId) {
            return;
          }
          if (!deps.has(sourceId)) {
            deps.set(sourceId, new Set());
          }
          deps.get(sourceId)?.add(String(match.id));
        });
    });
    return deps;
  }, [data.qualifiers]);

  const qualifierSlotsByMatch = React.useMemo(() => {
    const slotsByMatch = new Map<string, Set<string>>();
    const matchById = new Map<string, QualifierMatch>();
    data.qualifiers.forEach((match) => {
      matchById.set(String(match.id), match);
    });

    const collectSlots = (matchId: string, visited: Set<string>) => {
      if (visited.has(matchId)) {
        return new Set<string>();
      }
      visited.add(matchId);
      const slots = new Set<string>();
      const match = matchById.get(matchId);
      if (match?.winnerSlot) {
        slots.add(match.winnerSlot);
      }
      const deps = qualifierDependents.get(matchId);
      if (deps) {
        deps.forEach((dep) => {
          collectSlots(dep, visited).forEach((slot) => slots.add(slot));
        });
      }
      return slots;
    };

    matchById.forEach((_, matchId) => {
      slotsByMatch.set(matchId, collectSlots(matchId, new Set()));
    });

    return slotsByMatch;
  }, [data.qualifiers, qualifierDependents]);

  const groupMatchIdsByTeam = React.useMemo(() => {
    const map = new Map<string, Set<string>>();
    data.groupMatches.forEach((match) => {
      const home = match.homeTeam;
      const away = match.awayTeam;
      if (!map.has(home)) {
        map.set(home, new Set());
      }
      if (!map.has(away)) {
        map.set(away, new Set());
      }
      map.get(home)?.add(String(match.id));
      map.get(away)?.add(String(match.id));
    });
    return map;
  }, [data.groupMatches]);

  const groupIdsBySlot = React.useMemo(() => {
    const map = new Map<string, Set<string>>();
    data.groups.forEach((group) => {
      group.teams.forEach((team) => {
        if (!map.has(team)) {
          map.set(team, new Set());
        }
        map.get(team)?.add(group.id);
      });
    });
    return map;
  }, [data.groups]);

  const knockoutRootsByGroup = React.useMemo(() => {
    const map = new Map<string, Set<string>>();
    data.knockoutMatches.forEach((match) => {
      [match.homeLabel, match.awayLabel].forEach((label) => {
        const groupId = extractGroupId(label);
        if (!groupId) {
          return;
        }
        if (!map.has(groupId)) {
          map.set(groupId, new Set());
        }
        map.get(groupId)?.add(String(match.id));
      });
    });
    return map;
  }, [data.knockoutMatches]);

  const knockoutDependents = React.useMemo(() => {
    const deps = new Map<string, Set<string>>();
    data.knockoutMatches.forEach((match) => {
      [match.homeLabel, match.awayLabel].forEach((label) => {
        if (label.startsWith("Winner Match ")) {
          const from = label.replace("Winner Match ", "").trim();
          if (!deps.has(from)) {
            deps.set(from, new Set());
          }
          deps.get(from)?.add(String(match.id));
        } else if (label.startsWith("Loser Match ")) {
          const from = label.replace("Loser Match ", "").trim();
          if (!deps.has(from)) {
            deps.set(from, new Set());
          }
          deps.get(from)?.add(String(match.id));
        }
      });
    });
    return deps;
  }, [data.knockoutMatches]);

  const updateQualifierWinner = React.useCallback(
    (id: string | number, selection: WinnerSelection) => {
      let changed = false;
      const key = String(id);
      setQualifierWinners((prev) => {
        if ((prev[key] ?? null) === selection) {
          return prev;
        }
        changed = true;
        const next = { ...prev, [key]: selection };
        return clearDependentSelections(next, key, qualifierDependents);
      });
      if (changed) {
        setShowQualifierHint(false);
        setAutoQualifierWinners((prev) => {
          if (!prev[key]) {
            return prev;
          }
          const next = { ...prev };
          delete next[key];
          return next;
        });
        const affectedSlots =
          qualifierSlotsByMatch.get(String(id)) ?? new Set<string>();
        const affectedGroups = new Set<string>();
        affectedSlots.forEach((slot) => {
          const groups = groupIdsBySlot.get(slot);
          if (groups) {
            groups.forEach((groupId) => affectedGroups.add(groupId));
          }
        });

        if (affectedGroups.size > 0) {
          setGroupScores((prev) => {
            if (!Object.keys(prev).length) {
              return prev;
            }
            const next = { ...prev };
            affectedSlots.forEach((slot) => {
              const matchIds = groupMatchIdsByTeam.get(slot);
              if (!matchIds) {
                return;
              }
              matchIds.forEach((matchId) => {
                delete next[matchId];
              });
            });
            return next;
          });
          setAutoGroupScores((prev) => {
            const next = { ...prev };
            affectedSlots.forEach((slot) => {
              const matchIds = groupMatchIdsByTeam.get(slot);
              if (!matchIds) {
                return;
              }
              matchIds.forEach((matchId) => {
                delete next[matchId];
              });
            });
            return next;
          });

          setKnockoutWinners((prev) => {
            if (!Object.keys(prev).length) {
              return prev;
            }
            let next = { ...prev };
            affectedGroups.forEach((groupId) => {
              const rootMatches = knockoutRootsByGroup.get(groupId);
              if (rootMatches) {
                rootMatches.forEach((matchId) => {
                  next[matchId] = null;
                  next = clearDependentSelections(
                    next,
                    matchId,
                    knockoutDependents
                  );
                });
              }
            });
            const clearedIds = Object.keys(prev).filter(
              (matchId) => prev[matchId] && next[matchId] === null
            );
            if (clearedIds.length > 0) {
              setAutoKnockoutWinners((currentAuto) => {
                const nextAuto = { ...currentAuto };
                clearedIds.forEach((matchId) => {
                  delete nextAuto[matchId];
                });
                return nextAuto;
              });
            }
            return next;
          });
        }
      }
    },
    [
      qualifierDependents,
      qualifierSlotsByMatch,
      groupIdsBySlot,
      groupMatchIdsByTeam,
      knockoutRootsByGroup,
      knockoutDependents,
      setShowQualifierHint,
    ]
  );

  const updateKnockoutWinner = React.useCallback(
    (id: string | number, selection: WinnerSelection) => {
      const key = String(id);
      if (lockResultsActive && lockedKnockoutMatchIds.has(key)) {
        return;
      }
      setAutoKnockoutWinners((prev) => {
        if (!prev[key]) {
          return prev;
        }
        const next = { ...prev };
        delete next[key];
        return next;
      });
      setKnockoutWinners((prev) => {
        if ((prev[key] ?? null) === selection) {
          return prev;
        }
        const next = { ...prev, [key]: selection };
        return clearDependentSelections(next, key, knockoutDependents);
      });
    },
    [knockoutDependents, lockResultsActive, lockedKnockoutMatchIds]
  );

  const qualifierState = React.useMemo(
    () => resolveQualifierState(data.qualifiers, qualifierWinners),
    [data.qualifiers, qualifierWinners]
  );

  const slotWinners = qualifierState.slotWinners;
  const qualifierEntries = React.useMemo(() => {
    return Array.from(
      qualifierState.matches.reduce((map, match) => {
        if (!map.has(match.path)) {
          map.set(match.path, []);
        }
        map.get(match.path)?.push(match);
        return map;
      }, new Map<string, ResolvedQualifierMatch[]>())
    ).sort(([a], [b]) => {
      const order = [
        "IC Path 1",
        "IC Path 2",
        "UEFA Path A",
        "UEFA Path B",
        "UEFA Path C",
        "UEFA Path D",
      ];
      const indexA = order.indexOf(a);
      const indexB = order.indexOf(b);
      if (indexA !== -1 || indexB !== -1) {
        return (indexA === -1 ? 99 : indexA) - (indexB === -1 ? 99 : indexB);
      }
      return a.localeCompare(b);
    });
  }, [qualifierState.matches]);
  const firstQualifierPath = qualifierEntries[0]?.[0] ?? null;
  const qualifierPathsWithCtaMatches = React.useMemo(() => {
    const paths = new Set<string>();
    qualifierState.matches.forEach((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return;
      }
      const key = String(match.id);
      if ((qualifierWinners[key] ?? null) === null) {
        paths.add(match.path);
      }
    });
    return paths;
  }, [qualifierState.matches, qualifierWinners]);
  const qualifierSlots = React.useMemo(() => {
    const slots: string[] = [];
    const seen = new Set<string>();
    data.qualifiers.forEach((match) => {
      if (!match.winnerSlot || seen.has(match.winnerSlot)) {
        return;
      }
      seen.add(match.winnerSlot);
      slots.push(match.winnerSlot);
    });
    return slots;
  }, [data.qualifiers]);
  const qualifierPathBySlot = React.useMemo(() => {
    const map = new Map<string, string>();
    data.qualifiers.forEach((match) => {
      if (match.winnerSlot && match.path) {
        map.set(match.winnerSlot, match.path);
      }
    });
    return map;
  }, [data.qualifiers]);
  const qualifierQualifiedRows = React.useMemo(() => {
    const rows = qualifierSlots.map((slot) => ({
      slot,
      path: qualifierPathBySlot.get(slot) ?? "Qualifier",
      team: qualifierState.slotWinners.get(slot) ?? null,
    }));
    const filled = rows.slice(0, 6);
    while (filled.length < 6) {
      filled.push({ slot: `empty-${filled.length}`, path: "Qualifier", team: null });
    }
    return filled;
  }, [qualifierSlots, qualifierPathBySlot, qualifierState.slotWinners]);

  React.useEffect(() => {
    if (!qualifierEntries.length) {
      return;
    }
    const hasActive = qualifierEntries.some(([path]) => path === activeQualifierPath);
    if (!hasActive) {
      setActiveQualifierPath(qualifierEntries[0][0]);
    }
  }, [activeQualifierPath, qualifierEntries]);

  const hasUnpredictedQualifiers = React.useCallback(() => {
    return qualifierState.matches.some((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return false;
      }
      const key = String(match.id);
      return (qualifierWinners[key] ?? null) === null;
    });
  }, [qualifierState.matches, qualifierWinners]);

  const unpickableQualifierIds = React.useMemo(() => {
    return qualifierState.matches
      .filter(
        (match) =>
          !isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)
      )
      .map((match) => String(match.id));
  }, [qualifierState.matches]);

  React.useEffect(() => {
    if (unpickableQualifierIds.length === 0) {
      return;
    }
    setQualifierWinners((prev) => {
      let next = { ...prev };
      let changed = false;
      unpickableQualifierIds.forEach((matchId) => {
        if (next[matchId]) {
          next = clearDependentSelections(next, matchId, qualifierDependents);
          changed = true;
        }
      });
      return changed ? next : prev;
    });
    setAutoQualifierWinners((prev) => {
      if (!Object.keys(prev).length) {
        return prev;
      }
      const next = { ...prev };
      unpickableQualifierIds.forEach((matchId) => {
        delete next[matchId];
      });
      return next;
    });
  }, [qualifierDependents, unpickableQualifierIds]);

  const computeKnockoutContext = React.useCallback(
    (
      scores: Record<string, MatchScore>,
      slotWinnersOverride?: Map<string, string>
    ) => {
      const resolvedSlots = slotWinnersOverride ?? slotWinners;
      const resolvedMatches = data.groupMatches.map((match) => ({
        ...match,
        homeTeam: resolvedSlots.get(match.homeTeam) ?? match.homeTeam,
        awayTeam: resolvedSlots.get(match.awayTeam) ?? match.awayTeam,
      }));
      const resolvedGroupsLocal = data.groups.map((group) => ({
        ...group,
        teams: group.teams.map((team) => resolvedSlots.get(team) ?? team),
      }));
      const groupTablesLocal = resolvedGroupsLocal.map((group) => {
        const matches = groupMatchesFor(group.id, resolvedMatches);
        const { table, ranking } = buildGroupTable(group, matches, scores);
        const rows = ranking.map((team) => table[team]).filter(Boolean);
        return { group, ranking, table, rows };
      });
      const groupRankingsLocal: Record<string, string[]> = {};
      const groupCompletionLocal: Record<string, boolean> = {};
      groupTablesLocal.forEach((entry) => {
        groupRankingsLocal[entry.group.id] = entry.ranking;
        const matches = groupMatchesFor(entry.group.id, resolvedMatches);
        groupCompletionLocal[entry.group.id] = matches.every((match) => {
          const score = scores[String(match.id)];
          return score && score.home !== null && score.away !== null;
        });
      });
      const thirdPlaceEntries = bestThirdPlace(groupTablesLocal).entries;
      const thirdPlaceByGroupLocal: Record<string, string> = {};
      thirdPlaceEntries.forEach((entry) => {
        if (!thirdPlaceByGroupLocal[entry.group]) {
          thirdPlaceByGroupLocal[entry.group] = entry.team;
        }
      });
      const bestThirdGroups = thirdPlaceEntries.slice(0, 8);
      const qualifiedThirdGroupsLocal = new Set(
        bestThirdGroups.map((entry) => entry.group)
      );
      const groups = bestThirdGroups.map((entry) => entry.group).sort();
      const comboKey = groups.join("");
      const thirdPlaceAssignments =
        comboKey && data.roundOf32Combos[comboKey]
          ? data.roundOf32Combos[comboKey]
          : null;
      const allGroupMatchesComplete = resolvedMatches.every((match) => {
        const score = scores[String(match.id)];
        return score && score.home !== null && score.away !== null;
      });
      const labels = new Map<string, { home: string; away: string }>();
      data.knockoutMatches.forEach((match) => {
        const homeResolved = resolveKnockoutLabel({
          label: match.homeLabel,
          opponentLabel: match.awayLabel,
          groupRankings: groupRankingsLocal,
          thirdPlaceByGroup: thirdPlaceByGroupLocal,
          thirdPlaceAssignments,
          knockoutWinners: new Map(),
          knockoutLosers: new Map(),
          groupCompletion: groupCompletionLocal,
          allowThirdPlaceResolve: allGroupMatchesComplete,
          qualifiedThirdGroups: qualifiedThirdGroupsLocal,
          matchStageById,
        });
        const awayResolved = resolveKnockoutLabel({
          label: match.awayLabel,
          opponentLabel: match.homeLabel,
          groupRankings: groupRankingsLocal,
          thirdPlaceByGroup: thirdPlaceByGroupLocal,
          thirdPlaceAssignments,
          knockoutWinners: new Map(),
          knockoutLosers: new Map(),
          groupCompletion: groupCompletionLocal,
          allowThirdPlaceResolve: allGroupMatchesComplete,
          qualifiedThirdGroups: qualifiedThirdGroupsLocal,
          matchStageById,
        });
        labels.set(String(match.id), {
          home: homeResolved,
          away: awayResolved,
        });
      });
      return {
        labels,
        allGroupMatchesComplete,
        comboKey,
        bestThirdGroups: bestThirdGroups.map((entry) => entry.group),
        qualifiedThirdGroups: Array.from(qualifiedThirdGroupsLocal),
        thirdPlaceAssignments,
      thirdPlaceByGroup: thirdPlaceByGroupLocal,
      groupRankings: groupRankingsLocal,
      groupCompletion: groupCompletionLocal,
    };
  },
  [
    data.groupMatches,
    data.groups,
    data.knockoutMatches,
    data.roundOf32Combos,
    matchStageById,
    slotWinners,
  ]
);

  const clearKnockoutSelectionsByMatchIds = React.useCallback(
    (
      current: Record<string, WinnerSelection>,
      matchIds: Iterable<string>
    ) => {
      let next = { ...current };
      for (const matchId of matchIds) {
        next[matchId] = null;
        next = clearDependentSelections(next, matchId, knockoutDependents);
      }
      const clearedIds = Object.keys(current).filter(
        (id) => current[id] && next[id] === null
      );
      return { next, clearedIds };
    },
    [knockoutDependents]
  );

  const computeClearedKnockoutSelections = React.useCallback(
    (
      current: Record<string, WinnerSelection>,
      previousScores: Record<string, MatchScore>,
      nextScores: Record<string, MatchScore>,
      options?: {
        previousSlotWinners?: Map<string, string>;
        nextSlotWinners?: Map<string, string>;
      }
    ) => {
      if (!Object.keys(current).length) {
        return { nextWinners: current, clearedIds: [] as string[] };
      }
      const previousContext = computeKnockoutContext(
        previousScores,
        options?.previousSlotWinners
      );
      const nextContext = computeKnockoutContext(
        nextScores,
        options?.nextSlotWinners
      );
      const previousLabels = previousContext.labels;
      const nextLabels = nextContext.labels;
      const changedMatches = new Set<string>();
      data.knockoutMatches.forEach((match) => {
        const key = String(match.id);
        const before = previousLabels.get(key);
        const after = nextLabels.get(key);
        if (!before || !after) {
          return;
        }
        if (before.home !== after.home || before.away !== after.away) {
          changedMatches.add(key);
        }
      });
      if (changedMatches.size === 0) {
        return { nextWinners: current, clearedIds: [] as string[] };
      }
      const { next, clearedIds } = clearKnockoutSelectionsByMatchIds(
        current,
        changedMatches
      );
      return { nextWinners: next, clearedIds };
    },
    [clearKnockoutSelectionsByMatchIds, computeKnockoutContext, data.knockoutMatches, matchStageById]
  );

  const clearKnockoutOnGroupChange = React.useCallback(
    (nextScores: Record<string, MatchScore>) => {
      setKnockoutWinners((prev) => {
        const { nextWinners, clearedIds } = computeClearedKnockoutSelections(
          prev,
          groupScores,
          nextScores
        );
        if (clearedIds.length > 0) {
          setAutoKnockoutWinners((currentAuto) => {
            const nextAuto = { ...currentAuto };
            clearedIds.forEach((matchId) => {
              delete nextAuto[matchId];
            });
            return nextAuto;
          });
        }
        return nextWinners;
      });
    },
    [computeClearedKnockoutSelections, groupScores]
  );

  const createAutopredictSnapshot = React.useCallback(
    (): AutopredictSnapshot => ({
      qualifierWinners: { ...qualifierWinners },
      autoQualifierWinners: { ...autoQualifierWinners },
      groupScores: { ...groupScores },
      autoGroupScores: { ...autoGroupScores },
      knockoutWinners: { ...knockoutWinners },
      autoKnockoutWinners: { ...autoKnockoutWinners },
      funninessScore: 0,
    }),
    [
      autoGroupScores,
      autoKnockoutWinners,
      autoQualifierWinners,
      groupScores,
      knockoutWinners,
      qualifierWinners,
    ]
  );

  const commitAutopredictSnapshot = React.useCallback((snapshot: AutopredictSnapshot) => {
    setQualifierWinners(snapshot.qualifierWinners);
    setAutoQualifierWinners(snapshot.autoQualifierWinners);
    setGroupScores(snapshot.groupScores);
    setAutoGroupScores(snapshot.autoGroupScores);
    setKnockoutWinners(snapshot.knockoutWinners);
    setAutoKnockoutWinners(snapshot.autoKnockoutWinners);
  }, []);

  const enforceLockedResults = React.useCallback(() => {
    if (!lockResultsActive) {
      return;
    }
    let nextGroupScores = groupScores;
    let nextAutoGroupScores = autoGroupScores;
    let nextKnockoutWinners = knockoutWinners;
    let nextAutoKnockoutWinners = autoKnockoutWinners;
    let changed = false;

    for (const [matchId, lockedScore] of Object.entries(lockedGroupScores)) {
      const existing = nextGroupScores[matchId];
      if (
        existing?.home !== lockedScore.home ||
        existing?.away !== lockedScore.away
      ) {
        nextGroupScores = {
          ...nextGroupScores,
          [matchId]: { home: lockedScore.home, away: lockedScore.away },
        };
        changed = true;
      }
    }
    for (const matchId of Object.keys(lockedGroupScores)) {
      if (nextAutoGroupScores[matchId]) {
        if (nextAutoGroupScores === autoGroupScores) {
          nextAutoGroupScores = { ...nextAutoGroupScores };
        }
        delete nextAutoGroupScores[matchId];
        changed = true;
      }
    }

    if (nextGroupScores !== groupScores) {
      const cleared = computeClearedKnockoutSelections(
        nextKnockoutWinners,
        groupScores,
        nextGroupScores
      );
      nextKnockoutWinners = cleared.nextWinners;
      if (cleared.clearedIds.length > 0) {
        if (nextAutoKnockoutWinners === autoKnockoutWinners) {
          nextAutoKnockoutWinners = { ...nextAutoKnockoutWinners };
        }
        cleared.clearedIds.forEach((matchId) => {
          delete nextAutoKnockoutWinners[matchId];
        });
        changed = true;
      }
    }

    const sortedLockedKnockouts = Object.keys(lockedKnockoutWinners).sort(
      (a, b) => Number(a) - Number(b)
    );
    for (const matchId of sortedLockedKnockouts) {
      const lockedSelection = lockedKnockoutWinners[matchId] ?? null;
      const previous = nextKnockoutWinners[matchId] ?? null;
      if (previous !== lockedSelection) {
        const updated = { ...nextKnockoutWinners, [matchId]: lockedSelection };
        const cleared = clearDependentSelections(updated, matchId, knockoutDependents);
        const clearedIds = Object.keys(updated).filter(
          (id) => updated[id] && cleared[id] === null
        );
        nextKnockoutWinners = cleared;
        if (clearedIds.length > 0) {
          if (nextAutoKnockoutWinners === autoKnockoutWinners) {
            nextAutoKnockoutWinners = { ...nextAutoKnockoutWinners };
          }
          clearedIds.forEach((id) => {
            delete nextAutoKnockoutWinners[id];
          });
        }
        changed = true;
      }
      if (nextAutoKnockoutWinners[matchId]) {
        if (nextAutoKnockoutWinners === autoKnockoutWinners) {
          nextAutoKnockoutWinners = { ...nextAutoKnockoutWinners };
        }
        delete nextAutoKnockoutWinners[matchId];
        changed = true;
      }
    }

    if (!changed) {
      return;
    }
    setGroupScores(nextGroupScores);
    setAutoGroupScores(nextAutoGroupScores);
    setKnockoutWinners(nextKnockoutWinners);
    setAutoKnockoutWinners(nextAutoKnockoutWinners);
  }, [
    autoGroupScores,
    autoKnockoutWinners,
    clearDependentSelections,
    computeClearedKnockoutSelections,
    groupScores,
    knockoutDependents,
    knockoutWinners,
    lockResultsActive,
    lockedGroupScores,
    lockedKnockoutWinners,
  ]);

  React.useEffect(() => {
    enforceLockedResults();
  }, [enforceLockedResults]);

  const progressionFunnyPenalty = React.useCallback(
    (snapshot: AutopredictSnapshot) => {
      const terminalStages = new Map<string, FinalProgressionStage>();
      const slotWinnersLocal = resolveQualifierState(
        data.qualifiers,
        snapshot.qualifierWinners
      ).slotWinners;
      const resolvedGroupsLocal = data.groups.map((group) => ({
        ...group,
        teams: group.teams.map((team) => slotWinnersLocal.get(team) ?? team),
      }));
      const resolvedGroupMatchesLocal = data.groupMatches.map((match) => ({
        ...match,
        homeTeam: slotWinnersLocal.get(match.homeTeam) ?? match.homeTeam,
        awayTeam: slotWinnersLocal.get(match.awayTeam) ?? match.awayTeam,
      }));
      const groupTablesLocal = resolvedGroupsLocal.map((group) => {
        const matches = groupMatchesFor(group.id, resolvedGroupMatchesLocal);
        const { table, ranking } = buildGroupTable(group, matches, snapshot.groupScores);
        return { group, table, ranking };
      });
      const groupRankingsLocal: Record<string, string[]> = {};
      const groupCompletionLocal: Record<string, boolean> = {};
      for (const entry of groupTablesLocal) {
        groupRankingsLocal[entry.group.id] = entry.ranking;
        const matches = groupMatchesFor(entry.group.id, resolvedGroupMatchesLocal);
        groupCompletionLocal[entry.group.id] = matches.every((match) => {
          const score = snapshot.groupScores[String(match.id)];
          return score && score.home !== null && score.away !== null;
        });
      }
      for (const entry of groupTablesLocal) {
        if (!groupCompletionLocal[entry.group.id]) {
          continue;
        }
        const fourthPlacedTeam = entry.ranking[3];
        if (isConcreteTeam(fourthPlacedTeam)) {
          terminalStages.set(fourthPlacedTeam, "Group stage");
        }
      }
      const thirdPlaceEntriesLocal = bestThirdPlace(groupTablesLocal).entries;
      const thirdPlaceByGroupLocal: Record<string, string> = {};
      for (const entry of thirdPlaceEntriesLocal) {
        if (!thirdPlaceByGroupLocal[entry.group]) {
          thirdPlaceByGroupLocal[entry.group] = entry.team;
        }
      }
      const bestThirdGroupsLocal = thirdPlaceEntriesLocal.slice(0, 8);
      const qualifiedThirdGroupsLocal = new Set(
        bestThirdGroupsLocal.map((entry) => entry.group)
      );
      const comboKeyLocal = bestThirdGroupsLocal
        .map((entry) => entry.group)
        .sort()
        .join("");
      const thirdPlaceAssignmentsLocal = comboKeyLocal
        ? data.roundOf32Combos[comboKeyLocal] ?? null
        : null;
      const allGroupMatchesCompleteLocal = resolvedGroupMatchesLocal.every((match) => {
        const score = snapshot.groupScores[String(match.id)];
        return score && score.home !== null && score.away !== null;
      });

      if (allGroupMatchesCompleteLocal) {
        const qualifiedTeams = new Set<string>();
        for (const entry of groupTablesLocal) {
          entry.ranking.slice(0, 2).forEach((team) => qualifiedTeams.add(team));
        }
        for (const groupId of qualifiedThirdGroupsLocal) {
          const team = thirdPlaceByGroupLocal[groupId];
          if (team) {
            qualifiedTeams.add(team);
          }
        }
        for (const entry of groupTablesLocal) {
          for (const team of entry.ranking) {
            if (
              isConcreteTeam(team) &&
              !qualifiedTeams.has(team) &&
              !terminalStages.has(team)
            ) {
              terminalStages.set(team, "Group stage");
            }
          }
        }
      }

      const winners = new Map<number, string>();
      const losers = new Map<number, string>();
      const sorted = [...data.knockoutMatches].sort((a, b) => a.id - b.id);
      for (const match of sorted) {
        const homeResolved = resolveKnockoutLabel({
          label: match.homeLabel,
          opponentLabel: match.awayLabel,
          groupRankings: groupRankingsLocal,
          thirdPlaceByGroup: thirdPlaceByGroupLocal,
          thirdPlaceAssignments: thirdPlaceAssignmentsLocal,
          knockoutWinners: winners,
          knockoutLosers: losers,
          groupCompletion: groupCompletionLocal,
          allowThirdPlaceResolve: allGroupMatchesCompleteLocal,
          qualifiedThirdGroups: qualifiedThirdGroupsLocal,
          matchStageById,
        });
        const awayResolved = resolveKnockoutLabel({
          label: match.awayLabel,
          opponentLabel: match.homeLabel,
          groupRankings: groupRankingsLocal,
          thirdPlaceByGroup: thirdPlaceByGroupLocal,
          thirdPlaceAssignments: thirdPlaceAssignmentsLocal,
          knockoutWinners: winners,
          knockoutLosers: losers,
          groupCompletion: groupCompletionLocal,
          allowThirdPlaceResolve: allGroupMatchesCompleteLocal,
          qualifiedThirdGroups: qualifiedThirdGroupsLocal,
          matchStageById,
        });
        const winner = resolveWinner(
          match.id,
          homeResolved,
          awayResolved,
          {},
          false,
          snapshot.knockoutWinners
        );
        if (!winner) {
          continue;
        }
        const loser = winner === homeResolved ? awayResolved : homeResolved;
        winners.set(match.id, winner);
        losers.set(match.id, loser);
        if (!isConcreteTeam(loser) || !isConcreteTeam(winner)) {
          continue;
        }
        switch (match.stage) {
          case "Round of 32":
          case "Round of 16":
          case "Quarterfinal":
            terminalStages.set(loser, match.stage);
            break;
          case "Third place":
            terminalStages.set(winner, "Third place");
            terminalStages.set(loser, "Semifinal");
            break;
          case "Final":
            terminalStages.set(loser, "Reach Final");
            terminalStages.set(winner, "Champion");
            break;
          default:
            break;
        }
      }

      let penalty = 0;
      terminalStages.forEach((stage, team) => {
        const exactProbability = exactProgressionProbability(
          activeSimulationTeamProbabilities[team],
          stage
        );
        if (exactProbability === null) {
          return;
        }
        penalty += probabilityPenalty(exactProbability, progressionPlacementMultiplier(stage));
      });
      return penalty;
    },
    [activeSimulationTeamProbabilities, data.groupMatches, data.groups, data.knockoutMatches, data.qualifiers, data.roundOf32Combos, matchStageById]
  );

  const chooseAutopredictSnapshot = React.useCallback(
    (simulate: () => AutopredictSnapshot | null) => {
      const attempts = funnyRuns ?? 1;
      let best: AutopredictSnapshot | null = null;
      let bestScore = Number.POSITIVE_INFINITY;
      for (let i = 0; i < attempts; i += 1) {
        const candidate = simulate();
        if (!candidate) {
          continue;
        }
        const candidateScore = progressionFunnyPenalty(candidate);
        if (!best || candidateScore < bestScore) {
          best = candidate;
          bestScore = candidateScore;
        }
      }
      return best;
    },
    [funnyRuns, progressionFunnyPenalty]
  );

  const simulateGroupAutopredict = React.useCallback(
    (targetGroupId: string): AutopredictSnapshot | null => {
      const matches = groupMatchesFor(targetGroupId, resolvedGroupMatches);
      if (matches.length === 0) {
        return null;
      }
      const snapshot = createAutopredictSnapshot();
      const allPredicted = matches.every((match) => {
        const existing = snapshot.groupScores[String(match.id)];
        return existing && existing.home !== null && existing.away !== null;
      });
      if (allPredicted) {
        return null;
      }
      let changed = false;
      matches.forEach((match) => {
        const key = String(match.id);
        const existing = snapshot.groupScores[key];
        const hasScore =
          existing && existing.home !== null && existing.away !== null;
        if (hasScore) {
          return;
        }
        const matrix = resolveMatchScoreMatrix({
          probabilities: activeWinProbabilities,
          homeTeam: match.homeTeam,
          awayTeam: match.awayTeam,
          country: match.country,
        });
        if (!matrix) {
          return;
        }
        const sample = sampleScoreMatrixWithProbability(matrix);
        if (!sample) {
          return;
        }
        snapshot.groupScores[key] = { home: sample.home, away: sample.away };
        snapshot.autoGroupScores[key] = true;
        snapshot.funninessScore += groupScorePenalty(matrix, sample);
        changed = true;
      });
      if (!changed) {
        return null;
      }
      const cleared = computeClearedKnockoutSelections(
        snapshot.knockoutWinners,
        groupScores,
        snapshot.groupScores
      );
      snapshot.knockoutWinners = cleared.nextWinners;
      cleared.clearedIds.forEach((matchId) => {
        delete snapshot.autoKnockoutWinners[matchId];
      });
      return snapshot;
    },
    [
      computeClearedKnockoutSelections,
      createAutopredictSnapshot,
      activeWinProbabilities,
      groupScores,
      resolvedGroupMatches,
    ]
  );

  const simulateSectionGroupsAutopredict = React.useCallback((): AutopredictSnapshot | null => {
    const snapshot = createAutopredictSnapshot();
    const allPredicted = resolvedGroupMatches.every((match) => {
      const score = snapshot.groupScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
    if (allPredicted) {
      return null;
    }
    let changed = false;
    resolvedGroupMatches.forEach((match) => {
      const key = String(match.id);
      const existing = snapshot.groupScores[key];
      const hasScore =
        existing && existing.home !== null && existing.away !== null;
      if (hasScore) {
        return;
      }
      const matrix = resolveMatchScoreMatrix({
        probabilities: activeWinProbabilities,
        homeTeam: match.homeTeam,
        awayTeam: match.awayTeam,
        country: match.country,
      });
      if (!matrix) {
        return;
      }
      const sample = sampleScoreMatrixWithProbability(matrix);
      if (!sample) {
        return;
      }
      snapshot.groupScores[key] = { home: sample.home, away: sample.away };
      snapshot.autoGroupScores[key] = true;
      snapshot.funninessScore += groupScorePenalty(matrix, sample);
      changed = true;
    });
    if (!changed) {
      return null;
    }
    const cleared = computeClearedKnockoutSelections(
      snapshot.knockoutWinners,
      groupScores,
      snapshot.groupScores
    );
    snapshot.knockoutWinners = cleared.nextWinners;
    cleared.clearedIds.forEach((matchId) => {
      delete snapshot.autoKnockoutWinners[matchId];
    });
    return snapshot;
  }, [
    computeClearedKnockoutSelections,
    createAutopredictSnapshot,
    activeWinProbabilities,
    groupScores,
    resolvedGroupMatches,
  ]);

  const simulateQualifierAutopredict = React.useCallback(
    (path: string): AutopredictSnapshot | null => {
      const snapshot = createAutopredictSnapshot();
      const changedMatchIds = new Set<string>();
      const qualifierStateLocal = resolveQualifierState(
        data.qualifiers,
        snapshot.qualifierWinners
      );
      const matches = qualifierStateLocal.matches.filter((match) => match.path === path);
      if (matches.length === 0) {
        return null;
      }
      const allPredicted = matches.every((match) => {
        const key = String(match.id);
        return (snapshot.qualifierWinners[key] ?? null) !== null;
      });
      if (allPredicted) {
        return null;
      }
      const applyQualifierSelection = (matchId: string, selection: WinnerSelection) => {
        const prevSelection = snapshot.qualifierWinners[matchId] ?? null;
        if (prevSelection === selection) {
          return false;
        }
        const updated = { ...snapshot.qualifierWinners, [matchId]: selection };
        const cleared = clearDependentSelections(updated, matchId, qualifierDependents);
        const clearedIds = Object.keys(updated).filter(
          (id) => updated[id] && cleared[id] === null
        );
        snapshot.qualifierWinners = cleared;
        if (selection) {
          snapshot.autoQualifierWinners[matchId] = true;
        }
        clearedIds.forEach((id) => {
          delete snapshot.autoQualifierWinners[id];
          changedMatchIds.add(id);
        });
        changedMatchIds.add(matchId);
        return true;
      };
      let iteration = 0;
      let progress = true;
      while (progress && iteration < 10) {
        iteration += 1;
        progress = false;
        const qualifierStateIter = resolveQualifierState(
          data.qualifiers,
          snapshot.qualifierWinners
        );
        qualifierStateIter.matches
          .filter((match) => match.path === path)
          .forEach((match) => {
            const key = String(match.id);
            const isManual =
              snapshot.qualifierWinners[key] && !snapshot.autoQualifierWinners[key];
            if (isManual || snapshot.qualifierWinners[key]) {
              return;
            }
            if (
              isPlaceholderLabel(match.homeResolved) ||
              isPlaceholderLabel(match.awayResolved)
            ) {
              return;
            }
            const values = resolveMatchProbabilities({
              probabilities: activeWinProbabilities,
              homeTeam: match.homeResolved,
              awayTeam: match.awayResolved,
              allowDraw: false,
              neutralOverride: match.neutral,
            });
            const sample = sampleWinnerWithProbability(values);
            if (!sample) {
              return;
            }
            if (applyQualifierSelection(key, sample.selection)) {
              snapshot.funninessScore += probabilityPenalty(sample.probability);
              progress = true;
            }
          });
      }
      if (changedMatchIds.size === 0) {
        return null;
      }
      const affectedSlots = new Set<string>();
      changedMatchIds.forEach((matchId) => {
        const slots = qualifierSlotsByMatch.get(matchId);
        slots?.forEach((slot) => affectedSlots.add(slot));
      });
      const affectedGroups = new Set<string>();
      affectedSlots.forEach((slot) => {
        groupIdsBySlot.get(slot)?.forEach((groupId) => affectedGroups.add(groupId));
      });
      affectedSlots.forEach((slot) => {
        groupMatchIdsByTeam.get(slot)?.forEach((matchId) => {
          delete snapshot.groupScores[matchId];
          delete snapshot.autoGroupScores[matchId];
        });
      });
      if (affectedGroups.size > 0) {
        affectedGroups.forEach((groupId) => {
          knockoutRootsByGroup.get(groupId)?.forEach((matchId) => {
            snapshot.knockoutWinners[matchId] = null;
            snapshot.knockoutWinners = clearDependentSelections(
              snapshot.knockoutWinners,
              matchId,
              knockoutDependents
            );
          });
        });
        Object.keys(knockoutWinners)
          .filter(
            (matchId) =>
              knockoutWinners[matchId] && snapshot.knockoutWinners[matchId] === null
          )
          .forEach((matchId) => {
            delete snapshot.autoKnockoutWinners[matchId];
          });
      }
      return snapshot;
    },
    [
      clearDependentSelections,
      createAutopredictSnapshot,
      data.qualifiers,
      activeWinProbabilities,
      groupIdsBySlot,
      groupMatchIdsByTeam,
      knockoutDependents,
      knockoutRootsByGroup,
      knockoutWinners,
      qualifierDependents,
      qualifierSlotsByMatch,
    ]
  );

  const simulateSectionQualifiersAutopredict = React.useCallback(
    (): AutopredictSnapshot | null => {
      const snapshot = createAutopredictSnapshot();
      const qualifierStateLocal = resolveQualifierState(
        data.qualifiers,
        snapshot.qualifierWinners
      );
      const allPredicted = qualifierStateLocal.matches.every((match) => {
        if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
          return true;
        }
        const key = String(match.id);
        return (snapshot.qualifierWinners[key] ?? null) !== null;
      });
      if (allPredicted) {
        return null;
      }
      const changedQualifiers = new Set<string>();
      const applyQualifierSelection = (matchId: string, selection: WinnerSelection) => {
        const prevSelection = snapshot.qualifierWinners[matchId] ?? null;
        if (prevSelection === selection) {
          return { changed: false, clearedIds: [] as string[] };
        }
        const updated = { ...snapshot.qualifierWinners, [matchId]: selection };
        const cleared = clearDependentSelections(updated, matchId, qualifierDependents);
        const clearedIds = Object.keys(updated).filter(
          (id) => updated[id] && cleared[id] === null
        );
        snapshot.qualifierWinners = cleared;
        if (selection) {
          snapshot.autoQualifierWinners[matchId] = true;
        }
        clearedIds.forEach((id) => {
          delete snapshot.autoQualifierWinners[id];
        });
        return { changed: true, clearedIds };
      };
      let changed = false;
      let iteration = 0;
      while (iteration < 10) {
        iteration += 1;
        let qualifierProgress = false;
        const qualifierStateIter = resolveQualifierState(
          data.qualifiers,
          snapshot.qualifierWinners
        );
        qualifierStateIter.matches.forEach((match) => {
          const key = String(match.id);
          const isManual =
            snapshot.qualifierWinners[key] && !snapshot.autoQualifierWinners[key];
          const existingSelection = snapshot.qualifierWinners[key] ?? null;
          if (isManual || existingSelection) {
            return;
          }
          if (
            isPlaceholderLabel(match.homeResolved) ||
            isPlaceholderLabel(match.awayResolved)
          ) {
            return;
          }
          const values = resolveMatchProbabilities({
            probabilities: activeWinProbabilities,
            homeTeam: match.homeResolved,
            awayTeam: match.awayResolved,
            allowDraw: false,
            neutralOverride: match.neutral,
          });
          const sample = sampleWinnerWithProbability(values);
          if (!sample) {
            return;
          }
          const result = applyQualifierSelection(key, sample.selection);
          if (result.changed) {
            snapshot.funninessScore += probabilityPenalty(sample.probability);
            changed = true;
            qualifierProgress = true;
            changedQualifiers.add(key);
          }
        });
        if (!qualifierProgress) {
          break;
        }
      }
      if (!changed && changedQualifiers.size === 0) {
        return null;
      }
      if (changedQualifiers.size > 0) {
        const affectedSlots = new Set<string>();
        changedQualifiers.forEach((matchId) => {
          qualifierSlotsByMatch.get(matchId)?.forEach((slot) => affectedSlots.add(slot));
        });
        const affectedGroups = new Set<string>();
        affectedSlots.forEach((slot) => {
          groupIdsBySlot.get(slot)?.forEach((groupId) => affectedGroups.add(groupId));
        });
        affectedSlots.forEach((slot) => {
          groupMatchIdsByTeam.get(slot)?.forEach((matchId) => {
            delete snapshot.groupScores[matchId];
            delete snapshot.autoGroupScores[matchId];
          });
        });
        if (affectedGroups.size > 0) {
          affectedGroups.forEach((groupId) => {
            knockoutRootsByGroup.get(groupId)?.forEach((matchId) => {
              snapshot.knockoutWinners[matchId] = null;
              snapshot.knockoutWinners = clearDependentSelections(
                snapshot.knockoutWinners,
                matchId,
                knockoutDependents
              );
            });
          });
          Object.keys(knockoutWinners)
            .filter(
              (matchId) =>
                knockoutWinners[matchId] && snapshot.knockoutWinners[matchId] === null
            )
            .forEach((matchId) => {
              delete snapshot.autoKnockoutWinners[matchId];
            });
        }
      }
      return snapshot;
    },
    [
      clearDependentSelections,
      createAutopredictSnapshot,
      data.qualifiers,
      activeWinProbabilities,
      groupIdsBySlot,
      groupMatchIdsByTeam,
      knockoutDependents,
      knockoutRootsByGroup,
      knockoutWinners,
      qualifierDependents,
      qualifierSlotsByMatch,
    ]
  );

  const simulateSectionKnockoutsAutopredict = React.useCallback(
    (): AutopredictSnapshot | null => {
      const snapshot = createAutopredictSnapshot();
      const allPredicted = knockoutState.matches.every((match) => {
        if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
          return true;
        }
        const key = String(match.id);
        return (snapshot.knockoutWinners[key] ?? null) !== null;
      });
      if (allPredicted) {
        return null;
      }
      const applyKnockoutSelection = (matchId: string, selection: WinnerSelection) => {
        const prevSelection = snapshot.knockoutWinners[matchId] ?? null;
        if (prevSelection === selection) {
          return false;
        }
        const updated = { ...snapshot.knockoutWinners, [matchId]: selection };
        const cleared = clearDependentSelections(updated, matchId, knockoutDependents);
        const clearedIds = Object.keys(updated).filter(
          (id) => updated[id] && cleared[id] === null
        );
        snapshot.knockoutWinners = cleared;
        if (selection) {
          snapshot.autoKnockoutWinners[matchId] = true;
        }
        clearedIds.forEach((id) => {
          delete snapshot.autoKnockoutWinners[id];
        });
        return true;
      };
      const context = computeKnockoutContext(groupScores, slotWinners);
      const winners = new Map<number, string>();
      const losers = new Map<number, string>();
      const sorted = [...data.knockoutMatches].sort((a, b) => a.id - b.id);
      let changed = false;
      sorted.forEach((match) => {
        const key = String(match.id);
        const homeResolved = resolveKnockoutLabel({
          label: match.homeLabel,
          opponentLabel: match.awayLabel,
          groupRankings: context.groupRankings,
          thirdPlaceByGroup: context.thirdPlaceByGroup,
          thirdPlaceAssignments: context.thirdPlaceAssignments,
          knockoutWinners: winners,
          knockoutLosers: losers,
          groupCompletion: context.groupCompletion,
          allowThirdPlaceResolve: context.allGroupMatchesComplete,
          qualifiedThirdGroups: new Set(context.qualifiedThirdGroups),
          matchStageById,
        });
        const awayResolved = resolveKnockoutLabel({
          label: match.awayLabel,
          opponentLabel: match.homeLabel,
          groupRankings: context.groupRankings,
          thirdPlaceByGroup: context.thirdPlaceByGroup,
          thirdPlaceAssignments: context.thirdPlaceAssignments,
          knockoutWinners: winners,
          knockoutLosers: losers,
          groupCompletion: context.groupCompletion,
          allowThirdPlaceResolve: context.allGroupMatchesComplete,
          qualifiedThirdGroups: new Set(context.qualifiedThirdGroups),
          matchStageById,
        });
        const existingSelection = snapshot.knockoutWinners[key] ?? null;
        const isManual = existingSelection && !snapshot.autoKnockoutWinners[key];
        if (!isManual && !existingSelection) {
          if (!isPlaceholderLabel(homeResolved) && !isPlaceholderLabel(awayResolved)) {
            const values = resolveMatchProbabilities({
              probabilities: activeWinProbabilities,
              homeTeam: homeResolved,
              awayTeam: awayResolved,
              allowDraw: false,
              country: match.country,
            });
            const sample = sampleWinnerWithProbability(values);
            if (sample && applyKnockoutSelection(key, sample.selection)) {
              snapshot.funninessScore += probabilityPenalty(
                sample.probability,
                funnyWeightForStage(matchStageById[match.id])
              );
              changed = true;
            }
          }
        }
        const winner = resolveWinner(
          match.id,
          homeResolved,
          awayResolved,
          {},
          false,
          snapshot.knockoutWinners
        );
        if (winner) {
          winners.set(match.id, winner);
          losers.set(match.id, winner === homeResolved ? awayResolved : homeResolved);
        }
      });
      return changed ? snapshot : null;
    },
    [
      clearDependentSelections,
      computeKnockoutContext,
      createAutopredictSnapshot,
      data.knockoutMatches,
      activeWinProbabilities,
      groupScores,
      knockoutDependents,
      knockoutState.matches,
      matchStageById,
      slotWinners,
    ]
  );

  const simulateTournamentAutopredict = React.useCallback((): AutopredictSnapshot | null => {
    const snapshot = createAutopredictSnapshot();
    const allQualifiersPredicted = qualifierState.matches.every((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return true;
      }
      const key = String(match.id);
      return (snapshot.qualifierWinners[key] ?? null) !== null;
    });
    const allGroupsPredicted = resolvedGroupMatches.every((match) => {
      const score = snapshot.groupScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
    const allKnockoutsPredicted = knockoutState.matches.every((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return true;
      }
      const key = String(match.id);
      return (snapshot.knockoutWinners[key] ?? null) !== null;
    });
    if (allQualifiersPredicted && allGroupsPredicted && allKnockoutsPredicted) {
      return null;
    }
    const applyQualifierSelection = (matchId: string, selection: WinnerSelection) => {
      const prevSelection = snapshot.qualifierWinners[matchId] ?? null;
      if (prevSelection === selection) {
        return { changed: false, clearedIds: [] as string[] };
      }
      const updated = { ...snapshot.qualifierWinners, [matchId]: selection };
      const cleared = clearDependentSelections(updated, matchId, qualifierDependents);
      const clearedIds = Object.keys(updated).filter(
        (id) => updated[id] && cleared[id] === null
      );
      snapshot.qualifierWinners = cleared;
      if (selection) {
        snapshot.autoQualifierWinners[matchId] = true;
      }
      clearedIds.forEach((id) => {
        delete snapshot.autoQualifierWinners[id];
      });
      return { changed: true, clearedIds };
    };
    const applyKnockoutSelection = (matchId: string, selection: WinnerSelection) => {
      const prevSelection = snapshot.knockoutWinners[matchId] ?? null;
      if (prevSelection === selection) {
        return { changed: false, clearedIds: [] as string[] };
      }
      const updated = { ...snapshot.knockoutWinners, [matchId]: selection };
      const cleared = clearDependentSelections(updated, matchId, knockoutDependents);
      const clearedIds = Object.keys(updated).filter(
        (id) => updated[id] && cleared[id] === null
      );
      snapshot.knockoutWinners = cleared;
      if (selection) {
        snapshot.autoKnockoutWinners[matchId] = true;
      }
      clearedIds.forEach((id) => {
        delete snapshot.autoKnockoutWinners[id];
      });
      return { changed: true, clearedIds };
    };
    let iteration = 0;
    let changed = true;
    let anyChanged = false;
    while (changed && iteration < 10) {
      iteration += 1;
      changed = false;
      const previousSlotWinners = resolveQualifierState(
        data.qualifiers,
        snapshot.qualifierWinners
      ).slotWinners;
      const previousGroupScores = { ...snapshot.groupScores };
      let qualifierStateLocal = resolveQualifierState(
        data.qualifiers,
        snapshot.qualifierWinners
      );
      const changedQualifiers = new Set<string>();
      let qualifierProgress = true;
      let qualifierIterations = 0;
      while (qualifierProgress && qualifierIterations < 10) {
        qualifierProgress = false;
        qualifierIterations += 1;
        qualifierStateLocal = resolveQualifierState(
          data.qualifiers,
          snapshot.qualifierWinners
        );
        qualifierStateLocal.matches.forEach((match) => {
          const key = String(match.id);
          const isManual =
            snapshot.qualifierWinners[key] && !snapshot.autoQualifierWinners[key];
          const existingSelection = snapshot.qualifierWinners[key] ?? null;
          if (isManual || existingSelection) {
            return;
          }
          if (
            isPlaceholderLabel(match.homeResolved) ||
            isPlaceholderLabel(match.awayResolved)
          ) {
            return;
          }
          const values = resolveMatchProbabilities({
            probabilities: activeWinProbabilities,
            homeTeam: match.homeResolved,
            awayTeam: match.awayResolved,
            allowDraw: false,
            neutralOverride: match.neutral,
          });
          const sample = sampleWinnerWithProbability(values);
          if (!sample) {
            return;
          }
          const result = applyQualifierSelection(key, sample.selection);
          if (result.changed) {
            snapshot.funninessScore += probabilityPenalty(sample.probability);
            changed = true;
            anyChanged = true;
            qualifierProgress = true;
            changedQualifiers.add(key);
          }
        });
      }
      if (changedQualifiers.size > 0) {
        const affectedSlots = new Set<string>();
        changedQualifiers.forEach((matchId) => {
          qualifierSlotsByMatch.get(matchId)?.forEach((slot) => affectedSlots.add(slot));
        });
        const affectedGroups = new Set<string>();
        affectedSlots.forEach((slot) => {
          groupIdsBySlot.get(slot)?.forEach((groupId) => affectedGroups.add(groupId));
        });
        affectedSlots.forEach((slot) => {
          groupMatchIdsByTeam.get(slot)?.forEach((matchId) => {
            delete snapshot.groupScores[matchId];
            delete snapshot.autoGroupScores[matchId];
          });
        });
        if (affectedGroups.size > 0) {
          const rootsToClear = new Set<string>();
          affectedGroups.forEach((groupId) => {
            knockoutRootsByGroup.get(groupId)?.forEach((matchId) => rootsToClear.add(matchId));
          });
          if (rootsToClear.size > 0) {
            const cleared = clearKnockoutSelectionsByMatchIds(
              snapshot.knockoutWinners,
              rootsToClear
            );
            snapshot.knockoutWinners = cleared.next;
            cleared.clearedIds.forEach((matchId) => {
              delete snapshot.autoKnockoutWinners[matchId];
            });
            if (cleared.clearedIds.length > 0) {
              changed = true;
            }
          }
        }
        qualifierStateLocal = resolveQualifierState(data.qualifiers, snapshot.qualifierWinners);
      }
      const nextSlotWinners = qualifierStateLocal.slotWinners;
      const resolvedGroupMatchesLocal = data.groupMatches.map((match) => ({
        ...match,
        homeTeam: nextSlotWinners.get(match.homeTeam) ?? match.homeTeam,
        awayTeam: nextSlotWinners.get(match.awayTeam) ?? match.awayTeam,
      }));
      let groupScoresChanged = false;
      resolvedGroupMatchesLocal.forEach((match) => {
        const key = String(match.id);
        const existing = snapshot.groupScores[key];
        const hasScore = existing && existing.home !== null && existing.away !== null;
        const isManual = hasScore && !snapshot.autoGroupScores[key];
        if (isManual || hasScore) {
          return;
        }
        const matrix = resolveMatchScoreMatrix({
          probabilities: activeWinProbabilities,
          homeTeam: match.homeTeam,
          awayTeam: match.awayTeam,
          country: match.country,
        });
        if (!matrix) {
          return;
        }
        const sample = sampleScoreMatrixWithProbability(matrix);
        if (!sample) {
          return;
        }
        snapshot.groupScores[key] = { home: sample.home, away: sample.away };
        snapshot.autoGroupScores[key] = true;
        snapshot.funninessScore += groupScorePenalty(matrix, sample);
        groupScoresChanged = true;
        anyChanged = true;
      });
      if (groupScoresChanged || changedQualifiers.size > 0) {
        const clearedForGroups = computeClearedKnockoutSelections(
          snapshot.knockoutWinners,
          previousGroupScores,
          snapshot.groupScores,
          {
            previousSlotWinners,
            nextSlotWinners,
          }
        );
        snapshot.knockoutWinners = clearedForGroups.nextWinners;
        clearedForGroups.clearedIds.forEach((matchId) => {
          delete snapshot.autoKnockoutWinners[matchId];
        });
        if (clearedForGroups.clearedIds.length > 0) {
          changed = true;
        }
      }
      const nextContext = computeKnockoutContext(snapshot.groupScores, nextSlotWinners);
      const winners = new Map<number, string>();
      const losers = new Map<number, string>();
      const sorted = [...data.knockoutMatches].sort((a, b) => a.id - b.id);
      for (const match of sorted) {
        const key = String(match.id);
        const homeResolved = resolveKnockoutLabel({
          label: match.homeLabel,
          opponentLabel: match.awayLabel,
          groupRankings: nextContext.groupRankings,
          thirdPlaceByGroup: nextContext.thirdPlaceByGroup,
          thirdPlaceAssignments: nextContext.thirdPlaceAssignments,
          knockoutWinners: winners,
          knockoutLosers: losers,
          groupCompletion: nextContext.groupCompletion,
          allowThirdPlaceResolve: nextContext.allGroupMatchesComplete,
          qualifiedThirdGroups: new Set(nextContext.qualifiedThirdGroups),
          matchStageById,
        });
        const awayResolved = resolveKnockoutLabel({
          label: match.awayLabel,
          opponentLabel: match.homeLabel,
          groupRankings: nextContext.groupRankings,
          thirdPlaceByGroup: nextContext.thirdPlaceByGroup,
          thirdPlaceAssignments: nextContext.thirdPlaceAssignments,
          knockoutWinners: winners,
          knockoutLosers: losers,
          groupCompletion: nextContext.groupCompletion,
          allowThirdPlaceResolve: nextContext.allGroupMatchesComplete,
          qualifiedThirdGroups: new Set(nextContext.qualifiedThirdGroups),
          matchStageById,
        });
        const existingSelection = snapshot.knockoutWinners[key] ?? null;
        const isManual = existingSelection && !snapshot.autoKnockoutWinners[key];
        if (!isManual && !existingSelection) {
          if (!isPlaceholderLabel(homeResolved) && !isPlaceholderLabel(awayResolved)) {
            const values = resolveMatchProbabilities({
              probabilities: activeWinProbabilities,
              homeTeam: homeResolved,
              awayTeam: awayResolved,
              allowDraw: false,
              country: match.country,
            });
            const sample = sampleWinnerWithProbability(values);
            if (sample) {
              const result = applyKnockoutSelection(key, sample.selection);
              if (result.changed) {
                snapshot.funninessScore += probabilityPenalty(
                  sample.probability,
                  funnyWeightForStage(matchStageById[match.id])
                );
                changed = true;
                anyChanged = true;
              }
            }
          }
        }
        const winner = resolveWinner(
          match.id,
          homeResolved,
          awayResolved,
          {},
          false,
          snapshot.knockoutWinners
        );
        if (winner) {
          winners.set(match.id, winner);
          losers.set(match.id, winner === homeResolved ? awayResolved : homeResolved);
        }
      }
    }
    return anyChanged ? snapshot : null;
  }, [
    clearDependentSelections,
    clearKnockoutSelectionsByMatchIds,
    computeClearedKnockoutSelections,
    computeKnockoutContext,
    createAutopredictSnapshot,
    data.groupMatches,
    data.knockoutMatches,
    data.qualifiers,
    activeWinProbabilities,
    groupIdsBySlot,
    groupMatchIdsByTeam,
    knockoutDependents,
    knockoutRootsByGroup,
    knockoutState.matches,
    matchStageById,
    qualifierDependents,
    qualifierSlotsByMatch,
    qualifierState.matches,
    resolvedGroupMatches,
  ]);

  const updateGroupScore = React.useCallback(
    (id: string | number, side: "home" | "away", value: number | null) => {
      const key = String(id);
      if (lockResultsActive && lockedGroupMatchIds.has(key)) {
        return;
      }
      let changed = false;
      let nextScores: Record<string, MatchScore> | null = null;
      setGroupScores((prev) => {
        const prevScore = prev[key] ?? { home: null, away: null };
        const nextScore = { ...prevScore, [side]: value };
        if (
          prevScore.home === nextScore.home &&
          prevScore.away === nextScore.away
        ) {
          return prev;
        }
        changed = true;
        nextScores = { ...prev, [key]: nextScore };
        return nextScores;
      });
      if (changed) {
        setShowGroupHint(false);
        setAutoGroupScores((prev) => {
          if (!prev[key]) {
            return prev;
          }
          const next = { ...prev };
          delete next[key];
          return next;
        });
        if (nextScores) {
          clearKnockoutOnGroupChange(nextScores);
        }
      }
    },
    [clearKnockoutOnGroupChange, lockResultsActive, lockedGroupMatchIds]
  );

  const updateGroupScorePair = React.useCallback(
    (id: string | number, home: number | null, away: number | null) => {
      const key = String(id);
      if (lockResultsActive && lockedGroupMatchIds.has(key)) {
        return;
      }
      let changed = false;
      let nextScores: Record<string, MatchScore> | null = null;
      setGroupScores((prev) => {
        const prevScore = prev[key] ?? { home: null, away: null };
        const nextScore = { home, away };
        if (
          prevScore.home === nextScore.home &&
          prevScore.away === nextScore.away
        ) {
          return prev;
        }
        changed = true;
        nextScores = { ...prev, [key]: nextScore };
        return nextScores;
      });
      if (changed && nextScores) {
        setShowGroupHint(false);
        setAutoGroupScores((prev) => {
          if (!prev[key]) {
            return prev;
          }
          const next = { ...prev };
          delete next[key];
          return next;
        });
        clearKnockoutOnGroupChange(nextScores);
      }
    },
    [clearKnockoutOnGroupChange, lockResultsActive, lockedGroupMatchIds]
  );

  const resolvedGroups = React.useMemo(() => {
    return data.groups.map((group) => ({
      ...group,
      teams: group.teams.map((team) => slotWinners.get(team) ?? team),
    }));
  }, [data.groups, slotWinners]);

  const groupsWithUnresolvedParticipants = React.useMemo(() => {
    const unresolved = new Set<string>();
    resolvedGroups.forEach((group) => {
      if (group.teams.some((team) => !isConcreteTeam(team))) {
        unresolved.add(group.id);
      }
    });
    return unresolved;
  }, [resolvedGroups]);

  const groupQualifierPaths = React.useMemo(() => {
    const slotToPath = new Map<string, string>();
    data.qualifiers.forEach((match) => {
      if (match.winnerSlot && match.path) {
        slotToPath.set(match.winnerSlot, match.path);
      }
    });
    const groupPaths = new Map<string, Set<string>>();
    data.groups.forEach((group) => {
      group.teams.forEach((slot) => {
        if (slotWinners.get(slot)) {
          return;
        }
        const path = slotToPath.get(slot);
        if (!path) {
          return;
        }
        if (!groupPaths.has(group.id)) {
          groupPaths.set(group.id, new Set());
        }
        groupPaths.get(group.id)?.add(path);
      });
    });
    const normalized = new Map<string, string[]>();
    groupPaths.forEach((paths, groupId) => {
      normalized.set(groupId, Array.from(paths));
    });
    return normalized;
  }, [data.groups, data.qualifiers, slotWinners]);

  resolvedGroupMatches = React.useMemo(() => {
    return data.groupMatches.map((match) => ({
      ...match,
      homeTeam: slotWinners.get(match.homeTeam) ?? match.homeTeam,
      awayTeam: slotWinners.get(match.awayTeam) ?? match.awayTeam,
    }));
  }, [data.groupMatches, slotWinners]);

  const hasUnpredictedGroups = React.useCallback(() => {
    return resolvedGroupMatches.some((match) => {
      const score = groupScores[String(match.id)];
      return !score || score.home === null || score.away === null;
    });
  }, [groupScores, resolvedGroupMatches]);

  const hasAnyQualifierPredictions = React.useMemo(() => {
    return qualifierState.matches.some((match) => {
      const key = String(match.id);
      return (qualifierWinners[key] ?? null) !== null;
    });
  }, [qualifierState.matches, qualifierWinners]);

  const hasAnyGroupPredictions = React.useMemo(() => {
    return resolvedGroupMatches.some((match) => {
      if (lockResultsActive && lockedGroupMatchIds.has(String(match.id))) {
        return false;
      }
      const score = groupScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
  }, [groupScores, lockResultsActive, lockedGroupMatchIds, resolvedGroupMatches]);

  const qualifierPathPredictionStatus = React.useMemo(() => {
    const map = new Map<string, { hasUnpredicted: boolean; hasPredicted: boolean }>();
    qualifierState.matches.forEach((match) => {
      if (!match.path) {
        return;
      }
      const key = String(match.id);
      const selection = qualifierWinners[key] ?? null;
      const isPredicted = selection !== null;
      const isConcrete =
        isConcreteTeam(match.homeResolved) && isConcreteTeam(match.awayResolved);
      const existing = map.get(match.path) ?? {
        hasUnpredicted: false,
        hasPredicted: false,
      };
      if (isPredicted) {
        existing.hasPredicted = true;
      }
      if (isConcrete && !isPredicted) {
        existing.hasUnpredicted = true;
      }
      map.set(match.path, existing);
    });
    return map;
  }, [qualifierState.matches, qualifierWinners]);

  const groupTables = React.useMemo(() => {
    return resolvedGroups.map((group) => {
      const matches = groupMatchesFor(group.id, resolvedGroupMatches);
      const { table, ranking } = buildGroupTable(group, matches, groupScores);
      const rows = ranking.map((team) => table[team]).filter(Boolean);
      return { group, ranking, table, rows };
    });
  }, [resolvedGroups, resolvedGroupMatches, groupScores]);

  const groupsWithCtaMatches = React.useMemo(() => {
    const groups = new Set<string>();
    resolvedGroups.forEach((group) => {
      const matches = groupMatchesFor(group.id, resolvedGroupMatches);
      const hasCta = matches.some((match) => {
        if (!isConcreteTeam(match.homeTeam) || !isConcreteTeam(match.awayTeam)) {
          return false;
        }
        const score = groupScores[String(match.id)];
        return !score || score.home === null || score.away === null;
      });
      if (hasCta) {
        groups.add(group.id);
      }
    });
    return groups;
  }, [resolvedGroups, resolvedGroupMatches, groupScores]);

  const groupCompletion = React.useMemo(() => {
    const completion: Record<string, boolean> = {};
    data.groups.forEach((group) => {
      const matches = groupMatchesFor(group.id, resolvedGroupMatches);
      completion[group.id] = matches.every((match) => {
        const score = groupScores[String(match.id)];
        return score && score.home !== null && score.away !== null;
      });
    });
    return completion;
  }, [data.groups, resolvedGroupMatches, groupScores]);


  const groupRankings = React.useMemo(() => {
    const rankings: Record<string, string[]> = {};
    for (const entry of groupTables) {
      rankings[entry.group.id] = entry.ranking;
    }
    return rankings;
  }, [groupTables]);

  const thirdPlaceResults = React.useMemo(
    () => bestThirdPlace(groupTables),
    [groupTables]
  );
  const thirdPlaceEntries = thirdPlaceResults.entries;
  const thirdPlaceCutoffTies = React.useMemo(() => {
    const cutoffIndex = 7;
    const groups = new Map<
      string,
      { teams: string[]; indices: number[] }
    >();
    thirdPlaceEntries.forEach((entry, index) => {
      const key = `${entry.points}|${entry.gd}|${entry.gf}`;
      const group = groups.get(key) ?? { teams: [], indices: [] };
      group.teams.push(entry.team);
      group.indices.push(index);
      groups.set(key, group);
    });
    const tiedTeams = new Set<string>();
    groups.forEach((group) => {
      if (group.indices.length <= 1) {
        return;
      }
      const minIndex = Math.min(...group.indices);
      const maxIndex = Math.max(...group.indices);
      if (minIndex <= cutoffIndex && maxIndex > cutoffIndex) {
        group.teams.forEach((team) => tiedTeams.add(team));
      }
    });
    return tiedTeams;
  }, [thirdPlaceEntries]);
  const thirdPlaceRankingRows = React.useMemo(() => {
    const rowByTeam = new Map<string, GroupTableRow>();
    groupTables.forEach((entry) => {
      Object.values(entry.table).forEach((row) => {
        rowByTeam.set(row.team, row);
      });
    });
    return thirdPlaceEntries
      .map((entry, index) => {
        const row = rowByTeam.get(entry.team);
        if (!row) {
          return null;
        }
        return {
          ...row,
          position: index + 1,
          randomTiebreak: thirdPlaceCutoffTies.has(entry.team),
        };
      })
      .filter(
        (row): row is GroupTableRow & { randomTiebreak: boolean } => Boolean(row)
      );
  }, [groupTables, thirdPlaceEntries, thirdPlaceCutoffTies]);
  const bestThirdGroups = thirdPlaceEntries.slice(0, 8);
  const qualifiedThirdGroups = React.useMemo(
    () => new Set(bestThirdGroups.map((entry) => entry.group)),
    [bestThirdGroups]
  );
  const thirdPlaceByGroup = React.useMemo(() => {
    const mapping: Record<string, string> = {};
    for (const entry of thirdPlaceEntries) {
      if (!mapping[entry.group]) {
        mapping[entry.group] = entry.team;
      }
    }
    return mapping;
  }, [thirdPlaceEntries]);

  const allGroupMatchesComplete = React.useMemo(() => {
    return resolvedGroupMatches.every((match) => {
      const score = groupScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
  }, [resolvedGroupMatches, groupScores]);

  const thirdPlaceAssignments = React.useMemo(() => {
    const groups = bestThirdGroups.map((entry) => entry.group).sort();
    const comboKey = groups.join("");
    if (!comboKey) {
      return null;
    }
    return data.roundOf32Combos[comboKey] ?? null;
  }, [bestThirdGroups, data.roundOf32Combos]);

  const logRoundOf32Match = React.useCallback(
    (_match: ResolvedKnockoutMatch) => {},
    []
  );

  knockoutState = React.useMemo(() => {
    const winners = new Map<number, string>();
    const losers = new Map<number, string>();
    const resolvedMatches: ResolvedKnockoutMatch[] = [];
    const sorted = [...data.knockoutMatches].sort((a, b) => a.id - b.id);

    for (const match of sorted) {
      const homeResolved = resolveKnockoutLabel({
        label: match.homeLabel,
        opponentLabel: match.awayLabel,
        groupRankings,
        thirdPlaceByGroup,
        thirdPlaceAssignments,
        knockoutWinners: winners,
        knockoutLosers: losers,
        groupCompletion,
        allowThirdPlaceResolve: allGroupMatchesComplete,
        qualifiedThirdGroups,
        matchStageById,
      });
      const awayResolved = resolveKnockoutLabel({
        label: match.awayLabel,
        opponentLabel: match.homeLabel,
        groupRankings,
        thirdPlaceByGroup,
        thirdPlaceAssignments,
        knockoutWinners: winners,
        knockoutLosers: losers,
        groupCompletion,
        allowThirdPlaceResolve: allGroupMatchesComplete,
        qualifiedThirdGroups,
        matchStageById,
      });
      const isPickableMatch =
        isConcreteTeam(homeResolved) && isConcreteTeam(awayResolved);
      const winner = isPickableMatch
        ? resolveWinner(
            match.id,
            homeResolved,
            awayResolved,
            {},
            false,
            knockoutWinners
          )
        : undefined;
      if (winner) {
        winners.set(match.id, winner);
        const loser = winner === homeResolved ? awayResolved : homeResolved;
        losers.set(match.id, loser);
      }
      resolvedMatches.push({
        ...match,
        homeResolved,
        awayResolved,
        winner,
      });
    }
    return { winners, losers, matches: resolvedMatches };
  }, [
    data.knockoutMatches,
    groupRankings,
    thirdPlaceByGroup,
    thirdPlaceAssignments,
    knockoutWinners,
    allGroupMatchesComplete,
    matchStageById,
  ]);

  const unpickableKnockoutIds = React.useMemo(() => {
    return knockoutState.matches
      .filter(
        (match) =>
          !isConcreteTeam(match.homeResolved) ||
          !isConcreteTeam(match.awayResolved)
      )
      .map((match) => String(match.id));
  }, [knockoutState.matches]);

  const hasUnpredictedKnockouts = React.useCallback(() => {
    return knockoutState.matches.some((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return false;
      }
      const key = String(match.id);
      return (knockoutWinners[key] ?? null) === null;
    });
  }, [knockoutState.matches, knockoutWinners]);

  const hasAnyKnockoutPredictions = React.useMemo(() => {
    return knockoutState.matches.some((match) => {
      const key = String(match.id);
      if (lockResultsActive && lockedKnockoutMatchIds.has(key)) {
        return false;
      }
      return (knockoutWinners[key] ?? null) !== null;
    });
  }, [knockoutState.matches, knockoutWinners, lockResultsActive, lockedKnockoutMatchIds]);

  React.useEffect(() => {
    if (unpickableKnockoutIds.length === 0) {
      return;
    }
    setKnockoutWinners((prev) => {
      const { next, clearedIds } = clearKnockoutSelectionsByMatchIds(
        prev,
        unpickableKnockoutIds
      );
      if (clearedIds.length === 0) {
        return prev;
      }
      setAutoKnockoutWinners((currentAuto) => {
        if (!Object.keys(currentAuto).length) {
          return currentAuto;
        }
        const nextAuto = { ...currentAuto };
        clearedIds.forEach((matchId) => {
          delete nextAuto[matchId];
        });
        return nextAuto;
      });
      return next;
    });
  }, [clearKnockoutSelectionsByMatchIds, unpickableKnockoutIds]);

  React.useEffect(() => {
    if (process.env.NODE_ENV === "production") {
      return;
    }
    const context = computeKnockoutContext(groupScores);
    const engineRoundOf32 = new Map<string, { home: string; away: string }>();
    data.knockoutMatches.forEach((match) => {
      if (match.stage !== "Round of 32") {
        return;
      }
      const label = context.labels.get(String(match.id));
      if (!label) {
        return;
      }
      engineRoundOf32.set(String(match.id), label);
    });
    const uiRoundOf32 = new Map<string, { home: string; away: string }>();
    knockoutState.matches.forEach((match) => {
      if (match.stage !== "Round of 32") {
        return;
      }
      uiRoundOf32.set(String(match.id), {
        home: match.homeResolved ?? match.homeLabel,
        away: match.awayResolved ?? match.awayLabel,
      });
    });
    const mismatches: Array<{
      matchId: string;
      engine: { home: string; away: string };
      ui: { home: string; away: string } | null;
    }> = [];
    engineRoundOf32.forEach((engine, matchId) => {
      const ui = uiRoundOf32.get(matchId);
      if (!ui) {
        mismatches.push({ matchId, engine, ui: null });
        return;
      }
      if (engine.home !== ui.home || engine.away !== ui.away) {
        mismatches.push({ matchId, engine, ui });
      }
    });
    void mismatches;
  }, [
    computeKnockoutContext,
    data.knockoutMatches,
    groupScores,
    knockoutState.matches,
  ]);

  const knockoutMatchesByStage = React.useMemo(
    () => matchesByStage(knockoutState.matches),
    [knockoutState.matches]
  );

  const isTournamentComplete = React.useMemo(() => {
    const qualifiersDone = qualifierState.matches.every((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return true;
      }
      const key = String(match.id);
      return (qualifierWinners[key] ?? null) !== null;
    });
    const groupsDone = resolvedGroupMatches.every((match) => {
      const score = groupScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
    const knockoutsDone = knockoutState.matches.every((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return true;
      }
      const key = String(match.id);
      return (knockoutWinners[key] ?? null) !== null;
    });
    return qualifiersDone && groupsDone && knockoutsDone;
  }, [
    groupScores,
    knockoutState.matches,
    knockoutWinners,
    qualifierState.matches,
    qualifierWinners,
    resolvedGroupMatches,
  ]);

  const shareLink = React.useMemo(() => {
    if (typeof window === "undefined") {
      return "";
    }
    const token = encodeShareStateCompact({
      qualifiers: data.qualifiers,
      groupMatches: data.groupMatches,
      knockouts: data.knockoutMatches,
      qualifierWinners,
      groupScores,
      knockoutWinners,
    });
    return `${window.location.origin}${window.location.pathname}?p=${token}`;
  }, [
    data.groupMatches,
    data.knockoutMatches,
    data.qualifiers,
    groupScores,
    knockoutWinners,
    qualifierWinners,
  ]);

  const isKnockoutBracketReady = React.useMemo(() => {
    const roundOf32 = knockoutMatchesByStage.get("Round of 32") ?? [];
    if (roundOf32.length === 0) {
      return true;
    }
    return roundOf32.every(
      (match) =>
        isConcreteTeam(match.homeResolved) && isConcreteTeam(match.awayResolved)
    );
  }, [knockoutMatchesByStage]);

  const compactTight =
    compactKnockout &&
    knockoutContainerWidth !== null &&
    knockoutContainerWidth < 420;

  React.useLayoutEffect(() => {
    if (!showKnockoutContent) {
      setKnockoutContainerWidth(null);
      return;
    }
    const container = knockoutContainerRef.current;
    if (!container) {
      return;
    }
    const update = () => {
      const width = container.getBoundingClientRect().width;
      setKnockoutContainerWidth((prev) => (prev === width ? prev : width));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(container);
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("resize", update);
      observer.disconnect();
    };
  }, [showKnockoutContent]);

  React.useEffect(() => {
    if (!pendingSharedKnockouts.current || !isKnockoutBracketReady) {
      return;
    }
    setKnockoutWinners(pendingSharedKnockouts.current);
    setAutoKnockoutWinners({});
    pendingSharedKnockouts.current = null;
  }, [isKnockoutBracketReady]);

  const stageOrder = [
    "Round of 32",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Final",
  ];

  const thirdPlaceMatches = knockoutMatchesByStage.get("Third place") ?? [];

  const roundOf32Order = React.useMemo(() => {
    const matchById = new Map<number, ResolvedKnockoutMatch>();
    for (const matches of knockoutMatchesByStage.values()) {
      for (const match of matches) {
        matchById.set(match.id, match);
      }
    }

    const extractSource = (label: string) => {
      if (label.startsWith("Winner Match ")) {
        return Number(label.replace("Winner Match ", "").trim());
      }
      if (label.startsWith("Loser Match ")) {
        return Number(label.replace("Loser Match ", "").trim());
      }
      return null;
    };

    const stageMatches = (stage: string) =>
      knockoutMatchesByStage.get(stage) ?? [];

    const orderFromParent = (
      parents: ResolvedKnockoutMatch[],
      childStage: string
    ) => {
      const children = stageMatches(childStage);
      const childIds = new Set(children.map((match) => match.id));
      const used = new Set<number>();
      const order: number[] = [];
      for (const parent of parents) {
        const sources = [
          extractSource(parent.homeLabel),
          extractSource(parent.awayLabel),
        ];
        for (const source of sources) {
          if (source && childIds.has(source) && !used.has(source)) {
            order.push(source);
            used.add(source);
          }
        }
      }
      const remaining = children
        .map((match) => match.id)
        .filter((id) => !used.has(id))
        .sort((a, b) => a - b);
      return [...order, ...remaining];
    };

    const finalMatches = [...stageMatches("Final")].sort((a, b) => a.id - b.id);
    const semifinalOrder = orderFromParent(finalMatches, "Semifinal");
    const semifinalMatches = semifinalOrder
      .map((id) => matchById.get(id))
      .filter(Boolean) as ResolvedKnockoutMatch[];
    const quarterfinalOrder = orderFromParent(semifinalMatches, "Quarterfinal");
    const quarterfinalMatches = quarterfinalOrder
      .map((id) => matchById.get(id))
      .filter(Boolean) as ResolvedKnockoutMatch[];
    const round16Order = orderFromParent(quarterfinalMatches, "Round of 16");
    const round16Matches = round16Order
      .map((id) => matchById.get(id))
      .filter(Boolean) as ResolvedKnockoutMatch[];
    const round32Order = orderFromParent(round16Matches, "Round of 32");

    if (round32Order.length === 0) {
      return stageMatches("Round of 32")
        .map((match) => match.id)
        .sort((a, b) => a - b);
    }
    return round32Order;
  }, [knockoutMatchesByStage]);

  // Split matches into top and bottom halves for each stage
  // Matches are assigned based on which Round of 32 matches they descend from
  const splitMatchesByStage = React.useMemo(() => {
    const split: Record<string, { top: ResolvedKnockoutMatch[]; bottom: ResolvedKnockoutMatch[] }> = {};
    
    // Build lookup map from original match data for tracing ancestry
    const matchById = new Map<number, KnockoutMatch>();
    for (const match of data.knockoutMatches) {
      matchById.set(match.id, match);
    }
    
    // Split Round of 32 matches into top and bottom halves
    const round32Matches = knockoutMatchesByStage.get("Round of 32") ?? [];
    const orderedRound32 = roundOf32Order
      .map((id) => round32Matches.find((m) => m.id === id))
      .filter(Boolean) as ResolvedKnockoutMatch[];
    const midPoint = Math.ceil(orderedRound32.length / 2);
    const topRound32Ids = new Set(orderedRound32.slice(0, midPoint).map(m => m.id));
    const bottomRound32Ids = new Set(orderedRound32.slice(midPoint).map(m => m.id));
    
    split["Round of 32"] = {
      top: orderedRound32.slice(0, midPoint),
      bottom: orderedRound32.slice(midPoint),
    };
    
    // Helper function to extract source match ID from a label
    const extractSource = (label: string): number | null => {
      if (label.startsWith("Winner Match ")) {
        const id = Number(label.replace("Winner Match ", "").trim());
        return Number.isFinite(id) ? id : null;
      }
      if (label.startsWith("Loser Match ")) {
        const id = Number(label.replace("Loser Match ", "").trim());
        return Number.isFinite(id) ? id : null;
      }
      return null;
    };
    
    // Helper function to find all Round of 32 ancestors of a match
    const findRound32Ancestors = (matchId: number, visited: Set<number> = new Set()): Set<number> => {
      if (visited.has(matchId)) {
        return new Set();
      }
      visited.add(matchId);
      
      const match = matchById.get(matchId);
      if (!match) {
        return new Set();
      }
      
      // If this is a Round of 32 match, return itself
      if (match.stage === "Round of 32") {
        return new Set([matchId]);
      }
      
      // Otherwise, trace back through its sources
      const sources = [
        extractSource(match.homeLabel),
        extractSource(match.awayLabel),
      ].filter((id): id is number => id !== null);
      
      const ancestors = new Set<number>();
      for (const sourceId of sources) {
        const sourceAncestors = findRound32Ancestors(sourceId, visited);
        for (const ancestor of sourceAncestors) {
          ancestors.add(ancestor);
        }
      }
      return ancestors;
    };
    
    // For each subsequent stage, assign matches to top/bottom based on their Round of 32 ancestry
    for (const stage of ["Round of 16", "Quarterfinal", "Semifinal"]) {
      const matches = knockoutMatchesByStage.get(stage) ?? [];
      const topMatches: ResolvedKnockoutMatch[] = [];
      const bottomMatches: ResolvedKnockoutMatch[] = [];
      
      for (const match of matches) {
        const ancestors = findRound32Ancestors(match.id);
        const hasTopAncestor = Array.from(ancestors).some(id => topRound32Ids.has(id));
        const hasBottomAncestor = Array.from(ancestors).some(id => bottomRound32Ids.has(id));
        
        // If match has any top ancestor, it goes to top half
        // If it only has bottom ancestors, it goes to bottom half
        if (hasTopAncestor) {
          topMatches.push(match);
        } else if (hasBottomAncestor) {
          bottomMatches.push(match);
        } else {
          // Fallback: if we can't determine, assign based on match ID
          // (shouldn't happen in a well-formed bracket, but handle gracefully)
          if (match.id % 2 === 0) {
            topMatches.push(match);
          } else {
            bottomMatches.push(match);
          }
        }
      }
      
      split[stage] = { top: topMatches, bottom: bottomMatches };
    }
    
    return split;
  }, [knockoutMatchesByStage, roundOf32Order]);

  const knockoutEdges = React.useMemo(() => {
    const edges: Array<{ from: number; to: number }> = [];
    for (const match of data.knockoutMatches) {
      const labels = [match.homeLabel, match.awayLabel];
      for (const label of labels) {
        if (label.startsWith("Winner Match ")) {
          const from = Number(label.replace("Winner Match ", "").trim());
          if (Number.isFinite(from)) {
            edges.push({ from, to: match.id });
          }
        } else if (label.startsWith("Loser Match ")) {
          const from = Number(label.replace("Loser Match ", "").trim());
          if (Number.isFinite(from)) {
            edges.push({ from, to: match.id });
          }
        }
      }
    }
    return edges;
  }, [data.knockoutMatches]);

  React.useEffect(() => {
    if (!showKnockoutContent) {
      return;
    }
    const container = knockoutContainerRef.current;
    if (!container) {
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const rect = container.getBoundingClientRect();
        const connectorInset = 8;
        const paths: string[] = [];
        for (const edge of knockoutEdges) {
          const fromEl = knockoutRefs.current.get(edge.from);
          const toEl = knockoutRefs.current.get(edge.to);
          if (!fromEl || !toEl) {
            continue;
          }
          const fromRect = fromEl.getBoundingClientRect();
          const toRect = toEl.getBoundingClientRect();
          const fromStage = matchStageById[edge.from];
          const toStage = matchStageById[edge.to];
          
          // Determine if matches are on left or right side
          const fromMatches = knockoutMatchesByStage.get(fromStage) ?? [];
          const toMatches = knockoutMatchesByStage.get(toStage) ?? [];
          const fromMatch = fromMatches.find(m => m.id === edge.from);
          const toMatch = toMatches.find(m => m.id === edge.to);
          
          // Check if matches are in top (left) or bottom (right) half
          const fromIsTop = fromMatch && splitMatchesByStage[fromStage]?.top.some(m => m.id === edge.from);
          const toIsTop = toMatch && splitMatchesByStage[toStage]?.top.some(m => m.id === edge.to);
          const fromIsRight = !fromIsTop && fromStage !== "Final";
          const toIsRight = !toIsTop && toStage !== "Final";
          
          const startY = fromRect.top - rect.top + fromRect.height / 2;
          const endY = toRect.top - rect.top + toRect.height / 2;
          
          let path: string;
          const isRound32ToRound16 = fromStage === "Round of 32" && toStage === "Round of 16";
          const isRound16ToQuarter = fromStage === "Round of 16" && toStage === "Quarterfinal";
          const isQuarterToSemi = fromStage === "Quarterfinal" && toStage === "Semifinal";
          const isSemiToFinal = fromStage === "Semifinal" && (toStage === "Final" || toStage === "Third place");
          // In compact mode, tighten connector turn distance only when the layout is tight
          const horizontalDistance = compactKnockout ? (compactTight ? 12 : 20) : 30;
          // For R32→R16 in compact mode, use a fixed distance for symmetry
          // In non-compact mode, use midpoint (which worked before)
          const r32ToR16TurnDistance = compactKnockout ? 8 : undefined;
          
          if (fromIsRight) {
            // Right side: exit from LEFT edge, enter RIGHT edge of destination
            if (isRound32ToRound16) {
              // R32 → R16: match left-side inset for symmetry
              const startX = fromRect.left - rect.left + connectorInset;
              const endX = toRect.right - rect.left - connectorInset;
              // Only draw if we have valid coordinates (both elements are found and positioned)
              if (isFinite(startX) && isFinite(endX) && isFinite(startY) && isFinite(endY)) {
                if (compactKnockout) {
                  // R32 → R16: Use fixed distance from edge (goes left, so subtract)
                  const turnX = startX - r32ToR16TurnDistance!;
                  path = `M ${startX} ${startY} L ${turnX} ${startY} L ${turnX} ${endY} L ${endX} ${endY}`;
                } else {
                  // R32 → R16: Use midpoint calculation (non-compact mode)
                  const midX = startX + (endX - startX) * 0.5;
                  path = `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`;
                }
              } else {
                // Skip invalid path (coordinates are NaN or Infinity)
                continue;
              }
            } else {
              const startX = fromRect.left - rect.left + connectorInset;
              if (isRound16ToQuarter) {
                // R16 → Quarters: Exit from left, go short distance, turn right angle, enter right side
                const endX = toRect.right - rect.left - connectorInset;
                const turnX = startX - horizontalDistance;
                path = `M ${startX} ${startY} L ${turnX} ${startY} L ${turnX} ${endY} L ${endX} ${endY}`;
              } else if (isQuarterToSemi) {
                // Quarters → Semis: Exit from LEFT edge, go short distance left, then turn to enter SF from RIGHT
                const endX = toRect.right - rect.left - connectorInset;
                const turnX = startX - horizontalDistance;
                path = `M ${startX} ${startY} L ${turnX} ${startY} L ${turnX} ${endY} L ${endX} ${endY}`;
              } else if (isSemiToFinal) {
                // SF → Final/Third: Exit from LEFT, use fixed turn point for consistency
                const endX = toRect.right - rect.left - connectorInset;
                const turnX = startX - horizontalDistance;
                path = `M ${startX} ${startY} L ${turnX} ${startY} L ${turnX} ${endY} L ${endX} ${endY}`;
              } else {
                // Standard right-side connection
                const endX = toRect.right - rect.left - connectorInset;
                const midX = startX + (endX - startX) * 0.5;
                path = `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`;
              }
            }
          } else {
            // Left side or center: exit from RIGHT edge, enter LEFT edge
            if (isRound32ToRound16) {
              // R32 → R16: Use same approach as R16→QF for consistency (mirrors right side)
              const startX = fromRect.right - rect.left - connectorInset;
              const endX = toRect.left - rect.left + connectorInset;
              if (compactKnockout) {
                // R32 → R16: Use same fixed distance from edge (goes right, so add)
                const turnX = startX + r32ToR16TurnDistance!;
                path = `M ${startX} ${startY} L ${turnX} ${startY} L ${turnX} ${endY} L ${endX} ${endY}`;
              } else {
                // R32 → R16: Use midpoint calculation (non-compact mode)
                const midX = startX + (endX - startX) * 0.5;
                path = `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`;
              }
            } else {
              const startX = fromRect.right - rect.left - connectorInset;
              if (isRound16ToQuarter) {
                // R16 → Quarters: Exit from right, go short distance, turn right angle, enter left side
                const endX = toRect.left - rect.left + connectorInset;
                const turnX = startX + horizontalDistance;
                path = `M ${startX} ${startY} L ${turnX} ${startY} L ${turnX} ${endY} L ${endX} ${endY}`;
              } else if (isQuarterToSemi) {
                // Quarters → Semis: Exit from RIGHT edge, go short distance right, then turn to enter SF from LEFT
                const endX = toRect.left - rect.left + connectorInset;
                const turnX = startX + horizontalDistance;
                path = `M ${startX} ${startY} L ${turnX} ${startY} L ${turnX} ${endY} L ${endX} ${endY}`;
              } else if (isSemiToFinal) {
                // SF → Final/Third: Exit from RIGHT, use fixed turn point for consistency
                const endX = toRect.left - rect.left + connectorInset;
                const turnX = startX + horizontalDistance;
                path = `M ${startX} ${startY} L ${turnX} ${startY} L ${turnX} ${endY} L ${endX} ${endY}`;
              } else {
                // Standard left-side connection
                const endX = toRect.left - rect.left + connectorInset;
                const midX = startX + (endX - startX) * 0.5;
                path = `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`;
              }
            }
          }
          paths.push(path);
        }
        setKnockoutPaths(paths);
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(container);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [
    knockoutEdges,
    thirdPlaceOffset,
    finalCenterOverride,
    knockoutListHeight,
    matchStageById,
    knockoutMatchesByStage,
    splitMatchesByStage,
    compactKnockout,
    showKnockoutContent,
    compactTight,
  ]);

  React.useLayoutEffect(() => {
    if (!showKnockoutContent) {
      return;
    }
    const list = roundOf32ListRef.current;
    const container = knockoutContainerRef.current;
    if (!list || !container) {
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const listRect = list.getBoundingClientRect();
        const centers = new Map<number, number>();
        const round32Matches = roundOf32Order
          .map((id) =>
            (knockoutMatchesByStage.get("Round of 32") ?? []).find(
              (match) => match.id === id
            )
          )
          .filter(Boolean) as ResolvedKnockoutMatch[];
        for (const match of round32Matches) {
          const el = knockoutRefs.current.get(match.id);
          if (!el) {
            continue;
          }
          const rect = el.getBoundingClientRect();
          centers.set(
            match.id,
            rect.top - listRect.top + rect.height / 2
          );
          setKnockoutCardHeight((prev) =>
            prev && Math.abs(prev - rect.height) < 0.5 ? prev : rect.height
          );
        }
        const computed = new Map(centers);
        const extractSource = (label: string) => {
          if (label.startsWith("Winner Match ")) {
            return Number(label.replace("Winner Match ", "").trim());
          }
          if (label.startsWith("Loser Match ")) {
            return Number(label.replace("Loser Match ", "").trim());
          }
          return null;
        };
        const stageSequence = [
          "Round of 16",
          "Quarterfinal",
          "Semifinal",
          "Final",
          "Third place",
        ];
        for (const stage of stageSequence) {
          const matches = knockoutMatchesByStage.get(stage) ?? [];
          for (const match of matches) {
            const sources = [
              extractSource(match.homeLabel),
              extractSource(match.awayLabel),
            ].filter((id): id is number => Boolean(id));
            const sourceCenters = sources
              .map((id) => computed.get(id))
              .filter((value): value is number => typeof value === "number");
            if (sourceCenters.length === 0) {
              continue;
            }
            const avg =
              sourceCenters.reduce((sum, value) => sum + value, 0) /
              sourceCenters.length;
            computed.set(match.id, avg);
          }
        }
        const height = list.scrollHeight;
        if (height > 0) {
          setKnockoutListHeight((prev) => (prev === height ? prev : height));
        }
        const nextCenters: Record<number, number> = {};
        for (const [id, center] of computed.entries()) {
          nextCenters[id] = center;
        }
        setKnockoutCenters(nextCenters);
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(list);
    observer.observe(container);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [
    knockoutMatchesByStage,
    roundOf32Order,
    knockoutCardHeight,
    compactKnockout,
    showKnockoutContent,
    compactTight,
  ]);



  React.useLayoutEffect(() => {
    if (!showKnockoutContent) {
      setThirdPlaceOffset(null);
      return;
    }
    const finalMatch = (knockoutMatchesByStage.get("Final") ?? [])[0];
    const finalList = finalListRef.current;
    if (!finalMatch || !finalList) {
      setThirdPlaceOffset(null);
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const finalEl = knockoutRefs.current.get(finalMatch.id);
        if (!finalEl) {
          return;
        }
        const listRect = finalList.getBoundingClientRect();
        const containerRect = knockoutContainerRef.current?.getBoundingClientRect();
        if (!containerRect) {
          return;
        }
        const semifinalMatches = knockoutMatchesByStage.get("Semifinal") ?? [];
        const centers = semifinalMatches
          .map((match) => {
            const el = knockoutRefs.current.get(match.id);
            if (!el) {
              return null;
            }
            const rect = el.getBoundingClientRect();
            return rect.top - containerRect.top + rect.height / 2;
          })
          .filter((value): value is number => typeof value === "number");
        if (centers.length === 0) {
          return;
        }
        const semisAvg = centers.reduce((sum, value) => sum + value, 0) / centers.length;
        const listOffset = listRect.top - containerRect.top;
        // Position third place lower than semis, with same offset as final is above (symmetric)
        const finalOffset = isSmallScreen ? 56 : 72; // Distance above/below semis average
        // In compact mode, move third place up by one match height
        const compactAdjustment = compactKnockout ? (knockoutCardHeight ?? 64) : 0;
        const nextTop = semisAvg - listOffset + finalOffset - compactAdjustment;
        setThirdPlaceOffset((prev) => (prev === nextTop ? prev : nextTop));
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(finalList);
    if (knockoutContainerRef.current) {
      observer.observe(knockoutContainerRef.current);
    }
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [
    knockoutMatchesByStage,
    knockoutCenters,
    knockoutContainerRef,
    compactKnockout,
    knockoutCardHeight,
    isSmallScreen,
    showKnockoutContent,
    compactTight,
  ]);

  React.useLayoutEffect(() => {
    if (!showKnockoutContent) {
      setFinalCenterOverride(null);
      return;
    }
    const container = knockoutContainerRef.current;
    const finalList = finalListRef.current;
    const semifinalMatches = knockoutMatchesByStage.get("Semifinal") ?? [];
    if (!container || !finalList || semifinalMatches.length === 0) {
      setFinalCenterOverride(null);
      return;
    }
    let frame = 0;
    const compute = () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      frame = requestAnimationFrame(() => {
        const containerRect = container.getBoundingClientRect();
        const finalListRect = finalList.getBoundingClientRect();
        const cardHeight = isSmallScreen ? 56 : 64; // Match card height
        
        // Get SF centers
        const sfCenters = semifinalMatches
          .map((match) => {
            const el = knockoutRefs.current.get(match.id);
            if (!el) return null;
            const rect = el.getBoundingClientRect();
            return rect.top - containerRect.top + rect.height / 2;
          })
          .filter((value): value is number => typeof value === "number");
        if (sfCenters.length === 0) return;
        
        const sfMin = Math.min(...sfCenters);
        const sfMax = Math.max(...sfCenters);
        const sfAvg = (sfMin + sfMax) / 2;
        
        const listOffset = finalListRect.top - containerRect.top;
        
        // Position Final equidistant from semis as Third Place (symmetric)
        // Final CENTER = sfAvg - listOffset - 80 - cardHeight/2
        const finalOffset = isSmallScreen ? 56 : 72; // Same offset used for Third Place
        // In compact mode, move final down by one match height
        const compactAdjustment = compactKnockout ? (knockoutCardHeight ?? 64) : 0;
        const nextCenter = sfAvg - listOffset - finalOffset - cardHeight / 2 + compactAdjustment;
        
        setFinalCenterOverride((prev) => (prev === nextCenter ? prev : nextCenter));
      });
    };
    const observer = new ResizeObserver(compute);
    observer.observe(container);
    observer.observe(finalList);
    compute();
    window.addEventListener("resize", compute);
    return () => {
      window.removeEventListener("resize", compute);
      observer.disconnect();
      if (frame) {
        cancelAnimationFrame(frame);
      }
    };
  }, [
    knockoutMatchesByStage,
    knockoutCenters,
    compactKnockout,
    isSmallScreen,
    showKnockoutContent,
    compactTight,
  ]);

  const handleAutopredict = React.useCallback(() => {
    setShowQualifierHint(false);
    setShowGroupHint(false);
    const snapshot = chooseAutopredictSnapshot(simulateTournamentAutopredict);
    if (!snapshot) {
      return;
    }
    commitAutopredictSnapshot(snapshot);
  }, [
    chooseAutopredictSnapshot,
    commitAutopredictSnapshot,
    simulateTournamentAutopredict,
  ]);

  const handleResetAll = React.useCallback(() => {
    setQualifierWinners({});
    setGroupScores({});
    setKnockoutWinners({});
    setAutoQualifierWinners({});
    setAutoGroupScores({});
    setAutoKnockoutWinners({});
  }, []);

  const handleResetAutopredictions = React.useCallback(() => {
    const autoQualifierIds = Object.keys(autoQualifierWinners);
    let nextQualifierWinners = { ...qualifierWinners };
    autoQualifierIds.forEach((matchId) => {
      const updated = { ...nextQualifierWinners, [matchId]: null };
      nextQualifierWinners = clearDependentSelections(
        updated,
        matchId,
        qualifierDependents
      );
    });

    const affectedSlots = new Set<string>();
    autoQualifierIds.forEach((matchId) => {
      const slots = qualifierSlotsByMatch.get(matchId);
      if (!slots) {
        return;
      }
      slots.forEach((slot) => affectedSlots.add(slot));
    });
    const affectedGroups = new Set<string>();
    affectedSlots.forEach((slot) => {
      const groups = groupIdsBySlot.get(slot);
      if (groups) {
        groups.forEach((groupId) => affectedGroups.add(groupId));
      }
    });

    const nextGroupScores = { ...groupScores };
    Object.keys(autoGroupScores).forEach((matchId) => {
      delete nextGroupScores[matchId];
    });
    affectedSlots.forEach((slot) => {
      const matchIds = groupMatchIdsByTeam.get(slot);
      if (!matchIds) {
        return;
      }
      matchIds.forEach((matchId) => {
        delete nextGroupScores[matchId];
      });
    });

    let nextKnockoutWinners = { ...knockoutWinners };
    let nextAutoKnockoutWinners = { ...autoKnockoutWinners };
    const autoKnockoutIds = Object.keys(autoKnockoutWinners);
    if (autoKnockoutIds.length > 0) {
      const cleared = clearKnockoutSelectionsByMatchIds(
        nextKnockoutWinners,
        autoKnockoutIds
      );
      nextKnockoutWinners = cleared.next;
      nextAutoKnockoutWinners = {};
    }

    if (affectedGroups.size > 0) {
      const rootsToClear = new Set<string>();
      affectedGroups.forEach((groupId) => {
        const rootMatches = knockoutRootsByGroup.get(groupId);
        if (rootMatches) {
          rootMatches.forEach((matchId) => rootsToClear.add(matchId));
        }
      });
      if (rootsToClear.size > 0) {
        const cleared = clearKnockoutSelectionsByMatchIds(
          nextKnockoutWinners,
          rootsToClear
        );
        nextKnockoutWinners = cleared.next;
        cleared.clearedIds.forEach((matchId) => {
          delete nextAutoKnockoutWinners[matchId];
        });
      }
    }

    const nextSlotWinners = resolveQualifierState(
      data.qualifiers,
      nextQualifierWinners
    ).slotWinners;
    const clearedForGroups = computeClearedKnockoutSelections(
      nextKnockoutWinners,
      groupScores,
      nextGroupScores,
      {
        previousSlotWinners: slotWinners,
        nextSlotWinners,
      }
    );
    nextKnockoutWinners = clearedForGroups.nextWinners;
    clearedForGroups.clearedIds.forEach((matchId) => {
      delete nextAutoKnockoutWinners[matchId];
    });

    setQualifierWinners(nextQualifierWinners);
    setAutoQualifierWinners({});
    setGroupScores(nextGroupScores);
    setAutoGroupScores({});
    setKnockoutWinners(nextKnockoutWinners);
    setAutoKnockoutWinners(nextAutoKnockoutWinners);
  }, [
    autoGroupScores,
    autoKnockoutWinners,
    autoQualifierWinners,
    clearKnockoutSelectionsByMatchIds,
    computeClearedKnockoutSelections,
    data.qualifiers,
    groupIdsBySlot,
    groupMatchIdsByTeam,
    groupScores,
    knockoutRootsByGroup,
    knockoutWinners,
    qualifierDependents,
    qualifierSlotsByMatch,
    qualifierWinners,
    slotWinners,
  ]);

  const handleGroupAutopredict = React.useCallback(
    (groupId: string) => {
      setShowGroupHint(false);
      const snapshot = chooseAutopredictSnapshot(() => simulateGroupAutopredict(groupId));
      if (!snapshot) {
        return;
      }
      commitAutopredictSnapshot(snapshot);
    },
    [chooseAutopredictSnapshot, commitAutopredictSnapshot, simulateGroupAutopredict]
  );

  const handleGroupReset = React.useCallback(
    (groupId: string) => {
      const matches = groupMatchesFor(groupId, resolvedGroupMatches);
      if (matches.length === 0) {
        return;
      }
      const nextScores = { ...groupScores };
      const nextAutoScores = { ...autoGroupScores };
      let changed = false;
      matches.forEach((match) => {
        const key = String(match.id);
        if (nextScores[key]) {
          delete nextScores[key];
          changed = true;
        }
        if (nextAutoScores[key]) {
          delete nextAutoScores[key];
        }
      });
      if (!changed) {
        return;
      }
      setGroupScores(nextScores);
      setAutoGroupScores(nextAutoScores);
      clearKnockoutOnGroupChange(nextScores);
    },
    [autoGroupScores, clearKnockoutOnGroupChange, groupScores, resolvedGroupMatches]
  );

  const handleQualifierAutopredict = React.useCallback(
    (path: string) => {
      setShowQualifierHint(false);
      const snapshot = chooseAutopredictSnapshot(() => simulateQualifierAutopredict(path));
      if (!snapshot) {
        return;
      }
      commitAutopredictSnapshot(snapshot);
    },
    [chooseAutopredictSnapshot, commitAutopredictSnapshot, simulateQualifierAutopredict]
  );

  const handleQualifierReset = React.useCallback(
    (path: string) => {
      let nextQualifierWinners = { ...qualifierWinners };
      const nextAutoQualifierWinners = { ...autoQualifierWinners };
      const qualifierStateLocal = resolveQualifierState(
        data.qualifiers,
        nextQualifierWinners
      );
      const matches = qualifierStateLocal.matches.filter(
        (match) => match.path === path
      );
      if (matches.length === 0) {
        return;
      }
      const clearedMatchIds = new Set<string>();
      matches.forEach((match) => {
        const key = String(match.id);
        const updated = { ...nextQualifierWinners, [key]: null };
        const cleared = clearDependentSelections(
          updated,
          key,
          qualifierDependents
        );
        Object.keys(nextQualifierWinners).forEach((id) => {
          if (nextQualifierWinners[id] && cleared[id] === null) {
            clearedMatchIds.add(id);
          }
        });
        if (nextQualifierWinners[key]) {
          clearedMatchIds.add(key);
        }
        nextQualifierWinners = cleared;
      });
      if (clearedMatchIds.size === 0) {
        return;
      }
      clearedMatchIds.forEach((matchId) => {
        delete nextAutoQualifierWinners[matchId];
      });
      const affectedSlots = new Set<string>();
      clearedMatchIds.forEach((matchId) => {
        const slots = qualifierSlotsByMatch.get(matchId);
        if (slots) {
          slots.forEach((slot) => affectedSlots.add(slot));
        }
      });
      const affectedGroups = new Set<string>();
      affectedSlots.forEach((slot) => {
        const groups = groupIdsBySlot.get(slot);
        if (groups) {
          groups.forEach((groupId) => affectedGroups.add(groupId));
        }
      });
      const nextGroupScores = { ...groupScores };
      const nextAutoGroupScores = { ...autoGroupScores };
      affectedSlots.forEach((slot) => {
        const matchIds = groupMatchIdsByTeam.get(slot);
        if (!matchIds) {
          return;
        }
        matchIds.forEach((matchId) => {
          delete nextGroupScores[matchId];
          delete nextAutoGroupScores[matchId];
        });
      });
      let nextKnockoutWinners = { ...knockoutWinners };
      let nextAutoKnockoutWinners = { ...autoKnockoutWinners };
      if (affectedGroups.size > 0) {
        affectedGroups.forEach((groupId) => {
          const rootMatches = knockoutRootsByGroup.get(groupId);
          if (rootMatches) {
            rootMatches.forEach((matchId) => {
              nextKnockoutWinners[matchId] = null;
              nextKnockoutWinners = clearDependentSelections(
                nextKnockoutWinners,
                matchId,
                knockoutDependents
              );
            });
          }
        });
        const clearedIds = Object.keys(knockoutWinners).filter(
          (matchId) => knockoutWinners[matchId] && nextKnockoutWinners[matchId] === null
        );
        clearedIds.forEach((matchId) => {
          delete nextAutoKnockoutWinners[matchId];
        });
      }
      setQualifierWinners(nextQualifierWinners);
      setAutoQualifierWinners(nextAutoQualifierWinners);
      setGroupScores(nextGroupScores);
      setAutoGroupScores(nextAutoGroupScores);
      setKnockoutWinners(nextKnockoutWinners);
      setAutoKnockoutWinners(nextAutoKnockoutWinners);
    },
    [
      autoGroupScores,
      autoKnockoutWinners,
      autoQualifierWinners,
      data.qualifiers,
      groupIdsBySlot,
      groupMatchIdsByTeam,
      groupScores,
      knockoutDependents,
      knockoutRootsByGroup,
      knockoutWinners,
      qualifierDependents,
      qualifierSlotsByMatch,
      qualifierWinners,
    ]
  );

  const handleSectionQualifiersAutopredict = React.useCallback(() => {
    setShowQualifierHint(false);
    const snapshot = chooseAutopredictSnapshot(simulateSectionQualifiersAutopredict);
    if (!snapshot) {
      return;
    }
    commitAutopredictSnapshot(snapshot);
  }, [
    chooseAutopredictSnapshot,
    commitAutopredictSnapshot,
    simulateSectionQualifiersAutopredict,
  ]);

  const handleSectionQualifiersReset = React.useCallback(() => {
    if (!data.qualifiers.length) {
      return;
    }
    const nextQualifierWinners: Record<string, WinnerSelection> = {};
    const nextAutoQualifierWinners: Record<string, boolean> = {};
    const affectedSlots = new Set<string>();
    data.qualifiers.forEach((match) => {
      const slots = qualifierSlotsByMatch.get(String(match.id));
      if (slots) {
        slots.forEach((slot) => affectedSlots.add(slot));
      }
    });
    const affectedGroups = new Set<string>();
    affectedSlots.forEach((slot) => {
      const groups = groupIdsBySlot.get(slot);
      if (groups) {
        groups.forEach((groupId) => affectedGroups.add(groupId));
      }
    });
    const nextGroupScores = { ...groupScores };
    const nextAutoGroupScores = { ...autoGroupScores };
    affectedSlots.forEach((slot) => {
      const matchIds = groupMatchIdsByTeam.get(slot);
      if (!matchIds) {
        return;
      }
      matchIds.forEach((matchId) => {
        delete nextGroupScores[matchId];
        delete nextAutoGroupScores[matchId];
      });
    });
    let nextKnockoutWinners = { ...knockoutWinners };
    let nextAutoKnockoutWinners = { ...autoKnockoutWinners };
    if (affectedGroups.size > 0) {
      affectedGroups.forEach((groupId) => {
        const rootMatches = knockoutRootsByGroup.get(groupId);
        if (!rootMatches) {
          return;
        }
        rootMatches.forEach((matchId) => {
          nextKnockoutWinners[matchId] = null;
          nextKnockoutWinners = clearDependentSelections(
            nextKnockoutWinners,
            matchId,
            knockoutDependents
          );
        });
      });
      const clearedIds = Object.keys(knockoutWinners).filter(
        (matchId) => knockoutWinners[matchId] && nextKnockoutWinners[matchId] === null
      );
      clearedIds.forEach((matchId) => {
        delete nextAutoKnockoutWinners[matchId];
      });
    }
    setQualifierWinners(nextQualifierWinners);
    setAutoQualifierWinners(nextAutoQualifierWinners);
    setGroupScores(nextGroupScores);
    setAutoGroupScores(nextAutoGroupScores);
    setKnockoutWinners(nextKnockoutWinners);
    setAutoKnockoutWinners(nextAutoKnockoutWinners);
  }, [
    autoGroupScores,
    autoKnockoutWinners,
    data.qualifiers,
    groupIdsBySlot,
    groupMatchIdsByTeam,
    groupScores,
    knockoutDependents,
    knockoutRootsByGroup,
    knockoutWinners,
    qualifierSlotsByMatch,
  ]);

  const handleSectionGroupsAutopredict = React.useCallback(() => {
    setShowGroupHint(false);
    const snapshot = chooseAutopredictSnapshot(simulateSectionGroupsAutopredict);
    if (!snapshot) {
      return;
    }
    commitAutopredictSnapshot(snapshot);
  }, [
    chooseAutopredictSnapshot,
    commitAutopredictSnapshot,
    simulateSectionGroupsAutopredict,
  ]);

  React.useEffect(() => {
    if (!pendingGroupsAfterQualifiers.current) {
      return;
    }
    if (hasUnpredictedQualifiers()) {
      return;
    }
    pendingGroupsAfterQualifiers.current = false;
    if (hasUnpredictedGroups()) {
      handleSectionGroupsAutopredict();
    }
  }, [handleSectionGroupsAutopredict, hasUnpredictedGroups, hasUnpredictedQualifiers]);

  const handleSectionGroupsReset = React.useCallback(() => {
    if (!Object.keys(groupScores).length) {
      return;
    }
    setGroupScores({});
    setAutoGroupScores({});
    clearKnockoutOnGroupChange({});
  }, [clearKnockoutOnGroupChange, groupScores]);

  const handleSectionKnockoutsAutopredict = React.useCallback(() => {
    const snapshot = chooseAutopredictSnapshot(simulateSectionKnockoutsAutopredict);
    if (!snapshot) {
      return;
    }
    commitAutopredictSnapshot(snapshot);
  }, [
    chooseAutopredictSnapshot,
    commitAutopredictSnapshot,
    simulateSectionKnockoutsAutopredict,
  ]);

  const handleSectionKnockoutsReset = React.useCallback(() => {
    if (!Object.keys(knockoutWinners).length) {
      return;
    }
    setKnockoutWinners({});
    setAutoKnockoutWinners({});
  }, [knockoutWinners]);

  const knockoutBaseColumnWidth = compactKnockout ? 48 : isSmallScreen ? 152 : 200;
  const knockoutBaseGap = compactKnockout ? (compactTight ? 6 : 8) : 24;
  const knockoutSfPosition = compactKnockout ? (compactTight ? 2.75 : 3) : 2.8;
  const knockoutLeftBlockWidth =
    (knockoutSfPosition - 1) * (knockoutBaseColumnWidth + knockoutBaseGap) +
    knockoutBaseColumnWidth;
  const knockoutMinGapBetweenSFs = compactKnockout
    ? knockoutBaseColumnWidth * (compactTight ? 1.2 : 1.5)
    : knockoutBaseColumnWidth;
  const knockoutMinBracketWidth =
    knockoutLeftBlockWidth * 2 + knockoutMinGapBetweenSFs;
  const activeQualifierEntry =
    qualifierEntries.find(([path]) => path === activeQualifierPath) ??
    qualifierEntries[0];
  const activeQualifierPathValue =
    activeQualifierPath ?? activeQualifierEntry?.[0] ?? null;
  const showQualifierOnboarding =
    Boolean(showQualifierHint && firstQualifierPath) &&
    activeQualifierPathValue === firstQualifierPath;
  const qualifierPanelId = (path: string) =>
    `qualifier-panel-${path.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  const supergroupMinWidth = 500;
  const hasQualifiers = data.qualifiers.length > 0;
  const canAutopredictQualifiers = hasUnpredictedQualifiers();
  const canAutopredictGroups = hasUnpredictedGroups();
  const canAutopredictKnockouts = hasUnpredictedKnockouts();
  const isGroupStageReady = hasQualifiers ? !hasUnpredictedQualifiers() : true;
  const canAutopredictTournament =
    (hasQualifiers && canAutopredictQualifiers) ||
    canAutopredictGroups ||
    canAutopredictKnockouts;
  const canResetTournament =
    (hasQualifiers && hasAnyQualifierPredictions) ||
    hasAnyGroupPredictions ||
    hasAnyKnockoutPredictions;

  React.useEffect(() => {
    if (isGroupStageReady) {
      setShowGroupStageContent(true);
      setShowKnockoutSection(true);
    }
  }, [isGroupStageReady]);

  React.useEffect(() => {
    if (isKnockoutBracketReady) {
      if (process.env.NODE_ENV !== "production") {
        console.log("[predictor] knockout bracket ready");
      }
      setShowKnockoutContent(true);
    }
  }, [isKnockoutBracketReady]);

  return (
    <div
      className="flex flex-col gap-12"
      style={
        {
          "--cta-color": "#ef4444",
          "--cta-color-rgb": "239 68 68",
        } as React.CSSProperties
      }
    >
      <style jsx global>{`
        @keyframes hintPulse {
          0%,
          100% {
            box-shadow: 0 0 0 0 rgb(var(--cta-color-rgb) / 0.35);
          }
          50% {
            box-shadow: 0 0 0 8px rgb(var(--cta-color-rgb) / 0.25);
          }
        }
        .hint-pulse {
          animation: hintPulse 1.6s ease-in-out infinite;
        }
      `}</style>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => setShowPretournament((prev) => !prev)}
          role="switch"
          aria-checked={showingCurrent}
          aria-label={showingCurrent ? "Current" : "Pre-tournament"}
          className={cn(
            "relative h-6 w-11 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300",
            showingCurrent ? "bg-slate-900" : "bg-slate-300"
          )}
        >
          <span
            aria-hidden="true"
            className={cn(
              "absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform",
              showingCurrent ? "translate-x-5" : "translate-x-0"
            )}
          />
        </button>
        <span className="text-sm text-slate-700">
          {showingCurrent ? "Current" : "Pre-tournament"}
        </span>
        {showPretournament && !pretournamentData && !pretournamentLoadError ? (
          <span className="text-sm text-ink-400">Loading pre-tournament model output...</span>
        ) : null}
      </div>
      {pretournamentLoadError ? (
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          {pretournamentLoadError}
        </div>
      ) : null}
      {hasQualifiers ? (
        <>
      <div className="h-px w-full bg-slate-200/80" />
      <section className="space-y-3 sm:space-y-6">
        <div>
          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
            <div className="flex items-end gap-2 sm:gap-3">
              <span
                className="w-2 rounded-full bg-blue-300"
                style={{ height: "calc(1em * 2)" }}
                aria-hidden="true"
              />
              <h2 className="text-xl sm:text-2xl font-semibold text-ebony">
                Qualifier playoffs
              </h2>
            </div>
            <div className="flex min-h-[32px] flex-wrap items-center gap-2">
              <LoadingButton
                loading={Boolean(loadingKeys["section:qualifiers"])}
                disabled={!canAutopredictQualifiers}
                onClick={() =>
                  runAutopredictWithDelay(
                    "section:qualifiers",
                    handleSectionQualifiersAutopredict
                  )
                }
                className={cn(
                  "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                  canAutopredictQualifiers
                    ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                    : "text-slate-500"
                )}
              >
                Auto-predict qualifiers
              </LoadingButton>
              <button
                type="button"
                disabled={!hasAnyQualifierPredictions}
                onClick={handleSectionQualifiersReset}
                className={cn(
                  "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                  hasAnyQualifierPredictions
                    ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                    : "text-slate-500 cursor-default"
                )}
              >
                Reset qualifiers
              </button>
            </div>
          </div>
        </div>
        <div
          className="flex flex-col gap-3 sm:gap-6 md:flex-row md:flex-wrap md:items-stretch md:justify-start"
          style={
            {
              "--supergroup-min": `${supergroupMinWidth}px`,
            } as React.CSSProperties
          }
        >
          <div className="flex min-w-0 w-full md:w-auto flex-col md:flex-[3_1_0%] md:min-w-[var(--supergroup-min)]">
            {isGroupTabbed ? (
              <div className="relative flex w-full min-w-0 flex-col overflow-hidden rounded-xl bg-slate-50 ring-1 ring-slate-200 p-2 sm:p-4 flex-1">
                <div className="border-b border-slate-200 pb-3">
                <div className="overflow-visible pl-1 pr-2">
                  <div
                    ref={qualifierPathTabsRef}
                    role="tablist"
                    aria-label="Qualifier playoff tabs"
                    className="flex items-center gap-2 overflow-x-auto pb-2 pt-2 pl-1 pr-2"
                  >
                    {qualifierEntries.map(([path]) => {
                      const isActive = path === activeQualifierPathValue;
                      const isHighlighted = qualifierPathsWithCtaMatches.has(path);
                      const [pathFirst, pathSecond, pathThird, ...pathRest] =
                        path.split(" ");
                      const thirdLine = [pathThird, ...pathRest].filter(Boolean).join(" ");
                      const twoLine = [pathSecond, pathThird, ...pathRest]
                        .filter(Boolean)
                        .join(" ");
                      return (
                        <button
                          key={path}
                          type="button"
                          role="tab"
                          aria-selected={isActive}
                          aria-controls={qualifierPanelId(path)}
                          className={cn(
                            "rounded-full border px-2 sm:px-3 py-0.5 sm:py-1 text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide transition-colors",
                            isHighlighted && "ring-2 ring-[color:var(--cta-color)]",
                            isActive
                              ? "border-slate-900 bg-slate-900 text-white"
                              : cn(
                                  "bg-white text-slate-600 hover:bg-slate-100",
                                  isHighlighted ? "border-[color:var(--cta-color)]" : "border-slate-200"
                                )
                          )}
                          onClick={(e) => {
                            setActiveQualifierPath(path);
                            // Scroll button into view if partially visible
                            const container = qualifierPathTabsRef.current;
                            const button = e.currentTarget;
                            if (container) {
                              const containerRect = container.getBoundingClientRect();
                              const buttonRect = button.getBoundingClientRect();
                              
                              // Check if button is partially or fully outside the visible area
                              const isFullyVisible = 
                                buttonRect.left >= containerRect.left &&
                                buttonRect.right <= containerRect.right;
                              
                              if (!isFullyVisible) {
                                // Scroll the button into view
                                button.scrollIntoView({
                                  behavior: 'smooth',
                                  block: 'nearest',
                                  inline: 'center',
                                });
                              }
                            }
                          }}
                        >
                          <span className="flex flex-col items-center leading-[1.05] sm:hidden">
                            <span className="whitespace-nowrap">{pathFirst}</span>
                            {pathSecond ? (
                              <span className="whitespace-nowrap">{pathSecond}</span>
                            ) : null}
                            {thirdLine ? (
                              <span className="whitespace-nowrap">{thirdLine}</span>
                            ) : null}
                          </span>
                          <span className="hidden flex-col items-center leading-[1.1] sm:flex md:hidden">
                            <span className="whitespace-nowrap">{pathFirst}</span>
                            {twoLine ? (
                              <span className="whitespace-nowrap">{twoLine}</span>
                            ) : null}
                          </span>
                          <span className="hidden items-center gap-1 leading-none md:flex">
                            <span className="whitespace-nowrap">{pathFirst}</span>
                            {twoLine ? (
                              <span className="whitespace-nowrap">{twoLine}</span>
                            ) : null}
                          </span>
                        </button>
                      );
                    })}
                    </div>
                  </div>
                </div>
                {activeQualifierEntry && (
                  <div
                    id={qualifierPanelId(activeQualifierEntry[0])}
                    role="tabpanel"
                    className="pt-4 flex-1 flex flex-col min-h-0"
                  >
                    <QualifierPathBracket
                      key={activeQualifierEntry[0]}
                      path={activeQualifierEntry[0]}
                      matches={activeQualifierEntry[1]}
                      winnerSelections={qualifierWinners}
                      onWinnerSelect={updateQualifierWinner}
                      onAutoPredict={(pathId) =>
                        runAutopredictWithDelay(
                          `qual:${pathId}`,
                          () => handleQualifierAutopredict(pathId)
                        )
                      }
                      onReset={handleQualifierReset}
                      autoPredictLoading={Boolean(
                        loadingKeys[`qual:${activeQualifierEntry[0]}`]
                      )}
                      flags={data.flags}
                      getMatchProbabilityLabels={getMatchProbabilityLabels}
                      showTitle={false}
                      embedded
                      showHint={showQualifierOnboarding}
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="grid gap-3 sm:gap-6 lg:gap-6 grid-cols-[repeat(auto-fit,minmax(432px,1fr))] items-stretch">
                {qualifierEntries.map(([path, matches]) => (
                  <QualifierPathBracket
                    key={path}
                    path={path}
                    matches={matches}
                    winnerSelections={qualifierWinners}
                    onWinnerSelect={updateQualifierWinner}
                    onAutoPredict={(pathId) =>
                      runAutopredictWithDelay(
                        `qual:${pathId}`,
                        () => handleQualifierAutopredict(pathId)
                      )
                    }
                    onReset={handleQualifierReset}
                    autoPredictLoading={Boolean(loadingKeys[`qual:${path}`])}
                    flags={data.flags}
                    getMatchProbabilityLabels={getMatchProbabilityLabels}
                  />
                ))}
              </div>
            )}
          </div>
          <div className="flex min-w-0 w-full md:w-auto flex-col md:flex-[2_1_0%] md:min-w-[var(--supergroup-min)] space-y-2 sm:space-y-4 rounded-xl bg-slate-50 ring-1 ring-slate-200 p-2 sm:p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base sm:text-lg font-semibold text-slate-900">
                Qualified through playoffs
              </h3>
            </div>
            <div className="overflow-hidden rounded-xl bg-white ring-1 ring-slate-200 shadow-sm flex-1">
              <table className="w-full table-fixed text-sm">
                <colgroup>
                  <col />
                  <col style={{ width: "120px" }} />
                </colgroup>
          <thead className="bg-slate-200 border-b border-slate-200">
                  <tr>
                    <th className="px-1 sm:px-2 py-1.5 sm:py-2.5 text-left text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                      Team
                    </th>
                    <th className="px-1 sm:px-2 py-1.5 sm:py-2.5 text-left text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                      Path
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {qualifierQualifiedRows.map((row) => (
                    <tr key={row.slot}>
                      <td className="px-1 sm:px-2 py-1.5 sm:py-2.5">
                        <div className="flex min-w-0 items-center gap-1 sm:gap-2">
                          {row.team ? (
                            <TeamFlag
                              team={row.team}
                              flags={data.flags}
                              className="h-3.5 w-5 sm:h-4 sm:w-6 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] flex-shrink-0"
                            />
                          ) : (
                            <div className="h-3.5 w-5 sm:h-4 sm:w-6 rounded-sm bg-slate-100 ring-1 ring-slate-200 flex-shrink-0" />
                          )}
                          <span className={cn(
                            "min-w-0 truncate text-xs sm:text-sm",
                            row.team ? "font-medium text-slate-900" : "text-slate-400"
                          )}>
                            {row.team ? formatDisplayLabel(row.team) : "—"}
                          </span>
                        </div>
                      </td>
                      <td className={cn(
                        "px-1 sm:px-2 py-1.5 sm:py-2.5 text-xs sm:text-sm",
                        row.team ? "text-slate-700" : "text-slate-400"
                      )}>
                        {row.path ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <div className="h-px w-full bg-slate-200/80" />
        </>
      ) : null}

      <section className="space-y-3 sm:space-y-6">
        <div>
          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
            <div className="flex items-end gap-2 sm:gap-3">
              <span
                className="w-2 rounded-full bg-blue-300"
                style={{ height: "calc(1em * 2)" }}
                aria-hidden="true"
              />
              <h2 className="text-xl sm:text-2xl font-semibold text-ebony">Group stage</h2>
            </div>
            <div className="flex min-h-[32px] flex-wrap items-center gap-2">
              {isGroupStageReady ? (
                <>
                  <LoadingButton
                    loading={Boolean(loadingKeys["section:groups"])}
                    disabled={!canAutopredictGroups}
                    onClick={() =>
                      runAutopredictWithDelay(
                        "section:groups",
                        handleSectionGroupsAutopredict
                      )
                    }
                    className={cn(
                      "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                      canAutopredictGroups
                        ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                        : "text-slate-500"
                    )}
                  >
                    Auto-predict all groups
                  </LoadingButton>
                  <button
                    type="button"
                    disabled={!hasAnyGroupPredictions}
                    onClick={handleSectionGroupsReset}
                    className={cn(
                      "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                      hasAnyGroupPredictions
                        ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                        : "text-slate-500 cursor-default"
                    )}
                  >
                    Reset all groups
                  </button>
                </>
              ) : (
                <div className="inline-flex flex-wrap items-center gap-2 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-[11px] font-medium text-red-700">
                  <span>All qualifiers must be predicted.</span>
                  <LoadingButton
                    loading={Boolean(loadingKeys["section:qualifiers-group-gate"])}
                    disabled={!canAutopredictQualifiers}
                    onClick={() =>
                      runAutopredictWithDelay(
                        "section:qualifiers-group-gate",
                        handleSectionQualifiersAutopredict
                      )
                    }
                    className={cn(
                      "rounded-md bg-white px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-red-200",
                      canAutopredictQualifiers
                        ? "text-red-700 hover:bg-red-100"
                        : "text-red-300 cursor-default"
                    )}
                  >
                    Auto-predict qualifiers
                  </LoadingButton>
                </div>
              )}
            </div>
          </div>
        </div>
        {showGroupStageContent && (
          <div
            ref={groupCardsContainerRef}
            className={cn(
              "flex flex-col gap-3 sm:gap-6 md:flex-row md:flex-wrap md:items-stretch md:justify-start",
              thirdPlaceRankingRows.length > 0
                ? "md:gap-6"
                : ""
            )}
            style={
              {
                "--supergroup-min": `${supergroupMinWidth}px`,
              } as React.CSSProperties
            }
          >
            <div className="flex min-w-0 w-full md:w-auto flex-col md:flex-[3_1_0%] md:min-w-[var(--supergroup-min)]">
              <GroupStageCards
                groupTables={groupTables}
                resolvedGroupMatches={resolvedGroupMatches}
                groupScores={groupScores}
                lockedGroupMatchIds={lockedGroupMatchIds}
                updateGroupScore={updateGroupScore}
                updateGroupScorePair={updateGroupScorePair}
                winProbabilities={activeWinProbabilities}
                groupsWithUnresolvedParticipants={groupsWithUnresolvedParticipants}
                groupQualifierPaths={groupQualifierPaths}
                showGroupHint={showGroupHint}
                groupsWithCtaMatches={groupsWithCtaMatches}
              getMatchProbabilityLabels={getMatchProbabilityLabels}
              onGroupHintDismiss={() => setShowGroupHint(false)}
              loadingKeys={loadingKeys}
              runAutopredictWithDelay={runAutopredictWithDelay}
                handleGroupAutopredict={handleGroupAutopredict}
                handleGroupReset={handleGroupReset}
                handleQualifierAutopredict={handleQualifierAutopredict}
                qualifierPathPredictionStatus={qualifierPathPredictionStatus}
                groupCompletion={groupCompletion}
                qualifiedThirdGroups={qualifiedThirdGroups}
                allGroupMatchesComplete={allGroupMatchesComplete}
                flags={data.flags}
                isTabbed={isGroupTabbed}
                lockResultsActive={lockResultsActive}
              />
            </div>
            {thirdPlaceRankingRows.length > 0 && (
              <div className="flex min-w-0 w-full md:w-auto flex-col md:flex-[2_1_0%] md:min-w-[var(--supergroup-min)] space-y-2 sm:space-y-4 rounded-xl bg-slate-50 ring-1 ring-slate-200 p-2 sm:p-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-base sm:text-lg font-semibold text-slate-900">
                    Ranking of 3rd place teams
                  </h3>
                </div>
                <div className="flex w-full px-0.5 flex-1">
                  <GroupTable
                    group={{ id: "Third place", teams: [] }}
                    rows={thirdPlaceRankingRows}
                    highlightThird={false}
                    highlightWeakThird={false}
                    highlightTop={8}
                    showTieInfo={allGroupMatchesComplete}
                    flags={data.flags}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {showKnockoutSection && (
        <>
          <div className="h-px w-full bg-slate-200/80" />
          <section className="relative">
            {showCompactModeHint && compactModeHintPosition && (
              <div
                className={cn(
                  "pointer-events-none absolute z-30 transition-opacity duration-200 ease-out",
                  compactModeHintVisible ? "opacity-100" : "opacity-0"
                )}
                style={{
                  left: `${compactModeHintPosition.x}px`,
                  top: `${compactModeHintPosition.y}px`,
                  transform: "translateY(-100%)",
                }}
              >
                <div 
                  ref={compactModeHintBoxRef}
                  className="flex items-center gap-1 rounded-md bg-slate-900 px-1.5 sm:px-2 py-0.5 sm:py-1 text-[10px] sm:text-[11px] font-semibold text-white shadow-sm max-w-[240px] sm:max-w-none"
                >
                  <span className="sm:whitespace-nowrap">Toggle compact mode to see team names and probabilities.</span>
                  <button
                    type="button"
                    onClick={dismissCompactModeHint}
                    className="ml-1 flex h-4 w-4 items-center justify-center rounded hover:bg-slate-700 transition-colors pointer-events-auto"
                    aria-label="Dismiss hint"
                  >
                    <svg
                      className="h-3 w-3"
                      viewBox="0 0 20 20"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M15 5L5 15M5 5l10 10" />
                    </svg>
                  </button>
                </div>
                {compactModeHintArrowLeft !== null && (
                  <svg
                    className="absolute top-full -mt-px h-2 w-4 text-slate-900"
                    style={{
                      left: `${compactModeHintArrowLeft}px`,
                      transform: "translateX(-50%)",
                    }}
                    viewBox="0 0 20 8"
                    fill="none"
                    aria-hidden="true"
                  >
                    <path d="M0 0 L10 8 L20 0" fill="currentColor" />
                  </svg>
                )}
              </div>
            )}
            <div className="space-y-3 sm:space-y-6">
              <div>
                <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                  <div className="flex items-end gap-2 sm:gap-3">
                    <span
                      className="w-2 rounded-full bg-blue-300"
                      style={{ height: "calc(1em * 2)" }}
                      aria-hidden="true"
                    />
                    <h2 className="text-xl sm:text-2xl font-semibold text-ebony">Knockout stage</h2>
                  </div>
                  <div className="flex items-center gap-1.5 sm:gap-2">
                    <span className="text-sm font-medium text-slate-600">
                      Compact mode
                    </span>
                    <button
                      ref={compactModeToggleRef}
                      type="button"
                      onClick={() => {
                        hasUserSetCompactKnockout.current = true;
                        setCompactKnockout((prev) => !prev);
                        if (showCompactModeHint) {
                          dismissCompactModeHint();
                        }
                      }}
                      className={cn(
                        "relative inline-flex h-6 w-14 items-center rounded-full p-0.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 focus-visible:ring-offset-2 focus-visible:ring-offset-white",
                        compactKnockout ? "bg-slate-900" : "bg-slate-200"
                      )}
                      aria-pressed={compactKnockout}
                    >
                      <span
                        className={cn(
                          "flex h-5 w-5 items-center justify-center rounded-full bg-white text-[10px] font-semibold text-slate-700 shadow-sm transition-transform",
                          compactKnockout ? "translate-x-8" : "translate-x-0"
                        )}
                      >
                        {compactKnockout ? "ON" : "OFF"}
                      </span>
                    </button>
                  </div>
                  <div className="flex min-h-[56px] flex-wrap items-center gap-2 sm:min-h-0">
                    {isKnockoutBracketReady && (
                      <>
                        <LoadingButton
                          loading={Boolean(loadingKeys["section:knockouts"])}
                          disabled={!canAutopredictKnockouts}
                          onClick={() =>
                            runAutopredictWithDelay(
                              "section:knockouts",
                              handleSectionKnockoutsAutopredict
                            )
                          }
                          className={cn(
                            "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                            canAutopredictKnockouts
                              ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                              : "text-slate-500"
                          )}
                        >
                          Auto-predict knockout
                        </LoadingButton>
                        <button
                          type="button"
                          disabled={!hasAnyKnockoutPredictions}
                          onClick={handleSectionKnockoutsReset}
                          className={cn(
                            "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                            hasAnyKnockoutPredictions
                              ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                              : "text-slate-500 cursor-default"
                          )}
                        >
                          Reset knockout
                        </button>
                      </>
                    )}
                    {!isKnockoutBracketReady && (
                      <div className="inline-flex flex-wrap items-center gap-2 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-[11px] font-medium text-red-700">
                        <span>
                          All qualifier and group stage matches must be predicted.
                        </span>
                        <LoadingButton
                          loading={Boolean(loadingKeys["knockout:resolve"])}
                          disabled={!canAutopredictQualifiers && !canAutopredictGroups}
                          onClick={() =>
                            runAutopredictWithDelay("knockout:resolve", () => {
                              if (hasUnpredictedQualifiers()) {
                                pendingGroupsAfterQualifiers.current = true;
                                handleSectionQualifiersAutopredict();
                              }
                              if (!hasUnpredictedQualifiers() && hasUnpredictedGroups()) {
                                handleSectionGroupsAutopredict();
                              }
                            })
                          }
                          className={cn(
                            "rounded-md bg-white px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-red-200",
                            canAutopredictQualifiers || canAutopredictGroups
                              ? "text-red-700 hover:bg-red-100"
                              : "text-red-300 cursor-default"
                          )}
                        >
                          Auto-predict
                        </LoadingButton>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              {showKnockoutContent && (
                <>
                  <div className={cn(
                    "overflow-x-scroll overflow-y-visible pb-2 knockout-scroll",
                    compactKnockout && "max-w-[520px] lg:max-w-none mx-auto"
                  )} style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgb(203 213 225) rgb(241 245 249)' }}>
                    <div
                      ref={knockoutContainerRef}
                      className="relative px-0.5 lg:px-2"
                      style={{ 
                        minWidth: `${knockoutMinBracketWidth}px`,
                        maxWidth: compactKnockout ? "520px" : undefined,
                        marginLeft: compactKnockout ? "auto" : undefined,
                        marginRight: compactKnockout ? "auto" : undefined,
                      }}
                    >
                      {!compactKnockout && (
                        <div className="pointer-events-none absolute bottom-10 left-1/2 z-20 -translate-x-1/2">
                          <div className="pointer-events-auto flex flex-col items-center gap-2">
                            <div className="flex items-center gap-1.5 sm:gap-2">
                              <LoadingButton
                                loading={Boolean(loadingKeys.tournament)}
                                disabled={!canAutopredictTournament}
                                onClick={() =>
                                  runAutopredictWithDelay("tournament", handleAutopredict)
                                }
                                className={cn(
                                  "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                                  canAutopredictTournament
                                    ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                                    : "text-slate-500"
                                )}
                              >
                                Auto-predict tournament
                              </LoadingButton>
                              <button
                                type="button"
                                disabled={!canResetTournament}
                                onClick={handleResetAll}
                                className={cn(
                                  "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                                  canResetTournament
                                    ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                                    : "text-slate-500 cursor-default"
                                )}
                              >
                                Reset tournament
                              </button>
                            </div>
                            <div className="flex min-h-9 items-center">
                              {isTournamentComplete && (
                                <button
                                  type="button"
                                  onClick={() => {
                                    if (!shareLink) {
                                      setShareStatus("error");
                                      return;
                                    }
                                    if (navigator?.clipboard?.writeText) {
                                      navigator.clipboard
                                        .writeText(shareLink)
                                        .then(() => setShareStatus("copied"))
                                        .catch(() => setShareStatus("error"));
                                      return;
                                    }
                                    const ok = window.prompt("Copy link to share", shareLink);
                                    setShareStatus(ok ? "copied" : "error");
                                  }}
                                  className="inline-flex items-center gap-2 rounded-full bg-white/95 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-700 shadow-sm ring-1 ring-slate-200 hover:bg-white"
                                >
                                  <svg
                                    aria-hidden="true"
                                    viewBox="0 0 24 24"
                                    className="h-4 w-4 text-slate-600"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  >
                                    <path d="M10 13a5 5 0 0 0 7.07 0l2.83-2.83a5 5 0 0 0-7.07-7.07L10.5 5.5" />
                                    <path d="M14 11a5 5 0 0 0-7.07 0L4.1 13.83a5 5 0 0 0 7.07 7.07L13.5 18.5" />
                                  </svg>
                                  {shareStatus === "copied" ? "Link copied" : "Share prediction"}
                                </button>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    <svg
                      className="absolute inset-0 z-0 h-full w-full pointer-events-none"
                      aria-hidden="true"
                    >
                      {knockoutPaths.map((path, index) => (
                        <path
                          key={`${path}-${index}`}
                          d={path}
                          fill="none"
                          stroke="rgb(203 213 225)"
                          strokeWidth={1.5}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      ))}
                    </svg>
                    {/* Split bracket layout - left/right/center all absolutely positioned */}
                    <div className="z-10">
                      {(() => {
                        // In compact mode, cards are 40px wide with tighter spacing
                        const baseColumnWidth = knockoutBaseColumnWidth;
                        const baseGap = knockoutBaseGap;
                        const sfPosition = knockoutSfPosition;
                        const leftBlockWidth = knockoutLeftBlockWidth;
                        const minBracketWidth = knockoutMinBracketWidth;
                
                // Column positions within each block
                const getLeftPosition = (pos: number) => {
                  return `${(pos - 1) * (baseColumnWidth + baseGap)}px`;
                };
                
                const semifinalPosition = sfPosition;

                return (
                  <div 
                    className="relative w-full" 
                    style={{ 
                      minWidth: `${minBracketWidth}px`,
                      minHeight: knockoutListHeight
                        ? `${knockoutListHeight + (compactKnockout ? 12 : 40)}px`
                        : undefined,
                      maxWidth: compactKnockout ? '520px' : undefined,
                      marginLeft: compactKnockout ? 'auto' : undefined,
                      marginRight: compactKnockout ? 'auto' : undefined,
                    }}
                  >
                    {/* Left block: Top half bracket (R32 to Semis) - always left-aligned */}
                    <div className="absolute left-0 top-0" style={{ width: compactKnockout ? `${leftBlockWidth}px` : `calc(50% + ${leftBlockWidth / 2}px)` }}>
                      {stageOrder
                        .filter((stage) => stage !== "Final")
                        .map((stage) => {
                        const matches = splitMatchesByStage[stage]?.top ?? [];
                        const isRoundOf32 = stage === "Round of 32";
                        const cardHeight = knockoutCardHeight ?? 64;
                        const headerOffset = 20;
                        const labelGap = 28;
                        const columnHeight = knockoutListHeight
                          ? knockoutListHeight + headerOffset
                          : undefined;
                          // Map stages to positions from LEFT edge - R32 at left edge, SF towards center
                          const leftPositions: Record<string, number> = {
                            'Round of 32': 1,       // 0px from left (at left edge of page)
                            'Round of 16': 2,       // 224px from left
                            'Quarterfinal': 2.5,    // 336px from left
                            'Semifinal': semifinalPosition, // Slightly further from center
                          };
                          const pos = leftPositions[stage];
                          if (pos === undefined) {
                            return null;
                          }
                          return (
                            <div
                              key={`top-${stage}`}
                              className="absolute top-0"
                              style={{
                                left: getLeftPosition(pos),
                                width: `${baseColumnWidth}px`,
                                height: columnHeight,
                              }}
                            >
                        <div
                          ref={(el) => {
                            if (isRoundOf32) {
                              roundOf32ListRef.current = el;
                            }
                          }}
                          className={cn(
                            "relative",
                            isRoundOf32 ? "flex flex-col gap-4 pt-2" : "pt-4"
                          )}
                          style={
                            !isRoundOf32 && knockoutListHeight
                              ? {
                                  minHeight: `${knockoutListHeight}px`,
                                }
                              : undefined
                          }
                        >
                          {matches.map((match) => {
                            if (!match) {
                              return null;
                            }
                            const handleRoundOf32Click = isRoundOf32
                              ? () => logRoundOf32Match(match)
                              : undefined;
                            const center = knockoutCenters[match.id] ?? 0;
                            const top = isRoundOf32 ? undefined : center - cardHeight / 2;
                            const probabilities = getMatchProbabilityLabels({
                              homeTeam: match.homeResolved ?? match.homeLabel,
                              awayTeam: match.awayResolved ?? match.awayLabel,
                              allowDraw: false,
                              country: match.country,
                            });
                            return (
                              <div
                                key={match.id}
                                ref={(el) => {
                                  if (el) {
                                    knockoutRefs.current.set(match.id, el);
                                  } else {
                                    knockoutRefs.current.delete(match.id);
                                  }
                                }}
                                className={cn(
                                  "w-full flex justify-center",
                                  isRoundOf32 ? "relative" : "absolute left-0"
                                )}
                                style={top !== undefined ? { top } : undefined}
                                onClick={handleRoundOf32Click}
                              >
                                <KnockoutMatchCard
                                  homeTeam={match.homeResolved ?? match.homeLabel}
                                  awayTeam={match.awayResolved ?? match.awayLabel}
                                  winnerSelection={knockoutWinners[String(match.id)] ?? null}
                                  locked={lockResultsActive && lockedKnockoutMatchIds.has(String(match.id))}
                                  onWinnerSelect={(selection) =>
                                    updateKnockoutWinner(match.id, selection)
                                  }
                                  compact={isRoundOf32}
                                  flags={data.flags}
                                  homeWinProb={probabilities.homeWinProb}
                                  awayWinProb={probabilities.awayWinProb}
                                  drawProb={probabilities.drawProb}
                                  isFinal={false}
                                  compactMode={compactKnockout}
                                />
                              </div>
                            );
                          })}
                        </div>
                          </div>
                        );
                      })}

                    </div>

                    {/* Center: Final and Third place - absolutely centered */}
                    <div className="absolute left-1/2 -translate-x-1/2 z-10" style={{ width: `${baseColumnWidth}px` }}>
                      {stageOrder
                        .filter((stage) => stage === "Final")
                        .map((stage) => {
                          // Get champion (winner of Final) - only requires Final to be picked
                          const finalMatch = (knockoutMatchesByStage.get("Final") ?? [])[0];
                          const finalWinner = finalMatch ? knockoutWinners[String(finalMatch.id)] : null;
                          const champion = finalMatch && finalWinner !== null && finalWinner !== undefined
                            ? (finalWinner === "home"
                                ? (finalMatch.homeResolved ?? finalMatch.homeLabel)
                                : (finalMatch.awayResolved ?? finalMatch.awayLabel))
                            : null;
                          const matches = knockoutMatchesByStage.get(stage) ?? [];
                          const cardHeight = knockoutCardHeight ?? 64;
                          const headerOffset = 20;
                          const thirdPlaceMatchTop = stage === "Final" ? thirdPlaceOffset : null;
                          const labelGap = 28;
                          const championGap = 80; // Gap for champion block (increased to prevent intersection)
                          const finalStageHeight =
                            thirdPlaceMatchTop !== null
                              ? Math.max(
                                  knockoutListHeight ?? 0,
                                  thirdPlaceMatchTop + cardHeight
                                )
                              : knockoutListHeight ?? null;
                          const columnHeight =
                            knockoutListHeight && stage === "Final" && finalStageHeight
                              ? finalStageHeight + headerOffset
                              : knockoutListHeight
                                ? knockoutListHeight + headerOffset
                                : undefined;
                          return (
                            <div
                              key={stage}
                              className="relative"
                              style={{
                                width: `${baseColumnWidth}px`,
                                height: columnHeight,
                              }}
                            >
                        <div
                          ref={(el) => {
                            if (stage === "Final") {
                              finalListRef.current = el;
                            }
                          }}
                          className="relative pt-4"
                          style={
                            knockoutListHeight
                              ? {
                                  minHeight: `${
                                    stage === "Final" && finalStageHeight
                                      ? finalStageHeight
                                      : knockoutListHeight
                                  }px`,
                                }
                              : undefined
                          }
                        >
                          {/* Champion block - appears when Final has been picked */}
                          {champion && matches.length > 0 && (() => {
                            const finalMatch = matches[0];
                            const center =
                              stage === "Final" && finalCenterOverride !== null
                                ? finalCenterOverride
                                : knockoutCenters[finalMatch.id] ?? 0;
                            const finalTop = center - cardHeight / 2;
                            
                            // Position champion block
                            let championTop: number;
                            if (!compactKnockout && thirdPlaceMatchTop !== null) {
                              // In non-compact mode: center between Final and Third place
                              // Use the bottom of Final match and top of Third place match
                              const finalBottom = finalTop + cardHeight;
                              const thirdPlaceTop = thirdPlaceMatchTop;
                              const midpoint = (finalBottom + thirdPlaceTop) / 2;
                              // Total block height: CHAMPION text (~20px) + mb-3 (12px) + flag (24px) + gap-2 (8px) + team name (~20px) = ~84px
                              // Center the block by positioning its center at the midpoint
                              const championBlockHeight = 84;
                              championTop = midpoint - championBlockHeight / 2; // Center the entire block
                            } else {
                              // In compact mode: position above Final
                              const championBlockHeight = 80; // Approximate height of champion block content
                              championTop = finalTop - championGap - championBlockHeight;
                            }
                            
                            return (
                              <div
                                key="champion"
                                data-champion-block
                                className={cn(
                                  "absolute flex flex-col items-center",
                                  compactKnockout ? "" : "left-0 w-full"
                                )}
                                style={{ 
                                  top: championTop,
                                  ...(compactKnockout ? { 
                                    // Position relative to bracket container (392px), not Final column (48px)
                                    // Bracket center = 196px, champion (280px) should be centered there
                                    // Champion left should be at: 196px - 140px = 56px from bracket left
                                    // Final column (48px) is centered at bracket center (196px)
                                    // Final column left edge is at: 196px - 24px = 172px from bracket left
                                    // So champion left relative to Final column = 56px - 172px = -116px
                                    // Final column center is at 24px from Final column left
                                    // So from Final center: 24px - 140px = -116px from Final left
                                    // Using left: calc(50% - 140px) where 50% = 24px (Final center)
                                    // This gives: 24px - 140px = -116px, which positions champion left at bracket 56px ✓
                                    left: 'calc(50% - 140px)',
                                    width: '280px', 
                                    maxWidth: '280px',
                                    marginLeft: 0,
                                    marginRight: 0,
                                  } : {})
                                }}
                              >
                                {/* Gold gradient background - completely smooth fade from all sides */}
                                <div
                                  className="absolute pointer-events-none"
                                  style={{
                                    top: '-60px',
                                    left: '-60px',
                                    right: '-60px',
                                    bottom: '-60px',
                                    background: `
                                      radial-gradient(ellipse at center, 
                                        rgba(255, 215, 0, 0.12) 0%, 
                                        rgba(255, 215, 0, 0.08) 20%, 
                                        rgba(255, 215, 0, 0.05) 35%, 
                                        rgba(255, 215, 0, 0.03) 50%, 
                                        rgba(255, 215, 0, 0.015) 65%, 
                                        rgba(255, 215, 0, 0.008) 80%, 
                                        transparent 100%
                                      )
                                    `,
                                  }}
                                />
                                <div className="relative text-center text-sm font-semibold uppercase tracking-wide text-slate-600 mb-3 w-full" style={{ paddingLeft: 0, paddingRight: 0 }}>
                                  CHAMPION
                                </div>
                                <div className="relative flex flex-col items-center gap-2 w-full" style={{ paddingLeft: 0, paddingRight: 0 }}>
                                  <div 
                                    data-champion-flag
                                    className="relative flex items-center justify-center"
                                  >
                                    <TeamFlag
                                      team={champion}
                                      flags={data.flags}
                                      className="h-6 w-9 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
                                    />
                                    {/* Confetti renders directly inside the flag container */}
                                    {champion && (
                                      <ConfettiAnimation
                                        key={champion}
                                        duration={2000}
                                        champion={champion}
                                        funny={funnyRuns !== null}
                                      />
                                    )}
                                  </div>
                                  <div className="text-base font-bold text-slate-900 text-center break-words w-full" style={{ paddingLeft: '8px', paddingRight: '8px', boxSizing: 'border-box' }}>
                                    {champion}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                          {matches.map((match) => {
                            if (!match) {
                              return null;
                            }
                            const center =
                              stage === "Final" && finalCenterOverride !== null
                                ? finalCenterOverride
                                : knockoutCenters[match.id] ?? 0;
                            const top = center - cardHeight / 2;
                            const probabilities = getMatchProbabilityLabels({
                              homeTeam: match.homeResolved ?? match.homeLabel,
                              awayTeam: match.awayResolved ?? match.awayLabel,
                              allowDraw: false,
                              country: match.country,
                            });
                            return (
                              <div
                                key={match.id}
                                ref={(el) => {
                                  if (el) {
                                    knockoutRefs.current.set(match.id, el);
                                  } else {
                                    knockoutRefs.current.delete(match.id);
                                  }
                                }}
                                className="absolute left-0 w-full flex justify-center"
                                style={{ top }}
                              >
                                <div
                                  className="absolute left-0 w-full text-center text-xs font-semibold uppercase tracking-wide text-slate-600"
                                  style={{ top: -labelGap }}
                                >
                                  Final
                                </div>
                                <KnockoutMatchCard
                                  homeTeam={match.homeResolved ?? match.homeLabel}
                                  awayTeam={match.awayResolved ?? match.awayLabel}
                                  winnerSelection={knockoutWinners[String(match.id)] ?? null}
                                  locked={lockResultsActive && lockedKnockoutMatchIds.has(String(match.id))}
                                  onWinnerSelect={(selection) =>
                                    updateKnockoutWinner(match.id, selection)
                                  }
                                  compact={false}
                                  flags={data.flags}
                                  homeWinProb={probabilities.homeWinProb}
                                  awayWinProb={probabilities.awayWinProb}
                                  drawProb={probabilities.drawProb}
                                  isFinal={stage === "Final"}
                                  centerPlaceholders={stage === "Final"}
                                  compactMode={compactKnockout}
                                />
                              </div>
                            );
                          })}
                          {stage === "Final" &&
                            thirdPlaceMatches.length > 0 &&
                            thirdPlaceMatchTop !== null && (
                              <>
                            <div
                              className="absolute left-0 w-full flex justify-center"
                              style={{ top: thirdPlaceMatchTop }}
                            >
                                  {thirdPlaceMatches.map((match) => {
                                    const probabilities = getMatchProbabilityLabels({
                                      homeTeam: match.homeResolved ?? match.homeLabel,
                                      awayTeam: match.awayResolved ?? match.awayLabel,
                                      allowDraw: false,
                                      country: match.country,
                                    });
                                    return (
                                      <div
                                        key={match.id}
                                        ref={(el) => {
                                          if (el) {
                                            knockoutRefs.current.set(match.id, el);
                                          } else {
                                            knockoutRefs.current.delete(match.id);
                                          }
                                        }}
                                      >
                                        <KnockoutMatchCard
                                          homeTeam={match.homeResolved ?? match.homeLabel}
                                          awayTeam={match.awayResolved ?? match.awayLabel}
                                          winnerSelection={knockoutWinners[String(match.id)] ?? null}
                                          locked={lockResultsActive && lockedKnockoutMatchIds.has(String(match.id))}
                                          onWinnerSelect={(selection) =>
                                            updateKnockoutWinner(match.id, selection)
                                          }
                                          compact={false}
                                          flags={data.flags}
                                          homeWinProb={probabilities.homeWinProb}
                                          awayWinProb={probabilities.awayWinProb}
                                          drawProb={probabilities.drawProb}
                                          centerPlaceholders
                                          compactMode={compactKnockout}
                                        />
                                      </div>
                                    );
                                  })}
                                </div>
                                <div
                                  className="absolute left-0 w-full text-center text-xs font-semibold uppercase tracking-wide text-slate-600"
                                  style={{ top: thirdPlaceMatchTop + cardHeight + 12 }}
                                >
                                  Third place
                                </div>
                              </>
                            )}
                        </div>
                            </div>
                          );
                        })}
                    </div>

                    {/* Right block: Bottom half bracket (Semis to R32) - always right-aligned */}
                    <div className="absolute right-0 top-0" style={{ width: compactKnockout ? `${leftBlockWidth}px` : `calc(50% + ${leftBlockWidth / 2}px)` }}>
                      {stageOrder
                        .filter((stage) => stage !== "Final")
                        .map((stage) => {
                          const matches = splitMatchesByStage[stage]?.bottom ?? [];
                          const isRoundOf32 = stage === "Round of 32";
                          const cardHeight = knockoutCardHeight ?? 64;
                          const headerOffset = 20;
                          const columnHeight = knockoutListHeight
                            ? knockoutListHeight + headerOffset
                            : undefined;
                          // Map stages to logical positions (same as left block but mirrored)
                          const rightPositions: Record<string, number> = {
                            'Round of 32': 1,       // At right edge
                            'Round of 16': 2,       // One step in
                            'Quarterfinal': 2.5,    // Interleaved
                            'Semifinal': semifinalPosition, // Slightly further from center
                          };
                          const pos = rightPositions[stage];
                          if (pos === undefined) {
                            return null;
                          }
                          // Convert logical position to pixels from right edge
                          const posFromRight = (pos - 1) * (baseColumnWidth + baseGap);
                          return (
                            <div
                              key={`bottom-${stage}`}
                              className="absolute top-0"
                              style={{
                                right: posFromRight,
                                width: `${baseColumnWidth}px`,
                                height: columnHeight,
                              }}
                            >
                            <div
                              className={cn(
                                "relative",
                                isRoundOf32 ? "flex flex-col gap-4 pt-2" : "pt-4"
                              )}
                              style={
                                !isRoundOf32 && knockoutListHeight
                                  ? {
                                      minHeight: `${knockoutListHeight}px`,
                                    }
                                  : undefined
                              }
                            >
                              {matches.map((match) => {
                            if (!match) {
                              return null;
                            }
                            const handleRoundOf32Click = isRoundOf32
                              ? () => logRoundOf32Match(match)
                              : undefined;
                            const center = knockoutCenters[match.id] ?? 0;
                            const top = isRoundOf32 ? undefined : center - cardHeight / 2;
                            const probabilities = getMatchProbabilityLabels({
                              homeTeam: match.homeResolved ?? match.homeLabel,
                              awayTeam: match.awayResolved ?? match.awayLabel,
                              allowDraw: false,
                              country: match.country,
                            });
                            return (
                              <div
                                key={match.id}
                                ref={(el) => {
                                  if (el) {
                                    knockoutRefs.current.set(match.id, el);
                                  } else {
                                    knockoutRefs.current.delete(match.id);
                                  }
                                }}
                                className={cn(
                                  "w-full flex justify-center",
                                  isRoundOf32 ? "relative" : "absolute left-0"
                                )}
                                style={top !== undefined ? { top } : undefined}
                                onClick={handleRoundOf32Click}
                              >
                                <KnockoutMatchCard
                                  homeTeam={match.homeResolved ?? match.homeLabel}
                                  awayTeam={match.awayResolved ?? match.awayLabel}
                                  winnerSelection={knockoutWinners[String(match.id)] ?? null}
                                  locked={lockResultsActive && lockedKnockoutMatchIds.has(String(match.id))}
                                  onWinnerSelect={(selection) =>
                                    updateKnockoutWinner(match.id, selection)
                                  }
                                  compact={isRoundOf32}
                                  flags={data.flags}
                                  homeWinProb={probabilities.homeWinProb}
                                  awayWinProb={probabilities.awayWinProb}
                                  drawProb={probabilities.drawProb}
                                  isFinal={false}
                                  mirrored
                                  compactMode={compactKnockout}
                                />
                              </div>
                            );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
                {compactKnockout && (
                  <div className="pointer-events-none mt-0.5 flex w-full justify-center">
                    <div className="pointer-events-auto flex flex-col items-center gap-2">
                      <div className="flex items-center gap-1.5 sm:gap-2">
                        <LoadingButton
                          loading={Boolean(loadingKeys.tournament)}
                          disabled={!canAutopredictTournament}
                          onClick={() => runAutopredictWithDelay("tournament", handleAutopredict)}
                          className={cn(
                            "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                            canAutopredictTournament
                              ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                              : "text-slate-500"
                          )}
                        >
                          Auto-predict tournament
                        </LoadingButton>
                        <button
                          type="button"
                          disabled={!canResetTournament}
                          onClick={handleResetAll}
                          className={cn(
                            "rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide ring-1 ring-slate-200",
                            canResetTournament
                              ? "text-slate-600 hover:bg-slate-100 hover:text-slate-700"
                              : "text-slate-500 cursor-default"
                          )}
                        >
                          Reset tournament
                        </button>
                      </div>
                      <div className="flex min-h-9 items-center">
                        {isTournamentComplete && (
                          <button
                            type="button"
                            onClick={() => {
                              if (!shareLink) {
                                setShareStatus("error");
                                return;
                              }
                              if (navigator?.clipboard?.writeText) {
                                navigator.clipboard
                                  .writeText(shareLink)
                                  .then(() => setShareStatus("copied"))
                                  .catch(() => setShareStatus("error"));
                                return;
                              }
                              const ok = window.prompt("Copy link to share", shareLink);
                              setShareStatus(ok ? "copied" : "error");
                            }}
                            className="inline-flex items-center gap-2 rounded-full bg-white/95 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-700 shadow-sm ring-1 ring-slate-200 hover:bg-white"
                          >
                            <svg
                              aria-hidden="true"
                              viewBox="0 0 24 24"
                              className="h-4 w-4 text-slate-600"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <path d="M10 13a5 5 0 0 0 7.07 0l2.83-2.83a5 5 0 0 0-7.07-7.07L10.5 5.5" />
                              <path d="M14 11a5 5 0 0 0-7.07 0L4.1 13.83a5 5 0 0 0 7.07 7.07L13.5 18.5" />
                            </svg>
                            {shareStatus === "copied" ? "Link copied" : "Share prediction"}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
          </section>
        </>
      )}

    </div>
  );
}
