"use client";

import * as React from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";
import { FLAG_COLORS } from "@/lib/flag-colors";
import type {
  GroupDefinition,
  GroupMatch,
  KnockoutMatch,
  QualifierMatch,
  RoundOf32Combos,
  WinProbabilities,
  WorldCupPredictorData,
} from "@/lib/world-cup-predictor";

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
}> = ({ delay, duration, color }) => {
  const angle = React.useRef(Math.random() * 360);
  const distance = React.useRef(200 + Math.random() * 300);
  const rotation = React.useRef(Math.random() * 720 - 360);
  const size = React.useRef(8 + Math.random() * 6);
  const isCircle = React.useRef(Math.random() > 0.5);

  const x = Math.cos((angle.current * Math.PI) / 180) * distance.current;
  const y = Math.sin((angle.current * Math.PI) / 180) * distance.current;
  const finalRotation = rotation.current;

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
          transform: translate(-50%, -50%) translate(0, 0) rotate(0deg);
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
  }, [x, y, finalRotation]);

  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: '50%',
        top: '50%',
        width: `${size.current}px`,
        height: `${size.current}px`,
        backgroundColor: color,
        borderRadius: isCircle.current ? '50%' : '0%',
        animation: `${animationId.current} ${duration}ms ease-out ${delay}ms forwards`,
        opacity: 1,
      }}
    />
  );
};

// Confetti animation component - renders particles relative to its container
const ConfettiAnimation: React.FC<{ duration: number; champion: string }> = ({ duration, champion }) => {
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
        />
      ))}
    </>
  );
};

const HOST_TEAM_COUNTRIES: Record<string, string> = {
  USA: "United States",
  "United States": "United States",
  Canada: "Canada",
  Mexico: "Mexico",
};
const HOST_TEAMS = new Set(["USA", "Canada", "Mexico"]);
const TIEBREAK_TOOLTIP =
  "Table order has been chosen randomly but would be determined by Fair Play Points in reality.";

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
  className?: string;
  children: React.ReactNode;
};

const LoadingButton: React.FC<LoadingButtonProps> = ({
  loading,
  onClick,
  className,
  children,
}) => {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={loading}
      aria-busy={loading}
      className={cn(
        "relative overflow-hidden transition-colors",
        loading && "cursor-wait",
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
    .replace(/^Bosnia and Herzegovina$/i, "Bosnia and Herz.")
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
  // Always use 1dp if forced, or if value is very small or very large
  if (forceDecimal || percent < 0.5 || percent >= 99.5) {
    // If value would round to 0.0, display "<0.1"
    if (percent > 0 && percent < 0.05) {
      return "<0.1%";
    }
    const rounded = Number(percent.toFixed(1));
    const capped = Math.min(rounded, 99.9);
    return `${capped.toFixed(1)}%`;
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
  // If value is very small (from "<0.1%" parsing), display as "<0.1"
  if (value > 0 && value < 0.1) {
    return "<0.1";
  }
  // If value has decimal part, display with 1dp
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
}) {
  const { neutral, advantage } = resolveMatchNeutrality({
    homeTeam,
    awayTeam,
    country,
    neutralOverride,
  });
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
  entry: {
    p_home?: number;
    p_draw?: number;
    p_away?: number;
    p_home_pens?: number;
    p_away_pens?: number;
  } | undefined,
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
  if (!resolved?.entry?.score_matrix) {
    return null;
  }
  return resolved.flipped
    ? transposeScoreMatrix(resolved.entry.score_matrix)
    : resolved.entry.score_matrix;
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
        return { home: i, away: j };
      }
    }
  }
  return { home: 0, away: 0 };
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
        <Image
          src={flagPath}
          alt={`${team} flag`}
          fill
          className="object-cover"
          sizes="24px"
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
  const displayName =
    formatted === "Bosnia and Herzegovina" ? "Bosnia and Herz." : formatted;
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
    ? hasScore && score.home !== score.away
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

  const [isDrawHovered, setIsDrawHovered] = React.useState(false);

  if (orientation === "horizontal" && showScore) {
    const isScoreSet = score.home !== null && score.away !== null;
    const segments = normalizeProbabilitySegments({
      home: parseProbabilityLabel(homeProb),
      draw: allowDraw ? parseProbabilityLabel(drawProb ?? undefined) : null,
      away: parseProbabilityLabel(awayProb),
    });
    const formattedHome = formatDisplayLabel(homeTeam);
    const formattedAway = formatDisplayLabel(awayTeam);
    const displayHome =
      formattedHome === "Bosnia and Herzegovina" ? "Bosnia and Herz." : formattedHome;
    const displayAway =
      formattedAway === "Bosnia and Herzegovina" ? "Bosnia and Herz." : formattedAway;

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
        setScores(2, 1);
        return;
      }
      setScores(1, 2);
    };

    const handleDrawSelect = () => {
      if (!isPickableMatch || !allowDraw) {
        return;
      }
      if (hasScore && isDraw) {
        setScores(null, null);
        return;
      }
      setScores(1, 1);
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
              "w-8 h-7 rounded-md text-center text-sm font-semibold tabular-nums focus:outline-none focus:ring-2 focus:ring-blue-400 appearance-none [-moz-appearance:textfield] [-webkit-appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none transition-colors",
              !isScoreSet && "bg-slate-100 text-slate-400 placeholder:text-slate-400",
              isScoreSet && isWin && "bg-transparent text-blue-700",
              isScoreSet && isDraw && "bg-transparent text-blue-700",
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
            "group flex items-center gap-3 px-1.5 py-2 transition-all duration-200 w-full relative",
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
                  ? "bg-[linear-gradient(to_right,rgb(219,234,254)_0%,rgb(219,234,254)_50%,transparent_100%)]"
                  : "bg-[linear-gradient(to_left,rgb(219,234,254)_0%,rgb(219,234,254)_50%,transparent_100%)]"
              )}
            />
          )}
          {side === "away" && (
            <TeamFlag
              team={team}
              flags={flags}
              className="h-4 w-6 flex-shrink-0 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] relative z-10"
            />
          )}
          <span
            className={cn(
              "min-w-0 truncate text-sm leading-5 relative z-10",
              side === "home" && "text-right",
              !isPickableMatch && "text-slate-400",
              !isScoreSet && isPickableMatch && "font-medium text-slate-900",
              isScoreSet && isWin && "font-semibold text-slate-900",
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
              className="h-4 w-6 flex-shrink-0 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] relative z-10"
            />
          )}
        </button>
      );
    };

    // Calculate gradient position for blue highlight (aligned with knockouts: blue-100 at 50% opacity)
    const getGradientStyle = () => {
      if (!isScoreSet) return {};
      
      if (isDraw) {
        // For draws: gradient spans from home score to away score
        // Scores are roughly centered, so gradient should span the score area
        // Assuming score area is roughly 20% of width (scores + draw button)
        return {
          background: `linear-gradient(to right, 
            transparent 0%, 
            transparent 38%, 
            rgb(219, 234, 254) 44%, 
            rgb(219, 234, 254) 47%, 
            rgb(219, 234, 254) 53%, 
            rgb(219, 234, 254) 56%, 
            transparent 58%, 
            transparent 100%)`
        };
      } else if (homeIsWinner) {
        // For home wins: gradient starts from left edge, fades to white around home score (~40% from left)
        return {
          background: `linear-gradient(to right, 
            rgb(219, 234, 254) 0%, 
            rgb(219, 234, 254) 35%, 
            rgb(219, 234, 254) 38%, 
            rgba(255, 255, 255, 0) 42%, 
            transparent 45%, 
            transparent 100%)`
        };
      } else if (awayIsWinner) {
        // For away wins: gradient starts from right edge, fades to white around away score (~60% from left)
        return {
          background: `linear-gradient(to right, 
            transparent 0%, 
            transparent 55%, 
            rgba(255, 255, 255, 0) 58%, 
            rgb(219, 234, 254) 62%, 
            rgb(219, 234, 254) 65%, 
            rgb(219, 234, 254) 100%)`
        };
      }
      return {};
    };

    return (
      <div
        className={cn(
          "relative overflow-hidden rounded-xl shadow-sm transition-shadow hover:shadow",
          isPickableMatch && !isScoreSet && "bg-white ring-2 ring-[#ffb4a1]",
          isScoreSet && "bg-white ring-1 ring-slate-400",
          !isScoreSet && !isPickableMatch && "bg-white ring-1 ring-slate-200"
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
                rgb(219, 234, 254) 44%, 
                rgb(219, 234, 254) 56%, 
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
          <div className="relative flex items-center gap-1 px-1.5">
            {renderScoreInput("home", homeInputRef)}
            <button
              type="button"
              onClick={handleDrawSelect}
              disabled={!allowDraw || !isPickableMatch}
              onMouseEnter={() => setIsDrawHovered(true)}
              onMouseLeave={() => setIsDrawHovered(false)}
              className={cn(
                "flex flex-col items-center justify-center gap-1 px-2 py-1 rounded-md transition-colors",
                allowDraw && isPickableMatch && "cursor-pointer",
                !isPickableMatch && "cursor-default"
              )}
            >
              <div className="flex h-1 w-16 overflow-hidden rounded-full bg-slate-200">
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
              <div className="flex w-16 justify-between text-xs leading-none tabular-nums text-slate-600">
                <span>{segments ? formatSegmentDisplay(segments.home) : "--"}</span>
                <span>{segments ? formatSegmentDisplay(segments.draw) : "--"}</span>
                <span>{segments ? formatSegmentDisplay(segments.away) : "--"}</span>
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
        setScores(1, 1);
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
  cardWidthClass = "w-[192px]",
  containerRef,
  homeRowRef,
  awayRowRef,
  mirrored,
  compactMode,
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
}) {
  const placeholderHome = !isConcreteTeam(homeTeam);
  const placeholderAway = !isConcreteTeam(awayTeam);
  const isPickableMatch = !placeholderHome && !placeholderAway;
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
  const paddedRow = compact ? "py-1.5" : "py-0.5";
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
      <span className={cn("inline-flex h-[20px] max-w-full items-center truncate rounded-md bg-slate-50 px-2 text-[12px] leading-[20px] text-slate-500 ring-1 ring-slate-200", textAlign, centerPlaceholders && "justify-center")}>
        {formatDisplayLabel(team)}
      </span>
    );
  }
  return (
      <span
        className={cn(
          "block min-w-0 truncate text-sm leading-[20px]",
          textAlign,
          !isResolved && "font-medium text-slate-900",
          isResolved && isWinner && "font-semibold text-slate-900",
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
    const normalGradient = "bg-[linear-gradient(90deg,transparent_0%,rgb(219,234,254)_10%,rgb(219,234,254)_100%)]";
    const mirroredGradient = "bg-[linear-gradient(270deg,transparent_0%,rgb(219,234,254)_10%,rgb(219,234,254)_100%)]";
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
          centerPlaceholders && isLockedMatch ? "px-2 justify-center gap-0" : "gap-2",
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
                ? "bg-[linear-gradient(270deg,transparent_0%,rgb(219,234,254)_10%,rgb(219,234,254)_100%)]"
                : "bg-[linear-gradient(90deg,transparent_0%,rgb(219,234,254)_10%,rgb(219,234,254)_100%)]"
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
                className="h-4 w-6 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
              />
            )}
          </>
        ) : (
          <>
            {!isPlaceholder && (
              <TeamFlag
                team={team}
                flags={flags}
                className="h-4 w-6 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
              />
            )}
            <div className={cn("relative flex min-w-0 items-center", isPlaceholder && centerPlaceholders ? "flex-1 justify-center" : "flex-1")}>
              <span
                className={cn(
                  "absolute left-0 top-0 h-full w-1 rounded-full",
                  isResolved && isWinner
                    ? isChampionRow
                      ? "bg-amber-300"
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
        "flex h-[72px] w-7 flex-col items-center justify-center px-4 py-1 pointer-events-none",
        hideProbabilities && "invisible",
        hasSelection && "opacity-55"
      )}
      aria-hidden={hideProbabilities}
    >
      <span className="text-xs tabular-nums text-slate-600">
        {segments ? formatSegmentDisplay(segments.home) : "--"}
      </span>
      <div className="h-6 w-2 overflow-hidden rounded-full bg-slate-200/70">
        <div className="flex h-full flex-col">
          <div
            className="w-full bg-emerald-300/70"
            style={{ height: `${segments?.home ?? 0}%` }}
          />
          {showDraw && (
            <div
              className="w-full bg-slate-300/70"
              style={{ height: `${segments?.draw ?? 0}%` }}
            />
          )}
          <div
            className="w-full bg-rose-300/70"
            style={{ height: `${segments?.away ?? 0}%` }}
          />
        </div>
      </div>
      <span className="text-xs tabular-nums text-slate-600">
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
            "flex items-center justify-center p-1",
            isResolved && isWinner && !isChampionRow && "bg-blue-200",
            isChampionRow && "bg-amber-200",
            isPickableMatch ? "cursor-pointer" : "cursor-default"
          )}
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
            ? "bg-white ring-2 ring-[#ffb4a1]"
            : hasSelection
              ? "bg-white ring-1 ring-slate-400"
              : "bg-white ring-1 ring-slate-200"
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
          ? "bg-white ring-2 ring-[#ffb4a1]"
          : hasSelection
            ? "bg-white ring-1 ring-slate-400"
            : "bg-white ring-1 ring-slate-200"
      )}
    >
      <div className="flex h-[72px]">
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
  showTitle?: boolean;
  embedded?: boolean;
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
  const semisKey = React.useMemo(
    () => semis.map((match) => String(match.id)).join("|"),
    [semis]
  );
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const bracketRef = React.useRef<HTMLDivElement | null>(null);
  const matchRefs = React.useRef(new Map<string | number, HTMLDivElement>());
  const matchHomeRefs = React.useRef(new Map<string | number, HTMLDivElement>());
  const matchAwayRefs = React.useRef(new Map<string | number, HTMLDivElement>());
  const [paths, setPaths] = React.useState<string[]>([]);
  const [semisOffset, setSemisOffset] = React.useState(0);

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

  const content = (
    <div className="flex w-full flex-col gap-4">
        <div
          className={cn(
            "flex items-center gap-3",
            showTitle ? "justify-between" : "justify-start flex-wrap mb-4"
          )}
        >
        {showTitle && (
          <h3 className="text-sm font-semibold text-slate-900">{path}</h3>
        )}
        <div className="flex items-center gap-2 text-xs">
          <LoadingButton
            loading={autoPredictLoading}
            onClick={() => onAutoPredict(path)}
            className="rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
          >
            Auto-predict
          </LoadingButton>
          <button
            type="button"
            onClick={() => onReset(path)}
            className="rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
          >
            Reset
          </button>
        </div>
      </div>
      <div ref={bracketRef} className="relative w-full">
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
        <div className="relative z-10 grid w-full grid-cols-[max-content_max-content] gap-4">
          <div
            className="grid w-fit grid-rows-[72px_72px] gap-4"
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
              <div className="h-[72px]" />
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
              <div className="h-[72px]" />
            )}
          </div>
          <div className="grid w-fit grid-rows-[72px_auto] gap-4">
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
                <div className="rounded-lg px-3 py-2 bg-[radial-gradient(ellipse_at_center,rgba(219,234,254,0.7)_0%,rgba(219,234,254,0.35)_45%,rgba(219,234,254,0.15)_65%,transparent_100%)]">
                  <div className="flex flex-col items-center gap-2 text-slate-600">
                    <div className="text-[11px] font-semibold uppercase tracking-wide text-blue-700">
                      Qualified
                    </div>
                    <div className="flex items-center gap-2">
                      <TeamFlag
                        team={qualifiedTeam}
                        flags={flags}
                        className="h-5 w-7 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
                      />
                      <span className="text-[15px] font-semibold text-slate-900">
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
      className="relative flex w-full min-w-0 flex-col overflow-hidden rounded-xl bg-slate-50 ring-1 ring-slate-200 p-4"
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
  const [rowPositions, setRowPositions] = React.useState<number[]>([]);

  React.useEffect(() => {
    if (!tbodyRef.current) return;
    const rows = tbodyRef.current.querySelectorAll('tr');
    const positions: number[] = [];
    let currentTop = 0;
    rows.forEach((row) => {
      positions.push(currentTop);
      currentTop += row.getBoundingClientRect().height;
    });
    setRowPositions(positions);
  }, [rows]);

  // Get header height
  const headerHeight = 40; // Approximate, could be measured if needed

  return (
    <div className="w-full rounded-xl bg-white ring-1 ring-slate-200 shadow-sm overflow-hidden relative">
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
        <table className="w-full table-fixed text-sm">
          <colgroup>
            <col style={{ width: "40px" }} />
            <col />
            <col style={{ width: "36px" }} />
            <col style={{ width: "32px" }} />
            <col style={{ width: "32px" }} />
            <col style={{ width: "32px" }} />
            <col className="hidden lg:table-column" style={{ width: "36px" }} />
            <col className="hidden lg:table-column" style={{ width: "36px" }} />
            <col style={{ width: "36px" }} />
            <col style={{ width: "44px" }} />
          </colgroup>
          <thead className="bg-slate-200 border-b border-slate-200">
            <tr>
              <th className="px-2 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                Pos
              </th>
              <th className="px-2 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                Team
              </th>
              <th className="px-2 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                Pld
              </th>
              <th className="px-2 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                W
              </th>
              <th className="px-2 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                D
              </th>
              <th className="px-2 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                L
              </th>
              <th className="hidden lg:table-cell px-1 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                GF
              </th>
              <th className="hidden lg:table-cell px-1 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                GA
              </th>
              <th className="px-1 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
                GD
              </th>
              <th className="px-2 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wide text-slate-600">
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
                  <td className="px-2 py-2.5 text-center text-sm tabular-nums text-slate-600">
                    {row.position}
                  </td>
                  <td className="px-2 py-2.5">
                    <div className="flex min-w-0 items-center gap-2">
                      <TeamFlag
                        team={row.team}
                        flags={flags}
                        className="h-4 w-6 rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]"
                      />
                      <span className="min-w-0 truncate text-sm font-medium text-slate-900">
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
                  <td className="px-2 py-2.5 text-center text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.played}
                  </td>
                  <td className="px-2 py-2.5 text-center text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.wins}
                  </td>
                  <td className="px-2 py-2.5 text-center text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.draws}
                  </td>
                  <td className="px-2 py-2.5 text-center text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.losses}
                  </td>
                  <td className="hidden lg:table-cell px-1 py-2.5 text-center text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.gf}
                  </td>
                  <td className="hidden lg:table-cell px-1 py-2.5 text-center text-sm tabular-nums text-slate-700 whitespace-nowrap">
                    {row.ga}
                  </td>
                  <td className="px-1 py-2.5 text-center text-sm font-medium tabular-nums text-slate-700 whitespace-nowrap">
                    {row.gd > 0 ? `+${row.gd}` : row.gd}
                  </td>
                  <td className="px-2 py-2.5 text-center text-sm font-semibold tabular-nums text-slate-900 whitespace-nowrap">
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
  updateGroupScore: (id: string | number, side: "home" | "away", value: number | null) => void;
  updateGroupScorePair: (id: string | number, home: number | null, away: number | null) => void;
  getMatchProbabilityLabels: (params: {
    homeTeam: string;
    awayTeam: string;
    allowDraw: boolean;
    country?: string | null;
    neutralOverride?: boolean | null;
  }) => MatchProbabilityLabels;
  loadingKeys: Record<string, boolean>;
  runAutopredictWithDelay: (key: string, action: () => void) => void;
  handleGroupAutopredict: (groupId: string) => void;
  handleGroupReset: (groupId: string) => void;
  groupCompletion: Record<string, boolean>;
  qualifiedThirdGroups: Set<string>;
  allGroupMatchesComplete: boolean;
  flags: Record<string, string | null>;
  isTabbed: boolean;
};

function GroupStageCards({
  groupTables,
  resolvedGroupMatches,
  groupScores,
  updateGroupScore,
  updateGroupScorePair,
  getMatchProbabilityLabels,
  loadingKeys,
  runAutopredictWithDelay,
  handleGroupAutopredict,
  handleGroupReset,
  groupCompletion,
  qualifiedThirdGroups,
  allGroupMatchesComplete,
  flags,
  isTabbed,
}: GroupStageCardsProps) {
  const [activeGroupId, setActiveGroupId] = React.useState<string>(
    groupTables[0]?.group.id ?? ""
  );

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
    return (
      <>
        <div
          className={cn(
            "flex items-center gap-3",
            showTitle ? "justify-between" : "justify-start flex-wrap mb-4"
          )}
        >
          {showTitle && (
            <h3 className="text-lg font-semibold text-slate-900">
              Group {entry.group.id}
            </h3>
          )}
          <div className="flex items-center gap-2 text-xs">
            <LoadingButton
              loading={Boolean(loadingKeys[`group:${entry.group.id}`])}
              onClick={() =>
                runAutopredictWithDelay(
                  `group:${entry.group.id}`,
                  () => handleGroupAutopredict(entry.group.id)
                )
              }
              className="rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
            >
              Auto-predict
            </LoadingButton>
            <button
              type="button"
              onClick={() => handleGroupReset(entry.group.id)}
              className="rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
            >
              Reset
            </button>
          </div>
        </div>
        <div className="flex flex-col gap-4">
          <div className="flex w-full flex-col gap-3 px-0.5">
            {matches.map((match) => {
              const probabilities = getMatchProbabilityLabels({
                homeTeam: match.homeTeam,
                awayTeam: match.awayTeam,
                allowDraw: true,
                country: match.country,
              });
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
                  homeWinProb={probabilities.homeWinProb}
                  awayWinProb={probabilities.awayWinProb}
                  drawProb={probabilities.drawProb}
                  showDivider={false}
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
        className="relative flex w-full min-w-0 flex-col gap-4 overflow-hidden rounded-xl bg-slate-50 ring-1 ring-slate-200 p-4"
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
      <div className="relative flex w-full min-w-0 flex-col overflow-hidden rounded-xl bg-slate-50 ring-1 ring-slate-200 p-4">
        <div className="border-b border-slate-200 pb-3">
          <div
            role="tablist"
            aria-label="Group tabs"
            className="flex w-full min-w-0 items-center gap-2 overflow-x-auto pb-1"
          >
            {groupTables.map((entry) => {
              const isActive = entry.group.id === activeGroupId;
              return (
                <button
                  key={entry.group.id}
                  type="button"
                  role="tab"
                  aria-selected={isActive}
                  aria-controls={`group-panel-${entry.group.id}`}
                  className={cn(
                    "rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-wide transition-colors",
                    isActive
                      ? "border-slate-900 bg-slate-900 text-white"
                      : "border-slate-200 bg-white text-slate-600 hover:bg-slate-100"
                  )}
                  onClick={() => setActiveGroupId(entry.group.id)}
                >
                  Group {entry.group.id}
                </button>
              );
            })}
          </div>
        </div>
        <div id={`group-panel-${activeEntry.group.id}`} role="tabpanel" className="pt-4">
          {renderGroupContent(activeEntry, false)}
        </div>
      </div>
    );
  }

  return <>{groupTables.map((entry) => renderGroupCard(entry, true))}</>;
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

export function WorldCupPredictorPage({ data }: { data: WorldCupPredictorData }) {
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
  const pendingGroupsAfterQualifiers = React.useRef(false);
  const groupCardsContainerRef = React.useRef<HTMLDivElement | null>(null);
  const isGroupTabbed = true;
  const [activeQualifierPath, setActiveQualifierPath] = React.useState<string | null>(
    null
  );

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
        probabilities: data.winProbabilities,
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
    [data.winProbabilities]
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
    ]
  );

  const updateKnockoutWinner = React.useCallback(
    (id: string | number, selection: WinnerSelection) => {
      const key = String(id);
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
    [knockoutDependents]
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
        logChanges?: boolean;
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
      let changedMatchDetails: Array<{
        matchId: string;
        stage: string;
        rawLabels: { homeLabel: string; awayLabel: string } | null;
        winnerGroupKey: string | null;
        assignedGroup: string | null;
        nextAssignedGroup: string | null;
        previousThirdTeam: string | null;
        nextThirdTeam: string | null;
        before: { home: string; away: string } | undefined;
        after: { home: string; away: string } | undefined;
      }> = [];
      if (options?.logChanges) {
        changedMatchDetails = Array.from(changedMatches).map((matchId) => {
          const before = previousLabels.get(matchId);
          const after = nextLabels.get(matchId);
          const match = data.knockoutMatches.find(
            (entry) => String(entry.id) === matchId
          );
          const winnerGroupFromLabel = (label?: string) => {
            if (!label || !label.startsWith("Winner Group ")) {
              return null;
            }
            return `1${label.replace("Winner Group ", "").trim()}`;
          };
          const winnerGroupKey =
            winnerGroupFromLabel(match?.homeLabel) ??
            winnerGroupFromLabel(match?.awayLabel);
          const assignedGroup =
            winnerGroupKey && previousContext.thirdPlaceAssignments
              ? previousContext.thirdPlaceAssignments[winnerGroupKey]
              : null;
          const nextAssignedGroup =
            winnerGroupKey && nextContext.thirdPlaceAssignments
              ? nextContext.thirdPlaceAssignments[winnerGroupKey]
              : null;
          const previousThirdTeam = assignedGroup
            ? previousContext.thirdPlaceByGroup[assignedGroup]
            : null;
          const nextThirdTeam = nextAssignedGroup
            ? nextContext.thirdPlaceByGroup[nextAssignedGroup]
            : null;
          return {
            matchId,
            stage: matchStageById[Number(matchId)],
            rawLabels: match
              ? { homeLabel: match.homeLabel, awayLabel: match.awayLabel }
              : null,
            winnerGroupKey,
            assignedGroup,
            nextAssignedGroup,
            previousThirdTeam,
            nextThirdTeam,
            before,
            after,
          };
        });
      }
      const { next, clearedIds } = clearKnockoutSelectionsByMatchIds(
        current,
        changedMatches
      );
      if (options?.logChanges) {
        const describeSideChange = (
          label: string | undefined,
          beforeTeam: string | undefined,
          afterTeam: string | undefined
        ) => {
          if (!beforeTeam || !afterTeam || beforeTeam === afterTeam) {
            return null;
          }
          if (label) {
            return `${label} changed (${beforeTeam} -> ${afterTeam})`;
          }
          return `participants changed (${beforeTeam} -> ${afterTeam})`;
        };
        const qualifiedNext = new Set(nextContext.qualifiedThirdGroups ?? []);
        const thirdPlaceNote = (groupId: string) =>
          qualifiedNext.has(groupId) ? ` and ${groupId} is in top-8 thirds` : "";
        changedMatchDetails.forEach((detail) => {
          if (!detail.before || !detail.after) {
            return;
          }
          let reason = "";
          if (
            detail.assignedGroup &&
            detail.nextAssignedGroup &&
            detail.assignedGroup === detail.nextAssignedGroup &&
            detail.previousThirdTeam !== detail.nextThirdTeam
          ) {
            reason = `Match ${detail.matchId} cleared because Group ${detail.assignedGroup} third-place team changed (${detail.previousThirdTeam} -> ${detail.nextThirdTeam})${thirdPlaceNote(
              detail.assignedGroup
            )}.`;
          } else if (
            detail.assignedGroup &&
            detail.nextAssignedGroup &&
            detail.assignedGroup !== detail.nextAssignedGroup
          ) {
            reason = `Match ${detail.matchId} cleared because third-place assignment changed from Group ${detail.assignedGroup} to Group ${detail.nextAssignedGroup} (combo ${previousContext.comboKey} -> ${nextContext.comboKey}).`;
          } else {
            const parts = [
              describeSideChange(
                detail.rawLabels?.homeLabel,
                detail.before.home,
                detail.after.home
              ),
              describeSideChange(
                detail.rawLabels?.awayLabel,
                detail.before.away,
                detail.after.away
              ),
            ].filter(Boolean);
            reason = `Match ${detail.matchId} cleared because ${parts.length ? parts.join(" and ") : "participants changed"}.`;
          }
          console.log(`[predictor] ${reason}`);
        });
      }
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
          nextScores,
          { logChanges: true }
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

  const updateGroupScore = React.useCallback(
    (id: string | number, side: "home" | "away", value: number | null) => {
      let changed = false;
      let nextScores: Record<string, MatchScore> | null = null;
      const key = String(id);
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
    [clearKnockoutOnGroupChange]
  );

  const updateGroupScorePair = React.useCallback(
    (id: string | number, home: number | null, away: number | null) => {
      let changed = false;
      let nextScores: Record<string, MatchScore> | null = null;
      const key = String(id);
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
    [clearKnockoutOnGroupChange]
  );

  const resolvedGroups = React.useMemo(() => {
    return data.groups.map((group) => ({
      ...group,
      teams: group.teams.map((team) => slotWinners.get(team) ?? team),
    }));
  }, [data.groups, slotWinners]);

  const resolvedGroupMatches = React.useMemo(() => {
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

  const groupTables = React.useMemo(() => {
    return resolvedGroups.map((group) => {
      const matches = groupMatchesFor(group.id, resolvedGroupMatches);
      const { table, ranking } = buildGroupTable(group, matches, groupScores);
      const rows = ranking.map((team) => table[team]).filter(Boolean);
      return { group, ranking, table, rows };
    });
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
      .filter((row): row is GroupTableRow => Boolean(row));
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

  const knockoutState = React.useMemo(() => {
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
    if (mismatches.length > 0) {
      console.warn(
        `[predictor] reset label mismatch ${JSON.stringify({ mismatches })}`
      );
    }
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
          // In compact mode, subsequent rounds turn earlier (20px), R32→R16 uses fixed distance for true symmetry
          const horizontalDistance = compactKnockout ? 20 : 30;
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
  }, [knockoutEdges, thirdPlaceOffset, finalCenterOverride, knockoutListHeight, matchStageById, knockoutMatchesByStage, splitMatchesByStage, compactKnockout]);

  React.useLayoutEffect(() => {
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
  }, [knockoutMatchesByStage, roundOf32Order, knockoutCardHeight, compactKnockout]);



  React.useLayoutEffect(() => {
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
        const finalOffset = 72; // Distance above/below semis average
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
  }, [knockoutMatchesByStage, knockoutCenters, knockoutContainerRef, compactKnockout, knockoutCardHeight]);

  React.useLayoutEffect(() => {
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
        const cardHeight = 64; // Match card height
        
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
        const finalOffset = 72; // Same offset used for Third Place
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
  }, [knockoutMatchesByStage, knockoutCenters, compactKnockout]);

  const handleAutopredict = React.useCallback(() => {
    let nextQualifierWinners = { ...qualifierWinners };
    let nextAutoQualifierWinners = { ...autoQualifierWinners };
    let nextGroupScores = { ...groupScores };
    let nextAutoGroupScores = { ...autoGroupScores };
    let nextKnockoutWinners = { ...knockoutWinners };
    let nextAutoKnockoutWinners = { ...autoKnockoutWinners };
    const allQualifiersPredicted = qualifierState.matches.every((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return true;
      }
      const key = String(match.id);
      return (qualifierWinners[key] ?? null) !== null;
    });
    const allGroupsPredicted = resolvedGroupMatches.every((match) => {
      const score = groupScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
    const allKnockoutsPredicted = knockoutState.matches.every((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return true;
      }
      const key = String(match.id);
      return (knockoutWinners[key] ?? null) !== null;
    });
    if (allQualifiersPredicted && allGroupsPredicted && allKnockoutsPredicted) {
      nextQualifierWinners = {};
      nextAutoQualifierWinners = {};
      nextGroupScores = {};
      nextAutoGroupScores = {};
      nextKnockoutWinners = {};
      nextAutoKnockoutWinners = {};
    }

    const applyQualifierSelection = (matchId: string, selection: WinnerSelection) => {
      const prevSelection = nextQualifierWinners[matchId] ?? null;
      if (prevSelection === selection) {
        return { changed: false, clearedIds: [] as string[] };
      }
      const updated = { ...nextQualifierWinners, [matchId]: selection };
      const cleared = clearDependentSelections(updated, matchId, qualifierDependents);
      const clearedIds = Object.keys(updated).filter(
        (id) => updated[id] && cleared[id] === null
      );
      nextQualifierWinners = cleared;
      if (selection) {
        nextAutoQualifierWinners[matchId] = true;
      }
      clearedIds.forEach((id) => {
        delete nextAutoQualifierWinners[id];
      });
      return { changed: true, clearedIds };
    };

    const applyKnockoutSelection = (matchId: string, selection: WinnerSelection) => {
      const prevSelection = nextKnockoutWinners[matchId] ?? null;
      if (prevSelection === selection) {
        return { changed: false, clearedIds: [] as string[] };
      }
      const updated = { ...nextKnockoutWinners, [matchId]: selection };
      const cleared = clearDependentSelections(updated, matchId, knockoutDependents);
      const clearedIds = Object.keys(updated).filter(
        (id) => updated[id] && cleared[id] === null
      );
      nextKnockoutWinners = cleared;
      if (selection) {
        nextAutoKnockoutWinners[matchId] = true;
      }
      clearedIds.forEach((id) => {
        delete nextAutoKnockoutWinners[id];
      });
      return { changed: true, clearedIds };
    };

    const maxIterations = 10;
    let iteration = 0;
    let changed = true;

    while (changed && iteration < maxIterations) {
      iteration += 1;
      changed = false;

      const previousSlotWinners = resolveQualifierState(
        data.qualifiers,
        nextQualifierWinners
      ).slotWinners;
      const previousGroupScores = { ...nextGroupScores };

      let qualifierState = resolveQualifierState(
        data.qualifiers,
        nextQualifierWinners
      );
      const changedQualifiers = new Set<string>();
      let qualifierProgress = true;
      let qualifierIterations = 0;

      while (qualifierProgress && qualifierIterations < maxIterations) {
        qualifierProgress = false;
        qualifierIterations += 1;
        qualifierState = resolveQualifierState(
          data.qualifiers,
          nextQualifierWinners
        );
        qualifierState.matches.forEach((match) => {
          const key = String(match.id);
          const isManual =
            nextQualifierWinners[key] && !nextAutoQualifierWinners[key];
          const existingSelection = nextQualifierWinners[key] ?? null;
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
            probabilities: data.winProbabilities,
            homeTeam: match.homeResolved,
            awayTeam: match.awayResolved,
            allowDraw: false,
            neutralOverride: match.neutral,
          });
          const selection = sampleWinner(values);
          if (!selection) {
            return;
          }
          const result = applyQualifierSelection(key, selection);
          if (result.changed) {
            changed = true;
            qualifierProgress = true;
            changedQualifiers.add(key);
          }
        });
      }

      if (changedQualifiers.size > 0) {
        const affectedSlots = new Set<string>();
        changedQualifiers.forEach((matchId) => {
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
            if (cleared.clearedIds.length > 0) {
              changed = true;
            }
          }
        }

        qualifierState = resolveQualifierState(data.qualifiers, nextQualifierWinners);
      }

      const nextSlotWinners = qualifierState.slotWinners;
      const resolvedGroupMatches = data.groupMatches.map((match) => ({
        ...match,
        homeTeam: nextSlotWinners.get(match.homeTeam) ?? match.homeTeam,
        awayTeam: nextSlotWinners.get(match.awayTeam) ?? match.awayTeam,
      }));

      let groupScoresChanged = false;
      resolvedGroupMatches.forEach((match) => {
        const key = String(match.id);
        const existing = nextGroupScores[key];
        const hasScore =
          existing && existing.home !== null && existing.away !== null;
        const isManual = hasScore && !nextAutoGroupScores[key];
        if (isManual || hasScore) {
          return;
        }
        const matrix = resolveMatchScoreMatrix({
          probabilities: data.winProbabilities,
          homeTeam: match.homeTeam,
          awayTeam: match.awayTeam,
          country: match.country,
        });
        if (!matrix) {
          return;
        }
        const sample = sampleScoreMatrix(matrix);
        if (!sample) {
          return;
        }
        nextGroupScores[key] = { home: sample.home, away: sample.away };
        nextAutoGroupScores[key] = true;
        groupScoresChanged = true;
      });

      if (groupScoresChanged || changedQualifiers.size > 0) {
        const clearedForGroups = computeClearedKnockoutSelections(
          nextKnockoutWinners,
          previousGroupScores,
          nextGroupScores,
          {
            previousSlotWinners,
            nextSlotWinners,
          }
        );
        nextKnockoutWinners = clearedForGroups.nextWinners;
        clearedForGroups.clearedIds.forEach((matchId) => {
          delete nextAutoKnockoutWinners[matchId];
        });
        if (clearedForGroups.clearedIds.length > 0) {
          changed = true;
        }
      }

      const nextContext = computeKnockoutContext(
        nextGroupScores,
        nextSlotWinners
      );
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
        const existingSelection = nextKnockoutWinners[key] ?? null;
        const isManual = existingSelection && !nextAutoKnockoutWinners[key];
        if (!isManual && !existingSelection) {
          if (
            !isPlaceholderLabel(homeResolved) &&
            !isPlaceholderLabel(awayResolved)
          ) {
            const values = resolveMatchProbabilities({
              probabilities: data.winProbabilities,
              homeTeam: homeResolved,
              awayTeam: awayResolved,
              allowDraw: false,
              country: match.country,
            });
            const selection = sampleWinner(values);
            if (selection) {
              const result = applyKnockoutSelection(key, selection);
              if (result.changed) {
                changed = true;
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
          nextKnockoutWinners
        );
        if (winner) {
          winners.set(match.id, winner);
          const loser = winner === homeResolved ? awayResolved : homeResolved;
          losers.set(match.id, loser);
        }
      }
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
    autoQualifierWinners,
    clearKnockoutSelectionsByMatchIds,
    computeClearedKnockoutSelections,
    computeKnockoutContext,
    data.groupMatches,
    data.knockoutMatches,
    data.qualifiers,
    data.winProbabilities,
    groupIdsBySlot,
    groupMatchIdsByTeam,
    groupScores,
    knockoutDependents,
    knockoutRootsByGroup,
    knockoutState.matches,
    knockoutWinners,
    matchStageById,
    qualifierDependents,
    qualifierSlotsByMatch,
    qualifierState.matches,
    qualifierWinners,
    resolvedGroupMatches,
    slotWinners,
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
      const matches = groupMatchesFor(groupId, resolvedGroupMatches);
      if (matches.length === 0) {
        return;
      }
      let changed = false;
      const nextScores = { ...groupScores };
      const nextAutoScores = { ...autoGroupScores };
      const allPredicted = matches.every((match) => {
        const key = String(match.id);
        const existing = nextScores[key];
        const hasScore =
          existing && existing.home !== null && existing.away !== null;
        return hasScore;
      });
      if (allPredicted) {
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
      }
      matches.forEach((match) => {
        const key = String(match.id);
        const existing = nextScores[key];
        const hasScore =
          existing && existing.home !== null && existing.away !== null;
        if (hasScore) {
          return;
        }
        const matrix = resolveMatchScoreMatrix({
          probabilities: data.winProbabilities,
          homeTeam: match.homeTeam,
          awayTeam: match.awayTeam,
          country: match.country,
        });
        if (!matrix) {
          return;
        }
        const sample = sampleScoreMatrix(matrix);
        if (!sample) {
          return;
        }
        nextScores[key] = { home: sample.home, away: sample.away };
        nextAutoScores[key] = true;
        changed = true;
      });
      if (!changed) {
        return;
      }
      setGroupScores(nextScores);
      setAutoGroupScores(nextAutoScores);
      clearKnockoutOnGroupChange(nextScores);
    },
    [
      autoGroupScores,
      clearKnockoutOnGroupChange,
      data.winProbabilities,
      groupScores,
      resolvedGroupMatches,
    ]
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
      let nextQualifierWinners = { ...qualifierWinners };
      let nextAutoQualifierWinners = { ...autoQualifierWinners };
      const changedMatchIds = new Set<string>();

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
      const allPredicted = matches.every((match) => {
        const key = String(match.id);
        return (nextQualifierWinners[key] ?? null) !== null;
      });
      if (allPredicted) {
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
              changedMatchIds.add(id);
            }
          });
          if (nextQualifierWinners[key]) {
            changedMatchIds.add(key);
          }
          nextQualifierWinners = cleared;
        });
        changedMatchIds.forEach((matchId) => {
          delete nextAutoQualifierWinners[matchId];
        });
      }

      const applyQualifierSelection = (matchId: string, selection: WinnerSelection) => {
        const prevSelection = nextQualifierWinners[matchId] ?? null;
        if (prevSelection === selection) {
          return false;
        }
        const updated = { ...nextQualifierWinners, [matchId]: selection };
        const cleared = clearDependentSelections(
          updated,
          matchId,
          qualifierDependents
        );
        const clearedIds = Object.keys(updated).filter(
          (id) => updated[id] && cleared[id] === null
        );
        nextQualifierWinners = cleared;
        if (selection) {
          nextAutoQualifierWinners[matchId] = true;
        }
        clearedIds.forEach((id) => {
          delete nextAutoQualifierWinners[id];
          changedMatchIds.add(id);
        });
        changedMatchIds.add(matchId);
        return true;
      };

      const maxIterations = 10;
      let iteration = 0;
      let progress = true;
      while (progress && iteration < maxIterations) {
        iteration += 1;
        progress = false;
        const qualifierStateIter = resolveQualifierState(
          data.qualifiers,
          nextQualifierWinners
        );
        const pathMatches = qualifierStateIter.matches.filter(
          (match) => match.path === path
        );
        if (pathMatches.length === 0) {
          break;
        }
        pathMatches.forEach((match) => {
          const key = String(match.id);
          const isManual =
            nextQualifierWinners[key] && !nextAutoQualifierWinners[key];
          if (isManual || nextQualifierWinners[key]) {
            return;
          }
          if (
            isPlaceholderLabel(match.homeResolved) ||
            isPlaceholderLabel(match.awayResolved)
          ) {
            return;
          }
          const values = resolveMatchProbabilities({
            probabilities: data.winProbabilities,
            homeTeam: match.homeResolved,
            awayTeam: match.awayResolved,
            allowDraw: false,
            neutralOverride: match.neutral,
          });
          const selection = sampleWinner(values);
          if (!selection) {
            return;
          }
          if (applyQualifierSelection(key, selection)) {
            progress = true;
          }
        });
      }
      if (changedMatchIds.size === 0) {
        return;
      }
      const affectedSlots = new Set<string>();
      changedMatchIds.forEach((matchId) => {
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
      data.winProbabilities,
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
    let nextQualifierWinners = { ...qualifierWinners };
    let nextAutoQualifierWinners = { ...autoQualifierWinners };
    let nextGroupScores = { ...groupScores };
    let nextAutoGroupScores = { ...autoGroupScores };
    let nextKnockoutWinners = { ...knockoutWinners };
    let nextAutoKnockoutWinners = { ...autoKnockoutWinners };
    const qualifierStateLocal = resolveQualifierState(
      data.qualifiers,
      nextQualifierWinners
    );
    const allPredicted = qualifierStateLocal.matches.every((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return true;
      }
      const key = String(match.id);
      return (nextQualifierWinners[key] ?? null) !== null;
    });
    if (allPredicted) {
      nextQualifierWinners = {};
      nextAutoQualifierWinners = {};
    }

    const applyQualifierSelection = (matchId: string, selection: WinnerSelection) => {
      const prevSelection = nextQualifierWinners[matchId] ?? null;
      if (prevSelection === selection) {
        return { changed: false, clearedIds: [] as string[] };
      }
      const updated = { ...nextQualifierWinners, [matchId]: selection };
      const cleared = clearDependentSelections(updated, matchId, qualifierDependents);
      const clearedIds = Object.keys(updated).filter(
        (id) => updated[id] && cleared[id] === null
      );
      nextQualifierWinners = cleared;
      if (selection) {
        nextAutoQualifierWinners[matchId] = true;
      }
      clearedIds.forEach((id) => {
        delete nextAutoQualifierWinners[id];
      });
      return { changed: true, clearedIds };
    };

    const maxIterations = 10;
    let iteration = 0;
    let changed = false;
    const changedQualifiers = new Set<string>();

    while (iteration < maxIterations) {
      iteration += 1;
      let qualifierProgress = false;
      const qualifierStateLocal = resolveQualifierState(
        data.qualifiers,
        nextQualifierWinners
      );
      qualifierStateLocal.matches.forEach((match) => {
        const key = String(match.id);
        const isManual =
          nextQualifierWinners[key] && !nextAutoQualifierWinners[key];
        const existingSelection = nextQualifierWinners[key] ?? null;
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
          probabilities: data.winProbabilities,
          homeTeam: match.homeResolved,
          awayTeam: match.awayResolved,
          allowDraw: false,
          neutralOverride: match.neutral,
        });
        const selection = sampleWinner(values);
        if (!selection) {
          return;
        }
        const result = applyQualifierSelection(key, selection);
        if (result.changed) {
          changed = true;
          qualifierProgress = true;
          changedQualifiers.add(key);
        }
      });
      if (!qualifierProgress) {
        break;
      }
    }

    if (changedQualifiers.size > 0) {
      const affectedSlots = new Set<string>();
      changedQualifiers.forEach((matchId) => {
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
    }

    if (!changed && changedQualifiers.size === 0) {
      return;
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
    autoQualifierWinners,
    data.qualifiers,
    data.winProbabilities,
    groupIdsBySlot,
    groupMatchIdsByTeam,
    groupScores,
    knockoutDependents,
    knockoutRootsByGroup,
    knockoutWinners,
    qualifierDependents,
    qualifierSlotsByMatch,
    qualifierWinners,
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
    let changed = false;
    const nextScores = { ...groupScores };
    const nextAutoScores = { ...autoGroupScores };
    const allPredicted = resolvedGroupMatches.every((match) => {
      const score = nextScores[String(match.id)];
      return score && score.home !== null && score.away !== null;
    });
    if (allPredicted) {
      Object.keys(nextScores).forEach((key) => {
        delete nextScores[key];
        changed = true;
      });
      Object.keys(nextAutoScores).forEach((key) => {
        delete nextAutoScores[key];
      });
    }
    resolvedGroupMatches.forEach((match) => {
      const key = String(match.id);
      const existing = nextScores[key];
      const hasScore =
        existing && existing.home !== null && existing.away !== null;
      if (hasScore) {
        return;
      }
      const matrix = resolveMatchScoreMatrix({
        probabilities: data.winProbabilities,
        homeTeam: match.homeTeam,
        awayTeam: match.awayTeam,
        country: match.country,
      });
      if (!matrix) {
        return;
      }
      const sample = sampleScoreMatrix(matrix);
      if (!sample) {
        return;
      }
      nextScores[key] = { home: sample.home, away: sample.away };
      nextAutoScores[key] = true;
      changed = true;
    });
    if (!changed) {
      return;
    }
    setGroupScores(nextScores);
    setAutoGroupScores(nextAutoScores);
    clearKnockoutOnGroupChange(nextScores);
  }, [
    autoGroupScores,
    clearKnockoutOnGroupChange,
    data.winProbabilities,
    groupScores,
    resolvedGroupMatches,
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
    let nextKnockoutWinners = { ...knockoutWinners };
    let nextAutoKnockoutWinners = { ...autoKnockoutWinners };
    const allPredicted = knockoutState.matches.every((match) => {
      if (!isConcreteTeam(match.homeResolved) || !isConcreteTeam(match.awayResolved)) {
        return true;
      }
      const key = String(match.id);
      return (nextKnockoutWinners[key] ?? null) !== null;
    });
    if (allPredicted) {
      nextKnockoutWinners = {};
      nextAutoKnockoutWinners = {};
    }

    const applyKnockoutSelection = (matchId: string, selection: WinnerSelection) => {
      const prevSelection = nextKnockoutWinners[matchId] ?? null;
      if (prevSelection === selection) {
        return false;
      }
      const updated = { ...nextKnockoutWinners, [matchId]: selection };
      const cleared = clearDependentSelections(updated, matchId, knockoutDependents);
      const clearedIds = Object.keys(updated).filter(
        (id) => updated[id] && cleared[id] === null
      );
      nextKnockoutWinners = cleared;
      if (selection) {
        nextAutoKnockoutWinners[matchId] = true;
      }
      clearedIds.forEach((id) => {
        delete nextAutoKnockoutWinners[id];
      });
      return true;
    };

    const context = computeKnockoutContext(groupScores, slotWinners);
    const winners = new Map<number, string>();
    const losers = new Map<number, string>();
    const sorted = [...data.knockoutMatches].sort((a, b) => a.id - b.id);

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

      const existingSelection = nextKnockoutWinners[key] ?? null;
      const isManual = existingSelection && !nextAutoKnockoutWinners[key];
      if (!isManual && !existingSelection) {
        if (!isPlaceholderLabel(homeResolved) && !isPlaceholderLabel(awayResolved)) {
          const values = resolveMatchProbabilities({
            probabilities: data.winProbabilities,
            homeTeam: homeResolved,
            awayTeam: awayResolved,
            allowDraw: false,
            country: match.country,
          });
          const selection = sampleWinner(values);
          if (selection) {
            applyKnockoutSelection(key, selection);
          }
        }
      }

      const winner = resolveWinner(
        match.id,
        homeResolved,
        awayResolved,
        {},
        false,
        nextKnockoutWinners
      );
      if (winner) {
        winners.set(match.id, winner);
        const loser = winner === homeResolved ? awayResolved : homeResolved;
        losers.set(match.id, loser);
      }
    });

    setKnockoutWinners(nextKnockoutWinners);
    setAutoKnockoutWinners(nextAutoKnockoutWinners);
  }, [
    autoKnockoutWinners,
    computeKnockoutContext,
    data.knockoutMatches,
    data.winProbabilities,
    groupScores,
    knockoutDependents,
    knockoutState.matches,
    knockoutWinners,
    matchStageById,
    slotWinners,
  ]);

  const handleSectionKnockoutsReset = React.useCallback(() => {
    if (!Object.keys(knockoutWinners).length) {
      return;
    }
    setKnockoutWinners({});
    setAutoKnockoutWinners({});
  }, [knockoutWinners]);

  const knockoutBaseColumnWidth = compactKnockout ? 48 : 200;
  const knockoutBaseGap = compactKnockout ? 8 : 24;
  const knockoutSfPosition = compactKnockout ? 3 : 2.8;
  const knockoutLeftBlockWidth =
    (knockoutSfPosition - 1) * (knockoutBaseColumnWidth + knockoutBaseGap) +
    knockoutBaseColumnWidth;
  const knockoutMinGapBetweenSFs = compactKnockout
    ? knockoutBaseColumnWidth * 1.5
    : knockoutBaseColumnWidth;
  const knockoutMinBracketWidth =
    knockoutLeftBlockWidth * 2 + knockoutMinGapBetweenSFs;
  const activeQualifierEntry =
    qualifierEntries.find(([path]) => path === activeQualifierPath) ??
    qualifierEntries[0];
  const activeQualifierPathValue =
    activeQualifierPath ?? activeQualifierEntry?.[0] ?? null;
  const qualifierPanelId = (path: string) =>
    `qualifier-panel-${path.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

  return (
    <div className="flex flex-col gap-12">
      <div className="flex flex-wrap items-center justify-start gap-3">
        <LoadingButton
          loading={Boolean(loadingKeys.tournament)}
          onClick={() => runAutopredictWithDelay("tournament", handleAutopredict)}
          className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
        >
          Auto-predict tournament
        </LoadingButton>
        <button
          type="button"
          onClick={handleResetAll}
          className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
        >
          Reset
        </button>
      </div>
      <section className="space-y-6">
        <div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-2xl font-semibold text-ebony">
              Qualifier playoffs
            </h2>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <LoadingButton
              loading={Boolean(loadingKeys["section:qualifiers"])}
              onClick={() =>
                runAutopredictWithDelay(
                  "section:qualifiers",
                  handleSectionQualifiersAutopredict
                )
              }
              className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
            >
              Auto-predict qualifiers
            </LoadingButton>
            <button
              type="button"
              onClick={handleSectionQualifiersReset}
              className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
            >
              Reset
            </button>
          </div>
        </div>
        <div className="space-y-6">
          {isGroupTabbed ? (
            <div className="relative flex w-full min-w-0 flex-col overflow-hidden rounded-xl bg-slate-50 ring-1 ring-slate-200 p-4">
              <div className="border-b border-slate-200 pb-3">
                <div
                  role="tablist"
                  aria-label="Qualifier playoff tabs"
                  className="flex items-center gap-2 overflow-x-auto pb-1"
                >
                  {qualifierEntries.map(([path]) => {
                    const isActive = path === activeQualifierPathValue;
                    return (
                      <button
                        key={path}
                        type="button"
                        role="tab"
                        aria-selected={isActive}
                        aria-controls={qualifierPanelId(path)}
                        className={cn(
                          "rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-wide transition-colors",
                          isActive
                            ? "border-slate-900 bg-slate-900 text-white"
                            : "border-slate-200 bg-white text-slate-600 hover:bg-slate-100"
                        )}
                        onClick={() => setActiveQualifierPath(path)}
                      >
                        {path}
                      </button>
                    );
                  })}
                </div>
              </div>
              {activeQualifierEntry && (
                <div
                  id={qualifierPanelId(activeQualifierEntry[0])}
                  role="tabpanel"
                  className="pt-4"
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
                  />
                </div>
              )}
            </div>
          ) : (
            <div className="grid gap-6 lg:gap-6 grid-cols-[repeat(auto-fit,minmax(432px,1fr))]">
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
      </section>

      <section className="space-y-6">
        <div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-2xl font-semibold text-ebony">Group stage</h2>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <LoadingButton
              loading={Boolean(loadingKeys["section:groups"])}
              onClick={() =>
                runAutopredictWithDelay(
                  "section:groups",
                  handleSectionGroupsAutopredict
                )
              }
              className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
            >
              Auto-predict groups
            </LoadingButton>
            <button
              type="button"
              onClick={handleSectionGroupsReset}
              className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
            >
              Reset
            </button>
          </div>
        </div>
        <div
          ref={groupCardsContainerRef}
          className={cn(
            "grid gap-6",
            thirdPlaceRankingRows.length > 0
              ? "lg:grid-cols-[minmax(0,1fr)_minmax(0,520px)] lg:items-start"
              : "grid-cols-1"
          )}
        >
          <div className="min-w-0 space-y-6">
            <GroupStageCards
              groupTables={groupTables}
              resolvedGroupMatches={resolvedGroupMatches}
              groupScores={groupScores}
              updateGroupScore={updateGroupScore}
              updateGroupScorePair={updateGroupScorePair}
              getMatchProbabilityLabels={getMatchProbabilityLabels}
              loadingKeys={loadingKeys}
              runAutopredictWithDelay={runAutopredictWithDelay}
              handleGroupAutopredict={handleGroupAutopredict}
              handleGroupReset={handleGroupReset}
              groupCompletion={groupCompletion}
              qualifiedThirdGroups={qualifiedThirdGroups}
              allGroupMatchesComplete={allGroupMatchesComplete}
              flags={data.flags}
              isTabbed={isGroupTabbed}
            />
          </div>
          {thirdPlaceRankingRows.length > 0 && (
            <div className="space-y-4 rounded-xl bg-slate-50 ring-1 ring-slate-200 p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900">
                  Ranking of 3rd place teams
                </h3>
              </div>
              <div className="flex flex-wrap justify-center gap-4 lg:flex-nowrap lg:items-start lg:justify-between">
                <div className="flex w-full max-w-[520px] lg:max-w-none lg:flex-1">
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
            </div>
          )}
        </div>
      </section>

      <section className="space-y-6">
        <div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-2xl font-semibold text-ebony">Knockout stage</h2>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-slate-600">
                Compact mode
              </span>
              <button
                type="button"
                onClick={() => {
                  hasUserSetCompactKnockout.current = true;
                  setCompactKnockout((prev) => !prev);
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
          </div>
          {!isKnockoutBracketReady && (
            <div className="mt-3 inline-flex w-fit max-w-full items-center gap-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              <span>All qualifier and group stage matches must be predicted.</span>
              <LoadingButton
                loading={Boolean(loadingKeys["knockout:resolve"])}
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
                className="rounded-md bg-white px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-red-700 ring-1 ring-red-200 hover:bg-red-100"
              >
                Auto-predict
              </LoadingButton>
            </div>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <LoadingButton
              loading={Boolean(loadingKeys["section:knockouts"])}
              onClick={() =>
                runAutopredictWithDelay(
                  "section:knockouts",
                  handleSectionKnockoutsAutopredict
                )
              }
              className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
            >
              Auto-predict knockouts
            </LoadingButton>
            <button
              type="button"
              onClick={handleSectionKnockoutsReset}
              className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
            >
              Reset
            </button>
          </div>
        </div>
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
                
                const semifinalPosition = compactKnockout ? 3 : 2.8;

                return (
                  <div 
                    className="relative w-full" 
                    style={{ 
                      minWidth: `${minBracketWidth}px`,
                      minHeight: knockoutListHeight ? `${knockoutListHeight + 40}px` : undefined,
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
                                    {champion && <ConfettiAnimation key={champion} duration={2000} champion={champion} />}
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
      </section>

      <div className="flex flex-wrap items-center justify-start gap-3">
        <LoadingButton
          loading={Boolean(loadingKeys.tournament)}
          onClick={() => runAutopredictWithDelay("tournament", handleAutopredict)}
          className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
        >
          Auto-predict tournament
        </LoadingButton>
        <button
          type="button"
          onClick={handleResetAll}
          className="rounded-md bg-white px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-700"
        >
          Reset
        </button>
      </div>

    </div>
  );
}
