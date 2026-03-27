import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildScoreMatrix } from "./score-matrix.js";
import type { MatchProbabilityValues, WorldCupMatch } from "./types.js";
import {
  isPlaceholderLabel,
  loadMatches,
  loadTeamToFifaCode,
  loadWinProbabilities,
  resolveProbabilityEntry,
  teamToFlagEmoji,
} from "./data.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = path.resolve(MODULE_DIR, "..", "out");

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  const percent = value * 100;
  if (percent > 0 && percent < 0.05) {
    return "<0.1%";
  }
  if (percent < 0.5 || percent >= 99.5) {
    return `${Math.min(99.9, Number(percent.toFixed(1))).toFixed(1)}%`;
  }
  return `${Math.round(percent)}%`;
}

function resolveMatchProbabilities({
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
}): MatchProbabilityValues | null {
  const probabilities = loadWinProbabilities();
  if (isPlaceholderLabel(homeTeam) || isPlaceholderLabel(awayTeam)) {
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
  const entry = resolved.entry;
  const values = allowDraw
    ? {
        home: entry[6] ?? null,
        draw: entry[7] ?? null,
        away: entry[8] ?? null,
      }
    : {
        home: entry[9] ?? null,
        draw: null,
        away: entry[10] ?? null,
      };
  if (!resolved.flipped) {
    return values;
  }
  return { home: values.away, draw: values.draw, away: values.home };
}

function competitionLabel(match: WorldCupMatch) {
  return `${match.stage} · 2026 FIFA World Cup`;
}

function formatMatchDate(date: string) {
  return new Date(`${date}T00:00:00.000Z`).toLocaleDateString("en-GB", {
    timeZone: "UTC",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function formatLocation(match: WorldCupMatch) {
  if (match.city && match.country) {
    return `${match.city}, ${match.country}`;
  }
  return match.city || match.country || "TBD";
}

function formatScoreBucket(value: number, maxIndex: number) {
  return value === maxIndex ? `${value}+` : `${value}`;
}

function mostLikelyScoreline(matrix: number[][] | null) {
  if (!matrix || matrix.length === 0) {
    return null;
  }
  let bestHome = -1;
  let bestAway = -1;
  let bestValue = -1;
  for (let homeGoals = 0; homeGoals < matrix.length; homeGoals += 1) {
    const row = matrix[homeGoals] ?? [];
    for (let awayGoals = 0; awayGoals < row.length; awayGoals += 1) {
      const value = row[awayGoals] ?? 0;
      if (value > bestValue) {
        bestValue = value;
        bestHome = homeGoals;
        bestAway = awayGoals;
      }
    }
  }
  if (bestHome < 0 || bestAway < 0 || !(bestValue >= 0)) {
    return null;
  }
  const maxHome = matrix.length - 1;
  const maxAway = matrix.reduce((max, row) => Math.max(max, row.length), 0) - 1;
  return `${formatScoreBucket(bestHome, maxHome)}-${formatScoreBucket(bestAway, maxAway)} (${formatPercent(bestValue)})`;
}

function resolveMatchScoreMatrix(match: WorldCupMatch) {
  const probabilities = loadWinProbabilities();
  if (isPlaceholderLabel(match.home) || isPlaceholderLabel(match.away)) {
    return null;
  }
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam: match.home,
    awayTeam: match.away,
    country: match.country,
    neutralOverride: match.neutral,
  });
  if (!resolved) {
    return null;
  }
  const entry = resolved.entry;
  if (
    entry[3] === undefined ||
    entry[4] === undefined ||
    entry[5] === undefined
  ) {
    return null;
  }
  const matrix = buildScoreMatrix({
    nu: entry[3],
    lamH: entry[4],
    lamA: entry[5],
    maxGoals: probabilities.max_goals ?? 8,
  });
  if (!resolved.flipped) {
    return matrix;
  }
  const rows = matrix.length;
  const cols = matrix.reduce((max, row) => Math.max(max, row.length), 0);
  const transposed = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < matrix[i].length; j += 1) {
      transposed[j][i] = matrix[i][j] ?? 0;
    }
  }
  return transposed;
}

function postHeaderLine(match: WorldCupMatch, teamToFifaCode: Map<string, string>) {
  const homeFlag = teamToFlagEmoji(match.home, teamToFifaCode);
  const awayFlag = teamToFlagEmoji(match.away, teamToFifaCode);
  return `${homeFlag} ${match.home} vs. ${match.away} ${awayFlag}`;
}

function buildPostText(match: WorldCupMatch, teamToFifaCode: Map<string, string>) {
  const allowDraw = Boolean(match.group);
  const scoreMatrix = resolveMatchScoreMatrix(match);
  const ninetyValues = resolveMatchProbabilities({
    homeTeam: match.home,
    awayTeam: match.away,
    allowDraw: true,
    country: match.country,
    neutralOverride: match.neutral,
  });
  const qualifyValues = allowDraw
    ? null
    : resolveMatchProbabilities({
        homeTeam: match.home,
        awayTeam: match.away,
        allowDraw: false,
        country: match.country,
        neutralOverride: match.neutral,
      });

  const lines = [
    "⚽ MATCH PREVIEW ⚽",
    "",
    postHeaderLine(match, teamToFifaCode),
    competitionLabel(match),
    `${formatMatchDate(match.date)} · ${formatLocation(match)}`,
    "",
  ];

  if (!allowDraw) {
    lines.push("After 90'");
  }
  lines.push(`${match.home} ${formatPercent(ninetyValues?.home)}`);
  lines.push(`Draw ${formatPercent(ninetyValues?.draw)}`);
  lines.push(`${match.away} ${formatPercent(ninetyValues?.away)}`);

  if (!allowDraw) {
    lines.push("");
    lines.push("Full Time");
    lines.push(`${match.home} ${formatPercent(qualifyValues?.home)}`);
    lines.push(`${match.away} ${formatPercent(qualifyValues?.away)}`);
  }

  const likelyScoreline = mostLikelyScoreline(scoreMatrix);
  if (likelyScoreline) {
    lines.push("");
    lines.push(`Most likely score (after 90'): ${likelyScoreline}`);
  }

  return {
    text: lines.join("\n"),
    ninetyValues,
    qualifyValues,
  };
}

function targetPostTimeForMatch(matchDate: string) {
  const matchMidnightUtc = new Date(`${matchDate}T00:00:00.000Z`);
  return new Date(matchMidnightUtc.getTime() - 48 * 60 * 60 * 1000);
}

export function loadDueMatchPreviews(now = new Date(), windowHours = 1) {
  const teamToFifaCode = loadTeamToFifaCode();
  return loadMatches()
    .filter((match) => !isPlaceholderLabel(match.home) && !isPlaceholderLabel(match.away))
    .filter((match) => {
      const scheduledAt = targetPostTimeForMatch(match.date);
      const deltaMs = scheduledAt.getTime() - now.getTime();
      return deltaMs <= 0 && deltaMs > -windowHours * 60 * 60 * 1000;
    })
    .map((match) => {
      const generated = buildPostText(match, teamToFifaCode);
      const scheduledAt = targetPostTimeForMatch(match.date);
      return {
        match,
        competitionLabel: competitionLabel(match),
        ninetyValues: generated.ninetyValues,
        qualifyValues: generated.qualifyValues,
        scoreMatrix: resolveMatchScoreMatrix(match),
        postText: generated.text,
        imagePath: path.join(OUTPUT_DIR, `${match.id}-${match.home}-vs-${match.away}.png`.replaceAll("/", "-")),
        scheduledAtIso: scheduledAt.toISOString(),
        dedupeKey: `${match.id}|${match.date}|${match.home}|${match.away}`,
      };
    });
}

export function loadPreviewForMatchId(matchId: string) {
  const teamToFifaCode = loadTeamToFifaCode();
  const match = loadMatches().find((item) => item.id === matchId);
  if (!match) {
    throw new Error(`No match found with id ${matchId}`);
  }
  if (isPlaceholderLabel(match.home) || isPlaceholderLabel(match.away)) {
    throw new Error(`Match ${matchId} still contains placeholder teams.`);
  }
  const generated = buildPostText(match, teamToFifaCode);
  const scheduledAt = targetPostTimeForMatch(match.date);
  return {
    match,
    competitionLabel: competitionLabel(match),
    ninetyValues: generated.ninetyValues,
    qualifyValues: generated.qualifyValues,
    scoreMatrix: resolveMatchScoreMatrix(match),
    postText: generated.text,
    imagePath: path.join(OUTPUT_DIR, `${match.id}-${match.home}-vs-${match.away}.png`.replaceAll("/", "-")),
    scheduledAtIso: scheduledAt.toISOString(),
    dedupeKey: `${match.id}|${match.date}|${match.home}|${match.away}`,
  };
}
