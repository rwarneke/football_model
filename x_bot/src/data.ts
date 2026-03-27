import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import countries from "i18n-iso-countries";
import enLocale from "i18n-iso-countries/langs/en.json" with { type: "json" };
import type { CompactWinProbabilities, WorldCupMatch } from "./types.js";

countries.registerLocale(enLocale);

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(MODULE_DIR, "..", "..");
const PUBLIC_DIR = path.join(ROOT, "web", "public");
const REFERENCE_DIR = path.join(PUBLIC_DIR, "reference_data");
const MODEL_OUTPUT_DIR = path.join(PUBLIC_DIR, "model_output");
const FLAGS_DIR = path.join(PUBLIC_DIR, "flags");

const HOST_TEAM_COUNTRIES: Record<string, string> = {
  USA: "USA",
  "United States": "USA",
  Canada: "Canada",
  Mexico: "Mexico",
};

const TEAM_TO_ISO2_OVERRIDES: Record<string, string> = {
  USA: "US",
  "South Korea": "KR",
  "North Korea": "KP",
  "DR Congo": "CD",
  "Ivory Coast": "CI",
  "Cape Verde": "CV",
  Curacao: "CW",
  Iran: "IR",
  "Saudi Arabia": "SA",
  England: "GB",
  Scotland: "GB",
  Wales: "GB",
  "Northern Ireland": "GB",
  Kosovo: "XK",
};

function readCsv(filePath: string) {
  const contents = fs.readFileSync(filePath, "utf8").trim();
  const lines = contents ? contents.split(/\r?\n/) : [];
  if (lines.length === 0) {
    return [];
  }
  const headers = lines[0]?.split(",") ?? [];
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""])
    ) as Record<string, string>;
  });
}

function normalizeDate(value: string) {
  return value?.trim();
}

function qualifierTeamLabel(team: string, source: string, qualifierPath: string) {
  const trimmed = team?.trim();
  if (trimmed) {
    return trimmed;
  }
  const sourceTrimmed = source?.trim();
  if (sourceTrimmed) {
    return `Winner ${sourceTrimmed.toUpperCase()}`;
  }
  return qualifierPath?.trim() ? `${qualifierPath} TBD` : "TBD";
}

function formatQualifierStage(stage: string, qualifierPath: string) {
  const trimmedStage = stage.trim();
  const trimmedPath = qualifierPath.trim();
  if (!trimmedStage) {
    return trimmedPath;
  }
  if (!trimmedPath) {
    return trimmedStage;
  }
  if (trimmedStage.toLowerCase().includes("uefa") && trimmedPath.startsWith("UEFA ")) {
    return `${trimmedStage} ${trimmedPath.replace(/^UEFA\s+/, "")}`;
  }
  if (
    trimmedStage.toLowerCase().includes("inter-confederation") &&
    trimmedPath.startsWith("IC ")
  ) {
    return `IC Playoff ${trimmedPath.replace(/^IC\s+/, "")}`;
  }
  return `${trimmedStage} ${trimmedPath}`;
}

export function loadMatches(): WorldCupMatch[] {
  const groupMatches = readCsv(
    path.join(REFERENCE_DIR, "world_cup_2026_group_matches.csv")
  ).map((row) => ({
    id: row.match_id ?? "",
    date: normalizeDate(row.date ?? ""),
    stage: `Group ${row.group ?? ""}`.trim(),
    home: row.home_team ?? "",
    away: row.away_team ?? "",
    stadium: row.stadium ?? "",
    city: row.city ?? "",
    country: row.country ?? "",
    group: row.group ?? null,
    neutral: null,
  }));

  const knockoutMatches = readCsv(
    path.join(REFERENCE_DIR, "world_cup_2026_knockout_matches.csv")
  ).map((row) => ({
    id: row.match_id ?? "",
    date: normalizeDate(row.date ?? ""),
    stage: row.stage ?? "",
    home: row.home ?? "",
    away: row.away ?? "",
    stadium: row.stadium ?? "",
    city: row.city ?? "",
    country: row.country ?? "",
    group: null,
    neutral: null,
  }));

  const qualifierMatches = readCsv(
    path.join(REFERENCE_DIR, "world_cup_2026_remaining_qualifiers.csv")
  ).map((row, index) => ({
    id: `Q-${index + 1}`,
    date: normalizeDate(row.date ?? ""),
    stage: formatQualifierStage(row.stage ?? "", row.path ?? ""),
    home: qualifierTeamLabel(row.home_team ?? "", row.home_source ?? "", row.path ?? ""),
    away: qualifierTeamLabel(row.away_team ?? "", row.away_source ?? "", row.path ?? ""),
    stadium: row.stadium ?? "",
    city: row.city ?? "",
    country: row.country ?? "",
    group: null,
    neutral:
      row.neutral?.trim().toLowerCase() === "true"
        ? true
        : row.neutral?.trim().toLowerCase() === "false"
          ? false
          : null,
  }));

  return [...qualifierMatches, ...groupMatches, ...knockoutMatches]
    .filter((match) => match.date)
    .sort((a, b) => a.date.localeCompare(b.date));
}

export function loadWinProbabilities(): CompactWinProbabilities {
  return JSON.parse(
    fs.readFileSync(path.join(MODEL_OUTPUT_DIR, "win_probabilities.json"), "utf8")
  ) as CompactWinProbabilities;
}

export function isPlaceholderLabel(name: string) {
  const trimmed = name.trim();
  if (!trimmed) {
    return true;
  }
  return (
    /^Winner\b/i.test(trimmed) ||
    /^Runner-up\b/i.test(trimmed) ||
    /^3rd\b/i.test(trimmed) ||
    /^Loser\b/i.test(trimmed) ||
    /winner$/i.test(trimmed)
  );
}

export function normalizeCountry(value: string | null | undefined) {
  return value ? value.trim().toLowerCase() : "";
}

export function resolveMatchNeutrality({
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
  }
  return { neutral, advantage };
}

export function resolveProbabilityEntry({
  probabilities,
  homeTeam,
  awayTeam,
  country,
  neutralOverride,
}: {
  probabilities: CompactWinProbabilities;
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
  const teams = probabilities.teams;
  const homeId = teams.indexOf(homeTeam);
  const awayId = teams.indexOf(awayTeam);
  if (homeId === -1 || awayId === -1) {
    return null;
  }

  const entryMap = new Map<string, (typeof probabilities.entries)[number]>(
    probabilities.entries.map((entry) => [`${entry[0]}:${entry[1]}:${entry[2]}`, entry] as const)
  );
  const keyFor = (a: number, b: number, neutralFlag: boolean) =>
    `${a}:${b}:${neutralFlag ? 1 : 0}`;

  if (neutral) {
    const entry = entryMap.get(keyFor(homeId, awayId, true));
    return entry ? { entry, flipped: false } : null;
  }
  if (advantage === "home") {
    const entry = entryMap.get(keyFor(homeId, awayId, false));
    return entry ? { entry, flipped: false } : null;
  }
  if (advantage === "away") {
    const entry = entryMap.get(keyFor(awayId, homeId, false));
    return entry ? { entry, flipped: true } : null;
  }
  return null;
}

export function flagImagePath(team: string) {
  return path.join(FLAGS_DIR, `${team.replace(/ /g, "_")}.png`);
}

export function loadTeamToFifaCode(): Map<string, string> {
  const rows = readCsv(path.join(REFERENCE_DIR, "fifa_country_codes.csv"));
  return new Map(rows.map((row) => [row.team, row.fifa_code]));
}

export function teamToFlagEmoji(team: string, teamToFifaCode: Map<string, string>) {
  if (isPlaceholderLabel(team)) {
    return "🏳️";
  }
  const alpha2 =
    TEAM_TO_ISO2_OVERRIDES[team] ??
    countries.getAlpha2Code(team, "en") ??
    (() => {
      const fifaCode = teamToFifaCode.get(team);
      return fifaCode ? countries.alpha3ToAlpha2(fifaCode) ?? "" : "";
    })();
  if (!alpha2 || alpha2.length !== 2) {
    return "🏳️";
  }
  return alpha2
    .toUpperCase()
    .split("")
    .map((char) => String.fromCodePoint(127397 + char.charCodeAt(0)))
    .join("");
}
