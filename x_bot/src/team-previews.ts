import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import { loadWinProbabilities, resolveProbabilityEntry } from "./data.js";
import type { CompactWinProbabilities, MatchProbabilityValues } from "./types.js";

type StageProbabilities = Record<string, number>;
type OpponentProbabilities = Record<string, number>;
type TeamSimulationProbabilities = {
  stage_probability: StageProbabilities;
  group_stage_rank_probability: Record<string, number>;
  R32_opponent_probability?: OpponentProbabilities;
  R16_opponent_probability?: OpponentProbabilities;
  QF_opponent_probability?: OpponentProbabilities;
  SF_opponent_probability?: OpponentProbabilities;
  Final_opponent_probability?: OpponentProbabilities;
};

type TeamPreviewRecord = {
  team: string;
  group: string | null;
  data: TeamSimulationProbabilities;
  groupMatches: TeamGroupMatch[];
  ratings: TeamRatingSummary | null;
};

type GroupMatchRow = {
  id: string;
  group: string;
  date: string;
  homeTeam: string;
  awayTeam: string;
  country: string;
};

type TeamGroupMatch = {
  id: string;
  date: string;
  opponent: string;
  values: MatchProbabilityValues | null;
};

type RatingRow = {
  team: string;
  confederation: string | null;
  rating: number;
  attack: number;
  defense: number;
};

type TeamRatingSummary = RatingRow & {
  worldRank: number;
  confedRank: number | null;
  worldCupRank: number;
};

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(MODULE_DIR, "..", "..");
const MODEL_OUTPUT_DIR = path.join(REPO_ROOT, "model_output");
const REFERENCE_DIR = path.join(REPO_ROOT, "reference_data");
const WEB_FLAG_COLORS_PATH = path.join(REPO_ROOT, "web", "lib", "flag-colors.ts");
const FLAGS_DIR = path.join(REFERENCE_DIR, "flags");
const OUTPUT_DIR = path.resolve(MODULE_DIR, "..", "out", "team-previews");

const CARD_WIDTH = 900;
const CARD_HEIGHT = 1180;

const STAGE_ROWS = [
  { label: "Reach R32", probabilityKey: "Reach R32", opponentKey: "R32_opponent_probability" },
  { label: "Reach R16", probabilityKey: "Reach R16", opponentKey: "R16_opponent_probability" },
  { label: "Reach QF", probabilityKey: "Reach QF", opponentKey: "QF_opponent_probability" },
  { label: "Reach SF", probabilityKey: "Reach SF", opponentKey: "SF_opponent_probability" },
  { label: "Reach final", probabilityKey: "Reach Final", opponentKey: "Final_opponent_probability" },
  { label: "Champion", probabilityKey: "Champion", opponentKey: null },
] as const;

function readJson<T>(filePath: string): T {
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as T;
}

function readCsv(filePath: string): Record<string, string>[] {
  const contents = fs.readFileSync(filePath, "utf8").trim();
  if (!contents) {
    return [];
  }
  const lines = contents.split(/\r?\n/);
  const headers = lines[0]?.split(",") ?? [];
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function shouldUseDecimalPrecision(values: (number | null | undefined)[]) {
  return values.some((v) => v !== null && v !== undefined && Number.isFinite(v) && v * 100 < 0.5);
}

function formatPercent(value: number | null | undefined, forceDecimal = false): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  const percent = value * 100;
  if (percent < 0.1) {
    return "&lt;0.1%";
  }
  if (percent > 99.9) {
    return "&gt;99.9%";
  }
  if (forceDecimal || percent < 0.5 || percent >= 99.5) {
    return `${percent.toFixed(1)}%`;
  }
  return `${Math.round(percent)}%`;
}

function opponentChips(opponents: OpponentProbabilities | undefined) {
  const chips = Object.entries(opponents ?? {})
    .filter(([, value]) => value >= 0.001)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(
      ([team, probability]) =>
        `<div class="opponent-chip">${flagMarkup(team)}<span>${formatPercent(probability)}</span></div>`
    )
    .join("");
  return chips || `<span class="no-opponents">No opponent with at least 0.1% probability</span>`;
}

function slugifyTeam(team: string): string {
  return team
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^A-Za-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
}

function teamPreviewFileName(record: TeamPreviewRecord) {
  const rank = String(record.ratings?.worldCupRank ?? 99).padStart(2, "0");
  return `${rank}-${slugifyTeam(record.team)}.png`;
}

function flagDataUri(team: string): string | null {
  const fileName = `${team.replace(/ /g, "_")}.png`;
  const filePath = path.join(FLAGS_DIR, fileName);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  const buffer = fs.readFileSync(filePath);
  return `data:image/png;base64,${buffer.toString("base64")}`;
}

function flagMarkup(team: string, className = "mini-flag"): string {
  const flag = flagDataUri(team);
  if (!flag) {
    return `<span class="${className} flag-placeholder"></span>`;
  }
  return `<img class="${className}" src="${flag}" alt="">`;
}

function loadFlagColors(): Record<string, string[]> {
  const contents = fs.readFileSync(WEB_FLAG_COLORS_PATH, "utf8");
  const colors: Record<string, string[]> = {};
  const entryPattern = /"([^"]+)":\s*\[([^\]]*)\]/g;
  for (const match of contents.matchAll(entryPattern)) {
    const key = match[1];
    const values = Array.from(match[2].matchAll(/"([^"]+)"/g)).map((valueMatch) => valueMatch[1]);
    colors[key] = values;
  }
  return colors;
}

const FLAG_COLORS = loadFlagColors();

function getTeamFlagColors(team: string): string[] {
  if (FLAG_COLORS[team]) {
    return FLAG_COLORS[team];
  }
  const teamWithUnderscores = team.replace(/\s+/g, "_");
  if (FLAG_COLORS[teamWithUnderscores]) {
    return FLAG_COLORS[teamWithUnderscores];
  }
  return ["#FF0000", "#0000FF", "#FFFF00", "#00FF00"];
}

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const normalized = hex.trim().replace(/^#/, "");
  if (!/^[0-9a-fA-F]{6}$/.test(normalized)) {
    return null;
  }
  return {
    r: parseInt(normalized.slice(0, 2), 16),
    g: parseInt(normalized.slice(2, 4), 16),
    b: parseInt(normalized.slice(4, 6), 16),
  };
}

function relativeLuminance(hex: string) {
  const rgb = hexToRgb(hex);
  if (!rgb) {
    return 1;
  }
  const channels = [rgb.r, rgb.g, rgb.b].map((channel) => {
    const value = channel / 255;
    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function accentColorForTeam(team: string) {
  const colors = getTeamFlagColors(team);
  return colors.find((color) => relativeLuminance(color) < 0.86) ?? colors[0] ?? "#10b981";
}

function rgba(hex: string, alpha: number) {
  const rgb = hexToRgb(hex);
  if (!rgb) {
    return `rgba(16, 185, 129, ${alpha})`;
  }
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

function buildGroupLookup(): Map<string, string> {
  const rows = readCsv(path.join(REFERENCE_DIR, "world_cup_2026_groups.csv"));
  const groups = new Map<string, string>();
  for (const row of rows) {
    const group = row.group?.trim();
    const team = row.team?.trim();
    if (group && team) {
      groups.set(team, group);
    }
  }
  return groups;
}

function parseNumber(value: string | undefined): number | null {
  if (!value) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildConfederationLookup(): Map<string, string> {
  const rows = readCsv(path.join(REFERENCE_DIR, "confederations.csv"));
  const confederations = new Map<string, string>();
  for (const row of rows) {
    const team = row.team?.trim();
    const confederation = row.confederation?.trim();
    if (team && confederation && !confederations.has(team)) {
      confederations.set(team, confederation);
    }
  }
  return confederations;
}

function loadRatingSummaries(worldCupTeams: Set<string>): Map<string, TeamRatingSummary> {
  const confederations = buildConfederationLookup();
  const ratings: RatingRow[] = readCsv(path.join(MODEL_OUTPUT_DIR, "ratings_current.csv"))
    .map((row) => {
      const team = row.team?.trim() ?? "";
      const rating = parseNumber(row.rating);
      const attack = parseNumber(row.rating_attack);
      const defense = parseNumber(row.rating_defense);
      if (!team || rating === null || attack === null || defense === null) {
        return null;
      }
      return {
        team,
        confederation: confederations.get(team) ?? null,
        rating,
        attack,
        defense,
      };
    })
    .filter((row): row is RatingRow => Boolean(row));

  const byRating = [...ratings].sort((a, b) => b.rating - a.rating);
  const worldRanks = new Map(byRating.map((row, index) => [row.team, index + 1]));
  const worldCupRanks = new Map(
    byRating
      .filter((row) => worldCupTeams.has(row.team))
      .map((row, index) => [row.team, index + 1])
  );

  const confedRanks = new Map<string, number>();
  const confeds = new Set(ratings.map((row) => row.confederation).filter(Boolean));
  for (const confed of confeds) {
    byRating
      .filter((row) => row.confederation === confed)
      .forEach((row, index) => confedRanks.set(row.team, index + 1));
  }

  const summaries = new Map<string, TeamRatingSummary>();
  for (const row of ratings) {
    const worldRank = worldRanks.get(row.team);
    const worldCupRank = worldCupRanks.get(row.team);
    if (!worldRank || !worldCupRank) {
      continue;
    }
    summaries.set(row.team, {
      ...row,
      worldRank,
      confedRank: row.confederation ? confedRanks.get(row.team) ?? null : null,
      worldCupRank,
    });
  }
  return summaries;
}

function loadGroupMatches(): GroupMatchRow[] {
  return readCsv(path.join(REFERENCE_DIR, "world_cup_2026_group_matches.csv")).map((row) => ({
    id: row.match_id?.trim() ?? "",
    group: row.group?.trim() ?? "",
    date: row.date?.trim() ?? "",
    homeTeam: row.home_team?.trim() ?? "",
    awayTeam: row.away_team?.trim() ?? "",
    country: row.country?.trim() ?? "",
  }));
}

function resolveMatchProbabilities({
  probabilities,
  homeTeam,
  awayTeam,
  country,
}: {
  probabilities: CompactWinProbabilities;
  homeTeam: string;
  awayTeam: string;
  country?: string | null;
}): MatchProbabilityValues | null {
  const resolved = resolveProbabilityEntry({
    probabilities,
    homeTeam,
    awayTeam,
    country,
    neutralOverride: null,
  });
  if (!resolved) {
    return null;
  }
  const entry = resolved.entry;
  const values = {
    home: entry[6] ?? null,
    draw: entry[7] ?? null,
    away: entry[8] ?? null,
  };
  if (!resolved.flipped) {
    return values;
  }
  return { home: values.away, draw: values.draw, away: values.home };
}

function teamGroupMatches(
  team: string,
  groupMatches: GroupMatchRow[],
  probabilities: CompactWinProbabilities
): TeamGroupMatch[] {
  return groupMatches
    .filter((match) => match.homeTeam === team || match.awayTeam === team)
    .sort((a, b) => a.date.localeCompare(b.date) || a.id.localeCompare(b.id))
    .map((match) => {
      const isHome = match.homeTeam === team;
      const values = resolveMatchProbabilities({
        probabilities,
        homeTeam: match.homeTeam,
        awayTeam: match.awayTeam,
        country: match.country,
      });
      return {
        id: match.id,
        date: match.date,
        opponent: isHome ? match.awayTeam : match.homeTeam,
        values: values
          ? {
              home: isHome ? values.home : values.away,
              draw: values.draw,
              away: isHome ? values.away : values.home,
            }
          : null,
      };
    });
}

function loadPreviewRecords(): TeamPreviewRecord[] {
  const data = readJson<Record<string, TeamSimulationProbabilities>>(
    path.join(MODEL_OUTPUT_DIR, "simulation_team_probabilities.json")
  );
  const groupLookup = buildGroupLookup();
  const groupMatches = loadGroupMatches();
  const probabilities = loadWinProbabilities();
  const worldCupTeams = new Set(Object.keys(data));
  const ratings = loadRatingSummaries(worldCupTeams);
  return Object.keys(data)
    .sort((a, b) => a.localeCompare(b))
    .map((team) => ({
      team,
      group: groupLookup.get(team) ?? null,
      data: data[team],
      groupMatches: teamGroupMatches(team, groupMatches, probabilities),
      ratings: ratings.get(team) ?? null,
    }));
}

function progressRows(record: TeamPreviewRecord): string {
  const forceDecimal = shouldUseDecimalPrecision(
    STAGE_ROWS.map((row) => record.data.stage_probability[row.probabilityKey])
  );
  return STAGE_ROWS.map((row) => {
    const probability = formatPercent(record.data.stage_probability[row.probabilityKey], forceDecimal);
    if (row.probabilityKey === "Champion") {
      return `
        <div class="champion-row">
          <span>Champion</span>
          <strong>${probability}</strong>
        </div>
      `;
    }
    const chips =
      row.opponentKey == null
        ? ""
        : opponentChips(record.data[row.opponentKey] as OpponentProbabilities | undefined);
    return `
      <div class="progress-row">
        <div>
          <span class="label">${escapeHtml(row.label)}</span>
          <strong>${probability}</strong>
        </div>
        ${chips ? `<div class="opponents">${chips}</div>` : `<div></div>`}
      </div>
    `;
  }).join("");
}

function groupPositionTable(record: TeamPreviewRecord): string {
  const forceDecimal = shouldUseDecimalPrecision(
    Object.values(record.data.group_stage_rank_probability)
  );
  const positions = [
    ["1", "First"],
    ["2", "Second"],
    ["3", "Third"],
    ["4", "Fourth"],
  ] as const;
  return positions
    .map(([position, label]) => {
      const probability = record.data.group_stage_rank_probability[position] ?? 0;
      return `
        <div class="position-cell">
          <span>${label}</span>
          <strong>${formatPercent(probability, forceDecimal)}</strong>
        </div>
      `;
    })
    .join("");
}

function groupMatchRows(record: TeamPreviewRecord): string {
  return record.groupMatches
    .map((match) => {
      const values = match.values;
      const forceDecimal = shouldUseDecimalPrecision([values?.home, values?.draw, values?.away]);
      return `
        <div class="match-row">
          <div class="match-team">${flagMarkup(match.opponent)}<strong>${escapeHtml(match.opponent)}</strong></div>
          <div class="match-probs">
            <div class="prob-win"><span>Win</span><strong>${formatPercent(values?.home, forceDecimal)}</strong></div>
            <div class="prob-draw"><span>Draw</span><strong>${formatPercent(values?.draw, forceDecimal)}</strong></div>
            <div class="prob-loss"><span>Loss</span><strong>${formatPercent(values?.away, forceDecimal)}</strong></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function formatRating(value: number) {
  return Math.round(value).toString();
}

function ratingPanel(record: TeamPreviewRecord): string {
  const ratings = record.ratings;
  if (!ratings) {
    return "";
  }
  const confedLabel = ratings.confederation ?? "Confed";
  const confedRank = ratings.confedRank ? `#${ratings.confedRank}` : "--";
  return `
    <div class="ratings-panel">
      <div class="rating-metrics">
        <div><span>Overall</span><strong>${formatRating(ratings.rating)}</strong></div>
        <div><span>Attack</span><strong>${formatRating(ratings.attack)}</strong></div>
        <div><span>Defense</span><strong>${formatRating(ratings.defense)}</strong></div>
      </div>
      <div class="rating-ranks">
        <div><span>World</span><strong>#${ratings.worldRank}</strong></div>
        <div><span>${escapeHtml(confedLabel)}</span><strong>${confedRank}</strong></div>
        <div><span>WC</span><strong>#${ratings.worldCupRank}</strong></div>
      </div>
    </div>
  `;
}

function buildHtml(record: TeamPreviewRecord): string {
  const group = record.group ? `Group ${escapeHtml(record.group)}` : "World Cup 2026";
  const accent = accentColorForTeam(record.team);
  const teamNameClass = record.team === "Bosnia and Herzegovina" ? "team-name team-name-long" : "team-name";

  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        width: ${CARD_WIDTH}px;
        height: ${CARD_HEIGHT}px;
        overflow: hidden;
        background: #ffffff;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #0f172a;
      }
      .card {
        width: ${CARD_WIDTH}px;
        height: ${CARD_HEIGHT}px;
        padding: 30px 42px;
        background:
          linear-gradient(135deg, var(--team-accent-bg), rgba(255, 255, 255, 0.88) 38%, rgba(255, 241, 242, 0.84)),
          #ffffff;
        border: 1px solid #dbe3ef;
        display: flex;
        flex-direction: column;
      }
      .eyebrow {
        margin-top: 18px;
        font-size: 26px;
        line-height: 1;
        font-weight: 750;
        letter-spacing: 0;
        color: #475569;
        text-align: center;
      }
      .brand {
        display: inline-flex;
        align-items: flex-end;
        gap: 11px;
        color: #0f172a;
        align-self: center;
      }
      .brand svg {
        width: 34px;
        height: 34px;
        flex: 0 0 auto;
      }
      .brand span {
        position: relative;
        top: 2px;
        font-size: 34px;
        line-height: 1;
        font-weight: 850;
      }
      .header {
        display: grid;
        grid-template-columns: 138px minmax(0, 1fr) 238px;
        gap: 28px;
        align-items: center;
        margin-top: 16px;
      }
      .flag-wrap {
        width: 138px;
        height: 91px;
        border-radius: 7px;
        overflow: hidden;
        background: #e2e8f0;
        box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.12), 0 18px 38px var(--team-accent-soft);
      }
      .flag {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
      }
      .mini-flag {
        width: 30px;
        height: 20px;
        flex: 0 0 auto;
        border-radius: 2px;
        object-fit: cover;
        box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.10);
      }
      .flag-placeholder {
        background: repeating-linear-gradient(45deg, #e2e8f0, #e2e8f0 12px, #cbd5e1 12px, #cbd5e1 24px);
      }
      .team-name {
        margin: 0;
        font-size: 58px;
        line-height: 0.94;
        font-weight: 850;
        letter-spacing: 0;
        overflow-wrap: anywhere;
      }
      .team-name-long {
        font-size: 39px;
        line-height: 0.9;
      }
      .group {
        margin-top: 12px;
        font-size: 22px;
        font-weight: 700;
        color: #64748b;
      }
      .ratings-panel {
        align-self: stretch;
        display: grid;
        gap: 8px;
        padding-top: 2px;
      }
      .rating-metrics,
      .rating-ranks {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
      }
      .ratings-panel div div {
        min-width: 0;
        border-radius: 7px;
        background: rgba(15, 23, 42, 0.05);
        padding: 8px 6px;
        text-align: center;
      }
      .ratings-panel span {
        display: block;
        font-size: 10px;
        line-height: 1;
        font-weight: 850;
        color: #64748b;
        text-transform: uppercase;
        white-space: nowrap;
      }
      .ratings-panel strong {
        display: block;
        margin-top: 6px;
        font-size: 21px;
        line-height: 1;
      }
      .rating-ranks strong {
        font-size: 19px;
      }
      .section-title {
        margin-top: 22px;
        font-size: 16px;
        line-height: 1;
        font-weight: 850;
        color: #334155;
        text-transform: uppercase;
      }
      .matches {
        margin-top: 10px;
        display: grid;
        gap: 8px;
      }
      .match-row {
        min-height: 72px;
        display: grid;
        grid-template-columns: 1fr 360px;
        align-items: center;
        gap: 18px;
        padding: 11px 14px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.66);
      }
      .match-team {
        min-width: 0;
        display: flex;
        align-items: center;
        gap: 12px;
      }
      .match-team strong {
        font-size: 26px;
        line-height: 1;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
      }
      .match-probs {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
      }
      .match-probs div {
        border-radius: 6px;
        background: rgba(15, 23, 42, 0.05);
        padding: 8px 9px;
      }
      .match-probs .prob-win {
        background: rgba(16, 185, 129, 0.18);
      }
      .match-probs .prob-loss {
        background: rgba(244, 63, 94, 0.15);
      }
      .match-probs span {
        display: block;
        font-size: 12px;
        line-height: 1;
        font-weight: 850;
        color: #64748b;
        text-transform: uppercase;
      }
      .match-probs strong {
        display: block;
        margin-top: 6px;
        font-size: 22px;
        line-height: 1;
      }
      .progress {
        margin-top: 10px;
        display: grid;
        gap: 6px;
      }
      .progress-row {
        min-height: 55px;
        display: grid;
        grid-template-columns: 250px 1fr;
        gap: 20px;
        align-items: center;
        padding: 7px 14px;
        border-radius: 7px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.35);
      }
      .progress-row .label,
      .opponents span {
        display: block;
        font-size: 14px;
        line-height: 1.1;
        font-weight: 750;
        color: #64748b;
        text-transform: uppercase;
      }
      .progress-row strong {
        display: block;
        margin-top: 4px;
        font-size: 24px;
        line-height: 1;
      }
      .opponents {
        min-width: 0;
        display: grid;
        grid-template-columns: repeat(5, 78px);
        align-items: center;
        justify-content: end;
        column-gap: 8px;
      }
      .opponent-chip {
        width: 78px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        row-gap: 3px;
        padding: 0;
      }
      .opponents .opponent-chip span {
        display: block;
        font-size: 17px;
        line-height: 20px;
        font-weight: 850;
        color: #475569;
        text-transform: none;
        white-space: nowrap;
      }
      .opponents .no-opponents {
        grid-column: 1 / -1;
        display: block;
        font-size: 15px;
        line-height: 1;
        font-weight: 800;
        color: #64748b;
        text-transform: none;
        white-space: nowrap;
      }
      .champion-row {
        margin-top: 4px;
        min-height: 76px;
        padding: 16px 24px;
        border-radius: 8px;
        background: #0f172a;
        color: white;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 24px;
      }
      .champion-row span {
        font-size: 30px;
        font-weight: 800;
      }
      .champion-row strong {
        font-size: 44px;
        line-height: 1;
      }
      .positions-title {
        margin-top: 10px;
        font-size: 16px;
        line-height: 1;
        font-weight: 850;
        color: #334155;
        text-transform: uppercase;
      }
      .positions {
        margin-top: 8px;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
      }
      .position-cell {
        height: 58px;
        border-radius: 7px;
        background: rgba(15, 23, 42, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.42);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }
      .position-cell span {
        font-size: 14px;
        line-height: 1;
        font-weight: 850;
        color: #64748b;
        text-transform: uppercase;
      }
      .position-cell strong {
        margin-top: 6px;
        font-size: 23px;
        line-height: 1;
      }
    </style>
  </head>
  <body>
    <section
      class="card"
      style="--team-accent: ${accent}; --team-accent-soft: ${rgba(accent, 0.26)}; --team-accent-bg: ${rgba(accent, 0.16)};"
    >
      <div class="brand">
        <svg aria-hidden="true" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
          <rect width="32" height="32" rx="6" ry="6" fill="black" />
          <path
            d="M8 8 H24 M8 8 V24"
            stroke="white"
            stroke-width="3"
            stroke-linecap="round"
            stroke-linejoin="round"
            fill="none"
          />
        </svg>
        <span>TheBackPost</span>
      </div>
      <div class="eyebrow">World Cup 2026 Team Preview</div>
      <div class="header">
        <div class="flag-wrap">${flagMarkup(record.team, "flag")}</div>
        <div>
          <h1 class="${teamNameClass}">${escapeHtml(record.team)}</h1>
          <div class="group">${group}</div>
        </div>
        ${ratingPanel(record)}
      </div>
      <div class="section-title">Group stage opponents</div>
      <div class="matches">${groupMatchRows(record)}</div>
      <div class="section-title">Group stage position</div>
      <div class="positions">${groupPositionTable(record)}</div>
      <div class="section-title">Progression chances</div>
      <div class="progress">${progressRows(record)}</div>
    </section>
  </body>
</html>`;
}

function resolveBrowserPath() {
  const explicit = process.env.CHROMIUM_PATH?.trim();
  if (explicit) {
    return explicit;
  }

  const candidates = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "/opt/homebrew/bin/chromium",
    "chromium",
  ];

  for (const candidate of candidates) {
    if (candidate.includes("/") && fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return "chromium";
}

async function renderTeamPreviews() {
  const records = loadPreviewRecords();
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: resolveBrowserPath(),
    headless: true,
    args: ["--disable-gpu", "--hide-scrollbars"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: CARD_WIDTH, height: CARD_HEIGHT, deviceScaleFactor: 1 });
    for (const record of records) {
      await page.setContent(buildHtml(record), { waitUntil: "load" });
      const outputPath = path.join(OUTPUT_DIR, teamPreviewFileName(record));
      await page.screenshot({
        path: outputPath,
        type: "png",
        clip: { x: 0, y: 0, width: CARD_WIDTH, height: CARD_HEIGHT },
        omitBackground: false,
      });
      console.log(`Rendered ${record.team}: ${path.relative(REPO_ROOT, outputPath)}`);
    }
  } finally {
    await browser.close();
  }

  console.log(`\nRendered ${records.length} team preview images to ${path.relative(REPO_ROOT, OUTPUT_DIR)}`);
}

renderTeamPreviews().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
