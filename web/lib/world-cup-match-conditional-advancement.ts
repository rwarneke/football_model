import { readFile } from "node:fs/promises";
import path from "node:path";

export type MatchOutcomeConditionalAdvancement = {
  count: number;
  homeTeamProbability: number | null;
  awayTeamProbability: number | null;
};

export type MatchConditionalAdvancement = {
  matchId: number;
  stage: string;
  homeTeam: string;
  awayTeam: string;
  nextStage: string;
  basis: "full_time" | "after_90";
  outcomes: {
    home_win: MatchOutcomeConditionalAdvancement;
    draw: MatchOutcomeConditionalAdvancement;
    away_win: MatchOutcomeConditionalAdvancement;
  };
};

const PUBLIC_DIR = path.join(process.cwd(), "public");

function isErrnoException(error: unknown): error is NodeJS.ErrnoException {
  return typeof error === "object" && error !== null && "code" in error;
}

export async function loadWorldCupMatchConditionalAdvancement(
  modelOutputDir = "/model_output"
): Promise<Map<string, MatchConditionalAdvancement>> {
  const fullPath = path.join(
    PUBLIC_DIR,
    modelOutputDir.replace(/^\/+/, ""),
    "simulation_match_conditional_advancement.json"
  );
  let contents: string;
  try {
    contents = await readFile(fullPath, "utf8");
  } catch (error) {
    if (isErrnoException(error) && error.code === "ENOENT") {
      return new Map();
    }
    throw error;
  }
  const parsed = JSON.parse(contents) as {
    matches?: Record<
      string,
      {
        match_id: number;
        stage: string;
        home_team: string;
        away_team: string;
        next_stage: string;
        basis: "full_time" | "after_90";
        outcomes: {
          home_win: {
            count: number;
            home_team_probability: number | null;
            away_team_probability: number | null;
          };
          draw: {
            count: number;
            home_team_probability: number | null;
            away_team_probability: number | null;
          };
          away_win: {
            count: number;
            home_team_probability: number | null;
            away_team_probability: number | null;
          };
        };
      }
    >;
  };

  return new Map(
    Object.entries(parsed.matches ?? {}).map(([matchId, entry]) => [
      matchId,
      {
        matchId: entry.match_id,
        stage: entry.stage,
        homeTeam: entry.home_team,
        awayTeam: entry.away_team,
        nextStage: entry.next_stage,
        basis: entry.basis,
        outcomes: {
          home_win: {
            count: entry.outcomes.home_win.count,
            homeTeamProbability: entry.outcomes.home_win.home_team_probability,
            awayTeamProbability: entry.outcomes.home_win.away_team_probability,
          },
          draw: {
            count: entry.outcomes.draw.count,
            homeTeamProbability: entry.outcomes.draw.home_team_probability,
            awayTeamProbability: entry.outcomes.draw.away_team_probability,
          },
          away_win: {
            count: entry.outcomes.away_win.count,
            homeTeamProbability: entry.outcomes.away_win.home_team_probability,
            awayTeamProbability: entry.outcomes.away_win.away_team_probability,
          },
        },
      } satisfies MatchConditionalAdvancement,
    ])
  );
}
