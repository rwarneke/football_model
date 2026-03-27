export type WorldCupMatch = {
  id: string;
  date: string;
  stage: string;
  home: string;
  away: string;
  stadium: string;
  city: string;
  country: string;
  group: string | null;
  neutral: boolean | null;
};

export type MatchProbabilityValues = {
  home: number | null;
  draw: number | null;
  away: number | null;
};

export type MatchPreview = {
  match: WorldCupMatch;
  competitionLabel: string;
  ninetyValues: MatchProbabilityValues | null;
  qualifyValues: MatchProbabilityValues | null;
  scoreMatrix: number[][] | null;
  imagePath: string;
  postText: string;
  scheduledAtIso: string;
  dedupeKey: string;
};

export type PreviewBuildOptions = {
  variant?: boolean;
};

export type CompactWinProbabilities = {
  version: number;
  max_goals?: number;
  teams: string[];
  entries: Array<
    [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ]
  >;
};
