export type GroupDefinition = {
  id: string;
  teams: string[];
};

export type GroupMatch = {
  id: number;
  date: string;
  group: string;
  homeTeam: string;
  awayTeam: string;
  stadium: string;
  city: string;
  country: string;
};

export type KnockoutMatch = {
  id: number;
  stage: string;
  date: string;
  homeLabel: string;
  awayLabel: string;
  stadium: string;
  city: string;
  country: string;
};

export type QualifierMatch = {
  id: string;
  date: string;
  stage: string;
  path: string;
  round: string;
  homeTeam: string;
  awayTeam: string;
  homeSource: string;
  awaySource: string;
  winnerSlot: string;
  neutral: boolean;
};

export type RoundOf32Combos = Record<string, Record<string, string>>;

export type WinProbabilityEntry = {
  p_home?: number;
  p_draw?: number;
  p_away?: number;
  p_home_pens?: number;
  p_away_pens?: number;
  score_matrix?: number[][];
  nu?: number;
  lam_home?: number;
  lam_away?: number;
};

export type LegacyWinProbabilities = Record<
  string,
  Record<string, { home?: WinProbabilityEntry; neutral?: WinProbabilityEntry }>
>;

export type CompactWinProbabilities = {
  version: number;
  max_goals: number;
  teams: string[];
  entries: number[][];
};

export type WinProbabilities = LegacyWinProbabilities | CompactWinProbabilities;

export type TeamStageProbabilities = {
  stage_probability: Record<string, number>;
  group_stage_rank_probability: Record<string, number>;
  [key: string]: Record<string, number>;
};

export type WorldCupPredictorData = {
  groups: GroupDefinition[];
  groupMatches: GroupMatch[];
  knockoutMatches: KnockoutMatch[];
  roundOf32Combos: RoundOf32Combos;
  qualifiers: QualifierMatch[];
  flags: Record<string, string | null>;
  winProbabilities: WinProbabilities;
  simulationTeamProbabilities: Record<string, TeamStageProbabilities>;
};
