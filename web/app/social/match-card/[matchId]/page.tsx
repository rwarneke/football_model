import { notFound } from "next/navigation";
import { SocialMatchCard } from "@/components/social-match-card";
import { loadWorldCupPredictorData } from "@/lib/world-cup-predictor";
import type { WinProbabilities } from "@/lib/world-cup-predictor-types";
import type { WorldCupMatch } from "@/lib/world-cup-matches";

export const runtime = "edge";

async function loadSocialCardData(): Promise<{
  matches: WorldCupMatch[];
  winProbabilities: WinProbabilities;
}> {
  const data = await loadWorldCupPredictorData();

  const qualifierMatches: WorldCupMatch[] = data.qualifiers.map((match) => ({
    id: match.id,
    date: match.date,
    stage: `${match.stage}${match.path ? ` ${match.path}` : ""}`.trim(),
    home: match.homeTeam,
    away: match.awayTeam,
    stadium: "",
    city: "",
    country: "",
    group: null,
    neutral: match.neutral,
  }));

  const groupMatches: WorldCupMatch[] = data.groupMatches.map((match) => ({
    id: String(match.id),
    date: match.date,
    stage: `Group ${match.group}`.trim(),
    home: match.homeTeam,
    away: match.awayTeam,
    stadium: match.stadium,
    city: match.city,
    country: match.country,
    group: match.group,
    neutral: null,
  }));

  const knockoutMatches: WorldCupMatch[] = data.knockoutMatches.map((match) => ({
    id: String(match.id),
    date: match.date,
    stage: match.stage,
    home: match.homeLabel,
    away: match.awayLabel,
    stadium: match.stadium,
    city: match.city,
    country: match.country,
    group: null,
    neutral: null,
  }));

  const matches = [...qualifierMatches, ...groupMatches, ...knockoutMatches].sort((a, b) =>
    a.date.localeCompare(b.date)
  );

  return {
    matches,
    winProbabilities: data.winProbabilities,
  };
}

export default async function SocialMatchCardPage({
  params,
}: {
  params: Promise<{ matchId: string }>;
}) {
  const { matchId } = await params;
  const { matches, winProbabilities } = await loadSocialCardData();
  const match = matches.find((item) => item.id === matchId);

  if (!match) {
    notFound();
  }

  return (
    <main className="m-0 flex items-start justify-start bg-white p-0">
      <div
        id="social-card-shot"
        className="inline-block overflow-hidden rounded-[16px] bg-white p-[2px]"
      >
        <SocialMatchCard match={match} winProbabilities={winProbabilities} />
      </div>
    </main>
  );
}
