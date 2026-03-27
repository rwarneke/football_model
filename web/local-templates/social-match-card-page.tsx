import { notFound } from "next/navigation";
import { loadWorldCupMatches } from "@/lib/world-cup-matches";
import type { WinProbabilities } from "@/lib/world-cup-predictor-types";
import { SocialMatchCard } from "@/components/social-match-card";
import { readFile } from "node:fs/promises";
import path from "node:path";

async function loadWinProbabilities(): Promise<WinProbabilities> {
  const filePath = path.join(process.cwd(), "public", "model_output", "win_probabilities.json");
  const contents = await readFile(filePath, "utf8");
  return JSON.parse(contents) as WinProbabilities;
}

export default async function SocialMatchCardPage({
  params,
}: {
  params: Promise<{ matchId: string }>;
}) {
  const { matchId } = await params;
  const [matches, winProbabilities] = await Promise.all([
    loadWorldCupMatches(),
    loadWinProbabilities(),
  ]);
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
