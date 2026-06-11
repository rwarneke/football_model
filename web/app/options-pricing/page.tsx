import type { Metadata } from "next";
import { stat } from "node:fs/promises";
import path from "node:path";
import { WorldCupOptionPricingPage } from "@/components/world-cup-option-pricing-page";
import { loadWorldCupOptionPricing } from "@/lib/world-cup";

export const metadata: Metadata = {
  title: "World Cup 2026 Team Value Pricing",
  robots: {
    index: false,
    follow: false,
  },
};

async function modelOutputUpdatedLabel(dirName: string) {
  const stats = await stat(
    path.join(process.cwd(), "public", dirName.replace(/^\/+/, ""), "simulation_results.csv")
  );
  return stats.mtime.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

async function loadPretournamentPricing() {
  try {
    return await loadWorldCupOptionPricing("/model_output_pretournament");
  } catch {
    return await loadWorldCupOptionPricing("/model_output");
  }
}

async function pretournamentUpdatedLabel() {
  try {
    return await modelOutputUpdatedLabel("/model_output_pretournament");
  } catch {
    return await modelOutputUpdatedLabel("/model_output");
  }
}

export default async function OptionsPricingPage() {
  const [current, pretournament, currentUpdatedLabel, pretournamentUpdated] =
    await Promise.all([
      loadWorldCupOptionPricing("/model_output"),
      loadPretournamentPricing(),
      modelOutputUpdatedLabel("/model_output"),
      pretournamentUpdatedLabel(),
    ]);

  return (
    <main className="px-2 pb-16 pt-8 lg:px-6">
      <div className="flex w-full flex-col gap-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.3em] text-ink-400">
            FIFA WORLD CUP 2026
          </p>
          <h1 className="text-3xl font-semibold text-ebony md:text-4xl">
            Team Value Pricing
          </h1>
          <p className="text-base text-ink-200">
            Fair values from tournament simulations, split into progression and 90&apos; win value,
            with call and put prices on total team value at strikes 20, 40, 60, and 80.
          </p>
        </header>

        <WorldCupOptionPricingPage
          current={current}
          pretournament={pretournament}
          currentUpdatedLabel={currentUpdatedLabel}
          pretournamentUpdatedLabel={pretournamentUpdated}
        />
      </div>
    </main>
  );
}
