import type { Metadata } from "next";
import { WorldCupOptionPricingPage } from "@/components/world-cup-option-pricing-page";
import { loadWorldCupOptionPricing } from "@/lib/world-cup";

export const metadata: Metadata = {
  title: "World Cup 2026 Team Value Pricing",
  robots: {
    index: false,
    follow: false,
  },
};

export default async function OptionsPricingPage() {
  const pricing = await loadWorldCupOptionPricing();
  const lastUpdated = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

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
          <div className="flex items-center gap-4 text-sm text-ink-400">
            <span>Updated {lastUpdated}</span>
          </div>
        </header>

        <WorldCupOptionPricingPage {...pricing} />
      </div>
    </main>
  );
}
