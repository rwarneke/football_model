"use client";

import * as React from "react";
import { WorldCupProbabilitiesTable } from "@/components/world-cup-probabilities-table";
import type { WorldCupProbabilities } from "@/lib/world-cup";

type WorldCupProbabilitiesPageProps = {
  current: WorldCupProbabilities;
  pretournament: WorldCupProbabilities;
  currentUpdatedLabel: string;
  pretournamentUpdatedLabel: string;
};

export function WorldCupProbabilitiesPage({
  current,
  pretournament,
  currentUpdatedLabel,
  pretournamentUpdatedLabel,
}: WorldCupProbabilitiesPageProps) {
  const [query, setQuery] = React.useState("");
  const [showPretournament, setShowPretournament] = React.useState(false);
  const [probabilityMode, setProbabilityMode] = React.useState<"percent" | "decimal">(
    "percent"
  );
  const active = showPretournament ? pretournament : current;
  const updatedLabel = showPretournament
    ? pretournamentUpdatedLabel
    : currentUpdatedLabel;
  const showingCurrent = !showPretournament;

  const filtered = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return active.rows;
    }
    return active.rows.filter((row) => row.team.toLowerCase().includes(normalized));
  }, [active.rows, query]);

  return (
    <div className="space-y-4">
      <div className="flex w-full items-center gap-3">
        <button
          type="button"
          onClick={() => setShowPretournament((prev) => !prev)}
          role="switch"
          aria-checked={showingCurrent}
          aria-label={showingCurrent ? "Current" : "Pre-tournament"}
          className={`relative h-6 w-11 shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 ${
            showingCurrent ? "bg-slate-900" : "bg-slate-300"
          }`}
        >
          <span
            aria-hidden="true"
            className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
              showingCurrent ? "translate-x-5" : "translate-x-0"
            }`}
          />
        </button>
        <span className="shrink-0 text-sm text-slate-700">
          {showingCurrent ? "Current" : "Pre-tournament"}
        </span>
      </div>
      <div className="flex w-full items-center gap-3">
        <span className="shrink-0 text-sm text-ink-400">Updated {updatedLabel}</span>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search teams"
          className="min-w-0 w-full max-w-[25rem] flex-1 rounded-md bg-white px-3 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 md:w-64"
        />
        <div className="ml-auto flex w-40 shrink-0 items-center gap-2">
          <select
            value={probabilityMode}
            onChange={(event) =>
              setProbabilityMode(event.target.value as "percent" | "decimal")
            }
            className="w-full rounded-md bg-white px-2.5 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
          >
            <option value="percent">% Chance</option>
            <option value="decimal">Decimal Odds</option>
          </select>
        </div>
      </div>
      <WorldCupProbabilitiesTable
        columns={active.columns}
        rows={filtered}
        allRows={active.rows}
        probabilityMode={probabilityMode}
      />
    </div>
  );
}
