"use client";

import * as React from "react";
import { WorldCupProbabilitiesTable } from "@/components/world-cup-probabilities-table";
import type { WorldCupProbabilities } from "@/lib/world-cup";

type WorldCupProbabilitiesPageProps = WorldCupProbabilities;

export function WorldCupProbabilitiesPage({
  columns,
  rows,
}: WorldCupProbabilitiesPageProps) {
  const [query, setQuery] = React.useState("");
  const [probabilityMode, setProbabilityMode] = React.useState<"percent" | "decimal">(
    "percent"
  );

  const filtered = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return rows;
    }
    return rows.filter((row) => row.team.toLowerCase().includes(normalized));
  }, [query, rows]);

  return (
    <div className="space-y-4">
      <div className="flex w-full items-center gap-3">
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
        columns={columns}
        rows={filtered}
        probabilityMode={probabilityMode}
      />
    </div>
  );
}
