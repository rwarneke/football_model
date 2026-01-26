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

  const filtered = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return rows;
    }
    return rows.filter((row) => row.team.toLowerCase().includes(normalized));
  }, [query, rows]);

  return (
    <div className="space-y-4">
      <div className="flex w-full flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-slate-500">
          <span className="font-mono text-slate-700">{filtered.length}</span> shown
        </div>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter teams"
          className="w-full rounded-md bg-white px-3 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 md:w-64"
        />
      </div>
      <WorldCupProbabilitiesTable columns={columns} rows={filtered} />
    </div>
  );
}
