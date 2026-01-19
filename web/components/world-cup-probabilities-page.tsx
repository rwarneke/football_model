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
      <div className="flex w-full items-center justify-between gap-4">
        <div className="text-sm text-ink-400">
          <span className="font-mono">{filtered.length}</span> shown
        </div>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter teams"
          className="w-full rounded-lg border border-ink-900 bg-white px-3 py-1 text-sm text-ebony placeholder:text-ink-900/60 md:w-64"
        />
      </div>
      <WorldCupProbabilitiesTable columns={columns} rows={filtered} />
    </div>
  );
}
