"use client";

import * as React from "react";
import { RatingsTable } from "@/components/ratings-table";
import type { RatingRow } from "@/lib/ratings";

type RatingsPageProps = {
  data: RatingRow[];
};

export function RatingsPage({ data }: RatingsPageProps) {
  const [query, setQuery] = React.useState("");

  const dataWithRank = React.useMemo(
    () => data.map((row, index) => ({ ...row, rank: index + 1 })),
    [data]
  );

  const filtered = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return dataWithRank;
    }
    return dataWithRank.filter((row) =>
      row.team.toLowerCase().includes(normalized)
    );
  }, [dataWithRank, query]);

  return (
    <div className="space-y-4">
      <div className="flex w-full flex-wrap items-center justify-start gap-3">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search teams"
          className="w-full rounded-md bg-white px-3 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 md:w-64"
        />
      </div>
      <RatingsTable data={filtered} />
    </div>
  );
}
