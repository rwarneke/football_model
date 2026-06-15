"use client";

import * as React from "react";
import { RatingsTable } from "@/components/ratings-table";
import type { RatingRow } from "@/lib/ratings";

type RatingsPageProps = {
  data: RatingRow[];
};

export function RatingsPage({ data }: RatingsPageProps) {
  const [query, setQuery] = React.useState("");
  const [selectedConfederation, setSelectedConfederation] = React.useState<
    string | null
  >(null);

  const dataWithRank = React.useMemo(
    () => data.map((row, index) => ({ ...row, rank: index + 1 })),
    [data]
  );

  const confederations = React.useMemo(() => {
    const unique = new Set(
      dataWithRank
        .map((row) => row.confederation)
        .filter((value): value is string => Boolean(value))
    );
    return Array.from(unique).sort((a, b) => a.localeCompare(b));
  }, [dataWithRank]);

  const filtered = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return selectedConfederation
        ? dataWithRank.filter(
            (row) => row.confederation === selectedConfederation
          )
        : dataWithRank;
    }
    return dataWithRank.filter((row) => {
      if (selectedConfederation && row.confederation !== selectedConfederation) {
        return false;
      }
      return row.team.toLowerCase().includes(normalized);
    });
  }, [dataWithRank, query, selectedConfederation]);

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
      {confederations.length ? (
        <div className="flex flex-wrap items-center gap-1.5">
          {confederations.map((confederation) => {
            const isSelected = selectedConfederation === confederation;
            return (
              <button
                key={confederation}
                type="button"
                onClick={() =>
                  setSelectedConfederation((current) =>
                    current === confederation ? null : confederation
                  )
                }
                className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.06em] transition ${
                  isSelected
                    ? "border-emerald-500 bg-emerald-50 text-emerald-700"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-800"
                }`}
                aria-pressed={isSelected}
              >
                {confederation}
              </button>
            );
          })}
        </div>
      ) : null}
      <RatingsTable data={filtered} />
    </div>
  );
}
