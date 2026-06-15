"use client";

import * as React from "react";
import Image from "next/image";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  SortingState,
} from "@tanstack/react-table";

import {
  TableCell,
  TableHead,
  TableRow,
} from "@/components/ui/table";
import type { RatingRow } from "@/lib/ratings";
import {
  formatRatingValue,
  formatTiltValue,
  ratingPillStyle,
  tiltPillStyle,
} from "@/lib/rating-display";

const SKIP_INITIALS = new Set(["and", "of", "the"]);

function teamInitials(team: string) {
  const letters = team
    .split(/\s+/)
    .filter((word) => word && !SKIP_INITIALS.has(word.toLowerCase()))
    .map((word) => word[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();
  return letters || team.slice(0, 2).toUpperCase();
}

type RatingsTableProps = {
  data: Array<RatingRow & { rank?: number }>;
};

function MetricPill({
  value,
  formatter,
  style,
}: {
  value: number | null | undefined;
  formatter: (value: number | null | undefined) => string;
  style: React.CSSProperties;
}) {
  return (
    <span
      className="inline-flex min-w-[3.95rem] items-center justify-center rounded-full border px-1.5 py-1 text-[10px] font-mono font-semibold tabular-nums leading-none text-slate-700"
      style={style}
    >
      {formatter(value)}
    </span>
  );
}

export function RatingsTable({ data }: RatingsTableProps) {
  const primarySortId = "rating";
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: primarySortId, desc: true },
  ]);

  const standardDescForColumn = React.useCallback((columnId: string) => {
    if (columnId === "team" || columnId === "rank") {
      return false;
    }
    return true;
  }, []);

  const primarySorting = React.useCallback(
    () => [{ id: primarySortId, desc: standardDescForColumn(primarySortId) }],
    [primarySortId, standardDescForColumn]
  );

  const handleSortToggle = React.useCallback(
    (columnId: string) => {
      const primary = sorting[0];
      if (!primary || primary.id !== columnId) {
        const nextDesc = standardDescForColumn(columnId);
        if (columnId === primarySortId) {
          setSorting(primarySorting());
          return;
        }
        setSorting([{ id: columnId, desc: nextDesc }]);
        return;
      }

      const standardDesc = standardDescForColumn(columnId);
      if (primary.desc === standardDesc) {
        setSorting([{ id: columnId, desc: !standardDesc }]);
        return;
      }

      setSorting(primarySorting());
    },
    [primarySortId, primarySorting, sorting, standardDescForColumn]
  );

  const columns = React.useMemo<
    ColumnDef<RatingRow & { rank?: number }>[]
  >(
    () => [
      {
        id: "rank",
        header: "#",
        accessorFn: (row, index) => row.rank ?? index + 1,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        meta: { minWidthCh: 3 },
        cell: ({ row }) => (
          <span className="text-xs sm:text-sm font-mono tabular-nums text-slate-700">
            {row.original.rank ?? row.index + 1}
          </span>
        ),
      },
      {
        id: "team",
        header: () => (
          <span className="block pl-[calc(1.25rem+0.5rem)] sm:pl-[calc(1.5rem+0.625rem)]">
            Team
          </span>
        ),
        accessorFn: (row) => row.team,
        sortingFn: (a, b, id) => {
          const teamA = String(a.getValue(id) ?? "").toLowerCase();
          const teamB = String(b.getValue(id) ?? "").toLowerCase();
          return teamA.localeCompare(teamB);
        },
        cell: ({ row }) => (
          <div className="flex w-full items-center gap-2 sm:gap-2.5">
            <div className="relative h-3.5 w-5 shrink-0 overflow-hidden rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)] sm:h-4 sm:w-6">
              {row.original.flagPath ? (
                <Image
                  src={row.original.flagPath}
                  alt={`${row.original.team} flag`}
                  fill
                  className="object-cover"
                  sizes="24px"
                />
              ) : (
                <span className="flex h-full w-full items-center justify-center text-[8px] font-semibold uppercase text-slate-500 sm:text-[9px]">
                  {teamInitials(row.original.team)}
                </span>
              )}
            </div>
            <span className="block min-w-0 truncate text-xs font-medium text-slate-900 sm:text-sm">
              {row.original.team}
            </span>
          </div>
        ),
      },
      {
        id: "rating",
        header: () => (
          <span className="block text-center whitespace-nowrap">
            <span className="md:hidden">Rating</span>
            <span className="hidden md:inline">Rating</span>
          </span>
        ),
        accessorFn: (row) => row.rating ?? Number.NaN,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        sortDescFirst: true,
        meta: { isRating: true },
        cell: ({ row }) => (
          <div className="flex justify-center">
            <MetricPill
              value={row.original.rating}
              formatter={formatRatingValue}
              style={ratingPillStyle(row.original.rating)}
            />
          </div>
        ),
      },
      {
        id: "tilt",
        header: () => (
          <span className="block text-center whitespace-nowrap">
            <span className="md:hidden">Tilt</span>
            <span className="hidden md:inline">Tilt</span>
          </span>
        ),
        accessorFn: (row) => row.tilt ?? Number.NaN,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        sortDescFirst: true,
        meta: { isRating: true },
        cell: ({ row }) => (
          <div className="flex justify-center">
            <MetricPill
              value={row.original.tilt}
              formatter={formatTiltValue}
              style={tiltPillStyle(row.original.tilt)}
            />
          </div>
        ),
      },
    ],
    []
  );

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableMultiSort: true,
  });

  return (
    <div className="min-w-0 w-full overflow-clip rounded-xl bg-white ring-1 ring-slate-200 shadow-sm">
      <div className="table-scroll overflow-x-auto">
        <table className="w-max min-w-full table-fixed text-sm [--rating-col-width:clamp(5.2ch,7vw,7.5ch)] sm:[--rating-col-width:clamp(5.3ch,5.5vw,7.9ch)] [--rank-col-width:2.5rem] sm:[--rank-col-width:3rem] [--team-col-width:10.5rem] sm:[--team-col-width:14rem] md:[--team-col-width:16rem] lg:[--team-col-width:18rem]">
          <colgroup>
            <col style={{ width: "var(--rank-col-width)" }} />
            <col style={{ width: "var(--team-col-width)" }} />
            <col style={{ width: "var(--rating-col-width)" }} />
            <col style={{ width: "var(--rating-col-width)" }} />
          </colgroup>
          <thead className="sticky top-0 z-[50] border-b border-slate-200 bg-slate-200">
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow
                key={headerGroup.id}
                className="bg-slate-200 border-b border-slate-200"
              >
                {headerGroup.headers.map((header, index) => {
                  const isLastHeader = index === headerGroup.headers.length - 1;
                  const columnMeta = header.column.columnDef.meta as
                    | { minWidthCh?: number; isRating?: boolean }
                    | undefined;
                  return (
                  <TableHead
                    key={header.id}
                    className={`relative select-none ${
                      header.column.getCanSort()
                        ? "cursor-pointer hover:text-slate-900"
                        : "cursor-default"
                    } ${
                      header.id === "rank"
                        ? "text-right w-[var(--rank-col-width)] min-w-[var(--rank-col-width)] pr-2 sm:pr-3"
                        : header.id === "team"
                        ? "text-left w-[var(--team-col-width)] min-w-[var(--team-col-width)] max-w-[var(--team-col-width)]"
                        : columnMeta?.isRating
                        ? "text-center"
                        : "text-right"
                    } px-1 sm:px-2 py-1 sm:py-2 text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600 ${
                      header.id === "rank"
                        ? "sticky left-0 z-10 bg-slate-200 rounded-tl-xl pr-2 sm:pr-3"
                        : header.id === "team"
                        ? "pl-0 sm:pl-0"
                        : ""
                    } ${isLastHeader ? "rounded-tr-xl" : ""}`}
                    onClick={
                      header.column.getCanSort()
                        ? () => handleSortToggle(header.id)
                        : undefined
                    }
                    style={
                      columnMeta?.minWidthCh
                        ? {
                            minWidth: `${columnMeta.minWidthCh}ch`,
                          }
                        : columnMeta?.isRating
                        ? {
                            minWidth: "var(--rating-col-width)",
                            width: "var(--rating-col-width)",
                            maxWidth: "var(--rating-col-width)",
                          }
                        : undefined
                    }
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                  </TableHead>
                );
                })}
              </TableRow>
            ))}
          </thead>
          <tbody className="divide-y divide-slate-100">
            {table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className="border-b border-slate-100 transition-colors hover:bg-slate-50/70"
              >
                {row.getVisibleCells().map((cell) => {
                  const columnMeta = cell.column.columnDef.meta as
                    | { minWidthCh?: number; isRating?: boolean }
                    | undefined;
                  return (
                  <TableCell
                    key={cell.id}
                    className={`px-1 sm:px-2 py-1 sm:py-2 ${
                      cell.column.id === "rank"
                        ? "text-right w-[var(--rank-col-width)] min-w-[var(--rank-col-width)] pr-2 sm:pr-3"
                        : cell.column.id === "team"
                        ? "text-left w-[var(--team-col-width)] min-w-[var(--team-col-width)] max-w-[var(--team-col-width)] pl-0.5 sm:pl-1"
                        : "text-right"
                    } ${
                      cell.column.id === "rank"
                        ? "sticky left-0 z-10 bg-white"
                        : ""
                    }`}
                    style={{
                      ...(columnMeta?.minWidthCh
                        ? {
                            minWidth: `${columnMeta.minWidthCh}ch`,
                          }
                        : columnMeta?.isRating
                        ? {
                            minWidth: "var(--rating-col-width)",
                            width: "var(--rating-col-width)",
                            maxWidth: "var(--rating-col-width)",
                          }
                        : {}),
                    }}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                );
                })}
              </TableRow>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
