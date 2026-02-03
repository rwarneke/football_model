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

const ratingFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const ACCENT_DARK_RGB = "16, 185, 129";

function ratingBackground(value: number) {
  if (!Number.isFinite(value)) {
    return undefined;
  }
  const clamped = Math.max(0, Math.min(value, 100));
  let alpha = 0;
  if (clamped <= 50) {
    alpha = (clamped / 50) * 0.05;
  } else if (clamped <= 90) {
    alpha = 0.05 + ((clamped - 50) / 40) * 0.18;
  } else {
    const scaled = (clamped - 90) / 10;
    alpha = 0.3 + Math.pow(scaled, 1.4) * 0.25;
  }
  return { backgroundColor: `rgba(${ACCENT_DARK_RGB}, ${alpha})` };
}

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
        id: "flag",
        header: "",
        accessorFn: (row) => row.flagPath ?? "",
        enableSorting: false,
        meta: { minWidthCh: 2.5, isFlag: true, width: "2rem" },
        cell: ({ row }) => (
          <div className="flex pl-0.5 sm:pl-1 w-full">
            <div className="relative h-3.5 w-5 sm:h-4 sm:w-6 shrink-0 overflow-hidden rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
              {row.original.flagPath ? (
                <Image
                  src={row.original.flagPath}
                  alt={`${row.original.team} flag`}
                  fill
                  className="object-cover"
                  sizes="24px"
                />
              ) : (
                <span className="flex h-full w-full items-center justify-center text-[8px] sm:text-[9px] font-semibold uppercase text-slate-500">
                  {teamInitials(row.original.team)}
                </span>
              )}
            </div>
          </div>
        ),
      },
      {
        id: "team",
        header: "Team",
        accessorFn: (row) => row.team,
        sortingFn: (a, b, id) => {
          const teamA = String(a.getValue(id) ?? "").toLowerCase();
          const teamB = String(b.getValue(id) ?? "").toLowerCase();
          return teamA.localeCompare(teamB);
        },
        cell: ({ row }) => (
          <span className="block whitespace-nowrap text-xs sm:text-sm font-medium text-slate-900">
            {row.original.team}
          </span>
        ),
      },
      {
        id: "rating",
        header: () => (
          <span className="whitespace-nowrap">
            <span className="md:hidden">OVR</span>
            <span className="hidden md:inline">Overall</span>
          </span>
        ),
        accessorFn: (row) => row.rating ?? Number.NaN,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        sortDescFirst: true,
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-xs sm:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {Number.isFinite(row.original.rating)
              ? ratingFormatter.format(row.original.rating ?? 0)
              : ""}
          </span>
        ),
      },
      {
        id: "rating_attack",
        header: () => (
          <span className="whitespace-nowrap">
            <span className="md:hidden">ATT</span>
            <span className="hidden md:inline">Attack</span>
          </span>
        ),
        accessorFn: (row) => row.rating_attack ?? Number.NaN,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        sortDescFirst: true,
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-xs sm:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {Number.isFinite(row.original.rating_attack)
              ? ratingFormatter.format(row.original.rating_attack ?? 0)
              : ""}
          </span>
        ),
      },
      {
        id: "rating_defense",
        header: () => (
          <span className="whitespace-nowrap">
            <span className="md:hidden">DEF</span>
            <span className="hidden md:inline">Defense</span>
          </span>
        ),
        accessorFn: (row) => row.rating_defense ?? Number.NaN,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        sortDescFirst: true,
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-xs sm:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {Number.isFinite(row.original.rating_defense)
              ? ratingFormatter.format(row.original.rating_defense ?? 0)
              : ""}
          </span>
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
        <table className="w-full table-auto xl:table-fixed text-sm [--rating-col-width:clamp(6ch,9vw,14ch)] sm:[--rating-col-width:clamp(7ch,9vw,16ch)]">
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
                        ? "text-right w-[2.5rem] min-w-[2.5rem] sm:w-[3rem] sm:min-w-[3rem] pr-2 sm:pr-3"
                        : header.id === "flag"
                        ? "text-left w-[2rem] min-w-[2rem] sm:w-[3rem] sm:min-w-[3rem] pl-0.5 pr-1 sm:pl-1 sm:pr-2"
                        : header.id === "team"
                        ? "text-left min-w-[8rem] sm:min-w-[10rem]"
                        : "text-right"
                    } px-1 sm:px-2 py-1.5 sm:py-2.5 text-[10px] sm:text-[11px] font-semibold uppercase tracking-wide text-slate-600 ${
                      header.id === "rank"
                        ? "sticky left-0 z-10 bg-slate-200 rounded-tl-xl pr-2 sm:pr-3"
                        : header.id === "flag"
                        ? "sticky left-[2.5rem] sm:left-[3rem] z-10 bg-slate-200"
                        : header.id === "team"
                        ? "pl-0.5 sm:pl-1"
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
                    className={`px-1 sm:px-2 py-1.5 sm:py-2.5 ${
                      cell.column.id === "rank"
                        ? "text-right w-[2.5rem] min-w-[2.5rem] sm:w-[3rem] sm:min-w-[3rem] pr-2 sm:pr-3"
                        : cell.column.id === "flag"
                        ? "text-left w-[2rem] min-w-[2rem] sm:w-[3rem] sm:min-w-[3rem] pl-0.5 pr-1.5 sm:pl-1 sm:pr-2.5 overflow-hidden"
                        : cell.column.id === "team"
                        ? "text-left min-w-[8rem] sm:min-w-[10rem] pl-0.5 sm:pl-1"
                        : "text-right"
                    } ${
                      cell.column.id === "rank"
                        ? "sticky left-0 z-10 bg-white"
                        : cell.column.id === "flag"
                        ? "sticky left-[2.5rem] sm:left-[3rem] z-10 bg-white"
                        : ""
                    }`}
                    style={{
                      ...(columnMeta?.isRating
                        ? ratingBackground(cell.getValue<number>())
                        : {}),
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
