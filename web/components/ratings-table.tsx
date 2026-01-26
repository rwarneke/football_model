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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
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
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: "rating", desc: true },
  ]);

  const columns = React.useMemo<
    ColumnDef<RatingRow & { rank?: number }>[]
  >(
    () => [
      {
        id: "flag",
        header: "",
        accessorFn: (row) => row.flagPath ?? "",
        enableSorting: false,
        meta: { minWidthCh: 2.5, isFlag: true, width: "2.5rem" },
        cell: ({ row }) => (
          <div className="flex pl-2 w-full">
            <div className="relative h-4 w-6 shrink-0 overflow-hidden rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
              {row.original.flagPath ? (
                <Image
                  src={row.original.flagPath}
                  alt={`${row.original.team} flag`}
                  fill
                  className="object-cover"
                  sizes="24px"
                />
              ) : (
                <span className="flex h-full w-full items-center justify-center text-[9px] font-semibold uppercase text-slate-500">
                  {teamInitials(row.original.team)}
                </span>
              )}
            </div>
          </div>
        ),
      },
      {
        id: "rank",
        header: "Rank",
        accessorFn: (row) => row.rank ?? row.rating_rank,
        sortingFn: (a, b, id) => (b.getValue(id) ?? 0) - (a.getValue(id) ?? 0),
        cell: ({ row }) => (
          <span className="text-sm font-mono tabular-nums text-slate-700">
            {row.original.rank ?? row.index + 1}
          </span>
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
          <span className="min-w-0 truncate text-sm font-medium text-slate-900">
            {row.original.team}
          </span>
        ),
      },
      {
        id: "rating",
        header: "Overall",
        accessorFn: (row) => row.rating ?? Number.NaN,
        sortingFn: (a, b, id) => (a.getValue(id) ?? 0) - (b.getValue(id) ?? 0),
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {Number.isFinite(row.original.rating)
              ? ratingFormatter.format(row.original.rating ?? 0)
              : ""}
          </span>
        ),
      },
      {
        id: "rating_attack",
        header: "Attack",
        accessorFn: (row) => row.rating_attack ?? Number.NaN,
        sortingFn: (a, b, id) => (a.getValue(id) ?? 0) - (b.getValue(id) ?? 0),
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {Number.isFinite(row.original.rating_attack)
              ? ratingFormatter.format(row.original.rating_attack ?? 0)
              : ""}
          </span>
        ),
      },
      {
        id: "rating_defense",
        header: "Defense",
        accessorFn: (row) => row.rating_defense ?? Number.NaN,
        sortingFn: (a, b, id) => (a.getValue(id) ?? 0) - (b.getValue(id) ?? 0),
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
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
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableMultiSort: true,
  });

  return (
    <div className="min-w-0 w-full overflow-hidden rounded-xl bg-white ring-1 ring-slate-200 shadow-sm">
      <div className="table-scroll overflow-x-auto">
        <Table className="w-full table-auto xl:table-fixed text-sm [--rating-col-width:clamp(5ch,7vw,9ch)] sm:[--rating-col-width:clamp(6ch,7vw,10ch)]">
          <TableHeader className="border-b border-slate-200">
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow
                key={headerGroup.id}
                className="bg-slate-200 border-b border-slate-200"
              >
                {headerGroup.headers.map((header, index) => {
                  const isLastHeader = index === headerGroup.headers.length - 1;
                  return (
                  <TableHead
                    key={header.id}
                    className={`relative select-none ${
                      header.column.getCanSort()
                        ? "cursor-pointer hover:text-slate-900"
                        : "cursor-default"
                    } ${
                      header.id === "flag"
                        ? "text-left w-[3rem] min-w-[3rem] pl-1 pr-3"
                        : header.id === "team"
                        ? "text-left w-[12rem] min-w-[12rem] shrink-0"
                        : header.id === "rank"
                        ? "text-right whitespace-nowrap min-w-[4ch]"
                        : "text-right"
                    } px-2 py-2.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600 ${
                      header.id === "flag"
                        ? "sticky left-0 z-50 bg-slate-200 rounded-tl-xl"
                        : ""
                    } ${isLastHeader ? "rounded-tr-xl" : ""}`}
                    onClick={header.column.getToggleSortingHandler()}
                    style={
                      header.column.columnDef.meta?.minWidthCh
                        ? {
                            minWidth: `${header.column.columnDef.meta.minWidthCh}ch`,
                          }
                        : header.column.columnDef.meta?.isRating
                        ? {
                            minWidth: "var(--rating-col-width)",
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
          </TableHeader>
          <TableBody className="divide-y divide-slate-100">
            {table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className="border-b border-slate-100 transition-colors hover:bg-slate-50/70"
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    className={`px-2 py-2.5 ${
                      cell.column.id === "flag"
                        ? "text-left w-[3rem] min-w-[3rem] pl-1 pr-3 py-2.5 overflow-hidden"
                        : cell.column.id === "team"
                        ? "text-left w-[12rem] min-w-[12rem] shrink-0 pl-2"
                        : cell.column.id === "rank"
                        ? "text-right"
                        : "text-right"
                    } ${
                      cell.column.id === "flag"
                        ? "sticky left-0 z-40 bg-white"
                        : ""
                    }`}
                    style={{
                      ...(cell.column.columnDef.meta?.isRating
                        ? ratingBackground(cell.getValue<number>())
                        : {}),
                      ...(cell.column.columnDef.meta?.minWidthCh
                        ? {
                            minWidth: `${cell.column.columnDef.meta.minWidthCh}ch`,
                          }
                        : cell.column.columnDef.meta?.isRating
                        ? {
                            minWidth: "var(--rating-col-width)",
                          }
                        : {}),
                    }}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
