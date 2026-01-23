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
import { cn } from "@/lib/utils";
import type { RatingRow } from "@/lib/ratings";

const numberFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const ACCENT_DARK_RGB = "189, 110, 109";

function ratingBackground(value: number) {
  if (!Number.isFinite(value)) {
    return undefined;
  }
  const clamped = Math.max(0, Math.min(value, 100));
  let alpha = 0;
  if (clamped <= 90) {
    alpha = (clamped / 90) * 0.3;
  } else {
    alpha = 0.3 + ((clamped - 90) / 10) * 0.1;
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

  const columns = React.useMemo<ColumnDef<RatingRow>[]>(
    () => [
      {
        id: "rank",
        header: "Rank",
        accessorFn: (row) => row.rank ?? row.rating_rank,
        sortingFn: (a, b, id) => (b.getValue(id) ?? 0) - (a.getValue(id) ?? 0),
        cell: ({ row }) => (
          <span className="text-ink-200/80">
            {row.original.rank ?? row.index + 1}
          </span>
        ),
      },
      {
        accessorKey: "team",
        header: () => (
          <div className="flex items-center gap-3">
            <span className="inline-block h-5 w-7" aria-hidden="true" />
            <span>Team</span>
          </div>
        ),
        cell: ({ row }) => (
          <div className="flex items-center gap-3">
            <div className="relative h-5 w-7 shrink-0 overflow-hidden rounded-[1px] border border-ink-700 bg-ink-800">
              {row.original.flagPath ? (
                <Image
                  src={row.original.flagPath}
                  alt={`${row.original.team} flag`}
                  fill
                  className="object-cover"
                  sizes="28px"
                />
              ) : (
                <span className="flex h-full w-full items-center justify-center text-[10px] font-semibold uppercase tracking-wide text-ink-300">
                  {teamInitials(row.original.team)}
                </span>
              )}
            </div>
            <span className="text-ebony">{row.original.team}</span>
          </div>
        ),
      },
      {
        accessorKey: "rating",
        header: "Overall",
        cell: ({ getValue }) => (
          <span className="font-mono">
            {numberFormatter.format(getValue<number>())}
          </span>
        ),
      },
      {
        accessorKey: "rating_attack",
        header: "Attack",
        cell: ({ getValue }) => (
          <span className="font-mono">
            {numberFormatter.format(getValue<number>())}
          </span>
        ),
      },
      {
        accessorKey: "rating_defense",
        header: "Defense",
        cell: ({ getValue }) => (
          <span className="font-mono">
            {numberFormatter.format(getValue<number>())}
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
  });

  return (
    <div className="min-w-0 w-full overflow-hidden lg:rounded-xl lg:bg-white lg:ring-1 lg:ring-slate-200 lg:shadow-sm">
      <div className="table-scroll overflow-x-auto">
        <Table className="table-fixed w-full">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    className={cn(
                      "h-9 cursor-pointer select-none py-1.5",
                      header.column.getCanSort() && "hover:text-ebony",
                      "px-2",
                      header.id === "rank" && "w-10 sm:w-12 md:w-16",
                      header.id === "team" && "w-40 sm:w-56 md:w-80",
                      (header.id === "rating" ||
                        header.id === "rating_attack" ||
                        header.id === "rating_defense") &&
                        "w-16 sm:w-20 md:w-24 text-right px-1.5"
                    )}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.map((row) => (
              <TableRow key={row.id} className="h-9">
                {row.getVisibleCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    className={cn(
                      "px-2 py-1.5",
                      (cell.column.id === "rating" ||
                        cell.column.id === "rating_attack" ||
                        cell.column.id === "rating_defense") &&
                        "text-right px-1.5"
                    )}
                    style={
                      cell.column.id === "rating" ||
                      cell.column.id === "rating_attack" ||
                      cell.column.id === "rating_defense"
                        ? ratingBackground(cell.getValue<number>())
                        : undefined
                    }
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
