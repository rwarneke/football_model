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

const ratingFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

type RatingsTableProps = {
  data: RatingRow[];
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
        cell: ({ row }) => (
          <span className="text-ink-200/80">{row.index + 1}</span>
        ),
      },
      {
        accessorKey: "team",
        header: "Team",
        cell: ({ row }) => (
          <div className="flex items-center gap-3">
            <div className="relative h-5 w-7 overflow-hidden rounded-sm border border-ink-700 bg-ink-800">
              <Image
                src={row.original.flagPath}
                alt={`${row.original.team} flag`}
                fill
                className="object-cover"
                sizes="28px"
              />
            </div>
            <span className="text-white">{row.original.team}</span>
          </div>
        ),
      },
      {
        accessorKey: "rating",
        header: "Overall",
        cell: ({ getValue }) => ratingFormatter.format(getValue<number>()),
      },
      {
        accessorKey: "rating_attack",
        header: "Attack",
        cell: ({ getValue }) => numberFormatter.format(getValue<number>()),
      },
      {
        accessorKey: "rating_defense",
        header: "Defense",
        cell: ({ getValue }) => numberFormatter.format(getValue<number>()),
      },
      {
        accessorKey: "quality",
        header: "Form",
        cell: ({ getValue }) => numberFormatter.format(getValue<number>()),
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
    <div className="rounded-2xl border border-ink-800 bg-ink-800/60 shadow-soft">
      <div className="table-scroll overflow-x-auto">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    className={cn(
                      "cursor-pointer select-none",
                      header.column.getCanSort() && "hover:text-white"
                    )}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                    {header.column.getIsSorted() === "asc" && (
                      <span className="ml-2 text-ink-400">↑</span>
                    )}
                    {header.column.getIsSorted() === "desc" && (
                      <span className="ml-2 text-ink-400">↓</span>
                    )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
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
