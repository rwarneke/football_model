"use client";

import * as React from "react";
import Image from "next/image";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { WorldCupOptionPricing } from "@/lib/world-cup";

type WorldCupOptionPricingPageProps = WorldCupOptionPricing;

type TableRowData = WorldCupOptionPricing["rows"][number];

const moneyFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatMoney(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  return `$${moneyFormatter.format(value)}`;
}

function wrapHeaderLabel(label: string) {
  const words = label.split(" ").filter(Boolean);
  const content = words.join("\n");
  return (
    <span className="block w-full whitespace-pre-line 2xl:whitespace-nowrap">
      {content}
    </span>
  );
}

export function WorldCupOptionPricingPage({
  strikes,
  rows,
}: WorldCupOptionPricingPageProps) {
  const [query, setQuery] = React.useState("");

  const filtered = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return rows;
    }
    return rows.filter((row) => row.team.toLowerCase().includes(normalized));
  }, [query, rows]);

  const standardDescForColumn = React.useCallback((columnId: string) => {
    if (columnId === "team" || columnId === "group" || columnId === "flag") {
      return false;
    }
    return true;
  }, []);

  const primarySortId = "totalFairValue";
  const primarySorting = React.useMemo<SortingState>(
    () => [{ id: primarySortId, desc: true }],
    []
  );
  const [sorting, setSorting] = React.useState<SortingState>(() => primarySorting);

  const handleSortToggle = React.useCallback(
    (columnId: string) => {
      const primary = sorting[0];
      if (!primary || primary.id !== columnId) {
        setSorting([{ id: columnId, desc: standardDescForColumn(columnId) }]);
        return;
      }

      const standardDesc = standardDescForColumn(columnId);
      if (primary.desc === standardDesc) {
        setSorting([{ id: columnId, desc: !standardDesc }]);
        return;
      }

      setSorting(primarySorting);
    },
    [primarySorting, sorting, standardDescForColumn]
  );

  const tableColumns = React.useMemo<ColumnDef<TableRowData>[]>(
    () => [
      {
        id: "flag",
        header: "",
        accessorFn: (row) => row.flagPath,
        meta: { minWidthCh: 2.5, isFlag: true },
        cell: ({ row }) => (
          <div className="flex pl-1 xl:pl-2 w-full">
            <div className="relative h-3.5 w-5 xl:h-4 xl:w-6 shrink-0 overflow-hidden rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
              <Image
                src={row.original.flagPath}
                alt={`${row.original.team} flag`}
                fill
                className="object-cover"
                sizes="24px"
              />
            </div>
          </div>
        ),
      },
      {
        id: "team",
        header: () => wrapHeaderLabel("Team"),
        accessorFn: (row) => row.team,
        sortingFn: (a, b, id) =>
          String(a.getValue(id) ?? "")
            .toLowerCase()
            .localeCompare(String(b.getValue(id) ?? "").toLowerCase()),
        cell: ({ row }) => (
          <span className="min-w-0 truncate text-xs xl:text-sm font-medium text-slate-900">
            {row.original.team}
          </span>
        ),
      },
      {
        id: "group",
        header: () => (
          <span className="whitespace-nowrap">
            <span className="xl:hidden">Gr.</span>
            <span className="hidden xl:inline">Group</span>
          </span>
        ),
        accessorFn: (row) => row.group ?? "",
        meta: { isGroup: true },
        sortingFn: (a, b, id) =>
          String(a.getValue(id) ?? "").localeCompare(String(b.getValue(id) ?? "")),
        cell: ({ row }) => (
          <span className="font-mono text-xs xl:text-sm text-slate-700">
            {row.original.group ?? "—"}
          </span>
        ),
      },
      {
        id: "progressionFairValue",
        header: () => wrapHeaderLabel("Progression"),
        accessorFn: (row) => row.progressionFairValue,
        meta: { isValue: true },
        sortingFn: (a, b, id) => Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        cell: ({ row }) => (
          <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {formatMoney(row.original.progressionFairValue)}
          </span>
        ),
      },
      {
        id: "winFairValue",
        header: () => wrapHeaderLabel("90' Wins"),
        accessorFn: (row) => row.winFairValue,
        meta: { isValue: true },
        sortingFn: (a, b, id) => Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        cell: ({ row }) => (
          <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {formatMoney(row.original.winFairValue)}
          </span>
        ),
      },
      {
        id: "totalFairValue",
        header: () => wrapHeaderLabel("Total"),
        accessorFn: (row) => row.totalFairValue,
        meta: { isValue: true, emphasize: true },
        sortingFn: (a, b, id) => Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        cell: ({ row }) => (
          <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-900 font-semibold whitespace-nowrap">
            {formatMoney(row.original.totalFairValue)}
          </span>
        ),
      },
      ...strikes.map(
        (strike) =>
          ({
            id: `call-${strike}`,
            header: () => <span className="whitespace-nowrap">C{strike}</span>,
            accessorFn: (row: TableRowData) => row.calls[String(strike)] ?? 0,
            meta: { isValue: true },
            sortingFn: (a, b, id) =>
              Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
            cell: ({ row }: { row: { original: TableRowData } }) => (
              <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
                {formatMoney(row.original.calls[String(strike)])}
              </span>
            ),
          }) satisfies ColumnDef<TableRowData>
      ),
      ...strikes.map(
        (strike) =>
          ({
            id: `put-${strike}`,
            header: () => <span className="whitespace-nowrap">P{strike}</span>,
            accessorFn: (row: TableRowData) => row.puts[String(strike)] ?? 0,
            meta: { isValue: true },
            sortingFn: (a, b, id) =>
              Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
            cell: ({ row }: { row: { original: TableRowData } }) => (
              <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
                {formatMoney(row.original.puts[String(strike)])}
              </span>
            ),
          }) satisfies ColumnDef<TableRowData>
      ),
    ],
    [strikes]
  );

  const table = useReactTable({
    data: filtered,
    columns: tableColumns,
    state: { sorting },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableMultiSort: false,
  });

  return (
    <div className="space-y-4">
      <div className="flex w-full items-center gap-3">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search teams"
          className="min-w-0 w-full max-w-[25rem] flex-1 rounded-md bg-white px-3 py-1.5 text-sm text-slate-700 ring-1 ring-slate-200 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 md:w-64"
        />
      </div>

      <div className="min-w-0 w-full overflow-clip rounded-xl bg-white ring-1 ring-slate-200 shadow-sm">
        <div className="table-scroll overflow-x-auto">
          <table className="w-full table-auto xl:table-fixed text-sm [--value-col-width:clamp(7ch,8vw,11ch)]">
            <thead className="sticky top-0 z-[50] border-b border-slate-200 bg-slate-200">
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id} className="bg-slate-200 border-b border-slate-200">
                  {headerGroup.headers.map((header, index) => {
                    const isLastHeader = index === headerGroup.headers.length - 1;
                    const columnMeta = header.column.columnDef.meta as
                      | {
                          minWidthCh?: number;
                          isGroup?: boolean;
                          isValue?: boolean;
                        }
                      | undefined;
                    return (
                      <TableHead
                        key={header.id}
                        className={`relative cursor-pointer select-none hover:text-slate-900 ${
                          header.id === "flag"
                            ? "text-left w-[3rem] min-w-[3rem] pl-0.5 xl:pl-1 pr-1 xl:pr-2"
                            : header.id === "team"
                              ? "text-left w-[10rem] min-w-[7rem] xl:min-w-[10rem] shrink-0"
                              : columnMeta?.isGroup
                                ? "text-center whitespace-nowrap min-w-[3ch] xl:min-w-[4ch]"
                                : "text-right"
                        } px-1 xl:px-2 py-1.5 xl:py-2.5 text-[10px] xl:text-[11px] font-semibold uppercase tracking-wide text-slate-600 whitespace-normal 2xl:whitespace-nowrap ${
                          header.id === "flag" ? "sticky left-0 z-10 bg-slate-200 rounded-tl-xl" : ""
                        } ${isLastHeader ? "rounded-tr-xl" : ""}`}
                        onClick={() => handleSortToggle(header.id)}
                        style={
                          columnMeta?.minWidthCh
                            ? { minWidth: `${columnMeta.minWidthCh}ch` }
                            : columnMeta?.isValue
                              ? {
                                  minWidth: "var(--value-col-width)",
                                  maxWidth: "calc(var(--value-col-width) * 1.6)",
                                }
                              : undefined
                        }
                      >
                        <span className="block w-full">
                          {flexRender(header.column.columnDef.header, header.getContext())}
                        </span>
                      </TableHead>
                    );
                  })}
                </TableRow>
              ))}
            </thead>
            <TableBody className="divide-y divide-slate-100">
              {table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  className="border-b border-slate-100 transition-colors hover:bg-slate-50/70"
                >
                  {row.getVisibleCells().map((cell) => {
                    const columnMeta = cell.column.columnDef.meta as
                      | {
                          minWidthCh?: number;
                          isGroup?: boolean;
                          isValue?: boolean;
                          emphasize?: boolean;
                        }
                      | undefined;
                    return (
                      <TableCell
                        key={cell.id}
                        className={`px-1 xl:px-2 py-1.5 xl:py-2.5 ${
                          cell.column.id === "flag"
                            ? "text-left w-[3rem] min-w-[3rem] pl-0.5 xl:pl-1 pr-1.5 xl:pr-2.5 overflow-hidden"
                            : cell.column.id === "team"
                              ? "text-left w-[10rem] min-w-[7rem] xl:min-w-[10rem] shrink-0 pl-0.5 xl:pl-1"
                              : columnMeta?.isGroup
                                ? "text-center"
                                : "text-right"
                        } ${cell.column.id === "flag" ? "sticky left-0 z-10 bg-white" : ""}`}
                        style={
                          columnMeta?.minWidthCh
                            ? { minWidth: `${columnMeta.minWidthCh}ch` }
                            : columnMeta?.isValue
                              ? {
                                  minWidth: "var(--value-col-width)",
                                  maxWidth: "calc(var(--value-col-width) * 1.6)",
                                }
                              : undefined
                        }
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))}
              {table.getRowModel().rows.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={table.getAllLeafColumns().length}
                    className="px-3 py-8 text-center text-sm text-slate-500"
                  >
                    No teams match that search.
                  </TableCell>
                </TableRow>
              ) : null}
            </TableBody>
          </table>
        </div>
      </div>
    </div>
  );
}
