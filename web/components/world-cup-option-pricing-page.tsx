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

type WorldCupOptionPricingPageProps = {
  current: WorldCupOptionPricing;
  pretournament: WorldCupOptionPricing;
  currentUpdatedLabel: string;
  pretournamentUpdatedLabel: string;
};

type TableRowData = WorldCupOptionPricing["rows"][number];

const moneyFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const BLUE_RGB = "147, 197, 253";
const GREEN_RGB = "16, 185, 129";
const RED_RGB = "239, 68, 68";

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

function valueHeatBackground(value: number, maxValue: number, rgb: string) {
  if (!Number.isFinite(value) || !Number.isFinite(maxValue) || maxValue <= 0 || value <= 0) {
    return undefined;
  }
  const clamped = Math.max(0, Math.min(value / maxValue, 1));
  let alpha = 0;
  if (clamped <= 0.1) {
    const scaled = clamped / 0.1;
    alpha = 0.08 + Math.pow(scaled, 1.2) * 0.08;
  } else if (clamped <= 0.9) {
    const scaled = (clamped - 0.1) / 0.8;
    alpha = 0.16 + Math.pow(scaled, 1.35) * 0.54;
  } else {
    const scaled = (clamped - 0.9) / 0.1;
    alpha = 0.8 + Math.pow(scaled, 1.25) * 0.18;
  }
  return { backgroundColor: `rgba(${rgb}, ${alpha})` };
}

export function WorldCupOptionPricingPage({
  current,
  pretournament,
  currentUpdatedLabel,
  pretournamentUpdatedLabel,
}: WorldCupOptionPricingPageProps) {
  const [query, setQuery] = React.useState("");
  const [showPretournament, setShowPretournament] = React.useState(false);
  const showingCurrent = !showPretournament;
  const active = showPretournament ? pretournament : current;
  const updatedLabel = showingCurrent
    ? currentUpdatedLabel
    : pretournamentUpdatedLabel;
  const strikes = active.strikes;
  const rows = active.rows;

  const filtered = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return rows;
    }
    return rows.filter((row) => row.team.toLowerCase().includes(normalized));
  }, [query, rows]);

  const maxTotal = React.useMemo(
    () => rows.reduce((max, row) => Math.max(max, row.totalFairValue), 0),
    [rows]
  );
  const maxCall = React.useMemo(
    () =>
      rows.reduce(
        (max, row) =>
          Math.max(
            max,
            ...strikes.map((strike) => Number(row.calls[String(strike)] ?? 0))
          ),
        0
      ),
    [rows, strikes]
  );
  const maxPut = React.useMemo(
    () =>
      rows.reduce(
        (max, row) =>
          Math.max(
            max,
            ...strikes.map((strike) => Number(row.puts[String(strike)] ?? 0))
          ),
        0
      ),
    [rows, strikes]
  );

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
        meta: { isValue: true, emphasize: true, heat: "blue" },
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
            meta: { isValue: true, heat: "green", strike },
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
            meta: { isValue: true, heat: "red", strike },
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
  const isGroupedByGroup = sorting[0]?.id === "group";
  const groupBase = (value: string | null | undefined) =>
    String(value ?? "").replace(/\*/g, "");

  return (
    <div className="space-y-4">
      <div className="flex w-full items-center gap-3">
        <button
          type="button"
          onClick={() => setShowPretournament((prev) => !prev)}
          role="switch"
          aria-checked={showingCurrent}
          aria-label={showingCurrent ? "Current" : "Pre-tournament"}
          className={`relative h-6 w-11 shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 ${
            showingCurrent ? "bg-slate-900" : "bg-slate-300"
          }`}
        >
          <span
            aria-hidden="true"
            className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
              showingCurrent ? "translate-x-5" : "translate-x-0"
            }`}
          />
        </button>
        <span className="shrink-0 text-sm text-slate-700">
          {showingCurrent ? "Current" : "Pre-tournament"}
        </span>
      </div>
      <div className="flex w-full items-center gap-3">
        <span className="shrink-0 text-sm text-ink-400">Updated {updatedLabel}</span>
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
              {table.getRowModel().rows.map((row, index, allRows) => {
                const isGroupEnd =
                  isGroupedByGroup &&
                  index > 0 &&
                  index < allRows.length - 1 &&
                  groupBase(row.original.group) !==
                    groupBase(allRows[index + 1]?.original.group);
                return (
                <TableRow
                  key={row.id}
                  className={`border-b border-slate-100 transition-colors hover:bg-slate-50/70 ${
                    isGroupEnd ? "border-b-2 border-slate-200" : ""
                  }`}
                >
                  {row.getVisibleCells().map((cell) => {
                    const columnMeta = cell.column.columnDef.meta as
                      | {
                          minWidthCh?: number;
                          isGroup?: boolean;
                          isValue?: boolean;
                          emphasize?: boolean;
                          heat?: "blue" | "green" | "red";
                          strike?: number;
                        }
                      | undefined;
                    const rawValue = Number(cell.getValue() ?? 0);
                    const heatStyle =
                      columnMeta?.heat === "blue"
                        ? valueHeatBackground(rawValue, maxTotal, BLUE_RGB)
                        : columnMeta?.heat === "green"
                          ? valueHeatBackground(rawValue, maxCall, GREEN_RGB)
                          : columnMeta?.heat === "red"
                            ? valueHeatBackground(rawValue, maxPut, RED_RGB)
                            : undefined;
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
                        } ${cell.column.id === "flag" ? "sticky left-0 z-10 bg-white" : ""} ${
                          isGroupEnd ? "border-b-2 border-slate-200" : ""
                        }`}
                        style={
                          columnMeta?.minWidthCh
                            ? { minWidth: `${columnMeta.minWidthCh}ch`, ...heatStyle }
                            : columnMeta?.isValue
                              ? {
                                  minWidth: "var(--value-col-width)",
                                  maxWidth: "calc(var(--value-col-width) * 1.6)",
                                  ...heatStyle,
                                }
                              : heatStyle
                        }
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    );
                  })}
                </TableRow>
                );
              })}
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
