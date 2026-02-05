"use client";

import * as React from "react";
import Image from "next/image";
import {
  ColumnDef,
  Row,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  SortingState,
} from "@tanstack/react-table";

import {
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
type TableRowData = {
  team: string;
  flagPath: string;
  group?: string | null;
  ratingOverall?: number;
  ratingAttack?: number;
  ratingDefense?: number;
  values: Record<string, number>;
  statuses: Record<string, "G" | "U" | "I">;
};

type WorldCupProbabilitiesTableProps = {
  columns: string[];
  rows: TableRowData[];
};

const percentFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const percentFormatterTwoDp = new Intl.NumberFormat("en", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const ratingFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const ACCENT_DARK_RGB = "16, 185, 129";

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

function formatProbability(value: number, status: "G" | "U" | "I") {
  if (status === "G") {
    return "✓";
  }
  if (status === "I") {
    return "✕";
  }
  if (!Number.isFinite(value)) {
    return "";
  }
  if (value < 0.0001) {
    return "<0.01%";
  }
  if (value < 0.00995) {
    return `${percentFormatterTwoDp.format(value * 100)}%`;
  }
  if (value >= 0.9995) {
    return ">99.9%";
  }
  return `${percentFormatter.format(value * 100)}%`;
}

function probabilityBackground(value: number) {
  if (!Number.isFinite(value)) {
    return undefined;
  }
  const clamped = Math.max(0, Math.min(value, 1));
  let alpha = 0;
  if (clamped <= 0.9) {
    if (clamped <= 0.1) {
      const scaled = clamped / 0.1;
      alpha = 0.08 + Math.pow(scaled, 1.2) * 0.08;
    } else {
      const scaled = (clamped - 0.1) / 0.8;
      alpha = 0.16 + Math.pow(scaled, 1.35) * 0.54;
    }
  } else {
    const scaled = (clamped - 0.9) / 0.1;
    alpha = 0.8 + Math.pow(scaled, 1.25) * 0.18;
  }
  return { backgroundColor: `rgba(${ACCENT_LIGHT_RGB}, ${alpha})` };
}

const ACCENT_LIGHT_RGB = "147, 197, 253";

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

export function WorldCupProbabilitiesTable({
  columns,
  rows,
}: WorldCupProbabilitiesTableProps) {
  const primarySortId = "Champion";

  const standardDescForColumn = React.useCallback((columnId: string) => {
    if (columnId === "team" || columnId === "group" || columnId === "flag") {
      return false;
    }
    return true;
  }, []);

  const tiebreakOrder = React.useMemo(() => {
    const priority = [
      "Champion",
      "Reach Final",
      "Reach SF",
      "Reach QF",
      "Reach R16",
      "Reach R32",
      "Qualify",
    ];
    return priority.filter((col) => columns.includes(col));
  }, [columns]);

  const primarySorting = React.useMemo(() => {
    const tiebreaks = tiebreakOrder
      .filter((id) => id !== primarySortId)
      .map((id) => ({ id, desc: standardDescForColumn(id) }));
    return [
      { id: primarySortId, desc: standardDescForColumn(primarySortId) },
      ...tiebreaks,
    ];
  }, [primarySortId, standardDescForColumn, tiebreakOrder]);

  const [sorting, setSorting] = React.useState<SortingState>(() => primarySorting);

  const handleSortToggle = React.useCallback(
    (columnId: string) => {
      const primary = sorting[0];
      if (!primary || primary.id !== columnId) {
        const nextDesc = standardDescForColumn(columnId);
        if (columnId === primarySortId) {
          setSorting(primarySorting);
          return;
        }
        setSorting([{ id: columnId, desc: nextDesc }]);
        return;
      }

      const standardDesc = standardDescForColumn(columnId);
      if (primary.desc === standardDesc) {
        const tiebreaks =
          columnId === primarySortId
            ? tiebreakOrder
                .filter((id) => id !== columnId)
                .map((id) => ({ id, desc: standardDescForColumn(id) }))
            : [];
        setSorting([{ id: columnId, desc: !standardDesc }, ...tiebreaks]);
        return;
      }

      setSorting(primarySorting);
    },
    [primarySortId, primarySorting, sorting, standardDescForColumn, tiebreakOrder]
  );

  const tableColumns = React.useMemo<ColumnDef<TableRowData>[]>(
    () => [
      {
        id: "flag",
        header: "",
        accessorFn: (row) => row.flagPath ?? "",
        meta: { minWidthCh: 2.5, isFlag: true, width: "2.5rem" },
        cell: ({ row }) => (
          <div className="flex pl-1 xl:pl-2 w-full">
            <div className="relative h-3.5 w-5 xl:h-4 xl:w-6 shrink-0 overflow-hidden rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
              {row.original.flagPath ? (
                <Image
                  src={row.original.flagPath}
                  alt={`${row.original.team} flag`}
                  fill
                  className="object-cover"
                  sizes="24px"
                />
              ) : (
                <span className="flex h-full w-full items-center justify-center text-[8px] xl:text-[9px] font-semibold uppercase text-slate-500">
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
        sortingFn: (a, b, id) => {
          const valueA = String(a.getValue(id) ?? "");
          const valueB = String(b.getValue(id) ?? "");
          const isDesc = Boolean(sorting.find((entry) => entry.id === id)?.desc);
          const normalize = (value: string) => ({
            base: value.replace("*", ""),
            starred: value.includes("*"),
          });
          const normA = normalize(valueA);
          const normB = normalize(valueB);
          const baseCompare = normA.base.localeCompare(normB.base);
          if (baseCompare !== 0) {
            const desired = isDesc ? -baseCompare : baseCompare;
            return isDesc ? -desired : desired;
          }
          if (normA.starred === normB.starred) {
            return 0;
          }
          const desired = normA.starred ? 1 : -1;
          return isDesc ? -desired : desired;
        },
        cell: ({ row }: { row: Row<TableRowData> }) => {
          const group = row.original.group ?? "";
          if (!group.includes("*")) {
            return <span className="font-mono text-xs xl:text-sm text-slate-700">{group}</span>;
          }
          const [base, ...rest] = group.split("*");
          return (
            <span className="font-mono text-xs xl:text-sm text-slate-700">
              {base}
              <sup className="ml-[1px] text-[9px] xl:text-[10px]">*</sup>
              {rest.join("*")}
            </span>
          );
        },
      },
      ...columns.map((column) => ({
        id: column,
        header: () => {
          if (column === "Champion") {
            return (
              <span className="whitespace-nowrap">
                <span className="xl:hidden">Champ.</span>
                <span className="hidden xl:inline">Champion</span>
              </span>
            );
          }
          if (column === "Win round of 16") {
            return (
              <span className="whitespace-nowrap">
                <span className="xl:hidden">R16</span>
                <span className="hidden xl:inline">Win round of 16</span>
              </span>
            );
          }
          if (column === "Win round of 32") {
            return (
              <span className="whitespace-nowrap">
                <span className="xl:hidden">R32</span>
                <span className="hidden xl:inline">Win round of 32</span>
              </span>
            );
          }
          if (column === "Qualify") {
            return (
              <span className="whitespace-nowrap">
                <span className="xl:hidden">Qual.</span>
                <span className="hidden xl:inline">Qualify</span>
              </span>
            );
          }
          return column;
        },
        accessorFn: (row: TableRowData) => row.values[column],
        meta: {
          isProbability: true,
        },
        sortingFn: (a: Row<TableRowData>, b: Row<TableRowData>, id: string) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        cell: ({ row }: { row: Row<TableRowData> }) => {
          const value = row.original.values[column];
          const status = row.original.statuses[column] ?? "U";
          const formatted = formatProbability(value, status);
          if (formatted === "✓" || formatted === "✕") {
            return (
              <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700">
                {formatted}
              </span>
            );
          }
          return (
            <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">{formatted}</span>
          );
        },
      })),
      {
        id: "overall",
        header: () => (
          <span className="whitespace-nowrap">
            <span className="xl:hidden">OVR.</span>
            <span className="hidden xl:inline">Overall</span>
          </span>
        ),
        accessorFn: (row) => row.ratingOverall ?? Number.NaN,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {Number.isFinite(row.original.ratingOverall)
              ? ratingFormatter.format(row.original.ratingOverall ?? 0)
              : ""}
          </span>
        ),
      },
      {
        id: "attack",
        header: () => (
          <span className="whitespace-nowrap">
            <span className="xl:hidden">Att.</span>
            <span className="hidden xl:inline">Attack</span>
          </span>
        ),
        accessorFn: (row) => row.ratingAttack ?? Number.NaN,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {Number.isFinite(row.original.ratingAttack)
              ? ratingFormatter.format(row.original.ratingAttack ?? 0)
              : ""}
          </span>
        ),
      },
      {
        id: "defense",
        header: () => (
          <span className="whitespace-nowrap">
            <span className="xl:hidden">Def.</span>
            <span className="hidden xl:inline">Defense</span>
          </span>
        ),
        accessorFn: (row) => row.ratingDefense ?? Number.NaN,
        sortingFn: (a, b, id) =>
          Number(a.getValue(id) ?? 0) - Number(b.getValue(id) ?? 0),
        meta: { isRating: true },
        cell: ({ row }) => (
          <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
            {Number.isFinite(row.original.ratingDefense)
              ? ratingFormatter.format(row.original.ratingDefense ?? 0)
              : ""}
          </span>
        ),
      },
    ],
    [columns, sorting]
  );

  const table = useReactTable({
    data: rows,
    columns: tableColumns,
    state: { sorting },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableMultiSort: true,
  });

  const isGroupedByGroup = sorting[0]?.id === "group";
  const groupBase = (value: string | null | undefined) =>
    String(value ?? "").replace(/\*/g, "");
  const lastProbabilityId = columns[columns.length - 1];

  return (
    <div className="min-w-0 w-full overflow-clip rounded-xl bg-white ring-1 ring-slate-200 shadow-sm">
      <div className="table-scroll overflow-x-auto">
        <table className="w-full table-auto xl:table-fixed text-sm [--prob-col-width:clamp(4ch,6vw,8ch)] xl:[--prob-col-width:clamp(6ch,6vw,9ch)]">
          <thead className="sticky top-0 z-[50] border-b border-slate-200 bg-slate-200">
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id} className="bg-slate-200 border-b border-slate-200">
                {headerGroup.headers.map((header, index) => {
                  const isLastHeader = index === headerGroup.headers.length - 1;
                  const columnMeta = header.column.columnDef.meta as
                    | {
                        minWidthCh?: number;
                        isGroup?: boolean;
                        isProbability?: boolean;
                        isRating?: boolean;
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
                        : header.id === "overall" ||
                          header.id === "attack" ||
                          header.id === "defense"
                        ? "text-right whitespace-nowrap"
                        : "text-right"
                    } px-1 xl:px-2 py-1.5 xl:py-2.5 text-[10px] xl:text-[11px] font-semibold uppercase tracking-wide text-slate-600 ${
                      header.id === "flag"
                        ? "sticky left-0 z-10 bg-slate-200 rounded-tl-xl"
                        : ""
                    } ${isLastHeader ? "rounded-tr-xl" : ""}`}
                    onClick={() => handleSortToggle(header.id)}
                    style={
                      columnMeta?.minWidthCh
                        ? {
                            minWidth: `${columnMeta.minWidthCh}ch`,
                          }
                        : columnMeta?.isProbability || columnMeta?.isRating
                        ? {
                            minWidth: "var(--prob-col-width)",
                            maxWidth: "calc(var(--prob-col-width) * 2)",
                          }
                        : undefined
                    }
                  >
                    <span className="inline-flex items-center gap-1">
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                    </span>
                  </TableHead>
                );
                })}
              </TableRow>
            ))}
          </thead>
          <tbody className="divide-y divide-slate-100">
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
                  className="border-b border-slate-100 transition-colors hover:bg-slate-50/70"
                >
                {row.getVisibleCells().map((cell) => {
                  const columnMeta = cell.column.columnDef.meta as
                    | {
                        minWidthCh?: number;
                        isGroup?: boolean;
                        isProbability?: boolean;
                        isRating?: boolean;
                      }
                    | undefined;
                  return (
                  <TableCell
                    key={cell.id}
                    className={`px-1 xl:px-2 py-1.5 xl:py-2.5 ${
                      cell.column.id === "flag"
                        ? "text-left w-[3rem] min-w-[3rem] pl-0.5 xl:pl-1 pr-1.5 xl:pr-2.5 py-1.5 xl:py-2.5 overflow-hidden"
                        : cell.column.id === "team"
                        ? "text-left w-[10rem] min-w-[7rem] xl:min-w-[10rem] shrink-0 pl-0.5 xl:pl-1"
                        : columnMeta?.isGroup
                        ? "text-center min-w-[3ch] xl:min-w-[4ch]"
                        : "text-right"
                    } ${
                      columnMeta?.isProbability || columnMeta?.isRating ? "pl-4" : ""
                    } ${
                      cell.column.id === "flag"
                        ? "sticky left-0 z-10 bg-white"
                        : ""
                    } ${
                      isGroupEnd ? "border-b-2 border-slate-200" : ""
                    }`}
                    style={{
                      ...(columnMeta?.isProbability
                        ? probabilityBackground(cell.getValue<number>())
                        : {}),
                      ...(columnMeta?.isRating
                        ? ratingBackground(cell.getValue<number>())
                        : {}),
                      ...(columnMeta?.minWidthCh
                        ? {
                            minWidth: `${columnMeta.minWidthCh}ch`,
                          }
                        : columnMeta?.isProbability || columnMeta?.isRating
                        ? {
                            minWidth: "var(--prob-col-width)",
                            maxWidth: "calc(var(--prob-col-width) * 2)",
                          }
                        : {}),
                    }}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                );
                })}
                </TableRow>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
