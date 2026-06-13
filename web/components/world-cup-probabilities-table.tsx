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
import type { GroupRankProbabilities, OpponentProbabilities } from "@/lib/world-cup";
type TableRowData = {
  team: string;
  flagPath: string;
  group?: string | null;
  ratingOverall?: number;
  ratingAttack?: number;
  ratingDefense?: number;
  opponentProbabilities: OpponentProbabilities;
  groupRankProbabilities: GroupRankProbabilities;
  values: Record<string, number>;
  statuses: Record<string, "G" | "U" | "I">;
};

type WorldCupProbabilitiesTableProps = {
  columns: string[];
  rows: TableRowData[];
  probabilityMode: "percent" | "decimal";
};

const percentFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const ratingFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const STAGES = ["R32", "R16", "QF", "SF", "Final"] as const;
const MAX_OPPONENTS = 5;
const OPPONENT_THRESHOLD = 0.001;
const OPPONENT_CELL_MIN = "min-w-[2.5rem]";

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

function wrapHeaderLabel(label: string) {
  const words = label.split(" ").filter(Boolean);
  const content = words.join("\n");
  return (
    <span className="block w-full whitespace-pre-line 2xl:whitespace-nowrap">
      {content}
    </span>
  );
}

function formatDecimalOdds(value: number) {
  if (!Number.isFinite(value)) {
    return "";
  }
  if (value <= 0) {
    return ">1000";
  }
  const odds = 1 / value;
  if (odds > 1000) {
    return ">1000";
  }
  let fractionDigits = 0;
  if (odds < 1.0095) {
    fractionDigits = 3;
  } else if (odds < 10) {
    fractionDigits = 2;
  } else if (odds < 100) {
    fractionDigits = 1;
  }
  return odds.toFixed(fractionDigits);
}

function formatProbability(
  value: number,
  status: "G" | "U" | "I",
  mode: "percent" | "decimal"
) {
  if (status === "G") {
    return "✓";
  }
  if (status === "I") {
    return "✕";
  }
  if (!Number.isFinite(value)) {
    return "";
  }
  if (mode === "decimal") {
    return formatDecimalOdds(value);
  }
  if (value < 0.001) {
    return "<0.1%";
  }
  if (value >= 0.999) {
    return ">99.9";
  }
  return `${percentFormatter.format(value * 100)}%`;
}

function formatOpponentProbability(value: number, mode: "percent" | "decimal") {
  return formatProbability(value, "U", mode);
}

function probabilityBackground(value: number, maxValue: number) {
  if (!Number.isFinite(value)) {
    return undefined;
  }
  if (!Number.isFinite(maxValue) || maxValue <= 0) {
    return undefined;
  }
  const clamped = Math.max(0, Math.min(value / maxValue, 1));
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
  if (clamped <= 40) {
    alpha = (clamped / 40) * 0.1;
  } else if (clamped <= 90) {
    alpha = 0.1 + ((clamped - 40) / 50) * 0.3;
  } else {
    const scaled = (clamped - 90) / 10;
    alpha = 0.4 + scaled * 0.3;
  }
  return { backgroundColor: `rgba(${ACCENT_DARK_RGB}, ${alpha})` };
}

function flagPathForTeam(team: string) {
  return `/flags/${team.replace(/ /g, "_")}.png`;
}

type OpponentEntry = { team: string; probability: number };

function buildOpponentColumn(opponents: Record<string, number>) {
  const entries: OpponentEntry[] = Object.entries(opponents)
    .filter(([, value]) => Number.isFinite(value) && value > 0)
    .map(([team, probability]) => ({ team, probability }));
  entries.sort((a, b) => b.probability - a.probability);
  const top = entries
    .filter((entry) => entry.probability >= OPPONENT_THRESHOLD)
    .slice(0, MAX_OPPONENTS);
  const total = entries.reduce((sum, entry) => sum + entry.probability, 0);
  const topTotal = top.reduce((sum, entry) => sum + entry.probability, 0);
  return { top, other: Math.max(0, total - topTotal), total };
}

const GROUP_POSITIONS = ["1", "2", "3", "4"];

function formatGroupPositionLabel(position: string) {
  if (position === "1") return "1st";
  if (position === "2") return "2nd";
  if (position === "3") return "3rd";
  return `${position}th`;
}

function OpponentCell({
  entry,
  probabilityMode,
}: {
  entry?: OpponentEntry;
  probabilityMode: "percent" | "decimal";
}) {
  if (!entry) {
    return <span className="text-xs text-slate-400">—</span>;
  }
  const formatted = formatOpponentProbability(entry.probability, probabilityMode);
  return (
    <div className="flex items-center justify-center gap-2">
      <span className={`flex ${OPPONENT_CELL_MIN} shrink-0 justify-center`}>
        <span className="relative h-3.5 w-5 overflow-hidden rounded-sm border-0 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]">
          <Image
            src={flagPathForTeam(entry.team)}
            alt={`${entry.team} flag`}
            fill
            className="object-cover"
            sizes="20px"
          />
        </span>
      </span>
      <span className={`${OPPONENT_CELL_MIN} text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap text-right`}>
        {formatted}
      </span>
    </div>
  );
}

export function WorldCupProbabilitiesTable({
  columns,
  rows,
  probabilityMode,
}: WorldCupProbabilitiesTableProps) {
  const primarySortId = "Champion";
  const [expandedTeam, setExpandedTeam] = React.useState<string | null>(null);

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
  const probabilityColumnMax = React.useMemo(
    () =>
      Object.fromEntries(
        columns.map((column) => {
          const columnMax = rows.reduce(
            (max, row) => Math.max(max, Number(row.values[column] ?? 0)),
            0
          );
          return [column, Math.min(columnMax * 1.1, 1)];
        })
      ) as Record<string, number>,
    [columns, rows]
  );

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
        header: () => wrapHeaderLabel("Team"),
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
            for (const key of tiebreakOrder) {
              const probA = Number(a.original.values[key] ?? 0);
              const probB = Number(b.original.values[key] ?? 0);
              if (probA !== probB) {
                return isDesc ? probA - probB : probB - probA;
              }
            }
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
                <span className="md:hidden">Champ.</span>
                <span className="hidden md:inline">{wrapHeaderLabel("Champion")}</span>
              </span>
            );
          }
          if (column === "Win round of 16") {
            return wrapHeaderLabel("Win round of 16");
          }
          if (column === "Win round of 32") {
            return wrapHeaderLabel("Win round of 32");
          }
          if (column === "Qualify") {
            return wrapHeaderLabel("Qualify");
          }
          return wrapHeaderLabel(column);
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
          const formatted = formatProbability(value, status, probabilityMode);
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
            <span className="md:hidden">OVR.</span>
            <span className="hidden md:inline">{wrapHeaderLabel("Overall")}</span>
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
            <span className="md:hidden">Att.</span>
            <span className="hidden md:inline">{wrapHeaderLabel("Attack")}</span>
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
            <span className="md:hidden">Def.</span>
            <span className="hidden md:inline">{wrapHeaderLabel("Defense")}</span>
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
    [columns, probabilityMode, sorting, tiebreakOrder]
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
  const columnCount = table.getAllLeafColumns().length;

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
                    } px-1 xl:px-2 py-1.5 xl:py-2.5 text-[10px] xl:text-[11px] font-semibold uppercase tracking-wide text-slate-600 whitespace-normal 2xl:whitespace-nowrap ${
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
                    <span className="block w-full">
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
              const isExpanded = expandedTeam === row.original.team;
              return (
                <React.Fragment key={row.id}>
                  <TableRow
                    className={`border-b border-slate-100 transition-colors hover:bg-slate-50/70 ${
                      isExpanded ? "bg-slate-50/70" : ""
                    } cursor-pointer`}
                    onClick={() =>
                      setExpandedTeam((current) =>
                        current === row.original.team ? null : row.original.team
                      )
                    }
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
                              ? probabilityBackground(
                                  cell.getValue<number>(),
                                  probabilityColumnMax[cell.column.id] ?? 1
                                )
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
                  {isExpanded ? (
                    <TableRow>
                      <TableCell colSpan={columnCount} className="p-4">
                        <div className="rounded-lg bg-white p-4">
                          <div className="flex items-center justify-center pb-3">
                            <div className="inline-flex items-center gap-4 border-b border-slate-200 pb-2 px-3">
                            <span className="relative h-6 w-9 shrink-0 overflow-hidden rounded-sm shadow-[0_0_0_1px_rgba(15,23,42,0.12)]">
                              {row.original.flagPath ? (
                                <Image
                                  src={row.original.flagPath}
                                  alt={`${row.original.team} flag`}
                                  fill
                                  className="object-cover"
                                  sizes="36px"
                                />
                              ) : (
                                <span className="flex h-full w-full items-center justify-center text-[10px] font-semibold uppercase text-slate-500">
                                  {teamInitials(row.original.team)}
                                </span>
                              )}
                            </span>
                            <div>
                              <p className="text-base font-semibold text-slate-800">
                                {row.original.team}
                              </p>
                            </div>
                            </div>
                          </div>
                          <p className="text-center text-sm font-semibold text-slate-600 pb-3">
                            CHAMPION:{" "}
                            {formatProbability(
                              row.original.values["Champion"] ?? Number.NaN,
                              "U",
                              probabilityMode
                            )}
                          </p>
                          <div className="space-y-6">
                            <div>
                              <div className="pb-3">
                                <p className="text-xs uppercase tracking-[0.28em] text-slate-400">
                                  Group Rank
                                </p>
                              </div>
                              <div className="overflow-x-auto">
                                <table className="w-full table-fixed border-collapse text-center">
                                  <thead>
                                    <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-500">
                                      {GROUP_POSITIONS.map((position) => (
                                        <th key={position} className="py-2 font-semibold">
                                          {formatGroupPositionLabel(position)}
                                        </th>
                                      ))}
                                    </tr>
                                  </thead>
                                  <tbody>
                                    <tr>
                                      {GROUP_POSITIONS.map((position) => (
                                        <td key={position} className="py-2">
                                          <span className="text-xs xl:text-sm font-mono tabular-nums text-slate-700 whitespace-nowrap">
                                            {formatOpponentProbability(
                                              row.original.groupRankProbabilities[position] ?? 0,
                                              probabilityMode
                                            )}
                                          </span>
                                        </td>
                                      ))}
                                    </tr>
                                  </tbody>
                                </table>
                              </div>
                            </div>
                            <div>
                              <div className="pb-3">
                                <p className="text-xs uppercase tracking-[0.28em] text-slate-400">
                                  Knockout Opponents
                                </p>
                              </div>
                              <div className="overflow-x-auto">
                                <table className="w-full table-fixed border-collapse text-center">
                                  <thead>
                                    <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-500">
                                      {STAGES.map((stage) => (
                                        <th key={stage} className="py-2 font-semibold">
                                          {stage}
                                        </th>
                                      ))}
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {(() => {
                                      const columnsByStage = Object.fromEntries(
                                        STAGES.map((stage) => [
                                          stage,
                                          buildOpponentColumn(
                                            row.original.opponentProbabilities[stage]
                                          ),
                                        ])
                                      ) as Record<
                                        (typeof STAGES)[number],
                                        ReturnType<typeof buildOpponentColumn>
                                      >;
                                      const maxTopRows = Math.max(
                                        0,
                                        ...STAGES.map(
                                          (stage) => columnsByStage[stage].top.length
                                        )
                                      );
                                      const maxRows = maxTopRows + 1;
                                      const rowsByRank = Array.from(
                                        { length: maxRows },
                                        (_, rank) => ({
                                          key: `rank-${rank + 1}`,
                                          entries: Object.fromEntries(
                                            STAGES.map((stage) => [
                                              stage,
                                              columnsByStage[stage].top[rank],
                                            ])
                                          ) as Record<
                                            (typeof STAGES)[number],
                                            OpponentEntry | undefined
                                          >,
                                        })
                                      );
                                      const totalRow = {
                                        key: "total",
                                        values: Object.fromEntries(
                                          STAGES.map((stage) => [
                                            stage,
                                            columnsByStage[stage].total,
                                          ])
                                        ) as Record<(typeof STAGES)[number], number>,
                                      };

                                      return (
                                        <>
                                          {rowsByRank.map((rowData, rowIndex) => (
                                            <tr
                                              key={rowData.key}
                                              className={`border-b ${
                                                rowIndex === rowsByRank.length - 1
                                                  ? "border-slate-200"
                                                  : "border-slate-100"
                                              }`}
                                            >
                                              {STAGES.map((stage) => (
                                                <td key={stage} className="py-2">
                                                  {rowIndex <
                                                  columnsByStage[stage].top.length ? (
                                                    <OpponentCell
                                                      entry={rowData.entries[stage]}
                                                      probabilityMode={probabilityMode}
                                                    />
                                                  ) : rowIndex ===
                                                    columnsByStage[stage].top.length ? (
                                                    <div className="flex items-center justify-center gap-2">
                                                      <span
                                                        className={`${OPPONENT_CELL_MIN} text-[10px] uppercase tracking-wide text-slate-400 text-center`}
                                                      >
                                                        Other
                                                      </span>
                                                      <span
                                                        className={`${OPPONENT_CELL_MIN} text-xs xl:text-sm font-mono tabular-nums text-slate-600 whitespace-nowrap text-right`}
                                                      >
                                                        {formatOpponentProbability(
                                                          columnsByStage[stage].other,
                                                          probabilityMode
                                                        )}
                                                      </span>
                                                    </div>
                                                  ) : (
                                                    <span className="text-xs text-slate-300">—</span>
                                                  )}
                                                </td>
                                              ))}
                                            </tr>
                                          ))}
                                          <tr>
                                            {STAGES.map((stage) => (
                                              <td key={stage} className="py-2">
                                                <div className="flex items-center justify-center gap-2">
                                                  <span
                                                    className={`${OPPONENT_CELL_MIN} text-[10px] uppercase tracking-wide text-slate-500 text-center`}
                                                  >
                                                    Total
                                                  </span>
                                                  <span
                                                    className={`${OPPONENT_CELL_MIN} text-xs xl:text-sm font-mono tabular-nums text-slate-800 whitespace-nowrap text-right`}
                                                  >
                                                    {formatOpponentProbability(
                                                      totalRow.values[stage],
                                                      probabilityMode
                                                    )}
                                                  </span>
                                                </div>
                                              </td>
                                            ))}
                                          </tr>
                                        </>
                                      );
                                    })()}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          </div>
                        </div>
                      </TableCell>
                    </TableRow>
                  ) : null}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
