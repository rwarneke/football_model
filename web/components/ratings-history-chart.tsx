"use client";

import * as React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  ReferenceArea,
  Customized,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TooltipProps } from "recharts";

import type { RatingsHistoryPoint } from "@/lib/ratings";
import teamGroups from "@/lib/team-groups.json";

type RatingsHistoryChartProps = {
  data: RatingsHistoryPoint[];
  teams: string[];
};

type TeamGroupConfig = {
  categories: Array<{
    name: string;
    groups: Array<{
      id: string;
      label: string;
      teams: string[];
    }>;
  }>;
};

type PresetOption = {
  id: string;
  label: string;
  type: "rank" | "group";
  order?: "asc" | "desc";
  count?: number;
  teams?: string[];
  category: string;
};

function formatTick(value: number) {
  return String(new Date(value).getFullYear());
}

function formatTooltipLabel(value: number) {
  return new Date(value).getFullYear();
}

const tooltipNumberFormatter = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

type TooltipPayloadItem = {
  dataKey?: string | number;
  name?: string | number;
  value?: unknown;
  color?: string;
};

type RechartsTooltipProps = TooltipProps<
  number | string | Array<number | string>,
  string | number
>;
type RechartsPayload = NonNullable<RechartsTooltipProps["payload"]>[number];

function TooltipContent({
  active,
  payload,
  label,
  coordinate,
  viewBox,
  activeTeam,
}: RechartsTooltipProps & { activeTeam?: string | null }) {
  const contentRef = React.useRef<HTMLDivElement | null>(null);
  const [contentHeight, setContentHeight] = React.useState<number | null>(null);

  React.useLayoutEffect(() => {
    if (!contentRef.current) {
      return;
    }
    setContentHeight(contentRef.current.offsetHeight);
  }, [payload]);

  if (!active || !payload || payload.length === 0 || typeof label !== "number") {
    return null;
  }
  const rows = (payload ?? [])
    .map((item) => item as TooltipPayloadItem)
    .filter((item) => typeof item.value === "number")
    .sort((a, b) => (b.value as number) - (a.value as number));
  if (rows.length === 0) {
    return null;
  }
  let translateY = "12px";
  const coordY = typeof coordinate?.y === "number" ? coordinate.y : null;
  const viewBoxY = typeof viewBox?.y === "number" ? viewBox.y : null;
  const viewBoxHeight =
    typeof viewBox?.height === "number" ? viewBox.height : null;
  if (coordY !== null && viewBoxY !== null && viewBoxHeight !== null && contentHeight) {
    const bottom = viewBoxY + viewBoxHeight;
    const desiredTop = coordY + 12;
    const canFit = contentHeight <= viewBoxHeight;
    const top = canFit
      ? Math.min(desiredTop, bottom - contentHeight)
      : viewBoxY;
    translateY = `${top - coordY}px`;
  }
  return (
    <div
      ref={contentRef}
      className="rounded-[3px] border border-ink-700/40 bg-white px-3 py-2 text-sm text-ebony shadow-soft"
      style={{ transform: `translate(calc(-100% - 20px), ${translateY})` }}
    >
      <div className="text-xs text-ink-400">{formatTooltipLabel(label)}</div>
      <div className="mt-1 space-y-1">
        {rows.map((item) => {
          const name = item.name ?? item.dataKey ?? "";
          const isActive = activeTeam && name === activeTeam;
          return (
            <div key={name} className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <span
                  className="h-2 w-2 rounded-lg"
                  style={{ backgroundColor: item.color ?? "var(--color-accent-dark)" }}
                />
                <span className={isActive ? "font-semibold" : ""}>{name}</span>
              </div>
              <span className="font-mono font-medium">
                {tooltipNumberFormatter.format(item.value as number)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function colorForIndex(index: number) {
  const hue = (index * 47) % 360;
  return `hsl(${hue} 65% 45%)`;
}

export function RatingsHistoryChart({ data, teams }: RatingsHistoryChartProps) {
  const latestSnapshot = React.useMemo(() => {
    if (data.length === 0) {
      return null;
    }
    return data[data.length - 1];
  }, [data]);

  const defaultSelected = React.useMemo(() => {
    if (!latestSnapshot) {
      return [];
    }
    return [...teams]
      .sort((a, b) => {
        const valueA =
          typeof latestSnapshot[a] === "number"
            ? (latestSnapshot[a] as number)
            : -Infinity;
        const valueB =
          typeof latestSnapshot[b] === "number"
            ? (latestSnapshot[b] as number)
            : -Infinity;
        return valueB - valueA;
      })
      .slice(0, 10);
  }, [latestSnapshot, teams]);

  const [selectedTeams, setSelectedTeams] = React.useState<string[]>(
    () => defaultSelected
  );
  const [query, setQuery] = React.useState("");
  const [activePreset, setActivePreset] = React.useState<string>("best-10");
  const [xDomain, setXDomain] = React.useState<[number, number] | null>(null);
  const [yDomain, setYDomain] = React.useState<[number, number] | null>(null);
  const [refAreaLeft, setRefAreaLeft] = React.useState<number | null>(null);
  const [refAreaRight, setRefAreaRight] = React.useState<number | null>(null);
  const [refAreaTop, setRefAreaTop] = React.useState<number | null>(null);
  const [refAreaBottom, setRefAreaBottom] = React.useState<number | null>(null);
  const [isSelecting, setIsSelecting] = React.useState(false);
  const [lockedAxis, setLockedAxis] = React.useState<"x" | "y" | null>(null);
  const [activeTeam, setActiveTeam] = React.useState<string | null>(null);
  const [pinnedTooltip, setPinnedTooltip] = React.useState<{
    payload: RechartsPayload[];
    label: number;
    coordinate?: { x?: number; y?: number };
  } | null>(null);
  const [pinnedTeam, setPinnedTeam] = React.useState<string | null>(null);
  const [dragZoomEnabled, setDragZoomEnabled] = React.useState(true);
  const chartWrapperRef = React.useRef<HTMLDivElement | null>(null);
  const chartCardRef = React.useRef<HTMLDivElement | null>(null);
  const [chartCardHeight, setChartCardHeight] = React.useState<number | null>(null);
  const [isXl, setIsXl] = React.useState(false);
  const pinchRef = React.useRef<{
    startDistance: number;
    startCenter: { x: number; y: number };
    startXDomain: [number, number];
    startYDomain: [number, number];
  } | null>(null);
  const dragStartRef = React.useRef<{ x: number; y: number } | null>(null);
  const dragEndRef = React.useRef<{ x: number; y: number } | null>(null);
  const chartMetaRef = React.useRef<{
    xScale: { invert?: (value: number) => number | Date; domain: () => unknown[] };
    yScale: { invert?: (value: number) => number | Date; domain: () => unknown[] };
  } | null>(null);

  React.useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const mediaQuery = window.matchMedia("(min-width: 1280px)");
    const handleChange = () => setIsXl(mediaQuery.matches);
    handleChange();
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  React.useLayoutEffect(() => {
    if (!isXl || !chartCardRef.current) {
      setChartCardHeight(null);
      return;
    }
    const element = chartCardRef.current;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const target = entry.target as HTMLElement;
        const nextHeight = Math.round(target.getBoundingClientRect().height);
        setChartCardHeight((prev) => (prev === nextHeight ? prev : nextHeight));
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [isXl]);

  const visibleTeams = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const filtered = normalized
      ? teams.filter((team) => team.toLowerCase().includes(normalized))
      : teams;
    return [...filtered].sort((a, b) => a.localeCompare(b));
  }, [query, teams]);

  const selectedSet = React.useMemo(
    () => new Set(selectedTeams),
    [selectedTeams]
  );

  const toggleTeam = React.useCallback(
    (team: string) => {
      setSelectedTeams((prev) => {
        const next = new Set(prev);
        if (next.has(team)) {
          next.delete(team);
        } else {
          next.add(team);
        }
        return teams.filter((name) => next.has(name));
      });
    },
    [teams]
  );

  const resetZoom = React.useCallback(() => {
    setXDomain(null);
    setYDomain(null);
    setRefAreaLeft(null);
    setRefAreaRight(null);
    setRefAreaTop(null);
    setRefAreaBottom(null);
    setPinnedTooltip(null);
    setPinnedTeam(null);
  }, []);

  const chartMargin = { top: 12, right: 16, bottom: 16, left: 12 };
  const roundToTenth = (value: number) => Math.round(value * 10) / 10;
  const yTickCount = 6;

  const minRatingForTeams = React.useCallback(
    (teamList: string[]) => {
      let min = Number.POSITIVE_INFINITY;
      for (const row of data) {
        for (const team of teamList) {
          const value = row[team];
          if (typeof value === "number") {
            min = Math.min(min, value);
          }
        }
      }
      return Number.isFinite(min) ? min : 0;
    },
    [data]
  );

  const presets = React.useMemo(() => {
    const config = teamGroups as TeamGroupConfig;
    const presetOptions: PresetOption[] = [
      {
        id: "best-10",
        label: "Best 10",
        type: "rank",
        order: "desc",
        count: 10,
        category: "Rankings",
      },
      {
        id: "worst-10",
        label: "Worst 10",
        type: "rank",
        order: "asc",
        count: 10,
        category: "Rankings",
      },
    ];

    for (const category of config.categories) {
      for (const group of category.groups) {
        presetOptions.push({
          id: group.id,
          label: group.label,
          type: "group",
          teams: group.teams,
          category: category.name,
        });
      }
    }

    return presetOptions;
  }, []);

  const applyPreset = React.useCallback(
    (presetId: string) => {
      const preset = presets.find((option) => option.id === presetId);
      if (!preset) {
        return;
      }
      if (preset.type === "rank") {
        if (!latestSnapshot) {
          return;
        }
        const sorted = [...teams].sort((a, b) => {
          const valueA =
            typeof latestSnapshot[a] === "number"
              ? (latestSnapshot[a] as number)
              : preset.order === "asc"
              ? Number.POSITIVE_INFINITY
              : Number.NEGATIVE_INFINITY;
          const valueB =
            typeof latestSnapshot[b] === "number"
              ? (latestSnapshot[b] as number)
              : preset.order === "asc"
              ? Number.POSITIVE_INFINITY
              : Number.NEGATIVE_INFINITY;
          return preset.order === "asc" ? valueA - valueB : valueB - valueA;
        });
        const selection = sorted.slice(0, preset.count ?? 10);
        setSelectedTeams(selection);
        setYDomain(null);
        setActivePreset(preset.id);
        return;
      }

      const groupTeams = (preset.teams ?? []).filter((team) =>
        teams.includes(team)
      );
      setSelectedTeams(groupTeams);
      setYDomain(null);
      setActivePreset(preset.id);
    },
    [latestSnapshot, presets, teams, minRatingForTeams]
  );

  const dataExtent = React.useMemo<[number, number]>(() => {
    if (data.length === 0) {
      return [0, 1];
    }
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const row of data) {
      if (typeof row.date !== "number") {
        continue;
      }
      min = Math.min(min, row.date);
      max = Math.max(max, row.date);
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return [0, 1];
    }
    return [min, max];
  }, [data]);

  const updateChartMeta = React.useCallback((props: any) => {
    const xAxis = Object.values(props.xAxisMap ?? {})[0] as {
      scale: { invert?: (value: number) => number | Date; domain: () => unknown[] };
    };
    const yAxis = Object.values(props.yAxisMap ?? {})[0] as {
      scale: { invert?: (value: number) => number | Date; domain: () => unknown[] };
    };
    if (xAxis?.scale && yAxis?.scale) {
      chartMetaRef.current = {
        xScale: xAxis.scale,
        yScale: yAxis.scale,
      };
    }
  }, []);

  const ChartMeta = React.useCallback(
    (props: any) => {
      updateChartMeta(props);
      return null;
    },
    [updateChartMeta]
  );

  React.useEffect(() => {
    setYDomain(null);
  }, []);

  React.useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const media = window.matchMedia("(hover: hover) and (pointer: fine)");
    const update = () => setDragZoomEnabled(media.matches);
    update();
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", update);
      return () => media.removeEventListener("change", update);
    }
    media.addListener(update);
    return () => media.removeListener(update);
  }, []);

  // Prevent pinch zoom on devices that don't support click-drag zoom
  React.useEffect(() => {
    if (dragZoomEnabled) {
      return;
    }
    const wrapper = chartWrapperRef.current;
    if (!wrapper) {
      return;
    }

    const preventZoom = (e: TouchEvent) => {
      if (e.touches.length > 1) {
        e.preventDefault();
      }
    };

    const preventWheelZoom = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
      }
    };

    wrapper.addEventListener("touchstart", preventZoom, { passive: false });
    wrapper.addEventListener("touchmove", preventZoom, { passive: false });
    wrapper.addEventListener("wheel", preventWheelZoom, { passive: false });

    return () => {
      wrapper.removeEventListener("touchstart", preventZoom);
      wrapper.removeEventListener("touchmove", preventZoom);
      wrapper.removeEventListener("wheel", preventWheelZoom);
    };
  }, [dragZoomEnabled]);

  const yTicks = React.useMemo(() => {
    const [minRaw, maxRaw] = yDomain ?? [0, 100];
    const min = Math.floor(minRaw);
    const max = Math.ceil(maxRaw);
    if (max <= min) {
      return [min];
    }
    const step = Math.max(1, Math.round((max - min) / (yTickCount - 1)));
    const ticks: number[] = [];
    for (let value = min; value <= max; value += step) {
      if (value >= minRaw && value <= maxRaw) {
        ticks.push(value);
      }
    }
    if (ticks.length === 0) {
      ticks.push(Math.round(minRaw), Math.round(maxRaw));
      return Array.from(new Set(ticks)).sort((a, b) => a - b);
    }
    const last = ticks[ticks.length - 1];
    if (maxRaw >= minRaw && last !== Math.round(maxRaw)) {
      const finalTick = Math.round(maxRaw);
      if (finalTick >= minRaw && finalTick <= maxRaw) {
        ticks.push(finalTick);
      }
    }
    return ticks;
  }, [yDomain, yTickCount]);

  const currentXDomain = React.useMemo<[number, number]>(() => {
    const [minRaw, maxRaw] = xDomain ?? dataExtent;
    return [Math.min(minRaw, maxRaw), Math.max(minRaw, maxRaw)];
  }, [xDomain, dataExtent]);

  const xTicks = React.useMemo(() => {
    const [minRaw, maxRaw] = xDomain ?? dataExtent;
    const minDate = new Date(minRaw);
    const maxDate = new Date(maxRaw);
    if (Number.isNaN(minDate.getTime()) || Number.isNaN(maxDate.getTime())) {
      return [];
    }
    const minYear = minDate.getUTCFullYear();
    const maxYear = maxDate.getUTCFullYear();
    if (maxYear <= minYear) {
      return [Math.max(minRaw, Math.min(maxRaw, Date.UTC(minYear, 0, 1)))];
    }
    const span = maxYear - minYear;
    const roughStep = Math.max(1, Math.round(span / 9));
    const stepCandidates = [1, 2, 5, 10, 20, 25];
    const step =
      stepCandidates.find((candidate) => candidate >= roughStep) ?? 50;
    const first = Math.ceil(minYear / step) * step;
    const ticks: number[] = [];
    for (let year = first; year <= maxYear; year += step) {
      const tick = Date.UTC(year, 0, 1);
      if (tick >= minRaw && tick <= maxRaw) {
        ticks.push(tick);
      }
    }
    return ticks;
  }, [xDomain, dataExtent]);

  const toDomainValue = React.useCallback(
    (value: number | undefined, axis: "x" | "y") => {
      if (typeof value !== "number") {
        return null;
      }
      const meta = chartMetaRef.current;
      if (!meta) {
        return null;
      }
      const scale = axis === "x" ? meta.xScale : meta.yScale;
      if (typeof scale.invert !== "function") {
        return null;
      }
      const range = (() => {
        const candidate = (scale as unknown as { range?: () => number[] }).range;
        return typeof candidate === "function" ? candidate() : null;
      })();
      let clamped = value;
      if (range && range.length >= 2) {
        const min = Math.min(range[0] as number, range[range.length - 1] as number);
        const max = Math.max(range[0] as number, range[range.length - 1] as number);
        clamped = Math.min(Math.max(value, min), max);
      }
      const inverted = scale.invert(clamped);
      return inverted instanceof Date ? inverted.getTime() : (inverted as number);
    },
    []
  );

  const clampDomain = React.useCallback(
    (domain: [number, number], min: number, max: number) => {
      const span = domain[1] - domain[0];
      if (!Number.isFinite(span) || span <= 0) {
        return [min, max] as [number, number];
      }
      let nextMin = domain[0];
      let nextMax = domain[1];
      if (nextMin < min) {
        nextMin = min;
        nextMax = min + span;
      }
      if (nextMax > max) {
        nextMax = max;
        nextMin = max - span;
      }
      return [nextMin, nextMax] as [number, number];
    },
    []
  );

  const startPinch = React.useCallback(
    (touches: React.TouchList) => {
      if (touches.length < 2) {
        return;
      }
      const rect = chartWrapperRef.current?.getBoundingClientRect();
      if (!rect) {
        return;
      }
      const [first, second] = [touches[0], touches[1]];
      if (!first || !second) {
        return;
      }
      const dx = second.clientX - first.clientX;
      const dy = second.clientY - first.clientY;
      const startDistance = Math.hypot(dx, dy);
      if (!Number.isFinite(startDistance) || startDistance === 0) {
        return;
      }
      const startCenter = {
        x: (first.clientX + second.clientX) / 2 - rect.left,
        y: (first.clientY + second.clientY) / 2 - rect.top,
      };
      const meta = chartMetaRef.current;
      const rawXDomain =
        xDomain ??
        (meta?.xScale?.domain?.().map((value) =>
          value instanceof Date ? value.getTime() : (value as number)
        ) as number[]) ??
        dataExtent;
      const startXDomain: [number, number] = [
        Math.min(rawXDomain[0], rawXDomain[rawXDomain.length - 1]),
        Math.max(rawXDomain[0], rawXDomain[rawXDomain.length - 1]),
      ];
      const startYDomain: [number, number] = yDomain ?? [0, 100];
      pinchRef.current = {
        startDistance,
        startCenter,
        startXDomain,
        startYDomain,
      };
    },
    [dataExtent, xDomain, yDomain]
  );

  const handlePinchMove = React.useCallback(
    (touches: React.TouchList) => {
      if (touches.length < 2 || !pinchRef.current) {
        return;
      }
      const rect = chartWrapperRef.current?.getBoundingClientRect();
      if (!rect) {
        return;
      }
      const [first, second] = [touches[0], touches[1]];
      if (!first || !second) {
        return;
      }
      const dx = second.clientX - first.clientX;
      const dy = second.clientY - first.clientY;
      const distance = Math.hypot(dx, dy);
      if (!Number.isFinite(distance) || distance === 0) {
        return;
      }
      const scale = distance / pinchRef.current.startDistance;
      if (!Number.isFinite(scale) || scale <= 0) {
        return;
      }
      const center = {
        x: (first.clientX + second.clientX) / 2 - rect.left,
        y: (first.clientY + second.clientY) / 2 - rect.top,
      };
      const centerX = toDomainValue(center.x, "x");
      const centerY = toDomainValue(center.y, "y");
      if (typeof centerX === "number") {
        const span = pinchRef.current.startXDomain[1] - pinchRef.current.startXDomain[0];
        const nextSpan = Math.max(span / scale, 1);
        const nextXDomain = clampDomain(
          [centerX - nextSpan / 2, centerX + nextSpan / 2],
          dataExtent[0],
          dataExtent[1]
        );
        setXDomain(nextXDomain);
      }
      if (typeof centerY === "number") {
        const span = pinchRef.current.startYDomain[1] - pinchRef.current.startYDomain[0];
        const nextSpan = Math.max(span / scale, 1);
        const nextYDomain = clampDomain(
          [centerY - nextSpan / 2, centerY + nextSpan / 2],
          0,
          100
        );
        setYDomain(nextYDomain);
      }
    },
    [clampDomain, dataExtent, toDomainValue]
  );

  const endPinch = React.useCallback(() => {
    pinchRef.current = null;
  }, []);

  const closestTeamForPayload = React.useCallback(
    (payload: RechartsPayload[] | undefined, chartY: number | undefined) => {
      const meta = chartMetaRef.current;
      const yScale = meta?.yScale as ((value: number) => number) | undefined;
      if (!payload || payload.length === 0 || typeof chartY !== "number" || typeof yScale !== "function") {
        return null;
      }
      let closestTeam: string | null = null;
      let closestDistance = Infinity;
      for (const rawItem of payload) {
        const item = rawItem as TooltipPayloadItem;
        if (typeof item.value !== "number") {
          continue;
        }
        const yPx = yScale(item.value);
        const dist = Math.abs(yPx - chartY);
        if (dist < closestDistance) {
          closestDistance = dist;
          closestTeam = (item.name ?? item.dataKey ?? null) as string | null;
        }
      }
      const threshold = 8;
      return closestTeam && closestDistance <= threshold ? closestTeam : null;
    },
    []
  );

  function handlePointerDown(event: any) {
    if (!dragZoomEnabled) {
      // Completely disable zoom for devices that don't support click-drag zoom
      return;
    }
    if (!event) {
      return;
    }
    const xValue = toDomainValue(event.chartX, "x");
    const yValue = toDomainValue(event.chartY, "y");
    if (xValue === null || yValue === null) {
      return;
    }
    if (typeof event.chartX === "number" && typeof event.chartY === "number") {
      dragStartRef.current = { x: event.chartX, y: event.chartY };
      dragEndRef.current = { x: event.chartX, y: event.chartY };
    } else {
      dragStartRef.current = null;
      dragEndRef.current = null;
    }
    setIsSelecting(true);
    setActiveTeam(null);
    setPinnedTooltip(null);
    setPinnedTeam(null);
    setLockedAxis(null);
    setRefAreaLeft(xValue);
    setRefAreaRight(null);
    setRefAreaTop(yValue);
    setRefAreaBottom(null);
  }

  function handlePointerMove(event: any) {
    if (!dragZoomEnabled) {
      // Completely disable zoom for devices that don't support click-drag zoom
      return;
    }
    if (!event) {
      return;
    }

    if (isSelecting) {
      if (refAreaLeft === null) {
        return;
      }
      const xValue = toDomainValue(event.chartX, "x");
      const yValue = toDomainValue(event.chartY, "y");
      const dragStart = dragStartRef.current;
      const dragEnd =
        typeof event.chartX === "number" && typeof event.chartY === "number"
          ? { x: event.chartX, y: event.chartY }
          : dragEndRef.current;
      if (dragEnd) {
        dragEndRef.current = dragEnd;
      }

      const minPixel = 12;
      const dx =
        dragStart && dragEnd
          ? Math.abs(dragStart.x - dragEnd.x)
          : null;
      const dy =
        dragStart && dragEnd
          ? Math.abs(dragStart.y - dragEnd.y)
          : null;
      const shouldLockX =
        dx !== null && dx < minPixel && (dy === null || dy >= minPixel);
      const shouldLockY =
        dy !== null && dy < minPixel && (dx === null || dx >= minPixel);
      if (shouldLockX) {
        setLockedAxis("y");
      } else if (shouldLockY) {
        setLockedAxis("x");
      } else {
        setLockedAxis(null);
      }

      if (xValue !== null) {
        setRefAreaRight(xValue);
      }
      if (yValue !== null) {
        setRefAreaBottom(yValue);
      }
      return;
    }

    if (pinnedTooltip) {
      return;
    }

    const payload = event.activePayload as RechartsPayload[] | undefined;
    const closestTeam = closestTeamForPayload(payload, event.chartY);
    if (closestTeam) {
      if (activeTeam !== closestTeam) {
        setActiveTeam(closestTeam);
      }
    } else if (activeTeam) {
      setActiveTeam(null);
    }
  }

  function handlePointerUp() {
    if (!dragZoomEnabled) {
      // Completely disable zoom for devices that don't support click-drag zoom
      return;
    }
    endPinch();
    setIsSelecting(false);
    setLockedAxis(null);
    if (
      refAreaLeft === null ||
      refAreaRight === null ||
      refAreaTop === null ||
      refAreaBottom === null
    ) {
      dragStartRef.current = null;
      setRefAreaLeft(null);
      setRefAreaRight(null);
      setRefAreaTop(null);
      setRefAreaBottom(null);
      return;
    }

    const x1 = Math.min(refAreaLeft, refAreaRight);
    const x2 = Math.max(refAreaLeft, refAreaRight);
    const y1 = Math.min(refAreaTop, refAreaBottom);
    const y2 = Math.max(refAreaTop, refAreaBottom);

    const dragStart = dragStartRef.current;
    const dragEnd = dragEndRef.current;
    const minPixel = 12;
    const dx =
      dragStart && dragEnd
        ? Math.abs(dragStart.x - dragEnd.x)
        : null;
    const dy =
      dragStart && dragEnd
        ? Math.abs(dragStart.y - dragEnd.y)
        : null;

    const meta = chartMetaRef.current;
    const xDomainRaw =
      xDomain ??
      (meta?.xScale?.domain?.().map((value) =>
        value instanceof Date ? value.getTime() : (value as number)
      ) as number[]) ??
      [x1, x2];
    const xDomainCurrent: [number, number] = [
      Math.min(xDomainRaw[0] ?? x1, xDomainRaw[xDomainRaw.length - 1] ?? x2),
      Math.max(xDomainRaw[0] ?? x1, xDomainRaw[xDomainRaw.length - 1] ?? x2),
    ];
    const yDomainCurrent = yDomain ?? [0, 100];

    if (x1 === x2 && y1 === y2) {
      dragStartRef.current = null;
      setRefAreaLeft(null);
      setRefAreaRight(null);
      setRefAreaTop(null);
      setRefAreaBottom(null);
      return;
    }

    const domain =
      (meta?.xScale as unknown as { domain?: () => unknown[] })?.domain?.() ?? [];
    const firstDomainValue = domain[0];
    const lastDomainValue = domain[domain.length - 1];
    const rawMin =
      firstDomainValue instanceof Date
        ? firstDomainValue.getTime()
        : typeof firstDomainValue === "number"
        ? firstDomainValue
        : null;
    const rawMax =
      lastDomainValue instanceof Date
        ? lastDomainValue.getTime()
        : typeof lastDomainValue === "number"
        ? lastDomainValue
        : null;
    const minX = rawMin ?? x1;
    const maxX = rawMax ?? x2;
    const nextX1 = Math.max(minX, x1);
    const nextX2 = Math.min(maxX, x2);
    const nextY1 = Math.max(0, Math.min(100, y1));
    const nextY2 = Math.max(0, Math.min(100, y2));

    const shouldLockX = dx !== null && dx < minPixel && (dy === null || dy >= minPixel);
    const shouldLockY = dy !== null && dy < minPixel && (dx === null || dx >= minPixel);

    if (shouldLockX) {
      setXDomain(xDomainCurrent);
      setYDomain([roundToTenth(nextY1), roundToTenth(nextY2)]);
    } else if (shouldLockY) {
      setXDomain([nextX1, nextX2]);
      setYDomain(yDomainCurrent);
    } else {
      setXDomain([nextX1, nextX2]);
      setYDomain([roundToTenth(nextY1), roundToTenth(nextY2)]);
    }

    dragStartRef.current = null;
    dragEndRef.current = null;
    setRefAreaLeft(null);
    setRefAreaRight(null);
    setRefAreaTop(null);
    setRefAreaBottom(null);
  }

  function handlePointerLeave() {
    if (!dragZoomEnabled) {
      return;
    }
    endPinch();
    setIsSelecting(false);
    setLockedAxis(null);
    setActiveTeam(null);
    dragStartRef.current = null;
    dragEndRef.current = null;
  }

  function handleChartClick(event: any) {
    if (!event || isSelecting) {
      return;
    }
    const payload = event.activePayload as RechartsPayload[] | undefined;
    const label = event.activeLabel;
    if (!payload || payload.length === 0 || typeof label !== "number") {
      setPinnedTooltip(null);
      setPinnedTeam(null);
      return;
    }
    const closestTeam = closestTeamForPayload(payload, event.chartY);
    const coordinate =
      event.activeCoordinate ??
      (typeof event.chartX === "number" && typeof event.chartY === "number"
        ? { x: event.chartX, y: event.chartY }
        : undefined);
    setPinnedTooltip({ payload, label, coordinate });
    setPinnedTeam(closestTeam);
    if (closestTeam) {
      setActiveTeam(closestTeam);
    }
  }

  return (
    <div className="flex flex-col gap-6 xl:flex-row xl:items-stretch">
      <div
        ref={chartCardRef}
        className="relative flex flex-col gap-4 overflow-hidden rounded-xl bg-white p-4 ring-1 ring-slate-200 shadow-sm xl:flex-[3] xl:self-start"
      >
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold text-ebony">Ratings history</h2>
            <p className="hidden text-sm text-ink-400 sm:block">
              Drag across the chart area to zoom in on a time window.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 sm:flex-nowrap">
            <button
              type="button"
              className="flex min-w-[9.5rem] items-center justify-center rounded-lg border border-ink-700 px-3 py-1 text-sm text-ebony transition hover:bg-ink-800/60"
              onClick={() => {
                setSelectedTeams(teams);
                setYDomain(null);
                setActivePreset("");
              }}
            >
              Select all
            </button>
            <button
              type="button"
              className="flex min-w-[9.5rem] items-center justify-center rounded-lg border border-ink-700 px-3 py-1 text-sm text-ebony transition hover:bg-ink-800/60"
              onClick={() => {
                setSelectedTeams([]);
                setActivePreset("");
              }}
            >
              Unselect all
            </button>
            <button
              type="button"
              className="hidden min-w-[9.5rem] items-center justify-center rounded-lg border border-ink-700 px-3 py-1 text-sm text-ebony transition hover:bg-ink-800/60 sm:inline-flex"
              onClick={resetZoom}
            >
              Reset zoom
            </button>
          </div>
        </div>
        <div className="mt-4 aspect-square w-full max-h-[520px] select-none rounded-md">
          <div
            ref={chartWrapperRef}
            className="h-full w-full min-h-0 select-none touch-none"
            onTouchStart={dragZoomEnabled ? handlePointerDown : undefined}
            onTouchMove={dragZoomEnabled ? handlePointerMove : undefined}
            onTouchEnd={dragZoomEnabled ? handlePointerUp : undefined}
            onTouchCancel={dragZoomEnabled ? handlePointerLeave : undefined}
            style={{
              touchAction: dragZoomEnabled ? "auto" : "pan-x pan-y",
              WebkitUserSelect: "none",
              userSelect: "none",
            }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data}
                margin={chartMargin}
                onDoubleClick={resetZoom}
                onMouseDown={dragZoomEnabled ? handlePointerDown : undefined}
                onMouseMove={dragZoomEnabled ? handlePointerMove : undefined}
                onMouseUp={dragZoomEnabled ? handlePointerUp : undefined}
                onMouseLeave={dragZoomEnabled ? handlePointerLeave : undefined}
                onClick={handleChartClick}
              >
              <CartesianGrid stroke="#d1d5db" />
              <XAxis
                dataKey="date"
                type="number"
                scale="time"
                tickFormatter={formatTick}
                tickMargin={8}
                padding={{ left: 12, right: 12 }}
                domain={xDomain ?? ["dataMin", "dataMax"]}
                allowDataOverflow
                ticks={xTicks}
                stroke="#000000"
                tick={{ fill: "#000000" }}
                tickLine={{ stroke: "#000000" }}
                axisLine={{ stroke: "#000000" }}
              />
              <YAxis
                domain={yDomain ?? [0, 100]}
                allowDataOverflow
                ticks={yTicks}
                stroke="#000000"
                tick={{ fill: "#000000" }}
                tickLine={{ stroke: "#000000" }}
                axisLine={{ stroke: "#000000" }}
                tickMargin={6}
                tickFormatter={(value: number) => Math.round(value).toString()}
                allowDecimals={false}
              />
              {!isSelecting && (
                <Tooltip
                  labelFormatter={formatTooltipLabel}
                  formatter={(value: unknown) =>
                    typeof value === "number"
                      ? tooltipNumberFormatter.format(value)
                      : ""
                  }
                  allowEscapeViewBox={{ x: true, y: true }}
                  wrapperStyle={{ zIndex: 20 }}
                  contentStyle={{
                    background: "var(--color-primary-light)",
                    borderColor: "var(--color-accent-light)",
                    color: "var(--color-primary-dark)",
                  }}
                  active={pinnedTooltip ? true : undefined}
                  content={(props) => {
                    const mergedProps = pinnedTooltip
                      ? {
                          ...props,
                          active: true,
                          payload: pinnedTooltip.payload,
                          label: pinnedTooltip.label,
                          coordinate: pinnedTooltip.coordinate ?? props.coordinate,
                        }
                      : props;
                    return (
                      <TooltipContent
                        {...mergedProps}
                        activeTeam={pinnedTeam ?? activeTeam}
                      />
                    );
                  }}
                />
              )}
              <Customized component={ChartMeta} />
              {refAreaLeft !== null &&
                refAreaRight !== null &&
                refAreaTop !== null &&
                refAreaBottom !== null && (
                  <ReferenceArea
                    x1={
                      lockedAxis === "y"
                        ? currentXDomain[0]
                        : refAreaLeft
                    }
                    x2={
                      lockedAxis === "y"
                        ? currentXDomain[1]
                        : refAreaRight
                    }
                    y1={
                      lockedAxis === "x"
                        ? yDomain?.[0] ?? 0
                        : refAreaTop
                    }
                    y2={
                      lockedAxis === "x"
                        ? yDomain?.[1] ?? 100
                        : refAreaBottom
                    }
                    strokeOpacity={0.2}
                    fill="var(--color-accent-light)"
                    fillOpacity={0.3}
                  />
                )}
              {selectedTeams.map((team, index) => (
                <Line
                  key={team}
                  type="monotone"
                  dataKey={team}
                  stroke={colorForIndex(index)}
                  dot={false}
                  strokeWidth={1.2}
                  isAnimationActive={false}
                  strokeOpacity={
                    activeTeam && activeTeam !== team ? 0.15 : 1
                  }
                  connectNulls={false}
                />
              ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      <div
        className="relative flex flex-col gap-4 overflow-hidden rounded-xl bg-white p-4 ring-1 ring-slate-200 shadow-sm xl:flex-[2] xl:self-stretch xl:min-h-0"
        style={isXl && chartCardHeight ? { height: chartCardHeight } : undefined}
      >
        <div className="mb-3 flex flex-col gap-3 text-sm text-ink-400 md:flex-row md:items-end md:justify-between">
          <div className="flex w-full flex-wrap items-center gap-2 md:w-auto">
            <label className="text-xs uppercase tracking-[0.2em] text-ink-400">
              Presets
            </label>
            <select
              value={activePreset}
              onChange={(event) => applyPreset(event.target.value)}
              className="w-full rounded-lg border border-ink-700 bg-white px-3 py-1 text-sm text-ebony md:w-60"
            >
              <option value="" disabled>
                Custom selection
              </option>
              {Array.from(
                presets.reduce((map, option) => {
                  if (!map.has(option.category)) {
                    map.set(option.category, []);
                  }
                  map.get(option.category)?.push(option);
                  return map;
                }, new Map<string, PresetOption[]>())
              ).map(([category, options]) => (
                <optgroup key={category} label={category}>
                  {options.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search teams"
            className="w-full rounded-lg border border-ink-900 bg-white px-3 py-1 text-sm text-ebony placeholder:text-ink-900/60 md:w-64"
          />
        </div>
        <div className="grid h-[320px] content-start grid-cols-2 gap-x-4 gap-y-2 overflow-y-auto pr-2 text-sm text-ebony md:grid-cols-3 xl:flex-1 xl:min-h-0 xl:h-auto">
          {visibleTeams.map((team) => (
            <label key={team} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={selectedSet.has(team)}
              onChange={() => {
                toggleTeam(team);
                setActivePreset("");
              }}
              className="h-4 w-4 rounded border-ink-700 accent-accent-dark focus:ring-ink-700"
            />
              <span>{team}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
