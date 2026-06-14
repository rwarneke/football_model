import type { CSSProperties } from "react";

const RATING_RGB = "16, 185, 129";
const TILT_DEFENSIVE_RGB = "59, 130, 246";
const TILT_AGGRESSIVE_RGB = "234, 88, 12";
const DISPLAY_TILT_LIMIT = 10;

export function formatRatingValue(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  return value.toFixed(1);
}

export function formatTiltValue(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  const rounded = Math.abs(value) < 0.05 ? 0 : value;
  return rounded.toFixed(1);
}

export function ratingBackground(value: number | null | undefined): CSSProperties | undefined {
  if (value === null || value === undefined || !Number.isFinite(value)) {
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
  return { backgroundColor: `rgba(${RATING_RGB}, ${alpha})` };
}

export function tiltBackground(value: number | null | undefined): CSSProperties | undefined {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return undefined;
  }
  const clamped = Math.max(-DISPLAY_TILT_LIMIT, Math.min(value, DISPLAY_TILT_LIMIT));
  const magnitude = Math.abs(clamped) / DISPLAY_TILT_LIMIT;
  if (magnitude === 0) {
    return undefined;
  }
  const alpha = magnitude * 0.88;
  const rgb = clamped >= 0 ? TILT_AGGRESSIVE_RGB : TILT_DEFENSIVE_RGB;
  return { backgroundColor: `rgba(${rgb}, ${alpha})` };
}

export function ratingPillStyle(value: number | null | undefined): CSSProperties {
  return {
    ...(ratingBackground(value) ?? {}),
    borderColor: "rgba(148, 163, 184, 0.28)",
  };
}

export function tiltPillStyle(value: number | null | undefined): CSSProperties {
  return {
    ...(tiltBackground(value) ?? {}),
    borderColor: "rgba(148, 163, 184, 0.28)",
  };
}
