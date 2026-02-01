import { headers } from "next/headers";
import type {
  GroupDefinition,
  GroupMatch,
  KnockoutMatch,
  QualifierMatch,
  RoundOf32Combos,
  WinProbabilityEntry,
  WinProbabilities,
  WorldCupPredictorData,
} from "@/lib/world-cup-predictor-types";
import { loadWorldCupPredictorDataWithFetchers } from "@/lib/world-cup-predictor-loader";

export type {
  GroupDefinition,
  GroupMatch,
  KnockoutMatch,
  QualifierMatch,
  RoundOf32Combos,
  WinProbabilityEntry,
  WinProbabilities,
  WorldCupPredictorData,
};

const textCache = new Map<string, string>();
const textInFlight = new Map<string, Promise<string>>();
const jsonCache = new Map<string, unknown>();
const jsonInFlight = new Map<string, Promise<unknown>>();

async function fetchText(filePath: string) {
  const cached = textCache.get(filePath);
  if (cached) {
    return cached;
  }
  const inflight = textInFlight.get(filePath);
  if (inflight) {
    return inflight;
  }
  const headerList = headers();
  const forwardedHost = headerList.get("x-forwarded-host");
  const hostValue = forwardedHost ?? headerList.get("host");
  const proto = headerList.get("x-forwarded-proto") ?? "https";
  if (!hostValue) {
    throw new Error("Missing host header for data fetch");
  }
  const host = hostValue.startsWith("0.0.0.0")
    ? hostValue.replace(/^0\.0\.0\.0/, "127.0.0.1")
    : hostValue;
  const promise = (async () => {
    const res = await fetch(`${proto}://${host}${filePath}`, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Failed to load ${filePath}`);
    }
    const text = await res.text();
    textCache.set(filePath, text);
    return text;
  })();
  textInFlight.set(filePath, promise);
  try {
    return await promise;
  } finally {
    textInFlight.delete(filePath);
  }
}

async function fetchJson(filePath: string) {
  const cached = jsonCache.get(filePath);
  if (cached) {
    return cached;
  }
  const inflight = jsonInFlight.get(filePath);
  if (inflight) {
    return inflight;
  }
  const headerList = headers();
  const forwardedHost = headerList.get("x-forwarded-host");
  const hostValue = forwardedHost ?? headerList.get("host");
  const proto = headerList.get("x-forwarded-proto") ?? "https";
  if (!hostValue) {
    return null;
  }
  const host = hostValue.startsWith("0.0.0.0")
    ? hostValue.replace(/^0\.0\.0\.0/, "127.0.0.1")
    : hostValue;
  const promise = (async () => {
    const res = await fetch(`${proto}://${host}${filePath}`, { cache: "no-store" });
    if (!res.ok) {
      return null;
    }
    const data = await res.json();
    jsonCache.set(filePath, data);
    return data;
  })();
  jsonInFlight.set(filePath, promise);
  try {
    return await promise;
  } finally {
    jsonInFlight.delete(filePath);
  }
}

export async function loadWorldCupPredictorData(): Promise<WorldCupPredictorData> {
  return loadWorldCupPredictorDataWithFetchers(fetchText, fetchJson);
}
