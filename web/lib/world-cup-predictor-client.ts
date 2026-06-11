import type { WorldCupPredictorData } from "@/lib/world-cup-predictor-types";
import { fetchJsonWithGzipFallback } from "@/lib/fetch-json-gzip-client";
import { loadWorldCupPredictorDataWithFetchers } from "@/lib/world-cup-predictor-loader";

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
  const promise = (async () => {
    const res = await fetch(filePath);
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
  const promise = (async () => {
    const data = await fetchJsonWithGzipFallback(filePath);
    if (!data) {
      return null;
    }
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

export async function loadWorldCupPredictorDataClient(
  modelOutputDir = "/model_output"
): Promise<WorldCupPredictorData> {
  return loadWorldCupPredictorDataWithFetchers(fetchText, fetchJson, modelOutputDir);
}
