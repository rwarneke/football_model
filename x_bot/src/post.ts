import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { TwitterApi } from "twitter-api-v2";
import type { MatchPreview } from "./types.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const STATE_DIR = path.resolve(MODULE_DIR, "..", "state");
const STATE_PATH = path.join(STATE_DIR, "sent-posts.json");

type SentState = {
  sent: Record<string, { tweetId: string; postedAtIso: string }>;
};

function loadState(): SentState {
  if (!fs.existsSync(STATE_PATH)) {
    return { sent: {} };
  }
  return JSON.parse(fs.readFileSync(STATE_PATH, "utf8")) as SentState;
}

function saveState(state: SentState) {
  fs.mkdirSync(STATE_DIR, { recursive: true });
  fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2));
}

function requireEnv(name: string) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

export function loadCredsFile() {
  const credsPath = path.resolve(MODULE_DIR, "..", "creds");
  if (!fs.existsSync(credsPath)) {
    return;
  }
  const lines = fs.readFileSync(credsPath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) {
      continue;
    }
    const key = trimmed.slice(0, eqIndex).trim();
    const value = trimmed.slice(eqIndex + 1).trim();
    if (key && value && !process.env[key]) {
      process.env[key] = value;
    }
  }
}

function createClient() {
  return new TwitterApi({
    appKey: requireEnv("X_API_KEY"),
    appSecret: requireEnv("X_API_KEY_SECRET"),
    accessToken: requireEnv("X_ACCESS_TOKEN"),
    accessSecret: requireEnv("X_ACCESS_TOKEN_SECRET"),
  });
}

export async function postPreview(preview: MatchPreview, options?: { force?: boolean }) {
  loadCredsFile();
  const state = loadState();
  if (!options?.force && state.sent[preview.dedupeKey]) {
    return { skipped: true as const, tweetId: state.sent[preview.dedupeKey].tweetId };
  }

  const client = createClient();
  const mediaId = await client.v1.uploadMedia(preview.imagePath, { mimeType: "image/png" });
  const response = await client.v2.tweet({
    text: preview.postText,
    media: { media_ids: [mediaId] },
  });

  state.sent[preview.dedupeKey] = {
    tweetId: response.data.id,
    postedAtIso: new Date().toISOString(),
  };
  saveState(state);
  return { skipped: false as const, tweetId: response.data.id };
}
