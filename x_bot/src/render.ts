import fs from "node:fs";
import path from "node:path";
import http from "node:http";
import https from "node:https";
import puppeteer from "puppeteer-core";
import type { MatchPreview } from "./types.js";

function requestOk(urlString: string): Promise<boolean> {
  return new Promise((resolve) => {
    const url = new URL(urlString);
    const lib = url.protocol === "https:" ? https : http;
    const req = lib.request(
      url,
      { method: "GET" },
      (res) => {
        resolve((res.statusCode ?? 500) < 400);
        res.resume();
      }
    );
    req.on("error", () => resolve(false));
    req.end();
  });
}

async function waitForUrl(url: string, timeoutMs = 15_000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await requestOk(url)) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function resolveBrowserPath() {
  const explicit = process.env.CHROMIUM_PATH?.trim();
  if (explicit) {
    return explicit;
  }

  const candidates = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "/opt/homebrew/bin/chromium",
    "chromium",
  ];

  for (const candidate of candidates) {
    if (candidate.includes("/") && fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return "chromium";
}

export async function renderPreviewImage(preview: MatchPreview) {
  fs.mkdirSync(path.dirname(preview.imagePath), { recursive: true });
  const baseUrl = process.env.MATCH_CARD_BASE_URL ?? "http://127.0.0.1:3000";
  const url = `${baseUrl.replace(/\/$/, "")}/social/match-card/${encodeURIComponent(
    preview.match.id
  )}`;

  await waitForUrl(url);

  const browserPath = resolveBrowserPath();
  const browser = await puppeteer.launch({
    executablePath: browserPath,
    headless: true,
    args: ["--disable-gpu", "--hide-scrollbars"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 560, height: 900, deviceScaleFactor: 2 });
    await page.goto(url, { waitUntil: "networkidle0" });

    const card = await page.$("#social-card-shot");
    if (!card) {
      throw new Error(`Could not find #social-card-shot on ${url}`);
    }
    const cardBox = await card.boundingBox();
    if (!cardBox) {
      throw new Error(`Could not determine screenshot bounds for ${url}`);
    }

    const pageSize = page.viewport();
    const margin = 2;
    const clip = {
      x: Math.max(0, cardBox.x + 1),
      y: Math.max(0, cardBox.y),
      width: Math.max(
        1,
        Math.min(
          cardBox.width,
          (pageSize?.width ?? Math.ceil(cardBox.x + cardBox.width)) - Math.max(0, cardBox.x + 1)
        )
      ),
      height: Math.max(
        1,
        Math.min(
          cardBox.height,
          (pageSize?.height ?? Math.ceil(cardBox.y + cardBox.height)) - Math.max(0, cardBox.y)
        )
      ),
    };

    await page.screenshot({
      path: preview.imagePath,
      type: "png",
      omitBackground: false,
      clip: {
        x: Math.max(0, clip.x),
        y: Math.max(0, clip.y),
        width: Math.max(1, clip.width - margin),
        height: Math.max(1, clip.height),
      },
    });
  } finally {
    await browser.close();
  }
}
