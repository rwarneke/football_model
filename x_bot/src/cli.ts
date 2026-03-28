import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  loadDueMatchPreviews,
  loadPreviewForMatchId,
  loadPreviewForMatchIdWithOptions,
} from "./preview.js";
import { renderPreviewImage } from "./render.js";
import { loadCredsFile, postPreview } from "./post.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(MODULE_DIR, "..", "out");

function parseArgs(argv: string[]) {
  const nowArg = argv.find((arg) => arg.startsWith("--now="));
  const matchArg = argv.find((arg) => arg.startsWith("--match-id="));
  return {
    post: argv.includes("--post"),
    dryRun: argv.includes("--dry-run") || !argv.includes("--post"),
    force: argv.includes("--force"),
    variant: argv.includes("--variant"),
    nowIso: nowArg ? nowArg.slice("--now=".length) : null,
    matchId: matchArg ? matchArg.slice("--match-id=".length) : null,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  loadCredsFile();
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const now = args.nowIso ? new Date(args.nowIso) : new Date();
  if (Number.isNaN(now.getTime())) {
    throw new Error(`Invalid --now value: ${args.nowIso}`);
  }
  const previews = args.matchId
    ? [
        args.variant
          ? loadPreviewForMatchIdWithOptions(args.matchId, { variant: true })
          : loadPreviewForMatchId(args.matchId),
      ]
    : loadDueMatchPreviews(now, 1, { variant: args.variant });

  if (previews.length === 0) {
    console.log("No due match previews in the current 1-hour window.");
    return;
  }

  for (const preview of previews) {
    await renderPreviewImage(preview);
    console.log(`\n=== ${preview.match.home} vs ${preview.match.away} ===`);
    console.log(`Scheduled at: ${preview.scheduledAtIso}`);
    console.log(`Image: ${preview.imagePath}`);
    console.log(preview.postText);

    if (args.post) {
      const result = await postPreview(preview, { force: args.force });
      console.log(
        result.skipped
          ? `Skipped; already posted as ${result.tweetId}`
          : `Posted successfully: ${result.tweetId}`
      );
    }
  }

  if (args.dryRun) {
    console.log("\nDry run complete. No X posts were sent.");
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
