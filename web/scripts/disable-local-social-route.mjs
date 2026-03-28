import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const routePath = path.join(ROOT, "app", "social", "match-card", "[matchId]", "page.tsx");

if (fs.existsSync(routePath)) {
  fs.unlinkSync(routePath);
  console.log(`Removed local social route at ${path.relative(ROOT, routePath)}`);
} else {
  console.log("Local social route was already absent.");
}
