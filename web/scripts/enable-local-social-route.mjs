import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const templatePath = path.join(ROOT, "local-templates", "social-match-card-page.tsx");
const routePath = path.join(ROOT, "app", "social", "match-card", "[matchId]", "page.tsx");

fs.mkdirSync(path.dirname(routePath), { recursive: true });
fs.copyFileSync(templatePath, routePath);
console.log(`Enabled local social route at ${path.relative(ROOT, routePath)}`);
