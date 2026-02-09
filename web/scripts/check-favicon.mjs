import fs from "node:fs/promises";
import path from "node:path";

const ROOT = process.cwd();
const PUBLIC_DIR = path.join(ROOT, "public");

const REQUIRED_PNGS = [
  { name: "favicon-16.png", size: 16 },
  { name: "favicon-32.png", size: 32 },
  { name: "favicon-48.png", size: 48 },
];

const REQUIRED_FILES = ["favicon.ico"];

function fail(message) {
  console.error(`favicon check failed: ${message}`);
  process.exit(1);
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function readPngDimensions(buffer) {
  const PNG_SIGNATURE = "89504e470d0a1a0a";
  const signature = buffer.subarray(0, 8).toString("hex");
  if (signature !== PNG_SIGNATURE) {
    return null;
  }
  // IHDR starts at byte 8, chunk type at 12, width/height at 16/20.
  const chunkType = buffer.subarray(12, 16).toString("ascii");
  if (chunkType !== "IHDR") {
    return null;
  }
  const width = buffer.readUInt32BE(16);
  const height = buffer.readUInt32BE(20);
  return { width, height };
}

async function checkPngSize(name, size) {
  const filePath = path.join(PUBLIC_DIR, name);
  if (!(await fileExists(filePath))) {
    fail(`missing ${path.relative(ROOT, filePath)}`);
  }
  const buffer = await fs.readFile(filePath);
  const dims = readPngDimensions(buffer);
  if (!dims) {
    fail(`${name} is not a valid PNG`);
  }
  if (dims.width !== size || dims.height !== size) {
    fail(`${name} is ${dims.width}x${dims.height}, expected ${size}x${size}`);
  }
}

async function main() {
  for (const name of REQUIRED_FILES) {
    const filePath = path.join(PUBLIC_DIR, name);
    if (!(await fileExists(filePath))) {
      fail(`missing ${path.relative(ROOT, filePath)}`);
    }
  }

  for (const { name, size } of REQUIRED_PNGS) {
    await checkPngSize(name, size);
  }

  console.log("favicon check passed");
}

main().catch((err) => {
  fail(err?.message ?? String(err));
});
