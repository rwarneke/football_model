import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import pngToIco from "png-to-ico";

const ROOT = process.cwd();

// Where your source SVG lives
const SRC_SVG = path.join(ROOT, "assets", "favicon.svg");

// Where outputs go (Next.js uses /public)
const OUT_DIR = path.join(ROOT, "public");

// PNG sizes to generate
const PNG_SIZES = [
  { name: "favicon-16.png", size: 16 },
  { name: "favicon-32.png", size: 32 },
  { name: "favicon-48.png", size: 48 },
  { name: "favicon-64.png", size: 64 },
  { name: "apple-touch-icon.png", size: 180 },
  { name: "android-chrome-192.png", size: 192 },
  { name: "android-chrome-512.png", size: 512 },
];

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function fileExists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  if (!(await fileExists(SRC_SVG))) {
    console.error(`Missing source SVG: ${SRC_SVG}`);
    console.error("Create it at assets/favicon.svg (or change SRC_SVG).");
    process.exit(1);
  }

  await ensureDir(OUT_DIR);

  const svgBuf = await fs.readFile(SRC_SVG);

  // Copy SVG to public
  await fs.writeFile(path.join(OUT_DIR, "favicon.svg"), svgBuf);

  // Render PNGs
  const generatedPngPaths = [];
  for (const { name, size } of PNG_SIZES) {
    const outPath = path.join(OUT_DIR, name);

    // Using "contain" so your rounded-square tile isn't cropped.
    // If you want full-bleed, change fit to "cover".
    await sharp(svgBuf, { density: 1024 })
      .resize(size, size, { fit: "contain" })
      .png()
      .toFile(outPath);

    generatedPngPaths.push(outPath);
    console.log(`Wrote ${path.relative(ROOT, outPath)}`);
  }

  // Build a multi-size .ico from a subset (common Windows sizes)
  const icoSizes = [16, 32, 48, 64];
  const icoPngs = await Promise.all(
    icoSizes.map((s) =>
      sharp(svgBuf, { density: 1024 }).resize(s, s, { fit: "contain" }).png().toBuffer()
    )
  );

  const icoBuf = await pngToIco(icoPngs);
  const icoPath = path.join(OUT_DIR, "favicon.ico");
  await fs.writeFile(icoPath, icoBuf);
  console.log(`Wrote ${path.relative(ROOT, icoPath)}`);

  console.log("\nDone.");
  console.log("Tip: commit the /public outputs and reference them in <head>.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
