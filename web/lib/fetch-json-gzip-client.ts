"use client";

async function readGzipJson(res: Response) {
  const stream = res.body;
  if (!stream || typeof DecompressionStream === "undefined") {
    return null;
  }
  const decompressed = stream.pipeThrough(new DecompressionStream("gzip"));
  const text = await new Response(decompressed).text();
  return JSON.parse(text) as unknown;
}

export async function fetchJsonWithGzipFallback(filePath: string) {
  const gzipPath = filePath.endsWith(".json") ? `${filePath}.gz` : filePath;

  if (gzipPath !== filePath) {
    try {
      const gzipRes = await fetch(gzipPath);
      if (gzipRes.ok) {
        const data = await readGzipJson(gzipRes);
        if (data !== null) {
          return data;
        }
      }
    } catch {
      // Fall through to plain JSON fetch.
    }
  }

  const res = await fetch(filePath);
  if (!res.ok) {
    return null;
  }
  return (await res.json()) as unknown;
}
