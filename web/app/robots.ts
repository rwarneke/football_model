import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
      disallow: ["/options-pricing"],
    },
    sitemap: "https://thebackpost.net/sitemap.xml",
  };
}
