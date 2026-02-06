import type { MetadataRoute } from "next";

const baseUrl = "https://thebackpost.net";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: `${baseUrl}/`,
      changeFrequency: "daily",
      priority: 1,
    },
    {
      url: `${baseUrl}/current-ratings`,
      changeFrequency: "daily",
      priority: 0.9,
    },
    {
      url: `${baseUrl}/history`,
      changeFrequency: "weekly",
      priority: 0.6,
    },
    {
      url: `${baseUrl}/world-cup-2026`,
      changeFrequency: "weekly",
      priority: 0.8,
    },
    {
      url: `${baseUrl}/world-cup-2026/probabilities`,
      changeFrequency: "daily",
      priority: 0.9,
    },
    {
      url: `${baseUrl}/world-cup-2026/matches`,
      changeFrequency: "daily",
      priority: 0.9,
    },
    {
      url: `${baseUrl}/world-cup-2026/predictor`,
      changeFrequency: "weekly",
      priority: 0.7,
    },
  ];
}
