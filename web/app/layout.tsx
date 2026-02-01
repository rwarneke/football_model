import type { Metadata } from "next";
import { Bitter, Inter, Source_Code_Pro } from "next/font/google";
import "./globals.css";
import Link from "next/link";
import { Suspense } from "react";
import { SiteNav } from "@/components/site-nav";
import { RouteLoadingBar } from "@/components/route-loading-bar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const bitter = Bitter({
  subsets: ["latin"],
  variable: "--font-logo",
  weight: ["700"],
});
const sourceCodePro = Source_Code_Pro({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Global Soccer Ratings",
  description: "Current international soccer ratings and team strength metrics.",
  icons: {
    icon: [
      { url: "/favicon.ico" },
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon-32.png", sizes: "32x32", type: "image/png" },
      { url: "/favicon-16.png", sizes: "16x16", type: "image/png" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${bitter.variable} ${sourceCodePro.variable}`}>
      <body className="min-h-screen bg-white text-ebony antialiased">
        <Suspense fallback={null}>
          <RouteLoadingBar />
        </Suspense>
        <div className="fixed inset-x-0 top-0 z-[60] flex h-12 items-center justify-center border-b border-slate-200 bg-white">
          <Link
            href="/"
            className="flex items-end gap-1.5 text-[20px] leading-none font-bold text-ebony font-logo"
          >
            <svg
              aria-hidden="true"
              className="h-[18px] w-[18px]"
              viewBox="0 0 32 32"
              xmlns="http://www.w3.org/2000/svg"
            >
              <rect width="32" height="32" rx="6" ry="6" fill="black" />
              <path
                d="M8 8 H24 M8 8 V24"
                stroke="white"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            </svg>
            <span className="relative top-[2px]">TheBackPost</span>
          </Link>
        </div>
        <div className="flex min-h-screen">
          <aside className="relative z-[80] w-0 shrink-0 md:z-40 md:sticky md:top-12 md:h-[calc(100vh-3rem)] md:w-auto md:border-r md:border-slate-200 md:bg-white">
            <SiteNav />
          </aside>
          <div className="flex-1 min-w-0 pt-12">{children}</div>
        </div>
      </body>
    </html>
  );
}
