import type { Metadata } from "next";
import { Inter, Source_Code_Pro } from "next/font/google";
import "./globals.css";
import { SiteNav } from "@/components/site-nav";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const sourceCodePro = Source_Code_Pro({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Global Soccer Ratings",
  description: "Current international soccer ratings and team strength metrics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${sourceCodePro.variable}`}>
      <body className="min-h-screen bg-white text-ebony antialiased">
        <header className="border-b border-ink-700/40 bg-[var(--color-accent-dark)]">
          <div className="px-3 md:px-12">
            <div className="mx-auto w-full max-w-6xl">
              <SiteNav />
            </div>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
