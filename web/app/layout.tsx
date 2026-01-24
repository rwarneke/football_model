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
        <div className="fixed inset-x-0 top-0 z-30 h-16 border-b border-slate-200 bg-white md:hidden" />
        <div className="flex min-h-screen">
          <aside className="relative z-40 w-0 shrink-0 md:sticky md:top-0 md:h-screen md:w-auto md:border-r md:border-slate-200 md:bg-white">
            <SiteNav />
          </aside>
          <div className="flex-1 min-w-0 pt-16 md:pt-0">{children}</div>
        </div>
      </body>
    </html>
  );
}
