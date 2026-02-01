import type { Metadata } from "next";
import { Bitter, Inter, Source_Code_Pro } from "next/font/google";
import "./globals.css";
import { SiteNav } from "@/components/site-nav";

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
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${bitter.variable} ${sourceCodePro.variable}`}>
      <body className="min-h-screen bg-white text-ebony antialiased">
        <div className="fixed inset-x-0 top-0 z-[60] flex h-12 items-center justify-center border-b border-slate-200 bg-white md:hidden">
          <span className="text-xl font-bold text-ebony font-logo">
            TheBackPost
          </span>
        </div>
        <div className="flex min-h-screen">
          <aside className="relative z-[80] w-0 shrink-0 md:z-40 md:sticky md:top-0 md:h-screen md:w-auto md:border-r md:border-slate-200 md:bg-white">
            <SiteNav />
          </aside>
          <div className="flex-1 min-w-0 pt-12 md:pt-0">{children}</div>
        </div>
      </body>
    </html>
  );
}
