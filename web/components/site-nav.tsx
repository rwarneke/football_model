"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";
import { cn } from "@/lib/utils";

type NavGroup = {
  label: string;
  href?: string;
  children?: Array<{ label: string; href: string }>;
};

const navGroups: Array<
  NavGroup & { icon: string; children?: Array<{ label: string; href: string; icon: string }> }
> = [
  { label: "Home", href: "/", icon: "🏠" },
  {
    label: "World Football Ratings",
    icon: "⚽",
    children: [
      { label: "Current ratings", href: "/current-ratings", icon: "↕️" },
      { label: "Historical ratings", href: "/history", icon: "🕒" },
    ],
  },
  {
    label: "FIFA World Cup 2026",
    icon: "🏆",
    children: [
      { label: "Progression chances", href: "/world-cup-2026/probabilities", icon: "％" },
      { label: "Tournament predictor", href: "/world-cup-2026/predictor", icon: "🔀" },
    ],
  },
];

export function SiteNav() {
  const pathname = usePathname();
  const [open, setOpen] = React.useState(false);

  const handleNavSelect = React.useCallback(() => {
    setOpen(false);
  }, []);

  const isActive = (href: string) =>
    href === "/"
      ? pathname === "/"
      : pathname === href || pathname.startsWith(`${href}/`);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        className="md:hidden fixed left-4 top-3 z-50 flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-slate-200 text-slate-700 hover:bg-slate-100"
      >
        <span className="relative h-4 w-5">
          <span className="absolute left-0 top-0 h-0.5 w-full rounded-full bg-slate-700" />
          <span className="absolute left-0 top-[7px] h-0.5 w-full rounded-full bg-slate-700" />
          <span className="absolute left-0 bottom-0 h-0.5 w-full rounded-full bg-slate-700" />
        </span>
      </button>

      <nav
        className={cn(
          "fixed inset-y-0 left-0 z-40 h-full overflow-hidden bg-white shadow-sm ring-1 ring-slate-200 transition-[width] duration-200 md:static md:shadow-none md:ring-0",
          open ? "w-64" : "w-0",
          open ? "md:w-64" : "md:w-20"
        )}
      >
        <div className="flex h-full flex-col gap-4 px-2 pt-16 pb-4 md:px-3 md:pt-4">
          <button
            type="button"
            onClick={() => setOpen((prev) => !prev)}
            aria-expanded={open}
            className={cn(
              "hidden md:flex group items-center gap-3 rounded-lg px-2 py-2 text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300",
              open ? "justify-start" : "justify-center"
            )}
          >
            <span className="relative h-4 w-5 shrink-0">
              <span className="absolute left-0 top-0 h-0.5 w-full rounded-full bg-slate-700" />
              <span className="absolute left-0 top-[7px] h-0.5 w-full rounded-full bg-slate-700" />
              <span className="absolute left-0 bottom-0 h-0.5 w-full rounded-full bg-slate-700" />
            </span>
            <span
              className={cn(
                "text-xs font-semibold uppercase tracking-wide transition-all duration-200 overflow-hidden",
                open
                  ? "opacity-100 translate-x-0 max-w-[160px]"
                  : "pointer-events-none opacity-0 -translate-x-2 max-w-0"
              )}
            >
              Menu
            </span>
          </button>

        <div className="flex flex-col gap-3">
          {navGroups.map((group) => (
            <div key={group.label} className="space-y-2">
              {group.href ? (
                <Link
                  href={group.href}
                  onClick={handleNavSelect}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-2 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100 md:px-2",
                    isActive(group.href) && "bg-slate-100 text-slate-900",
                    open ? "justify-start" : "justify-center"
                  )}
                >
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-200 text-base">
                    {group.icon}
                  </span>
                  <span
                    className={cn(
                      "whitespace-nowrap transition-all duration-200 overflow-hidden",
                      open
                        ? "opacity-100 translate-x-0 max-w-[180px]"
                        : "pointer-events-none opacity-0 -translate-x-2 max-w-0"
                    )}
                  >
                    {group.label}
                  </span>
                </Link>
              ) : (
                <div className={cn("flex items-center gap-3 px-2 md:px-2", open ? "justify-start" : "justify-center")}>
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-200 text-base">
                    {group.icon}
                  </span>
                  <span
                    className={cn(
                      "whitespace-nowrap text-xs font-semibold uppercase tracking-wide text-slate-500 transition-all duration-200 overflow-hidden",
                      open
                        ? "opacity-100 translate-x-0 max-w-[180px]"
                        : "pointer-events-none opacity-0 -translate-x-2 max-w-0"
                    )}
                  >
                    {group.label}
                  </span>
                </div>
              )}
              {group.children && (
                <div className={cn("flex flex-col gap-1", open ? "pl-9" : "pl-0")}>
                  {group.children.map((child) => (
                    <Link
                      key={child.href}
                      href={child.href}
                      onClick={handleNavSelect}
                      className={cn(
                        "flex items-center gap-3 rounded-lg px-2 py-1.5 text-sm text-slate-600 hover:bg-slate-100 hover:text-slate-900 md:px-2",
                        isActive(child.href) && "bg-slate-100 text-slate-900",
                        open ? "justify-start" : "justify-center"
                      )}
                    >
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-slate-200 text-[11px]">
                        {child.icon}
                      </span>
                      <span
                        className={cn(
                          "whitespace-nowrap transition-all duration-200 overflow-hidden",
                          open
                            ? "opacity-100 translate-x-0 max-w-[160px]"
                            : "pointer-events-none opacity-0 -translate-x-2 max-w-0"
                        )}
                      >
                        {child.label}
                      </span>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </nav>
    </>
  );
}
