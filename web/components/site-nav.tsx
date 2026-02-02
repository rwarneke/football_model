"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";
import { cn } from "@/lib/utils";
import { Home, Trophy, ArrowUpDown, Clock, Percent, Shuffle, Globe } from "lucide-react";

type NavIcon = React.ComponentType<{ className?: string }>;

type NavChild = {
  label: string;
  href: string;
  icon: NavIcon;
};

type NavGroup = {
  label: string;
  href?: string;
  icon: NavIcon;
  children?: NavChild[];
};

const navGroups: NavGroup[] = [
  { label: "Home", href: "/", icon: Home },
  {
    label: "World Football Ratings",
    icon: Globe,
    children: [
      { label: "Current ratings", href: "/current-ratings", icon: ArrowUpDown },
      { label: "Historical ratings", href: "/history", icon: Clock },
    ],
  },
  {
    label: "FIFA World Cup 2026",
    icon: Trophy,
    children: [
      { label: "Progression chances", href: "/world-cup-2026/probabilities", icon: Percent },
      { label: "Tournament predictor", href: "/world-cup-2026/predictor", icon: Shuffle },
    ],
  },
];

export function SiteNav() {
  const pathname = usePathname();
  const [open, setOpen] = React.useState(false);
  const touchStartXRef = React.useRef<number | null>(null);
  const touchStartYRef = React.useRef<number | null>(null);

  const handleTouchStart = (event: React.TouchEvent<HTMLElement>) => {
    if (!open) return;
    const touch = event.touches[0];
    touchStartXRef.current = touch.clientX;
    touchStartYRef.current = touch.clientY;
  };

  const handleTouchEnd = (event: React.TouchEvent<HTMLElement>) => {
    if (!open || touchStartXRef.current === null || touchStartYRef.current === null) {
      return;
    }

    const touch = event.changedTouches[0];
    const deltaX = touch.clientX - touchStartXRef.current;
    const deltaY = touch.clientY - touchStartYRef.current;

    // Only consider mostly horizontal swipes
    if (Math.abs(deltaX) > Math.abs(deltaY) && deltaX < -40) {
      // Swipe left to close
      setOpen(false);
    }

    touchStartXRef.current = null;
    touchStartYRef.current = null;
  };

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
        className="lg:hidden fixed left-3.5 top-1.5 z-[70] flex h-9 w-9 items-center justify-center text-slate-900"
      >
        <span className="relative h-4 w-5">
          <span className="absolute left-0 top-0 h-[2px] w-full rounded-full bg-current" />
          <span className="absolute left-0 top-[7px] h-[2px] w-full rounded-full bg-current" />
          <span className="absolute left-0 bottom-0 h-[2px] w-full rounded-full bg-current" />
        </span>
      </button>

      {open && (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 z-[55] bg-black/20 lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      <nav
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
        className={cn(
          "fixed inset-y-0 left-0 z-[60] h-full overflow-hidden bg-white shadow-sm ring-1 ring-slate-200 transition-[width] duration-200 lg:static lg:shadow-none lg:ring-0",
          open ? "w-64" : "w-0",
          open ? "lg:w-64" : "lg:w-20"
        )}
      >
        <div className="flex h-full flex-col gap-4 px-2 pt-16 pb-4 lg:px-3 lg:pt-4">
          <button
            type="button"
            onClick={() => setOpen((prev) => !prev)}
            aria-expanded={open}
            className={cn(
              "hidden lg:flex group items-center gap-3 rounded-lg px-2 py-2 text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300",
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
                    "flex items-center gap-3 rounded-lg px-2 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100 lg:px-2",
                    isActive(group.href) && "bg-slate-100 text-slate-900",
                    open ? "justify-start" : "justify-center"
                  )}
                >
                  <group.icon className="h-6 w-6 shrink-0 text-slate-700" />
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
                <div className={cn("flex items-center gap-3 px-2 lg:px-2", open ? "justify-start" : "justify-center")}>
                  <group.icon className="h-6 w-6 shrink-0 text-slate-700" />
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
                        "flex items-center gap-3 rounded-lg px-2 py-1.5 text-sm text-slate-600 hover:bg-slate-100 hover:text-slate-900 lg:px-2",
                        isActive(child.href) && "bg-slate-100 text-slate-900",
                        open ? "justify-start" : "justify-center"
                      )}
                    >
                      <child.icon className="h-5 w-5 shrink-0 text-slate-600" />
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
