"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Current Ratings" },
  { href: "/history", label: "Ratings History" },
  { href: "/world-cup-2026", label: "World Cup 2026" },
];

export function SiteNav() {
  const pathname = usePathname();

  return (
    <nav className="flex w-full items-center gap-6 py-4 text-sm">
      {links.map((link) => {
        const isActive =
          link.href === "/"
            ? pathname === "/"
            : pathname === link.href || pathname.startsWith(`${link.href}/`);
        return (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
              "transition hover:text-white",
              isActive ? "font-semibold text-white" : "text-white"
            )}
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
