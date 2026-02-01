"use client";

import * as React from "react";
import { usePathname, useSearchParams } from "next/navigation";

export function RouteLoadingBar() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [progress, setProgress] = React.useState(0);
  const [visible, setVisible] = React.useState(false);
  const firstLoad = React.useRef(true);
  const timers = React.useRef<number[]>([]);

  const startProgress = React.useCallback(() => {
    if (visible) {
      return;
    }
    setVisible(true);
    setProgress(20);
    const t1 = window.setTimeout(() => setProgress(55), 120);
    const t2 = window.setTimeout(() => setProgress(80), 280);
    timers.current.push(t1, t2);
  }, [visible]);

  const finishProgress = React.useCallback(() => {
    const t3 = window.setTimeout(() => {
      setProgress(100);
      const t4 = window.setTimeout(() => {
        setVisible(false);
        setProgress(0);
      }, 220);
      timers.current.push(t4);
    }, 120);
    timers.current.push(t3);
  }, []);

  React.useEffect(() => {
    if (firstLoad.current) {
      firstLoad.current = false;
      return;
    }

    finishProgress();

    return () => {
      timers.current.forEach((timer) => window.clearTimeout(timer));
      timers.current = [];
    };
  }, [pathname, searchParams, finishProgress]);

  React.useEffect(() => {
    const shouldStartForLink = (link: HTMLAnchorElement) => {
      const href = link.getAttribute("href");
      if (!href || href.startsWith("#") || href.startsWith("mailto:") || href.startsWith("tel:")) {
        return false;
      }
      if (link.getAttribute("target") && link.getAttribute("target") !== "_self") {
        return false;
      }
      const url = new URL(href, window.location.href);
      if (url.origin !== window.location.origin) {
        return false;
      }
      const current = `${window.location.pathname}${window.location.search}`;
      const next = `${url.pathname}${url.search}`;
      return current !== next;
    };

    const shouldStartFromEvent = (event: Event) => {
      if (event.defaultPrevented) {
        return false;
      }
      if (event instanceof MouseEvent) {
        if (event.button !== 0) {
          return false;
        }
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
          return false;
        }
      } else if (event instanceof PointerEvent) {
        if (event.button !== 0) {
          return false;
        }
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
          return false;
        }
      }
      return true;
    };

    const handleStartFromEvent = (event: Event) => {
      if (!shouldStartFromEvent(event)) {
        return;
      }
      const target = event.target as HTMLElement | null;
      const link = target?.closest("a") as HTMLAnchorElement | null;
      if (!link || !shouldStartForLink(link)) {
        return;
      }
      startProgress();
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.key !== "Enter") {
        return;
      }
      handleStartFromEvent(event);
    };

    const handlePopState = () => {
      startProgress();
    };

    document.addEventListener("pointerdown", handleStartFromEvent, { capture: true });
    document.addEventListener("touchstart", handleStartFromEvent, { capture: true });
    document.addEventListener("click", handleStartFromEvent, { capture: true });
    document.addEventListener("keydown", handleKeyDown, { capture: true });
    window.addEventListener("popstate", handlePopState);
    return () => {
      document.removeEventListener("pointerdown", handleStartFromEvent, { capture: true } as EventListenerOptions);
      document.removeEventListener("touchstart", handleStartFromEvent, { capture: true } as EventListenerOptions);
      document.removeEventListener("click", handleStartFromEvent, { capture: true } as EventListenerOptions);
      document.removeEventListener("keydown", handleKeyDown, { capture: true } as EventListenerOptions);
      window.removeEventListener("popstate", handlePopState);
    };
  }, [startProgress]);

  if (!visible) {
    return null;
  }

  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-[90] h-0.5 bg-transparent">
      <div
        className="h-full bg-slate-900 transition-[width] duration-200 ease-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}
