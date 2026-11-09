import { useSyncExternalStore } from "react";

/** Hash routes without a router library: #/console, #/console/24, #/copilot/<id>, #/results, #/how. */
export type View = "console" | "copilot" | "results" | "how";

export interface Route {
  view: View;
  param: string | null;
}

export const VIEWS: { view: View; label: string }[] = [
  { view: "console", label: "Review queue" },
  { view: "copilot", label: "Copilot" },
  { view: "results", label: "Results" },
  { view: "how", label: "How it works" },
];

const DEFAULT: Route = { view: "console", param: null };
const EVENT = "payguard:route";

/** null = not a route hash (an in-page anchor such as #paypal): the current view stays. */
export function parseHash(hash: string): Route | null {
  if (hash === "" || hash === "#" || hash === "#/") return DEFAULT;
  if (!hash.startsWith("#/")) return null;
  const [view, ...rest] = hash.slice(2).split("/");
  const known = VIEWS.find((v) => v.view === view);
  if (!known) return DEFAULT;
  let param: string | null = rest.join("/") || null;
  if (param) {
    try {
      param = decodeURIComponent(param);
    } catch {
      /* keep raw */
    }
  }
  return { view: known.view, param };
}

export const hrefFor = (view: View, param?: string | number | null) =>
  `#/${view}${param !== undefined && param !== null && param !== "" ? `/${encodeURIComponent(String(param))}` : ""}`;

let current: Route = parseHash(typeof window === "undefined" ? "" : window.location.hash) ?? DEFAULT;
const key = (r: Route) => `${r.view}\n${r.param ?? ""}`;

function snapshot(): string {
  const r = parseHash(window.location.hash);
  if (r && key(r) !== key(current)) current = r;
  return key(current);
}

function subscribe(cb: () => void): () => void {
  window.addEventListener("hashchange", cb);
  window.addEventListener(EVENT, cb);
  return () => {
    window.removeEventListener("hashchange", cb);
    window.removeEventListener(EVENT, cb);
  };
}

export function useRoute(): Route {
  useSyncExternalStore(subscribe, snapshot, () => key(DEFAULT));
  return current;
}

/** Change the selection inside a view without adding a history entry (arrow-key navigation). */
export function replaceRoute(view: View, param?: string | number | null): void {
  try {
    window.history.replaceState(null, "", hrefFor(view, param));
  } catch {
    window.location.hash = hrefFor(view, param);
  }
  window.dispatchEvent(new Event(EVENT));
}
