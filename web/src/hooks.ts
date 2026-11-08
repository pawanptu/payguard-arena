import { useCallback, useEffect, useRef, useState } from "react";

export interface AsyncState<T> {
  /** "loading" only before the first successful load; refreshes keep the previous data visible. */
  status: "loading" | "ready" | "error";
  data: T | null;
  error: string | null;
  /** True while any request (first load or refresh) is in flight. */
  busy: boolean;
  /** Re-run the request. The promise never rejects. */
  reload: () => Promise<void>;
}

export function useAsync<T>(fn: () => Promise<T>): AsyncState<T> {
  const [state, setState] = useState<Omit<AsyncState<T>, "reload">>({
    status: "loading",
    data: null,
    error: null,
    busy: true,
  });
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const seq = useRef(0);

  const run = useCallback(async () => {
    const id = ++seq.current;
    setState((s) => ({ status: s.status === "error" ? "loading" : s.status, data: s.data, error: null, busy: true }));
    try {
      const data = await fnRef.current();
      if (id === seq.current) setState({ status: "ready", data, error: null, busy: false });
    } catch (e: unknown) {
      if (id === seq.current) {
        setState((s) => ({ status: "error", data: s.data, error: e instanceof Error ? e.message : String(e), busy: false }));
      }
    }
  }, []);

  useEffect(() => {
    void run();
    return () => {
      seq.current++; // ignore results that arrive after unmount (or the StrictMode first pass)
    };
  }, [run]);

  return { ...state, reload: run };
}

/** Live `matchMedia` query, e.g. useMediaQuery("(max-width: 639px)"). */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => (typeof window !== "undefined" ? window.matchMedia(query).matches : false));
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setMatches(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return matches;
}
