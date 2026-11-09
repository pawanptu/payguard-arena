import type {
  ConsoleExplanations,
  ConsoleQueue,
  CopilotSessions,
  EpisodeEvent,
  EventsPage,
  ExplanationsResult,
  Payout,
  RoundsResponse,
  Summary,
  WebhooksResponse,
} from "./types";

/** An HTTP failure that keeps the status and the server's `{"error": "..."}` code. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  constructor(url: string, status: number, code: string | null) {
    super(`${url} returned HTTP ${status}${code ? ` (${code})` : ""}`);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    let code: string | null = null;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "error" in body && typeof body.error === "string") code = body.error;
    } catch {
      /* not JSON */
    }
    throw new ApiError(url, res.status, code);
  }
  return (await res.json()) as T;
}

export const fetchSummary = () => getJson<Summary>("/api/replay/summary");

export async function fetchAllEvents(pageSize = 1000): Promise<EpisodeEvent[]> {
  const out: EpisodeEvent[] = [];
  for (;;) {
    const page = await getJson<EventsPage>(`/api/replay/events?offset=${out.length}&limit=${pageSize}`);
    out.push(...page.events);
    if (page.events.length === 0 || out.length >= page.total) return out;
  }
}

export const fetchPayouts = () => getJson<{ source: string; payouts: Payout[] }>("/api/payouts");

export const fetchWebhooks = (limit = 50) => getJson<WebhooksResponse>(`/api/webhooks/recent?limit=${limit}`);

/* ---------- Ops console ---------- */

export async function fetchQueue(): Promise<ConsoleQueue> {
  try {
    return await getJson<ConsoleQueue>("/api/console/queue");
  } catch (e) {
    if (e instanceof ApiError && e.status === 404 && e.code === "not_built") {
      throw new Error("The review queue has not been built yet.");
    }
    throw e;
  }
}

/** 404 {"error": "not_built"} resolves to `{ built: false }`; every other failure rejects. */
export async function fetchExplanations(): Promise<ExplanationsResult> {
  try {
    return { built: true, data: await getJson<ConsoleExplanations>("/api/console/explanations") };
  } catch (e) {
    if (e instanceof ApiError && e.status === 404 && e.code === "not_built") return { built: false };
    throw e;
  }
}

export const fetchCopilotSessions = () => getJson<CopilotSessions>("/api/copilot/sessions");

export const fetchRounds = () => getJson<RoundsResponse>("/api/results/rounds");
