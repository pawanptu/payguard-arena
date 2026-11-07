import type { EpisodeEvent, EventsPage, Payout, Summary } from "./types";

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${url} returned HTTP ${res.status}`);
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
