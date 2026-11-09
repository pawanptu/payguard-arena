import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { fetchCopilotSessions, fetchExplanations, fetchQueue } from "../api";
import { useAsync, useMediaQuery } from "../hooks";
import type { AsyncState } from "../hooks";
import { hrefFor, replaceRoute, useRoute } from "../route";
import { scoreboard, useReview } from "../review";
import type { Score } from "../review";
import type { ConsoleCase, ConsoleQueue, CopilotSession, ExplanationsResult } from "../types";
import { pct, SEP, usd } from "../format";
import { TIPS } from "../tips";
import InfoTip from "../components/InfoTip";
import { CardHeader, ErrorCard, Segmented, Skeleton } from "../components/ui";
import CaseDetail from "./CaseDetail";
import QueueList, { rowDomId } from "./QueueList";
import { specialistInfo } from "./labels";

type Filter = "all" | "hold" | "step";

const PAGE = 50;

function Kpi({ label, tip, children, sub, title }: { label: string; tip?: ReactNode; children: ReactNode; sub: ReactNode; title?: string }) {
  return (
    <div className="kpi" title={title}>
      <div className="kpi-head">
        <span className="kpi-label">{label}</span>
        {tip && <InfoTip label={label}>{tip}</InfoTip>}
      </div>
      <div className="kpi-value">{children}</div>
      <p className="kpi-sub">{sub}</p>
    </div>
  );
}

function KpiStrip({ q, score, onReset }: { q: ConsoleQueue; score: Score; onReset: () => void }) {
  const k = q.counts;
  const stoppedShare = k.attack_usd_total > 0 ? (k.attack_usd_stopped / k.attack_usd_total) * 100 : null;
  const fraudSub = `of ${usd(k.attack_usd_total)} in fraud${stoppedShare !== null ? `${SEP}${pct(stoppedShare)}` : ""}${SEP}${usd(k.attack_usd_leaked)} got through`;
  return (
    <div className="kpis kpis-console" role="group" aria-label="Replay summary">
      <Kpi label="Cases" sub={`${k.HOLD.toLocaleString("en-US")} on hold${SEP}${k.STEP_UP.toLocaleString("en-US")} stepped up`}>
        {k.cases.toLocaleString("en-US")}
      </Kpi>
      <Kpi label="Fraud stopped" tip={TIPS.leakage} sub={fraudSub} title={fraudSub}>
        {usd(k.attack_usd_stopped)}
      </Kpi>
      <Kpi label="Honest payouts held" tip={TIPS.honestHeldKpi} sub="Waiting for a reviewer">
        {k.cases_honest.toLocaleString("en-US")}
      </Kpi>
      <Kpi
        label="Your calls vs the swarm"
        tip={TIPS.scoreboard}
        sub={
          score.reviewed > 0 ? (
            <>
              Right calls on {score.reviewed} reviewed
              <button type="button" className="link-btn" onClick={onReset}>
                Reset score
              </button>
            </>
          ) : (
            "You and the agents, same cases"
          )
        }
      >
        {score.reviewed > 0 ? (
          <>
            {score.yours}/{score.reviewed} <span className="kpi-vs">vs</span> {score.swarm}/{score.reviewed}
          </>
        ) : (
          <span className="kpi-empty">Make a call to start</span>
        )}
      </Kpi>
    </div>
  );
}

function ConsoleSkeleton() {
  return (
    <div aria-busy="true" className="console-load">
      <span className="visually-hidden">Loading the review queue</span>
      <div className="kpis kpis-console">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="skel-tile" />
        ))}
      </div>
      <div className="console-grid">
        <div className="card">
          <Skeleton className="skel-controls" />
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="skel-row" />
          ))}
        </div>
        <div className="card">
          <Skeleton className="skel-title" />
          <Skeleton className="skel-chart" />
        </div>
      </div>
    </div>
  );
}

/** event_id -> id of the first recorded copilot session that looked at it (via a tool call's args). */
function sessionIndex(sessions: CopilotSession[]): Map<number, string> {
  const m = new Map<number, string>();
  for (const s of sessions) {
    for (const t of s.turns) {
      if (t.role !== "tool_call" || !t.args) continue;
      const raw = (t.args as Record<string, unknown>).event_id;
      const id = typeof raw === "number" ? raw : typeof raw === "string" && /^\d+$/.test(raw) ? Number(raw) : null;
      if (id !== null && !m.has(id)) m.set(id, s.id);
    }
  }
  return m;
}

export default function ConsoleView() {
  const queue = useAsync(fetchQueue);
  const explanations = useAsync(fetchExplanations);
  const sessions = useAsync(fetchCopilotSessions);
  const route = useRoute();
  const narrow = useMediaQuery("(max-width: 1023px)");

  // On a phone an open case replaces the whole list screen, page head included.
  const routeId = route.view === "console" && route.param !== null && /^\d+$/.test(route.param) ? Number(route.param) : null;
  const detailOnPhone = narrow && routeId !== null && Boolean(queue.data?.cases.some((c) => c.event_id === routeId));

  return (
    <div className="console-page">
      {detailOnPhone ? (
        <h1 className="visually-hidden">Payout review</h1>
      ) : (
        <div className="page-head">
          <h1>Payout review</h1>
          <p className="page-sub">Risky payouts the defender agents flagged. Pick one, read the evidence, then make your call.</p>
        </div>
      )}
      {queue.status === "error" ? (
        <ErrorCard what="the review queue" message={queue.error} onRetry={() => void queue.reload()} />
      ) : queue.data === null ? (
        <ConsoleSkeleton />
      ) : (
        <Loaded q={queue.data} explanations={explanations} sessions={sessions.data?.sessions ?? []} />
      )}
    </div>
  );
}

function Loaded({
  q,
  explanations,
  sessions,
}: {
  q: ConsoleQueue;
  explanations: AsyncState<ExplanationsResult>;
  sessions: CopilotSession[];
}) {
  const route = useRoute();
  const narrow = useMediaQuery("(max-width: 1023px)");
  const review = useReview();
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [cursor, setCursor] = useState<number | null>(null);
  const [limit, setLimit] = useState(PAGE);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const pushed = useRef(false);

  const specialists = useMemo(() => specialistInfo(q), [q]);
  const cases = useMemo(
    () => [...q.cases].sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity)),
    [q.cases],
  );
  const byId = useMemo(() => new Map(cases.map((c) => [c.event_id, c])), [cases]);
  const sessionFor = useMemo(() => sessionIndex(sessions), [sessions]);
  const decidedIds = useMemo(() => new Set(Object.keys(review.decisions).map(Number)), [review.decisions]);
  const score = useMemo(() => scoreboard(cases, review.decisions), [cases, review.decisions]);

  const counts = useMemo(
    () => ({ all: cases.length, hold: cases.filter((c) => c.verdict === "HOLD").length, step: cases.filter((c) => c.verdict === "STEP_UP").length }),
    [cases],
  );
  const visible = useMemo(() => {
    const term = search.trim().replace(/^#/, "");
    return cases.filter(
      (c) =>
        (filter === "all" || (filter === "hold" ? c.verdict === "HOLD" : c.verdict === "STEP_UP")) &&
        (term === "" || String(c.event_id).includes(term)),
    );
  }, [cases, filter, search]);

  // A new filter or search starts the phone list at the first page again.
  useEffect(() => setLimit(PAGE), [filter, search]);

  // The case opened through the route (#/console/24). Unknown ids fall back to the list.
  const routeId = route.view === "console" && route.param !== null && /^\d+$/.test(route.param) ? Number(route.param) : null;
  const opened: ConsoleCase | null = routeId !== null ? byId.get(routeId) ?? null : null;
  const firstVisible = visible[0]?.event_id ?? null;
  const visibleIds = useMemo(() => new Set(visible.map((c) => c.event_id)), [visible]);
  // Desktop: list and detail are side by side, the selected case is always shown. Phones: list OR detail.
  const selected: ConsoleCase | null = narrow ? opened : opened ?? (firstVisible !== null ? byId.get(firstVisible) ?? null : null);
  const activeId = narrow
    ? cursor !== null && visibleIds.has(cursor)
      ? cursor
      : firstVisible
    : selected && visibleIds.has(selected.event_id)
      ? selected.event_id
      : firstVisible;

  // Phones render a page of rows at a time (the page scrolls, the list does not); the active row is always rendered.
  const activeIdx = activeId === null ? -1 : visible.findIndex((c) => c.event_id === activeId);
  const effLimit = narrow ? Math.max(limit, activeIdx + 1) : visible.length;
  const shown = narrow ? visible.slice(0, effLimit) : visible;
  const remaining = visible.length - shown.length;

  // Selecting through the filter: when the open case is filtered out on desktop, the first visible one takes over.
  useEffect(() => {
    if (!narrow && opened && !visibleIds.has(opened.event_id) && firstVisible !== null) replaceRoute("console", firstVisible);
  }, [narrow, opened, visibleIds, firstVisible]);

  const open = (id: number, push: boolean) => {
    if (push) {
      pushed.current = true;
      window.location.hash = hrefFor("console", id);
    } else replaceRoute("console", id);
  };
  const onSelect = (id: number) => {
    if (narrow) {
      setCursor(id);
      open(id, true);
    } else open(id, false);
  };
  const move = (delta: number) => {
    if (visible.length === 0) return;
    const i = Math.max(0, visible.findIndex((c) => c.event_id === activeId));
    const next = visible[Math.min(visible.length - 1, Math.max(0, i + delta))].event_id;
    if (narrow) setCursor(next);
    else open(next, false);
  };
  const onListKey = (e: KeyboardEvent<HTMLUListElement>) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    switch (e.key) {
      case "ArrowDown":
      case "j":
        e.preventDefault();
        move(1);
        break;
      case "ArrowUp":
      case "k":
        e.preventDefault();
        move(-1);
        break;
      case "Home":
        e.preventDefault();
        move(-visible.length);
        break;
      case "End":
        e.preventDefault();
        move(visible.length);
        break;
      case "Enter":
        e.preventDefault();
        if (activeId !== null) {
          if (narrow) open(activeId, true);
          else headingRef.current?.focus();
        }
        break;
    }
  };

  // Moving between the phone's list and detail screens: focus the new screen's heading, and on the way back
  // bring the case you just left into view.
  const wasOpen = useRef(false);
  useEffect(() => {
    if (!narrow) return;
    const isOpen = selected !== null;
    if (isOpen && !wasOpen.current) {
      window.scrollTo({ top: 0 });
      headingRef.current?.focus({ preventScroll: true });
    } else if (!isOpen && wasOpen.current) {
      pushed.current = false;
      const id = cursor;
      window.requestAnimationFrame(() => {
        if (id !== null) document.getElementById(rowDomId(id))?.scrollIntoView({ block: "nearest" });
      });
    }
    wasOpen.current = isOpen;
  }, [narrow, selected, cursor]);

  const back = () => {
    if (pushed.current) {
      pushed.current = false;
      window.history.back();
    } else replaceRoute("console");
  };

  const clearFilters = () => {
    setFilter("all");
    setSearch("");
  };

  const showList = !narrow || selected === null;
  const showDetail = selected !== null;
  const copilotHref = selected ? hrefFor("copilot", sessionFor.get(selected.event_id)) : hrefFor("copilot");
  const src = q.source;

  return (
    <>
      {!(narrow && selected) && <KpiStrip q={q} score={score} onReset={review.reset} />}
      <div className="console-grid">
        {showList && (
          <section className="card queue" aria-labelledby="queue-h">
            <CardHeader
              id="queue-h"
              title="Review queue"
              tipLabel="the review queue"
              tip={`Holds first, then the most dollars at risk. Expected dollars at risk = risk score x amount. Use the arrow keys or J and K to move through the list${narrow ? ", Enter to open a case" : ""}.`}
              subtitle={
                <>
                  <span aria-live="polite">
                    {visible.length.toLocaleString("en-US")} of {cases.length.toLocaleString("en-US")} cases
                  </span>
                  <span className="kbd-hint" aria-hidden="true">{SEP}↑↓ or J/K to move</span>
                </>
              }
            />
            <div className="controls queue-controls">
              <Segmented
                label="Filter by verdict"
                value={filter}
                onChange={setFilter}
                options={[
                  { value: "all", label: `All ${counts.all}` },
                  { value: "hold", label: `Hold ${counts.hold}` },
                  { value: "step", label: `Step-up ${counts.step}` },
                ]}
              />
              <label className="search">
                <span className="visually-hidden">Search by payout number</span>
                <input
                  type="search"
                  inputMode="numeric"
                  placeholder="Payout number, e.g. 3759"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "ArrowDown") {
                      e.preventDefault();
                      document.querySelector<HTMLElement>(".qlist")?.focus();
                    }
                  }}
                />
              </label>
            </div>
            <QueueList
              cases={shown}
              activeId={activeId}
              decided={decidedIds}
              specialists={specialists}
              onSelect={onSelect}
              onKeyDown={onListKey}
              onClear={clearFilters}
              label="Cases to review, holds first, then the most dollars at risk"
            />
            {remaining > 0 && (
              <button type="button" className="btn show-more" onClick={() => setLimit(effLimit + PAGE)}>
                Show {Math.min(PAGE, remaining)} more{SEP}{remaining.toLocaleString("en-US")} left
              </button>
            )}
          </section>
        )}
        {showDetail && selected && (
          <CaseDetail
            key={selected.event_id}
            ref={headingRef}
            c={selected}
            queue={q}
            specialists={specialists}
            explanations={explanations}
            decision={review.decisions[String(selected.event_id)]}
            revealed={review.revealed.includes(selected.event_id)}
            copilotHref={copilotHref}
            narrow={narrow}
            onDecide={(v) => review.decide(selected.event_id, v)}
            onReveal={() => review.reveal(selected.event_id)}
            onBack={narrow ? back : undefined}
          />
        )}
        {!selected && !narrow && (
          <div className="card">
            <div className="empty" role="status">
              <p className="empty-title">Select a payout</p>
              <p className="empty-cause">Pick a case in the list, or use the arrow keys or J and K.</p>
            </div>
          </div>
        )}
      </div>
      {!(narrow && selected) && (
        <p className="console-foot muted">
          Replay of {src.parity_checked_events.toLocaleString("en-US")} payouts, checked against the published replay.
          <InfoTip label="the source">
            {src.defender} on {src.split} (seed {src.seed}).
          </InfoTip>
        </p>
      )}
    </>
  );
}
