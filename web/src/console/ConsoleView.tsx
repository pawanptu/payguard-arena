import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { fetchCopilotSessions, fetchExplanations, fetchQueue } from "../api";
import { useAsync, useMediaQuery } from "../hooks";
import type { AsyncState } from "../hooks";
import { hrefFor, replaceRoute, useRoute } from "../route";
import { scoreboard, useReview } from "../review";
import type { ConsoleCase, ConsoleQueue, CopilotSession, ExplanationsResult } from "../types";
import { pct, SEP, usd } from "../format";
import InfoTip from "../components/InfoTip";
import { EmptyState, ErrorCard, Segmented, SimPill, Skeleton } from "../components/ui";
import { InboxIcon } from "../components/Icons";
import CaseDetail from "./CaseDetail";
import QueueList from "./QueueList";
import { num, specialistInfo } from "./labels";

type Filter = "all" | "hold" | "step";

function Kpi({ label, tip, children, sub }: { label: string; tip?: ReactNode; children: ReactNode; sub: ReactNode }) {
  return (
    <div className="kpi">
      <div className="kpi-head">
        <span className="kpi-label">{label}</span>
        {tip && <InfoTip label={label}>{tip}</InfoTip>}
        <SimPill />
      </div>
      <div className="kpi-value">{children}</div>
      <p className="kpi-sub">{sub}</p>
    </div>
  );
}

function KpiStrip({ q }: { q: ConsoleQueue }) {
  const k = q.counts;
  const stoppedShare = k.attack_usd_total > 0 ? (k.attack_usd_stopped / k.attack_usd_total) * 100 : null;
  return (
    <>
    <p className="kpi-caption"><SimPill /> Simulated replay numbers</p>
    <div className="kpis kpis-console" role="group" aria-label="Replay summary (simulated)">
      <Kpi label="Cases in queue" sub={`${k.HOLD.toLocaleString("en-US")} HOLD${SEP}${k.STEP_UP.toLocaleString("en-US")} STEP_UP`}>
        {k.cases.toLocaleString("en-US")}
      </Kpi>
      <Kpi
        label="Fraud USD stopped"
        tip="Attack payouts the defender held or stepped up, in simulated dollars, over the whole replayed episode. Ground truth comes from the simulator."
        sub={
          <>
            of {usd(k.attack_usd_total)} attack volume{stoppedShare !== null && <>{SEP}{pct(stoppedShare)}</>}
          </>
        }
      >
        {usd(k.attack_usd_stopped)}
      </Kpi>
      <Kpi
        label="Honest payouts held for review"
        tip="Friction, not an error: legitimate payouts that wait for a reviewer. This is the price of catching fraud and it is reported next to the fraud numbers."
        sub="Friction, not an error: honest payouts waiting for review"
      >
        {k.cases_honest.toLocaleString("en-US")}
      </Kpi>
      <Kpi
        label="Thresholds"
        tip="The swarm's aggregated risk, 0 to 1. At or above the step-up line the payout needs extra verification; at or above the hold line it is held."
        sub="risk at or above each line"
      >
        <span className="kpi-thr">
          <span>Step-up {num(q.thresholds.step_up)}</span>
          <span>Hold {num(q.thresholds.hold)}</span>
        </span>
      </Kpi>
    </div>
    </>
  );
}

function ConsoleSkeleton() {
  return (
    <div aria-busy="true">
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

  return (
    <>
      <div className="page-head">
        <h1>Payout review console</h1>
        <p className="page-sub">
          Payouts the defender swarm held or stepped up, replayed from a simulated episode. Pick a case, read the evidence,
          then make your own call.
        </p>
      </div>
      {queue.status === "error" ? (
        <ErrorCard what="the review queue" message={queue.error} onRetry={() => void queue.reload()} />
      ) : queue.data === null ? (
        <ConsoleSkeleton />
      ) : (
        <Loaded q={queue.data} explanations={explanations} sessions={sessions.data?.sessions ?? []} />
      )}
    </>
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
  const headingRef = useRef<HTMLHeadingElement>(null);

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

  // Selecting through the filter: when the open case is filtered out on desktop, the first visible one takes over.
  useEffect(() => {
    if (!narrow && opened && !visibleIds.has(opened.event_id) && firstVisible !== null) replaceRoute("console", firstVisible);
  }, [narrow, opened, visibleIds, firstVisible]);

  const open = (id: number, push: boolean) => {
    if (push) window.location.hash = hrefFor("console", id);
    else replaceRoute("console", id);
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

  // Moving between the phone's list and detail screens: put focus on the new screen's heading.
  const wasOpen = useRef(false);
  useEffect(() => {
    if (!narrow) return;
    const isOpen = selected !== null;
    if (isOpen && !wasOpen.current) {
      window.scrollTo({ top: 0 });
      headingRef.current?.focus({ preventScroll: true });
    }
    wasOpen.current = isOpen;
  }, [narrow, selected]);

  const showList = !narrow || selected === null;
  const showDetail = !narrow ? selected !== null : selected !== null;
  const copilotHref = selected ? hrefFor("copilot", sessionFor.get(selected.event_id)) : hrefFor("copilot");

  return (
    <>
      {!(narrow && selected) && <KpiStrip q={q} />}
      <div className="console-grid">
        {showList && (
          <section className="card queue" aria-labelledby="queue-h">
            <header className="card-head">
              <div className="card-head-main">
                <h2 id="queue-h">
                  Review queue
                  <InfoTip label="the review queue">
                    Highest risk first. Use the arrow keys or J and K to move through the list{narrow ? ", Enter to open a case" : ""}.
                  </InfoTip>
                </h2>
                <p className="card-sub">
                  {visible.length.toLocaleString("en-US")} of {cases.length.toLocaleString("en-US")} cases{SEP}simulated
                </p>
              </div>
            </header>
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
                <span className="visually-hidden">Search by event id</span>
                <input
                  type="search"
                  inputMode="numeric"
                  placeholder="Event id, e.g. 24"
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
              cases={visible}
              activeId={activeId}
              decided={decidedIds}
              specialists={specialists}
              onSelect={onSelect}
              onKeyDown={onListKey}
              label="Cases to review, highest risk first"
            />
            <Scoreboard score={score} onReset={review.reset} />
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
            onDecide={(v) => review.decide(selected.event_id, v)}
            onReveal={() => review.reveal(selected.event_id)}
            onBack={narrow ? () => replaceRoute("console") : undefined}
          />
        )}
        {!selected && !narrow && (
          <div className="card">
            <EmptyState icon={<InboxIcon size={22} />} title="No case selected" cause="No cases match the current filter." />
          </div>
        )}
      </div>
      <p className="console-foot muted">
        Source: {q.source.defender} on {q.source.split} (seed {q.source.seed}){SEP}{q.source.parity_checked_events.toLocaleString("en-US")} events
        checked against the published replay.
      </p>
    </>
  );
}

function Scoreboard({ score, onReset }: { score: ReturnType<typeof scoreboard>; onReset: () => void }) {
  return (
    <div className="score" role="group" aria-label="Session scoreboard">
      <div className="score-head">
        <span className="score-title">
          Session scoreboard
          <InfoTip label="the scoreboard">
            A call matches the truth when it holds back an attack (Hold or Step-up) or releases an honest payout. The swarm is scored on the
            same cases you reviewed. Everything stays in this browser.
          </InfoTip>
        </span>
        <button type="button" className="link-btn" onClick={onReset}>
          Reset
        </button>
      </div>
      <dl className="score-grid">
        <div>
          <dt>Reviewed</dt>
          <dd>{score.reviewed}</dd>
        </div>
        <div>
          <dt>You matched</dt>
          <dd>
            {score.yours}
            <span className="muted"> / {score.reviewed}</span>
          </dd>
        </div>
        <div>
          <dt>Swarm, same cases</dt>
          <dd>
            {score.swarm}
            <span className="muted"> / {score.reviewed}</span>
          </dd>
        </div>
      </dl>
    </div>
  );
}
