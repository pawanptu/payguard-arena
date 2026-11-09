import { forwardRef, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { AsyncState } from "../hooks";
import type { CaseExplanation, ConsoleCase, ConsoleQueue, ExplanationsResult, SpecialistInfo, VerdictName } from "../types";
import { SPECIALIST_KEYS } from "../types";
import { days1, num1, plural, SEP, usd } from "../format";
import { TIPS } from "../tips";
import InfoTip from "../components/InfoTip";
import { CheckIcon, CrossIcon, DiamondIcon, InboxIcon, SquareIcon } from "../components/Icons";
import { EmptyState, Pill, Skeleton, VERDICT_WORD, VerdictPill } from "../components/ui";
import RiskMeter from "./RiskMeter";
import { callMatchesTruth } from "../review";
import { clamp01, DECIDED_BY, num, pct0, statLabel } from "./labels";

/* ---------- Why it was flagged ---------- */

function Why({ state, id }: { state: AsyncState<ExplanationsResult>; id: number }) {
  if (state.status === "loading" || (state.status === "ready" && state.data === null)) {
    return (
      <div aria-busy="true">
        <span className="visually-hidden">Loading the explanation</span>
        <Skeleton className="skel-line" style={{ width: "92%", marginTop: 0 }} />
        <Skeleton className="skel-line" style={{ width: "80%" }} />
        <Skeleton className="skel-line" style={{ width: "60%" }} />
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="inline-error" role="alert">
        <p>Couldn&apos;t load the explanation.</p>
        <button type="button" className="btn" onClick={() => void state.reload()}>Retry</button>
      </div>
    );
  }
  const res = state.data as ExplanationsResult;
  if (!res.built) {
    return (
      <EmptyState
        icon={<InboxIcon size={22} />}
        title="No AI explanation yet"
        cause="The evidence below is complete without it."
      />
    );
  }
  const e: CaseExplanation | undefined = res.data.explanations[String(id)];
  if (!e) {
    return (
      <EmptyState
        icon={<InboxIcon size={22} />}
        title="No explanation for this case yet"
        cause="The evidence below is still complete."
      />
    );
  }
  const ai = e.source === "llm";
  return (
    <div className="why">
      <div className="why-meta">
        <Pill variant="chip" className={ai ? "src-ai" : "src-template"}>
          {ai ? "AI explanation" : "Standard explanation"}
        </Pill>
        <InfoTip label={ai ? "the AI explanation" : "the standard explanation"}>
          {ai ? `Written ahead of time by ${res.data.model}. ${TIPS.grounding}` : TIPS.template}
        </InfoTip>
      </div>
      <p className="why-summary">{e.summary}</p>
      {e.evidence.length > 0 && (
        <ul className="why-evidence">
          {e.evidence.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      )}
      <p className="why-line">
        <strong>Check next:</strong> {e.check_next}
      </p>
      <p className="why-line why-action">
        <strong>Suggested action:</strong> <VerdictPill verdict={e.suggested_action} />
      </p>
    </div>
  );
}

/* ---------- How each specialist voted ---------- */

function Evidence({ c, specialists }: { c: ConsoleCase; specialists: Map<string, SpecialistInfo> }) {
  const rows = SPECIALIST_KEYS.map((k) => ({
    key: k,
    info: specialists.get(k),
    score: c.scores?.[k] ?? 0,
    weight: c.weights?.[k] ?? 0,
    share: c.vote_share?.[k] ?? 0,
  })).sort((a, b) => b.share - a.share || b.score - a.score || a.key.localeCompare(b.key));
  const layerInfo = specialists.get("layer");
  const fired = c.layer?.fired ?? [];
  return (
    <>
      <p className="spec-legend">
        What each agent saw
        <InfoTip label="the specialist columns">
          Score: {TIPS.agentScore} Trust weight: {TIPS.trustWeight} Share of the risk: {TIPS.shareOfRisk}
        </InfoTip>
      </p>
      <ul className="spec-list">
        {rows.map((r) => {
          const silent = r.score <= 0;
          return (
            <li key={r.key} className={`spec${silent ? " silent" : ""}`}>
              <div className="spec-head">
                <span className="spec-name">{r.info?.label ?? r.key}</span>
                {r.info?.what && <InfoTip label={r.info.label}>{r.info.what}</InfoTip>}
                {silent && <span className="spec-quiet">No signal</span>}
              </div>
              <dl className="spec-metrics">
                <div className="spec-m spec-score">
                  <dt>Score</dt>
                  <dd>
                    <span className="sbar" aria-hidden="true">
                      <span className="sbar-fill" style={{ width: `${clamp01(r.score) * 100}%` }} />
                    </span>
                    <span className="spec-num">{num(r.score)}</span>
                  </dd>
                </div>
                <div className="spec-m">
                  <dt>Trust weight</dt>
                  <dd className="spec-num">{num(r.weight)}</dd>
                </div>
                <div className="spec-m">
                  <dt>Share of the risk</dt>
                  <dd className="spec-num">{pct0(r.share)}</dd>
                </div>
              </dl>
            </li>
          );
        })}
      </ul>
      <div className="layer">
        <div className="layer-head">
          <span className="spec-name">
            {layerInfo?.label ?? "Anomaly layer"}
            <InfoTip label="the anomaly layer">{TIPS.anomalyLayer}</InfoTip>
          </span>
          {c.layer && <VerdictPill verdict={c.layer.verdict} />}
        </div>
        <p className="spec-what">{TIPS.anomalyWhat}</p>
        {fired.length > 0 ? (
          <ul className="chips" aria-label="Checks that fired">
            {fired.map((f) => (
              <li key={f}>
                <Pill variant="chip" title={f}>{statLabel(f)}</Pill>
              </li>
            ))}
          </ul>
        ) : (
          <p className="spec-what">No checks fired.</p>
        )}
      </div>
    </>
  );
}

/* ---------- Account facts ---------- */

function sharing(n: number, what: string, one: string): string {
  return n <= 1 ? one : `${plural(n, "account")} use the same ${what}.`;
}

function Facts({ c }: { c: ConsoleCase }) {
  const f = c.features;
  if (!f) return null;
  const facts: string[] = [
    `The account is ${num1(f.age_days)} days old and has ${plural(f.n_prior, "earlier payout")}.`,
    `In the last 24 hours it had ${plural(f.window_n, "payout")} worth ${usd(f.window_usd)} in total.`,
    `This payout is ${num(f.amount_ratio)}× the account's usual amount.`,
    sharing(f.device_workers, "device", "Only this account uses its device."),
    sharing(f.destination_workers, "payout destination", "Only this account uses its payout destination."),
    `Requester ${c.requester_id} is tied to ${plural(f.requester_workers, "account")}.`,
    `This account is tied to ${plural(f.worker_requesters, "requester")}.`,
  ];
  return (
    <ul className="facts">
      {facts.map((t) => (
        <li key={t}>{t}</li>
      ))}
    </ul>
  );
}

/* ---------- What really happened ---------- */

function Match({ ok, who, call }: { ok: boolean; who: string; call: string }) {
  return (
    <li className={`check ${ok ? "check-ok" : "check-no"}`}>
      {ok ? <CheckIcon size={14} /> : <CrossIcon size={14} />}
      <span>
        {who}: <strong>{call}</strong>
        {SEP}
        {ok ? "was right" : "was wrong"}
      </span>
    </li>
  );
}

function TruthPanel({ c, decision }: { c: ConsoleCase; decision: VerdictName | undefined }) {
  const ref = useRef<HTMLElement>(null);
  // The panel appears after a click in the action bar: bring it into view if it is scrolled out.
  useEffect(() => {
    ref.current?.scrollIntoView({ block: "nearest" });
  }, []);
  const t = c.truth;
  if (!t) return null;
  const o = c.outcome;
  let outcome: string | null = null;
  if (o) {
    const after = o.delay_h != null ? ` after ${num1(o.delay_h)} h` : "";
    if (t.is_attack) {
      outcome = o.paid
        ? `This attack payout was paid${after}. That is leakage: fraud that got through.`
        : "This attack payout was never paid: it was stopped.";
    } else {
      outcome = o.paid
        ? `This honest payout was paid${after}. The wait is the cost of the review.`
        : "This honest payout was never paid.";
    }
  }
  return (
    <section ref={ref} className="truth" aria-label="What really happened" aria-live="polite">
      <div className="truth-head">
        <h3>What really happened</h3>
      </div>
      <p className="truth-line">
        <Pill variant={t.is_attack ? "leaked" : "release"}>{t.is_attack ? "Attack" : "Honest payout"}</Pill>{" "}
        {t.is_attack ? `Attack, family ${t.family}.` : "A legitimate payout."}
      </p>
      {outcome && <p className="truth-line">{outcome}</p>}
      <ul className="checks truth-checks" aria-label="Were the calls right?">
        {decision && <Match ok={callMatchesTruth(decision, t.is_attack)} who="Your call" call={VERDICT_WORD[decision]} />}
        <Match ok={callMatchesTruth(c.verdict, t.is_attack)} who="The agents' call" call={VERDICT_WORD[c.verdict]} />
      </ul>
    </section>
  );
}

/* ---------- Reviewer actions ---------- */

const ACTIONS: { v: VerdictName; label: string; icon: ReactNode; cls: string }[] = [
  { v: "RELEASE", label: "Release", icon: <CheckIcon size={16} />, cls: "release" },
  { v: "STEP_UP", label: "Step up", icon: <DiamondIcon size={16} />, cls: "step" },
  { v: "HOLD", label: "Hold", icon: <SquareIcon size={16} />, cls: "hold" },
];

function Actions({ decision, onDecide }: { decision: VerdictName | undefined; onDecide: (v: VerdictName) => void }) {
  return (
    <div className="actions" role="group" aria-label="Your call on this payout">
      <div className="actions-row">
        {ACTIONS.map((a) => (
          <button
            key={a.v}
            type="button"
            className={`act act-${a.cls}${decision === a.v ? " on" : ""}`}
            aria-pressed={decision === a.v}
            onClick={() => onDecide(a.v)}
          >
            {a.icon}
            {a.label}
          </button>
        ))}
        <p className="actions-note">Saved in this browser only. Nothing is sent.</p>
      </div>
    </div>
  );
}

/* ---------- Detail ---------- */

const CaseDetail = forwardRef<
  HTMLHeadingElement,
  {
    c: ConsoleCase;
    queue: ConsoleQueue;
    specialists: Map<string, SpecialistInfo>;
    explanations: AsyncState<ExplanationsResult>;
    decision: VerdictName | undefined;
    revealed: boolean;
    copilotHref: string;
    narrow: boolean;
    onDecide: (v: VerdictName) => void;
    onReveal: () => void;
    onBack?: () => void;
  }
>(function CaseDetail({ c, queue, specialists, explanations, decision, revealed, copilotHref, narrow, onDecide, onReveal, onBack }, headingRef) {
  const showTruth = Boolean(decision) || revealed;
  const [specOpen, setSpecOpen] = useState(!narrow);
  return (
    <section className="card detail" aria-labelledby="case-h">
      {onBack && (
        <div className="detail-bar">
          <button type="button" className="back-btn" onClick={onBack}>
            <span aria-hidden="true">←</span> Queue
          </button>
          <span className="detail-bar-id">#{c.event_id}</span>
          <VerdictPill verdict={c.verdict} />
        </div>
      )}
      <div className="detail-body">
        <header className="detail-head">
          <div className="detail-title">
            <h2 id="case-h" ref={headingRef} tabIndex={-1}>
              Payout #{c.event_id}
            </h2>
            <p className="detail-amt">{usd(c.amount)}</p>
            <p className="card-sub">
              Worker {c.worker_id}{SEP}Requester {c.requester_id}{SEP}Day {days1(c.t)} of the replay
            </p>
          </div>
          <div className="detail-side">
            <VerdictPill verdict={c.verdict} />
            <p className="detail-by">
              Flagged by: {DECIDED_BY[c.decided_by] ?? c.decided_by}
              <InfoTip label="who flagged it">
                {TIPS.swarm} {TIPS.anomalyLayer}
              </InfoTip>
            </p>
            <div className="detail-links">
              <a className="btn btn-link" href={copilotHref}>Ask copilot about #{c.event_id}</a>
              {!showTruth && Boolean(c.truth) && (
                <button type="button" className="btn" onClick={onReveal}>
                  Show the answer
                </button>
              )}
            </div>
          </div>
        </header>

        {showTruth && <TruthPanel c={c} decision={decision} />}

        <RiskMeter risk={c.risk} stepUp={queue.thresholds.step_up} hold={queue.thresholds.hold} />

        <section className="dsec" aria-labelledby="why-h">
          <h3 id="why-h">Why it was flagged</h3>
          <Why state={explanations} id={c.event_id} />
        </section>

        <section className="dsec" aria-labelledby="facts-h">
          <h3 id="facts-h">Account facts</h3>
          <Facts c={c} />
        </section>

        <details
          className="dsec spec-details"
          open={specOpen}
          onToggle={(e) => setSpecOpen((e.currentTarget as HTMLDetailsElement).open)}
        >
          <summary>How each specialist voted</summary>
          <Evidence c={c} specialists={specialists} />
        </details>
      </div>
      <footer className="detail-foot">
        <Actions decision={decision} onDecide={onDecide} />
      </footer>
    </section>
  );
});

export default CaseDetail;
