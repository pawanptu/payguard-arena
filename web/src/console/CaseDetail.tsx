import { forwardRef } from "react";
import type { ReactNode } from "react";
import type { AsyncState } from "../hooks";
import type { CaseExplanation, ConsoleCase, ConsoleQueue, ExplanationsResult, SpecialistInfo, VerdictName } from "../types";
import { SPECIALIST_KEYS } from "../types";
import { num1, plural, SEP, usd } from "../format";
import InfoTip from "../components/InfoTip";
import { CheckIcon, CrossIcon, DiamondIcon, InboxIcon, SquareIcon } from "../components/Icons";
import { EmptyState, Pill, Skeleton, VerdictPill } from "../components/ui";
import RiskMeter from "./RiskMeter";
import { callMatchesTruth } from "../review";
import { clamp01, DECIDED_BY, num, pct0, statLabel } from "./labels";

const VERDICT_WORD: Record<VerdictName, string> = { RELEASE: "Release", STEP_UP: "Step-up", HOLD: "Hold" };

/* ---------- Why it was held ---------- */

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
        <p>Couldn&apos;t load the AI explanations.</p>
        <button type="button" className="btn" onClick={() => void state.reload()}>Retry</button>
      </div>
    );
  }
  const res = state.data as ExplanationsResult;
  if (!res.built) {
    return (
      <EmptyState
        icon={<InboxIcon size={22} />}
        title="AI explanations are not built yet"
        cause="The evidence below is complete without them."
      />
    );
  }
  const e: CaseExplanation | undefined = res.data.explanations[String(id)];
  if (!e) {
    return <EmptyState icon={<InboxIcon size={22} />} title="No explanation for this case" cause="The evidence below still applies." />;
  }
  const ai = e.source === "llm";
  return (
    <div className="why">
      <div className="why-meta">
        <Pill variant="chip" className={ai ? "src-ai" : "src-template"}>
          {ai ? `AI explanation${SEP}${res.data.model}${SEP}cached` : "Template (AI text failed the grounding check)"}
        </Pill>
        <InfoTip label="how explanations are checked">{res.data.grounding}</InfoTip>
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

/* ---------- Evidence by specialist ---------- */

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
      <ul className="spec-list">
        {rows.map((r) => {
          const silent = r.score <= 0;
          return (
            <li key={r.key} className={`spec${silent ? " silent" : ""}`}>
              <div className="spec-head">
                <span className="spec-name">{r.info?.label ?? r.key}</span>
                {silent && <span className="spec-quiet">Silent</span>}
              </div>
              <p className="spec-what">{r.info?.what}</p>
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
                  <dt>Reputation weight</dt>
                  <dd className="spec-num">{num(r.weight)}</dd>
                </div>
                <div className="spec-m">
                  <dt>Share of the vote</dt>
                  <dd className="spec-num">{pct0(r.share)}</dd>
                </div>
              </dl>
            </li>
          );
        })}
      </ul>
      <div className="layer">
        <div className="layer-head">
          <span className="spec-name">{layerInfo?.label ?? "Anomaly layer"}</span>
          {c.layer && <VerdictPill verdict={c.layer.verdict} />}
        </div>
        {layerInfo && <p className="spec-what">{layerInfo.what}</p>}
        {fired.length > 0 ? (
          <ul className="chips" aria-label="Statistics that fired">
            {fired.map((f) => (
              <li key={f}>
                <Pill variant="chip" title={f}>{statLabel(f)}</Pill>
              </li>
            ))}
          </ul>
        ) : (
          <p className="spec-what">No statistic crossed its cut-off.</p>
        )}
      </div>
    </>
  );
}

/* ---------- Facts at decision time ---------- */

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

/* ---------- Truth and outcome (replay mode) ---------- */

function Match({ ok, who, call }: { ok: boolean; who: string; call: string }) {
  return (
    <li className={`check ${ok ? "check-ok" : "check-no"}`}>
      {ok ? <CheckIcon size={14} /> : <CrossIcon size={14} />}
      <span>
        {who}: <strong>{call}</strong>
        {SEP}
        {ok ? "matched the truth" : "did not match the truth"}
      </span>
    </li>
  );
}

function TruthPanel({ c, decision }: { c: ConsoleCase; decision: VerdictName | undefined }) {
  const t = c.truth;
  if (!t) return null;
  const o = c.outcome;
  let outcome: string | null = null;
  if (o) {
    const after = o.delay_h != null ? ` after ${num1(o.delay_h)} h` : "";
    if (t.is_attack) {
      outcome = o.paid
        ? `In the simulation this attack payout was paid${after}. That counts as leakage.`
        : "In the simulation this attack payout was never paid: it was stopped.";
    } else {
      outcome = o.paid
        ? `In the simulation this honest payout was paid${after}. The wait is the cost of the review.`
        : "In the simulation this honest payout was never paid.";
    }
  }
  return (
    <section className="truth" aria-label="What the simulator says" aria-live="polite">
      <div className="truth-head">
        <h3>What the simulator says</h3>
        <Pill variant="sim">SIM</Pill>
      </div>
      <p className="truth-line">
        <Pill variant={t.is_attack ? "leaked" : "release"}>{t.is_attack ? "Attack" : "Honest payout"}</Pill>{" "}
        {t.is_attack ? `Ground truth: attack, family ${t.family}.` : "Ground truth: a legitimate payout."}
      </p>
      {outcome && <p className="truth-line">{outcome}</p>}
      <ul className="checks truth-checks" aria-label="Did the calls match the truth?">
        {decision && <Match ok={callMatchesTruth(decision, t.is_attack)} who="Your call" call={VERDICT_WORD[decision]} />}
        <Match ok={callMatchesTruth(c.verdict, t.is_attack)} who="The swarm's call" call={VERDICT_WORD[c.verdict]} />
      </ul>
      <p className="truth-note">In replay mode nothing is sent.</p>
    </section>
  );
}

/* ---------- Reviewer actions ---------- */

const ACTIONS: { v: VerdictName; label: string; icon: ReactNode; cls: string }[] = [
  { v: "RELEASE", label: "Release", icon: <CheckIcon size={16} />, cls: "release" },
  { v: "STEP_UP", label: "Step-up", icon: <DiamondIcon size={16} />, cls: "step" },
  { v: "HOLD", label: "Keep on hold", icon: <SquareIcon size={16} />, cls: "hold" },
];

function Actions({
  decision,
  showReveal,
  onDecide,
  onReveal,
}: {
  decision: VerdictName | undefined;
  showReveal: boolean;
  onDecide: (v: VerdictName) => void;
  onReveal: () => void;
}) {
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
        {showReveal && (
          <button type="button" className="link-btn" onClick={onReveal}>
            Reveal truth
          </button>
        )}
      </div>
      <p className="actions-note">In replay mode a click is saved in this browser only. Nothing is sent.</p>
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
    onDecide: (v: VerdictName) => void;
    onReveal: () => void;
    onBack?: () => void;
  }
>(function CaseDetail({ c, queue, specialists, explanations, decision, revealed, copilotHref, onDecide, onReveal, onBack }, headingRef) {
  const showTruth = Boolean(decision) || revealed;
  return (
    <section className="card detail" aria-labelledby="case-h">
      <div className="detail-body">
        {onBack && (
          <button type="button" className="link-btn back" onClick={onBack}>
            ← Back to the queue
          </button>
        )}
        <header className="detail-head">
          <div className="detail-title">
            <h2 id="case-h" ref={headingRef} tabIndex={-1}>
              Payout #{c.event_id}
            </h2>
            <p className="detail-amt">{usd(c.amount)}</p>
            <p className="card-sub">
              Worker {c.worker_id}{SEP}Requester {c.requester_id}{SEP}arrived at simulated hour {num1(c.t)}
            </p>
          </div>
          <div className="detail-side">
            <VerdictPill verdict={c.verdict} />
            <p className="detail-by">
              Decided by: {DECIDED_BY[c.decided_by] ?? c.decided_by}
              <InfoTip label="who decided">
                The swarm votes from four specialists. The anomaly layer can only raise the severity of that verdict, never lower it.
              </InfoTip>
            </p>
            <a className="btn btn-link" href={copilotHref}>Ask the copilot</a>
          </div>
        </header>

        {showTruth && <TruthPanel c={c} decision={decision} />}

        <RiskMeter risk={c.risk} stepUp={queue.thresholds.step_up} hold={queue.thresholds.hold} />

        <section className="dsec" aria-labelledby="why-h">
          <h3 id="why-h">Why it was held</h3>
          <Why state={explanations} id={c.event_id} />
        </section>

        <section className="dsec" aria-labelledby="ev-h">
          <h3 id="ev-h">Evidence by specialist</h3>
          <Evidence c={c} specialists={specialists} />
        </section>

        <section className="dsec" aria-labelledby="facts-h">
          <h3 id="facts-h">Facts at decision time</h3>
          <Facts c={c} />
        </section>
      </div>
      <footer className="detail-foot">
        <Actions decision={decision} showReveal={!showTruth && Boolean(c.truth)} onDecide={onDecide} onReveal={onReveal} />
      </footer>
    </section>
  );
});

export default CaseDetail;
