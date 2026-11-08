import type { ReactNode } from "react";
import type { ResultRow } from "../types";
import { caveatsFor, evaluateTests, headlineFor, MARGIN } from "../verdict";
import type { TestResult } from "../verdict";
import { defenderLabel, num1, plural, SEP, signed1 } from "../format";
import InfoTip from "./InfoTip";
import { CheckIcon, CrossIcon, InboxIcon } from "./Icons";
import { EmptyState, LivePill, Pill } from "./ui";

export interface LiveStats {
  /** Live payouts in the replay. */
  n: number;
  /** Verified webhooks matching a live payout; null while webhooks are unavailable. */
  k: number | null;
}

/** Payout stats, or why they are not available yet. */
export type LiveState = LiveStats | "loading" | "error";

const SUB = "AI defenders decide RELEASE / STEP_UP / HOLD on instant gig payouts while attackers evolve.";

function Check({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <li className={`check ${ok ? "check-ok" : "check-no"}`}>
      {ok ? <CheckIcon size={14} /> : <CrossIcon size={14} />}
      <span className="visually-hidden">{ok ? "Passed: " : "Not passed: "}</span>
      <span>{children}</span>
    </li>
  );
}

/** Full-width PayPal proof strip under the result cards. */
function ProofStrip({ live }: { live: LiveState }) {
  if (typeof live === "string") {
    return (
      <div className="proof proof-muted">
        <LivePill />
        <span>{live === "loading" ? "PayPal sandbox: checking payouts…" : "PayPal sandbox: payouts unavailable"}</span>
      </div>
    );
  }
  if (live.n === 0) {
    return (
      <div className="proof proof-muted">
        <LivePill />
        <span>PayPal sandbox: no live payouts in this replay yet</span>
      </div>
    );
  }
  return (
    <a className="proof" href="#paypal">
      <LivePill />
      <span>
        PayPal sandbox: {plural(live.n, "live payout")}
        {live.k !== null && <>{SEP}{plural(live.k, "signature-verified webhook")}</>}
        {SEP}See receipts <span aria-hidden="true">→</span>
      </span>
    </a>
  );
}

function ResultCard({ r, rows }: { r: TestResult; rows: ResultRow[] }) {
  const { test, v } = r;
  const singleName = v.single.defender.replace(/^single_/, "");
  const caveats = caveatsFor(rows, r);
  const titleId = `test-${test.split}`;
  return (
    <section className="test-card" aria-labelledby={titleId}>
      <div className="test-head">
        <h2 id={titleId}>
          {test.title}
          {test.decisive && (
            <InfoTip label="how this was tested">
              Rule fixed before this run: beat static rules and the best single defender by at least 3.0 objective
              points, without blocking more honest workers than static rules. Lower objective is better.
            </InfoTip>
          )}
        </h2>
      </div>
      <div className="test-pill">
        <Pill variant={v.met ? "met" : "notmet"} className="pill-lg">
          {v.met ? <CheckIcon size={16} /> : <CrossIcon size={16} />}
          {test.pillLabel}: {v.met ? "Met" : "Not met"}
        </Pill>
      </div>
      <div className="test-num">
        <span className="test-value">{num1(v.primary.objective)}</span>
        <span className="test-unit">objective{SEP}{defenderLabel(v.primary.defender)}</span>
      </div>
      <div className="test-meta">
        <p>Lower is better{SEP}static rules {num1(v.static.objective)}</p>
        <p>
          Attack leakage {num1(v.primary.leakage_pct)}% (static rules {num1(v.static.leakage_pct)}%)
        </p>
      </div>
      {/* One wrapper = the card's 5th subgrid row; a 6th child would land on top of the checks. */}
      <div className="test-foot">
        <ul className="checks" aria-label={`${test.title}: checks`}>
          <Check ok={v.passStatic}>
            vs static rules: {signed1(v.mStatic)} (needs +{MARGIN.toFixed(1)})
          </Check>
          <Check ok={v.passSingle}>
            vs best single on TRAIN ({singleName}): {signed1(v.mSingle)} (needs +{MARGIN.toFixed(1)})
          </Check>
          <Check ok={v.passHonest}>
            Honest blocked {v.primary.honest_block_rate.toFixed(1)}% ≤ static {v.static.honest_block_rate.toFixed(1)}%
          </Check>
        </ul>
        {caveats.length > 0 && (
          <ul className="test-caveats" aria-label={`${test.title}: caveats`}>
            {caveats.map((c) => (
              <li key={c} className="test-caveat">
                {c}
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

export default function Hero({ results, live }: { results: ResultRow[]; live: LiveState }) {
  const tests = evaluateTests(results);

  if (tests.length === 0) {
    return (
      <div className="hero">
        <h1 className="visually-hidden">PayGuard Arena</h1>
        <p className="hero-sub">{SUB}</p>
        <div className="card hero-empty">
          <EmptyState
            icon={<InboxIcon size={22} />}
            title="Held-out results are not in this replay yet"
            cause="This replay has no held-out rows for the swarm, static rules and best single defender."
          />
        </div>
      </div>
    );
  }

  return (
    <div className="hero">
      <h1>{headlineFor(tests)}</h1>
      <p className="hero-sub">{SUB}</p>

      <div className="test-cards">
        {tests.map((t) => (
          <ResultCard key={t.test.split} r={t} rows={results} />
        ))}
      </div>

      <ProofStrip live={live} />
    </div>
  );
}
