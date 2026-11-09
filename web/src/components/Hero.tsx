import type { ReactNode } from "react";
import type { ResultRow } from "../types";
import { caveatsFor, evaluateTests, MARGIN } from "../verdict";
import type { TestResult } from "../verdict";
import { defenderLabel, num1, SEP, signed1 } from "../format";
import InfoTip from "./InfoTip";
import { CheckIcon, CrossIcon, InboxIcon } from "./Icons";
import { EmptyState, Pill } from "./ui";

function Check({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <li className={`check ${ok ? "check-ok" : "check-no"}`}>
      {ok ? <CheckIcon size={14} /> : <CrossIcon size={14} />}
      <span className="visually-hidden">{ok ? "Passed: " : "Not passed: "}</span>
      <span>{children}</span>
    </li>
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
        <h3 id={titleId}>{test.title}</h3>
        {test.decisive && (
          <InfoTip label="how this was tested">
            Rule fixed before this run: beat static rules and the best single agent by at least {MARGIN.toFixed(1)} score points, without
            holding more honest payouts than static rules. Lower score is better.
          </InfoTip>
        )}
      </div>
      <div className="test-pill">
        <Pill variant={v.met ? "met" : "notmet"} className="pill-lg">
          {v.met ? <CheckIcon size={16} /> : <CrossIcon size={16} />}
          {test.pillLabel}: {v.met ? "Met" : "Not met"}
        </Pill>
      </div>
      <div className="test-num">
        <span className="test-value">{num1(v.primary.objective)}</span>
        <span className="test-unit">Score{SEP}{defenderLabel(v.primary.defender)}</span>
      </div>
      <div className="test-meta">
        <p>Lower is better{SEP}static rules {num1(v.static.objective)}</p>
        <p>
          Fraud let through {num1(v.primary.leakage_pct)}% (static rules {num1(v.static.leakage_pct)}%)
        </p>
      </div>
      {/* One wrapper = the card's 5th subgrid row; a 6th child would land on top of the checks. */}
      <div className="test-foot">
        <ul className="checks" aria-label={`${test.title}: checks`}>
          <Check ok={v.passStatic}>
            Margin over static rules: {signed1(v.mStatic)} points (needs +{MARGIN.toFixed(1)})
          </Check>
          <Check ok={v.passSingle}>
            Margin over best single agent, chosen on training data ({singleName}): {signed1(v.mSingle)} points (needs +{MARGIN.toFixed(1)})
          </Check>
          <Check ok={v.passHonest}>
            Honest payouts held {v.primary.honest_block_rate.toFixed(1)}% against static rules {v.static.honest_block_rate.toFixed(1)}% (must not be higher)
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

/** The two original held-out test cards (they repeat rounds 1 and 2 on the Results tab). */
export default function TestCards({ results }: { results: ResultRow[] }) {
  const tests = evaluateTests(results);

  if (tests.length === 0) {
    return (
      <div className="card hero-empty">
        <EmptyState
          icon={<InboxIcon size={22} />}
          title="Test results are not available yet"
          cause="The replay has no rows for the swarm, static rules and best single agent."
        />
      </div>
    );
  }

  return (
    <div className="test-cards">
      {tests.map((t) => (
        <ResultCard key={t.test.split} r={t} rows={results} />
      ))}
    </div>
  );
}
