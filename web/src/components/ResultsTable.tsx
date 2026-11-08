import { useMemo, useState } from "react";
import type { ResultRow, Split } from "../types";
import { defenderLabel, num1, num2, ordinal, pct, SEP, SPLIT_LABEL, SPLIT_ORDER } from "../format";
import { pickBestSingle, PRIMARY_DEFENDER, SINGLE_PREFIX, TESTS } from "../verdict";
import InfoTip from "./InfoTip";
import { InboxIcon, StarIcon } from "./Icons";
import { CardHeader, EmptyState, Pill, Segmented, Switch } from "./ui";

type Key = Exclude<keyof ResultRow, "defender" | "split">;

interface Col {
  key: Key;
  label: string;
  tip: string;
  fmt: (n: number) => string;
}

const MAIN_COLS: Col[] = [
  { key: "objective", label: "Objective", tip: "Leakage + 0.5 × honest blocked + honest unpaid. Lower is better.", fmt: num2 },
  { key: "leakage_pct", label: "Leakage %", tip: "Share of attack value paid out.", fmt: pct },
  { key: "honest_block_rate", label: "Honest blocked %", tip: "Honest workers put on HOLD.", fmt: pct },
  { key: "honest_unpaid_rate", label: "Honest unpaid %", tip: "Honest workers never paid.", fmt: pct },
];
const MORE_COLS: Col[] = [
  { key: "honest_median_ttp_h", label: "Median time to pay (h)", tip: "Hours until an honest worker is paid.", fmt: num1 },
  {
    key: "immunity_attempts",
    label: "Attack payouts before containment",
    tip: "Attack payouts that slipped through before the pattern was caught.",
    fmt: num1,
  },
  { key: "never_contained_rate", label: "Never contained %", tip: "Attack rings never stopped.", fmt: pct },
];

/** swarm_r2 was designed after the round-1 result on TEST_HELDOUT was known, so its row there is not a clean test. */
const POST_HOC: Partial<Record<Split, string>> = { TEST_HELDOUT: "swarm_r2" };

/** The defender pinned as "Primary" for a split: the one under test there, else the round-1 primary. */
const primaryFor = (split: Split) => TESTS.find((t) => t.split === split)?.primary ?? PRIMARY_DEFENDER;

interface Note {
  text: string;
  tip: string;
  tipLabel: string;
}

/** A muted marker next to a defender name: why a row is not a clean test, or what a single_* row is (D-038). */
function noteFor(r: ResultRow, ruleSingle?: ResultRow, splitBest?: ResultRow): Note | null {
  if (POST_HOC[r.split] === r.defender)
    return { text: "Post-hoc", tipLabel: "why this row is not a clean test", tip: "Designed after the round-1 result on this split was known." };
  if (r === ruleSingle)
    return { text: "Chosen on TRAIN", tipLabel: "which single the rule uses", tip: "The best single specialist on TRAIN. The pre-registered rule compares against this one." };
  if (r === splitBest)
    return { text: "Not pre-registered", tipLabel: "why this single is shown", tip: "The best single specialist on this split, shown for comparison. The rule's single was fixed on TRAIN before the test." };
  return null;
}

export default function ResultsTable({ rows }: { rows: ResultRow[] }) {
  const available = useMemo(() => SPLIT_ORDER.filter((s) => rows.some((r) => r.split === s)), [rows]);
  const [picked, setPicked] = useState<Split | null>(null);
  const [more, setMore] = useState(false);

  if (rows.length === 0) {
    return (
      <section id="results" className="card" aria-labelledby="res-h">
        <CardHeader id="res-h" title="Results" />
        <EmptyState icon={<InboxIcon size={22} />} title="No results" cause="This replay has no result rows." />
      </section>
    );
  }

  // SPLIT_ORDER lists the newest held-out split first, so it is the default whenever the replay has rows for it.
  const split: Split = picked && available.includes(picked) ? picked : available[0];
  const inSplit = rows.filter((r) => r.split === split);
  const primary = inSplit.find((r) => r.defender === primaryFor(split));
  const singles = inSplit.filter((r) => r.defender.startsWith(SINGLE_PREFIX));
  const ruleSingle = pickBestSingle(rows, singles);
  const splitBest = singles.length ? singles.reduce((a, b) => (b.objective < a.objective ? b : a)) : undefined;
  const ordered = [
    ...(primary ? [primary] : []),
    ...inSplit.filter((r) => r !== primary).sort((a, b) => a.objective - b.objective),
  ];
  // Only the Objective column is marked, on its minimum value(s) within the split.
  const bestObjective = inSplit.length > 1 ? Math.min(...inSplit.map((r) => r.objective)) : null;
  const rank = primary ? inSplit.filter((r) => r.objective < primary.objective).length + 1 : null;
  const subtitle =
    `${SPLIT_LABEL[split]}${SEP}${ordered.length} defenders` +
    (primary && rank ? `${SEP}${defenderLabel(primary.defender)} ranks ${ordinal(rank)}` : "");
  const cols = more ? [...MAIN_COLS, ...MORE_COLS] : MAIN_COLS;

  return (
    <section id="results" className="card" aria-labelledby="res-h">
      <CardHeader
        id="res-h"
        title="Results"
        subtitle={subtitle}
        actions={
          <>
            <Segmented
              label="Data split"
              options={available.map((s) => ({ value: s, label: SPLIT_LABEL[s] }))}
              value={split}
              onChange={setPicked}
            />
            <Switch checked={more} onChange={setMore} label="More metrics" />
          </>
        }
      />
      <div className="table-wrap results-wrap" tabIndex={0} role="region" aria-label={`Results table, ${SPLIT_LABEL[split]} split`}>
        <table className="results">
          <caption className="visually-hidden">Defender results, {SPLIT_LABEL[split]} split. Lower is better.</caption>
          <thead>
            <tr>
              <th scope="col">Defender</th>
              {cols.map((c) => (
                <th scope="col" className="num" key={c.key}>
                  <span className="th-in">
                    {c.label}
                    <InfoTip label={c.label}>{c.tip}</InfoTip>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ordered.map((r) => {
              const isPrimary = r === primary;
              const note = noteFor(r, ruleSingle, splitBest);
              return (
                <tr key={`${r.split}|${r.defender}`} className={isPrimary ? "row-primary" : ""}>
                  <th scope="row" title={r.defender}>
                    <span className="def">
                      {defenderLabel(r.defender)}
                      {isPrimary && <Pill variant="headline">Primary</Pill>}
                      {note && (
                        <span className="def-note">
                          {note.text}
                          <InfoTip label={note.tipLabel}>{note.tip}</InfoTip>
                        </span>
                      )}
                    </span>
                  </th>
                  {cols.map((c) => {
                    const isBest = c.key === "objective" && bestObjective !== null && r.objective === bestObjective;
                    return (
                      <td key={c.key} className={`num${isBest ? " best" : ""}`}>
                        {isBest && (
                          <>
                            <StarIcon size={12} />
                            <span className="visually-hidden">best </span>
                          </>
                        )}
                        {c.fmt(r[c.key])}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
