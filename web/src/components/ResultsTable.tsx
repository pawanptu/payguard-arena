import type { ResultRow, Split } from "../types";
import { num1, num2, pct, splitLabel } from "../format";

type Key = Exclude<keyof ResultRow, "defender" | "split">;

const COLS: { key: Key; label: string; fmt: (n: number) => string }[] = [
  { key: "objective", label: "Objective", fmt: num2 },
  { key: "leakage_pct", label: "Leakage", fmt: pct },
  { key: "honest_block_rate", label: "Honest blocked", fmt: pct },
  { key: "honest_unpaid_rate", label: "Honest unpaid", fmt: pct },
  { key: "honest_median_ttp_h", label: "Median time to pay (h)", fmt: num1 },
  { key: "immunity_attempts", label: "Attacks paid before contained", fmt: num1 },
  { key: "never_contained_rate", label: "Never contained", fmt: pct },
];

const SPLIT_ORDER: Split[] = ["TRAIN", "TEST_SEEN", "TEST_HELDOUT"];

export default function ResultsTable({ rows, objective }: { rows: ResultRow[]; objective: string }) {
  const splits = SPLIT_ORDER.filter((s) => rows.some((r) => r.split === s));
  // Lowest value per (split, column) is the best for every column here.
  const best = new Map<string, number>();
  for (const s of splits) {
    const inSplit = rows.filter((r) => r.split === s);
    if (inSplit.length < 2) continue;
    for (const c of COLS) best.set(`${s}|${c.key}`, Math.min(...inSplit.map((r) => r[c.key])));
  }
  const isBest = (r: ResultRow, k: Key) => best.get(`${r.split}|${k}`) === r[k];

  return (
    <section className="card" aria-labelledby="res-h">
      <h2 id="res-h">Results by defender and split</h2>
      <p className="muted small">
        Objective = <code>{objective}</code>. Lower is better in every column; the best value in each split is
        marked <span className="best-key">best</span> and shaded.
      </p>
      {rows.length === 0 ? (
        <p className="muted">This replay has no results.</p>
      ) : (
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Results table, scrolls sideways on narrow screens">
          <table>
            <caption className="visually-hidden">Defender results per split</caption>
            <thead>
              <tr>
                <th scope="col">Defender</th>
                <th scope="col">Split</th>
                {COLS.map((c) => (
                  <th scope="col" className="num" key={c.key}>{c.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {splits.flatMap((s) =>
                rows
                  .filter((r) => r.split === s)
                  .map((r) => (
                    <tr key={`${s}|${r.defender}`} className={isBest(r, "objective") ? "row-best" : ""}>
                      <th scope="row"><code>{r.defender}</code></th>
                      <td>{splitLabel(r.split)}</td>
                      {COLS.map((c) => (
                        <td key={c.key} className={`num ${isBest(r, c.key) ? "best" : ""}`}>
                          {c.fmt(r[c.key])}
                          {isBest(r, c.key) && <span className="best-tag"> best</span>}
                        </td>
                      ))}
                    </tr>
                  )),
              )}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
