import { useEffect, useRef } from "react";
import type { KeyboardEvent } from "react";
import type { ConsoleCase, SpecialistInfo, VerdictName } from "../types";
import { usd } from "../format";
import { EmptyState, VerdictPill } from "../components/ui";
import { InboxIcon } from "../components/Icons";
import { clamp01, num, topSpecialistKey } from "./labels";

const RISK_CLASS: Record<VerdictName, string> = { HOLD: "hold", STEP_UP: "step", RELEASE: "release" };

export const rowDomId = (id: number) => `case-opt-${id}`;

/** A listbox of cases. Focus stays on the list; the active row is announced through aria-activedescendant. */
export default function QueueList({
  cases,
  activeId,
  decided,
  specialists,
  onSelect,
  onKeyDown,
  onClear,
  label,
}: {
  cases: ConsoleCase[];
  activeId: number | null;
  decided: ReadonlySet<number>;
  specialists: Map<string, SpecialistInfo>;
  onSelect: (id: number) => void;
  onKeyDown: (e: KeyboardEvent<HTMLUListElement>) => void;
  onClear: () => void;
  label: string;
}) {
  const listRef = useRef<HTMLUListElement>(null);
  const first = useRef(true);

  // Keep the active row visible (keyboard moves, deep links, filter changes). The desktop list scrolls inside its
  // pane; below 1024 px the page scrolls instead.
  useEffect(() => {
    if (activeId === null) return;
    const el = document.getElementById(rowDomId(activeId));
    const box = listRef.current;
    const initial = first.current;
    first.current = false;
    if (!el || !box) return;
    if (box.scrollHeight > box.clientHeight + 1) {
      const top = el.offsetTop;
      const bottom = top + el.offsetHeight;
      if (top < box.scrollTop) box.scrollTop = top;
      else if (bottom > box.scrollTop + box.clientHeight) box.scrollTop = bottom - box.clientHeight;
    } else if (!initial) {
      el.scrollIntoView({ block: "nearest" });
    }
  }, [activeId, cases]);

  if (cases.length === 0) {
    return (
      <div className="qempty">
        <EmptyState icon={<InboxIcon size={22} />} title="Nothing to show" cause="Change the filter or clear the search." />
        <button type="button" className="btn" onClick={onClear}>Clear filter</button>
      </div>
    );
  }

  return (
    <ul
      ref={listRef}
      className="qlist"
      role="listbox"
      tabIndex={0}
      aria-label={label}
      aria-activedescendant={activeId !== null ? rowDomId(activeId) : undefined}
      onKeyDown={onKeyDown}
    >
      {cases.map((c) => {
        const active = c.event_id === activeId;
        const done = decided.has(c.event_id);
        const top = specialists.get(topSpecialistKey(c));
        return (
          <li
            key={c.event_id}
            id={rowDomId(c.event_id)}
            role="option"
            aria-selected={active}
            className={`qrow${active ? " active" : ""}`}
            title={top ? `Main signal: ${top.label}` : undefined}
            onClick={() => {
              listRef.current?.focus({ preventScroll: true });
              onSelect(c.event_id);
            }}
          >
            <span className="qrow-id">#{c.event_id}</span>
            <span className="qrow-amt">{usd(c.amount)}</span>
            <span className="qrisk" title={`Risk ${num(c.risk)}`}>
              <span className="qrisk-bar" aria-hidden="true">
                <span className={`qrisk-fill ${RISK_CLASS[c.verdict]}`} style={{ width: `${clamp01(c.risk) * 100}%` }} />
              </span>
              <span className="qrisk-num">{num(c.risk)}</span>
            </span>
            <span className="qrow-end">
              {done && (
                <span className="qdot" title="You decided this case">
                  <span className="visually-hidden">Reviewed</span>
                </span>
              )}
              <VerdictPill verdict={c.verdict} />
            </span>
          </li>
        );
      })}
    </ul>
  );
}
