import { useRef } from "react";
import type { CSSProperties, KeyboardEvent, ReactNode } from "react";
import type { VerdictName } from "../types";
import { AlertIcon, CheckIcon, DiamondIcon, SquareIcon } from "./Icons";
import InfoTip from "./InfoTip";

/* ---------- Pill ---------- */

export type PillVariant = "release" | "step" | "hold" | "sim" | "live" | "headline" | "met" | "notmet" | "verified" | "sample" | "leaked" | "chip";

export function Pill({ variant, children, className = "", title }: { variant: PillVariant; children: ReactNode; className?: string; title?: string }) {
  return (
    <span className={`pill pill-${variant} ${className}`} title={title}>
      {children}
    </span>
  );
}

const VERDICT_VARIANT: Record<VerdictName, PillVariant> = { RELEASE: "release", STEP_UP: "step", HOLD: "hold" };
const VERDICT_ICON: Record<VerdictName, ReactNode> = {
  RELEASE: <CheckIcon size={12} />,
  STEP_UP: <DiamondIcon size={12} />,
  HOLD: <SquareIcon size={12} />,
};

/** Verdict is carried by icon shape and text as well as colour. */
export function VerdictPill({ verdict, count }: { verdict: VerdictName; count?: number }) {
  return (
    <Pill variant={VERDICT_VARIANT[verdict]} className="pill-verdict">
      {VERDICT_ICON[verdict]}
      <span>{verdict}</span>
      {count !== undefined && <span className="pill-count">{count.toLocaleString("en-US")}</span>}
    </Pill>
  );
}

export function LivePill() {
  return (
    <Pill variant="live">
      <span className="live-dot" aria-hidden="true" />
      LIVE
    </Pill>
  );
}

export function SimPill() {
  return <Pill variant="sim">SIM</Pill>;
}

/* ---------- Segmented (radiogroup with arrow keys) ---------- */

export interface SegOption<T extends string> {
  value: T;
  label: string;
}

export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const idx = Math.max(0, options.findIndex((o) => o.value === value));

  const move = (to: number) => {
    const n = (to + options.length) % options.length;
    onChange(options[n].value);
    refs.current[n]?.focus();
  };
  const onKey = (e: KeyboardEvent) => {
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        e.preventDefault();
        move(idx + 1);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        e.preventDefault();
        move(idx - 1);
        break;
      case "Home":
        e.preventDefault();
        move(0);
        break;
      case "End":
        e.preventDefault();
        move(options.length - 1);
        break;
    }
  };

  return (
    <div className="seg" role="radiogroup" aria-label={label} onKeyDown={onKey}>
      {options.map((o, i) => (
        <button
          key={o.value}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="button"
          role="radio"
          aria-checked={i === idx}
          tabIndex={i === idx ? 0 : -1}
          className={`seg-btn${i === idx ? " on" : ""}`}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ---------- Switch ---------- */

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button type="button" role="switch" aria-checked={checked} className="switch" onClick={() => onChange(!checked)}>
      <span className="switch-track" aria-hidden="true">
        <span className="switch-thumb" />
      </span>
      <span>{label}</span>
    </button>
  );
}

/* ---------- Card header ---------- */

export function CardHeader({
  id,
  title,
  tip,
  tipLabel,
  subtitle,
  actions,
}: {
  id: string;
  title: string;
  tip?: ReactNode;
  tipLabel?: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="card-head">
      <div className="card-head-main">
        <h2 id={id}>
          {title}
          {tip && <InfoTip label={tipLabel ?? title}>{tip}</InfoTip>}
        </h2>
        {subtitle && <p className="card-sub">{subtitle}</p>}
      </div>
      {actions && <div className="card-actions">{actions}</div>}
    </header>
  );
}

/* ---------- Loading, empty and error states ---------- */

export function Skeleton({ className = "", style }: { className?: string; style?: CSSProperties }) {
  return <div className={`skel ${className}`} style={style} aria-hidden="true" />;
}

export function EmptyState({ icon, title, cause }: { icon?: ReactNode; title: string; cause?: string }) {
  return (
    <div className="empty" role="status">
      {icon && <div className="empty-icon">{icon}</div>}
      <p className="empty-title">{title}</p>
      {cause && <p className="empty-cause">{cause}</p>}
    </div>
  );
}

export function ErrorCard({ what, message, onRetry }: { what: string; message: string | null; onRetry: () => void }) {
  return (
    <div className="card error-card" role="alert">
      <div className="error-row">
        <AlertIcon size={18} />
        <p className="error-title">Couldn&apos;t load {what}.</p>
        <button type="button" className="btn" onClick={onRetry}>Retry</button>
      </div>
      {message && (
        <details className="error-details">
          <summary>Details</summary>
          <pre>{message}</pre>
        </details>
      )}
    </div>
  );
}
