import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { InfoIcon } from "./Icons";

const MAX_W = 280;
const EDGE = 8;

/**
 * A real button that toggles a tooltip bubble. Opens on hover and keyboard focus, toggles on tap,
 * closes on Esc / outside tap, stays open while the pointer is over the bubble, and is clamped to the viewport.
 */
export default function InfoTip({ label, children }: { label: string; children: ReactNode }) {
  const id = useId();
  const wrapRef = useRef<HTMLSpanElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const tipRef = useRef<HTMLSpanElement>(null);
  const timer = useRef<number | undefined>(undefined);
  const pointer = useRef<string>("key");
  const hovered = useRef(false);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ left: 0, top: 0, width: MAX_W });

  const show = () => {
    window.clearTimeout(timer.current);
    setOpen(true);
  };
  const hideSoon = () => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setOpen(false), 140);
  };

  useLayoutEffect(() => {
    if (!open || !btnRef.current || !tipRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    const width = Math.min(MAX_W, window.innerWidth - EDGE * 2);
    const left = Math.max(EDGE, Math.min(r.left + r.width / 2 - width / 2, window.innerWidth - width - EDGE));
    const h = tipRef.current.offsetHeight;
    const below = r.bottom + 6;
    const top = below + h > window.innerHeight - EDGE && r.top - h - 6 > EDGE ? r.top - h - 6 : below;
    setPos({ left, top, width });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onDown = (e: PointerEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const close = () => setOpen(false);
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    window.addEventListener("scroll", close, { passive: true, capture: true });
    window.addEventListener("resize", close);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [open]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  return (
    <span
      ref={wrapRef}
      className="tip"
      onPointerEnter={(e) => {
        pointer.current = e.pointerType;
        if (e.pointerType === "mouse") {
          hovered.current = true;
          show();
        }
      }}
      onPointerLeave={(e) => {
        if (e.pointerType === "mouse") {
          hovered.current = false;
          if (!btnRef.current?.matches(":focus-visible")) hideSoon();
        }
      }}
    >
      <button
        ref={btnRef}
        type="button"
        className="tip-btn"
        aria-label={`About ${label}`}
        aria-describedby={id}
        aria-expanded={open}
        onPointerDown={(e) => {
          pointer.current = e.pointerType || "mouse";
        }}
        onKeyDown={() => {
          pointer.current = "key";
        }}
        onFocus={(e) => {
          if (e.currentTarget.matches(":focus-visible")) show();
        }}
        onBlur={() => {
          // Close at once on blur (no two bubbles open while tabbing), unless the pointer is still over the tip.
          window.clearTimeout(timer.current);
          if (!hovered.current) setOpen(false);
        }}
        onClick={() => {
          // Mouse: hover already opened it, a click keeps it open. Touch and keyboard: toggle.
          if (pointer.current === "mouse") show();
          else setOpen((o) => !o);
        }}
      >
        <InfoIcon size={14} />
      </button>
      <span
        ref={tipRef}
        id={id}
        role="tooltip"
        className={`tip-bubble${open ? " open" : ""}`}
        style={{ left: pos.left, top: pos.top, width: pos.width }}
      >
        {children}
      </span>
    </span>
  );
}
