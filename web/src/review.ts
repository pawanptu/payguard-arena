import { useCallback, useEffect, useState } from "react";
import type { ConsoleCase, VerdictName } from "./types";

/** Reviewer decisions in replay mode. They live in this browser only (localStorage); the app works without it. */
const KEY = "payguard-console-review-v1";

export interface ReviewState {
  decisions: Record<string, VerdictName>;
  revealed: number[];
}

const EMPTY: ReviewState = { decisions: {}, revealed: [] };
const VERDICTS: VerdictName[] = ["RELEASE", "STEP_UP", "HOLD"];

function read(): ReviewState {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return EMPTY;
    const v: unknown = JSON.parse(raw);
    if (!v || typeof v !== "object") return EMPTY;
    const o = v as { decisions?: unknown; revealed?: unknown };
    const decisions: Record<string, VerdictName> = {};
    if (o.decisions && typeof o.decisions === "object") {
      for (const [id, d] of Object.entries(o.decisions)) {
        if (VERDICTS.includes(d as VerdictName)) decisions[id] = d as VerdictName;
      }
    }
    const revealed = Array.isArray(o.revealed) ? o.revealed.filter((x): x is number => typeof x === "number") : [];
    return { decisions, revealed };
  } catch {
    return EMPTY;
  }
}

function write(s: ReviewState) {
  try {
    if (Object.keys(s.decisions).length === 0 && s.revealed.length === 0) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable: decisions stay in memory for this visit */
  }
}

export function useReview() {
  const [state, setState] = useState<ReviewState>(read);
  useEffect(() => write(state), [state]);

  const decide = useCallback(
    (id: number, v: VerdictName) => setState((s) => ({ ...s, decisions: { ...s.decisions, [String(id)]: v } })),
    [],
  );
  const reveal = useCallback(
    (id: number) => setState((s) => (s.revealed.includes(id) ? s : { ...s, revealed: [...s.revealed, id] })),
    [],
  );
  const reset = useCallback(() => setState(EMPTY), []);
  return { ...state, decide, reveal, reset };
}

/** A call matches the truth when it holds back an attack (Hold or Step-up) or releases an honest payout. */
export const callMatchesTruth = (call: VerdictName, isAttack: boolean) => (isAttack ? call !== "RELEASE" : call === "RELEASE");

export interface Score {
  reviewed: number;
  yours: number;
  swarm: number;
}

export function scoreboard(cases: ConsoleCase[], decisions: Record<string, VerdictName>): Score {
  let reviewed = 0;
  let yours = 0;
  let swarm = 0;
  for (const c of cases) {
    const d = decisions[String(c.event_id)];
    if (!d || !c.truth) continue;
    reviewed++;
    if (callMatchesTruth(d, c.truth.is_attack)) yours++;
    if (callMatchesTruth(c.verdict, c.truth.is_attack)) swarm++;
  }
  return { reviewed, yours, swarm };
}
