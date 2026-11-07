import { useEffect, useMemo, useRef, useState } from "react";
import type { EpisodeEvent, EpisodeMeta, VerdictName } from "../types";
import { day, num1, usd } from "../format";

const SPEEDS = [
  { label: "6 h/s", hoursPerSec: 6 },
  { label: "1 day/s", hoursPerSec: 24 },
  { label: "3 days/s", hoursPerSec: 72 },
];
const TICK_MS = 100;
const VISIBLE_ROWS = 60;

const GLYPH: Record<VerdictName, string> = { RELEASE: "✓", STEP_UP: "◆", HOLD: "■" };

export function VerdictBadge({ verdict }: { verdict: VerdictName }) {
  return (
    <span className={`badge v-${verdict}`}>
      <span aria-hidden="true">{GLYPH[verdict]} </span>
      {verdict}
    </span>
  );
}

/** Index of the first event with t > clock (events are sorted by time). */
function countUpTo(events: EpisodeEvent[], clock: number): number {
  let lo = 0;
  let hi = events.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (events[mid].t <= clock) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

interface Props {
  events: EpisodeEvent[];
  episode: EpisodeMeta;
}

export default function DecisionFeed({ events, episode }: Props) {
  const maxT = events.length ? events[events.length - 1].t : 0;
  const reduceMotion = useMemo(
    () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
    [],
  );
  const [clock, setClock] = useState(0);
  const [playing, setPlaying] = useState(!reduceMotion);
  const [speed, setSpeed] = useState(1);
  const [reveal, setReveal] = useState(true);
  const [announce, setAnnounce] = useState("");
  const clockRef = useRef(clock);
  clockRef.current = clock;

  useEffect(() => {
    if (!playing) return;
    const h = SPEEDS[speed].hoursPerSec;
    let last = performance.now();
    const id = window.setInterval(() => {
      // Real elapsed time, so a throttled background tab still plays at the chosen speed.
      const now = performance.now();
      const next = Math.min(maxT, clockRef.current + (h * (now - last)) / 1000);
      last = now;
      setClock(next);
      if (next >= maxT) {
        setPlaying(false);
        setAnnounce("Replay finished");
      }
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [playing, speed, maxT]);

  const shown = countUpTo(events, clock);

  // Cumulative counters, computed once from the replay events.
  const cum = useMemo(() => {
    const n = events.length;
    const rel = new Int32Array(n + 1);
    const step = new Int32Array(n + 1);
    const hold = new Int32Array(n + 1);
    const atk = new Int32Array(n + 1);
    const atkPaid = new Int32Array(n + 1);
    events.forEach((e, i) => {
      rel[i + 1] = rel[i] + (e.verdict === "RELEASE" ? 1 : 0);
      step[i + 1] = step[i] + (e.verdict === "STEP_UP" ? 1 : 0);
      hold[i + 1] = hold[i] + (e.verdict === "HOLD" ? 1 : 0);
      atk[i + 1] = atk[i] + (e.is_attack ? 1 : 0);
      atkPaid[i + 1] = atkPaid[i] + (e.is_attack && e.paid ? 1 : 0);
    });
    return { rel, step, hold, atk, atkPaid };
  }, [events]);

  const rows = useMemo(() => {
    const out: EpisodeEvent[] = [];
    for (let i = shown - 1; i >= 0 && out.length < VISIBLE_ROWS; i--) out.push(events[i]);
    return out;
  }, [events, shown]);

  const finished = clock >= maxT && maxT > 0;
  const toggle = () => {
    if (finished) {
      setClock(0);
      setPlaying(true);
      setAnnounce("Replay restarted");
    } else {
      setPlaying(!playing);
      setAnnounce(playing ? `Paused at ${day(clock)}` : "Playing");
    }
  };

  if (events.length === 0) {
    return (
      <section className="card" aria-labelledby="feed-h">
        <h2 id="feed-h">Decision feed</h2>
        <p className="muted">This replay has no episode events.</p>
      </section>
    );
  }

  return (
    <section className="card" aria-labelledby="feed-h">
      <h2 id="feed-h">Decision feed</h2>
      <p className="muted small">
        Episode replay: seed {episode.seed}, split {episode.split}, defender <code>{episode.defender}</code>,{" "}
        {episode.n_events.toLocaleString("en-US")} payout requests over {day(maxT).replace("day ", "")} days.
      </p>

      <div className="controls">
        <button type="button" className="btn primary" onClick={toggle}>
          {finished ? "Replay again" : playing ? "Pause" : "Play"}
        </button>
        <label className="inline">
          <span>Speed</span>
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s, i) => (
              <option key={s.label} value={i}>{s.label}</option>
            ))}
          </select>
        </label>
        <label className="inline check">
          <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} />
          <span>Reveal ground truth</span>
        </label>
      </div>

      <label className="scrub">
        <span>Replay position: {day(clock)} of {day(maxT).replace("day ", "")} days ({shown.toLocaleString("en-US")} events)</span>
        <input
          type="range"
          min={0}
          max={maxT}
          step={1}
          value={clock}
          onChange={(e) => setClock(Number(e.target.value))}
        />
      </label>
      <p className="visually-hidden" role="status" aria-live="polite">{announce}</p>

      <dl className="tally">
        <div><dt>Released</dt><dd>{cum.rel[shown]}</dd></div>
        <div><dt>Step-up</dt><dd>{cum.step[shown]}</dd></div>
        <div><dt>Hold</dt><dd>{cum.hold[shown]}</dd></div>
        {reveal && (
          <>
            <div><dt>Attacks so far</dt><dd>{cum.atk[shown]}</dd></div>
            <div><dt>Attacks paid out</dt><dd>{cum.atkPaid[shown]}</dd></div>
          </>
        )}
      </dl>

      <div className="feed" tabIndex={0} role="region" aria-label="Latest decisions, newest first">
        {rows.length === 0 && <p className="muted pad">Press Play to start the replay.</p>}
        <ul className="feed-list">
          {rows.map((e) => (
            <li key={e.id} className={`feed-row ${reveal && e.is_attack ? "attack" : ""}`}>
              <span className="c-time">{day(e.t)}</span>
              <VerdictBadge verdict={e.verdict} />
              <span className="c-amt">{usd(e.amount)}</span>
              <span className="c-who muted">worker {e.worker_id} / requester {e.requester_id}</span>
              <span className="c-paid">
                {e.paid ? `paid${e.paid_at != null ? ` at ${day(e.paid_at)}` : ""}` : "never paid"}
              </span>
              {reveal && e.is_attack && (
                <span className="truth">
                  ATTACK ({e.family}) - ground truth, revealed after the fact
                  {!e.paid ? "; blocked" : "; leaked"}
                </span>
              )}
              {e.signals && (
                <span className="signals muted">
                  {Object.entries(e.signals).map(([k, v]) => `${k}=${num1(v)}`).join(" ")}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
      {reveal && (
        <p className="muted small">Ground truth (attack or honest) is hidden from defenders; it is shown here after the fact.</p>
      )}
    </section>
  );
}
