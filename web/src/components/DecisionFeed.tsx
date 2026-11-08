import { memo, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { EpisodeEvent, EpisodeMeta, Split } from "../types";
import { days1, daysTotal, defenderLabel, num1, SEP, usd } from "../format";
import { AlertIcon, InboxIcon, PauseIcon, PlayIcon, RestartIcon, ShieldIcon } from "./Icons";
import { CardHeader, EmptyState, Pill, Segmented, Switch, VerdictPill } from "./ui";

const SPEEDS = [
  { value: "0", label: "6 h/s", hoursPerSec: 6 },
  { value: "1", label: "1 day/s", hoursPerSec: 24 },
  { value: "2", label: "3 days/s", hoursPerSec: 72 },
];
const TICK_MS = 100;
const VISIBLE_ROWS = 60;

const SPLIT_PREFIX: Record<string, string> = { TRAIN: "Train", TEST_SEEN: "Seen-family", TEST_HELDOUT: "Held-out", TEST_HELDOUT2: "Round-2 held-out" };

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

/** U2: the replay opens paused at the first attack of any family (undefined: start of the episode). */
function startEvent(events: EpisodeEvent[]): EpisodeEvent | undefined {
  return events.find((e) => e.is_attack);
}

function subtitle(events: EpisodeEvent[], episode: EpisodeMeta): string {
  const families = new Set(events.filter((e) => e.is_attack).map((e) => e.family));
  const prefix = SPLIT_PREFIX[episode.split as Split] ?? episode.split;
  const head = families.size === 1 ? `${prefix} ${[...families][0]} family` : `${prefix} episode`;
  return `${head}${SEP}seed ${episode.seed}${SEP}${defenderLabel(episode.defender)}${SEP}${episode.n_events.toLocaleString("en-US")} payouts`;
}

interface RowProps {
  e: EpisodeEvent;
  reveal: boolean;
  enter: boolean;
}

const FeedRow = memo(function FeedRow({ e, reveal, enter }: RowProps) {
  const attack = reveal && e.is_attack;
  const signals = e.signals ? Object.entries(e.signals).map(([k, v]) => `${k}=${num1(v)}`).join(" ") : undefined;
  return (
    <li className={`feed-row${attack ? " attack" : ""}${enter ? " enter" : ""}`} title={signals}>
      <span className="c-time" title={`Day ${days1(e.t)}`}>{days1(e.t)}d</span>
      <VerdictPill verdict={e.verdict} />
      <span className="c-amt">{usd(e.amount)}</span>
      <span className="c-who">worker {e.worker_id} / requester {e.requester_id}</span>
      <span className="c-paid">{e.paid ? "Paid" : "Never paid"}</span>
      {attack && (
        <span className="c-tag">
          {e.paid ? <AlertIcon size={12} /> : <ShieldIcon size={12} />}
          Attack{SEP}{e.family}{SEP}{e.paid ? "leaked" : "blocked"}
        </span>
      )}
    </li>
  );
});

interface Props {
  events: EpisodeEvent[];
  episode: EpisodeMeta;
}

export default function DecisionFeed({ events, episode }: Props) {
  const maxT = events.length ? events[events.length - 1].t : 0;
  const start = useMemo(() => startEvent(events), [events]);
  const [clock, setClock] = useState(() => start?.t ?? 0);
  const [playing, setPlaying] = useState(false); // U2: never autoplays, reduced motion or not
  const [pristine, setPristine] = useState(true); // true until the user presses Play or scrubs
  const [speed, setSpeed] = useState("1");
  const [reveal, setReveal] = useState(true);
  const [announce, setAnnounce] = useState("");
  const clockRef = useRef(clock);
  clockRef.current = clock;

  const hoursPerSec = SPEEDS.find((s) => s.value === speed)?.hoursPerSec ?? 24;

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const id = window.setInterval(() => {
      // Real elapsed time, so a throttled background tab still plays at the chosen speed.
      const now = performance.now();
      const next = Math.min(maxT, clockRef.current + (hoursPerSec * (now - last)) / 1000);
      last = now;
      setClock(next);
      if (next >= maxT) {
        setPlaying(false);
        setAnnounce("Replay finished");
      }
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [playing, hoursPerSec, maxT]);

  const shown = countUpTo(events, clock);

  // Cumulative counters, computed once from the replay events.
  const cum = useMemo(() => {
    const n = events.length;
    const rel = new Int32Array(n + 1);
    const step = new Int32Array(n + 1);
    const hold = new Int32Array(n + 1);
    const leaked = new Int32Array(n + 1);
    events.forEach((e, i) => {
      rel[i + 1] = rel[i] + (e.verdict === "RELEASE" ? 1 : 0);
      step[i + 1] = step[i] + (e.verdict === "STEP_UP" ? 1 : 0);
      hold[i + 1] = hold[i] + (e.verdict === "HOLD" ? 1 : 0);
      leaked[i + 1] = leaked[i] + (e.is_attack && e.paid ? 1 : 0);
    });
    return { rel, step, hold, leaked };
  }, [events]);

  const rows = useMemo(() => {
    const out: EpisodeEvent[] = [];
    for (let i = shown - 1; i >= 0 && out.length < VISIBLE_ROWS; i--) out.push(events[i]);
    return out;
  }, [events, shown]);

  const sub = useMemo(() => subtitle(events, episode), [events, episode]);

  if (events.length === 0) {
    return (
      <section className="card" aria-labelledby="feed-h">
        <CardHeader id="feed-h" title="Replay" />
        <EmptyState icon={<InboxIcon size={22} />} title="No episode events" cause="This replay has no episode events." />
      </section>
    );
  }

  const finished = clock >= maxT && maxT > 0;
  const toggle = () => {
    setPristine(false);
    if (finished) {
      setClock(0);
      setPlaying(true);
      setAnnounce("Replay restarted");
    } else {
      setPlaying(!playing);
      setAnnounce(playing ? `Paused at day ${days1(clock)}` : "Playing");
    }
  };

  const fillPct = maxT > 0 ? Math.min(100, (clock / maxT) * 100) : 0;
  const readout = `Day ${days1(clock)} / ${daysTotal(maxT)}`;
  const animate = playing && speed === "0";

  return (
    <section className="card feed-card" aria-labelledby="feed-h">
      <CardHeader
        id="feed-h"
        title="Replay"
        subtitle={sub}
        tipLabel="ground truth"
        tip="Defenders never see the attack label. It is revealed here after the fact."
        actions={<Switch checked={reveal} onChange={setReveal} label="Reveal ground truth" />}
      />

      <div className="controls">
        <button
          type="button"
          className="icon-btn icon-btn-primary"
          onClick={toggle}
          aria-label={finished ? "Replay again" : playing ? "Pause replay" : "Play replay"}
        >
          {finished ? <RestartIcon size={18} /> : playing ? <PauseIcon size={18} /> : <PlayIcon size={18} />}
        </button>
        {pristine && !playing && (
          <span className="chip-caption">{reveal && start ? `First attack${SEP}press Play` : `Paused${SEP}press Play`}</span>
        )}
        <Segmented label="Playback speed" options={SPEEDS} value={speed} onChange={setSpeed} />
      </div>

      <div className="scrub">
        <input
          type="range"
          min={0}
          max={maxT}
          step={1}
          value={clock}
          aria-label="Replay position"
          aria-valuetext={`Day ${days1(clock)} of ${daysTotal(maxT)}`}
          style={{ "--fill": `${fillPct}%` } as CSSProperties}
          onChange={(e) => {
            setPristine(false);
            setClock(Number(e.target.value));
          }}
        />
        <span className="scrub-readout" aria-hidden="true">{readout}</span>
      </div>
      <p className="visually-hidden" role="status" aria-live="polite">{announce}</p>

      <ul className="tally" aria-label="Totals so far">
        <li><VerdictPill verdict="RELEASE" count={cum.rel[shown]} /></li>
        <li><VerdictPill verdict="STEP_UP" count={cum.step[shown]} /></li>
        <li><VerdictPill verdict="HOLD" count={cum.hold[shown]} /></li>
        {reveal && (
          <li>
            <Pill variant="leaked">
              <AlertIcon size={12} />
              <span>Leaked attacks</span>
              <span className="pill-count">{cum.leaked[shown].toLocaleString("en-US")}</span>
            </Pill>
          </li>
        )}
      </ul>

      <div className="feed" tabIndex={0} role="region" aria-label="Latest decisions, newest first">
        {rows.length === 0 && <p className="muted pad">Press Play to start the replay.</p>}
        <ul className="feed-list">
          {rows.map((e, i) => (
            <FeedRow key={e.id} e={e} reveal={reveal} enter={animate && i === 0} />
          ))}
        </ul>
      </div>
    </section>
  );
}
