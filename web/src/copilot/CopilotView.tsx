import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { fetchCopilotSessions } from "../api";
import { useAsync } from "../hooks";
import { hrefFor, replaceRoute, useRoute } from "../route";
import type { CopilotSession, CopilotTurn } from "../types";
import { plural, SEP } from "../format";
import InfoTip from "../components/InfoTip";
import { CheckIcon, CopyIcon, CrossIcon, InboxIcon } from "../components/Icons";
import { EmptyState, ErrorCard, Pill, Skeleton } from "../components/ui";

const RUN_LIVE = 'python -m scripts.copilot ask "Why was payout #24 held?"';

type Call = Extract<CopilotTurn, { role: "tool_call" }>;
type Result = Extract<CopilotTurn, { role: "tool_result" }>;

const toolCallCount = (s: CopilotSession) => s.turns.filter((t) => t.role === "tool_call").length;

function dateOf(iso: string): string {
  const m = /^(\d{4}-\d{2}-\d{2})/.exec(iso);
  return m ? m[1] : iso;
}

const ENV_LABEL: Record<string, string> = { "paypal-sandbox": "PayPal sandbox" };

/* ---------- Source badges ---------- */

function SourceBadge({ source }: { source: string }) {
  switch (source) {
    case "paypal-agent-toolkit":
      return <span className="src src-toolkit">PayPal Agent Toolkit</span>;
    case "payguard-rest":
      return (
        <span className="src-wrap">
          <span className="src src-rest">PayGuard REST (Payouts GET)</span>
          <InfoTip label="the PayGuard REST source">
            The Agent Toolkit has no Payouts tools, so payout status comes from our own read-only REST call.
          </InfoTip>
        </span>
      );
    case "payguard-console":
      return <span className="src src-console">Console data</span>;
    default:
      return <span className="src src-console">{source}</span>;
  }
}

/* ---------- Transcript ---------- */

function pretty(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2) ?? "null";
  } catch {
    return String(v);
  }
}

function ToolCard({ call, result }: { call: Call; result: Result | undefined }) {
  const hasArgs = call.args && Object.keys(call.args).length > 0;
  return (
    <article className="tool" aria-label={`Tool call ${call.step}: ${call.tool}`}>
      <header className="tool-head">
        <span className="tool-step" aria-hidden="true">{call.step}</span>
        <code className="tool-name">{call.tool}</code>
        <SourceBadge source={String(call.source)} />
        {result ? (
          <span className="tool-status">
            <Pill variant={result.ok ? "met" : "leaked"}>
              {result.ok ? <CheckIcon size={12} /> : <CrossIcon size={12} />}
              {result.ok ? "ok" : "error"}
            </Pill>
            <span className="tool-ms">{result.ms} ms</span>
          </span>
        ) : (
          <span className="tool-status muted">no result recorded</span>
        )}
      </header>
      <div className="tool-body">
        <p className="tool-label">Arguments</p>
        {hasArgs ? <pre className="code">{JSON.stringify(call.args)}</pre> : <p className="muted tool-none">No arguments</p>}
        {result && !result.ok && (
          <p className="tool-error">
            <strong>Error:</strong> {result.error ?? "unknown error"}
          </p>
        )}
        {result && result.ok && result.data !== undefined && (
          <details className="tool-result">
            <summary>Result</summary>
            <pre className="code">{pretty(result.data)}</pre>
          </details>
        )}
      </div>
    </article>
  );
}

function Transcript({ s }: { s: CopilotSession }) {
  const results = useMemo(() => {
    const m = new Map<number, Result>();
    for (const t of s.turns) if (t.role === "tool_result") m.set(t.step, t);
    return m;
  }, [s]);
  const out: ReactNode[] = [];
  s.turns.forEach((t, i) => {
    if (t.role === "user") {
      out.push(
        <div key={i} className="bubble bubble-user">
          <p className="bubble-who">You asked</p>
          <p>{t.text}</p>
        </div>,
      );
    } else if (t.role === "tool_call") {
      out.push(<ToolCard key={i} call={t} result={results.get(t.step)} />);
    } else if (t.role === "assistant") {
      out.push(
        <div key={i} className="bubble bubble-answer">
          <p className="bubble-who">Copilot answer</p>
          {t.text.split(/\n{2,}/).map((para, j) => (
            <p key={j}>{para}</p>
          ))}
        </div>,
      );
    }
  });
  return <div className="transcript">{out}</div>;
}

/* ---------- Header strip ---------- */

function InfoStrip({ s }: { s: CopilotSession }) {
  return (
    <div className="cp-strip" role="group" aria-label="Session setup">
      <dl className="cp-facts">
        <div>
          <dt>Environment</dt>
          <dd>{ENV_LABEL[s.environment] ?? s.environment}</dd>
        </div>
        <div>
          <dt>LLM</dt>
          <dd>{s.llm.model}</dd>
        </div>
        <div>
          <dt>Toolkit</dt>
          <dd>
            <code>{s.toolkit.package}</code> {s.toolkit.version}
          </dd>
        </div>
        <div>
          <dt>Access</dt>
          <dd>
            <Pill variant="met">{s.toolkit.read_only ? "Read-only" : "Read and write"}</Pill>
          </dd>
        </div>
      </dl>
      <p className="cp-redact">
        <strong>Redaction:</strong> {s.redaction}
      </p>
    </div>
  );
}

/* ---------- Run it live ---------- */

function RunLive() {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    try {
      void navigator.clipboard
        .writeText(RUN_LIVE)
        .then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1600);
        })
        .catch(() => undefined);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <section className="card runlive" aria-labelledby="live-h">
      <h2 id="live-h" className="runlive-h">Run it live</h2>
      <p className="card-sub">
        The public site only replays recorded sessions. To ask your own question against the PayPal sandbox, run the copilot locally:
      </p>
      <div className="codeblock">
        <pre className="code"><code>{RUN_LIVE}</code></pre>
        <button type="button" className="icon-btn icon-btn-sm" onClick={copy} aria-label="Copy the command" title="Copy">
          {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
        </button>
        <span className="visually-hidden" role="status">{copied ? "Copied" : ""}</span>
      </div>
    </section>
  );
}

/* ---------- View ---------- */

export default function CopilotView() {
  const state = useAsync(fetchCopilotSessions);
  const route = useRoute();

  return (
    <>
      <div className="page-head">
        <h1>Ops copilot</h1>
        <p className="page-sub">
          Recorded questions to an AI assistant built on the PayPal Agent Toolkit, answered against the PayPal sandbox. Every tool it
          called is shown with where the data came from.
        </p>
      </div>
      {state.status === "error" ? (
        <ErrorCard what="the copilot sessions" message={state.error} onRetry={() => void state.reload()} />
      ) : state.data === null ? (
        <div className="cp-grid" aria-busy="true">
          <span className="visually-hidden">Loading the recorded sessions</span>
          <div className="card"><Skeleton className="skel-title" />{[0, 1, 2].map((i) => <Skeleton key={i} className="skel-row" />)}</div>
          <div className="card"><Skeleton className="skel-title" /><Skeleton className="skel-chart" /></div>
        </div>
      ) : state.data.sessions.length === 0 ? (
        <>
          <div className="card">
            <EmptyState
              icon={<InboxIcon size={22} />}
              title="No recorded sessions yet."
              cause={`Run it live: ${RUN_LIVE.replace("Why was payout #24 held?", "...")}`}
            />
          </div>
          <RunLive />
        </>
      ) : (
        <Loaded sessions={state.data.sessions} param={route.view === "copilot" ? route.param : null} />
      )}
    </>
  );
}

function Loaded({ sessions, param }: { sessions: CopilotSession[]; param: string | null }) {
  const current = sessions.find((s) => s.id === param) ?? sessions[0];
  return (
    <>
      <InfoStrip s={current} />
      <div className="cp-grid">
        <nav className="card cp-list" aria-label="Recorded sessions">
          <h2 className="cp-list-h">Sessions</h2>
          <ul>
            {sessions.map((s) => {
              const on = s.id === current.id;
              return (
                <li key={s.id}>
                  <a
                    href={hrefFor("copilot", s.id)}
                    className={`cp-item${on ? " on" : ""}`}
                    aria-current={on ? "true" : undefined}
                    onClick={(e) => {
                      // Selecting is not navigation: no history entry per click.
                      e.preventDefault();
                      replaceRoute("copilot", s.id);
                    }}
                  >
                    <span className="cp-item-title">{s.title}</span>
                    <span className="cp-item-meta">
                      {dateOf(s.recorded_at)}{SEP}{plural(toolCallCount(s), "tool call")}
                    </span>
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>
        <section className="card cp-main" aria-labelledby="cp-title">
          <header className="card-head">
            <div className="card-head-main">
              <h2 id="cp-title">{current.title}</h2>
              <p className="card-sub">
                Recorded {dateOf(current.recorded_at)}{SEP}{plural(toolCallCount(current), "tool call")}{SEP}replayed, not live
              </p>
            </div>
          </header>
          <Transcript s={current} />
        </section>
      </div>
      <RunLive />
    </>
  );
}
