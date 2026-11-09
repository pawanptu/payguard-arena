import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { fetchCopilotSessions } from "../api";
import { useAsync } from "../hooks";
import { hrefFor, replaceRoute, useRoute } from "../route";
import type { CopilotSession, CopilotTurn } from "../types";
import { plural, SEP } from "../format";
import { TIPS } from "../tips";
import InfoTip from "../components/InfoTip";
import { CheckIcon, CopyIcon, CrossIcon, InboxIcon } from "../components/Icons";
import { CardHeader, EmptyState, ErrorCard, Pill, Skeleton } from "../components/ui";

const RUN_LIVE = 'python -m scripts.copilot ask "Why was payout #24 held?"';

type Call = Extract<CopilotTurn, { role: "tool_call" }>;
type Result = Extract<CopilotTurn, { role: "tool_result" }>;

const toolCalls = (s: CopilotSession) => s.turns.filter((t): t is Call => t.role === "tool_call");

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
          <span className="src src-rest">PayGuard lookup</span>
          <InfoTip label="PayGuard lookup">{TIPS.payguardLookup}</InfoTip>
        </span>
      );
    case "payguard-console":
      return <span className="src src-console">PayGuard data</span>;
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
              {result.ok ? "OK" : "Failed"}
            </Pill>
            <span className="tool-ms">{result.ms} ms</span>
          </span>
        ) : (
          <span className="tool-status muted">No result recorded</span>
        )}
      </header>
      <div className="tool-body">
        <p className="tool-label">Input</p>
        {hasArgs ? <pre className="code">{JSON.stringify(call.args)}</pre> : <p className="muted tool-none">No input</p>}
        {result && !result.ok && (
          <p className="tool-error">
            <strong>Error:</strong> {result.error ?? "unknown error"}
          </p>
        )}
        {result && result.ok && result.data !== undefined && (
          <details className="tool-result">
            <summary>Output</summary>
            <pre className="code">{pretty(result.data)}</pre>
          </details>
        )}
      </div>
    </article>
  );
}

function Paragraphs({ text }: { text: string }) {
  return (
    <>
      {text.split(/\n{2,}/).map((para, j) => (
        <p key={j}>{para}</p>
      ))}
    </>
  );
}

/** One chip line: how many calls, where the data came from, and that nothing could be changed. */
function Provenance({ s }: { s: CopilotSession }) {
  const calls = toolCalls(s);
  const sources = [...new Set(calls.map((c) => String(c.source)))];
  return (
    <ul className="provenance" aria-label="Where the answer came from">
      <li><Pill variant="chip">{plural(calls.length, "tool call")}</Pill></li>
      {sources.map((src) => (
        <li key={src}><SourceBadge source={src} /></li>
      ))}
      <li>
        <Pill variant="met">
          {s.toolkit.read_only ? "Read-only" : "Read and write"}
        </Pill>
        {s.toolkit.read_only && <InfoTip label="read-only">{TIPS.readOnly}</InfoTip>}
      </li>
      <li><Pill variant="chip">{ENV_LABEL[s.environment] ?? s.environment}</Pill></li>
    </ul>
  );
}

function Transcript({ s }: { s: CopilotSession }) {
  const results = useMemo(() => {
    const m = new Map<number, Result>();
    for (const t of s.turns) if (t.role === "tool_result") m.set(t.step, t);
    return m;
  }, [s]);
  const calls = toolCalls(s);
  const question = s.turns.find((t) => t.role === "user");
  const answers = s.turns.filter((t) => t.role === "assistant");
  const showQuestion = question && question.role === "user" && question.text.trim() !== s.title.trim();
  const tools: ReactNode =
    calls.length > 0 ? (
      <details className="cp-tools" open={answers.length === 0}>
        <summary>Show the {plural(calls.length, "tool call")}</summary>
        <div className="transcript">
          {calls.map((c) => (
            <ToolCard key={c.step} call={c} result={results.get(c.step)} />
          ))}
        </div>
      </details>
    ) : null;
  return (
    <div className="transcript">
      {showQuestion && question.role === "user" && (
        <div className="bubble bubble-user">
          <p className="bubble-who">Question</p>
          <p>{question.text}</p>
        </div>
      )}
      {answers.map((a, i) => (
        <div key={i} className="bubble bubble-answer">
          <p className="bubble-who">Answer</p>
          {a.role === "assistant" && <Paragraphs text={a.text} />}
        </div>
      ))}
      <Provenance s={s} />
      {tools}
      <details className="cp-setup">
        <summary>Session setup</summary>
        <dl className="cp-facts">
          <div>
            <dt>Environment</dt>
            <dd>{ENV_LABEL[s.environment] ?? s.environment}</dd>
          </div>
          <div>
            <dt>Model</dt>
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
            <dd>{s.toolkit.read_only ? "Read-only" : "Read and write"}</dd>
          </div>
        </dl>
        <p className="cp-redact">
          <strong>Redaction:</strong> {s.redaction}
        </p>
      </details>
    </div>
  );
}

/* ---------- Run it yourself ---------- */

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
    <details className="card runlive">
      <summary>Run it yourself</summary>
      <p className="card-sub">
        This site only replays recordings. To ask your own question against the PayPal sandbox, run this on your machine:
      </p>
      <div className="codeblock">
        <pre className="code"><code>{RUN_LIVE}</code></pre>
        <button type="button" className="icon-btn icon-btn-sm" onClick={copy} aria-label="Copy the command" title="Copy">
          {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
        </button>
        <span className="visually-hidden" role="status">{copied ? "Copied" : ""}</span>
      </div>
    </details>
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
          An AI assistant built on the PayPal Agent Toolkit answers ops questions from the PayPal sandbox. Every tool it used is shown.
        </p>
      </div>
      {state.status === "error" ? (
        <ErrorCard what="the copilot sessions" message={state.error} onRetry={() => void state.reload()} />
      ) : state.data === null ? (
        <div aria-busy="true">
          <span className="visually-hidden">Loading the recorded sessions</span>
          <div className="sess-row">
            {[0, 1, 2].map((i) => <Skeleton key={i} className="skel-session" />)}
          </div>
          <div className="card"><Skeleton className="skel-title" /><Skeleton className="skel-chart" /></div>
        </div>
      ) : state.data.sessions.length === 0 ? (
        <>
          <div className="card">
            <EmptyState
              icon={<InboxIcon size={22} />}
              title="No recorded sessions yet"
              cause="Sessions are recorded against the PayPal sandbox and appear here once added."
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
      <nav className="sess-row" aria-label="Recorded sessions">
        {sessions.map((s) => {
          const on = s.id === current.id;
          return (
            <a
              key={s.id}
              href={hrefFor("copilot", s.id)}
              className={`sess${on ? " on" : ""}`}
              aria-current={on ? "true" : undefined}
              onClick={(e) => {
                // Selecting is not navigation: no history entry per click.
                e.preventDefault();
                replaceRoute("copilot", s.id);
              }}
            >
              <span className="sess-title">{s.title}</span>
              <span className="sess-meta">
                {dateOf(s.recorded_at)}{SEP}{plural(toolCalls(s).length, "tool call")}
              </span>
            </a>
          );
        })}
      </nav>
      <section className="card cp-main" aria-labelledby="cp-title">
        <CardHeader
          id="cp-title"
          title={current.title}
          subtitle={`Recorded ${dateOf(current.recorded_at)}${SEP}${plural(toolCalls(current).length, "tool call")}${SEP}not live`}
        />
        <Transcript s={current} />
      </section>
      <RunLive />
    </>
  );
}
