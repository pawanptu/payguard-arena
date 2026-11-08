import type { Summary } from "../types";
import { shortTime } from "../format";
import InfoTip from "./InfoTip";
import { ShieldMark, WarnIcon } from "./Icons";
import { Pill } from "./ui";
import ThemeToggle from "./ThemeToggle";

const NAV = [
  { href: "#overview", label: "Overview" },
  { href: "#replay", label: "Replay" },
  { href: "#results", label: "Results" },
  { href: "#paypal", label: "PayPal" },
];

/** Sticky block: the always-visible SIMULATED strip (28px) over the nav bar (52px). */
export default function TopBlock({ summary }: { summary: Summary | null }) {
  return (
    <div className="top-block">
      <div className="strip" role="note">
        <div className="strip-in">
          <WarnIcon size={14} />
          <span className="strip-text">
            <strong>SIMULATED</strong>
            <span className="strip-long"> replay on synthetic data. Only rows marked LIVE used the PayPal sandbox (test money).</span>
            <span className="strip-short"> · synthetic data</span>
          </span>
          {summary?.synthetic_fixture && <Pill variant="sample">Sample data</Pill>}
          {summary && (
            <InfoTip label="this replay">
              Generated {shortTime(summary.generated_at)}. Schema v{summary.schema_version}.
            </InfoTip>
          )}
        </div>
      </div>
      <div className="bar">
        <div className="bar-in">
          <a className="brand" href="#overview">
            <ShieldMark size={20} />
            <span>PayGuard Arena</span>
          </a>
          <nav className="nav" aria-label="Sections">
            {NAV.map((n) => (
              <a key={n.href} href={n.href}>{n.label}</a>
            ))}
          </nav>
          <ThemeToggle />
        </div>
      </div>
    </div>
  );
}
