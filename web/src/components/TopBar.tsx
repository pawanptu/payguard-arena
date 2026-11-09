import { hrefFor, VIEWS } from "../route";
import type { View } from "../route";
import { TIPS } from "../tips";
import { ShieldMark } from "./Icons";
import InfoTip from "./InfoTip";
import ThemeToggle from "./ThemeToggle";

/** Sticky top bar: brand, the four view tabs, the "Simulated data" chip and the theme toggle. */
export default function TopBar({ view }: { view: View }) {
  return (
    <header className="top-block">
      <div className="bar">
        <div className="bar-in">
          <a className="brand" href={hrefFor("console")} aria-label="PayGuard, payout fraud ops console">
            <ShieldMark size={22} />
            <span className="brand-text">
              <span className="brand-name">PayGuard</span>
              <span className="brand-sub">Payout fraud ops console</span>
            </span>
          </a>
          <nav className="tabs" aria-label="Views">
            {VIEWS.map((v) => (
              <a key={v.view} href={hrefFor(v.view)} aria-current={v.view === view ? "page" : undefined}>
                {v.label}
              </a>
            ))}
          </nav>
          <span className="sim-chip">
            Simulated data
            <InfoTip label="simulated data">{TIPS.simulated}</InfoTip>
          </span>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
