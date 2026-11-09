import { hrefFor, VIEWS } from "../route";
import type { View } from "../route";
import { ShieldMark, WarnIcon } from "./Icons";
import ThemeToggle from "./ThemeToggle";

/** Sticky block: the permanent replay-mode banner over the product bar and the four view tabs. */
export default function TopBar({ view }: { view: View }) {
  return (
    <header className="top-block">
      <div className="strip" role="note">
        <div className="strip-in">
          <WarnIcon size={14} />
          <span className="strip-text">
            <strong>Replay mode:</strong> simulated synthetic data. This public site never sends payouts.
          </span>
        </div>
      </div>
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
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
