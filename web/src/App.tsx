import { useEffect, useRef } from "react";
import { useRoute } from "./route";
import type { View } from "./route";
import TopBar from "./components/TopBar";
import ConsoleView from "./console/ConsoleView";
import CopilotView from "./copilot/CopilotView";
import ResultsView from "./results/ResultsView";
import HowView from "./HowView";

const TITLES: Record<View, string> = {
  console: "Console",
  copilot: "Copilot",
  results: "Results",
  how: "How it works",
};

export default function App() {
  const route = useRoute();
  const mainRef = useRef<HTMLElement>(null);
  const first = useRef(true);

  useEffect(() => {
    document.title = `${TITLES[route.view]} - PayGuard`;
    // After a tab change, start at the top of the new view and move focus there (not on the first load).
    if (first.current) {
      first.current = false;
      return;
    }
    window.scrollTo({ top: 0 });
    mainRef.current?.focus({ preventScroll: true });
  }, [route.view]);

  return (
    <>
      <a
        className="skip"
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          mainRef.current?.focus();
        }}
      >
        Skip to content
      </a>
      <TopBar view={route.view} />
      <main id="main" ref={mainRef} tabIndex={-1} className="container">
        {route.view === "console" && <ConsoleView />}
        {route.view === "copilot" && <CopilotView />}
        {route.view === "results" && <ResultsView />}
        {route.view === "how" && <HowView />}
      </main>
      <footer className="foot">PayGuard Arena · PayPal AI Hackathon 2026</footer>
    </>
  );
}
