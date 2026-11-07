export default function ModeToggle() {
  return (
    <div className="seg" role="group" aria-label="Data mode">
      <button type="button" className="seg-btn on" aria-pressed="true">Replay</button>
      <button type="button" className="seg-btn" disabled aria-pressed="false" title="Live mode is coming soon">
        Live <span className="soon">(coming soon)</span>
      </button>
    </div>
  );
}
