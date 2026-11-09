import InfoTip from "../components/InfoTip";
import { riskTip } from "../tips";
import { clamp01, num } from "./labels";

/** Risk 0-1 on a track split by the step-up and hold thresholds. Text and ticks carry the meaning, not colour alone. */
export default function RiskMeter({ risk, stepUp, hold }: { risk: number; stepUp: number; hold: number }) {
  const r = clamp01(risk);
  const s = clamp01(stepUp);
  const h = clamp01(hold);
  const zone = r >= h ? "Hold zone" : r >= s ? "Step-up zone" : "Release zone";
  return (
    <div className="meter">
      <div
        className="meter-track"
        role="img"
        aria-label={`Risk ${num(risk)} on a scale from 0 to 1, in the ${zone.toLowerCase()}. Step-up from ${num(stepUp)}, hold from ${num(hold)}.`}
      >
        <span className="meter-zone release" style={{ left: 0, width: `${s * 100}%` }} />
        <span className="meter-zone step" style={{ left: `${s * 100}%`, width: `${(h - s) * 100}%` }} />
        <span className="meter-zone hold" style={{ left: `${h * 100}%`, width: `${(1 - h) * 100}%` }} />
        <span className="meter-tick" style={{ left: `${s * 100}%` }} />
        <span className="meter-tick" style={{ left: `${h * 100}%` }} />
        <span className="meter-marker" style={{ left: `${r * 100}%` }} />
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span style={{ left: 0 }}>0</span>
        <span style={{ left: `${s * 100}%` }}>Step-up {num(stepUp)}</span>
        <span style={{ left: `${h * 100}%` }}>Hold {num(hold)}</span>
        <span style={{ left: "100%" }}>1</span>
      </div>
      <p className="meter-read">
        <strong>Risk {num(risk)}</strong> <span className="muted">· {zone}</span>
        <InfoTip label="the risk score">{riskTip(num(stepUp), num(hold))}</InfoTip>
      </p>
    </div>
  );
}
