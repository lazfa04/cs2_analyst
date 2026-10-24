import { WinStep } from "./api";

interface Props {
  steps: WinStep[];
  startTick: number;
  endTick: number;
  tickRate: number;
  currentTick: number;
  onSeek: (tick: number) => void;
}

// P(T wins) over the round as a step line: the chance only changes at events.
export default function WinChart({ steps, startTick, endTick, tickRate, currentTick, onSeek }: Props) {
  const W = 720, H = 190, padL = 38, padR = 10, padT = 12, padB = 24;
  const dur = Math.max(1, (endTick - startTick) / tickRate);
  const x = (tick: number) => padL + ((tick - startTick) / tickRate / dur) * (W - padL - padR);
  const y = (p: number) => padT + (1 - p) * (H - padT - padB);

  let d = "";
  steps.forEach((s, i) => {
    const px = x(s.tick), py = y(s.p_t_win);
    if (i === 0) d += `M${px},${py}`;
    else d += ` L${px},${y(steps[i - 1].p_t_win)} L${px},${py}`;
  });
  if (steps.length) d += ` L${x(endTick)},${y(steps[steps.length - 1].p_t_win)}`;

  const click = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const frac = ((e.clientX - r.left) / r.width * W - padL) / (W - padL - padR);
    onSeek(Math.round(startTick + Math.min(1, Math.max(0, frac)) * dur * tickRate));
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="winchart" onClick={click}>
      {[0, 0.25, 0.5, 0.75, 1].map((p) => (
        <g key={p}>
          <line x1={padL} x2={W - padR} y1={y(p)} y2={y(p)} className={p === 0.5 ? "mid" : "grid"} />
          <text x={padL - 6} y={y(p) + 4} className="axis" textAnchor="end">{p * 100}%</text>
        </g>
      ))}
      <text x={padL + 4} y={padT + 10} className="axis tlabel">T wins</text>
      <text x={padL + 4} y={H - padB - 4} className="axis ctlabel">CT wins</text>
      <path d={d} className="pline" />
      {steps.slice(1).map((s) => (
        <circle key={s.step} cx={x(s.tick)} cy={y(s.p_t_win)} r={Math.min(9, 3 + Math.abs(s.swing_t_pp) / 6)}
                className={`dot ${s.actor_side === "CT" ? "ct" : "t"}`}
                onClick={(e) => { e.stopPropagation(); onSeek(s.tick); }}>
          <title>{`${s.time_s}s · ${s.event === "kill" ? `${s.actor} kills ${s.victim}` : `${s.actor} ${s.event}s`} · ${s.alive} · T ${Math.round(s.p_t_win * 100)}% (${s.swing_t_pp > 0 ? "+" : ""}${s.swing_t_pp} for T)`}</title>
        </circle>
      ))}
      <line x1={x(currentTick)} x2={x(currentTick)} y1={padT} y2={H - padB} className="cursor" />
      <text x={W - padR} y={H - 6} className="axis" textAnchor="end">{Math.round(dur)}s</text>
      <text x={padL} y={H - 6} className="axis">0s</text>
    </svg>
  );
}
