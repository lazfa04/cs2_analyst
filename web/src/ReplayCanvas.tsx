import { useEffect, useRef, useState } from "react";
import { Frame, Replay } from "./api";

const COLORS = { CT: "#5aa9ff", T: "#f2a33a" } as const;
const SMOKE_RADIUS = 144;   // world units
const FIRE_RADIUS = 110;

interface Props {
  replay: Replay;
  tick: number;
  showNames: boolean;
}

function frameAt(ticks: number[], tick: number): [number, number] {
  // index of the last frame at or before `tick`, and how far we are towards the next one
  let lo = 0, hi = ticks.length - 1;
  if (tick <= ticks[0]) return [0, 0];
  if (tick >= ticks[hi]) return [hi, 0];
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (ticks[mid] <= tick) lo = mid; else hi = mid;
  }
  return [lo, (tick - ticks[lo]) / (ticks[lo + 1] - ticks[lo])];
}

function lerpAngle(a: number, b: number, t: number) {
  const d = ((b - a + 540) % 360) - 180;
  return a + d * t;
}

export default function ReplayCanvas({ replay, tick, showNames }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    const im = new Image();
    im.onload = () => setImg(im);
    im.src = replay.radar_url;
  }, [replay.radar_url]);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const size = c.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== size * dpr) { c.width = size * dpr; c.height = size * dpr; }
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr * size / 1024, 0, 0, dpr * size / 1024, 0, 0);
    ctx.clearRect(0, 0, 1024, 1024);
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, 1024, 1024);
    if (img) { ctx.globalAlpha = 0.85; ctx.drawImage(img, 0, 0, 1024, 1024); ctx.globalAlpha = 1; }
    const k = 1 / replay.units_per_px;

    for (const s of replay.smokes) if (tick >= s.start && tick <= s.end) {
      ctx.fillStyle = "rgba(200,205,215,0.55)";
      ctx.beginPath(); ctx.arc(s.xy[0], s.xy[1], SMOKE_RADIUS * k, 0, Math.PI * 2); ctx.fill();
    }
    for (const f of replay.fires) if (tick >= f.start && tick <= f.end) {
      ctx.fillStyle = "rgba(255,110,40,0.45)";
      ctx.beginPath(); ctx.arc(f.xy[0], f.xy[1], FIRE_RADIUS * k, 0, Math.PI * 2); ctx.fill();
    }
    const plant = replay.bomb.find((b) => b.event === "finish_plant" && b.tick <= tick && b.xy);
    if (plant?.xy) {
      ctx.fillStyle = "#ff3b3b";
      ctx.fillRect(plant.xy[0] - 7, plant.xy[1] - 7, 14, 14);
      ctx.font = "bold 16px sans-serif"; ctx.fillText("C4", plant.xy[0] + 10, plant.xy[1] + 6);
    }
    // deaths so far, and a tracer for kills in the last second
    for (const kl of replay.kills) {
      if (kl.tick > tick) continue;
      const [vx, vy] = kl.victim_xy;
      ctx.strokeStyle = COLORS[kl.victim_side] ?? "#aaa"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(vx - 7, vy - 7); ctx.lineTo(vx + 7, vy + 7); ctx.moveTo(vx + 7, vy - 7); ctx.lineTo(vx - 7, vy + 7); ctx.stroke();
      const [ax, ay] = kl.attacker_xy;
      if (ax !== null && ay !== null && tick - kl.tick < replay.tick_rate) {
        ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(vx, vy); ctx.stroke();
      }
    }

    const [i, t] = frameAt(replay.ticks, tick);
    for (const p of replay.players) {
      const a: Frame = p.frames[i];
      if (!a) continue;
      const b: Frame = p.frames[i + 1] ?? a;
      const x = b ? a[0] + (b[0] - a[0]) * t : a[0];
      const y = b ? a[1] + (b[1] - a[1]) * t : a[1];
      const yaw = (b ? lerpAngle(a[2], b[2], t) : a[2]) * Math.PI / 180;
      const col = COLORS[p.side];
      if (a[7] > 0.5) { // flashed
        ctx.fillStyle = "rgba(255,255,255,0.7)"; ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2); ctx.fill();
      }
      ctx.strokeStyle = col; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + Math.cos(yaw) * 22, y - Math.sin(yaw) * 22); ctx.stroke();
      ctx.fillStyle = col; ctx.strokeStyle = "#0b0e14"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      if (a[5]) { ctx.fillStyle = "#ff3b3b"; ctx.fillRect(x - 4, y - 4, 8, 8); }        // bomb carrier
      if (a[6]) { ctx.strokeStyle = "#7CFC00"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI * 2); ctx.stroke(); }
      // health bar
      ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(x - 12, y + 12, 24, 4);
      ctx.fillStyle = a[3] > 50 ? "#4ade80" : a[3] > 20 ? "#facc15" : "#ef4444"; ctx.fillRect(x - 12, y + 12, 24 * a[3] / 100, 4);
      if (showNames) {
        ctx.font = "bold 15px sans-serif"; ctx.fillStyle = "#fff"; ctx.strokeStyle = "rgba(0,0,0,0.85)"; ctx.lineWidth = 4;
        ctx.strokeText(p.name, x + 13, y - 10); ctx.fillText(p.name, x + 13, y - 10);
      }
    }
  });

  return <canvas ref={ref} className="radar" />;
}
