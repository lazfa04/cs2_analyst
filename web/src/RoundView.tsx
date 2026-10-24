import { useEffect, useRef, useState } from "react";
import { api, Replay, WinProb } from "./api";
import Cs2Modal from "./Cs2Modal";
import ReplayCanvas from "./ReplayCanvas";
import WinChart from "./WinChart";

const SPEEDS = [0.5, 1, 2, 4];

export default function RoundView({ id, n }: { id: string; n: number }) {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [wp, setWp] = useState<WinProb | null>(null);
  const [wpMsg, setWpMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [names, setNames] = useState(true);
  const [cs2, setCs2] = useState<{ tick: number; label: string } | null>(null);
  const last = useRef<number | null>(null);

  useEffect(() => {
    api.replay(id, n).then((r) => { setReplay(r); setTick(r.start_tick); }).catch((e) => setError(e.message));
    api.winprob(id, n).then(setWp).catch((e) => setWpMsg(e.message));
  }, [id, n]);

  const endTick = replay ? replay.ticks[replay.ticks.length - 1] ?? replay.end_tick : 0;

  useEffect(() => {
    if (!playing || !replay) return;
    let raf = 0;
    const step = (now: number) => {
      const dt = last.current === null ? 0 : (now - last.current) / 1000;
      last.current = now;
      setTick((t) => {
        const next = t + dt * replay.tick_rate * speed;
        if (next >= endTick) { setPlaying(false); return endTick; }
        return next;
      });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => { cancelAnimationFrame(raf); last.current = null; };
  }, [playing, speed, replay, endTick]);

  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.code === "Space") { e.preventDefault(); setPlaying((p) => !p); } };
    addEventListener("keydown", key);
    return () => removeEventListener("keydown", key);
  }, []);

  if (error) return <div className="page"><a href={`#/demo/${id}`} className="back">← Match</a><div className="error">{error}</div></div>;
  if (!replay) return <div className="page muted">Loading round {n}… (the first replay of a match reads the demo once, ~5 seconds)</div>;

  const seconds = (t: number) => ((t - replay.start_tick) / replay.tick_rate).toFixed(1);
  const swingAt = (t: number) => wp?.steps.find((s) => s.tick === t);
  const seek = (t: number) => { setTick(t); };

  return (
    <div className="page round">
      <div className="roundnav">
        <a href={`#/demo/${id}`} className="back">← Match</a>
        <a href={`#/demo/${id}/round/${n - 1}`} className={n <= 1 ? "disabled" : ""}>‹ Round {n - 1}</a>
        <h2>Round {n} <span className={replay.winner_side === "CT" ? "ctc" : "tc"}>{replay.winner_side} win</span>
          <span className="muted"> · {replay.reason?.replace(/_/g, " ")}</span></h2>
        <a href={`#/demo/${id}/round/${n + 1}`}>Round {n + 1} ›</a>
      </div>

      <div className="roundgrid">
        <div>
          <ReplayCanvas replay={replay} tick={tick} showNames={names} />
          <div className="controls">
            <button onClick={() => { if (tick >= endTick) setTick(replay.start_tick); setPlaying(!playing); }}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
            <input type="range" min={replay.start_tick} max={endTick} value={tick} step={1}
                   onChange={(e) => seek(Number(e.target.value))} />
            <span className="time">{seconds(tick)}s</span>
            {SPEEDS.map((s) => <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>{s}x</button>)}
            <label className="muted"><input type="checkbox" checked={names} onChange={(e) => setNames(e.target.checked)} /> names</label>
            <button onClick={() => setCs2({ tick: Math.round(tick), label: `Round ${n} at ${seconds(tick)}s` })}>Watch in CS2</button>
          </div>
        </div>

        <div className="side">
          <h3>Win probability</h3>
          {wp ? (
            <WinChart steps={wp.steps} startTick={replay.start_tick} endTick={endTick} tickRate={replay.tick_rate}
                      currentTick={tick} onSeek={seek} />
          ) : <p className="muted">{wpMsg ?? "Loading…"}</p>}
          {wp && <p className="muted">Starts at {Math.round(wp.steps[0].p_t_win * 100)}% for T (the buys). Dot size = how much the event swung the round; click to jump there.</p>}

          <h3>Kill feed</h3>
          <ul className="feed">
            {replay.kills.map((k) => {
              const s = swingAt(k.tick);
              const forKiller = s && k.attacker_side ? (k.attacker_side === "T" ? s.swing_t_pp : -s.swing_t_pp) : null;
              return (
                <li key={k.tick + k.victim} className={tick >= k.tick ? "" : "future"}>
                  <button className="link" onClick={() => seek(k.tick - replay.tick_rate * 2)}>{seconds(k.tick)}s</button>
                  <span className={k.attacker_side === "CT" ? "ctc" : "tc"}>{k.attacker ?? "world"}</span>
                  <span className="weapon">{k.weapon}{k.headshot ? " ◎" : ""}</span>
                  <span className={k.victim_side === "CT" ? "ctc" : "tc"}>{k.victim}</span>
                  {forKiller !== null && <span className={`swing ${forKiller >= 0 ? "good" : "bad"}`}>{forKiller >= 0 ? "+" : ""}{forKiller.toFixed(1)}</span>}
                  <button className="mini" title="Watch this moment in CS2"
                          onClick={() => setCs2({ tick: k.tick, label: `Round ${n}: ${k.attacker} kills ${k.victim}` })}>CS2</button>
                </li>
              );
            })}
            {replay.bomb.map((b) => (
              <li key={b.tick} className={tick >= b.tick ? "" : "future"}>
                <button className="link" onClick={() => seek(b.tick - replay.tick_rate * 2)}>{seconds(b.tick)}s</button>
                <span>{b.name} {b.event === "finish_plant" ? "plants the bomb" : "defuses"}</span>
              </li>
            ))}
          </ul>
          <p className="muted">Swing = change in the killer's team's chance to win the round, in points.</p>
        </div>
      </div>
      {cs2 && <Cs2Modal id={id} tick={cs2.tick} label={cs2.label} onClose={() => setCs2(null)} />}
    </div>
  );
}
