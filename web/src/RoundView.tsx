import { useEffect, useRef, useState } from "react";
import { api, Replay, RoundList, WinProb } from "./api";
import AskPanel from "./AskPanel";
import Cs2Modal from "./Cs2Modal";
import { savedTab } from "./Match";
import ReplayCanvas from "./ReplayCanvas";
import Timeline from "./Timeline";
import WinChart from "./WinChart";

const SPEEDS = [0.5, 1, 2, 4];

// `at`: open paused 3 seconds before this tick (coach cards link to the moment of a death).
export default function RoundView({ id, n, at }: { id: string; n: number; at?: number }) {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [wp, setWp] = useState<WinProb | null>(null);
  const [wpMsg, setWpMsg] = useState<string | null>(null);
  const [rounds, setRounds] = useState<RoundList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [names, setNames] = useState(true);
  const [cs2, setCs2] = useState<{ tick: number; label: string } | null>(null);
  const last = useRef<number | null>(null);

  useEffect(() => {
    api.replay(id, n).then((r) => { setReplay(r); setTick(at ? Math.max(r.start_tick, at - 3 * r.tick_rate) : r.start_tick); }).catch((e) => setError(e.message));
    api.winprob(id, n).then(setWp).catch((e) => setWpMsg(e.message));
    api.rounds(id).then(setRounds).catch(() => {});
  }, [id, n, at]);

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
    const key = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT";
      if (e.code === "Space" && !typing) { e.preventDefault(); setPlaying((p) => !p); }
    };
    addEventListener("keydown", key);
    return () => removeEventListener("keydown", key);
  }, []);

  const matchHref = `#/demo/${id}/${savedTab.get() || "board"}`;
  if (error) return <div className="page"><a href={matchHref} className="back">← Match</a><div className="error">{error}</div></div>;
  if (!replay) return <div className="page"><p className="loading">Loading round {n}… the first replay of a match reads the demo once, about 5 seconds.</p></div>;

  const seconds = (t: number) => ((t - replay.start_tick) / replay.tick_rate).toFixed(1);
  const swingAt = (t: number) => wp?.steps.find((s) => s.tick === t);
  const seek = (t: number) => { setTick(t); };

  const lastRound = rounds?.rounds.length ?? Infinity;
  const nowKill = [...replay.kills].reverse().find((k) => k.tick <= tick && tick - k.tick < replay.tick_rate * 2);

  return (
    <div className="page round">
      <a href={matchHref} className="back">← Match</a>
      <div className="roundhead" style={{ marginTop: 12 }}>
        <h1>Round {n}</h1>
        <span className="result">
          <span className={`result-badge ${replay.winner_side === "CT" ? "ct" : "t"}`}>{replay.winner_side} win</span>
          {!/^(ct|t)_win$/i.test(replay.reason ?? "") && <span className="reason">{replay.reason?.replace(/_/g, " ")}</span>}
        </span>
        <span className="nav">
          <a className={`btn ${n <= 1 ? "disabled" : ""}`} href={`#/demo/${id}/round/${n - 1}`}>‹ Prev</a>
          <a className={`btn ${n >= lastRound ? "disabled" : ""}`} href={`#/demo/${id}/round/${n + 1}`}>Next ›</a>
        </span>
      </div>
      {rounds && <Timeline id={id} rounds={rounds} current={n} />}

      <div className="roundgrid">
        <div className="radarbox">
          <ReplayCanvas replay={replay} tick={tick} showNames={names} />
          <div className="controls">
            <button className="play" title="Play / pause (space)" onClick={() => { if (tick >= endTick) setTick(replay.start_tick); setPlaying(!playing); }}>{playing ? "❚❚" : "▶"}</button>
            <input type="range" min={replay.start_tick} max={endTick} value={tick} step={1}
                   onChange={(e) => seek(Number(e.target.value))} />
            <span className="time">{seconds(tick)}s</span>
            <div className="seg">{SPEEDS.map((s) => <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>{s}x</button>)}</div>
            <label><input type="checkbox" checked={names} onChange={(e) => setNames(e.target.checked)} /> Names</label>
            <button onClick={() => setCs2({ tick: Math.round(tick), label: `Round ${n} at ${seconds(tick)}s` })}>Watch in CS2</button>
          </div>
        </div>

        <div className="stack">
          <section className="panel">
            <div className="sechead"><h3>Win probability</h3>
              {wp && <span className="aside muted small">Starts at {Math.round(wp.steps[0].p_t_win * 100)}% for T (the buys)</span>}</div>
            {wp ? (
              <WinChart steps={wp.steps} startTick={replay.start_tick} endTick={endTick} tickRate={replay.tick_rate}
                        currentTick={tick} onSeek={seek} />
            ) : <p className={wpMsg ? "muted" : "loading"}>{wpMsg ?? "Loading…"}</p>}
            {wp && <p className="note">Dot size = how much the event swung the round. Click the chart to jump there.</p>}
          </section>

          <section className="panel">
            <div className="sechead"><h3>Kill feed</h3><span className="aside muted small">Swing for the killer's team, in points</span></div>
            <ul className="feed">
              {replay.kills.map((k) => {
                const s = swingAt(k.tick);
                const forKiller = s && k.attacker_side ? (k.attacker_side === "T" ? s.swing_t_pp : -s.swing_t_pp) : null;
                return (
                  <li key={k.tick + k.victim} className={tick < k.tick ? "future" : k === nowKill ? "now" : ""}>
                    <button className="link ts" onClick={() => seek(k.tick - replay.tick_rate * 2)}>{seconds(k.tick)}s</button>
                    <span className={`who ${k.attacker_side === "CT" ? "ctc" : "tc"}`}>{k.attacker ?? "world"}</span>
                    <span className="weapon">{k.weapon}{k.headshot && <span className="hs"> ◎</span>}</span>
                    <span className={`who ${k.victim_side === "CT" ? "ctc" : "tc"}`}>{k.victim}</span>
                    {forKiller !== null && <span className={`swing ${forKiller >= 0 ? "good" : "bad"}`}>{forKiller >= 0 ? "+" : ""}{forKiller.toFixed(1)}</span>}
                    <button className="mini" style={forKiller === null ? { marginLeft: "auto" } : undefined} title="Watch this moment in CS2"
                            onClick={() => setCs2({ tick: k.tick, label: `Round ${n}: ${k.attacker} kills ${k.victim}` })}>CS2</button>
                  </li>
                );
              })}
              {replay.bomb.map((b) => (
                <li key={b.tick} className={tick >= b.tick ? "" : "future"}>
                  <button className="link ts" onClick={() => seek(b.tick - replay.tick_rate * 2)}>{seconds(b.tick)}s</button>
                  <span className="bombev">{b.name} {b.event === "finish_plant" ? "plants the bomb" : "defuses"}</span>
                </li>
              ))}
            </ul>
          </section>
          <AskPanel id={id} round={n} players={replay.players.map((p) => p.name)} />
        </div>
      </div>
      {cs2 && <Cs2Modal id={id} tick={cs2.tick} label={cs2.label} onClose={() => setCs2(null)} />}
    </div>
  );
}
