import { useEffect, useState } from "react";
import { api, Coach } from "./api";
import Cs2Modal from "./Cs2Modal";

// A player's costliest flagged deaths, from measured facts. Free: no model call.
export default function CoachCards({ id, player }: { id: string; player: string }) {
  const [coach, setCoach] = useState<Coach | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cs2, setCs2] = useState<{ tick: number; label: string } | null>(null);

  useEffect(() => {
    if (!player) return;
    setBusy(true); setError(null); setCoach(null);
    api.coach(id, player).then(setCoach).catch((e) => setError(e.message)).finally(() => setBusy(false));
  }, [id, player]);

  const profile = coach ? Object.values(coach.spacing_profile)[0] : undefined;

  return (
    <section className="coach">
      <div className="coachhead">
        <h3>Coach: deaths worth a second look</h3>
      </div>
      {!player && <p className="muted">Pick a player above to see their costliest deaths, with what the demo measured and a question to ask yourself. Free, no AI call.</p>}
      {busy && <p className="muted">Reviewing deaths… (the first time reads the demo for positions, ~5 seconds)</p>}
      {error && <div className="error">{error}</div>}
      {coach && (
        <>
          <p className="muted">
            {coach.deaths} deaths reviewed · {coach.cards.length + coach.more_flagged} worth a look
            ({coach.by_whom.player} point at {player}, {coach.by_whom.teammates} at teammates) · sorted by {coach.sorted_by}
            {profile && ` · on T, a round spent apart from the team (before the plant) in ${profile.rounds_apart}/${profile.rounds} rounds${profile.lurker ? " (lurker)" : ""}`}
          </p>
          <div className="coachgrid">
            {coach.cards.map((c) => (
              <article key={c.tick} className={`coachcard ${c.points_at}`}>
                <div className="cardtop">
                  <a className="rbadge" href={`#/demo/${id}/round/${c.round}`}>R{c.round}</a>
                  <span className={c.side === "CT" ? "ctc" : "tc"}>{c.side}</span>
                  <span className="muted">{c.time_s !== null ? `${c.time_s}s` : ""} · {c.killer ?? "world"} ({c.weapon})</span>
                  {c.swing_pp !== null && <span className="swing bad">{c.swing_pp.toFixed(1)} pp</span>}
                </div>
                <h4>{c.title}</h4>
                <span className={`whose ${c.points_at}`}>{c.points_at === "player" ? `on ${player}` : "on teammates"}</span>
                <ul className="facts">{c.facts.map((f) => <li key={f}>{f}</li>)}</ul>
                <div className="chips">{c.context.map((x) => <span key={x} className="ctx">{x}</span>)}</div>
                <p className="question">{c.question}</p>
                <div className="cardbtns">
                  <a className="btn" href={`#/demo/${id}/round/${c.round}/at/${c.tick}`}>▶ Replay the moment</a>
                  <button onClick={() => setCs2({ tick: c.tick, label: `Round ${c.round}: ${c.killer} kills ${player}` })}>Watch in CS2</button>
                </div>
              </article>
            ))}
          </div>
          {coach.cards.length === 0 && <p>No flagged deaths. {coach.notes.length ? "" : "Clean game, or nothing the demo can measure."}</p>}
          {coach.more_flagged > 0 && <p className="muted">+{coach.more_flagged} more flagged deaths with smaller costs.</p>}
          {coach.not_shown.length > 0 && (
            <details className="notshown"><summary>Not shown</summary><ul>{coach.not_shown.map((n) => <li key={n}>{n}</li>)}</ul></details>
          )}
          {coach.notes.map((n) => <p key={n} className="notice">{n}</p>)}
          <details className="notshown"><summary>How deaths are flagged</summary><p className="muted">{coach.rules}</p></details>
        </>
      )}
      {cs2 && <Cs2Modal id={id} tick={cs2.tick} label={cs2.label} onClose={() => setCs2(null)} />}
    </section>
  );
}
