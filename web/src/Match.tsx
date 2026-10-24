import { useEffect, useMemo, useState } from "react";
import { api, Overview, RoundList, SwingPlayer } from "./api";
import AskPanel from "./AskPanel";
import CoachCards from "./CoachCards";

type SortKey = "rating" | "swing";

export default function Match({ id }: { id: string }) {
  const [ov, setOv] = useState<Overview | null>(null);
  const [rounds, setRounds] = useState<RoundList | null>(null);
  const [swing, setSwing] = useState<Record<string, SwingPlayer> | null>(null);
  const [swingMsg, setSwingMsg] = useState<string | null>(null);
  const [sort, setSort] = useState<SortKey>("rating");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.overview(id).then(setOv).catch((e) => setError(e.message));
    api.rounds(id).then(setRounds).catch(() => {});
    api.swing(id)
      .then((s) => setSwing(Object.fromEntries(s.players.map((p) => [p.name, p]))))
      .catch((e) => setSwingMsg(e.message));
  }, [id]);

  const teams = useMemo(() => {
    if (!ov) return [];
    const names = Object.keys(ov.score);
    return names.map((t) => {
      const rows = ov.scoreboard.filter((r) => r.team === t);
      const val = (r: typeof rows[number]) => (sort === "swing" && swing?.[r.name] ? swing[r.name].swing_per_round_pp : r.rating);
      return { team: t, score: ov.score[t], rows: [...rows].sort((a, b) => val(b) - val(a)) };
    });
  }, [ov, swing, sort]);

  if (error) return <div className="page"><div className="error">{error}</div></div>;
  if (!ov) return <div className="page muted">Loading…</div>;

  return (
    <div className="page">
      <a href="#/" className="back">← Matches</a>
      <div className="matchhead">
        <span className="map">{ov.map?.replace("de_", "")}</span>
        {teams.map((t, i) => (
          <span key={t.team} className="teamscore">
            {i > 0 && <span className="vs">:</span>}
            <span className="tname">{t.team}</span> <b>{t.score.rounds_won}</b>
          </span>
        ))}
      </div>

      <div className="sortbar">
        Rank by:
        <button className={sort === "rating" ? "on" : ""} onClick={() => setSort("rating")}>Rating</button>
        <button className={sort === "swing" ? "on" : ""} onClick={() => setSort("swing")} disabled={!swing}>Swing</button>
        {swingMsg && <span className="muted"> Swing unavailable: {swingMsg}</span>}
      </div>

      {teams.map((t) => (
        <table key={t.team} className="board">
          <thead>
            <tr>
              <th className="left">{t.team}</th><th>Rating</th><th title="Average change in the team's round win chance from this player's plays, percentage points per round">Swing/rd</th>
              <th>K</th><th>D</th><th>A</th><th>ADR</th><th>KAST</th><th>Open K–D</th><th>Clutch</th><th>3k</th><th>4k</th><th>Ace</th>
            </tr>
          </thead>
          <tbody>
            {t.rows.map((r) => {
              const s = swing?.[r.name];
              return (
                <tr key={r.name}>
                  <td className="left">{r.name}</td>
                  <td className={r.rating >= 1.15 ? "good" : r.rating < 0.85 ? "bad" : ""}>{r.rating.toFixed(2)}</td>
                  <td className={s ? (s.swing_per_round_pp > 1 ? "good" : s.swing_per_round_pp < -1 ? "bad" : "") : "muted"}>
                    {s ? `${s.swing_per_round_pp > 0 ? "+" : ""}${s.swing_per_round_pp.toFixed(2)}` : "–"}
                  </td>
                  <td>{r.kills}</td><td>{r.deaths}</td><td>{r.assists}</td><td>{r.adr}</td><td>{r.kast}%</td>
                  <td>{r.opening_kills}–{r.opening_deaths}</td><td>{r.clutches_won}/{r.clutches_played}</td>
                  <td>{r.k3 || ""}</td><td>{r.k4 || ""}</td><td>{r.aces || ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ))}

      <CoachCards id={id} players={ov.scoreboard.map((r) => r.name)} />

      <AskPanel id={id} players={ov.scoreboard.map((r) => r.name)} />

      {rounds && (
        <>
          <h2>Rounds</h2>
          <div className="roundstrip">
            {rounds.rounds.map((r, i) => {
              const newHalf = i > 0 && rounds.halves.some((h) => Number(h.rounds.split("-")[0]) === r.round);
              const winnerBuy = r.teams.find((t) => t.team === r.winner)?.buy;
              const multi = ov.multikills.rounds.filter((m) => m.round === r.round);
              return (
                <a key={r.round} href={`#/demo/${id}/round/${r.round}`}
                   className={`tile ${r.winner_side === "CT" ? "ct" : "t"} ${newHalf ? "half" : ""}`}
                   title={`Round ${r.round}: ${r.winner} (${r.winner_side}) won, ${r.reason}. Buys: ${r.teams.map((t) => `${t.team} ${t.buy}`).join(", ")}${multi.length ? ". " + multi.map((m) => `${m.player} ${m.type}`).join(", ") : ""}`}>
                  <span className="rn">{r.round}</span>
                  <span className="buy">{winnerBuy === "full" ? "" : winnerBuy}</span>
                  {r.bomb_planted && <span className="plant">●</span>}
                  {multi.some((m) => m.kills >= 4) && <span className="star">★</span>}
                </a>
              );
            })}
          </div>
          <p className="muted legend">Blue: CT won · orange: T won · ● bomb planted · ★ 4k or ace · label: winner's buy when it wasn't a full buy. Click a round for the replay.</p>
        </>
      )}
    </div>
  );
}
