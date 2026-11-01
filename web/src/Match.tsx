import { useEffect, useMemo, useState } from "react";
import { api, Overview, RoundList, ScoreRow, Side, SwingPlayer } from "./api";
import AskPanel from "./AskPanel";
import CoachCards from "./CoachCards";
import { mapLabel, radarUrl } from "./Home";
import PlayerPanel from "./PlayerPanel";
import Timeline, { halfStarts } from "./Timeline";

const store = (area: Storage | undefined, key: string) => ({
  get: () => { try { return area?.getItem(key) ?? ""; } catch { return ""; } },
  set: (v: string) => { try { area?.setItem(key, v); } catch { /* private mode */ } },
});
const savedPlayer = store(globalThis.localStorage, "cs2.player");
export const savedTab = store(globalThis.sessionStorage, "cs2.tab");

export const TABS = [
  { id: "board", label: "Scoreboard" },
  { id: "rounds", label: "Rounds" },
  { id: "player", label: "Player & coach" },
  { id: "ask", label: "Ask the analyst" },
] as const;
export type Tab = (typeof TABS)[number]["id"];

type SortKey = "rating" | "swing";
type TeamInfo = { team: string; side: Side; score: Overview["score"][string]; rows: ScoreRow[] };

export default function Match({ id, tab: tabProp }: { id: string; tab?: Tab }) {
  const [ov, setOv] = useState<Overview | null>(null);
  const [rounds, setRounds] = useState<RoundList | null>(null);
  const [swing, setSwing] = useState<Record<string, SwingPlayer> | null>(null);
  const [swingMsg, setSwingMsg] = useState<string | null>(null);
  const [sort, setSort] = useState<SortKey>("rating");
  const [error, setError] = useState<string | null>(null);
  const [player, setPlayer] = useState(savedPlayer.get());
  const tab: Tab = tabProp ?? (TABS.find((t) => t.id === savedTab.get())?.id) ?? "board";

  useEffect(() => { savedTab.set(tab); }, [tab]);
  useEffect(() => {
    api.overview(id).then(setOv).catch((e) => setError(e.message));
    api.rounds(id).then(setRounds).catch(() => {});
    api.swing(id)
      .then((s) => setSwing(Object.fromEntries(s.players.map((p) => [p.name, p]))))
      .catch((e) => setSwingMsg(e.message));
  }, [id]);

  const teams: TeamInfo[] = useMemo(() => {
    if (!ov) return [];
    return Object.keys(ov.score).map((t) => {
      const rows = ov.scoreboard.filter((r) => r.team === t);
      const val = (r: ScoreRow) => (sort === "swing" && swing?.[r.name] ? swing[r.name].swing_per_round_pp : r.rating);
      const side: Side = rounds?.halves[0]?.sides[t] ?? "CT";
      return { team: t, side, score: ov.score[t], rows: [...rows].sort((a, b) => val(b) - val(a)) };
    });
  }, [ov, swing, sort, rounds]);

  if (error) return <div className="page"><a href="#/" className="back">← Matches</a><div className="error">{error}</div></div>;
  if (!ov) return <div className="page"><p className="loading">Loading match…</p></div>;

  const known = ov.scoreboard.some((r) => r.name === player);
  const pick = (name: string) => { setPlayer(name); savedPlayer.set(name); };
  const mvp = [...ov.scoreboard].sort((a, b) => b.rating - a.rating)[0]?.name;
  const [a, b] = teams;

  return (
    <div className="page">
      <a href="#/" className="back">← Matches</a>

      <section className="hero cut" style={{ marginTop: 12 }}>
        <div className="art" style={{ backgroundImage: `url(${radarUrl(ov.map)})` }} />
        <div className="herotop">
          <span className="mapname">{mapLabel(ov.map)} · {ov.rounds_played} rounds</span>
        </div>
        {a && b && (
          <div className="scoreline">
            <TeamBlock t={a} won={a.score.rounds_won > b.score.rounds_won} />
            <div className="score">
              <span className={a.score.rounds_won < b.score.rounds_won ? "lose" : ""}>{a.score.rounds_won}</span>
              <span className="colon">:</span>
              <span className={b.score.rounds_won < a.score.rounds_won ? "lose" : ""}>{b.score.rounds_won}</span>
            </div>
            <TeamBlock t={b} won={b.score.rounds_won > a.score.rounds_won} right />
          </div>
        )}
        {rounds && <Timeline id={id} rounds={rounds} multikills={ov.multikills.rounds} />}
      </section>

      <nav className="tabs">
        {TABS.map((t) => (
          <a key={t.id} href={`#/demo/${id}/${t.id}`} className={tab === t.id ? "on" : ""}>
            {t.label}
            {t.id === "player" && known && <span className="badge">{player}</span>}
          </a>
        ))}
      </nav>

      {/* Panels stay mounted so switching tabs doesn't refetch FACEIT/Leetify or the coach review. */}
      <div hidden={tab !== "board"}>
        <div className="toolbar">
          <span className="eyebrow">Rank by</span>
          <div className="seg">
            <button className={sort === "rating" ? "on" : ""} onClick={() => setSort("rating")}>Rating</button>
            <button className={sort === "swing" ? "on" : ""} onClick={() => setSort("swing")} disabled={!swing}>Swing</button>
          </div>
          {swingMsg && <span className="muted small">Swing unavailable: {swingMsg}</span>}
          <span className="muted small" style={{ marginLeft: "auto" }}>Click a name to open their player tab.</span>
        </div>
        {teams.map((t) => (
          <Board key={t.team} t={t} swing={swing} mvp={mvp} me={known ? player : ""}
                 onPick={(n) => { pick(n); location.hash = `#/demo/${id}/player`; }} />
        ))}
      </div>

      <div hidden={tab !== "rounds"}>
        {rounds ? <RoundsTable id={id} rounds={rounds} ov={ov} teams={teams} /> : <p className="loading">Loading rounds…</p>}
      </div>

      <div hidden={tab !== "player"}>
        <div className="pickers">
          {teams.map((t) => (
            <div key={t.team} className="pteam">
              <div className="eyebrow"><span className={`side-tag ${t.side.toLowerCase()}`}>{t.side}</span>{t.team}</div>
              <div className="pills">
                {t.rows.map((r) => (
                  <button key={r.name} className={`pill ${t.side.toLowerCase()} ${player === r.name ? "on" : ""}`} onClick={() => pick(r.name)}>{r.name}</button>
                ))}
              </div>
            </div>
          ))}
        </div>
        <p className="note">Side shown is the starting side. Your pick is remembered and also sets who "I" means in Ask.</p>
        {known && <PlayerPanel id={id} player={player} />}
        <CoachCards id={id} player={known ? player : ""} />
      </div>

      <div hidden={tab !== "ask"}>
        <AskPanel id={id} players={ov.scoreboard.map((r) => r.name)} player={player || undefined} />
      </div>
    </div>
  );
}

function TeamBlock({ t, won, right }: { t: TeamInfo; won: boolean; right?: boolean }) {
  const other: Side = t.side === "CT" ? "T" : "CT";
  const half = (s: Side) => (s === "CT" ? t.score.ct_rounds_won : t.score.t_rounds_won);
  return (
    <div className={`team ${right ? "right" : ""}`}>
      <span className={`tname ${t.side === "CT" ? "ctc" : "tc"}`} title={t.team}>{t.team}</span>
      <span className="halves">
        <span><span className={t.side === "CT" ? "ctc" : "tc"}>{t.side}</span> <b>{half(t.side)}</b></span>
        <span><span className={other === "CT" ? "ctc" : "tc"}>{other}</span> <b>{half(other)}</b></span>
      </span>
      {won && <span className="winner-tag">WIN</span>}
    </div>
  );
}

function Board({ t, swing, mvp, me, onPick }: { t: TeamInfo; swing: Record<string, SwingPlayer> | null; mvp?: string; me: string; onPick: (name: string) => void }) {
  return (
    <div className="tablewrap">
      <table className="board">
        <thead>
          <tr>
            <th className="left teamhead"><span className={`side-tag ${t.side.toLowerCase()}`}>{t.side}</span>{t.team}<span className="tscore">{t.score.rounds_won}</span></th>
            <th>Rating</th><th title="Average change in the team's round win chance from this player's plays, percentage points per round">Swing/rd</th>
            <th className="sep">K</th><th>D</th><th>A</th><th>ADR</th><th>KAST</th>
            <th className="sep">Open K–D</th><th>Clutch</th><th>3k</th><th>4k</th><th>Ace</th>
          </tr>
        </thead>
        <tbody>
          {t.rows.map((r) => {
            const s = swing?.[r.name];
            const rc = r.rating >= 1.15 ? "good" : r.rating < 0.85 ? "bad" : "";
            return (
              <tr key={r.name} className={r.name === me ? "me" : ""}>
                <td className="left pname">
                  <button className="link" style={{ color: "inherit" }} onClick={() => onPick(r.name)}>{r.name}</button>
                  {r.name === mvp && <span className="mvp">MVP</span>}
                </td>
                <td className={rc}><span className="rating"><span className="bar"><i style={{ width: `${Math.min(1, r.rating / 2) * 100}%` }} /></span><b>{r.rating.toFixed(2)}</b></span></td>
                <td className={s ? (s.swing_per_round_pp > 1 ? "good" : s.swing_per_round_pp < -1 ? "bad" : "") : "muted"}>
                  {s ? `${s.swing_per_round_pp > 0 ? "+" : ""}${s.swing_per_round_pp.toFixed(2)}` : "–"}
                </td>
                <td className="sep">{r.kills}</td><td>{r.deaths}</td><td>{r.assists}</td><td>{r.adr}</td><td>{r.kast}%</td>
                <td className="sep">{r.opening_kills}–{r.opening_deaths}</td><td>{r.clutches_won}/{r.clutches_played}</td>
                <td className={r.k3 ? "" : "zero"}>{r.k3 || "·"}</td><td className={r.k4 ? "" : "zero"}>{r.k4 || "·"}</td><td className={r.aces ? "" : "zero"}>{r.aces || "·"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RoundsTable({ id, rounds, ov, teams }: { id: string; rounds: RoundList; ov: Overview; teams: TeamInfo[] }) {
  const starts = halfStarts(rounds);
  return (
    <>
      <div className="tablewrap">
        <table className="board rtable">
          <thead>
            <tr>
              <th className="left">#</th><th className="left">Winner</th><th className="left">How</th>
              {teams.map((t) => <th key={t.team} className="left sep" title={`${t.team}'s buy`}>{t.team}</th>)}
              <th className="left sep">First kill</th><th className="left">Multi-kills</th><th>Score</th>
            </tr>
          </thead>
          <tbody>
            {rounds.rounds.map((r) => {
              const multi = ov.multikills.rounds.filter((m) => m.round === r.round);
              return (
                <tr key={r.round} className={starts.has(r.round) ? "half" : ""} onClick={() => { location.hash = `#/demo/${id}/round/${r.round}`; }}>
                  <td className="left rn"><a href={`#/demo/${id}/round/${r.round}`}>{r.round}</a></td>
                  <td className="left"><span className="winner"><span className={`side-tag ${r.winner_side.toLowerCase()}`}>{r.winner_side}</span>{r.winner}</span></td>
                  <td className="left muted" style={{ textTransform: "capitalize" }}>{r.reason.replace(/_/g, " ")}{r.bomb_planted && <span className="bad"> · planted</span>}</td>
                  {teams.map((t) => {
                    const tb = r.teams.find((x) => x.team === t.team);
                    return (
                      <td key={t.team} className="left sep">
                        {tb?.buy && <span className={`buytag ${tb.buy.toLowerCase()}`}>{tb.buy}</span>}
                        {tb?.equipment != null && <span className="dim small"> ${(tb.equipment / 1000).toFixed(1)}k</span>}
                      </td>
                    );
                  })}
                  <td className="left sep muted">{r.first_kill_team ?? "–"}</td>
                  <td className="left"><span className="multis">{multi.map((m) => <span key={m.player} className="ctx"><b>{m.type}</b> {m.player}</span>)}</span></td>
                  <td className="num">{teams.map((t) => r.score_after[t.team] ?? "–").join(" : ")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="legend">
        <span><i className="ct" />CT won</span><span><i className="t" />T won</span><span><i className="pin" />Bomb planted</span>
        <span>Click a round for the 2D replay and win probability.</span>
      </div>
    </>
  );
}
