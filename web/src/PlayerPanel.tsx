import { useEffect, useState } from "react";
import { api, CompareRow, PlayerContext } from "./api";

const levelClass = (lvl?: number) => (!lvl ? "" : lvl >= 10 ? "lv10" : lvl >= 8 ? "lv8" : lvl >= 4 ? "lv4" : lvl >= 2 ? "lv2" : "lv1");
const fmt = (v: number | null | undefined, digits = 1) => (v === null || v === undefined ? "–" : Number.isInteger(v) ? String(v) : v.toFixed(digits));
// e.g. 259.04 with 0 digits -> "259"

// Leetify's rating block mixes 0-100 scores (aim, positioning, utility) and per-round contributions
// (clutch, opening: 0.05 = +5 per 100 rounds). Show each the way Leetify's own site does.
const LEETIFY_LABELS: Record<string, string> = { ct_leetify: "CT rating", t_leetify: "T rating" };

function leetifyValue(v: number) {
  if (Math.abs(v) <= 1) return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}`;
  return v.toFixed(0);
}

function verdict(r: CompareRow) {
  const base = r.recent ?? r.usual;
  if (r.this_match === null || base === null || base === 0) return "";
  const diff = (r.this_match - base) / Math.abs(base);
  if (Math.abs(diff) < 0.08) return "same";
  return (diff > 0) === r.higher_is_better ? "good" : "bad";
}

export default function PlayerPanel({ id, player }: { id: string; player: string }) {
  const [ctx, setCtx] = useState<PlayerContext | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!player) return;
    setBusy(true); setError(null); setCtx(null);
    api.context(id, player).then(setCtx).catch((e) => setError(e.message)).finally(() => setBusy(false));
  }, [id, player]);

  if (!player) return null;
  const f = ctx?.faceit;
  const l = ctx?.leetify;
  const rec = f?.recent;

  return (
    <section className="profile">
      {busy && <p className="muted">Loading FACEIT and Leetify…</p>}
      {error && <div className="error">{error}</div>}
      {ctx && (
        <>
          <div className="profilehead">
            <div className="svc">
              <div className="svcname">FACEIT</div>
              {f?.error ? <p className="muted small">{f.error}</p> : (
                <div className="faceit">
                  <span className={`level ${levelClass(f?.level)}`}>{f?.level ?? "?"}</span>
                  <div>
                    <a href={`https://www.faceit.com/en/players/${f?.nickname}`} target="_blank" rel="noreferrer"><b>{f?.nickname}</b></a>
                    <div className="muted small">{f?.elo} Elo · {fmt(f?.lifetime?.matches, 0)} matches · {fmt(f?.lifetime?.win_rate_pct, 0)}% won</div>
                  </div>
                </div>
              )}
              {rec && rec.count > 0 && (
                <div className="form" title="Latest first">
                  {rec.matches.slice(0, 10).map((m, i) => (
                    <span key={i} className={`wl ${m.result === "win" ? "w" : "l"}`}
                          title={`${m.map?.replace("de_", "")} ${m.score} · ${m.kills ?? "?"}-${m.deaths ?? "?"} · ${m.adr ?? "?"} ADR`}>
                      {m.result === "win" ? "W" : "L"}
                    </span>
                  ))}
                  <span className="muted small">{rec.wins}/{rec.count} won</span>
                </div>
              )}
            </div>
            <div className="svc">
              <div className="svcname">Leetify</div>
              {l?.error ? <p className="muted small">{l.error}</p> : (
                <>
                  <a href={`https://leetify.com/app/profile/${ctx.steam64}`} target="_blank" rel="noreferrer"><b>{l?.name}</b></a>
                  <div className="muted small">{l?.total_matches} matches tracked{typeof l?.winrate === "number" ? ` · ${fmt(l.winrate <= 1 ? l.winrate * 100 : l.winrate, 0)}% won` : ""}</div>
                  <div className="chips">
                    {Object.entries(l?.rating ?? {}).filter(([, v]) => typeof v === "number").map(([k, v]) => (
                      <span key={k} className="ctx" title={Math.abs(v as number) <= 1 ? "Per-round contribution: +5 means +5 per 100 rounds" : "Leetify score"}>
                        {LEETIFY_LABELS[k] ?? k.replace(/_/g, " ")} <b>{leetifyValue(v as number)}</b>
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>

          <table className="board compare">
            <thead>
              <tr><th className="left">{ctx.player}: this match vs usual</th><th>This match</th>
                <th>Last {ctx.recent_count ?? 20} (FACEIT)</th><th>Usual</th><th className="left">Usual from</th></tr>
            </thead>
            <tbody>
              {ctx.comparison.map((r) => {
                const d = r.stat === "K/D" ? 2 : r.stat.includes("($)") ? 0 : 1;
                return (
                <tr key={r.stat}>
                  <td className="left">{r.stat}</td>
                  <td className={verdict(r)}>
                    {fmt(r.this_match, d)}
                    {r.stat.startsWith("Opening") && <span className="muted small"> ({ctx.this_match.opening_duels})</span>}
                  </td>
                  <td>{fmt(r.recent, d)}</td>
                  <td>{fmt(r.usual, d)}</td>
                  <td className="left muted small">{r.usual === null ? "–" : r.usual_source}</td>
                </tr>
                );
              })}
            </tbody>
          </table>
          <p className="muted small">Green/red: this match against the last {ctx.recent_count ?? 20} FACEIT matches where there are any, else the usual number. Leetify measures trading and utility its own way, so those rows are roughly comparable, not identical.</p>
        </>
      )}
    </section>
  );
}
