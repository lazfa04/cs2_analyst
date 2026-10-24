import { useEffect, useState } from "react";
import { Answer, api } from "./api";

const MODELS = [
  { id: "claude-haiku-5-5", label: "Haiku (≈$0.005)" },
  { id: "claude-sonnet-5-5", label: "Sonnet (≈$0.04, better)" },
];

const MATCH_SUGGESTIONS = [
  "How did I play this match?",
  "Which rounds did we throw, and why?",
  "Which of my kills mattered most?",
  "Why did we win or lose this match?",
];
const ROUND_SUGGESTIONS = [
  "What went wrong for my team this round?",
  "What decided this round?",
  "Did I do anything that cost us this round?",
  "Was this round lost on the buy?",
];

const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

type Entry = Answer & { round?: number };

// Rounds mentioned in an answer become links to that round's replay.
function linkRounds(text: string, id: string) {
  const parts = text.split(/(\brounds? \d+(?:(?:,| and| to|-) ?\d+)*)/gi);
  return parts.map((part, i) => {
    if (!/^rounds? \d/i.test(part)) return <span key={i}>{part}</span>;
    const nums: string[] = part.match(/\d+/g) ?? [];
    return (
      <span key={i}>
        {part.split(/(\d+)/).map((bit, j) =>
          nums.includes(bit) ? <a key={j} href={`#/demo/${id}/round/${bit}`}>{bit}</a> : <span key={j}>{bit}</span>)}
      </span>
    );
  });
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

const ICON: Record<string, string> = {
  verified: "✓", supported: "✓", failed: "✗", unsupported: "⚠", unchecked: "?", not_judged: "–",
};

export default function AskPanel({ id, round, players, player: chosen }: { id: string; round?: number; players: string[]; player?: string }) {
  const histKey = `cs2.ask.${id}`;
  const [question, setQuestion] = useState("");
  const [player, setPlayer] = useState(store.get("cs2.player") ?? "");
  const [model, setModel] = useState(store.get("cs2.model") ?? MODELS[0].id);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Entry[]>(() => {
    try { return JSON.parse(store.get(histKey) ?? "[]"); } catch { return []; }
  });

  useEffect(() => { if (player) store.set("cs2.player", player); }, [player]);
  useEffect(() => { if (chosen) setPlayer(chosen); }, [chosen]);
  useEffect(() => { store.set("cs2.model", model); }, [model]);

  const ask = async (q: string) => {
    if (!q.trim() || busy) return;
    setBusy(true); setError(null);
    try {
      const a = await api.ask(id, { question: q, player: player || undefined, round, model });
      const next = [{ ...a, round }, ...history].slice(0, 20);
      setHistory(next);
      store.set(histKey, JSON.stringify(next));
      setQuestion("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const shownHistory = round === undefined ? history : history.filter((h) => h.round === round);

  return (
    <div className="ask">
      <h3>Ask the analyst{round !== undefined && <span className="muted"> · about round {round}</span>}</h3>
      <div className="askrow">
        <select value={player} onChange={(e) => setPlayer(e.target.value)} title="Who 'I' and 'my' refer to">
          <option value="">I am… (optional)</option>
          {players.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </div>
      <div className="askrow">
        <input value={question} onChange={(e) => setQuestion(e.target.value)} disabled={busy}
               onKeyDown={(e) => { if (e.key === "Enter") ask(question); e.stopPropagation(); }}
               placeholder={round !== undefined ? "e.g. why did we lose this round?" : "e.g. how did I play?"} />
        <button onClick={() => ask(question)} disabled={busy || !question.trim()}>{busy ? "Thinking…" : "Ask"}</button>
      </div>
      <div className="chips">
        {(round !== undefined ? ROUND_SUGGESTIONS : MATCH_SUGGESTIONS).map((s) => (
          <button key={s} className="chip" disabled={busy} onClick={() => ask(s)}>{s}</button>
        ))}
      </div>
      {busy && <div className="notice">The agent is pulling stats with its tools, then every fact it states gets checked against them and a reviewer reads the conclusions. Usually 15–30 seconds.</div>}
      {error && <div className="error">{error}</div>}

      {shownHistory.map((a, i) => {
        const shown = a.claims.filter((c) => !["failed", "unsupported"].includes(c.status));
        const dropped = a.claims.filter((c) => ["failed", "unsupported"].includes(c.status));
        const flagged = [...a.summary_contradictions, ...a.summary_issues];
        return (
          <div className="answer" key={i}>
            <div className="q">{a.question.replace(/^\(I'm looking at round \d+\.\) /, "")}</div>
            <div className="a">{a.summary ? linkRounds(a.summary, id) : <span className="bad">No answer ({a.stop_reason}).</span>}</div>
            {shown.length > 0 && (
              <details>
                <summary>{plural(shown.filter((c) => c.kind === "fact").length, "checked fact")}, {plural(shown.filter((c) => c.kind === "interpretation").length, "conclusion")}</summary>
                <ul className="claims">
                  {shown.map((c) => (
                    <li key={c.id} className={c.status}><span className="ic">{ICON[c.status]}</span>{linkRounds(c.text, id)}</li>
                  ))}
                </ul>
              </details>
            )}
            {dropped.length > 0 && (
              <div className="dropped">
                <b>Didn't hold up:</b>
                <ul className="claims">{dropped.map((c) => <li key={c.id} className={c.status}><span className="ic">{ICON[c.status]}</span>{c.text}{c.note && <span className="muted"> ({c.note})</span>}</li>)}</ul>
              </div>
            )}
            {flagged.length > 0 && (
              <div className="dropped"><b>In the answer but not backed by a checked fact:</b>
                <ul className="claims">{flagged.map((f) => <li key={f} className="unsupported"><span className="ic">⚠</span>{f}</li>)}</ul>
              </div>
            )}
            <div className="meta muted">
              {a.tools_used.map((t) => t.tool.replace(/^get_/, "")).join(" · ")} · {a.model.includes("haiku") ? "Haiku" : "Sonnet"} ·{" "}
              {a.usage.cost_usd !== null ? `$${a.usage.cost_usd.toFixed(3)}` : "cost unknown"} · {a.usage.elapsed_s}s
            </div>
          </div>
        );
      })}
    </div>
  );
}
