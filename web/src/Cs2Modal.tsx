import { useEffect, useState } from "react";
import { api, Cs2Info } from "./api";

export default function Cs2Modal({ id, tick, label, onClose }: { id: string; tick: number; label: string; onClose: () => void }) {
  const [info, setInfo] = useState<Cs2Info | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<number | null>(null);

  const load = () => api.cs2(id, tick).then(setInfo).catch((e) => setError(e.message));
  useEffect(() => { load(); }, [id, tick]);

  const prepare = async () => {
    setBusy(true); setError(null);
    try { await api.cs2Prepare(id); await load(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const copy = (text: string, i: number) => navigator.clipboard.writeText(text).then(() => setCopied(i));

  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Watch in CS2</h3>
        <p className="muted">{label}</p>
        {error && <div className="error">{error}</div>}
        {info && !info.game_dir && (
          <p>CS2's folder wasn't found. Add <code>CS2_GAME_DIR=...\Counter-Strike Global Offensive\game\csgo</code> to your .env and restart the app.</p>
        )}
        {info?.game_dir && !info.prepared && (
          <p>Step 0: CS2 can only play demos inside its own folder.{" "}
            <button onClick={prepare} disabled={busy}>{busy ? "Copying…" : "Copy this demo to CS2"}</button>
          </p>
        )}
        {info?.game_dir && info.prepared && <p className="good">✓ The demo is in CS2's folder.</p>}
        {info && (
          <ol className="cmds">
            <li>Open CS2 and press <kbd>~</kbd> for the console.</li>
            {info.commands.map((c, i) => (
              <li key={c}>
                Paste {i === 0 ? "this to load the demo" : "this to jump to 5 seconds before the moment"}:
                <div className="cmd"><code>{c}</code><button onClick={() => copy(c, i)}>{copied === i ? "Copied" : "Copy"}</button></div>
              </li>
            ))}
          </ol>
        )}
        <button className="close" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
