import { useEffect, useState } from "react";
import { api, DemoSummary } from "./api";

export const mapLabel = (map?: string) => map?.replace(/^de_/, "") ?? "";
export const radarUrl = (map?: string) => (map ? `/api/maps/${map}/radar.png` : undefined);

export default function Home() {
  const [demos, setDemos] = useState<DemoSummary[] | null>(null);
  const [files, setFiles] = useState<{ name: string; size_mb: number; parsed: boolean }[]>([]);
  const [folder, setFolder] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);

  const load = () => {
    api.demos().then(setDemos).catch((e) => { setDemos([]); setError(e.message); });
    api.files().then((f) => { setFiles(f.files); setFolder(f.demo_dir); }).catch(() => {});
  };
  useEffect(load, []);

  const run = async (label: string, fn: () => Promise<DemoSummary>) => {
    setBusy(label); setError(null);
    try {
      const d = await fn();
      location.hash = `#/demo/${d.demo_id}`;
    } catch (e) {
      setError((e as Error).message);
      load();
    } finally {
      setBusy(null);
    }
  };
  const upload = (f?: File) => { if (f && !busy) run(f.name, () => api.upload(f)); };
  const unparsed = files.filter((f) => !f.parsed);

  return (
    <div className="page">
      <div className="pagehead">
        <h1>Matches</h1>
        {demos && demos.length > 0 && <span className="count">{demos.length}</span>}
      </div>
      {error && <div className="error">{error}</div>}
      {busy && <div className="notice loading">Parsing {busy}… about 10 seconds for a full match.</div>}

      {demos === null ? <p className="loading">Loading matches…</p> : (
        <div className="mapgrid">
          {demos.map((d) => (
            <a key={d.demo_id} className="mapcard cut" href={`#/demo/${d.demo_id}`}>
              <div className="art" style={{ backgroundImage: `url(${radarUrl(d.map)})` }} />
              <span className="rounds">{d.rounds_played} RDS</span>
              <div className="mapname">{mapLabel(d.map)}</div>
              <div className="file" title={d.source_name}>{d.source_name}</div>
            </a>
          ))}
          {demos.length === 0 && <div className="empty">No parsed demos yet. Parse one from your folder or upload a .dem below.</div>}
        </div>
      )}

      <div className="section homecols">
        <section className="panel">
          <div className="sechead">
            <h3>Demo folder</h3>
            {unparsed.length > 0 && <span className="aside muted small">{unparsed.length} not parsed</span>}
          </div>
          <div className="folder">{folder ?? "Set CS2_DEMO_DIR in .env to list a folder here."}</div>
          <ul className="filelist">
            {unparsed.map((f) => (
              <li key={f.name}>
                <span className="fname" title={f.name}>{f.name} <span className="dim">{f.size_mb} MB</span></span>
                <button disabled={!!busy} onClick={() => run(f.name, () => api.ingest(f.name))}>Parse</button>
              </li>
            ))}
            {files.length > 0 && unparsed.length === 0 && <li className="muted">Everything in the folder is parsed.</li>}
          </ul>
        </section>

        <section className="panel">
          <div className="sechead"><h3>Upload</h3></div>
          <label className={`drop ${over ? "over" : ""}`}
                 onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
                 onDrop={(e) => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files?.[0]); }}>
            <input type="file" accept=".dem" disabled={!!busy} onChange={(e) => upload(e.target.files?.[0])} />
            <span className="big">Drop a .dem here</span>
            <span className="muted small">or click to choose a file</span>
          </label>
          <p className="note">FACEIT and HLTV downloads come compressed (.dem.gz / .rar / .zst): extract the .dem first.</p>
        </section>
      </div>
    </div>
  );
}
