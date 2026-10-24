import { useEffect, useState } from "react";
import { api, DemoSummary } from "./api";

export default function Home() {
  const [demos, setDemos] = useState<DemoSummary[]>([]);
  const [files, setFiles] = useState<{ name: string; size_mb: number; parsed: boolean }[]>([]);
  const [folder, setFolder] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    api.demos().then(setDemos).catch((e) => setError(e.message));
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

  return (
    <div className="page">
      <h1>Matches</h1>
      {error && <div className="error">{error}</div>}
      {busy && <div className="notice">Parsing {busy}… (about 10 seconds for a full match)</div>}
      <div className="cards">
        {demos.map((d) => (
          <a key={d.demo_id} className="card" href={`#/demo/${d.demo_id}`}>
            <div className="card-map">{d.map?.replace("de_", "")}</div>
            <div className="card-file">{d.source_name}</div>
            <div className="muted">{d.rounds_played} rounds</div>
          </a>
        ))}
        {demos.length === 0 && <p className="muted">No parsed demos yet. Parse one below.</p>}
      </div>

      <h2>Demos in your folder</h2>
      <p className="muted">{folder ? folder : "Set CS2_DEMO_DIR in .env to list a folder here."}</p>
      <ul className="filelist">
        {files.filter((f) => !f.parsed).map((f) => (
          <li key={f.name}>
            <span>{f.name} <span className="muted">({f.size_mb} MB)</span></span>
            <button disabled={!!busy} onClick={() => run(f.name, () => api.ingest(f.name))}>Parse</button>
          </li>
        ))}
        {files.length > 0 && files.every((f) => f.parsed) && <li className="muted">Everything in the folder is parsed.</li>}
      </ul>

      <h2>Upload a demo</h2>
      <label className="upload">
        <input type="file" accept=".dem" disabled={!!busy}
               onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f.name, () => api.upload(f)); }} />
        <span>Choose a .dem file</span>
      </label>
      <p className="muted">FACEIT and HLTV downloads come compressed (.dem.gz / .rar / .zst): extract the .dem first.</p>
    </div>
  );
}
