// Typed wrappers around the Python backend (src/cs2_analyst/web/app.py).

export type Side = "CT" | "T";

export interface DemoSummary {
  demo_id: string;
  map: string;
  rounds_played: number;
  source_name: string;
  ingested_at: string;
}

export interface ScoreRow {
  name: string;
  team: string;
  rating: number;
  team_rank: number;
  match_rank: number;
  kills: number;
  deaths: number;
  assists: number;
  adr: number;
  kast: number;
  hs_pct: number;
  opening_kills: number;
  opening_deaths: number;
  clutches_won: number;
  clutches_played: number;
  k3: number;
  k4: number;
  aces: number;
}

export interface Overview {
  map: string;
  rounds_played: number;
  score: Record<string, { rounds_won: number; ct_rounds_won: number; t_rounds_won: number }>;
  scoreboard: ScoreRow[];
  multikills: { rounds: { round: number; player: string; kills: number; type: string; victims: string[] }[] };
}

export interface SwingPlayer {
  name: string;
  team: string;
  swing_per_round_pp: number;
  swing_rank: number;
  rating_rank?: number;
  biggest_plays: { round: number; tick: number; what: string; other: string | null; swing_pp: number }[];
  costliest_moments: { round: number; tick: number; what: string; other: string | null; swing_pp: number }[];
}

export interface RoundRow {
  round: number;
  winner: string;
  winner_side: Side;
  reason: string;
  first_kill_team: string | null;
  bomb_planted: string | null;
  teams: { team: string; side: Side; buy: string | null; equipment: number | null }[];
  score_after: Record<string, number>;
}

export interface RoundList {
  final_score: Record<string, number>;
  halves: { half: number; rounds: string; overtime: boolean; sides: Record<string, Side>; rounds_won: Record<string, number> }[];
  rounds: RoundRow[];
}

export interface WinStep {
  step: number;
  tick: number;
  time_s: number;
  event: string;
  actor: string | null;
  actor_side: Side | null;
  victim: string | null;
  alive: string;
  p_t_win: number;
  swing_t_pp: number;
  why: { reason: string }[];
}

export interface WinProb {
  round: number;
  winner_side: Side;
  steps: WinStep[];
  biggest_swing: WinStep | null;
}

// one frame: [x, y, yaw, hp, weapon, has_bomb, defusing, flashed_s]
export type Frame = [number, number, number, number, string | null, boolean, boolean, number] | null;

export interface Replay {
  round: number;
  map: string;
  tick_rate: number;
  start_tick: number;
  end_tick: number;
  winner_side: Side;
  reason: string;
  ticks: number[];
  players: { steamid: string; name: string; side: Side; frames: Frame[] }[];
  kills: {
    tick: number;
    attacker: string | null;
    attacker_side: Side | null;
    victim: string;
    victim_side: Side;
    weapon: string;
    headshot: boolean;
    victim_xy: [number, number];
    attacker_xy: [number | null, number | null];
  }[];
  smokes: { start: number; end: number; xy: [number, number] }[];
  fires: { start: number; end: number; xy: [number, number] }[];
  bomb: { tick: number; event: string; name: string; xy: [number, number] | null }[];
  radar_url: string;
  units_per_px: number;
}

export interface Cs2Info {
  game_dir: string | null;
  prepared: boolean;
  commands: string[];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail ?? `${r.status} ${r.statusText}`);
  return body as T;
}

export const api = {
  demos: () => req<DemoSummary[]>("/api/demos"),
  files: () => req<{ demo_dir: string | null; files: { name: string; size_mb: number; parsed: boolean }[] }>("/api/files"),
  ingest: (file: string) =>
    req<DemoSummary>("/api/ingest", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file }) }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<DemoSummary>("/api/upload", { method: "POST", body: fd });
  },
  overview: (id: string) => req<Overview>(`/api/demos/${id}/overview`),
  rounds: (id: string) => req<RoundList>(`/api/demos/${id}/rounds`),
  swing: (id: string) => req<{ players: SwingPlayer[]; note: string }>(`/api/demos/${id}/swing`),
  model: () => req<{ trained: boolean; message?: string; demos?: number; rounds?: number }>("/api/model"),
  winprob: (id: string, n: number) => req<WinProb>(`/api/demos/${id}/rounds/${n}/winprob`),
  replay: (id: string, n: number) => req<Replay>(`/api/demos/${id}/rounds/${n}/replay`),
  cs2: (id: string, tick?: number) => req<Cs2Info>(`/api/demos/${id}/cs2${tick !== undefined ? `?tick=${tick}` : ""}`),
  cs2Prepare: (id: string) => req<{ prepared: boolean; path: string }>(`/api/demos/${id}/cs2/prepare`, { method: "POST" }),
};
