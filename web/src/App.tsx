import { useEffect, useState } from "react";
import Home from "./Home";
import Match from "./Match";
import RoundView from "./RoundView";

// Tiny hash router: #/  ·  #/demo/<id>  ·  #/demo/<id>/round/<n>  ·  #/demo/<id>/round/<n>/at/<tick>
type Route = { page: "home" } | { page: "match"; id: string } | { page: "round"; id: string; n: number; at?: number };
function parse(hash: string): Route {
  const m = hash.match(/^#\/demo\/([^/]+)(?:\/round\/(\d+)(?:\/at\/(\d+))?)?/);
  if (!m) return { page: "home" };
  if (!m[2]) return { page: "match", id: m[1] };
  return { page: "round", id: m[1], n: Number(m[2]), at: m[3] ? Number(m[3]) : undefined };
}

export default function App() {
  const [route, setRoute] = useState(parse(location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(location.hash));
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return (
    <div className="app">
      <header className="topbar">
        <a href="#/" className="brand">cs2-analyst</a>
        <span className="tagline">demos · swing · win probability · 2D replay</span>
      </header>
      <main>
        {route.page === "home" && <Home />}
        {route.page === "match" && <Match id={route.id} />}
        {route.page === "round" && <RoundView id={route.id} n={route.n} at={route.at} key={`${route.id}-${route.n}-${route.at ?? ""}`} />}
      </main>
    </div>
  );
}
