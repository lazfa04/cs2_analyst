import { useEffect, useState } from "react";
import Home from "./Home";
import Match from "./Match";
import RoundView from "./RoundView";

// Tiny hash router: #/  ·  #/demo/<id>  ·  #/demo/<id>/round/<n>
function parse(hash: string): { page: "home" } | { page: "match"; id: string } | { page: "round"; id: string; n: number } {
  const m = hash.match(/^#\/demo\/([^/]+)(?:\/round\/(\d+))?/);
  if (!m) return { page: "home" };
  return m[2] ? { page: "round", id: m[1], n: Number(m[2]) } : { page: "match", id: m[1] };
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
        {route.page === "round" && <RoundView id={route.id} n={route.n} key={`${route.id}-${route.n}`} />}
      </main>
    </div>
  );
}
