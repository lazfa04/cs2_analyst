import { useEffect, useState } from "react";
import Home from "./Home";
import Match, { Tab, TABS } from "./Match";
import RoundView from "./RoundView";

// Tiny hash router: #/  ·  #/demo/<id>[/<tab>]  ·  #/demo/<id>/round/<n>  ·  #/demo/<id>/round/<n>/at/<tick>
type Route = { page: "home" } | { page: "match"; id: string; tab?: Tab } | { page: "round"; id: string; n: number; at?: number };
function parse(hash: string): Route {
  const m = hash.match(/^#\/demo\/([^/]+)(?:\/round\/(\d+)(?:\/at\/(\d+))?|\/([a-z]+))?/);
  if (!m) return { page: "home" };
  if (m[2]) return { page: "round", id: m[1], n: Number(m[2]), at: m[3] ? Number(m[3]) : undefined };
  return { page: "match", id: m[1], tab: TABS.find((t) => t.id === m[4])?.id };
}

export default function App() {
  const [route, setRoute] = useState(parse(location.hash));
  useEffect(() => {
    const on = () => { setRoute(parse(location.hash)); };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  useEffect(() => { scrollTo(0, 0); }, [route.page, route.page !== "home" ? route.id : ""]);
  return (
    <div className="app">
      <header className="topbar">
        <a href="#/" className="brand">
          <span className="mark"><i /><i /></span>CS2<span className="slash">/</span>Analyst
        </a>
        <span className="tagline">demos · swing · win probability · 2D replay</span>
        <nav><a href="#/">Matches</a></nav>
      </header>
      <main>
        {route.page === "home" && <Home />}
        {route.page === "match" && <Match id={route.id} tab={route.tab} key={route.id} />}
        {route.page === "round" && <RoundView id={route.id} n={route.n} at={route.at} key={`${route.id}-${route.n}-${route.at ?? ""}`} />}
      </main>
    </div>
  );
}
