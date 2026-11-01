import { Fragment } from "react";
import { Overview, RoundList } from "./api";

type Multi = Overview["multikills"]["rounds"];

export const halfStarts = (rounds: RoundList) => new Set(rounds.halves.slice(1).map((h) => Number(h.rounds.split("-")[0])));

// One tile per round, coloured by the winning side, with a gap at each half. Shared by the match hero and the round page.
export default function Timeline({ id, rounds, multikills = [], current }: { id: string; rounds: RoundList; multikills?: Multi; current?: number }) {
  const starts = halfStarts(rounds);
  return (
    <div className="timeline">
      {rounds.rounds.map((r) => {
        const winnerBuy = r.teams.find((t) => t.team === r.winner)?.buy;
        const multi = multikills.filter((m) => m.round === r.round);
        return (
          <Fragment key={r.round}>
            {starts.has(r.round) && <span className="gap" />}
            <a href={`#/demo/${id}/round/${r.round}`}
               className={`rt ${r.winner_side === "CT" ? "ct" : "t"} ${current === r.round ? "cur" : ""}`}
               title={`Round ${r.round}: ${r.winner} (${r.winner_side}) won, ${r.reason.replace(/_/g, " ")}. Buys: ${r.teams.map((t) => `${t.team} ${t.buy}`).join(", ")}${multi.length ? ". " + multi.map((m) => `${m.player} ${m.type}`).join(", ") : ""}`}>
              {r.round}
              {winnerBuy && winnerBuy !== "full" && <span className="buy">{winnerBuy}</span>}
              {r.bomb_planted && <span className="pin" />}
              {multi.some((m) => m.kills >= 4) && <span className="star">★</span>}
            </a>
          </Fragment>
        );
      })}
    </div>
  );
}
