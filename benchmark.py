#!/usr/bin/env python3
"""Measure the MCTS agent against a baseline over a match of N games.

Seats alternate every game so neither agent keeps the first-move advantage,
and the RNG is seeded so a run can be reproduced.

    python benchmark.py --games 100 --time-limit 1.0
    python benchmark.py --opponent mcts --games 40 --baseline-time 0.1
"""

from __future__ import annotations

import argparse
import random
import time

from src.agents import MCTSAgent, RandomAgent
from src.driver import play_game


def build(kind: str, seat: int, time_limit: float):
    if kind == "random":
        return RandomAgent(seat)
    if kind == "mcts":
        return MCTSAgent(seat, name="MCTS-baseline", time_limit=time_limit)
    raise ValueError(f"unknown agent kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--time-limit", type=float, default=1.0,
                        help="seconds per action for the agent under test")
    parser.add_argument("--opponent", choices=["random", "mcts"], default="random")
    parser.add_argument("--baseline-time", type=float, default=0.1,
                        help="seconds per action for an MCTS opponent")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    wins = losses = draws = 0
    sims = 0
    search_time = 0.0
    started = time.time()

    for game in range(args.games):
        seat = 1 if game % 2 == 0 else 2
        agent = MCTSAgent(seat, time_limit=args.time_limit)
        opponent = build(args.opponent, 3 - seat, args.baseline_time)

        p1, p2 = (agent, opponent) if seat == 1 else (opponent, agent)
        result = play_game(p1, p2)

        if result == seat:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

        sims += agent.total_simulations
        search_time += agent.total_search_time

        print(f"  game {game + 1:>3}/{args.games}  "
              f"seat {seat}  ->  W{wins} L{losses} D{draws}", flush=True)

    played = wins + losses + draws
    print()
    print(f"MCTS ({args.time_limit}s/action) vs {args.opponent}")
    print(f"  games        {played}")
    print(f"  wins         {wins}  ({wins / played:.1%})")
    print(f"  losses       {losses}  ({losses / played:.1%})")
    print(f"  draws        {draws}  ({draws / played:.1%})")
    print(f"  sims/second  {sims / search_time:,.0f}")
    print(f"  total sims   {sims:,}")
    print(f"  wall clock   {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
