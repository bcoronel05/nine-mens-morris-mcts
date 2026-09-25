#!/usr/bin/env python3
"""Play a game against the MCTS agent in the terminal.

    python play.py                 # you are X, agent thinks 1s per move
    python play.py --time-limit 3  # give the agent a longer budget
"""

from __future__ import annotations

import argparse

from src.agents import MCTSAgent
from src.game import NineMensMorris


def ask(prompt: str, valid: list[int]) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) in valid:
            return int(raw)
        print(f"  pick one of: {sorted(valid)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=1.0)
    args = parser.parse_args()

    state = NineMensMorris()
    agent = MCTSAgent(2, time_limit=args.time_limit)
    print("You are X (player 1). The agent is O (player 2).\n")

    while not state.is_terminal():
        print(state.render(), "\n")

        if state.cur == 1:
            if state.pending_capture:
                targets = state.capture_targets(1)
                print(f"You closed a mill. Removable: {sorted(targets)}")
                state.apply_capture(ask("remove > ", targets))
            else:
                moves = state.legal_moves(1)
                if moves[0][0] == "place":
                    state.apply_move(("place", None, ask("place > ", [m[2] for m in moves])))
                else:
                    sources = sorted({m[1] for m in moves})
                    src = ask(f"move from {sources} > ", sources)
                    dests = sorted(m[2] for m in moves if m[1] == src)
                    state.apply_move(("move", src, ask(f"       to {dests} > ", dests)))
        else:
            choice: dict = {}
            if state.pending_capture:
                agent.choose_capture(state.copy(), choice)
                target = choice.get("capture_choice")
                if target not in state.capture_targets(2):
                    target = state.capture_targets(2)[0]
                print(f"agent removes {target}\n")
                state.apply_capture(target)
            else:
                agent.choose_move(state.copy(), choice)
                action = choice.get("move_choice")
                if action not in state.legal_moves(2):
                    action = state.legal_moves(2)[0]
                print(f"agent plays {action}\n")
                state.apply_move(action)

    print(state.render(), "\n")
    result = state.winner()
    print("draw" if result == 0 else ("you win" if result == 1 else "agent wins"))
    if agent.total_search_time > 0:
        print(f"agent averaged {agent.simulations_per_second:,.0f} simulations/second")


if __name__ == "__main__":
    main()
