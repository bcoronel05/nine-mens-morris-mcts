"""Game driver.

Runs one game between two agents using the same callback protocol the
course tournament used: the driver owns the state, hands a read-only view
to the agent, and reads the agent's choice out of a mutable dict.
"""

from __future__ import annotations

from .game import NineMensMorris

MAX_PLIES = 400


def play_game(agent1, agent2, verbose: bool = False) -> int:
    """Play one game. Returns 1, 2 or 0 (draw)."""
    state = NineMensMorris()
    agents = {1: agent1, 2: agent2}

    for _ in range(MAX_PLIES):
        result = state.winner()
        if result is not None:
            if verbose:
                print(state.render())
                print(f"result: {'draw' if result == 0 else f'player {result} wins'}")
            return result

        agent = agents[state.cur]

        if state.pending_capture:
            choice: dict = {}
            agent.choose_capture(state.copy(), choice)
            target = choice.get("capture_choice")
            legal = state.capture_targets(state.cur)
            if target not in legal:
                target = legal[0]          # driver repairs an illegal choice
            state.apply_capture(target)
        else:
            choice = {}
            agent.choose_move(state.copy(), choice)
            action = choice.get("move_choice")
            legal = state.legal_moves(state.cur)
            if action not in legal:
                action = legal[0]
            state.apply_move(action)

        if verbose:
            print(state.render())
            print("-" * 30)

    return 0   # ply cap reached, score as a draw
