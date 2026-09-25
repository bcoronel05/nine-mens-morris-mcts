"""Uniform-random agent. The floor that any search agent must clear."""

from __future__ import annotations

import random


class RandomAgent:
    """Picks uniformly among legal actions. No search, no state."""

    def __init__(self, agent_num: int, name: str = "Random"):
        self.name = name
        self.agent_number = agent_num
        self.number_of_wins = 0

    def choose_move(self, state, move_choice_dict) -> None:
        moves = state.legal_moves(self.agent_number)
        if not moves:
            return None
        move_choice_dict["move_choice"] = random.choice(moves)

    def choose_capture(self, state, capture_choice_dict) -> None:
        captures = [m for m in state.legal_moves(self.agent_number) if m[0] == "capture"]
        if not captures:
            return None
        capture_choice_dict["capture_choice"] = random.choice(captures)[2]
