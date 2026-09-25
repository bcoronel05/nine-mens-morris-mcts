"""UCT-based Monte Carlo Tree Search agent for Nine Men's Morris.

Originally written for a course tournament, where each agent gets a fixed
wall-clock budget per action and the driver calls back into the agent
through `choose_move` / `choose_capture`.

The search is textbook UCT:

  1. Selection    descend via UCB1 while the node is fully expanded
  2. Expansion    pop one untried action, create the child
  3. Simulation   random playout, capped at ROLLOUT_DEPTH plies, falling
                  back to a material heuristic if no terminal state is hit
  4. Backprop     walk to the root, storing each node's value from the
                  point of view of the player who moved into it

There is no neural network here and no learned prior: node values come
from random rollouts, and the exploration term is plain UCB1.
"""

from __future__ import annotations

import math
import random
import time

ROLLOUT_DEPTH = 25          # plies simulated before the heuristic takes over
EXPLORATION = math.sqrt(2)  # UCB1 constant
MATERIAL_WEIGHT = 0.05      # value per piece of material advantage


class MCTSNode:
    """One node of the search tree.

    `wins` is accumulated from the perspective of the player who made
    `parent_action`, so a parent can select among its children with a plain
    max() instead of negating at selection time.
    """

    __slots__ = ("state", "parent", "parent_action", "children",
                 "wins", "visits", "untried_actions")

    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_actions = state.legal_moves(state.cur)

    def uct_select_child(self) -> "MCTSNode":
        log_parent = math.log(self.visits)
        return max(
            self.children,
            key=lambda c: c.wins / c.visits + EXPLORATION * math.sqrt(log_parent / c.visits),
        )

    def expand(self) -> "MCTSNode":
        action = self.untried_actions.pop(random.randrange(len(self.untried_actions)))
        next_state = self.state.copy()
        if action[0] == "capture":
            next_state.apply_capture(action[2])
        else:
            next_state.apply_move(action)

        child = MCTSNode(next_state, self, action)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        self.visits += 1
        self.wins += result


class MCTSAgent:
    """Search agent driven by a wall-clock budget per action."""

    def __init__(self, agent_num: int, name: str = "MCTS", time_limit: float = 1.0):
        self.name = name
        self.agent_number = agent_num
        self.time_limit = time_limit
        self.number_of_wins = 0
        self.total_simulations = 0   # instrumentation, for honest sims/sec
        self.total_search_time = 0.0

    # --- rollout ----------------------------------------------------------

    def sim_playout(self, state) -> float:
        """Random playout from `state`, returning a value in [0, 1] from
        this agent's point of view."""
        current_state = state.copy()
        for _ in range(ROLLOUT_DEPTH):
            if current_state.is_terminal():
                break
            moves = current_state.legal_moves(current_state.cur)
            if not moves:
                break
            action = random.choice(moves)
            if action[0] == "capture":
                current_state.apply_capture(action[2])
            else:
                current_state.apply_move(action)

        winner = current_state.winner()
        if winner == self.agent_number:
            return 1.0
        if winner == 0:
            return 0.5               # draw
        if winner is not None:
            return 0.0

        # No terminal state inside the horizon: fall back to material count.
        opponent = 3 - self.agent_number
        mine = (current_state.on_board.get(self.agent_number, 0)
                + current_state.in_hand.get(self.agent_number, 0))
        theirs = (current_state.on_board.get(opponent, 0)
                  + current_state.in_hand.get(opponent, 0))
        score = 0.5 + (mine - theirs) * MATERIAL_WEIGHT
        return max(0.0, min(1.0, score))

    # --- search -----------------------------------------------------------

    def get_best_action(self, state):
        """Run UCT until the time budget is spent, then return the root
        child with the most visits."""
        root = MCTSNode(state)
        start_time = time.time()
        simulations = 0

        while time.time() - start_time < self.time_limit:
            node = root

            # 1. selection
            while not node.untried_actions and node.children:
                node = node.uct_select_child()

            # 2. expansion
            if node.untried_actions:
                node = node.expand()

            # 3. simulation
            result = self.sim_playout(node.state)
            simulations += 1

            # 4. backpropagation
            curr = node
            while curr is not None:
                if curr.parent is None:
                    curr.update(result)
                else:
                    player_who_moved = curr.parent.state.cur
                    if player_who_moved == self.agent_number:
                        curr.update(result)
                    else:
                        curr.update(1.0 - result)
                curr = curr.parent

        self.total_simulations += simulations
        self.total_search_time += time.time() - start_time

        if not root.children:
            legal = state.legal_moves(self.agent_number)
            return random.choice(legal) if legal else None
        return max(root.children, key=lambda c: c.visits).parent_action

    # --- driver interface -------------------------------------------------

    def choose_move(self, state, move_choice_dict) -> None:
        moves = state.legal_moves(self.agent_number)
        if not moves:
            return None
        best_action = self.get_best_action(state)
        move_choice_dict["move_choice"] = best_action or random.choice(moves)

    def choose_capture(self, state, capture_choice_dict) -> None:
        """Pick a piece to remove after closing a mill.

        The driver only calls this when a capture is pending, so every legal
        action here is a capture; the fallback stays inside that set.
        """
        captures = [m for m in state.legal_moves(self.agent_number) if m[0] == "capture"]
        if not captures:
            return None

        best_action = self.get_best_action(state)
        if best_action and best_action[0] == "capture":
            capture_choice_dict["capture_choice"] = best_action[2]
        else:
            capture_choice_dict["capture_choice"] = random.choice(captures)[2]

    # --- stats ------------------------------------------------------------

    @property
    def simulations_per_second(self) -> float:
        if self.total_search_time <= 0:
            return 0.0
        return self.total_simulations / self.total_search_time
