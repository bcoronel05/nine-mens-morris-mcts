"""Nine Men's Morris game engine.

Board layout (24 points, indices 0-23):

    0-----------1-----------2
    |           |           |
    |   3-------4-------5   |
    |   |       |       |   |
    |   |   6---7---8   |   |
    |   |   |       |   |   |
    9--10--11      12--13--14
    |   |   |       |   |   |
    |   |  15--16--17   |   |
    |   |       |       |   |
    |  18------19------20   |
    |           |           |
    21---------22----------23

Rules implemented:
  * Placing phase: each player places 9 pieces on empty points.
  * Moving phase: a piece slides to an adjacent empty point.
  * Flying: a player reduced to exactly 3 pieces may move anywhere.
  * Forming a mill (3 in a line) removes one enemy piece. Pieces inside a
    mill are protected unless every enemy piece is in a mill.
  * Loss: reduced to 2 pieces, or left with no legal move.
  * Draw: DRAW_PLY_LIMIT plies pass with no capture.

Actions are 3-tuples so a single format covers every move type:
    ("place",  None, to)      place a piece from hand onto `to`
    ("move",   frm,  to)      slide/fly a piece from `frm` to `to`
    ("capture", None, target) remove the enemy piece on `target`
"""

from __future__ import annotations

# --- board topology -------------------------------------------------------

ADJACENT: dict[int, tuple[int, ...]] = {
    0: (1, 9),
    1: (0, 2, 4),
    2: (1, 14),
    3: (4, 10),
    4: (1, 3, 5, 7),
    5: (4, 13),
    6: (7, 11),
    7: (4, 6, 8),
    8: (7, 12),
    9: (0, 10, 21),
    10: (3, 9, 11, 18),
    11: (6, 10, 15),
    12: (8, 13, 17),
    13: (5, 12, 14, 20),
    14: (2, 13, 23),
    15: (11, 16),
    16: (15, 17, 19),
    17: (12, 16),
    18: (10, 19),
    19: (16, 18, 20, 22),
    20: (13, 19),
    21: (9, 22),
    22: (19, 21, 23),
    23: (14, 22),
}

MILLS: tuple[tuple[int, int, int], ...] = (
    # horizontal
    (0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11),
    (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23),
    # vertical
    (0, 9, 21), (3, 10, 18), (6, 11, 15), (1, 4, 7),
    (16, 19, 22), (8, 12, 17), (5, 13, 20), (2, 14, 23),
)

# Mills indexed by the point they contain, so mill checks touch 2-3 lines
# instead of all 16. This is on the MCTS hot path.
MILLS_AT: tuple[tuple[tuple[int, int, int], ...], ...] = tuple(
    tuple(mill for mill in MILLS if point in mill) for point in range(24)
)

PIECES_PER_PLAYER = 9
DRAW_PLY_LIMIT = 100


class NineMensMorris:
    """A full game state. Cheap to copy, which is what MCTS needs."""

    __slots__ = (
        "board", "in_hand", "on_board", "cur",
        "pending_capture", "plies_since_capture",
    )

    def __init__(self) -> None:
        self.board = [0] * 24          # 0 = empty, 1 = player 1, 2 = player 2
        self.in_hand = {1: PIECES_PER_PLAYER, 2: PIECES_PER_PLAYER}
        self.on_board = {1: 0, 2: 0}
        self.cur = 1
        self.pending_capture = False
        self.plies_since_capture = 0

    # --- copying ----------------------------------------------------------

    def copy(self) -> "NineMensMorris":
        """Shallow-but-complete copy. Avoids deepcopy, which is ~40x slower
        and dominates the profile of any MCTS that uses it."""
        new = NineMensMorris.__new__(NineMensMorris)
        new.board = self.board[:]
        new.in_hand = {1: self.in_hand[1], 2: self.in_hand[2]}
        new.on_board = {1: self.on_board[1], 2: self.on_board[2]}
        new.cur = self.cur
        new.pending_capture = self.pending_capture
        new.plies_since_capture = self.plies_since_capture
        return new

    # --- mills ------------------------------------------------------------

    def forms_mill(self, point: int, player: int) -> bool:
        """True if `player` owning `point` completes a line."""
        board = self.board
        for mill in MILLS_AT[point]:
            if board[mill[0]] == player and board[mill[1]] == player and board[mill[2]] == player:
                return True
        return False

    def in_mill(self, point: int) -> bool:
        """True if the piece standing on `point` is part of a mill."""
        player = self.board[point]
        if player == 0:
            return False
        return self.forms_mill(point, player)

    def capture_targets(self, player: int) -> list[int]:
        """Enemy pieces `player` is allowed to remove.

        Pieces in mills are protected, unless every enemy piece is in a mill.
        """
        opponent = 3 - player
        pieces = [p for p in range(24) if self.board[p] == opponent]
        unprotected = [p for p in pieces if not self.in_mill(p)]
        return unprotected if unprotected else pieces

    # --- move generation --------------------------------------------------

    def legal_moves(self, player: int) -> list[tuple]:
        """All actions available to `player`. Empty list if it is not their
        turn, so an agent asking out of turn gets a safe answer."""
        if player != self.cur:
            return []

        if self.pending_capture:
            return [("capture", None, t) for t in self.capture_targets(player)]

        board = self.board
        if self.in_hand[player] > 0:
            return [("place", None, p) for p in range(24) if board[p] == 0]

        sources = [p for p in range(24) if board[p] == player]
        moves = []
        if self.on_board[player] == 3:          # flying phase
            empties = [p for p in range(24) if board[p] == 0]
            for src in sources:
                for dst in empties:
                    moves.append(("move", src, dst))
        else:
            for src in sources:
                for dst in ADJACENT[src]:
                    if board[dst] == 0:
                        moves.append(("move", src, dst))
        return moves

    # --- applying actions -------------------------------------------------

    def apply_move(self, action: tuple) -> None:
        """Apply a "place" or "move" action for the player to move.

        If the action closes a mill and the opponent has a removable piece,
        the turn does NOT pass: `pending_capture` becomes True and the same
        player must follow up with apply_capture().
        """
        kind, frm, to = action
        player = self.cur

        if kind == "place":
            if self.in_hand[player] <= 0:
                raise ValueError(f"player {player} has no pieces in hand")
            if self.board[to] != 0:
                raise ValueError(f"point {to} is occupied")
            self.board[to] = player
            self.in_hand[player] -= 1
            self.on_board[player] += 1
        elif kind == "move":
            if self.board[frm] != player:
                raise ValueError(f"point {frm} does not hold a player-{player} piece")
            if self.board[to] != 0:
                raise ValueError(f"point {to} is occupied")
            if self.on_board[player] != 3 and to not in ADJACENT[frm]:
                raise ValueError(f"{frm} -> {to} is not an adjacent slide")
            self.board[frm] = 0
            self.board[to] = player
        else:
            raise ValueError(f"apply_move got a {kind!r} action")

        self.plies_since_capture += 1

        if self.forms_mill(to, player) and self.capture_targets(player):
            self.pending_capture = True
            return

        self.cur = 3 - player

    def apply_capture(self, point: int) -> None:
        """Remove the enemy piece on `point`, then pass the turn."""
        if not self.pending_capture:
            raise ValueError("no capture is pending")
        player = self.cur
        opponent = 3 - player
        if self.board[point] != opponent:
            raise ValueError(f"point {point} does not hold a player-{opponent} piece")
        if point not in self.capture_targets(player):
            raise ValueError(f"point {point} is protected by a mill")

        self.board[point] = 0
        self.on_board[opponent] -= 1
        self.pending_capture = False
        self.plies_since_capture = 0
        self.cur = opponent

    def apply(self, action: tuple) -> None:
        """Dispatch on action type. Convenience wrapper for drivers."""
        if action[0] == "capture":
            self.apply_capture(action[2])
        else:
            self.apply_move(action)

    # --- terminal detection -----------------------------------------------

    def winner(self) -> int | None:
        """1 or 2 if that player has won, 0 for a draw, None if unfinished."""
        for player in (1, 2):
            if self.in_hand[player] == 0 and self.on_board[player] < 3:
                return 3 - player
        if self.plies_since_capture >= DRAW_PLY_LIMIT:
            return 0
        if not self.legal_moves(self.cur):
            return 3 - self.cur       # stalemated player loses
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None

    # --- display ----------------------------------------------------------

    def render(self) -> str:
        """ASCII board. '.' is empty, 'X' is player 1, 'O' is player 2."""
        sym = {0: ".", 1: "X", 2: "O"}
        b = [sym[v] for v in self.board]
        return (
            f"{b[0]}-----------{b[1]}-----------{b[2]}\n"
            f"|           |           |\n"
            f"|   {b[3]}-------{b[4]}-------{b[5]}   |\n"
            f"|   |       |       |   |\n"
            f"|   |   {b[6]}---{b[7]}---{b[8]}   |   |\n"
            f"|   |   |       |   |   |\n"
            f"{b[9]}--{b[10]}--{b[11]}       {b[12]}--{b[13]}--{b[14]}\n"
            f"|   |   |       |   |   |\n"
            f"|   |   {b[15]}---{b[16]}---{b[17]}   |   |\n"
            f"|   |       |       |   |\n"
            f"|   {b[18]}-------{b[19]}-------{b[20]}   |\n"
            f"|           |           |\n"
            f"{b[21]}-----------{b[22]}-----------{b[23]}\n"
            f"\nin hand  X:{self.in_hand[1]} O:{self.in_hand[2]}   "
            f"on board X:{self.on_board[1]} O:{self.on_board[2]}   "
            f"to move: {sym[self.cur]}"
            + ("  (capture pending)" if self.pending_capture else "")
        )

    def __repr__(self) -> str:
        return (
            f"<NineMensMorris cur={self.cur} "
            f"hand={self.in_hand} board={self.on_board} "
            f"pending_capture={self.pending_capture}>"
        )
