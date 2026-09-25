"""Engine rule tests. Run with: python -m unittest discover tests"""

import unittest

from src.game import ADJACENT, MILLS, NineMensMorris


class TestTopology(unittest.TestCase):
    def test_adjacency_is_symmetric(self):
        for point, neighbours in ADJACENT.items():
            for n in neighbours:
                self.assertIn(point, ADJACENT[n], f"{point}-{n} is one-way")

    def test_edge_count(self):
        edges = sum(len(v) for v in ADJACENT.values()) // 2
        self.assertEqual(edges, 32)

    def test_mill_count_and_membership(self):
        self.assertEqual(len(MILLS), 16)
        # every point sits on exactly two mills, one horizontal one vertical
        for point in range(24):
            self.assertEqual(sum(point in mill for mill in MILLS), 2)


class TestPlacing(unittest.TestCase):
    def test_opening_moves(self):
        s = NineMensMorris()
        moves = s.legal_moves(1)
        self.assertEqual(len(moves), 24)
        self.assertTrue(all(m[0] == "place" for m in moves))

    def test_out_of_turn_returns_nothing(self):
        s = NineMensMorris()
        self.assertEqual(s.legal_moves(2), [])

    def test_place_updates_counts_and_passes_turn(self):
        s = NineMensMorris()
        s.apply_move(("place", None, 0))
        self.assertEqual(s.board[0], 1)
        self.assertEqual(s.in_hand[1], 8)
        self.assertEqual(s.on_board[1], 1)
        self.assertEqual(s.cur, 2)

    def test_cannot_place_on_occupied_point(self):
        s = NineMensMorris()
        s.apply_move(("place", None, 0))
        with self.assertRaises(ValueError):
            s.apply_move(("place", None, 0))


class TestMills(unittest.TestCase):
    def test_closing_a_mill_holds_the_turn(self):
        s = NineMensMorris()
        for action in [("place", None, 0), ("place", None, 3),
                       ("place", None, 1), ("place", None, 4)]:
            s.apply_move(action)
        self.assertFalse(s.pending_capture)
        s.apply_move(("place", None, 2))          # completes 0-1-2
        self.assertTrue(s.pending_capture)
        self.assertEqual(s.cur, 1)
        self.assertTrue(all(m[0] == "capture" for m in s.legal_moves(1)))

    def test_capture_passes_the_turn_and_resets_the_clock(self):
        s = NineMensMorris()
        for action in [("place", None, 0), ("place", None, 3),
                       ("place", None, 1), ("place", None, 4),
                       ("place", None, 2)]:
            s.apply_move(action)
        s.apply_capture(3)
        self.assertEqual(s.board[3], 0)
        self.assertEqual(s.on_board[2], 1)
        self.assertEqual(s.cur, 2)
        self.assertEqual(s.plies_since_capture, 0)
        self.assertFalse(s.pending_capture)

    def test_mill_pieces_are_protected_unless_all_are_in_mills(self):
        s = NineMensMorris()
        s.board = [0] * 24
        s.board[3] = s.board[4] = s.board[5] = 2   # enemy mill
        s.board[10] = 2                            # loose enemy piece
        s.board[0] = s.board[1] = 1
        s.in_hand = {1: 0, 2: 0}
        s.on_board = {1: 2, 2: 4}
        s.cur = 1
        self.assertEqual(s.capture_targets(1), [10])

        s.board[10] = 0
        s.on_board[2] = 3
        self.assertEqual(sorted(s.capture_targets(1)), [3, 4, 5])


class TestMovingAndFlying(unittest.TestCase):
    def test_moves_are_adjacent_only(self):
        s = NineMensMorris()
        s.board = [0] * 24
        s.board[0] = 1
        s.board[4] = s.board[5] = s.board[13] = 1
        s.board[21] = s.board[22] = s.board[23] = 2
        s.in_hand = {1: 0, 2: 0}
        s.on_board = {1: 4, 2: 3}
        s.cur = 1
        destinations = {m[2] for m in s.legal_moves(1) if m[1] == 0}
        self.assertEqual(destinations, {1, 9})

    def test_three_pieces_may_fly(self):
        s = NineMensMorris()
        s.board = [0] * 24
        s.board[0] = s.board[1] = s.board[2] = 1
        s.board[21] = s.board[22] = s.board[23] = 2
        s.in_hand = {1: 0, 2: 0}
        s.on_board = {1: 3, 2: 3}
        s.cur = 1
        destinations = {m[2] for m in s.legal_moves(1) if m[1] == 0}
        self.assertEqual(len(destinations), 18)   # every empty point


class TestTerminal(unittest.TestCase):
    def test_two_pieces_loses(self):
        s = NineMensMorris()
        s.board = [0] * 24
        s.board[0] = s.board[1] = 1
        s.board[21] = s.board[22] = s.board[23] = 2
        s.in_hand = {1: 0, 2: 0}
        s.on_board = {1: 2, 2: 3}
        s.cur = 1
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.winner(), 2)

    def test_no_legal_moves_loses(self):
        s = NineMensMorris()
        s.board = [0] * 24
        # Player 1 holds 0, 1, 2, 14 and every exit is occupied. Four pieces,
        # so flying does not apply and the player is genuinely stuck.
        s.board[0] = s.board[1] = s.board[2] = s.board[14] = 1
        s.board[9] = s.board[4] = s.board[13] = s.board[23] = 2
        s.in_hand = {1: 0, 2: 0}
        s.on_board = {1: 4, 2: 4}
        s.cur = 1
        self.assertEqual(s.legal_moves(1), [])
        self.assertEqual(s.winner(), 2)

    def test_quiet_game_is_a_draw(self):
        s = NineMensMorris()
        s.board = [0] * 24
        s.board[0] = s.board[1] = s.board[9] = 1
        s.board[21] = s.board[22] = s.board[23] = 2
        s.in_hand = {1: 0, 2: 0}
        s.on_board = {1: 3, 2: 3}
        s.cur = 1
        s.plies_since_capture = 100
        self.assertEqual(s.winner(), 0)


class TestCopy(unittest.TestCase):
    def test_copy_is_independent(self):
        s = NineMensMorris()
        s.apply_move(("place", None, 0))
        clone = s.copy()
        clone.apply_move(("place", None, 5))
        self.assertEqual(s.board[5], 0)
        self.assertEqual(s.in_hand[2], 9)
        self.assertEqual(clone.in_hand[2], 8)


if __name__ == "__main__":
    unittest.main()
