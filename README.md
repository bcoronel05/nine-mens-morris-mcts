# Nine Men's Morris — MCTS Agent

A Monte Carlo Tree Search agent for Nine Men's Morris, with a from-scratch
game engine, a benchmark harness and a terminal client. Pure Python 3.10+,
no dependencies.

The agent was written for a course tournament, where each entry gets a fixed
wall-clock budget per action. This repository packages that agent with the
engine and tooling needed to actually run and measure it.

```
git clone https://github.com/<user>/nine-mens-morris-mcts.git
cd nine-mens-morris-mcts
python play.py
```

## What the agent does

Textbook UCT, no learned components:

| Stage | Implementation |
|---|---|
| Selection | UCB1, `wins/visits + √2 · √(ln N / n)`, descending while the node is fully expanded |
| Expansion | one random untried action popped per iteration |
| Simulation | uniform-random playout capped at 25 plies |
| Evaluation | terminal result if reached, otherwise a material-balance heuristic clipped to [0, 1] |
| Backpropagation | value stored from the perspective of the player who moved into each node |
| Action choice | root child with the highest visit count |

The search runs until its wall-clock budget is spent rather than to a fixed
iteration count, so it degrades gracefully on slower hardware.

Two details worth calling out, because they are the usual sources of silent
bugs in two-player MCTS:

**Perspective on backpropagation.** Each node stores its value from the point
of view of the player who *moved into* it, read from `parent.state.cur`
rather than assumed from alternation. Nine Men's Morris does not alternate
strictly — closing a mill keeps the turn for a follow-up capture — so
assuming `player = depth % 2` would silently invert values on capture nodes.
Reading the mover from the parent state handles this for free, and lets
`uct_select_child` use a plain `max()`.

**Heuristic at the horizon.** Random playouts rarely reach a terminal state
within 25 plies from a midgame position, so most rollouts return the material
heuristic instead. The value is `0.5 + 0.05 · (my pieces − their pieces)`,
counting both hand and board. It is deliberately weak: with a small
coefficient it breaks ties toward material without overwhelming the parts of
the value estimate that come from real terminal results.

## Benchmarks

Measured with `benchmark.py`, seats alternating every game, seed 1.
These come from a single run on one machine; `sims/second` is hardware
dependent, so re-run rather than quoting these numbers.

| Opponent | Budget | Games | Win | Loss | Draw | sims/s |
|---|---|---|---|---|---|---|
| Uniform random | 0.2 s/action | 30 | 100.0% | 0.0% | 0.0% | ~10,000 |
| MCTS @ 0.02 s/action | 0.2 s/action | 20 | 90.0% | 10.0% | 0.0% | ~9,400 |

Reproduce:

```bash
python benchmark.py --games 30 --time-limit 0.2 --opponent random --seed 1
python benchmark.py --games 20 --time-limit 0.2 --opponent mcts --baseline-time 0.02 --seed 1
```

Beating uniform random is a floor, not an achievement — it confirms the
search is wired up correctly, nothing more. The 10× budget matchup is the
more informative number: it says roughly how much strength this
implementation buys per unit of extra thinking time, and a 90% result there
means throughput is translating into play quality rather than being wasted
on a broken selection rule.

## Layout

```
src/
  game.py              engine: topology, move generation, mills, terminal detection
  driver.py            runs one game between two agents
  agents/
    mcts_agent.py      the UCT agent
    random_agent.py    uniform-random baseline
benchmark.py           match runner, reports win rate and throughput
play.py                human vs agent in the terminal
tests/test_game.py     engine rule tests
```

## The engine

24 points, 32 edges, 16 mills. Placing, moving, flying at three pieces, mill
capture with the standard protection rule, loss on two pieces or on being
stuck, and a draw after 100 plies without a capture.

`copy()` is hand-written rather than `copy.deepcopy`, which matters more than
it looks: MCTS copies a state on every expansion and every rollout step, and
`deepcopy` on this structure is roughly 40× slower than copying a 24-element
list and two small dicts. States use `__slots__` for the same reason.

Agents talk to the driver through a callback protocol carried over from the
tournament platform — `choose_move(state, out_dict)` and
`choose_capture(state, out_dict)` — and receive a copy of the state, so an
agent cannot mutate the real game. The driver validates every returned action
and substitutes a legal one if an agent returns garbage.

Run the tests:

```bash
python -m unittest discover -s tests -t .
```

## Known limits

- No transposition table. Positions reachable by different move orders are
  searched as separate nodes.
- The tree is discarded between moves instead of reusing the subtree under
  the chosen action, which throws away most of the previous search.
- Rollouts are uniform-random with no playout policy, so the value estimates
  are noisy and the 25-ply cap means most of them fall through to the
  material heuristic.
- The heuristic ignores mills, mobility and blocked pieces — all of which
  matter more than raw material in the midgame.
- `choose_capture` runs a full search from the capture state, spending a
  second budget on what is usually a low-branching decision.

## License

MIT
