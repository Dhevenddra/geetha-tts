"""Exact edit distance from a hypothesis to a reference *lattice*.

A dual-script reference is not one string, it is a small lattice: literal stretches with
parallel branches wherever a span has more than one legitimate reading::

    {ஓகே|okay} {மேம்|ma'am} {டூ|two} நிமிஷம்

Scoring means finding the cheapest alignment between the hypothesis and *any* path
through that lattice. Two obvious approaches both fail:

* **Expand every path and score each.** Exact, but real DiarBench segments carry
  fourteen or more code-mixed words, which is 16,384 paths.
* **Resolve each site greedily, one at a time.** Linear, but it lands in local minima --
  measured on random small references it reported 5 errors where the true answer was 3,
  because the best reading of one site depends on how its neighbours aligned.

So this module does the standard thing instead: Levenshtein over a DAG. The lattice is
built with node ids increasing along every edge, which makes the node order
topological, and the usual edit-distance recurrence is relaxed over
(hypothesis position, lattice node). Exact, and O(len(hypothesis) x edges).

The cheapest path is reconstructed afterwards, so the caller learns which reading each
site was actually scored against -- required for the released raw outputs.
"""

from __future__ import annotations

from typing import Callable, Sequence

from geetha.scoring.metrics import ErrorCounts

__all__ = ["Lattice", "build_lattice", "lattice_edit_counts"]

_INF = float("inf")


class Lattice:
    """A single-source, single-sink DAG whose edges carry reference tokens.

    An edge token of ``None`` is an epsilon edge: it joins branches back together and
    consumes nothing.
    """

    __slots__ = ("node_count", "sink", "incoming")

    def __init__(self, node_count: int, sink: int, incoming: list[list[tuple[int, str | None]]]):
        self.node_count = node_count
        self.sink = sink
        self.incoming = incoming


def build_lattice(slots: Sequence[Sequence[Sequence[str]]]) -> Lattice:
    """Build the lattice for a sequence of slots.

    Each slot is a list of alternative token sequences — one slot per word (or per
    character, depending on the unit being scored), and one alternative per legitimate
    reading of it. A slot with a single alternative is a literal stretch.

    Tokenisation happens before this point, by whole words, so an alternation with a
    suffix glued to it (``{டோன்|tone}ல``) arrives as one slot with the readings
    ``டோன்ல`` and ``toneல`` rather than as a word followed by a stray suffix.
    """
    incoming: list[list[tuple[int, str | None]]] = [[]]
    frontier = 0

    def new_node() -> int:
        incoming.append([])
        return len(incoming) - 1

    for alternatives in slots:
        if len(alternatives) == 1:
            for token in alternatives[0]:
                nxt = new_node()
                incoming[nxt].append((frontier, token))
                frontier = nxt
            continue

        branch_ends: list[int] = []
        for tokens in alternatives:
            node = frontier
            for token in tokens:
                nxt = new_node()
                incoming[nxt].append((node, token))
                node = nxt
            branch_ends.append(node)
        join = new_node()
        for end in branch_ends:
            # Several readings can tokenise identically; one epsilon each is harmless
            # and keeps branch order, and therefore tie-breaking, stable.
            incoming[join].append((end, None))
        frontier = join

    return Lattice(node_count=len(incoming), sink=frontier, incoming=incoming)


def lattice_edit_counts(
    hypothesis: Sequence[str],
    lattice: Lattice,
) -> tuple[ErrorCounts, list[str]]:
    """Cheapest alignment of ``hypothesis`` to any path through ``lattice``.

    Returns the error counts and the reference tokens of the chosen path. Ties go to the
    reading that appears first in the reference, so a score is reproducible.
    """
    rows = len(hypothesis) + 1
    nodes = lattice.node_count

    # cost[i][v] and the backpointer that produced it.
    cost: list[list[float]] = [[_INF] * nodes for _ in range(rows)]
    # (previous_i, previous_node, token_consumed_from_reference)
    back: list[list[tuple[int, int, str | None] | None]] = [
        [None] * nodes for _ in range(rows)
    ]
    # Error breakdown carried alongside, as (subs, dels, ins, reference_length).
    detail: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0)] * nodes for _ in range(rows)
    ]
    cost[0][0] = 0.0

    def offer(
        i: int,
        v: int,
        value: float,
        source: tuple[int, int, str | None],
        counts: tuple[int, int, int, int],
    ) -> None:
        if value < cost[i][v]:
            cost[i][v] = value
            back[i][v] = source
            detail[i][v] = counts

    for i in range(rows):
        for v in range(nodes):
            if i > 0:
                # Insertion: the hypothesis token has nothing behind it in the reference.
                if cost[i - 1][v] < _INF:
                    s, d, ins, length = detail[i - 1][v]
                    offer(i, v, cost[i - 1][v] + 1, (i - 1, v, None), (s, d, ins + 1, length))
                # Match or substitution across one reference token.
                for u, token in lattice.incoming[v]:
                    if token is None or cost[i - 1][u] == _INF:
                        continue
                    same = hypothesis[i - 1] == token
                    s, d, ins, length = detail[i - 1][u]
                    offer(
                        i,
                        v,
                        cost[i - 1][u] + (0 if same else 1),
                        (i - 1, u, token),
                        (s if same else s + 1, d, ins, length + 1),
                    )
            # Same row: a reference token the hypothesis is missing, and epsilon joins.
            # Node ids increase along every edge, so cost[i][u] is already final here.
            for u, token in lattice.incoming[v]:
                if cost[i][u] == _INF:
                    continue
                s, d, ins, length = detail[i][u]
                if token is None:
                    offer(i, v, cost[i][u], (i, u, None), (s, d, ins, length))
                else:
                    offer(i, v, cost[i][u] + 1, (i, u, token), (s, d + 1, ins, length + 1))

    end = cost[rows - 1][lattice.sink]
    if end == _INF:  # pragma: no cover - a built lattice always has a path
        raise ValueError("no path through the reference lattice")

    subs, dels, ins, length = detail[rows - 1][lattice.sink]

    chosen: list[str] = []
    i, v = rows - 1, lattice.sink
    while not (i == 0 and v == 0):
        step = back[i][v]
        if step is None:  # pragma: no cover
            break
        prev_i, prev_v, token = step
        if token is not None:
            chosen.append(token)
        i, v = prev_i, prev_v
    chosen.reverse()

    return ErrorCounts(subs, dels, ins, length), chosen
