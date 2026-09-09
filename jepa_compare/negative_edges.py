"""Evaluation-time negative edge strategies from DyGLib / EdgeBank.

Poursafaei et al. (NeurIPS 2022) evaluate dynamic link prediction against
negatives that are harder than uniformly corrupted destinations.  DyGLib's
``NegativeEdgeSampler`` implements them; this module mirrors its
``historical`` strategy so every method in the comparison scores the same
negatives.  It is deliberately NumPy-only so the sampling core can be checked
without a PyTorch installation.

Protocol (DyGLib ``utils/utils.py`` at commit 3aacc36, ``historical_sample``):

* the sampler is built on the complete event stream (train + validation +
  test) and seeded once per split: 0 for validation, 2 for test, with
  ``reset_random_state()`` before each evaluation pass;
* an evaluation batch consists of ``batch_size`` (200) consecutive positive
  events; ``t_start``/``t_end`` are the timestamps of its first/last event;
* the candidate pool is ``{edges with t <= t_start} - {edges with
  t_start <= t <= t_end}`` (both endpoints fixed, ordered pairs);
* ``size`` distinct edges are drawn uniformly from the pool.  When the pool is
  smaller than ``size`` every pool edge is used and the remainder is filled
  with uniformly random ``(unique source, unique destination)`` pairs that do
  not appear among the batch positives (without replacement unless that is
  impossible); random fill precedes the pool edges in the returned arrays;
* the i-th negative is scored at the timestamp of the i-th batch positive.

Only the random draws differ from DyGLib: DyGLib iterates Python sets whose
order is implementation-defined and materialises ``unique_src x unique_dst``
(hundreds of millions of pairs on Flights).  The sampler below draws the same
distributions with rejection sampling and keeps its history incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


NEGATIVE_STRATEGIES = ("random", "historical")


def normalize_negative_strategy(value: object) -> str:
    """Validate ``link.negative_strategy``; ``inductive`` is not implemented yet."""
    strategy = str(value).lower()
    if strategy in NEGATIVE_STRATEGIES:
        return strategy
    if strategy == "inductive":
        raise NotImplementedError(
            "link.negative_strategy 'inductive' is not implemented yet; "
            "use 'random' or 'historical'"
        )
    raise ValueError(
        f"unknown link.negative_strategy {value!r}; expected one of "
        f"{list(NEGATIVE_STRATEGIES)}"
    )


def _as_int64(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer node ids")
    return array.astype(np.int64, copy=False)


def _as_float64(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


@dataclass(frozen=True)
class SnapshotNegativeEdges:
    """Negatives for one evaluation snapshot, aligned with its events.

    Row ``i`` corresponds to the ``i``-th event of the snapshot in chronological
    order.  The positive endpoints are stored so every consumer can verify it
    walks the same event stream before pairing positives with negatives.
    ``from_pool`` is False for DyGLib's random fill.
    """

    positive_sources: np.ndarray
    positive_destinations: np.ndarray
    sources: np.ndarray
    destinations: np.ndarray
    from_pool: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.positive_sources,
            self.positive_destinations,
            self.sources,
            self.destinations,
            self.from_pool,
        )
        lengths = {int(np.asarray(array).shape[0]) for array in arrays}
        if len(lengths) != 1:
            raise ValueError("negative edge arrays must share one length")
        object.__setattr__(
            self, "positive_sources", _as_int64(self.positive_sources, "positive_sources")
        )
        object.__setattr__(
            self,
            "positive_destinations",
            _as_int64(self.positive_destinations, "positive_destinations"),
        )
        object.__setattr__(self, "sources", _as_int64(self.sources, "sources"))
        object.__setattr__(
            self, "destinations", _as_int64(self.destinations, "destinations")
        )
        object.__setattr__(
            self, "from_pool", np.asarray(self.from_pool, dtype=bool)
        )

    def __len__(self) -> int:
        return int(self.sources.shape[0])

    def check_alignment(
        self,
        positive_sources: np.ndarray,
        positive_destinations: np.ndarray,
        *,
        context: str = "evaluation",
    ) -> None:
        """Raise unless the caller's positives are exactly the table's positives."""
        sources = _as_int64(positive_sources, "positive_sources")
        destinations = _as_int64(positive_destinations, "positive_destinations")
        if (
            sources.shape != self.positive_sources.shape
            or not np.array_equal(sources, self.positive_sources)
            or not np.array_equal(destinations, self.positive_destinations)
        ):
            raise ValueError(
                f"{context} positives do not match the negative edge table; "
                "the table is built over the chronological raw event stream of "
                "each target snapshot"
            )


def concatenate_negative_edges(
    parts: Sequence[SnapshotNegativeEdges],
) -> SnapshotNegativeEdges:
    if not parts:
        raise ValueError("at least one snapshot is required")
    return SnapshotNegativeEdges(
        np.concatenate([part.positive_sources for part in parts]),
        np.concatenate([part.positive_destinations for part in parts]),
        np.concatenate([part.sources for part in parts]),
        np.concatenate([part.destinations for part in parts]),
        np.concatenate([part.from_pool for part in parts]),
    )


class HistoricalNegativeEdgeSampler:
    """DyGLib ``NegativeEdgeSampler(negative_sample_strategy="historical")``.

    ``sources``/``destinations``/``timestamps`` describe the complete event
    stream in chronological order.  DyGLib's evaluation loop calls ``sample``
    with non-decreasing batch start times, so the history set is maintained
    incrementally; an earlier start time (a new evaluation pass) rebuilds it.
    """

    def __init__(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        seed: int,
    ) -> None:
        self.sources = _as_int64(sources, "sources")
        self.destinations = _as_int64(destinations, "destinations")
        self.timestamps = _as_float64(timestamps, "timestamps")
        if not (
            self.sources.shape == self.destinations.shape == self.timestamps.shape
        ):
            raise ValueError("sources, destinations and timestamps must align")
        if self.sources.size == 0:
            raise ValueError("the event stream is empty")
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError("the event stream must be sorted by timestamp")
        self.unique_sources = np.unique(self.sources)
        self.unique_destinations = np.unique(self.destinations)
        self._source_set = set(self.unique_sources.tolist())
        self._destination_set = set(self.unique_destinations.tolist())
        self.seed = int(seed)
        self.random_state = np.random.RandomState(self.seed)
        # Unique edges with timestamp <= the most recent batch start, in
        # first-seen order, plus the number of stream events folded in so far.
        self._history_edges: list[tuple[int, int]] = []
        self._history_index: dict[tuple[int, int], int] = {}
        self._history_cursor = 0

    def reset_random_state(self) -> None:
        """Mirror DyGLib: evaluation passes restart the seeded generator."""
        self.random_state = np.random.RandomState(self.seed)

    def _edges_between(self, start_time: float, end_time: float) -> set[tuple[int, int]]:
        low = int(np.searchsorted(self.timestamps, start_time, side="left"))
        high = int(np.searchsorted(self.timestamps, end_time, side="right"))
        return set(
            zip(self.sources[low:high].tolist(), self.destinations[low:high].tolist())
        )

    def _advance_history(self, batch_start_time: float) -> None:
        stop = int(np.searchsorted(self.timestamps, batch_start_time, side="right"))
        if stop < self._history_cursor:
            # DyGLib rebuilds the history from the whole stream on every call,
            # so moving backwards (a new evaluation pass) is legal; rebuild.
            self._history_edges = []
            self._history_index = {}
            self._history_cursor = 0
        for source, destination in zip(
            self.sources[self._history_cursor : stop].tolist(),
            self.destinations[self._history_cursor : stop].tolist(),
        ):
            edge = (source, destination)
            if edge not in self._history_index:
                self._history_index[edge] = len(self._history_edges)
                self._history_edges.append(edge)
        self._history_cursor = stop

    def historical_pool(
        self, batch_start_time: float, batch_end_time: float
    ) -> set[tuple[int, int]]:
        """Candidate pool of one batch (exposed for protocol checks)."""
        self._advance_history(batch_start_time)
        current = self._edges_between(batch_start_time, batch_end_time)
        return {edge for edge in self._history_edges if edge not in current}

    def _choose_from_history(
        self, size: int, excluded: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Uniform draw of ``size`` distinct history edges outside ``excluded``."""
        total = len(self._history_edges)
        chosen: list[tuple[int, int]] = []
        taken: set[int] = set()
        while len(chosen) < size:
            draws = self.random_state.randint(
                0, total, size=2 * (size - len(chosen)) + 8
            )
            for index in draws.tolist():
                if index in taken:
                    continue
                edge = self._history_edges[index]
                if edge in excluded:
                    continue
                taken.add(index)
                chosen.append(edge)
                if len(chosen) == size:
                    break
        return chosen

    def random_fill(
        self, size: int, batch_sources: np.ndarray, batch_destinations: np.ndarray
    ) -> list[tuple[int, int]]:
        """DyGLib ``random_sample_with_collision_check``.

        Uniform over ``unique_sources x unique_destinations`` minus the batch
        positives, without replacement unless fewer such pairs exist.
        """
        if size <= 0:
            return []
        batch_pairs = set(
            zip(
                _as_int64(batch_sources, "batch_sources").tolist(),
                _as_int64(batch_destinations, "batch_destinations").tolist(),
            )
        )
        colliding = sum(
            1
            for source, destination in batch_pairs
            if source in self._source_set and destination in self._destination_set
        )
        complement = (
            len(self.unique_sources) * len(self.unique_destinations) - colliding
        )
        if complement <= 0:
            raise ValueError("no random negative edge is available for this batch")
        with_replacement = complement < size
        chosen: list[tuple[int, int]] = []
        chosen_set: set[tuple[int, int]] = set()
        while len(chosen) < size:
            count = 2 * (size - len(chosen)) + 8
            source_rows = self.random_state.randint(
                0, len(self.unique_sources), size=count
            )
            destination_rows = self.random_state.randint(
                0, len(self.unique_destinations), size=count
            )
            for source, destination in zip(
                self.unique_sources[source_rows].tolist(),
                self.unique_destinations[destination_rows].tolist(),
            ):
                edge = (source, destination)
                if edge in batch_pairs:
                    continue
                if not with_replacement and edge in chosen_set:
                    continue
                chosen.append(edge)
                chosen_set.add(edge)
                if len(chosen) == size:
                    break
        return chosen

    def sample(
        self,
        size: int,
        batch_sources: np.ndarray,
        batch_destinations: np.ndarray,
        batch_start_time: float,
        batch_end_time: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(sources, destinations, from_pool)`` for one batch."""
        if size < 0:
            raise ValueError("size must be non-negative")
        if batch_end_time < batch_start_time:
            raise ValueError("batch_end_time precedes batch_start_time")
        self._advance_history(batch_start_time)
        current = self._edges_between(batch_start_time, batch_end_time)
        overlap = sum(1 for edge in current if edge in self._history_index)
        pool_size = len(self._history_edges) - overlap
        if size > pool_size:
            pool = [edge for edge in self._history_edges if edge not in current]
            fill = self.random_fill(size - pool_size, batch_sources, batch_destinations)
            edges = fill + pool
            from_pool = np.concatenate(
                [np.zeros(len(fill), dtype=bool), np.ones(len(pool), dtype=bool)]
            )
        else:
            edges = self._choose_from_history(size, current)
            from_pool = np.ones(len(edges), dtype=bool)
        sources = np.fromiter((edge[0] for edge in edges), dtype=np.int64, count=len(edges))
        destinations = np.fromiter(
            (edge[1] for edge in edges), dtype=np.int64, count=len(edges)
        )
        return sources, destinations, from_pool


def sample_stream_negatives(
    sampler: HistoricalNegativeEdgeSampler,
    sources: np.ndarray,
    destinations: np.ndarray,
    timestamps: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one negative per positive of an evaluation split.

    The split stream is consumed in DyGLib's evaluation order: consecutive
    ``batch_size`` positives per batch, the batch time span taken from its
    first and last event.  The generator is reset first, as DyGLib does at the
    start of every evaluation pass.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sources = _as_int64(sources, "sources")
    destinations = _as_int64(destinations, "destinations")
    timestamps = _as_float64(timestamps, "timestamps")
    if not (sources.shape == destinations.shape == timestamps.shape):
        raise ValueError("split arrays must align")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("the evaluation split must be sorted by timestamp")
    count = int(sources.shape[0])
    negative_sources = np.empty(count, dtype=np.int64)
    negative_destinations = np.empty(count, dtype=np.int64)
    from_pool = np.empty(count, dtype=bool)
    sampler.reset_random_state()
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch_sources, batch_destinations, batch_pool = sampler.sample(
            stop - start,
            sources[start:stop],
            destinations[start:stop],
            float(timestamps[start]),
            float(timestamps[stop - 1]),
        )
        negative_sources[start:stop] = batch_sources
        negative_destinations[start:stop] = batch_destinations
        from_pool[start:stop] = batch_pool
    return negative_sources, negative_destinations, from_pool


@dataclass(frozen=True)
class NegativeEdgeTable:
    """Per-snapshot evaluation negatives shared by every compared method.

    ``entries`` maps a snapshot time to its negatives, or to ``None`` for
    snapshots that are never evaluation targets (training targets and pure
    context snapshots keep sampling random negatives).  Looking up a snapshot
    that is absent from the table is an error, so a consumer cannot silently
    drift back to random negatives.
    """

    strategy: str
    entries: Mapping[int, SnapshotNegativeEdges | None]
    summary: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def for_snapshot(self, time: int) -> SnapshotNegativeEdges | None:
        try:
            return self.entries[int(time)]
        except KeyError as error:
            raise KeyError(
                f"snapshot {time} is not covered by the {self.strategy} negative "
                "edge table"
            ) from error

    def for_snapshots(self, times: Sequence[int]) -> SnapshotNegativeEdges:
        """Concatenate the negatives of evaluation targets in the given order."""
        parts = []
        for time in times:
            entry = self.for_snapshot(time)
            if entry is None:
                raise ValueError(
                    f"snapshot {time} has no {self.strategy} negatives; only "
                    "validation/test target snapshots do"
                )
            parts.append(entry)
        return concatenate_negative_edges(parts)


def build_negative_edge_table(
    snapshot_streams: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    evaluation_targets: Mapping[str, Sequence[int]],
    seeds: Mapping[str, int],
    *,
    strategy: str,
    batch_size: int,
) -> NegativeEdgeTable:
    """Build the table from per-snapshot ``(time, sources, destinations, timestamps)``.

    ``snapshot_streams`` must list every snapshot of the graph; the streams are
    concatenated in time order to form DyGLib's ``full_data``.
    ``evaluation_targets`` maps a split name to the times of its target
    snapshots (in order) and ``seeds`` gives that split's sampler seed.
    """
    strategy = normalize_negative_strategy(strategy)
    if strategy == "random":
        raise ValueError("the random strategy does not use a negative edge table")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if set(evaluation_targets) != set(seeds):
        raise ValueError("every evaluation split needs exactly one sampler seed")
    ordered = sorted(snapshot_streams, key=lambda item: int(item[0]))
    times = [int(item[0]) for item in ordered]
    if len(set(times)) != len(times):
        raise ValueError("snapshot times must be unique")
    per_snapshot: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for time, sources, destinations, timestamps in ordered:
        sources = _as_int64(sources, f"sources of snapshot {time}")
        destinations = _as_int64(destinations, f"destinations of snapshot {time}")
        timestamps = _as_float64(timestamps, f"timestamps of snapshot {time}")
        if not (sources.shape == destinations.shape == timestamps.shape):
            raise ValueError(f"snapshot {time} event arrays do not align")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError(
                f"snapshot {time} events are not chronological; the negative "
                "edge table requires time-sorted query events"
            )
        per_snapshot[int(time)] = (sources, destinations, timestamps)
    full_sources = np.concatenate([per_snapshot[time][0] for time in times])
    full_destinations = np.concatenate([per_snapshot[time][1] for time in times])
    full_timestamps = np.concatenate([per_snapshot[time][2] for time in times])
    if np.any(np.diff(full_timestamps) < 0):
        raise ValueError("snapshots are not chronological across the stream")

    entries: dict[int, SnapshotNegativeEdges | None] = {time: None for time in times}
    summary: dict[str, dict[str, float]] = {}
    claimed: set[int] = set()
    for split_name, target_times in evaluation_targets.items():
        target_times = [int(time) for time in target_times]
        if not target_times:
            raise ValueError(f"{split_name} has no target snapshots")
        if any(time not in per_snapshot for time in target_times):
            raise ValueError(f"{split_name} targets are missing from the stream")
        if target_times != sorted(target_times):
            raise ValueError(f"{split_name} targets must be chronological")
        overlap = claimed.intersection(target_times)
        if overlap:
            raise ValueError(
                f"snapshots {sorted(overlap)} are targets of two evaluation splits"
            )
        claimed.update(target_times)
        split_sources = np.concatenate([per_snapshot[t][0] for t in target_times])
        split_destinations = np.concatenate([per_snapshot[t][1] for t in target_times])
        split_timestamps = np.concatenate([per_snapshot[t][2] for t in target_times])
        sampler = HistoricalNegativeEdgeSampler(
            full_sources, full_destinations, full_timestamps, seed=int(seeds[split_name])
        )
        negative_sources, negative_destinations, from_pool = sample_stream_negatives(
            sampler, split_sources, split_destinations, split_timestamps, batch_size
        )
        cursor = 0
        for time in target_times:
            count = int(per_snapshot[time][0].shape[0])
            rows = slice(cursor, cursor + count)
            entries[time] = SnapshotNegativeEdges(
                per_snapshot[time][0],
                per_snapshot[time][1],
                negative_sources[rows],
                negative_destinations[rows],
                from_pool[rows],
            )
            cursor += count
        events = int(split_sources.shape[0])
        summary[split_name] = {
            "events": float(events),
            "batches": float(-(-events // batch_size)),
            "pool_fraction": float(from_pool.mean()) if events else 0.0,
            "unique_negative_edges": float(
                len(set(zip(negative_sources.tolist(), negative_destinations.tolist())))
            ),
        }
    return NegativeEdgeTable(strategy=strategy, entries=entries, summary=summary)
