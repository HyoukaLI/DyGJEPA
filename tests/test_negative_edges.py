"""Protocol checks for the historical/inductive negative samplers against DyGLib.

The reference class below is DyGLib's ``NegativeEdgeSampler`` (utils/utils.py
at commit 3aacc36, MIT licence) reduced to the historical and inductive
strategies.  It is kept verbatim, including its set-based bookkeeping, so the
tests compare our incremental sampler against the upstream definition rather
than against a paraphrase of it.  These tests only need NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

from jepa_compare.negative_edges import (
    DyGLibNegativeEdgeSampler,
    HistoricalNegativeEdgeSampler,
    NegativeEdgeTable,
    SnapshotNegativeEdges,
    build_negative_edge_table,
    normalize_negative_strategy,
    sample_stream_negatives,
)


class UpstreamNegativeEdgeSampler:
    """Upstream historical/inductive sampler (verbatim logic, random-fill included)."""

    def __init__(self, src_node_ids, dst_node_ids, interact_times, seed,
                 last_observed_time=None, negative_sample_strategy="historical"):
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time
        self.possible_edges = set(
            (src_node_id, dst_node_id)
            for src_node_id in self.unique_src_node_ids
            for dst_node_id in self.unique_dst_node_ids
        )
        if self.negative_sample_strategy == "inductive":
            self.observed_edges = self.get_unique_edges_between_start_end_time(
                self.earliest_time, self.last_observed_time
            )
        self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time, end_time):
        selected_time_interval = np.logical_and(
            self.interact_times >= start_time, self.interact_times <= end_time
        )
        return set(
            (src_node_id, dst_node_id)
            for src_node_id, dst_node_id in zip(
                self.src_node_ids[selected_time_interval],
                self.dst_node_ids[selected_time_interval],
            )
        )

    def random_sample_with_collision_check(self, size, batch_src_node_ids, batch_dst_node_ids):
        batch_edges = set(
            (batch_src_node_id, batch_dst_node_id)
            for batch_src_node_id, batch_dst_node_id in zip(batch_src_node_ids, batch_dst_node_ids)
        )
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0
        random_edge_indices = self.random_state.choice(
            len(possible_random_edges), size=size, replace=len(possible_random_edges) < size
        )
        return (
            np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]),
            np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices]),
        )

    def pool(self, current_batch_start_time, current_batch_end_time):
        historical_edges = self.get_unique_edges_between_start_end_time(
            start_time=self.earliest_time, end_time=current_batch_start_time
        )
        current_batch_edges = self.get_unique_edges_between_start_end_time(
            start_time=current_batch_start_time, end_time=current_batch_end_time
        )
        if self.negative_sample_strategy == "inductive":
            # unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
            return historical_edges - self.observed_edges - current_batch_edges
        return historical_edges - current_batch_edges

    def sample(self, size, batch_src_node_ids, batch_dst_node_ids,
               current_batch_start_time, current_batch_end_time):
        # historical_sample and inductive_sample share this body upstream; only
        # the pool differs (see ``pool``).
        return self.historical_sample(size, batch_src_node_ids, batch_dst_node_ids,
                                      current_batch_start_time, current_batch_end_time)

    def historical_sample(self, size, batch_src_node_ids, batch_dst_node_ids,
                          current_batch_start_time, current_batch_end_time):
        unique_historical_edges = self.pool(current_batch_start_time, current_batch_end_time)
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(
                size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
            )
            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(
                len(unique_historical_edges), size=size, replace=False
            )
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)


def synthetic_stream(
    seed: int,
    *,
    events: int,
    num_sources: int,
    num_destinations: int,
    source_offset: int = 0,
    destination_offset: int = 0,
    coarse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A stream with heavy edge repetition; ``coarse`` yields many tied stamps."""
    rng = np.random.default_rng(seed)
    # Zipf-like preferences produce repeated edges as in interaction data.
    source_weights = 1.0 / np.arange(1, num_sources + 1)
    destination_weights = 1.0 / np.arange(1, num_destinations + 1)
    sources = rng.choice(
        num_sources, size=events, p=source_weights / source_weights.sum()
    ) + source_offset
    destinations = rng.choice(
        num_destinations, size=events, p=destination_weights / destination_weights.sum()
    ) + destination_offset
    if coarse:
        timestamps = np.sort(rng.integers(0, 12, size=events)).astype(np.float64)
    else:
        timestamps = np.sort(rng.uniform(0.0, 1000.0, size=events))
    return sources.astype(np.int64), destinations.astype(np.int64), timestamps


def split_batches(count: int, batch_size: int):
    for start in range(0, count, batch_size):
        yield start, min(start + batch_size, count)


@pytest.mark.parametrize(
    "stream_kwargs",
    [
        dict(events=600, num_sources=12, num_destinations=9, destination_offset=12),
        dict(events=600, num_sources=10, num_destinations=10, coarse=True),
        dict(events=300, num_sources=4, num_destinations=3, destination_offset=4),
    ],
)
def test_candidate_pool_matches_dyglib_definition(stream_kwargs) -> None:
    sources, destinations, timestamps = synthetic_stream(3, **stream_kwargs)
    split_start = int(0.6 * len(sources))
    ours = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=0)
    reference = UpstreamNegativeEdgeSampler(sources, destinations, timestamps, seed=0)
    batch_size = 20
    for start, stop in split_batches(len(sources) - split_start, batch_size):
        rows = slice(split_start + start, split_start + stop)
        t_start, t_end = float(timestamps[rows][0]), float(timestamps[rows][-1])
        assert ours.historical_pool(t_start, t_end) == reference.pool(t_start, t_end)


@pytest.mark.parametrize(
    "stream_kwargs, batch_size",
    [
        (dict(events=600, num_sources=12, num_destinations=9, destination_offset=12), 25),
        (dict(events=600, num_sources=10, num_destinations=10, coarse=True), 25),
        (dict(events=300, num_sources=4, num_destinations=3, destination_offset=4), 5),
    ],
)
def test_samples_follow_dyglib_pool_and_fill_rules(stream_kwargs, batch_size) -> None:
    sources, destinations, timestamps = synthetic_stream(5, **stream_kwargs)
    # Start evaluating almost immediately so early batches exercise DyGLib's
    # random-fill branch while later ones draw from a pool larger than a batch.
    split_start = 5
    ours = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=2)
    reference = UpstreamNegativeEdgeSampler(sources, destinations, timestamps, seed=2)
    possible = reference.possible_edges
    saw_fill = saw_full_pool = False
    for start, stop in split_batches(len(sources) - split_start, batch_size):
        rows = slice(split_start + start, split_start + stop)
        batch_sources = sources[rows]
        batch_destinations = destinations[rows]
        t_start, t_end = float(timestamps[rows][0]), float(timestamps[rows][-1])
        size = stop - start
        pool = reference.pool(t_start, t_end)
        batch_pairs = set(zip(batch_sources.tolist(), batch_destinations.tolist()))

        neg_src, neg_dst, from_pool = ours.sample(
            size, batch_sources, batch_destinations, t_start, t_end
        )
        ref_src, ref_dst = reference.historical_sample(
            size, batch_sources, batch_destinations, t_start, t_end
        )
        assert neg_src.shape == ref_src.shape == (size,)
        assert neg_src.dtype == np.int64 and neg_dst.dtype == np.int64
        edges = list(zip(neg_src.tolist(), neg_dst.tolist()))
        if size > len(pool):
            saw_fill = True
            fill = size - len(pool)
            # Random fill precedes the complete pool, exactly as upstream.
            assert not from_pool[:fill].any() and from_pool[fill:].all()
            assert set(edges[fill:]) == pool and len(set(edges[fill:])) == len(pool)
            ref_edges = list(zip(ref_src.tolist(), ref_dst.tolist()))
            assert set(ref_edges[fill:]) == pool
            for edge in edges[:fill]:
                assert edge in possible and edge not in batch_pairs
            if len(possible - batch_pairs) >= fill:
                assert len(set(edges[:fill])) == fill
        else:
            saw_full_pool = True
            assert from_pool.all()
            assert len(set(edges)) == size
            assert set(edges) <= pool
    assert saw_fill and saw_full_pool


def test_sampling_is_seeded_and_resettable() -> None:
    sources, destinations, timestamps = synthetic_stream(
        7, events=400, num_sources=8, num_destinations=6, destination_offset=8
    )
    split_start = 200
    batch = slice(split_start, split_start + 30)
    args = (
        30,
        sources[batch],
        destinations[batch],
        float(timestamps[batch][0]),
        float(timestamps[batch][-1]),
    )
    first = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=0)
    second = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=0)
    other = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=1)
    a = first.sample(*args)
    b = second.sample(*args)
    c = other.sample(*args)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not (np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))
    # reset_random_state replays the draw, as DyGLib does per evaluation pass.
    first.reset_random_state()
    replay = first.sample(*args)
    assert np.array_equal(a[0], replay[0]) and np.array_equal(a[1], replay[1])


def test_pool_draws_are_uniform_without_replacement() -> None:
    sources, destinations, timestamps = synthetic_stream(
        11, events=500, num_sources=8, num_destinations=8, destination_offset=8
    )
    sampler = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=0)
    batch = slice(400, 410)
    t_start, t_end = float(timestamps[batch][0]), float(timestamps[batch][-1])
    pool = sorted(sampler.historical_pool(t_start, t_end))
    assert len(pool) > 10
    counts = {edge: 0 for edge in pool}
    trials = 3000
    for trial in range(trials):
        sampler.random_state = np.random.RandomState(trial)
        neg_src, neg_dst, _ = sampler.sample(
            5, sources[batch], destinations[batch], t_start, t_end
        )
        edges = list(zip(neg_src.tolist(), neg_dst.tolist()))
        assert len(set(edges)) == 5
        for edge in edges:
            counts[edge] += 1
    expected = trials * 5 / len(pool)
    frequencies = np.array(list(counts.values()), dtype=np.float64)
    # Loose uniformity band: every pool edge is drawn at about the same rate.
    assert np.all(np.abs(frequencies - expected) < 0.25 * expected + 4 * np.sqrt(expected))


def test_random_fill_matches_collision_check_semantics() -> None:
    # Two sources, two destinations -> four possible pairs. The batch holds
    # (0, 2) and (1, 3), so only two pairs remain for the fill; asking for
    # three forces DyGLib's ``replace=True`` branch.
    sources = np.array([0, 1, 0, 1], dtype=np.int64)
    destinations = np.array([2, 3, 2, 3], dtype=np.int64)
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    sampler = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=0)
    fill = sampler.random_fill(2, sources[:2], destinations[:2])
    assert set(fill) == {(0, 3), (1, 2)}
    with_replacement = sampler.random_fill(3, sources[:2], destinations[:2])
    assert len(with_replacement) == 3
    assert set(with_replacement) <= {(0, 3), (1, 2)}
    with pytest.raises(ValueError):  # every possible pair is a batch positive
        sampler.random_fill(
            1, np.array([0, 1, 0, 1]), np.array([2, 3, 3, 2])
        )


def test_history_rewinds_like_dyglib_when_batches_go_backwards() -> None:
    sources, destinations, timestamps = synthetic_stream(
        13, events=200, num_sources=5, num_destinations=5, destination_offset=5
    )
    sampler = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=0)
    reference = UpstreamNegativeEdgeSampler(sources, destinations, timestamps, seed=0)
    sampler.sample(5, sources[150:155], destinations[150:155], timestamps[150], timestamps[154])
    # DyGLib recomputes the history per call, so an earlier batch must see
    # only the edges before it even after a later batch was processed.
    assert sampler.historical_pool(timestamps[50], timestamps[54]) == reference.pool(
        timestamps[50], timestamps[54]
    )
    with pytest.raises(ValueError):
        HistoricalNegativeEdgeSampler(sources, destinations, timestamps[::-1], seed=0)


def test_stream_negatives_use_positive_timestamps_positionally() -> None:
    sources, destinations, timestamps = synthetic_stream(
        17, events=500, num_sources=8, num_destinations=6, destination_offset=8
    )
    split = slice(300, 500)
    sampler = HistoricalNegativeEdgeSampler(sources, destinations, timestamps, seed=2)
    reference = UpstreamNegativeEdgeSampler(sources, destinations, timestamps, seed=2)
    neg_src, neg_dst, from_pool = sample_stream_negatives(
        sampler, sources[split], destinations[split], timestamps[split], batch_size=40
    )
    assert neg_src.shape == (200,) and from_pool.shape == (200,)
    for start, stop in split_batches(200, 40):
        rows = slice(300 + start, 300 + stop)
        pool = reference.pool(float(timestamps[rows][0]), float(timestamps[rows][-1]))
        batch_edges = set(zip(neg_src[start:stop].tolist(), neg_dst[start:stop].tolist()))
        assert batch_edges & pool == set(
            edge
            for edge, is_pool in zip(
                zip(neg_src[start:stop].tolist(), neg_dst[start:stop].tolist()),
                from_pool[start:stop].tolist(),
            )
            if is_pool
        )
        # A negative never coincides with a positive of its own batch.
        positives = set(zip(sources[rows].tolist(), destinations[rows].tolist()))
        assert batch_edges.isdisjoint(positives)
    # Rerunning is deterministic: the generator is reset inside.
    again = sample_stream_negatives(
        sampler, sources[split], destinations[split], timestamps[split], batch_size=40
    )
    assert np.array_equal(again[0], neg_src) and np.array_equal(again[1], neg_dst)


def test_negative_edge_table_covers_targets_and_rejects_unknown_snapshots() -> None:
    streams = []
    for time in range(6):
        sources, destinations, timestamps = synthetic_stream(
            20 + time, events=50, num_sources=6, num_destinations=5, destination_offset=6
        )
        streams.append((time, sources, destinations, timestamps + 1000.0 * time))
    table = build_negative_edge_table(
        streams,
        {"validation": [3], "test": [4, 5]},
        {"validation": 0, "test": 2},
        strategy="historical",
        batch_size=20,
    )
    assert table.strategy == "historical"
    assert table.for_snapshot(0) is None and table.for_snapshot(2) is None
    validation = table.for_snapshot(3)
    assert isinstance(validation, SnapshotNegativeEdges)
    assert len(validation) == 50
    assert np.array_equal(validation.positive_sources, streams[3][1])
    test = table.for_snapshots([4, 5])
    assert len(test) == 100
    assert np.array_equal(test.positive_destinations, np.concatenate([streams[4][2], streams[5][2]]))
    assert set(table.summary) == {"validation", "test"}
    assert table.summary["test"]["events"] == 100.0 and table.summary["test"]["batches"] == 5.0
    assert 0.0 <= table.summary["test"]["pool_fraction"] <= 1.0
    validation.check_alignment(streams[3][1], streams[3][2])
    with pytest.raises(ValueError):
        validation.check_alignment(streams[3][2], streams[3][1])
    with pytest.raises(KeyError):
        table.for_snapshot(99)
    with pytest.raises(ValueError):
        table.for_snapshots([2])
    with pytest.raises(ValueError):
        build_negative_edge_table(
            streams, {"validation": [3], "test": [3, 4]}, {"validation": 0, "test": 2},
            strategy="historical", batch_size=20,
        )
    with pytest.raises(ValueError):
        build_negative_edge_table(
            streams, {"validation": [3]}, {"validation": 0},
            strategy="random", batch_size=20,
        )


def test_table_test_split_history_includes_validation_events() -> None:
    # An edge that first appears in the validation split is a historical
    # negative candidate for the test split (DyGLib builds the sampler over
    # full_data), but never for the validation split itself.
    streams = []
    for time in range(4):
        sources = np.array([0, 1, 0, 1], dtype=np.int64)
        destinations = np.array([4, 5, 4, 5], dtype=np.int64)
        if time == 2:
            sources = np.array([0, 2, 0, 2], dtype=np.int64)
            destinations = np.array([4, 6, 4, 6], dtype=np.int64)
        streams.append((time, sources, destinations, np.arange(4, dtype=np.float64) + 10.0 * time))
    table = build_negative_edge_table(
        streams,
        {"validation": [2], "test": [3]},
        {"validation": 0, "test": 2},
        strategy="historical",
        batch_size=4,
    )
    validation = table.for_snapshot(2)
    test = table.for_snapshot(3)
    assert (2, 6) not in set(zip(validation.sources.tolist(), validation.destinations.tolist()))
    # Test batch positives are (0,4),(1,5): pool = {(2,6)} plus random fill.
    test_edges = set(zip(test.sources.tolist(), test.destinations.tolist()))
    assert (2, 6) in test_edges
    assert test.from_pool.sum() == 1


def test_normalize_negative_strategy() -> None:
    assert normalize_negative_strategy("Random") == "random"
    assert normalize_negative_strategy("historical") == "historical"
    assert normalize_negative_strategy("Inductive") == "inductive"
    with pytest.raises(ValueError):
        normalize_negative_strategy("hist")
    with pytest.raises(ValueError):
        DyGLibNegativeEdgeSampler(
            np.array([0, 1]), np.array([2, 3]), np.array([0.0, 1.0]), seed=0, strategy="random"
        )
    with pytest.raises(ValueError):  # inductive needs the observed boundary
        DyGLibNegativeEdgeSampler(
            np.array([0, 1]), np.array([2, 3]), np.array([0.0, 1.0]), seed=0, strategy="inductive"
        )


def inductive_pair(sources, destinations, timestamps, seed, boundary_index):
    """Our sampler and the upstream oracle sharing one observed boundary."""
    last_observed_time = float(timestamps[boundary_index])
    ours = DyGLibNegativeEdgeSampler(
        sources, destinations, timestamps, seed=seed,
        strategy="inductive", last_observed_time=last_observed_time,
    )
    reference = UpstreamNegativeEdgeSampler(
        sources, destinations, timestamps, seed=seed,
        last_observed_time=last_observed_time, negative_sample_strategy="inductive",
    )
    return ours, reference


@pytest.mark.parametrize(
    "stream_kwargs, expect_pool",
    [
        (dict(events=600, num_sources=12, num_destinations=9, destination_offset=12), True),
        (dict(events=600, num_sources=10, num_destinations=10, coarse=True), True),
        # 4 x 3 possible edges are all observed before the split: the inductive
        # pool is empty for every batch and DyGLib falls back to random fill.
        (dict(events=300, num_sources=4, num_destinations=3, destination_offset=4), False),
    ],
)
def test_inductive_pool_matches_dyglib_definition(stream_kwargs, expect_pool) -> None:
    sources, destinations, timestamps = synthetic_stream(21, **stream_kwargs)
    split_start = int(0.6 * len(sources))
    # The observed boundary is the last event before the split; with coarse
    # timestamps it can tie with the split's first events, exactly as DyGLib's
    # time-based observed_edges would.
    ours, reference = inductive_pair(sources, destinations, timestamps, 0, split_start - 1)
    assert reference.observed_edges == set(
        zip(sources[: split_start].tolist(), destinations[: split_start].tolist())
    ) or stream_kwargs.get("coarse", False)
    batch_size = 20
    non_empty = 0
    for start, stop in split_batches(len(sources) - split_start, batch_size):
        rows = slice(split_start + start, split_start + stop)
        t_start, t_end = float(timestamps[rows][0]), float(timestamps[rows][-1])
        pool = reference.pool(t_start, t_end)
        assert ours.candidate_pool(t_start, t_end) == pool
        # Inductive candidates are never edges observed up to the boundary.
        assert pool.isdisjoint(reference.observed_edges)
        non_empty += bool(pool)
    assert (non_empty > 0) == expect_pool


@pytest.mark.parametrize(
    "stream_kwargs, batch_size, expect_pool",
    [
        (dict(events=600, num_sources=12, num_destinations=9, destination_offset=12), 10, True),
        (dict(events=600, num_sources=10, num_destinations=10, coarse=True), 10, True),
        (dict(events=300, num_sources=4, num_destinations=3, destination_offset=4), 3, False),
    ],
)
def test_inductive_samples_follow_dyglib_pool_and_fill_rules(
    stream_kwargs, batch_size, expect_pool
) -> None:
    sources, destinations, timestamps = synthetic_stream(23, **stream_kwargs)
    split_start = int(0.4 * len(sources))
    ours, reference = inductive_pair(sources, destinations, timestamps, 2, split_start - 1)
    possible = reference.possible_edges
    saw_fill = saw_full_pool = False
    for start, stop in split_batches(len(sources) - split_start, batch_size):
        rows = slice(split_start + start, split_start + stop)
        batch_sources, batch_destinations = sources[rows], destinations[rows]
        t_start, t_end = float(timestamps[rows][0]), float(timestamps[rows][-1])
        size = stop - start
        pool = reference.pool(t_start, t_end)
        batch_pairs = set(zip(batch_sources.tolist(), batch_destinations.tolist()))
        neg_src, neg_dst, from_pool = ours.sample(
            size, batch_sources, batch_destinations, t_start, t_end
        )
        ref_src, ref_dst = reference.sample(
            size, batch_sources, batch_destinations, t_start, t_end
        )
        assert neg_src.shape == ref_src.shape == (size,)
        edges = list(zip(neg_src.tolist(), neg_dst.tolist()))
        assert not (set(edges) & reference.observed_edges & set(edges[int((~from_pool).sum()):]))
        if size > len(pool):
            saw_fill = True
            fill = size - len(pool)
            assert not from_pool[:fill].any() and from_pool[fill:].all()
            assert set(edges[fill:]) == pool and len(set(edges[fill:])) == len(pool)
            assert set(list(zip(ref_src.tolist(), ref_dst.tolist()))[fill:]) == pool
            for edge in edges[:fill]:
                assert edge in possible and edge not in batch_pairs
        else:
            saw_full_pool = True
            assert from_pool.all()
            assert len(set(edges)) == size and set(edges) <= pool
    assert saw_fill and saw_full_pool == expect_pool


def test_inductive_table_uses_the_split_boundaries() -> None:
    # Snapshots 0-1 (training) only contain (0,4) and (1,5).  Snapshot 2
    # (validation) introduces (2,6); snapshot 3 (test) introduces (3,7) and
    # repeats (2,6); snapshot 4 (test) repeats old edges but not (3,7).
    edges = {
        0: ([0, 1, 0, 1], [4, 5, 4, 5]),
        1: ([1, 0, 1, 0], [5, 4, 5, 4]),
        2: ([0, 2, 0, 2], [4, 6, 4, 6]),
        3: ([0, 3, 2, 3], [4, 7, 6, 7]),
        4: ([0, 2, 0, 1], [4, 6, 4, 5]),
    }
    streams = [
        (
            time,
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            np.arange(4, dtype=np.float64) + 10.0 * time,
        )
        for time, (src, dst) in edges.items()
    ]
    table = build_negative_edge_table(
        streams,
        {"validation": [2], "test": [3, 4]},
        {"validation": 0, "test": 2},
        strategy="inductive",
        batch_size=2,
    )
    assert table.strategy == "inductive"
    # DyGLib: validation sampler observes up to the last training event, the
    # test sampler up to the last validation event.
    assert table.summary["validation"]["last_observed_time"] == 13.0
    assert table.summary["test"]["last_observed_time"] == 23.0
    assert table.summary["validation"]["observed_edges"] == 2.0  # (0,4), (1,5)
    assert table.summary["test"]["observed_edges"] == 3.0  # + (2,6)

    # Validation: (2,6) is the only edge first seen after the training
    # boundary, and it occurs in both validation batches, so no inductive
    # candidate exists and every negative is random fill.
    validation = table.for_snapshot(2)
    assert not validation.from_pool.any()

    # Test: (2,6) was observed by the validation boundary, so it is never an
    # inductive negative; (3,7) first appears in the test period and becomes
    # a candidate for the snapshot-4 batches, which do not contain it.
    test = table.for_snapshots([3, 4])
    pool_edges = {
        edge
        for edge, is_pool in zip(
            zip(test.sources.tolist(), test.destinations.tolist()), test.from_pool.tolist()
        )
        if is_pool
    }
    assert pool_edges == {(3, 7)}
    assert not table.for_snapshot(3).from_pool.any()
    assert table.for_snapshot(4).from_pool.sum() == 2  # one pool edge per batch


def test_inductive_table_needs_history_before_the_first_target() -> None:
    streams = [
        (t, np.array([0, 1]), np.array([2, 3]), np.array([0.0, 1.0]) + 10.0 * t)
        for t in range(3)
    ]
    with pytest.raises(ValueError):
        build_negative_edge_table(
            streams, {"validation": [0]}, {"validation": 0},
            strategy="inductive", batch_size=2,
        )
    table = build_negative_edge_table(
        streams, {"validation": [1], "test": [2]}, {"validation": 0, "test": 2},
        strategy="inductive", batch_size=2,
    )
    assert table.summary["validation"]["last_observed_time"] == 1.0
    assert table.summary["test"]["last_observed_time"] == 11.0
