# Copyright (c) 2021-2024 The Chan Zuckerberg Initiative Foundation
# Copyright (c) 2021-2024 TileDB, Inc.
#
# Licensed under the MIT License.
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.testing import assert_array_almost_equal
from pytest import fixture, raises

from tests._utils import default, param, parametrize, sweep
from tiledbsoma_ml.query_ids import QueryIDs, SamplingMethod, SamplingMethods

seed = default(None)
method = default("stochastic_rounding")
freqs = default(None)


@fixture
def query_ids(n: int) -> QueryIDs:
    """Generate a simple |QueryIDs| for testing, with a given ``obs_joinids`` size."""
    obs_joinids = np.arange(n)
    return QueryIDs(
        obs_joinids=obs_joinids,
        var_joinids=np.arange(3),
    )


@fixture
def check(
    query_ids: QueryIDs,
    fracs: List[float],
    seed: int | None,
    method: SamplingMethod,
    expected: str | List[List[int]],
):
    """Verify :meth:`tiledbsoma_ml.QueryIDs.random_split`'s output matches ``expected``, given the other arguments."""
    if isinstance(expected, str):
        with raises(AssertionError, match=expected):
            query_ids.random_split(*fracs, seed=seed, method=method)
    else:
        splits = query_ids.random_split(*fracs, seed=seed, method=method)
        split_ids = [split.obs_joinids.tolist() for split in splits]
        assert split_ids == expected


@fixture
def check_sizes(
    query_ids: QueryIDs,
    fracs: List[float],
    seeds: Sequence[int],
    method: SamplingMethod,
    expected: Dict[Tuple[int, ...], int],
    freqs: List[int] | None,
):
    split_size_hist = {}
    for seed in seeds:
        splits = query_ids.random_split(*fracs, seed=seed, method=method)
        sizes = tuple(len(split.obs_joinids) for split in splits)
        split_size_hist[sizes] = split_size_hist.get(sizes, 0) + 1

    assert split_size_hist == expected

    if freqs is not None:
        sums = np.zeros(len(fracs))
        for sizes, num in split_size_hist.items():
            sums += np.array(sizes) * num
        actual_freqs = sums / len(seeds) / len(query_ids.obs_joinids)
        assert_array_almost_equal(actual_freqs, freqs)


# fmt: off

@sweep(method=["deterministic", "stochastic_rounding"])
@parametrize("n,fracs,seed,expected", [
    (10, [ .7, .3    ], 111, [[0, 1, 4, 5, 7, 8, 9], [2, 3, 6]]),
    (10, [ .7, .3    ], 222, [[0, 3, 4, 5, 7, 8, 9], [1, 2, 6]]),
    (10, [ .7, .3    ], 333, [[1, 2, 4, 5, 7, 8, 9], [0, 3, 6]]),
    (10, [ .7, .2, .1], 111, [[0, 1, 4, 5, 7, 8, 9], [2, 6], [3]]),
    (10, [ .7, .2, .1], 222, [[0, 3, 4, 5, 7, 8, 9], [1, 6], [2]]),
    (10, [ .7, .2, .1], 333, [[1, 2, 4, 5, 7, 8, 9], [3, 6], [0]]),
])
def test_split__even_rounding(check):
    """When ``fracs`` divides ``n`` into integer-sized splits, ``"deterministic"`` and `` "stochastic_rounding"`` are
    equivalent."""
    pass


@param(n=15, fracs=[.7, .3])
@parametrize("method,seed,expected", [
    (      "deterministic", 111, [[0, 1, 4, 5,    8, 9, 10, 11, 13, 14], [2, 3, 6, 7, 12]]),
    ("stochastic_rounding", 111, [[0, 1, 4, 5, 6, 8, 9, 10, 11, 13, 14], [2, 3,    7, 12]]),
    (      "deterministic", 222, [[   1, 3, 4, 5, 7, 8, 10, 12, 13, 14], [0, 2, 6, 9, 11]]),
    ("stochastic_rounding", 222, [[0, 1, 3, 4, 5, 7, 8, 10, 12, 13, 14], [   2, 6, 9, 11]]),
    (      "deterministic", 333, [[1, 2, 4, 5, 7, 8, 9, 10, 11, 13    ], [0, 3, 6, 12, 14]]),
    ("stochastic_rounding", 333, [[1, 2, 4, 5, 7, 8, 9, 10, 11, 13, 14], [0, 3, 6, 12    ]]),
    (      "deterministic", 444, [[0, 1, 2, 4, 5, 7, 11, 12, 13, 14], [3, 6, 8, 9, 10]]),
    ("stochastic_rounding", 444, [[0, 1, 2, 4, 5, 7, 11, 12, 13, 14], [3, 6, 8, 9, 10]]),
])
def test_split__uneven_rounding(check):
    """Target split sizes: [ 10.5, 4.5 ].

    ``"deterministic"`` always emits 10 and 5, ``"stochastic_rounding"`` should emit [10, 5] and [11, 4] with equal
    probability.
    """
    pass


@param(n=10, fracs=[.7, .3], method="multinomial")
@parametrize("seed,expected", [
    (111, [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]),
    (222, [[0, 3, 4, 5, 7, 8, 9], [1, 2, 6]]),
    (333, [[1, 2, 4, 5, 8, 9], [0, 3, 6, 7]]),
    (444, [[0, 1, 3, 4, 5, 6, 7], [2, 8, 9]]),
    (555, [[1, 2, 3, 4, 6, 9], [0, 5, 7, 8]]),
    (666, [[0, 1, 2, 3, 4, 6, 8, 9], [5, 7]]),
])
def test_split__multinomial(check):
    pass


@param(n=1, fracs=[.7, .2, .1], seeds=range(10000), method="multinomial", expected={
    (1, 0, 0): 7057,  # ≈ .7
    (0, 1, 0): 1906,  # ≈ .2
    (0, 0, 1): 1037,  # ≈ .1
})
def test_split__multinomial_size_hist_1(check_sizes):
    pass


@param(n=4, fracs=[.25, .75], seeds=range(10000), method="multinomial", expected={
    (0, 4): 3075,
    (1, 3): 4245,  # mode
    (2, 2): 2182,
    (3, 1): 451,
    (4, 0): 47,
})
def test_split__multinomial_size_hist_4(check_sizes):
    pass


@param(n=1, fracs=[.7, .2, .1], seeds=range(10000), method="stochastic_rounding", expected={
    (1, 0, 0): 7057,  # ≈ .7
    (0, 1, 0): 1975,  # ≈ .2
    (0, 0, 1):  968,  # ≈ .1
})
def test_split__stochastic_rounding_size_hist_1(check_sizes):
    pass


@param(n=4, fracs=[.4, .3, .2, .1], seeds=range(10000), method="stochastic_rounding", expected={
    (2, 1, 1, 0): 3756,
    (1, 1, 1, 1): 2374,
    (2, 1, 0, 1): 1515,
    (1, 2, 1, 0): 1123,
    (2, 2, 0, 0): 765,
    (1, 2, 0, 1): 467,
}, freqs=[0.4009, 0.308875, 0.181325, 0.1089])
def test_split__stochastic_rounding_size_hist_4(check_sizes):
    pass


@param(n=15, fracs=[.7, .2, .1])
@parametrize(
    "method,seed,expected", [
        (      "deterministic", 111, [[0, 1, 4, 5,    8, 9, 10, 11, 13, 14], [   6, 7, 12], [2, 3]]),
        ("stochastic_rounding", 111, [[0, 1, 4, 5, 6, 8, 9, 10, 11, 13, 14], [2,    7, 12], [   3]]),
        (      "deterministic", 222, [[   1, 3, 4, 5, 7, 8, 10, 12, 13, 14], [0, 9, 11], [2, 6]]),
        ("stochastic_rounding", 222, [[0, 1, 3, 4, 5, 7, 8, 10, 12, 13, 14], [6, 9, 11], [2   ]]),
        (      "deterministic", 333, [[1, 2, 4, 5, 7, 8, 9, 10, 11, 13    ], [3, 6,     14], [0, 12]]),
        ("stochastic_rounding", 333, [[1, 2, 4, 5, 7, 8, 9, 10, 11, 13, 14], [3, 6, 12],     [0    ]]),
    ]
)
def test_split__uneven_rounding3(check):
    pass


@sweep(method=SamplingMethods)
@parametrize("n,fracs,expected", [
    (10, [ .7  ], "Fractions must sum to 1"),
    (10, [ .999], "Fractions must sum to 1"),
    (10, [     ], "Fractions must sum to 1"),
])
def test_split__errors(check):
    pass


@param(fracs=[1])
@parametrize("n,expected", [
    ( 1, [[0]]),
    (10, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
])
def test_split__solos(check):
    pass
