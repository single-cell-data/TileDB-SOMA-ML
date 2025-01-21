import numpy as np
from pytest import raises

from tests._utils import parametrize
from tiledbsoma_ml.query_ids import QueryIDs


# fmt: off
@parametrize(
    "n,seed,fracs,expected", [
        (10,  111, [0.7, 0.3], [[0, 1, 4, 5, 7, 8, 9], [2, 3, 6]]),
        (10,  111, [0.7, 0.2, 0.1], [[0, 1, 4, 5, 7, 8, 9], [2, 6], [3]]),
        (10,  222, [0.7, 0.3], [[0, 3, 4, 5, 7, 8, 9], [1, 2, 6]]),
        (10,  222, [0.7, 0.2, 0.1], [[0, 3, 4, 5, 7, 8, 9], [1, 6], [2]]),
        (10, None, [1], [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (15,  111, [0.7, 0.3], [[0, 1, 4, 5, 8, 9, 10, 11, 13, 14], [2, 3, 6, 7, 12]]),
        (15,  111, [0.7, 0.2, 0.1], [[0, 1, 4, 5, 8, 9, 10, 11, 13, 14], [6, 7, 12], [2, 3]]),
        (10, None, [0.7], "Fractions must sum to 1"),
        (10, None, [0.999], "Fractions must sum to 1"),
        (10, None, [], "Fractions must sum to 1"),
    ]
)
# fmt: on
def test_split(n, fracs, seed, expected):
    obs_joinids = np.arange(n)
    query_ids = QueryIDs(
        obs_joinids=obs_joinids,
        var_joinids=np.arange(3),
    )
    if isinstance(expected, str):
        with raises(AssertionError, match=expected):
            query_ids.split(*fracs, seed=seed)
    else:
        splits = query_ids.split(*fracs, seed=seed)
        split_ids = [split.obs_joinids.tolist() for split in splits]
        assert split_ids == expected
