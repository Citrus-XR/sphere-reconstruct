from __future__ import annotations

import numpy as np
import pytest
from sphere_reconstruct.colmap.sequence_alignment import robust_similarity


@pytest.mark.parametrize("count", [40, 41])
def test_alignment_rejects_two_modes_without_a_strict_majority(count):
    source = np.random.default_rng(151).normal(size=(count, 3))
    target = source.copy()
    target[20:40] += [10, 0, 0]
    if count == 41:
        target[40] += [0, 10, 0]

    with pytest.raises(ValueError, match="majority-supported"):
        robust_similarity(source, target, 0.01)
