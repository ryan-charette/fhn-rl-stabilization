import numpy as np
import pytest

from sarsa_lambda.features import StateActionFeatureVectorWithTile


def test_tile_feature_vector_marks_one_tile_per_tiling() -> None:
    features = StateActionFeatureVectorWithTile(
        state_low=np.array([0.0, 0.0]),
        state_high=np.array([1.0, 1.0]),
        num_actions=2,
        num_tilings=4,
        tile_width=np.array([0.5, 0.5]),
    )

    active = features.active_indices(np.array([0.25, 0.75]), done=False, action=1)
    vector = features(np.array([0.25, 0.75]), done=False, action=1)

    assert vector.shape == (features.feature_vector_len(),)
    assert active.shape == (features.num_tilings,)
    assert np.count_nonzero(vector) == features.num_tilings
    np.testing.assert_array_equal(np.flatnonzero(vector), active)


def test_tile_feature_vector_rejects_invalid_actions() -> None:
    features = StateActionFeatureVectorWithTile(
        state_low=np.array([-1.0]),
        state_high=np.array([1.0]),
        num_actions=2,
        num_tilings=2,
        tile_width=np.array([0.5]),
    )

    with pytest.raises(ValueError, match="action must be"):
        features.active_indices(np.array([0.0]), done=False, action=2)
