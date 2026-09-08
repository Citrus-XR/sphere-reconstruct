"""Internal image-normalization artifact の error contract。"""

import pytest

from sphere_reconstruct.pipeline import prepared_images
from sphere_reconstruct.pipeline.errors import LocalizedError


def test_missing_rectified_catalog_has_localization_key(tmp_path):
    with pytest.raises(LocalizedError) as captured:
        prepared_images.load_catalog(tmp_path)

    assert captured.value.key == "error.rectify_required"
