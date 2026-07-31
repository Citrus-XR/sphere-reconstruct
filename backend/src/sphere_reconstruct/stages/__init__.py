"""stages パッケージ.

このパッケージが import されるときに全ステージ実装が `@register` を通じて
manifest レジストリに登録される. Worker エントリから import する.

FastAPI 側からは import しない (Torch などを引きずり込まないため).
"""

from . import (
    align_reconstruction,  # noqa: F401
    dense_initialization,  # noqa: F401
    export_dataset,  # noqa: F401
    extract_features,  # noqa: F401
    extract_frames,  # noqa: F401
    generate_masks,  # noqa: F401
    inspect_source,  # noqa: F401
    match_features,  # noqa: F401
    position_ground,  # noqa: F401
    prepare_images,  # noqa: F401
    reconstruct,  # noqa: F401
    rectify_fisheye,  # noqa: F401
    restore_metric_scale,  # noqa: F401
)
