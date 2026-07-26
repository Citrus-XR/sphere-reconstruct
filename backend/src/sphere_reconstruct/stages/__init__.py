"""stages パッケージ.

このパッケージが import されるときに全ステージ実装が `@register` を通じて
manifest レジストリに登録される. Worker エントリから import する.

FastAPI 側からは import しない (Torch などを引きずり込まないため).
"""

from . import (
    align_reconstruction,  # noqa: F401
    denoise_frames,  # noqa: F401
    export_dataset,  # noqa: F401
    extract_features,  # noqa: F401
    extract_frames,  # noqa: F401
    generate_masks,  # noqa: F401
    inspect_source,  # noqa: F401
    match_features,  # noqa: F401
    reconstruct,  # noqa: F401
    reproject_views,  # noqa: F401
)
