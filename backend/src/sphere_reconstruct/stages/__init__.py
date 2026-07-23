"""stages パッケージ.

このパッケージが import されるときに全ステージ実装が `@register` を通じて
manifest レジストリに登録される. Worker エントリから import する.

FastAPI 側からは import しない (Torch などを引きずり込まないため).
"""

from . import inspect_source  # noqa: F401
from . import extract_frames  # noqa: F401
from . import reproject_views  # noqa: F401
from . import generate_masks  # noqa: F401
