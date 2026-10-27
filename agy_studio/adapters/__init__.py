"""Model adapters (BUILD_SPEC_v3 §5; AGY_INTERFACES §4).

Adapters are only ever loaded through :meth:`agy_studio.registry.Registry.adapter`, which imports
``agy_studio.adapters.<module>`` and calls its module-level ``create_adapter(spec, **kw)``
(kw: ``project``, ``fake``, ``client``, ``registry``). ``fake`` defaults to the ``AGY_FAKE_GENERATION``
environment variable; fake adapters are deterministic, local and never touch the network.

The existing generators (``vertex_veo_director.py``, ``pro_audio_director.py``) are wrapped, never edited, and
are imported lazily — only on real (non-fake) code paths.
"""

from .base import (
    FAKE_ENV,
    AdapterError,
    PollResult,
    UnsupportedFeatureError,
    VideoAdapter,
    fake_mode,
    resolve_fake,
)

__all__ = [
    "FAKE_ENV",
    "AdapterError",
    "PollResult",
    "UnsupportedFeatureError",
    "VideoAdapter",
    "fake_mode",
    "resolve_fake",
]
