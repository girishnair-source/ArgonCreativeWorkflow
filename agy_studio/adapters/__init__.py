"""Model adapters (BUILD_SPEC_v3 §5; AGY_INTERFACES §4).

Adapters are only ever loaded through :meth:`agy_studio.registry.Registry.adapter`, which imports
``agy_studio.adapters.<module>`` and calls its module-level ``create_adapter(spec, **kw)``
(kw: ``project``, ``fake``, ``client``, ``registry``). ``fake=True`` selects the deterministic unit-test double and is
only honoured under pytest (:class:`FakeModeUnavailable` otherwise) — there is no offline production mode.

The existing generators (``vertex_veo_director.py``, ``pro_audio_director.py``) are wrapped, never edited, and
are imported lazily — only on real (non-fake) code paths.
"""

from .auth import AuthError, require_vertex_auth
from .base import (
    AdapterError,
    FakeModeUnavailable,
    PollResult,
    UnsupportedFeatureError,
    VideoAdapter,
    fake_mode,
    resolve_fake,
)

__all__ = [
    "AdapterError",
    "FakeModeUnavailable",
    "AuthError",
    "PollResult",
    "UnsupportedFeatureError",
    "VideoAdapter",
    "fake_mode",
    "require_vertex_auth",
    "resolve_fake",
]
