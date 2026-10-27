"""``python -m agy_studio <command>`` entry point (see :mod:`agy_studio.cli`)."""

import sys

from .cli import main

sys.exit(main())
