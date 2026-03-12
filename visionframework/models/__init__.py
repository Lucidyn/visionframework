"""
Model components: backbones, necks, and heads.

Importing this package triggers registration of all built-in components
into the global registries.
"""

from . import backbones  # noqa: F401 — registers backbone classes
from . import necks      # noqa: F401 — registers neck classes
from . import heads      # noqa: F401 — registers head classes
