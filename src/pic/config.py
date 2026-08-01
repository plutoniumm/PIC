"""Back-compat shim -> pic.config (module aliased in sys.modules)."""
import sys
import pic.config as _real

sys.modules[__name__] = _real
