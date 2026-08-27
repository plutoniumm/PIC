"""Back-compat shim -> pic.wiring (module aliased in sys.modules)."""
import sys
import pic.wiring as _real

sys.modules[__name__] = _real
