"""Back-compat shim -> pic.interface (module aliased in sys.modules)."""
import sys
import pic.interface as _real

sys.modules[__name__] = _real
