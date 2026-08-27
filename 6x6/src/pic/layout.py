"""Back-compat shim -> pic.layout (module aliased in sys.modules)."""
import sys
import pic.layout as _real

sys.modules[__name__] = _real
