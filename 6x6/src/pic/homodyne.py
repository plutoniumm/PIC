"""Back-compat shim -> pic.homodyne (module aliased in sys.modules)."""
import sys
import pic.homodyne as _real

sys.modules[__name__] = _real
