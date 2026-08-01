"""Back-compat shim -> pic.acquisition (module aliased in sys.modules)."""
import sys
import pic.acquisition as _real

sys.modules[__name__] = _real
