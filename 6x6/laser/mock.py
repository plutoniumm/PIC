"""Back-compat shim -- MockLaser moved to `pic.devices.mock`."""
import sys
import pic.devices.mock as _real

sys.modules[__name__] = _real
