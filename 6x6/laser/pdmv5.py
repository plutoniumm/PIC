"""Back-compat shim -- the PDMv5 wire protocol moved to `pic.devices.pdmv5`."""
import sys
import pic.devices.pdmv5 as _real

sys.modules[__name__] = _real
