"""Back-compat shim -- the laser device moved to `pic.devices.laser`.
`from laser.laser import Laser` still works; new code should use `from pic import Laser`."""
import sys
import pic.devices.laser as _real

sys.modules[__name__] = _real
