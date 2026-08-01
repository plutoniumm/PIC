"""Back-compat shim -- the dual-device experiment layer moved into `pic.session`.

`from template import open_devices, laser_session, run_experiment` still works; new code
should use `from pic import open_devices, laser_session, run_experiment` (or the `Rig`
facade). `_FakeLaser` is now an alias of the canonical `pic.devices.mock.MockLaser` -- the
separate minimal mock it used to name was deleted (that duplication is what the merge fixed).
"""
from pic.session import (  # noqa: F401
    open_devices,
    laser_session,
    run_experiment,
    run_step_response,
    sweep_vectors,
    estimate_sweep_seconds,
    parse_vector,
    load_vectors,
    parse_channels,
    parse_levels,
    _resolve_pic_port,
    main,
)
from pic.devices.mock import MockLaser as _FakeLaser  # noqa: F401  back-compat alias

if __name__ == "__main__":
    main()
