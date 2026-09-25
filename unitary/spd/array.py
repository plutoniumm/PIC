"""SPDs by chip id: one `SPD` per detector, `SPDs` for all of them at once.

Every SPD is a Vega board on USB speaking the same VAUL protocol, so a detector is named by
its USB serial number (chip id) and nothing else -- the device path changes with the socket
and the OS, the id does not.

    with SPDs() as spds:                  # every SPD on USB
        spds.count(0.5)                   # {id: Count} over the same half second, each
        spds.normalised(0.5)              # {id: rate / that detector's stored max}
        spds.send("start")                # to all
        for sid, vault in spds.vaults(timeout=5):
            print(sid, vault.count, vault.rate)

The max is one rate per detector in MAX_PATH, taken with the mesh set to send it the most
light (`python -m spd max`), so a normalised reading is 0..1 on the scale of that detector.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from .vega import Count, SPDError, Vega, _rig_ids

MAX_PATH = Path("pic_data/spd_max.json")


def load_max(path=MAX_PATH) -> dict:
    try:
        return {k: v["rate"] for k, v in json.loads(Path(path).read_text()).items()}
    except (OSError, ValueError):
        return {}


def save_max(rates: dict, path=MAX_PATH):
    """Record `rates` (counts/s per chip id) as each detector's max, keeping the others."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        d = {}
    at = time.strftime("%Y-%m-%d %H:%M")
    d.update({k: {"rate": float(r), "at": at} for k, r in rates.items()})
    Path(path).write_text(json.dumps(d, indent=1))


class SPD(Vega):
    """One detector, by chip id. `SPD.mock(id)` is the same object over a physical mock."""

    def __init__(self, chip_id: str, **kw):
        self.id = str(chip_id)
        super().__init__(port=kw.pop("port", self.id), **kw)

    @classmethod
    def mock(cls, chip_id: str = "mock-spd", seed: int = 0, count_hz=None, **kw):
        from .mock import MockVega

        return cls(chip_id, ser=MockVega(chip_id, seed=seed, count_hz=count_hz), **kw)

    def __repr__(self):
        return f"SPD({self.id!r})"


def discover(exclude=None) -> list[str]:
    """Chip ids of every USB serial device that is not a known rig instrument: the SPDs."""
    from serial.tools import list_ports

    exclude = _rig_ids() if exclude is None else set(exclude)
    return sorted(
        p.serial_number
        for p in list_ports.comports()
        if p.vid and p.serial_number and p.serial_number not in exclude
    )


class SPDs:
    """Several SPDs read together. Vaults from all of them merge into one queue, tagged by
    chip id, so a consumer sees them in arrival order across detectors."""

    def __init__(self, ids=None, *, mock: int = 0, count_hz=None, csv=None, on_text=None):
        if mock:
            self.spds = [
                SPD.mock(f"mock-spd-{i}", seed=i, count_hz=count_hz) for i in range(mock)
            ]
        else:
            ids = discover() if ids is None else list(ids)
            if not ids:
                raise SPDError("no SPD on USB: every serial device is a known rig instrument")
            self.spds = [SPD(i) for i in ids]
        for s in self.spds:
            # one csv for all, with the id per row; console text prefixed by who said it
            s.csv = None
            s.on_text = (lambda sid: lambda t: (on_text or _echo)(sid, t))(s.id)
        self.csv = csv
        self.q: queue.Queue = queue.Queue()
        self._pumps = []

    def __getitem__(self, chip_id) -> SPD:
        return next(s for s in self.spds if s.id == chip_id)

    def __iter__(self):
        return iter(self.spds)

    def __len__(self):
        return len(self.spds)

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.spds]

    def open(self):
        for s in self.spds:
            s.open()
            t = threading.Thread(target=self._pump, args=(s,), daemon=True, name=f"spd-{s.id}")
            t.start()
            self._pumps.append(t)
        if self.csv:
            import csv, os

            if not os.path.exists(self.csv):
                with open(self.csv, "w", newline="") as f:
                    csv.writer(f).writerow(("Timestamp_Unix", "SPD", "Burst_ID", "TOF_ps"))
        return self

    def close(self):
        for s in self.spds:
            s.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _pump(self, s: SPD):
        while s._thread is not None:
            try:
                v = s.get(timeout=0.2)
            except TimeoutError:
                continue
            except SPDError as e:
                self.q.put((s.id, e))
                return
            if self.csv:
                import csv

                with open(self.csv, "a", newline="") as f:
                    csv.writer(f).writerows([v.t, s.id, v.burst, int(x)] for x in v.tof_ps)
            self.q.put((s.id, v))

    def send(self, cmd: str, ids=None, timeout: float | None = 5.0):
        """To every SPD, or to `ids`; each waits for its own gap between vaults."""
        for s in self.spds:
            if ids is None or s.id in ids:
                s.send(cmd, timeout)

    def get(self, timeout: float | None = None):
        """(chip id, vault) from whichever detector finished one next."""
        try:
            sid, v = self.q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no vault from any SPD within {timeout} s")
        if isinstance(v, Exception):
            raise SPDError(f"SPD {sid}: {v}") from v
        return sid, v

    def vaults(self, timeout: float | None = None):
        while True:
            try:
                yield self.get(timeout)
            except TimeoutError:
                return

    def count(self, seconds: float = 1.0) -> dict[str, Count]:
        """Every detector's detections over the same `seconds`. Each board frames on its own
        clock, so they are read in parallel and agree to within a frame."""
        out, errs = {}, []

        def one(s):
            try:
                out[s.id] = s.count(seconds)
            except SPDError as e:
                errs.append(f"{s.id}: {e}")

        ts = [threading.Thread(target=one, args=(s,)) for s in self.spds]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs:
            raise SPDError("; ".join(errs))
        return out

    def normalised(self, seconds: float = 1.0, maxes=None) -> dict[str, float]:
        """Rate over `seconds` as a fraction of each detector's stored max. Not clipped: a
        reading above 1 says the max was not the brightest setting after all."""
        maxes = load_max() if maxes is None else maxes
        missing = [i for i in self.ids if not maxes.get(i)]
        if missing:
            raise SPDError(f"no max stored for {missing}: measure it with `python -m spd max`")
        return {i: c.rate / maxes[i] for i, c in self.count(seconds).items()}


def _echo(sid, text):
    print(f"[{sid}] {text}", end="" if text.endswith("\n") else "\n", flush=True)


def _selftest():
    with SPDs(mock=3) as spds:
        seen = {}
        for sid, v in spds.vaults(timeout=2):
            seen[sid] = seen.get(sid, 0) + 1
            if len(seen) == 3 and min(seen.values()) >= 2:
                break
        assert set(seen) == set(spds.ids), seen
        spds.send("status", ids=[spds.ids[0]])
        assert spds[spds.ids[0]].id == spds.ids[0]
    import tempfile

    with SPDs(mock=4, count_hz=2000.0) as spds:
        c = spds.count(0.5)
        assert set(c) == set(spds.ids) and all(x.frames == 25 for x in c.values())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "max.json"
            save_max({i: 2000.0 for i in spds.ids[:2]}, p)
            save_max({i: 4000.0 for i in spds.ids[2:]}, p)  # adds, does not replace
            n = spds.normalised(1.0, load_max(p))
        assert all(abs(n[i] - 1) < 0.1 for i in spds.ids[:2]), n
        assert all(abs(n[i] - 0.5) < 0.05 for i in spds.ids[2:]), n
        try:
            spds.normalised(0.1, {})
            raise AssertionError("normalised without a max")
        except SPDError:
            pass
    return seen
