"""SPDs by chip id: one `SPD` per detector, `SPDs` for all of them at once.

Every SPD is a Vega board on USB speaking the same VAUL protocol, so a detector is named by
its USB serial number (chip id) and nothing else -- the device path changes with the socket
and the OS, the id does not.

    with SPDs() as spds:                  # every SPD on USB
        spds.count(0.5)                   # {id: Count} over the same half second, each
        spds.normalised(0.5)              # {id: (rate - dark) / (max - dark)}
        spds.send("start")                # to all
        for sid, vault in spds.vaults(timeout=5):
            print(sid, vault.count, vault.rate)

Each detector's dark and max rates live in MAX_PATH (`python -m spd max`): the max with the
light set as bright as the user set it, the dark with the switch parked on its dark channel.
A normalised reading is then 0 dark and 1 at that max, on the scale of that detector.
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
    """{chip id: (dark, max)} in counts/s. A detector with no dark on file has dark None."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return {k: (v.get("dark"), v.get("rate")) for k, v in d.items()}


def save_max(rates: dict | None = None, darks: dict | None = None, path=MAX_PATH):
    """Record max and/or dark rates (counts/s per chip id), keeping every other entry."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        d = {}
    at = time.strftime("%Y-%m-%d %H:%M")
    for key, vals in (("rate", rates), ("dark", darks)):
        for k, r in (vals or {}).items():
            d.setdefault(k, {}).update({key: float(r), key + "_at": at})
    Path(path).write_text(json.dumps(d, indent=1))


def normalise(rate: float, law) -> float:
    """(rate - dark) / (max - dark): 0 dark, 1 at the stored max. Not clipped: a reading
    above 1 says the max was not the brightest setting after all, one below 0 is Poisson."""
    dark, mx = law
    return (rate - dark) / (mx - dark)


def check_law(ids, maxes) -> dict:
    """{id: (dark, max)} for `ids`, or a refusal naming what is missing."""
    bad = [i for i in ids if not maxes.get(i) or None in maxes[i]]
    if bad:
        raise SPDError(f"no dark and max stored for {bad}: `python -m spd dark` laser off, then `python -m spd max <id>` per output")
    flat = [i for i in ids if maxes[i][1] <= maxes[i][0]]
    if flat:
        raise SPDError(f"stored max not above dark for {flat}: re-run `python -m spd max <id>` with its output lit")
    return {i: maxes[i] for i in ids}


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
    ids = {chip_id(p.serial_number) for p in list_ports.comports() if p.vid and p.serial_number}
    return sorted(ids - exclude)


def chip_id(sn: str) -> str:
    """Windows' FTDI driver appends a channel letter to the 8-character chip id
    (DQ00QQ2C -> DQ00QQ2CA); the id is what the map and the dark/max file are keyed on."""
    return sn[:8] if len(sn) == 9 and sn[-1] in "AB" else sn


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
            self.spds = [i if isinstance(i, SPD) else SPD(i) for i in ids]
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
        clock, so they are read in parallel and agree to within a frame. One that drops out
        fails the whole read by its id: a read with an output silently missing is not one."""
        out, errs = {}, []

        def one(s):
            try:
                out[s.id] = s.count(seconds)
            except Exception as e:  # a pulled cable is a SerialException, not an SPDError
                errs.append(f"SPD {s.id} dropped out: {e}")

        ts = [threading.Thread(target=one, args=(s,)) for s in self.spds]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs:
            raise SPDError("; ".join(errs))
        return out

    def normalised(self, seconds: float = 1.0, maxes=None) -> dict[str, float]:
        """Rate over `seconds` on each detector's stored dark-to-max scale; see `normalise`."""
        law = check_law(self.ids, load_max() if maxes is None else maxes)
        return {i: normalise(c.rate, law[i]) for i, c in self.count(seconds).items()}


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
            save_max({i: 2000.0 for i in spds.ids[:2]}, path=p)
            save_max({i: 4000.0 for i in spds.ids[2:]}, path=p)  # adds, does not replace
            try:
                spds.normalised(0.1, load_max(p))
                raise AssertionError("normalised without a dark")
            except SPDError as e:
                assert "dark and max" in str(e), e
            save_max(darks={i: 0.0 for i in spds.ids[:2]}, path=p)
            save_max(darks={i: 1000.0 for i in spds.ids[2:]}, path=p)  # keeps the max
            law = load_max(p)
            assert law[spds.ids[3]] == (1000.0, 4000.0), law
            n = spds.normalised(1.0, law)
        assert all(abs(n[i] - 1) < 0.1 for i in spds.ids[:2]), n
        assert all(abs(n[i] - 1 / 3) < 0.05 for i in spds.ids[2:]), n  # (2000-1000)/(4000-1000)
        for bad in ({}, {i: (5.0, 5.0) for i in spds.ids}):
            try:
                spds.normalised(0.1, bad)
                raise AssertionError(f"normalised on {bad}")
            except SPDError:
                pass
    # a callable rate: the detector follows whatever light a model puts on it, frame by frame
    lit = [100.0]
    with SPDs([SPD.mock("m", count_hz=lambda: lit[0])]) as spds:
        a = spds.count(0.5)["m"].rate
        lit[0] = 2000.0
        b = spds.count(0.5)["m"].rate
    assert a < 300 < 1500 < b, (a, b)
    # one of four pulled mid-run: the read fails, and names it
    with SPDs(mock=4, count_hz=100.0, on_text=lambda sid, t: None) as spds:
        spds.count(0.1)
        gone = spds.spds[2]

        def pulled(*_):
            raise OSError("device reports readiness to read but returned no data")

        gone.ser.read = pulled
        try:
            spds.count(0.1)
            raise AssertionError("a read with one SPD gone came back")
        except SPDError as e:
            assert f"SPD {gone.id} dropped out" in str(e) and "mock-spd-0" not in str(e), e
    return seen
