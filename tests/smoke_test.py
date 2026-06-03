"""Hardware-free smoke test for the lib/ + src/ scaffold. Run from the repo root:
    /usr/local/Caskroom/miniconda/base/envs/pic/bin/python tests/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from lib.pic import PIC, MockPIC, PICConfig, acquisition
from src import mzi, influence, characterize, inverse, data

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and bool(cond)

print("== src.mzi ==")
t1, t2 = 0.7, -1.3
M = mzi.mzi(t1, t2)
check("mzi == closed form", np.allclose(M, mzi.mzi_closed(t1, t2)))
check("mzi unitary", np.allclose(M.conj().T @ M, np.eye(2), atol=1e-12))
check("attenuator |t| == |sin D|", np.isclose(abs(mzi.attenuator(t1, t2)), abs(np.sin((t1 - t2) / 2))))
N = 6
layers = mzi.clements_layers(N)
params = {(li, m): (np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi))
          for li, layer in enumerate(layers) for m in layer}
U = mzi.mesh_unitary(N, params, layers)
check("6x6 mesh unitary", np.allclose(U.conj().T @ U, np.eye(N), atol=1e-9))

print("== src.characterize.fit_fringe (recover phi2, phi0) ==")
v = np.linspace(0, 4, 41)
phi2_true, phi0_true = np.pi / 1.5 ** 2, 0.8
p = 0.1 + 0.08 * np.cos(phi2_true * v ** 2 + phi0_true) + np.random.default_rng(0).normal(0, 1e-3, v.size)
fit = characterize.fit_fringe(v, p)
check("fit_fringe recovers phi2", fit and abs(fit["phi2"] - phi2_true) < 0.05)
check("fit_fringe Vpi ~ 1.5", fit and abs(fit["Vpi"] - 1.5) < 0.1)

print("== lib.pic.acquisition.fit_homodyne ==")
hp = 0.05 + 0.03 * np.cos((np.pi / 1.5 ** 2) * v ** 2 + 1.1)
h = acquisition.fit_homodyne(v, hp)
check("fit_homodyne returns fields", h is not None and h["B"] > 0)

print("== src.influence.eta2_matrix ==")
rng = np.random.default_rng(1)
Xt = rng.choice(np.arange(0, 4.5, 0.5), size=(2000, 8))
Yt = np.column_stack([np.sin(Xt[:, 0]) + 0.01 * rng.normal(size=2000),   # driven by ch0
                      0.01 * rng.normal(size=2000)])                      # noise only
E = influence.eta2_matrix(Xt, Yt)
check("eta2 finds ch0 drives out0", int(np.argmax(E[:, 0])) == 0)
check("eta2 ~0 for pure-noise out1", E[:, 1].max() < 0.05)

print("== lib.pic.MockPIC + src.inverse.MonteCarloInverse ==")
# synthetic 64->14 forward: a few channels drive a few PDs through cos^2 fringes
rngf = np.random.default_rng(2)
Wsel = rngf.integers(0, 64, size=(14, 3))
phi2f = rngf.uniform(0.5, 1.5, size=64)
def forward(v):
    y = np.zeros(14)
    for k in range(14):
        if k in (0, 2, 7, 11):      # damaged PDs read ~0
            continue
        s = sum(np.cos(phi2f[c] * v[c] ** 2) for c in Wsel[k])
        y[k] = 0.05 * (1 + np.cos(s)) + 0.02
    return y
pic = MockPIC(forward, noise=0.0, config=PICConfig(num_dac=64)).open()
target_v = rngf.choice(np.arange(0, 4.5, 0.5), size=64)
desired = pic.measure(target_v)
check("MockPIC returns 10 live PDs", desired.shape == (10,))
mc = inverse.MonteCarloInverse(seed=3)
seeds = [rngf.choice(np.arange(0, 4.5, 0.5), size=64) for _ in range(5)]
best = mc.run(pic.measure, desired, seeds, iters=25, per_seed=4)
check("MC inverse reduces loss", best["loss"] < np.linalg.norm(pic.measure(seeds[0]) - desired))
print(f"     MC best loss = {best['loss']:.4f} at iter {best['iter']}")
pic.close()

print("== src.data ==")
check("data.live drops 4 damaged", data.live(np.zeros((3, 14))).shape == (3, 10))

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
