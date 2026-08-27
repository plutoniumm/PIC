"""Wire-by-wire trace of one 6x6 structure ("6x6P") from the GDS into a netlist + geometry render."""

from __future__ import annotations
import json, math, collections
import gdstk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

GDS = "NUS/AD_SiPhIS_180423.gds"
WG_LAYER = (10, 0)  # routing-waveguide core
TOL = 0.30  # vertex-coincidence tolerance (um)
IC_SNAP = 3.0  # input-coupler port sits ~1um off its waveguide start

LOCAL_PORTS = {
    "AMF_PSOI_Si2X2MMI_Cband_v3p0": [
        ("L1", 0, 0.667),
        ("L2", 0, -0.667),
        ("R3", 75, 0.667),
        ("R4", 75, -0.667),
    ],
    "AMF_PSOI_Si1X2MMI_Cband_v3p0": [("IN", 0, 0), ("O2", 55, 0.9), ("O3", 55, -0.9)],
    "AMF_PSOI_PowMonitor_Cband_Cell_v3p5": [("IN", 0, 0)],
    "input_coupler": [("P", 0, 0)],
}
ROLE = {
    "AMF_PSOI_Si2X2MMI_Cband_v3p0": "mmi2x2",
    "AMF_PSOI_Si1X2MMI_Cband_v3p0": "mmi1x2",
    "AMF_PSOI_PowMonitor_Cband_Cell_v3p5": "pd",
    "input_coupler": "input",
    "AMF_PSOI_SiGC1D_Cband_v3p0": "grating",
}
COLOR = {
    "mmi2x2": "#1f77b4",
    "mmi1x2": "#2ca02c",
    "pd": "#000000",
    "input": "#9467bd",
    "grating": "#ff7f0e",
    "heater": "#d62728",
}


def xf(o, rot, xr, lx, ly):
    if xr:
        ly = -ly
    c, s = math.cos(rot), math.sin(rot)
    return (o[0] + c * lx - s * ly, o[1] + s * lx + c * ly)


def build():
    lib = gdstk.read_gds(GDS)
    top = next(c for c in lib.cells if c.name == "top")

    comps, ports = [], []
    for i, r in enumerate(top.references):
        comps.append(
            {
                "id": i,
                "cell": r.cell.name,
                "role": ROLE.get(r.cell.name, "other"),
                "x": round(r.origin[0], 3),
                "y": round(r.origin[1], 3),
                "rot": round(r.rotation, 4),
                "xrefl": int(r.x_reflection),
            }
        )
        for pn, lx, ly in LOCAL_PORTS.get(r.cell.name, []):
            px, py = xf(r.origin, r.rotation, int(r.x_reflection), lx, ly)
            ports.append({"comp": i, "port": pn, "x": px, "y": py})

    wpolys = top.get_polygons(depth=None, layer=WG_LAYER[0], datatype=WG_LAYER[1])

    # union-find over device ports and waveguide polygons by vertex coincidence
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        parent[find(a)] = find(b)

    def key(x, y):
        return (round(x / TOL), round(y / TOL))

    loc = collections.defaultdict(list)
    vtx = []
    for k, p in enumerate(ports):
        find(("P", k))
        loc[key(p["x"], p["y"])].append(("P", k))
    for j, poly in enumerate(wpolys):
        find(("W", j))
        for vx, vy in poly.points:
            loc[key(vx, vy)].append(("W", j))
            vtx.append((vx, vy, ("W", j)))
    for (kx, ky), _ in list(loc.items()):
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near += loc.get((kx + dx, ky + dy), [])
        for n in near[1:]:
            union(near[0], n)
    # snap input-coupler ports onto the nearest waveguide vertex (~1um gap by design)
    for k, p in enumerate(ports):
        if comps[p["comp"]]["role"] == "input":
            bx, by, bn = min(
                vtx, key=lambda v: (v[0] - p["x"]) ** 2 + (v[1] - p["y"]) ** 2
            )
            if (bx - p["x"]) ** 2 + (by - p["y"]) ** 2 <= IC_SNAP**2:
                union(("P", k), bn)

    # nets = device ports sharing a root; edges between components
    root_ports = collections.defaultdict(list)
    for k in range(len(ports)):
        root_ports[find(("P", k))].append(k)
    edges = collections.defaultdict(set)
    nets = []
    for ks in root_ports.values():
        net = [(ports[k]["comp"], ports[k]["port"]) for k in ks]
        nets.append(net)
        cs = {c for c, _ in net}
        for a in cs:
            for b in cs:
                if a != b:
                    edges[a].add(b)
    return top, comps, ports, wpolys, edges, nets, root_ports


def isolate(comps, edges, x0=15700, y0=1961.5):
    start = next(
        c["id"]
        for c in comps
        if c["role"] == "input" and abs(c["x"] - x0) < 1 and abs(c["y"] - y0) < 1
    )
    seen, q = {start}, [start]
    while q:
        u = q.pop()
        for v in edges[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return start, seen


def main():
    top, comps, ports, wpolys, edges, nets, root_ports = build()
    start, sub = isolate(comps, edges)
    subc = [comps[i] for i in sub]
    xs = [c["x"] for c in subc]
    ys = [c["y"] for c in subc]
    bx0, bx1, by0, by1 = min(xs) - 60, max(xs) + 160, min(ys) - 40, max(ys) + 40

    # heaters (phase shifters) inside the structure bbox -> arms
    heaters = [
        c
        for c in comps
        if c["role"] == "other"
        and c["cell"] == "cell_thermal_phase_shifter"
        and bx0 < c["x"] < bx1
        and by0 < c["y"] < by1
    ]

    # terminations: a 2x2 output port (R3/R4) whose net touches no other device
    port_net_root = {}
    for r, ks in root_ports.items():
        for k in ks:
            port_net_root[k] = r
    netsize = {r: len({ports[k]["comp"] for k in ks}) for r, ks in root_ports.items()}
    terminations = []
    for k, p in enumerate(ports):
        c = comps[p["comp"]]
        if c["id"] in sub and c["role"] == "mmi2x2" and p["port"] in ("R3", "R4"):
            if netsize[find_root(port_net_root, k)] == 1:
                terminations.append(
                    {
                        "comp": c["id"],
                        "port": p["port"],
                        "x": round(p["x"], 2),
                        "y": round(p["y"], 2),
                    }
                )

    role_counts = collections.Counter(c["role"] for c in subc)
    summary = {
        "n_components": len(sub),
        "mmi2x2 (MZI couplers)": role_counts["mmi2x2"],
        "mmi1x2 (splitters/taps)": role_counts["mmi1x2"],
        "pd (photodiodes)": role_counts["pd"],
        "input couplers": role_counts["input"],
        "heaters (phase shifters)": len(heaters),
        "terminations (dangling 2x2 outputs)": len(terminations),
        "bbox": [
            round(min(xs), 1),
            round(min(ys), 1),
            round(max(xs), 1),
            round(max(ys), 1),
        ],
    }
    print("6x6P trace summary:")
    for k, v in summary.items():
        print(f"  {k:38s}: {v}")

    sub_set = set(sub)
    out_comps = [dict(c, is_heater=False) for c in subc] + [
        dict(h, role="heater", is_heater=True) for h in heaters
    ]
    out_nets = []
    for net in nets:
        members = [{"comp": c, "port": p} for c, p in net if c in sub_set]
        if members:
            out_nets.append(members)
    with open("pic_data/netlist_6x6P.json", "w") as f:
        json.dump(
            {
                "summary": summary,
                "components": out_comps,
                "nets": out_nets,
                "input_coupler_id": start,
            },
            f,
            indent=1,
        )
    print("wrote pic_data/netlist_6x6P.json")

    fig, ax = plt.subplots(figsize=(26, 5))
    sub_wp = [
        p.points
        for p in wpolys
        if bx0 < p.points[:, 0].mean() < bx1 and by0 < p.points[:, 1].mean() < by1
    ]
    ax.add_collection(
        PolyCollection(sub_wp, facecolors="#b9d3ee", edgecolors="none", zorder=1)
    )
    for c in subc:
        ax.scatter(
            c["x"],
            c["y"],
            c=COLOR[c["role"]],
            s=26 if c["role"] != "pd" else 70,
            marker=(
                "o"
                if c["role"] == "mmi2x2"
                else (
                    "^"
                    if c["role"] == "mmi1x2"
                    else ("*" if c["role"] == "pd" else "D")
                )
            ),
            zorder=4,
            edgecolors="none",
        )
    for h in heaters:
        ax.scatter(h["x"] + 58, h["y"], c=COLOR["heater"], s=8, marker="s", zorder=3)
    for t in terminations:
        ax.scatter(
            t["x"],
            t["y"],
            facecolors="none",
            edgecolors="#c0392b",
            s=90,
            lw=1.6,
            marker="o",
            zorder=5,
        )
    leg = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR["mmi2x2"],
            markersize=8,
            label="2x2 MMI (MZI coupler)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=COLOR["mmi1x2"],
            markersize=8,
            label="1x2 MMI (splitter/tap)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="k",
            markersize=12,
            label="photodiode",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=COLOR["input"],
            markersize=8,
            label="input coupler",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLOR["heater"],
            markersize=7,
            label="heater (phase shifter)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#c0392b",
            markersize=10,
            label="termination (dangling out)",
        ),
    ]
    ax.legend(handles=leg, loc="upper left", fontsize=8, ncol=3)
    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by0, by1)
    ax.set_aspect("equal")
    ax.set_title(
        f"6x6P traced from GDS — {summary['n_components']} devices, "
        f"{summary['heaters (phase shifters)']} heaters, {summary['terminations (dangling 2x2 outputs)']} terminations "
        "(input at right, flows left)"
    )
    fig.savefig("schematic_ref/gds_6x6P_geom.png", dpi=130, bbox_inches="tight")
    print("wrote schematic_ref/gds_6x6P_geom.png")


def find_root(pmap, k):
    return pmap[k]


if __name__ == "__main__":
    main()
