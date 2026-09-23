// every server error carries `where`: the part of the rig to look at first
const why = (d) => (d.where ? `[${d.where}] ` : "") + d.error;
const ico = (n) => `<i class="ico" style="--i:url(/icons/${n}.svg)"></i>`;
const $ = (s) => document.querySelector(s),
  U = $("#U"),
  X = $("#X"),
  MAXC = 8,
  HCAP = 25;
const gauss = () => {
  let u = 0,
    v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

for (let i = 0; i < 16; i++) {
  const e = document.createElement("input");
  e.type = "text";
  e.value = "0";
  e.id = "u" + i;
  // What is SHOWN is rounded to 5 dp; what is SENT is not. Rounding a Gram-Schmidt matrix
  // to 4 dp leaves ||U^T U - I|| at ~1.4e-4, so a displayed-precision target would be
  // refused by the 1e-5 gate -- the generator would be producing matrices its own page
  // rejects. Typing in a cell makes the typed number the truth for that cell.
  e.addEventListener("input", () => {
    Utrue[[...U.children].indexOf(e)] = parseFloat(e.value) || 0;
    orth();
  });
  e.addEventListener("blur", () => {
    e.value = Utrue[[...U.children].indexOf(e)].toFixed(5);
    orth();
  });
  U.append(e);
}
let Utrue = new Array(16).fill(0);
const uv = () => Utrue.slice();
const setU = (a) => {
  Utrue = [...a].map(Number);
  [...U.children].forEach((e, i) => (e.value = Utrue[i].toFixed(5)));
  orth();
};
function haar() {
  // real orthogonal via Gram-Schmidt on a gaussian matrix
  let m = [...Array(4)].map(() => [...Array(4)].map(gauss));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < i; j++) {
      const d = m[i].reduce((s, x, k) => s + x * m[j][k], 0);
      m[i] = m[i].map((x, k) => x - d * m[j][k]);
    }
    const n = Math.hypot(...m[i]);
    m[i] = m[i].map((x) => x / n);
  }
  return m.flat();
}
function orth() {
  // the server's gate, mirrored so the number is visible before the click
  const a = uv();
  let s = 0;
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let d = 0;
      for (let k = 0; k < 4; k++) d += a[k * 4 + i] * a[k * 4 + j];
      s += (d - (i === j ? 1 : 0)) ** 2;
    }
  const dev = Math.sqrt(s),
    ok = dev <= 1e-5;
  $("#orth").innerHTML = ok
    ? ""
    : "&#8214;U<sup>T</sup>U &minus; I&#8214; = " +
      dev.toExponential(2) +
      " &middot; over the 1e-5 tolerance, the run will be refused";
  return dev;
}
$("#rand").onclick = () => setU(haar());
$("#eye").onclick = () => setU([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

const cols = () => [...X.querySelectorAll(".col.vec")];
function addCol(vals) {
  const n = cols().length;
  if (n >= MAXC) return;
  const c = document.createElement("div");
  c.className = "col vec";
  const h = document.createElement("div");
  h.className = "colhead";
  const t = document.createElement("span");
  t.textContent = "x";
  h.append(t);
  const rm = document.createElement("button");
  rm.className = "mini";
  rm.textContent = "×";
  rm.title = "remove this vector";
  rm.onclick = () => {
    c.remove();
    relabel();
  };
  h.append(rm);
  c.append(h);
  for (let i = 0; i < 4; i++) {
    const e = document.createElement("input");
    e.type = "text";
    e.value = (vals ? vals[i] : 0).toFixed(3);
    c.append(e);
  }
  X.append(c);
  relabel();
}
function relabel() {
  const cs = cols();
  cs.forEach((c, i) => {
    c.querySelector(".colhead span").textContent = "x" + (i + 1);
    c.querySelector(".mini").style.visibility = cs.length > 1 ? "visible" : "hidden";
  });
  $("#addx").disabled = cs.length >= MAXC;
  $("#xnote").textContent = cs.length + "/" + MAXC;
}
function setX(rows) {
  cols().forEach((c) => c.remove());
  (rows && rows.length ? rows : [[1, 0, 0, 0]]).slice(0, MAXC).forEach((r) => addCol(r));
}
const rails = document.createElement("div");
rails.className = "col rails";
rails.innerHTML =
  '<div class="colhead"></div>' +
  [0, 1, 2, 3].map((i) => '<div class="cell">in ' + i + "</div>").join("");
X.append(rails);
addCol([1, 0, 0, 0]);
$("#addx").onclick = () => addCol([...Array(4)].map(gauss));
$("#randx").onclick = () =>
  cols().forEach((c) =>
    [...c.querySelectorAll("input")].forEach((e) => (e.value = gauss().toFixed(3))),
  );
const xv = () =>
  cols().map((c) => [...c.querySelectorAll("input")].map((e) => parseFloat(e.value) || 0));

const f = (v, d) =>
  v === null || v === undefined || !isFinite(v) ? "--" : v.toFixed(d === undefined ? 4 : d);
const esc = (s) =>
  String(s).replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );
const svg = (t, at, kids) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", t);
  for (const k in at) e.setAttribute(k, at[k]);
  if (kids) for (const c of [].concat(kids)) e.append(c);
  return e;
};
const tip = (el, s) => {
  el.append(svg("title", {}, []));
  el.lastChild.textContent = s;
  return el;
};
// rounded at the data end, square at the baseline: the bar's length is the datum and a
// radius on the zero end would shorten it
const capBar = (x, y, w, h, r, up) => {
  r = Math.min(r, w / 2, h);
  return up
    ? `M${x} ${y + h}V${y + r}a${r} ${r} 0 0 1 ${r} ${-r}h${w - 2 * r}a${r} ${r} 0 0 1 ${r} ${r}V${y + h}Z`
    : `M${x} ${y}V${y + h - r}a${r} ${r} 0 0 0 ${r} ${r}h${w - 2 * r}a${r} ${r} 0 0 0 ${r} ${-r}V${y}Z`;
};

function railChart(v, label, rails) {
  // One chart per x, three bars per output rail: the mesh holding U outright, U cut into
  // tiles the host sums, and the CPU. Each measured bar is scored against ITS OWN ideal:
  // normal can only measure |U|^2 x, so marking it against Ux would fault it for a sign it
  // had no way to see.
  const N = v.normal,
    T = v.tiled,
    C = v.computed,
    D = v.dpnn || []; // null when no checkpoint: the bar is simply absent
  const g = svg("svg", {
    class: "fig",
    viewBox: "0 0 640 272",
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "aria-label": "normal, tiled and computed per output rail for " + label,
  });
  const vals = N.device.concat(T.device, C, N.ideal, D),
    neg = vals.some((x) => x < 0);
  const peak = Math.max(0.1, ...vals.map(Math.abs));
  const hi = Math.ceil(peak * 10) / 10,
    lo = neg ? -hi : 0;
  const x0 = 52,
    x1 = 620,
    y0 = 30,
    y1 = 150;
  const sy = (q) => y1 - ((q - lo) / (hi - lo)) * (y1 - y0),
    zy = sy(0);
  const cap = svg("text", {
    x: x0,
    y: 20,
    "font-size": 16,
    fill: "var(--dim)",
    "letter-spacing": ".18em",
    "font-family": "IBM Plex Sans Condensed, system-ui, sans-serif",
    "font-weight": 600,
  });
  cap.textContent = label.toUpperCase();
  g.append(cap);
  const fd = (q) => (isFinite(q) ? q.toFixed(3) : "--");
  const fl = svg("text", {
    x: x1,
    y: 20,
    "text-anchor": "end",
    "font-size": 16,
    fill: "var(--ink-2)",
  });
  fl.textContent = `fidelity  F ${fd(N.fid)}  T ${fd(T.fid)}`;
  g.append(tip(fl, "|<measured,ideal>| / (||measured|| ||ideal||), each against its own ideal"));
  const ticks = neg ? [lo, 0, hi] : [0, hi];
  for (const q of ticks)
    if (q !== 0)
      g.append(
        svg("line", {
          x1: x0,
          y1: sy(q),
          x2: x1,
          y2: sy(q),
          stroke: "var(--line)",
          "stroke-width": 1,
        }),
      );
  g.append(
    svg("line", { x1: x0, y1: zy, x2: x1, y2: zy, stroke: "var(--dim)", "stroke-width": 1.5 }),
  );
  for (const q of ticks) {
    const t = svg("text", {
      x: x0 - 10,
      y: sy(q) + 5,
      "text-anchor": "end",
      "font-size": 16,
      fill: "var(--dim)",
    });
    t.textContent = q.toFixed(1);
    g.append(t);
  }
  const n = C.length,
    gw = (x1 - x0) / n,
    bw = Math.min(22, gw * 0.19);
  const pct = (m, i) => (Math.abs(i) > 1e-9 ? (Math.abs(m - i) / Math.abs(i)) * 100 : NaN);
  for (let r = 0; r < n; r++) {
    const cx = x0 + gw * (r + 0.5),
      out = rails && rails.length > r ? rails[r] : r;
    [
      [N.device[r], "var(--s1)", 0, "full"],
      [T.device[r], "var(--s2)", 1, "tiled"],
      [C[r], "var(--ref)", 2, "cpu"],
      [D[r], "var(--s3)", 3, "dpnn"],
    ].forEach(([q, fill, k, who]) => {
      if (q === undefined || q === null) return;
      const x = cx + (k - 2) * (bw + 3),
        y = Math.min(zy, sy(q)),
        h = Math.abs(sy(q) - zy) || 1;
      g.append(
        tip(
          svg("path", {
            class: "bar-" + who,
            d: capBar(x, y, bw, h, 4, q >= 0),
            fill: fill,
            stroke: "var(--bg-2)",
            "stroke-width": 2,
            "stroke-linejoin": "round",
          }),
          `${who} out ${out} = ${q.toFixed(4)}`,
        ),
      );
    });
    const t = svg("text", {
      x: cx,
      y: y1 + 24,
      "text-anchor": "middle",
      "font-size": 16,
      fill: "var(--dim)",
    });
    t.textContent = "out " + out;
    g.append(t);
    [
      [pct(N.device[r], N.ideal[r]), "var(--s1)", 48, "F", "full"],
      [pct(T.device[r], T.ideal[r]), "var(--s2)", 70, "T", "tiled"],
      [D.length ? pct(D[r], N.device[r]) : NaN, "var(--s3)", 92, "D", "dpnn"],
    ].forEach(([e, fill, dy, k, who]) => {
      if (who === "dpnn" && !D.length) return;
      const q = svg("text", {
        class: "bar-" + who,
        x: cx,
        y: y1 + dy,
        "text-anchor": "middle",
        "font-size": 16,
        fill: fill,
      });
      q.textContent = isFinite(e) ? `${k} ${e.toFixed(1)}%` : `${k} --`;
      g.append(q);
    });
  }
  return g;
}

function resultBody(e, onx) {
  // a zero x has no relative error, so it is kept out of the plots; it still gets its row
  const d = e.d,
    V = d.vectors,
    G = V.filter((v) => !v.null),
    nz = V.length - G.length;
  const box = document.createElement("div");
  box.className = "hb";
  const charts = document.createElement("div");
  G.forEach((v) => charts.append(railChart(v, "x" + (V.indexOf(v) + 1), d.rails_out)));
  box.innerHTML = "";
  box.append(charts);
  const lg = document.createElement("div");
  lg.className = "lg";
  lg.innerHTML =
    `<span class="bar-full" title="mesh holds U, one program, intensity only: measures |U|^2 x">` +
    `<i style="background:var(--s1)"></i>full</span>` +
    `<span class="bar-tiled" title="U cut into 2x2 tiles the host sums: signed Ux">` +
    `<i style="background:var(--s2)"></i>tiled</span>` +
    `<span class="bar-cpu" title="Ux on the CPU"><i style="background:var(--ref)"></i>cpu</span>` +
    (d.dpnn && d.dpnn.error
      ? `<span class="bar-dpnn v-warn">dpnn: ${esc(d.dpnn.error)}</span>`
      : d.dpnn
        ? `<span class="bar-dpnn${d.dpnn.source === "sim" ? " v-warn" : ""}" title="the DPNN's prediction of the Full program">` +
          `<i style="background:var(--s3)"></i>dpnn (${esc(String(d.dpnn.source))}, R² ${Number(d.dpnn.r2).toFixed(2)})</span>`
        : "");
  box.append(lg);
  if (nz) {
    const n = document.createElement("div");
    n.className = "note tight";
    n.textContent = `${nz} of ${V.length} vectors are zero and are absent from the plots.`;
    box.append(n);
  }
  const rows = document.createElement("div");
  rows.className = "xg";
  rows.style.marginTop = "22px";
  rows.innerHTML = V.map((v, i) => {
    return (
      `<div class="xb"><div class="xh"><span class="spacer"></span>` +
      `<button class="rl vx" data-i="${i}" title="load x${i + 1} alone into the inputs">` +
      `restore x</button></div><table><thead><tr><th>rail</th><th class="bar-full">full</th>` +
      `<th class="bar-tiled">tiled</th><th class="bar-cpu">cpu</th>` +
      (v.dpnn ? `<th class="bar-dpnn">dpnn</th>` : "") +
      `</tr></thead><tbody>` +
      v.computed
        .map(
          (c, r) =>
            `<tr><td>out ${d.rails_out[r]}</td>` +
            `<td class="bar-full">${f(v.normal.device[r])}</td>` +
            `<td class="bar-tiled">${f(v.tiled.device[r])}</td><td class="bar-cpu">${f(c)}</td>` +
            (v.dpnn ? `<td class="bar-dpnn">${f(v.dpnn[r])}</td>` : "") +
            `</tr>`,
        )
        .join("") +
      `</tbody></table></div>`
    );
  }).join("");
  rows.querySelectorAll(".vx").forEach((b) => (b.onclick = () => onx(e.X[+b.dataset.i])));
  box.append(rows);
  return box;
}

// Cap: HCAP runs. Each entry is the request plus the server's reply, a few kB; the cap is
// there so a long session cannot fill the origin's quota, and a quota throw on write halves
// the list rather than losing it. Every read and write is guarded: private windows, cleared
// site data and blocked storage all throw or hand back nothing.
const HKEY = "pic4x4.ui.runs.v2"; // v1 had one hosting per run
let HIST = [];
function hload() {
  try {
    const s = localStorage.getItem(HKEY);
    if (!s) return [];
    const a = JSON.parse(s);
    // an entry from an older page version would throw inside drawHist and blank the tab
    return Array.isArray(a)
      ? a.filter((e) => e && e.d && Array.isArray(e.d.vectors) && e.U && e.X).slice(0, HCAP)
      : [];
  } catch (e) {
    return [];
  }
}
function hsave() {
  try {
    localStorage.setItem(HKEY, JSON.stringify(HIST));
  } catch (e) {
    try {
      HIST = HIST.slice(0, Math.max(1, HIST.length >> 1));
      localStorage.setItem(HKEY, JSON.stringify(HIST));
    } catch (e2) {}
  }
}
function hclear() {
  HIST = [];
  try {
    localStorage.removeItem(HKEY);
  } catch (e) {}
  drawHist();
}

function summary(e) {
  const s = e.d.spread,
    t = new Date(e.t);
  const hh =
    String(t.getHours()).padStart(2, "0") +
    ":" +
    String(t.getMinutes()).padStart(2, "0") +
    ":" +
    String(t.getSeconds()).padStart(2, "0");
  return (
    `<span class="when">${hh}</span>` +
    `<span class="state${e.mock ? "" : " bench"}">${e.mock ? "mock" : "bench"}</span>` +
    `<span>${e.d.vectors.length}&times;</span>` +
    `<span>rel F ${f(s.normal.median, 3)}</span>` +
    `<span>T ${f(s.tiled.median, 3)}</span><span class="sp2"></span>`
  );
}

function toMV() {
  scrollTo({ top: 0, behavior: "smooth" });
} // restore lives on this page
// one x, alone, into the input grid. U is left where it is: this restores a vector, not a
// measurement, and silently swapping the target under it would be the other thing.
const restoreX = (x) => {
  setX([x]);
  toMV();
};

function drawHist() {
  const box = $("#hist");
  box.innerHTML = "";
  $("#hcount").textContent = HIST.length ? `${HIST.length}/${HCAP}` : "";
  if (!HIST.length) {
    box.innerHTML = '<div class="note" style="margin:0">no runs yet</div>';
    return;
  }
  HIST.forEach((e, i) => {
    const d = document.createElement("details");
    d.className = "run";
    const s = document.createElement("summary");
    s.innerHTML = summary(e);
    const rb = document.createElement("button");
    rb.className = "rl";
    rb.textContent = "restore run";
    rb.title = "put this U and every x back into the inputs";
    rb.onclick = (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      setU(e.U);
      setX(e.X);
      toMV();
    };
    s.append(rb);
    d.append(s);
    // built on first open: HCAP entries of charts is a lot. The newest is built here and
    // now rather than by firing the toggle, so the one open card is drawn even where the
    // event does not reach a node that is not in the document yet.
    const build = () => {
      if (d.children.length > 1) return;
      const b = resultBody(e, restoreX);
      const u = document.createElement("div");
      u.className = "two";
      u.style.margin = "10px 0 16px";
      u.innerHTML =
        `<div><div class="eyebrow">U</div><div class="umini">` +
        e.U.map((v) => `<div>${f(v, 3)}</div>`).join("") +
        `</div></div>` +
        `<div><div class="eyebrow">x</div><div class="umini" ` +
        `style="grid-template-columns:repeat(${e.X.length},1fr);max-width:${80 * e.X.length}px">` +
        [0, 1, 2, 3].map((r) => e.X.map((c) => `<div>${f(c[r], 3)}</div>`).join("")).join("") +
        `</div></div>`;
      b.prepend(u);
      d.append(b);
    };
    d.addEventListener("toggle", () => {
      if (d.open) build();
    });
    if (i === 0) {
      build();
      d.open = true;
    } // exactly one, and it is the newest
    box.append(d);
  });
}
$("#clr").onclick = hclear;

$("#run").onclick = async () => {
  const b = $("#run");
  b.disabled = true;
  $("#stat").textContent = "running…";
  $("#oops").classList.add("hide");
  const U0 = uv(),
    X0 = xv(),
    mock = document.querySelector("#mk input:checked").value === "mock";
  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ U: U0, X: X0, mock: mock, dbm: +$("#lzdbm").value }),
    });
    const d = await r.json();
    if (d.error) {
      $("#msg").textContent = why(d);
      $("#oops").classList.remove("hide");
    } else {
      HIST.unshift({ t: Date.now(), U: U0, X: X0, mock: mock, d: d });
      if (HIST.length > HCAP) HIST.length = HCAP;
      hsave();
      drawHist();
    } // drawHist opens the newest and closes everything older
  } catch (e) {
    $("#msg").textContent = `[UI server] not reachable or crashed: ${e}`;
    $("#oops").classList.remove("hide");
  }
  b.disabled = false;
  $("#stat").textContent = "";
};

let CAL = null,
  POLL = null;
const RAILY = (r) => 44 + 46 * r;
const MW = 24; // half-width of an element box
const PHI_OFF = MW + 18; // a phi sits this far left of its element, on the upper arm
const SC = { green: "var(--ok)", yellow: "var(--warn)", red: "var(--bad)", grey: "none" };
const WORD = { green: "good", yellow: "weak", red: "broken", grey: "uncal" };
const GREEK = { theta: "θ", phi: "φ", alpha: "α" };
const ROLE = {
  theta: "internal arm phase of",
  phi: "upper-arm input phase into",
  aux: "spare, redundant with",
  out: "output screen phase on",
};

// Every coordinate below comes off `S.elements` and `h.place`, which the server derives
// from clements.MESH/COLUMN and layout.THETA_DAC/PHI_DAC. Nothing here decides which rails
// an element couples or which element a heater belongs to; drawing it from memory is how
// the picture and the wiring part company.
function meshFig(S) {
  const W = 660,
    H = 226,
    X0 = 140,
    XEND = 594,
    PDX = 600;
  const dx = Math.min(106, (XEND - 36 - X0) / Math.max(1, S.ncol));
  const cx = (c) => X0 + dx * c;
  const g = svg("svg", {
    class: "fig",
    viewBox: `0 0 ${W} ${H}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "aria-label":
      `${S.nmode} rails carrying ${S.elements.length} MZI elements in ` +
      `${S.ncol} columns; each heater is drawn on the element it drives, coloured by state,` +
      ` photodiodes at the outputs`,
  });
  for (let r = 0; r < S.nmode; r++) {
    g.append(
      svg("line", {
        x1: 78,
        y1: RAILY(r),
        x2: XEND,
        y2: RAILY(r),
        stroke: "var(--line)",
        "stroke-width": 3,
      }),
    );
    g.append(
      svg("line", {
        class: "rail",
        x1: 78,
        y1: RAILY(r),
        x2: XEND,
        y2: RAILY(r),
        stroke: "var(--accent)",
        "stroke-width": 1.5,
        opacity: 0.5,
      }),
    );
    const t = svg("text", {
      x: 70,
      y: RAILY(r) + 4,
      "text-anchor": "end",
      "font-size": 16,
      fill: "var(--dim)",
    });
    t.textContent = "in " + r;
    g.append(t);
  }
  for (const e of S.elements) {
    const x = cx(e.col),
      ya = RAILY(e.rails[0]),
      yb = RAILY(e.rails[1]);
    g.append(
      tip(
        svg("rect", {
          x: x - MW,
          y: ya - 15,
          width: 2 * MW,
          height: yb - ya + 30,
          rx: 11,
          fill: "var(--pane-2)",
          stroke: "var(--line)",
          "stroke-width": 1.5,
        }),
        `MZI${e.k} · column ${e.col} · couples rails ${e.rails.join(" and ")}`,
      ),
    );
    // the bar IS the coupling. Without it the box reads as a decal over two rails that
    // never meet, which is the one thing the picture has to get across
    g.append(
      svg("line", {
        x1: x,
        y1: ya,
        x2: x,
        y2: yb,
        stroke: "var(--line)",
        "stroke-width": 7,
        "stroke-linecap": "round",
      }),
    );
  }
  for (const h of S.heaters) {
    const p = h.place;
    if (!p) continue;
    const inbox = p.kind === "theta";
    const hx = cx(p.col) - (p.kind === "phi" ? PHI_OFF : 0);
    const hy = inbox ? (RAILY(p.rails[0]) + RAILY(p.rails[1])) / 2 : RAILY(p.rails[0]);
    // a rail phase rides a segment of one arm; an internal phase sits between two
    if (inbox)
      g.append(
        svg("line", {
          x1: hx,
          y1: RAILY(p.rails[0]),
          x2: hx,
          y2: RAILY(p.rails[1]),
          stroke: SC[h.status] === "none" ? "var(--off)" : SC[h.status],
          "stroke-width": 2,
          opacity: 0.45,
        }),
      );
    else
      g.append(
        svg("line", {
          x1: hx - 13,
          y1: hy,
          x2: hx + 13,
          y2: hy,
          stroke: "var(--ink-2)",
          "stroke-width": 5,
          "stroke-linecap": "round",
          opacity: 0.32,
        }),
      );
    // a bonded pair is one heater on two wires, so it is one dot with a second ring
    if (h.dacs.length > 1)
      g.append(
        svg("circle", {
          cx: hx,
          cy: hy,
          r: 9.5,
          fill: "none",
          stroke: SC[h.status] === "none" ? "var(--off)" : SC[h.status],
          "stroke-width": 1,
          opacity: 0.5,
        }),
      );
    const what =
      p.elem >= 0
        ? `MZI${p.elem} (rails ${S.elements[p.elem].rails.join(",")}, ` + `column ${p.col})`
        : `rail ${p.rails[0]}, column ${p.col}`;
    g.append(
      tip(
        svg("circle", {
          cx: hx,
          cy: hy,
          r: 6.5,
          fill: h.status === "grey" ? "var(--bg)" : SC[h.status],
          stroke: h.status === "grey" ? "var(--off)" : "var(--bg-2)",
          "stroke-width": 2,
        }),
        `${h.pad} · DAC ${h.dacs.join("+")} · ${ROLE[p.kind]} ${what} · ` +
          `${WORD[h.status]} · ${h.why}`,
      ),
    );
    const t = svg("text", {
      x: hx,
      y: hy - 13,
      "text-anchor": "middle",
      "font-size": 16,
      fill: "var(--ink-2)",
    });
    t.textContent = h.pad;
    g.append(t);
  }
  for (const p of S.pds) {
    g.append(
      tip(
        svg("rect", {
          x: PDX,
          y: RAILY(p.i) - 9,
          width: 18,
          height: 18,
          rx: 5,
          fill: p.status === "grey" ? "var(--bg)" : SC[p.status],
          stroke: p.status === "grey" ? "var(--off)" : "var(--bg-2)",
          "stroke-width": 2,
        }),
        `PD${p.i} · ${WORD[p.status]} · ${p.why}`,
      ),
    );
    const t = svg("text", {
      x: PDX + 26,
      y: RAILY(p.i) + 4,
      "font-size": 16,
      fill: "var(--ink-2)",
    });
    t.textContent = "PD" + p.i;
    g.append(t);
  }
  for (let c = 0; c < S.ncol; c++) {
    const t = svg("text", {
      x: cx(c),
      y: H - 8,
      "text-anchor": "middle",
      "font-size": 16,
      fill: "var(--dim)",
      "letter-spacing": ".14em",
    });
    t.textContent = "col " + c;
    g.append(t);
  }
  return g;
}

function calTables(S) {
  // state is the dot, detail is on hover: the row carries numbers, not words
  const dot = (h) =>
    `<span class="dot s-${h.status}" title="${esc(WORD[h.status] + ": " + h.why)}"></span>`;
  const sub = (r, i) => `${GREEK[r] || esc(r)}${i < 0 ? "" : `<sub>${i}</sub>`}`;
  const pdchip = (h) =>
    h.pd === null || h.pd === undefined
      ? ""
      : `<span class="chip" title="${esc(
          `fringe seen on PD${h.pd} with port ${h.port} lit` +
            (h.from_dac === null || h.from_dac === undefined
              ? ""
              : `; fit joined by pad from pre-rewire DAC ${h.from_dac}`),
        )}">${ico("pds")}${h.pd}</span>`;
  // the status dot and the reload button share one slot: hover a row and the dot becomes it
  const sw = (x, attr, what) =>
    `<span class="sw">${dot(x)}<button class="rl" ${attr} title="re-characterize this ${what}">${ico("reload")}</button></span>`;
  const pm = (v, sd, k, d) =>
    v === null || v === undefined
      ? "--"
      : `${f(k * v, d)}${sd === null || sd === undefined ? "" : ` <span class="dim">± ${f(k * sd, d)}</span>`}`;
  $("#htab").innerHTML =
    "<thead><tr><th>heater</th><th>DAC</th><th>role</th><th>span / π</th></tr></thead><tbody>" +
    S.heaters
      .map(
        (h) =>
          `<tr data-pad="${esc(h.pad)}"${h.disagree ? ' class="warnrow"' : ""}><td class="nm">${sw(h, `data-ch="${h.dacs.join(",")}"`, "heater")}${esc(h.pad)}${pdchip(h)}</td>` +
          `<td>${h.dacs.join("+")}</td><td>${sub(h.role, h.index)}</td>` +
          `<td title="fit span ${f(h.fit_span, 2)} π">${pm(h.span_pi, h.span_sd, 1, 2)}</td>` +
          `</tr>`,
      )
      .join("") +
    "</tbody>";
  $("#ptab").innerHTML =
    "<thead><tr><th>detector</th><th>full / mV</th><th>dark / mV</th><th>contrast / dB</th></tr></thead><tbody>" +
    S.pds
      .map(
        (p) =>
          `<tr data-pad="PD${p.i}"><td class="nm">${sw(p, `data-pd="${p.i}"`, "detector")}PD${p.i}</td>` +
          `<td>${pm(p.port_full_v, p.noise_v, 1e3, 1)}</td>` +
          `<td>${pm(p.dark_v, p.noise_v, 1e3, 3)}</td>` +
          `<td>${f(p.contrast_db, 1)}</td>` +
          `</tr>`,
      )
      .join("") +
    "</tbody>";
  paintRows();
  document
    .querySelectorAll(".rl[data-ch],.rl[data-pd]")
    .forEach((b) => (b.onclick = () => recal(b)));
}

// what each predictor can be refreshed with; the heaters have two ways, sweep and recenter
const PBTN = {
  heaters: [
    ["sweep", "full sweep", "Vpi and phi0 for every heater"],
    ["pd", "PD sweep", "every heater at 0 V: dark level and full scale per detector"],
    ["recenter", "recenter", "phi0 only, from the well; Vpi held"],
  ],
  table: [["table", "sync", "re-measure a few stored states, carry the table onto today"]],
  dpnn: [["dpnn", "train", "light the laser, sample the chip, fit the DPNN"]],
};
async function loadPreds() {
  let P;
  try {
    P = await (await fetch("/api/predictors")).json();
  } catch (e) {
    return;
  }
  if (P.error) return ($("#preds").textContent = why(P));
  $("#preds").innerHTML = P.map(
    (p) =>
      `<div class="dcard"><div class="dh">${ico(p.id === "heaters" ? "heater" : p.id)}${esc(p.name)}` +
      `<span class="grow"></span><span class="v-${p.status}">${dicon(p.status)}</span></div>` +
      dbody(p) +
      `<div class="calv">${PBTN[p.id]
        .map(
          ([l, t, tip]) =>
            `<button class="mini" data-level="${l}" title="${esc(tip)}">${ico("reload")}${t}</button>`,
        )
        .join("")}</div></div>`,
  ).join("");
  document
    .querySelectorAll("#preds [data-level], [data-level=all]")
    .forEach((b) => (b.onclick = () => recal(b)));
}

async function recal(b) {
  b.disabled = true;
  const body = b.dataset.level
    ? { level: b.dataset.level }
    : b.dataset.pd !== undefined
      ? { pd: +b.dataset.pd }
      : { channels: b.dataset.ch };
  body.mock = document.querySelector("#mk input:checked").value === "mock";
  try {
    const r = await fetch("/api/recal", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    $("#recal").textContent = d.ok ? `started: ${d.cmd}` : `${why(d)}\n${d.cmd || ""}`;
    if (d.ok) jobPoll();
  } catch (e) {
    $("#recal").textContent = String(e);
  }
  b.disabled = false;
}

function drawCal(S) {
  $("#cstamp").textContent = S.stamp;
  const box = $("#mesh");
  box.innerHTML = "";
  box.append(meshFig(S));
  $("#cwarn").textContent = S.warnings.join("\n");
  calTables(S);
  CAL = S;
} // last, so a render that threw is retried on the next poll rather than latched

async function loadCal() {
  try {
    const r = await fetch("/api/calib");
    const d = await r.json();
    if (d.error) {
      $("#cwarn").textContent = why(d);
      return;
    }
    if (CAL && CAL.rev === d.rev) return; // nothing moved, keep the DOM
    drawCal(d);
  } catch (e) {
    $("#cwarn").textContent = String(e);
  }
}

// Status by shape as well as by colour, so a row survives grayscale and colour blindness:
// check, bang, cross, a dashed ring for "no reading", a square for "no software can decide
// this". The five reuse the existing ok/warn/bad/off tokens and add no hue.
const DICO = {
  pass: '<circle cx="8" cy="8" r="6.4"/><path d="M5.2 8.3 7.3 10.4 10.9 5.9"/>',
  warn: '<path d="M8 2.6 14.4 13.4H1.6Z"/><path d="M8 6.4v3.1"/><path d="M8 11.5h.01"/>',
  fail: '<circle cx="8" cy="8" r="6.4"/><path d="M5.8 5.8 10.2 10.2M10.2 5.8 5.8 10.2"/>',
  unknown: '<circle cx="8" cy="8" r="6.4" stroke-dasharray="2.6 2.4"/>',
};
const WHAT = { pass: "pass", warn: "warn", fail: "fail", unknown: "not run" };
const dicon = (s) =>
  `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" ` +
  `stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" role="img" ` +
  `aria-label="${s}">${DICO[s] || DICO.unknown}</svg>`;

let DROWS = null,
  DARM = false,
  DBUSY = false;
const dmock = () => false; // diagnostics are always the bench: a mocked check is not a check

// "headline · label: value · free text". A label line is a quiet key/value, free text is a
// problem in the row's colour, "a → b" is drawn as a path and "a|b|c" as table cells.
function dbody(r) {
  const parts = r.value.split(" · ");
  // a first segment that is itself "label: value" means the row has no headline
  const head = parts[0].indexOf(": ") > 0 ? "" : parts.shift(),
    rest = parts;
  const kv = (v) => {
    const i = v.indexOf(": ");
    if (i < 0 || i > 24) return `<div class="dmsg v-${r.status}">${esc(v)}</div>`;
    const k = esc(v.slice(0, i)),
      val = v.slice(i + 2);
    if (val.includes("|"))
      return `<div class="dtr"><span>${k}</span>${val
        .split("|")
        .map((c) => `<span>${esc(c)}</span>`)
        .join("")}</div>`;
    const shown = val.includes(" → ")
      ? val
          .split(" → ")
          .map((p) => `<span class="chip">${esc(p)}</span>`)
          .join("<i>→</i>")
      : `<span class="chip">${esc(val)}</span>`;
    return `<div class="dkv"><span>${k}</span><span>${shown}</span></div>`;
  };
  return (head ? `<div class="dhead">${esc(head)}</div>` : "") + rest.map(kv).join("");
}

function drawDiag() {
  if (!DROWS) return;
  const st = (r) =>
    `<span class="v-${r.status}" title="${esc(WHAT[r.status] || r.status)}">${dicon(r.status)}</span>`;
  $("#dgrid").innerHTML = DROWS.filter((r) => r.id !== "rail")
    .map(
      (r) =>
        `<div class="dcard" title="${esc(r.tip)}"><div class="dh">${ico(r.id)}${esc(r.name)}` +
        `<span class="grow"></span>${st(r)}<button class="rl" data-d="${r.id}"` +
        `${DBUSY ? " disabled" : ""} title="re-run this check">${ico("reload")}</button></div>` +
        dbody(r) +
        `</div>`,
    )
    .join("");
  const rail = DROWS.find((r) => r.id === "rail");
  if (rail)
    $("#drailv").innerHTML =
      `<div class="dcard"><div class="dh">${st(rail)}</div>` + dbody(rail) + `</div>`;
  $("#dall").disabled = $("#drail").disabled = DBUSY;
  $("#dgrid")
    .querySelectorAll(".rl[data-d]")
    .forEach((b) => (b.onclick = () => drun([b.dataset.d])));
}
$("#drail").onclick = () => drun(["rail"]);

// Every request is explicit. There is no poll on this tab: the switch row walks the mirror
// and the rail row puts every heater at its clamp.
async function drun(ids) {
  if (DBUSY) return;
  const mock = dmock();
  if (
    ids.includes("rail") &&
    !confirm("Power test: every heater at its clamp for 30 s while the TEC is watched. Continue?")
  )
    return;
  DBUSY = true;
  ids.forEach((i) => {
    const r = DROWS.find((x) => x.id === i);
    if (r) {
      r.status = "unknown";
      r.value = "running…";
    }
  });
  drawDiag();
  try {
    const r = await fetch("/api/diag", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ checks: ids, mock: mock }),
    });
    const d = await r.json();
    if (d.error) $("#dnote").textContent = why(d);
    else {
      $("#dnote").textContent = "";
      d.rows.forEach((n) => {
        const k = DROWS.findIndex((x) => x.id === n.id);
        if (k >= 0) DROWS[k] = n;
      });
    }
  } catch (e) {
    $("#dnote").textContent = String(e);
  }
  DBUSY = false;
  drawDiag();
}

async function dload() {
  try {
    const r = await fetch("/api/diag");
    const d = await r.json();
    if (d.error) {
      $("#dnote").textContent = why(d);
      return;
    }
    DROWS = d.rows;
    DARM = !!d.armed;
    drawDiag();
  } catch (e) {
    $("#dnote").textContent = String(e);
  }
}
$("#dall").onclick = () => DROWS && drun(DROWS.filter((r) => r.auto).map((r) => r.id));

// Which page this is comes from the server, not a click, so a reload lands back here.
const CUR = document.body.dataset.page;
// Bench or mock is one choice for the whole app, so it lives in the navbar and survives a
// page change. Guarded: storage can throw or come back empty.
const MKEY = "pic4x4.ui.mode";
try {
  const m = localStorage.getItem(MKEY);
  if (m) document.querySelector(`#mk input[value="${m}"]`).checked = true;
} catch (e) {}
document.querySelectorAll("#mk input").forEach(
  (i) =>
    (i.onchange = () => {
      document.body.classList.toggle("bench", i.value === "hw");
      try {
        localStorage.setItem(MKEY, i.value);
      } catch (e) {}
    }),
);
document.body.classList.toggle("bench", document.querySelector("#mk input:checked").value === "hw");
$("#t-" + CUR).classList.add("on");
if (CUR === "cal")
  POLL = setInterval(() => {
    if (!document.hidden) loadCal();
  }, 4000);
if (CUR === "diag") dload(); // the catalogue only; nothing is measured

setU(haar());
HIST = hload();
drawHist();
if (CUR === "cal") {
  loadCal();
  loadPreds();
}

// Always the bench: a mock laser would forget its state between requests.
// On any error the tint is left alone: "cannot tell" must never read as "off".
const lz = (b) =>
  fetch("/api/laser", b && { method: "POST", body: JSON.stringify(b) })
    .then((r) => r.json())
    .then((d) => {
      if (d.held) {
        // lit/unlit from the job's own log; null means nobody can tell, so the tint stays
        if (d.on !== null) document.body.classList.toggle("lasing", d.on);
        const s = d.on === null ? "unknown" : d.on ? "on" : "off";
        $("#lzs").innerHTML = `<span class="v-warn">${s} · held by ${esc(d.held)}</span>`;
        return;
      }
      if (d.error) {
        $("#lzs").innerHTML =
          `<span class="v-fail">[${esc(d.where || "laser")}] ${esc(d.error)}</span>`;
        return;
      }
      const on = !!d.on;
      $("#lzon").classList.toggle("on", on);
      document.body.classList.toggle("lasing", on);
      $("#lzon b").textContent = on ? "on" : "off";
      // the key is an icon: green on, red off; the text is only what needs reading
      $("#lzkey").className = `ico key v-${d.key ? "pass" : "fail"}`;
      $("#lzkey").title = `key ${d.key ? "on" : "off"}`;
      $("#lzs").innerHTML =
        (d.bnc && d.ext ? "" : `<span class="v-fail">interlock open</span>`) +
        `<span>${d.mA} mA</span>`;
    })
    .catch((e) => {
      $("#lzs").innerHTML =
        `<span class="v-fail">[UI server] not reachable: ${esc(String(e))}</span>`;
    });
$("#lzon").onclick = () =>
  lz({ on: !$("#lzon").classList.contains("on"), dbm: +$("#lzdbm").value });
$("#lzdbm").onchange = () =>
  $("#lzon").classList.contains("on") && lz({ on: true, dbm: +$("#lzdbm").value });
lz();
// ponytail: polls the serial port every 5 s; push from the server if that ever contends with a run
setInterval(lz, 5000);

// Four global show/hide flags, one per bar series, applied to every run as body classes.
const BKEY = "pic4x4.ui.bars";
const BARS = ["full", "tiled", "cpu", "dpnn"];
function setBars(hidden) {
  BARS.forEach((b) => document.body.classList.toggle("hide-" + b, hidden.includes(b)));
  document.querySelectorAll("#bars input").forEach((i) => (i.checked = !hidden.includes(i.value)));
  try {
    localStorage.setItem(BKEY, JSON.stringify(hidden));
  } catch (e) {}
}
let HIDDEN = [];
try {
  HIDDEN = JSON.parse(localStorage.getItem(BKEY) || "[]").filter((b) => BARS.includes(b));
} catch (e) {}
document
  .querySelectorAll("#bars input")
  .forEach(
    (i) =>
      (i.onchange = () =>
        setBars(
          [...document.querySelectorAll("#bars input")]
            .filter((x) => !x.checked)
            .map((x) => x.value),
        )),
  );
setBars(HIDDEN);

// One calibration job at a time. Every page polls it; body classes then switch off whatever
// would fight the job for a port: `job` for any job, `jobhw` for one on the bench.
let JOBWAS = false,
  JOBROWS = {};
// a row the running job will touch: queued (dimmed), running (spinner in the dot's slot),
// done (checked until the job ends and the new fit is read back)
function paintRows() {
  document.querySelectorAll("tr[data-pad]").forEach((tr) => {
    ["queued", "running", "done"].forEach((s) =>
      tr.classList.toggle("q-" + s, JOBROWS[tr.dataset.pad] === s),
    );
  });
}
async function jobPoll() {
  let d;
  try {
    d = await (await fetch("/api/job")).json();
  } catch (e) {
    return;
  }
  const run = !!d.running,
    hw = run && !d.mock;
  document.body.classList.toggle("job", run);
  document.body.classList.toggle("jobhw", hw);
  const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  // expected is this job's median past duration on this rig; none until it has run once
  const times =
    `elapsed ${mmss(d.elapsed || 0)}` +
    (d.expected
      ? ` · expected ${mmss(d.expected)} · remaining ${run ? mmss(Math.max(0, d.expected - d.elapsed)) : "0:00"}`
      : " · expected unknown (first run)");
  $("#job").classList.toggle("hide", !run && !d.external);
  const ch = d.chain && d.chain.running ? ` · step ${d.chain.at + 1}/${d.chain.steps.length}` : "";
  $("#jobt").textContent = run
    ? `${d.what}${d.mock ? " (mock)" : ""}${ch} · ${mmss(d.elapsed || 0)}` +
      (d.expected ? ` of ~${mmss(d.expected)}` : "")
    : d.external
      ? `pic process outside this page: pid ${d.external.join(", ")}`
      : "";
  $("#jobstop").classList.toggle("hide", !run);
  if (d.cmd) {
    $("#joblog").classList.remove("hide");
    $("#joblog").textContent =
      `$ ${d.cmd}\n${times}\n\n` +
      ((d.tail || []).length ? d.tail.join("\n") : run ? "(starting…)" : "") +
      (run ? "" : `\n-- ${d.rc === 0 ? "finished" : "exited " + d.rc} · log ${d.log}`);
  }
  JOBROWS = run ? d.rows || {} : {};
  paintRows();
  if (JOBWAS && !run && CUR === "cal") {
    loadCal(); // new fits are on disk
    loadPreds();
  }
  JOBWAS = run;
}
$("#jobstop").onclick = async () => {
  if (!confirm("Stop the calibration job? Its partial results are not written.")) return;
  await fetch("/api/job/stop", { method: "POST", body: "{}" });
  jobPoll();
};
jobPoll();
setInterval(jobPoll, 2000);

// Die temperature in the header: the mean of 5 TEC readings, every 5 s. While a job or run
// holds the TEC port the last value stays, dimmed, rather than a second reader splitting it.
async function tecPoll() {
  const e = $("#tect");
  try {
    const d = await (await fetch("/api/tec")).json();
    if (d.held) {
      e.style.opacity = 0.5;
      e.title = "TEC port held by a running job; showing the last reading";
      return;
    }
    e.style.opacity = 1;
    if (d.error) {
      e.textContent = "TEC ?";
      e.className = "v-fail";
      e.title = why(d);
      return;
    }
    e.textContent = `${d.c.toFixed(2)} °C`;
    e.className = Math.abs(d.c - d.set) <= d.tol ? "v-pass" : "v-warn";
    e.title = `die ${d.c} °C, setpoint ${d.set} ± ${d.tol}, TEC drive ${d.drive} V (mean of ${d.n})`;
  } catch (err) {
    e.textContent = "TEC ?";
    e.className = "v-fail";
  }
}
tecPoll();
setInterval(tecPoll, 5000);
