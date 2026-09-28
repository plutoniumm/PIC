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
$("#eye").onclick = () =>
  setU([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

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
    c.querySelector(".mini").style.visibility =
      cs.length > 1 ? "visible" : "hidden";
  });
  $("#addx").disabled = cs.length >= MAXC;
  $("#xnote").textContent = cs.length + "/" + MAXC;
}
function setX(rows) {
  cols().forEach((c) => c.remove());
  (rows && rows.length ? rows : [[1, 0, 0, 0]])
    .slice(0, MAXC)
    .forEach((r) => addCol(r));
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
    [...c.querySelectorAll("input")].forEach(
      (e) => (e.value = gauss().toFixed(3)),
    ),
  );
const xv = () =>
  cols().map((c) =>
    [...c.querySelectorAll("input")].map((e) => parseFloat(e.value) || 0),
  );

const f = (v, d) =>
  v === null || v === undefined || !isFinite(v)
    ? "--"
    : v.toFixed(d === undefined ? 4 : d);
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

const pc = (e) => (isFinite(e) ? (e * 100).toFixed(1) + "%" : "--");

// the colour, icon and name a hosting goes by on the page
const WHO = { full: "full", tile: "tiled" };
const BARC = { full: "var(--s1)", tiled: "var(--s2)" };

function railChart(v, who, label, rails, W) {
  // Two bars per output rail: the PIC in the hosting Settings holds, and the CPU's ideal
  // for it. Full can only measure |U|^2 x, so its ideal is that, not Ux.
  // Fixed axis -1..1: x is unit norm, so every exact output is inside it; a noisy reading
  // past it is clipped here and shown exactly in the tables.
  // W is the drawn width in px, so 16 in the viewBox is 16 on screen.
  const P = v.pic;
  const x0 = 44,
    x1 = W - 2,
    y0 = 10,
    y1 = 128,
    H = y1 + 48;
  const g = svg("svg", {
    class: "fig",
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label": `${who} and cpu per output rail for ${label}`,
  });
  const lo = -1,
    hi = 1;
  const clip = (q) => Math.max(lo, Math.min(hi, q));
  const sy = (q) => y1 - ((clip(q) - lo) / (hi - lo)) * (y1 - y0),
    zy = sy(0);
  for (const q of [lo, 0, hi]) {
    if (q !== 0)
      g.append(
        svg("line", { x1: x0, y1: sy(q), x2: x1, y2: sy(q), class: "gl" }),
      );
    const t = svg("text", {
      x: x0 - 8,
      y: sy(q) + 5,
      "text-anchor": "end",
      class: "ax",
    });
    t.textContent = q.toFixed(1);
    g.append(t);
  }
  g.append(svg("line", { x1: x0, y1: zy, x2: x1, y2: zy, class: "zl" }));
  const n = P.ideal.length,
    gw = (x1 - x0) / n,
    bw = Math.min(26, gw * 0.24);
  const rel = (m, i) =>
    Math.abs(i) > 1e-9 ? Math.abs(m - i) / Math.abs(i) : NaN;
  for (let r = 0; r < n; r++) {
    const cx = x0 + gw * (r + 0.5),
      out = rails && rails.length > r ? rails[r] : r;
    [
      [P.device[r], P.ideal[r], BARC[who], 0, who],
      [P.ideal[r], null, "var(--ref)", 1, "cpu"],
    ].forEach(([q, ideal, fill, k, w]) => {
      if (q === undefined || q === null || !isFinite(q)) return;
      const x = cx + (k - 1) * (bw + 3) + 1.5, // the pair centred on the rail
        y = Math.min(zy, sy(q)),
        h = Math.abs(sy(q) - zy) || 1;
      const why =
        ideal === null || !isFinite(ideal)
          ? ""
          : ` · ${pc(rel(q, ideal))} off its ideal ${ideal.toFixed(4)}`;
      g.append(
        tip(
          svg("path", {
            class: "bar-" + w,
            d: capBar(x, y, bw, h, 4, q >= 0),
            fill: fill,
            stroke: "var(--bg-2)",
            "stroke-width": 1.5,
          }),
          `${w} out ${out} = ${q.toFixed(4)}${why}`,
        ),
      );
    });
    const t = svg("text", {
      x: cx,
      y: y1 + 40,
      "text-anchor": "middle",
      class: "ax",
    });
    t.textContent = "out " + out;
    g.append(t);
  }
  return g;
}

// y per output row, one column per x: what the chart draws, exact and unclipped
const sg = (q) => (q === null || !isFinite(q) ? "--" : (q >= 0 ? "+" : "") + q.toFixed(2));
function yTable(V, rails, who) {
  // one table: per x, the PIC reading beside the CPU's ideal
  return (
    `<table class="yt"><thead><tr><th></th>` +
    V.map((v, i) => `<th colspan="2">x${i + 1}</th>`).join("") +
    `</tr><tr><th></th>` +
    V.map(() => `<th class="bar-${who}">PIC</th><th class="bar-cpu">CPU</th>`).join("") +
    `</tr></thead><tbody>` +
    [0, 1, 2, 3]
      .map(
        (r) =>
          `<tr><td>${rails[r]}</td>` +
          V.map((v) => `<td>${sg(v.pic.device[r])}</td><td>${sg(v.pic.ideal[r])}</td>`).join("") +
          `</tr>`,
      )
      .join("") +
    `</tbody></table>`
  );
}

function resultBody(e, onx, cw) {
  const d = e.d,
    V = d.vectors,
    who = WHO[d.hosting],
    box = document.createElement("div");
  box.className = "hb";
  const meta = document.createElement("div");
  meta.className = "hm";
  meta.innerHTML = `<span class="grow"></span><button class="rl vt" title="U and x">${ico("table")}</button>`;
  // tables to the right of the charts when there is room, under them when not
  const tw = 40 + 110 * V.length,
    side = cw - tw - 20 >= 320,
    aw = side ? cw - tw - 20 : cw;
  // two charts a row when there is room; each is drawn at the width it will be shown at
  const two = aw >= 760 && V.length > 1,
    W = Math.max(300, Math.floor(two ? (aw - 20) / 2 : aw));
  const main = document.createElement("div");
  main.className = "hmain" + (side ? " side" : "");
  const charts = document.createElement("div");
  charts.className = "charts" + (two ? " two-up" : "");
  V.forEach((v, i) => {
    const c = document.createElement("div");
    c.className = "chart";
    const s = v.pic;
    c.innerHTML =
      `<div class="chd"><button class="rl vx" title="load x${i + 1} alone into the inputs">` +
      `x${i + 1}</button>` +
      (v.null
        ? `<span class="note">zero, not plotted</span>`
        : `<span class="bar-${who}" title="${who}: fidelity |<measured,ideal>| / (||measured|| ||ideal||), ` +
          `relative error ${f(s.rel, 3)}, signs kept ${pc(s.sign)}">` +
          `${mico(who)} ${f(s.fid, 3)}</span>`) +
      `</div>`;
    c.querySelector(".vx").onclick = () => onx(e.X[i]);
    if (!v.null) c.append(railChart(v, who, "x" + (i + 1), d.rails_out, W));
    charts.append(c);
  });
  const tabs = document.createElement("div");
  tabs.className = "ytabs";
  tabs.innerHTML = yTable(V, d.rails_out, who);
  main.append(charts, tabs);
  // the inputs behind the picture: U and x, folded away
  const more = document.createElement("div");
  more.className = "more hide";
  more.innerHTML =
    `<div class="two"><div><div class="eyebrow">U</div><div class="umini">` +
    e.U.map((q) => `<div>${f(q, 3)}</div>`).join("") +
    `</div></div><div><div class="eyebrow">x</div><div class="umini" ` +
    `style="grid-template-columns:repeat(${e.X.length},1fr);max-width:${80 * e.X.length}px">` +
    [0, 1, 2, 3]
      .map((r) => e.X.map((c) => `<div>${f(c[r], 3)}</div>`).join(""))
      .join("") +
    `</div></div></div>`;
  // the meta row is lifted into the run's summary, so a click here must not fold the run
  meta.querySelector(".vt").onclick = (ev) => {
    ev.preventDefault();
    ev.currentTarget.classList.toggle("on", !more.classList.toggle("hide"));
  };
  box.append(meta, main, more);
  return box;
}

// Cap: HCAP runs. Each entry is the request plus the server's reply, a few kB; the cap is
// there so a long session cannot fill the origin's quota, and a quota throw on write halves
// the list rather than losing it. Every read and write is guarded: private windows, cleared
// site data and blocked storage all throw or hand back nothing.
const HKEY = "pic4x4.ui.runs.v3"; // v2 carried both hostings per run
let HIST = [];
function hload() {
  try {
    const s = localStorage.getItem(HKEY);
    if (!s) return [];
    const a = JSON.parse(s);
    // an entry from an older page version would throw inside drawHist and blank the tab
    return Array.isArray(a)
      ? a
          .filter((e) => e && e.d && WHO[e.d.hosting] && Array.isArray(e.d.vectors) && e.U && e.X)
          .slice(0, HCAP)
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

// full = the mesh holding all of U (a square); tiled = U in four 2x2 blocks (a square in four)
const mico = (who) =>
  `<i class="ico mico s-${who}" title="${who === "full" ? "full: one program holds all of U" : "tiled: U as four 2x2 blocks, summed on the host"}" style="--i:url(/icons/${who}.svg)"></i>`;
// One number per run in the run row: its fidelity, the median over the x vectors of
// |<measured, ideal>| / (|measured| |ideal|). 1 is perfect.
function spark(e) {
  const V = e.d.vectors.filter((v) => !v.null),
    who = WHO[e.d.hosting];
  const fs = V.map((v) => v.pic.fid)
    .filter(isFinite)
    .sort((a, b) => a - b);
  const med = fs.length ? fs[Math.floor((fs.length - 1) / 2)] : NaN;
  // ponytail: fixed fidelity tiers, a judgment call and not a server gate; move them if a
  // good bench run reads as weak
  const tierF = (x) =>
    !isFinite(x) ? "bad" : x >= 0.95 ? "ok" : x >= 0.8 ? "warn" : "bad";
  const t = `${who} fidelity, median over x\nper x: ${V.map((v) => f(v.pic.fid, 3)).join(" ")}`;
  return `<span class="q" title="${esc(t)}">${mico(who)}<b class="t-${tierF(med)}">${f(med, 3)}</b></span>`;
}
function summary(e) {
  const t = new Date(e.t);
  return (
    `<span class="when" title="${t.toLocaleString()}">${agoTxt((Date.now() - e.t) / 60000)}</span>` +
    `<i class="mode ${e.mock ? "m-mock" : "m-bench"}" title="${e.mock ? "mock" : "bench"}"></i>` +
    spark(e)
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
    rb.innerHTML = ico("reload");
    rb.title = "restore run: put this U and every x back into the inputs";
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
      const b = resultBody(e, restoreX, box.clientWidth - 28);
      s.insertBefore(b.querySelector(".hm"), rb); // the values toggle, shown while open
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
  if (hotBench()) return;
  if (document.body.classList.contains("bench"))
    document.body.classList.add("lasing");
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
      body: JSON.stringify({
        U: U0,
        X: X0,
        mock: mock,
        dbm: +$("#lzdbm").value,
      }),
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
  lz(); // the run switched the laser off; show it now, not at the next poll
};

let CAL = null,
  POLL = null;
const RAILY = (r) => 44 + 46 * r;
const MW = 24; // half-width of an element box
const PHI_OFF = MW + 18; // a phi sits this far left of its element, on the upper arm
const SC = {
  green: "var(--ok)",
  yellow: "var(--warn)",
  red: "var(--bad)",
  grey: "none",
};
const WORD = { green: "good", yellow: "weak", red: "broken", grey: "uncal" };
const GREEK = { theta: "θ", phi: "φ", alpha: "α" };

function calTables(S) {
  // state is the dot, detail is on hover: the row carries numbers, not words
  const dot = (h) =>
    `<span class="dot s-${h.status}" title="${esc(WORD[h.status] + ": " + h.why)}"></span>`;
  const sub = (r, i) =>
    `${GREEK[r] || esc(r)}${i < 0 ? "" : `<sub>${i}</sub>`}`;
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
    "<thead><tr><th>heater</th><th>DAC</th><th>role</th><th>φ<sub>0</sub> / π</th><th>span / π</th></tr></thead><tbody>" +
    S.heaters
      .map(
        (h) =>
          `<tr data-pad="${esc(h.pad)}"${h.disagree ? ' class="warnrow"' : ""}><td class="nm">${sw(h, `data-ch="${h.dacs.join(",")}"`, "heater")}${esc(h.pad)}${pdchip(h)}</td>` +
          `<td>${h.dacs.join("+")}</td><td>${sub(h.role, h.index)}</td>` +
          // the phase offset the heater law programs volts from, as swept and fitted
          `<td class="c-phi">${pm(h.phi0 === null || h.phi0 === undefined ? null : h.phi0 / Math.PI, h.phi0_sd === undefined || h.phi0_sd === null ? null : h.phi0_sd / Math.PI, 1, 3)}</td>` +
          `<td class="c-span" title="fit span ${f(h.fit_span, 2)} π">${pm(h.span_pi, h.span_sd, 1, 2)}</td>` +
          `</tr>`,
      )
      .join("") +
    "</tbody>";
  if (S.spds) {
    // SPD mode: dark and each input's reference from `pic spdnorm`, counts/s
    const c = (x) => (x == null ? "–" : Math.round(x).toLocaleString());
    $("#ptab").innerHTML =
      "<thead><tr><th>detector</th><th>dark /s</th><th>P0 /s</th><th>P1 /s</th><th>P2 /s</th><th>P3 /s</th></tr></thead><tbody>" +
      S.spds
        .map(
          (p) =>
            `<tr data-pad="PD${p.i}" title="${p.id}${p.at ? " · " + p.at : ""}"><td class="nm">PD${p.i} ${p.id}</td>` +
            `<td>${c(p.dark)}</td>` +
            p.ports.map((r) => `<td>${c(r)}</td>`).join("") +
            `</tr>`,
        )
        .join("") +
      "</tbody>";
  } else
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
let EST = {};
const estTxt = (s) =>
  s == null
    ? "~?"
    : s < 90
      ? `~${Math.round(s)} s`
      : `~${Math.round(s / 60)} min`;
const PBTN = {
  heaters: [
    ["sweep", "sweep", "play", "Vpi and phi0 for every heater"],
    ["recenter", "recenter", "reload", "phi0 only, from the well; Vpi held"],
  ],
  // a mode without photodiodes (SPD) will simply not list this card
  pds: [
    [
      "pd",
      "sweep",
      "play",
      "PD: dark and full scale · SPD: dark, then each input once",
    ],
  ],
  table: [
    [
      "table",
      "sync",
      "reload",
      "re-measure a few stored states, carry the table onto today",
    ],
    [
      "capture",
      "recapture",
      "play",
      "a fresh table: 200 random states, all four ports each",
    ],
  ],
};
const PICO = { heaters: "heater", pds: "pds", table: "table" };
const mmss = (t) =>
  `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
let PJOB = {}; // the last /api/job, so a re-render of the tiles keeps the running one running

// minutes since "N min ago" or a "YYYY-MM-DD hh:mm" stamp; null if neither
function ageMin(v) {
  const t = Date.parse(v.replace(" ", "T"));
  if (!isNaN(t)) return (Date.now() - t) / 60000;
  const g = v.match(/([\d.]+)\s*(s|sec|min|h|hr|hrs|hour|hours|day|days)\b/);
  return g
    ? +g[1] *
        ({ s: 1 / 60, sec: 1 / 60, min: 1, day: 1440, days: 1440 }[g[2]] ?? 60)
    : null;
}
const agoTxt = (m) =>
  m < 1
    ? "just now"
    : m < 90
      ? `${Math.round(m)} min ago`
      : m < 1440
        ? `${Math.round(m / 60)} hrs ago`
        : m < 2880
          ? "1 day ago"
          : `${Math.round(m / 1440)} days ago`;

// Quality as a fill against its bar: [shown number, unit, fill 0..1, threshold 0..1, segments].
const QUAL = {
  heaters: (v) => {
    const m = v.match(/(\d+)\/(\d+)/);
    return m && [`${m[1]}/${m[2]}`, "fitted", m[1] / m[2], null, +m[2]];
  },
  pds: (v) => {
    const m = v.match(/(\d+)\/(\d+) detectors/);
    return m && [`${m[1]}/${m[2]}`, "healthy", m[1] / m[2], null, +m[2]];
  },
  table: (v) => {
    const m = v.match(/error ([\d.]+)/); // 1.0 is what answering zero scores
    return m && [m[1], "error", Math.max(0, 1 - m[1]), 0.5];
  },
};
// the problem sentence: whatever segment is not a fact
const pprob = (v) =>
  v
    .split(" · ")
    .slice(1)
    .filter((s) => !s.includes(": ") && !/^R²/.test(s));

// A ring that drains over a day: the map moves overnight, so age is the first thing to see.
// While the tile's job runs the same ring becomes its progress.
const ring = (f, cls) =>
  `<svg class="ring ${cls}" viewBox="0 0 44 44"><circle class="rt" cx="22" cy="22" r="19" pathLength="100"/>` +
  `<circle class="rf" cx="22" cy="22" r="19" pathLength="100" stroke-dasharray="${(f * 100).toFixed(1)} 100"/></svg>`;

function ptile(p) {
  const seg =
    p.value.split(" · ").find((s) => /ago$|^age: |^last /.test(s)) || "";
  const m = ageMin(seg.slice(seg.indexOf(": ") + 2));
  const fresh = m == null ? 0 : Math.max(0, 1 - m / 1440);
  const q = (QUAL[p.id] || (() => null))(p.value);
  const btns = PBTN[p.id] || [];
  // every action says how long it takes, from this bench's own past runs
  const dur = (l) => `<span class="pdur">${estTxt(EST[l])}</span>`;
  const btn = ([l, t, i, tip]) =>
    p.rec === l
      ? `<button class="go pgo" data-level="${l}" title="${esc(tip)}">${ico(i)}${t}${dur(l)}</button>`
      : `<button class="mini pgh" data-level="${l}" title="${esc(`${t} (${estTxt(EST[l])}): ${tip}`)}" aria-label="${esc(t)}">${ico(i)}${dur(l)}</button>`;
  const meter = !q
    ? ""
    : q[4]
      ? `<div class="pseg">${Array.from({ length: q[4] }, (_, i) => `<i class="${i < q[2] * q[4] ? "on" : ""}"></i>`).join("")}</div>`
      : `<div class="pmeter"><i style="width:${(q[2] * 100).toFixed(0)}%"></i><s style="left:${q[3] * 100}%"></s></div>`;
  return (
    `<div class="ptile s-${p.status}${p.rec ? " need" : ""}" data-p="${p.id}">` +
    `<div class="ptop"><div class="pring" title="${m == null ? "age unknown" : "measured " + agoTxt(m)}">` +
    ring(fresh, fresh > 0.5 ? "f-ok" : fresh > 0 ? "f-mid" : "f-old") +
    ico(PICO[p.id] || p.id) +
    `<span class="pst v-${p.status}" title="${esc(WHAT[p.status] || p.status)}">${dicon(p.status)}</span>` +
    `</div><div class="pid"><div class="pn">${esc(p.name)}</div>` +
    `<div class="pa">${m == null ? "" : agoTxt(m)}</div></div></div>` +
    (q
      ? `<div class="pq"><b>${esc(q[0])}</b><small>${esc(q[1])}</small></div>` +
        meter
      : "") +
    pprob(p.value)
      .map((t) => `<div class="ptag v-${p.status}">${esc(t)}</div>`)
      .join("") +
    // actions: quiet related ones as one icon group, the recommended one last, on the right
    `<div class="calv pact"><span class="grow"></span>` +
    (btns.some((b) => b[0] !== p.rec)
      ? `<span class="bgroup">${btns
          .filter((b) => b[0] !== p.rec)
          .map(btn)
          .join("")}</span>`
      : "") +
    btns
      .filter((b) => b[0] === p.rec)
      .map(btn)
      .join("") +
    `</div></div>`
  );
}

const LTILE = {
  sweep: "heaters",
  recenter: "heaters",
  pd: "pds",
  table: "table",
  capture: "table",
};
// progress 0..1, or null for "running, length unknown"; the server's own figure first
function jobFrac(d) {
  if (d.progress != null) return Math.min(1, d.progress);
  for (const l of [...(d.tail || [])].reverse()) {
    const g = l.match(/(\d+)\/(\d+) states|prescan (\d+)\/(\d+)/);
    if (g) return Math.min(1, (g[1] ?? g[3]) / (g[2] ?? g[4]));
  }
  return d.expected ? Math.min(1, (d.elapsed || 0) / d.expected) : null;
}
function paintBusy() {
  const d = PJOB,
    run = !!d.running,
    f = run && !d.waiting ? jobFrac(d) : null,
    id = run ? LTILE[d.level] : null;
  const left =
    f && f > 0.02
      ? `${mmss(((d.elapsed || 0) * (1 - f)) / f)} left`
      : mmss(d.elapsed || 0);
  document.querySelectorAll("#preds .ptile").forEach((t) => {
    const on = t.dataset.p === id;
    t.classList.toggle("busy", on);
    t.classList.toggle("ind", on && f == null);
    t.style.setProperty("--pf", (f ?? 0).toFixed(3));
    const rf = t.querySelector(".ring .rf"),
      pa = t.querySelector(".pa");
    if (on) {
      if (!t.dataset.age) t.dataset.age = pa.textContent;
      rf.setAttribute(
        "stroke-dasharray",
        `${f == null ? 25 : (f * 100).toFixed(1)} 100`,
      );
      // the job's own units first (round 2/4 · 120/200 pts), then the percentage and time left
      const stage = d.stage ? `${d.stage} · ` : "";
      pa.textContent = d.waiting
        ? `${stage}TEC ${$("#tect").textContent}`
        : f == null
          ? `${stage}${left}`
          : `${stage}${Math.round(f * 100)}% · ${left}`;
    } else if (t.dataset.age != null) {
      pa.textContent = t.dataset.age;
      delete t.dataset.age;
    }
  });
  document
    .querySelectorAll("[data-level]")
    .forEach((b) =>
      b.classList.toggle("spin", run && b.dataset.level === d.level),
    );
}

async function loadPreds() {
  let P;
  try {
    P = await (await fetch("/api/predictors")).json();
  } catch (e) {
    return;
  }
  if (P.error) return ($("#preds").textContent = why(P));
  if (!Array.isArray(P)) {
    EST = P.est || {};
    P = P.preds;
  }
  const all = $("[data-level=all]");
  if (all) {
    all.querySelector(".pdur")?.remove();
    all.insertAdjacentHTML(
      "beforeend",
      `<span class="pdur">${estTxt(EST.all)}</span>`,
    );
  }
  $("#preds").innerHTML = P.map(ptile).join("");
  paintBusy();
  document
    .querySelectorAll("#preds [data-level], [data-level=all]")
    .forEach((b) => (b.onclick = () => recal(b)));
}

const hotBench = () =>
  document.body.classList.contains("hot") &&
  document.body.classList.contains("bench");
async function recal(b) {
  if (hotBench()) return;
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
    $("#recal").textContent = d.ok ? "" : why(d); // a start shows on its button, not as text
    if (d.ok) jobPoll();
  } catch (e) {
    $("#recal").textContent = String(e);
  }
  b.disabled = false;
}

function drawCal(S) {
  // a pic process outside this page may be writing these files: lock, do not warn
  document.body.classList.toggle("locked", !!S.locked);
  calTables(S);
  CAL = S;
} // last, so a render that threw is retried on the next poll rather than latched

async function loadCal() {
  try {
    const r = await fetch("/api/calib");
    const d = await r.json();
    if (d.error) return;
    if (CAL && CAL.rev === d.rev) return; // nothing moved, keep the DOM
    drawCal(d);
  } catch (e) {} // the next poll retries
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
    if (i < 0 || i > 24)
      return `<div class="dmsg v-${r.status}">${esc(v)}</div>`;
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
  return (
    // a lone headline on a warn or fail is the fix itself, so it takes the row's colour
    (head
      ? `<div class="dhead${rest.length || !/warn|fail/.test(r.status) ? "" : " v-" + r.status}">${esc(head)}</div>`
      : "") +
    rest.map(kv).join("")
  );
}

function drawDiag() {
  if (!DROWS) return;
  const st = (r) =>
    `<span class="v-${r.status}" title="${esc(WHAT[r.status] || r.status)}">${dicon(r.status)}</span>`;
  $("#dgrid").innerHTML = DROWS.filter((r) => r.id !== "rail")
    .map(
      (r) =>
        `<div class="dcard" title="${esc(r.detail && r.detail !== r.value ? r.detail + "\n\n" + r.tip : r.tip)}"><div class="dh">${ico(r.id)}${esc(r.name)}` +
        `<span class="grow"></span><button class="rl" data-d="${r.id}"` +
        `${DBUSY ? " disabled" : ""} title="re-run this check">${ico("reload")}</button>${st(r)}</div>` +
        dbody(r) +
        `</div>`,
    )
    .join("");
  const rail = DROWS.find((r) => r.id === "rail");
  if (rail)
    $("#drailv").innerHTML =
      `<div class="dcard dline">${st(rail)}<div>${dbody(rail)}</div></div>`;
  $("#dall").disabled = $("#drail").disabled = DBUSY;
  $("#dgrid")
    .querySelectorAll(".rl[data-d]")
    .forEach((b) => (b.onclick = () => drun([b.dataset.d])));
}
$("#drail").onclick = () => drun(["rail"]);

// Every request is explicit except the DAC pins row, re-read every 5 min: it only writes a
// few sub-mA codes and zeroes them, and the server refuses it while anything holds the board.
// The switch row walks the mirror and the rail row puts every heater at its clamp.
setInterval(() => DROWS && !DBUSY && drun(["dac"]), 300000);
async function drun(ids) {
  if (DBUSY) return;
  const mock = dmock();
  if (
    ids.includes("rail") &&
    !confirm(
      "Power test: every heater at a quarter of its power for 30 s while the TEC is watched. Continue?",
    )
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
    // a check started before this page loaded: stay busy and re-read until it lands
    DBUSY = DROWS.some((r) => r.running);
    if (DBUSY) setTimeout(dload, 2000);
    drawDiag();
  } catch (e) {
    $("#dnote").textContent = String(e);
  }
}
$("#dall").onclick = () =>
  DROWS && drun(DROWS.filter((r) => r.auto).map((r) => r.id));

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
document.body.classList.toggle(
  "bench",
  document.querySelector("#mk input:checked").value === "hw",
);
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
        $("#lzs").innerHTML =
          `<span class="v-warn">${s} · held by ${esc(d.held)}</span>`;
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
  $("#lzon").classList.contains("on") &&
  lz({ on: true, dbm: +$("#lzdbm").value });
lz();
// ponytail: polls the serial port every 5 s; push from the server if that ever contends with a run
setInterval(lz, 5000);


// One calibration job at a time. Every page polls it; body classes then switch off whatever
// would fight the job for a port: `job` for any job, `jobhw` for one on the bench.
let JOBWAS = false,
  JOBROWS = {},
  JOBFITS = {};
// a row the running job will touch: queued (dimmed), running (spinner in the dot's slot),
// done (checked until the job ends and the new fit is read back)
function paintRows() {
  document.querySelectorAll("tr[data-pad]").forEach((tr) => {
    // a fit this job just reported, shown in its row before the file is written
    const fit = JOBFITS[tr.dataset.pad];
    if (fit) {
      const put = (cls, v, d) => {
        const c = tr.querySelector(cls);
        if (c && v !== null && v !== undefined) {
          c.textContent = Number(v).toFixed(d);
          c.classList.add("fresh", fit.ok ? "v-pass" : "v-fail");
        }
      };
      put(".c-phi", fit.phi0_pi, 3);
      put(".c-span", fit.span, 2);
    }
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
  $("#job").classList.toggle("hide", !run && !d.external);
  const ch =
    d.chain && d.chain.running
      ? ` · step ${d.chain.at + 1}/${d.chain.steps.length}`
      : "";
  const tecNow = $("#tect").textContent;
  $("#jobt").textContent = run
    ? d.waiting
      ? `${d.what}${ch} · waiting for TEC · ${tecNow}`
      : `${d.what}${d.mock ? " (mock)" : ""}${ch} · ${mmss(d.elapsed || 0)}` +
        (d.expected ? ` of ~${mmss(d.expected)}` : "")
    : d.external
      ? `pic process outside this page: pid ${d.external.join(", ")}`
      : "";
  $("#jobstop").classList.toggle("hide", !run);
  if ($("#pstop")) $("#pstop").classList.toggle("hide", !run);
  // the running tile shows its own progress; this bar is for other jobs and for a failure
  const jl = $("#joblog"),
    f = run ? (d.waiting ? null : jobFrac(d)) : 1;
  jl.className =
    "pjob " +
    (run ? "run" : "bad") +
    (!d.cmd || d.rc === 0 || (run && LTILE[d.level]) ? " hide" : "");
  if (d.cmd)
    jl.innerHTML =
      `<span class="pw">${esc(d.what || "")}</span>` +
      `<div class="pbar${f == null ? " ind" : ""}"><i style="width:${(f ?? 1) * 100}%"></i></div>` +
      `<span>${run ? mmss(d.elapsed || 0) : `stopped · exit ${d.rc}`}</span>`;
  PJOB = d;
  paintBusy();
  JOBROWS = run ? d.rows || {} : {};
  JOBFITS = run ? d.fits || {} : {};
  paintRows();
  if (JOBWAS && !run && CUR === "cal") {
    loadCal(); // new fits are on disk
    loadPreds();
  }
  JOBWAS = run;
}
const stopJob = async () => {
  if (
    !confirm("Stop the calibration job? Its partial results are not written.")
  )
    return;
  await fetch("/api/job/stop", { method: "POST", body: "{}" });
  jobPoll();
};
$("#jobstop").onclick = stopJob;
if ($("#pstop")) $("#pstop").onclick = stopJob;
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
    $("#tecnow").textContent = e.textContent;
    $("#tecnow").className =
      "lzs " + (Math.abs(d.c - d.set) > d.tol ? "v-warn" : "v-pass");
    if (document.activeElement !== $("#tecset")) $("#tecset").value = d.set;
    if ($("#techw") && document.activeElement !== $("#techw")) $("#techw").value = d.hw;
    // a bench job's own heater load warms the die: that is expected, not news, so only an
    // idle rig out of band shows the flame
    const hot =
      Math.abs(d.c - d.set) > d.tol &&
      !document.body.classList.contains("jobhw");
    e.className = hot ? "v-warn" : "v-pass";
    // everything that waits on the TEC is disabled now rather than failing later
    document.body.classList.toggle("hot", hot);
    // a greyed button says why on hover, not just by being grey
    const wait = `waiting for the die: ${d.c.toFixed(2)} °C, needs ${d.set} ± ${d.tol}`;
    document
      .querySelectorAll("#preds button, [data-level=all], #run")
      .forEach((b) => {
        if (hot && b.dataset.tip === undefined) b.dataset.tip = b.title || "";
        if (hot) b.title = wait;
        else if (b.dataset.tip !== undefined) {
          b.title = b.dataset.tip;
          delete b.dataset.tip;
        }
      });
    document
      .querySelectorAll(".hotico")
      .forEach(
        (i) =>
          (i.title = `die at ${d.c.toFixed(2)} °C, outside ${d.set} ± ${d.tol}: waits for the TEC`),
      );
    e.title =
      `die ${d.c} °C, setpoint ${d.set} ± ${d.tol}, TEC drive ${d.drive} V (mean of ${d.n})` +
      (d.status ? `\n${d.status}` : "");
  } catch (err) {
    e.textContent = "TEC ?";
    e.className = "v-fail";
  }
}
tecPoll();
setInterval(tecPoll, 5000);

// Raw: every print, the server's own and every job's, newest file first, re-read every 5 s.
async function rawPoll() {
  const sel = $("#logsel");
  const pick = sel.value;
  let d;
  try {
    d = await (
      await fetch(
        "/api/logs" + (pick ? "?name=" + encodeURIComponent(pick) : ""),
      )
    ).json();
  } catch (e) {
    return;
  }
  if (d.error) return ($("#logtext").textContent = why(d));
  const opts = d.files
    .map(
      (f) =>
        `<option value="${esc(f.name)}">${esc(f.name)} · ${esc(f.ago)}</option>`,
    )
    .join("");
  if (sel.innerHTML !== opts) {
    sel.innerHTML = opts;
    sel.value = pick || (d.files[0] && d.files[0].name) || "";
  }
  if (!pick && sel.value) return rawPoll(); // first load: fetch the newest file's text
  const box = $("#logtext"),
    atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.textContent = d.text || "";
  if (atEnd) box.scrollTop = box.scrollHeight; // follow the tail unless you scrolled up
}

// Events: the same prints as one row each. A structured event carries its component `c`, the
// state `s` it entered and its numbers; a raw print is shown as printed. A run of progress
// lines from one source is one row holding the latest value, a traceback is one expandable row.
let LEV = [],
  LT = 0,
  LDIRTY = true;
const LOPEN = new Set();
const LPATH = {
  wait: '<circle cx="8" cy="8" r="6.4"/><path d="M8 4.6V8l2.3 1.6"/>',
  progress:
    '<circle cx="8" cy="8" r="6.4" opacity=".3"/><path d="M8 1.6a6.4 6.4 0 0 1 6.4 6.4"/>',
  info: '<circle cx="8" cy="8" r="1.7" fill="currentColor"/>',
};
const lsvg = (p) =>
  `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" ` +
  `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
const CICO = {
  laser: "laser",
  tec: "tec",
  board: "board",
  switch: "switch",
  session: "key",
  job: "play",
  heater: "heater",
  pd: "pds",
  table: "table",
  fit: "calib",
  sync: "reload",
  server: "diag",
};
const LF = {
  all: () => true,
  error: (e) => e.lvl === "error",
  warn: (e) => e.lvl === "warn",
  laser: (e) => e.c === "laser" || e.kind === "laser",
  jobs: (e) => e.src !== "server",
};
// the lit/dark edge of the laser, from its state, or from the text of an older server's line
const lzOn = (e) =>
  e.c === "laser"
    ? e.s === "on"
      ? "on"
      : e.s === "off"
        ? "off"
        : ""
    : /LASER ON|^hw 1:/.test(e.text)
      ? "on"
      : /LASER OFF|^hw 0:/.test(e.text)
        ? "off"
        : "";
const num = (v, d) =>
  typeof v === "number" && isFinite(v) ? v.toFixed(d) : null;
// An event's numbers, compact, for the events whose line they say in full (TERSE); the line
// itself moves to the tooltip: "H4 · Vπ 5.28 · φ₀ 0.62π · ✓" over a terminal-width fit report.
// Every other structured event: its fields as small key/value chips beside the message. Long
// values (a traceback, an error string) stay in the folded detail, not in a chip.
const LSKIP = new Set(["k", "n", "trace", "job"]);
function lchips(e) {
  const f = e.fields || {};
  return Object.entries(f)
    .filter(([k, v]) => !LSKIP.has(k) && v !== null && String(v).length <= 40)
    .slice(0, 8)
    .map(([k, v]) => {
      const s =
        typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v;
      return `<span class="lchip">${esc(k.replace(/_/g, " "))} <b>${esc(String(s))}</b></span>`;
    })
    .join("");
}
function lfields(e) {
  const o = [],
    put = (s) => s && o.push(s);
  if (e.pad) put(`<b>${esc(e.pad)}</b>`);
  if (e.pd != null && e.c === "heater") put(`PD${e.pd}`);
  put(num(e.vpi, 2) && `Vπ ${num(e.vpi, 2)}`);
  if (e.phi0_pi != null)
    put(
      `φ₀ ${e.was_pi != null ? num(e.was_pi, 2) + "→" : ""}${num(e.phi0_pi, 2)}π`,
    );
  put(num(e.span, 2) && `span ${num(e.span, 2)}π`);
  put(num(e.depth_mv, 1) && `${num(e.depth_mv, 1)} mV`);
  put(num(e.r2_physics ?? e.r2, 3) && `R² ${num(e.r2_physics ?? e.r2, 3)}`);
  put(num(e.points, 0) && `${e.points} pts`);
  put(num(e.dbm, 1) && `${e.dbm >= 0 ? "+" : ""}${num(e.dbm, 1)} dBm`);
  put(num(e.mA, 0) && `${num(e.mA, 0)} mA`);
  put(num(e.temp ?? e.chip_c, 2) && `${num(e.temp ?? e.chip_c, 2)} °C`);
  put(num(e.moved, 4) && `Δ ${num(e.moved, 4)}`);
  put(num(e.secs, 0) && e.c !== "laser" && `${num(e.secs, 0)} s`);
  if (e.rc) put(`exit ${e.rc}`);
  if (e.ok === true) put('<i class="v-pass">✓</i>');
  if (e.ok === false) put('<i class="v-warn">✗</i>');
  return o.join('<i class="lgd">·</i>');
}
// A temperature field named `c` overwrites the component `c` on the wire (pic.log merges fields
// into the event), so a number there is a temperature and the component follows from the state.
const CSTATE = { setpoint: "tec", paused: "session", resumed: "session" };
const lfix = (e) => {
  if (e.c != null && typeof e.c !== "string") {
    e.temp = e.c;
    e.c = CSTATE[e.s] || "tec";
    if (typeof e.kind !== "string") e.kind = e.c;
  }
  return e;
};
const TERSE = new Set([
  "heater:prescan",
  "heater:queued",
  "heater:sweeping",
  "heater:fit",
  "heater:recal",
  "fit:fit",
  "fit:points",
  "table:progress",
  "laser:set",
]);
function lrow(g) {
  const e = g.e,
    raw = e.lvl === "raw",
    lz = lzOn(e),
    cls =
      e.lvl === "error"
        ? "fail"
        : e.lvl === "warn"
          ? "warn"
          : e.lvl === "ok"
            ? "pass"
            : raw
              ? "raw"
              : "info",
    icon = CICO[e.c]
      ? ico(CICO[e.c])
      : e.kind === "laser"
        ? ico("laser")
        : e.lvl === "error"
          ? lsvg(DICO.fail)
          : e.kind === "wait"
            ? lsvg(LPATH.wait)
            : raw
              ? ""
              : lsvg(LPATH[e.kind] || LPATH.info),
    k = e.k ?? (g.n && /(\d+)\s*\/\s*(\d+)/.exec(e.text)?.[1]),
    n = e.n ?? (g.n && /(\d+)\s*\/\s*(\d+)/.exec(e.text)?.[2]),
    // numbers only where they replace the line; beside it they would only repeat it
    fx =
      !raw && TERSE.has(`${e.c}:${e.s}`) ? lfields(e) : !raw ? lchips(e) : "",
    terse = !!fx && TERSE.has(`${e.c}:${e.s}`),
    body =
      `<time>${new Date(e.t * 1000).toLocaleTimeString("en-GB", { hour12: false })}</time>` +
      `<span class="lgi" title="${esc([e.c || e.kind, e.lvl].join(" · "))}">${icon}</span>` +
      `<span class="lgt"${terse ? ` title="${esc(e.text)}"` : ""}>` +
      (e.s
        ? `<span class="lgst">${esc(String(e.s).replace(/_/g, " "))}</span>`
        : "") +
      (terse ? "" : `<span class="lgx">${esc(e.text)}</span>`) +
      (fx ? `<span class="lgn">${fx}</span>` : "") +
      (k != null && n && !(!terse && e.text.includes(`${k}/${n}`))
        ? `<span class="lgk">${k}/${n}</span>`
        : "") +
      (g.n > 1 ? `<em>×${g.n}</em>` : "") +
      (k != null && n && g.n
        ? `<i class="lgp" style="--p:${Math.min(1, k / n)}"></i>`
        : "") +
      `</span>`,
    c = `lgr k-${e.kind} v-${cls}${lz ? " lz-" + lz : ""}`;
  // a dict dump or a long message folds to two lines; the whole of it is one click away
  const det =
    e.detail || (e.fields && e.fields.trace) || (e.text.length > 160 && e.text);
  if (!det) return `<div class="${c}">${body}</div>`;
  const key = e.t + e.src;
  return (
    `<details class="${c}" data-k="${esc(key)}"${LOPEN.has(key) ? " open" : ""}>` +
    `<summary>${body}</summary><pre>${esc(det)}</pre></details>`
  );
}
function lgDraw() {
  const box = $("#lglist"),
    f = () => true, // one list, everything, newest at the bottom
    mine = LEV,
    atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  const groups = [];
  for (const e of mine.filter(f)) {
    const g = groups[groups.length - 1];
    if (e.kind === "progress" && g && g.n && g.e.src === e.src && g.e.c === e.c)
      ((g.e = e), g.n++);
    else groups.push({ e, n: e.kind === "progress" ? 1 : 0 });
  }
  let day = "",
    prev = "",
    html = "";
  for (const g of groups) {
    const d = new Date(g.e.t * 1000).toLocaleDateString("en-GB", {
      weekday: "short",
      day: "numeric",
      month: "short",
    });
    if (d !== day)
      ((html += `<div class="lgday">${d}</div>`), (day = d), (prev = ""));
    // the source once per run of its rows: a job's lines read as one block under its name
    if (g.e.src !== prev)
      html += `<div class="lgfrom" title="${esc(g.e.job || g.e.src)}">${esc(g.e.src)}</div>`;
    html += lrow(g);
    prev = g.e.src;
  }
  box.innerHTML =
    html || `<div class="lgnone">${lsvg(LPATH.info)}nothing here</div>`;
  if (atEnd) box.scrollTop = box.scrollHeight; // follow the tail unless you scrolled up
  LDIRTY = false;
}
async function evPoll() {
  let d;
  try {
    const r = await fetch("/api/events?after=" + LT);
    if (!r.ok) throw r.status; // an older server without events: the raw files still work
    d = await r.json();
  } catch (e) {
    if (!LEV.length) return lgView("raw");
    return;
  }
  if (d.events && d.events.length) {
    LEV = LEV.concat(d.events.map(lfix)).slice(-3000);
    LT = LEV[LEV.length - 1].t;
    LDIRTY = true;
  }
  if (LDIRTY) lgDraw();
}
function lgView(v) {
  document.querySelector(`[name=lgv][value=${v}]`).checked = true;
  const raw = v === "raw";
  $("#lgraw").classList.toggle("hide", !raw);
  $("#lglist").classList.toggle("hide", raw);
  logPoll();
}
const logPoll = () =>
  document.querySelector("[name=lgv]:checked").value === "raw"
    ? rawPoll()
    : evPoll();
if (CUR === "logs") {
  $("#logsel").onchange = rawPoll;
  $("#lgv").onchange = (ev) => lgView(ev.target.value);
  $("#lglist").addEventListener(
    "toggle",
    (ev) => {
      const k = ev.target.dataset.k;
      if (k) ev.target.open ? LOPEN.add(k) : LOPEN.delete(k);
    },
    true,
  );
  logPoll();
  setInterval(logPoll, 5000);
}

// The operating temperature: what runs gate on. The controller's own target is #techw.
$("#tecset").onchange = async () => {
  const v = $("#tecset").value;
  if (v === "") return;
  const d = await (
    await fetch("/api/tec", {
      method: "POST",
      body: JSON.stringify({ set: +v }),
    })
  ).json();
  if (d.error) alert(why(d));
  tecPoll();
};

// PD or SPD: which detectors every run reads through. The server refuses while a job runs.
async function detPoll() {
  try {
    const d = await (await fetch("/api/detectors")).json();
    const i = document.querySelector(`#det input[value="${d.mode}"]`);
    if (i) i.checked = true;
  } catch (e) {}
}
document.querySelectorAll("#det input").forEach(
  (i) =>
    (i.onchange = async () => {
      const d = await (
        await fetch("/api/detectors", { method: "POST", body: JSON.stringify({ mode: i.value }) })
      ).json();
      if (d.error) alert(why(d));
      detPoll();
    }),
);
if ($("#det")) detPoll();

// Full or tiled: the one hosting a Run press measures. Refused while a job runs.
async function hostPoll() {
  try {
    const d = await (await fetch("/api/hosting")).json();
    const i = document.querySelector(`#host input[value="${d.mode}"]`);
    if (i) i.checked = true;
  } catch (e) {}
}
document.querySelectorAll("#host input").forEach(
  (i) =>
    (i.onchange = async () => {
      const d = await (
        await fetch("/api/hosting", { method: "POST", body: JSON.stringify({ mode: i.value }) })
      ).json();
      if (d.error) alert(why(d));
      hostPoll();
    }),
);
if ($("#host")) hostPoll();

$("#techw").onchange = async () => {
  const v = $("#techw").value;
  if (v === "") return;
  const d = await (
    await fetch("/api/tec", { method: "POST", body: JSON.stringify({ hw: +v }) })
  ).json();
  if (d.error) alert(why(d));
  tecPoll();
};
