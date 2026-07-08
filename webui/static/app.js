/* Zak-OTFS Waveform Studio — frontend logic */
"use strict";

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------ plotly theming */
const FONT = { family: "Inter, sans-serif", color: "#8b98c0", size: 11 };
const LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: FONT,
  margin: { l: 48, r: 12, t: 8, b: 38 },
  height: 260,
};
const CFG = { displayModeBar: false, responsive: true };
const grid = { gridcolor: "#1e2a4a", zerolinecolor: "#1e2a4a" };

/* ------------------------------------------------ control wiring */
const sliderOuts = [
  ["pilot_boost_db", "pilot_boost_out", (v) => `${v} dB`],
  ["speed_kmh", "speed_out", (v) => `${v} km/h`],
  ["fc_ghz", "fc_out", (v) => `${(+v).toFixed(1)} GHz`],
  ["resid_doppler_hz", "resid_out", (v) => `${(v / 1000).toFixed(1)} kHz`],
  ["k_factor_db", "k_out", (v) => `${v} dB`],
  ["cust_doppler_hz", "cdop_out", (v) => `${(v / 1000).toFixed(1)} kHz`],
  ["cust_echo_delay_us", "cdel_out", (v) => `${v} µs`],
  ["cust_echo_gain_db", "cgain_out", (v) => `${v} dB`],
  ["snr_db", "snr_out", (v) => `${v} dB`],
];
for (const [inp, out, fmt] of sliderOuts) {
  $(inp).addEventListener("input", () => ($(out).textContent = fmt($(inp).value)));
}

$("channel").addEventListener("change", () => {
  const c = $("channel").value;
  $("ch-terrestrial").classList.toggle("hidden", !(c === "eva" || c === "etu"));
  $("ch-ntn").classList.toggle("hidden", c !== "ntn");
  $("ch-custom").classList.toggle("hidden", c !== "custom");
});

document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("tab-frame").classList.toggle("hidden", t.dataset.tab !== "frame");
    $("tab-sweep").classList.toggle("hidden", t.dataset.tab !== "sweep");
  })
);

function params() {
  return {
    M: +$("M").value, N: +$("N").value, scs_khz: +$("scs_khz").value,
    qam_order: +$("qam_order").value, pilot_boost_db: +$("pilot_boost_db").value,
    channel: $("channel").value, speed_kmh: +$("speed_kmh").value,
    fc_ghz: +$("fc_ghz").value, resid_doppler_hz: +$("resid_doppler_hz").value,
    k_factor_db: +$("k_factor_db").value,
    cust_doppler_hz: +$("cust_doppler_hz").value,
    cust_echo_delay_us: +$("cust_echo_delay_us").value,
    cust_echo_gain_db: +$("cust_echo_gain_db").value,
    csi: $("csi").value, detector: $("detector").value,
    snr_db: +$("snr_db").value, seed: +$("seed").value,
  };
}

/* --------------------------------------------------- run a frame */
async function runFrame() {
  const btn = $("run");
  btn.disabled = true;
  $("run-status").textContent = "modulating · channel · equalising…";
  const t0 = performance.now();
  try {
    const r = await fetch("/api/run_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params()),
    });
    if (!r.ok) throw new Error((await r.json()).error || r.statusText);
    render(await r.json());
    $("run-status").textContent =
      `done in ${((performance.now() - t0) / 1000).toFixed(2)} s`;
  } catch (e) {
    $("run-status").textContent = "✗ " + e.message;
  } finally {
    btn.disabled = false;
  }
}
$("run").addEventListener("click", runFrame);

/* --------------------------------------------------- rendering */
function fmtBer(b) {
  if (b === 0) return "0";
  return b.toExponential(1).replace("e-", "e-");
}

function render(d) {
  const f = d.frame;
  const berCls = (b) => (b < 1e-3 ? "good" : b > 3e-2 ? "bad" : "");
  $("metrics").innerHTML = `
    <div class="metric ${berCls(d.otfs.ber)}"><div class="k">OTFS BER</div>
      <div class="v">${fmtBer(d.otfs.ber)}</div><div class="s">uncoded · ${f.bits} bits</div></div>
    <div class="metric ${berCls(d.ofdm.ber)}"><div class="k">OFDM BER</div>
      <div class="v">${fmtBer(d.ofdm.ber)}</div><div class="s">ideal one-tap CSI</div></div>
    <div class="metric accent"><div class="k">OTFS EVM</div>
      <div class="v">${d.otfs.evm_pct}%</div><div class="s">OFDM ${d.ofdm.evm_pct}%</div></div>
    <div class="metric"><div class="k">PAPR</div>
      <div class="v">${d.otfs.papr_db}</div><div class="s">dB · OFDM ${d.ofdm.papr_db} dB</div></div>
    <div class="metric"><div class="k">Frame</div>
      <div class="v">${f.frame_ms}<span style="font-size:13px"> ms</span></div>
      <div class="s">fs ${f.fs_khz} kHz · CP ${f.cp}</div></div>
    <div class="metric"><div class="k">DD resolution</div>
      <div class="v">${f.doppler_res_hz}<span style="font-size:13px"> Hz</span></div>
      <div class="s">${f.delay_res_us} µs · guard k±${f.k_max} l±${f.l_max}</div></div>`;

  const heat = (el, z) =>
    Plotly.react(el, [{ z, type: "heatmap", colorscale: "Viridis", showscale: false }],
      { ...LAYOUT, xaxis: { title: "Doppler bin k", ...grid },
        yaxis: { title: "delay bin l", ...grid } }, CFG);
  heat("plot-tx", d.tx_grid);
  heat("plot-rx", d.rx_grid);

  const constPlot = (el, pts, color) =>
    Plotly.react(el, [
      { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), mode: "markers",
        type: "scatter", marker: { size: 4, color, opacity: 0.65 }, name: "rx" },
      { x: d.ideal_const.map((p) => p[0]), y: d.ideal_const.map((p) => p[1]),
        mode: "markers", type: "scatter",
        marker: { symbol: "cross-thin", size: 10, color: "#fb7185",
                  line: { width: 1.6, color: "#fb7185" } }, name: "ideal" },
    ], { ...LAYOUT, showlegend: false,
         xaxis: { title: "I", range: [-1.8, 1.8], ...grid },
         yaxis: { title: "Q", range: [-1.8, 1.8], scaleanchor: "x", ...grid } }, CFG);
  constPlot("plot-const-otfs", d.otfs.const, "#22d3ee");
  constPlot("plot-const-ofdm", d.ofdm.const, "#fbbf24");

  const row = (p) =>
    `<tr><td>path</td><td>${p.l}</td><td>${p.fd_hz}</td><td>${p.gain}</td></tr>`;
  $("paths").innerHTML = `
    <table class="paths">
      <tr><th></th><th>delay bin</th><th>Doppler (Hz)</th><th>|gain|</th></tr>
      <tr class="section"><td colspan="4">true channel</td></tr>
      ${d.paths.true.map(row).join("")}
      <tr class="section"><td colspan="4">pilot estimate ${d.paths.est.length ? "" : "(n/a for this CSI mode)"}</td></tr>
      ${d.paths.est.map(row).join("")}
    </table>`;

  Plotly.react("plot-psd", [
    { x: d.psd.f_khz, y: d.psd.db, mode: "lines",
      line: { color: "#8b5cf6", width: 1.4 }, fill: "tozeroy",
      fillcolor: "rgba(139,92,246,.09)" }],
    { ...LAYOUT, xaxis: { title: "frequency (kHz)", ...grid },
      yaxis: { title: "PSD (dB rel. peak)", range: [-95, 4], ...grid } }, CFG);
}

/* --------------------------------------------------- BER sweep */
let sweepTimer = null;
async function runSweep() {
  const btn = $("run-sweep");
  btn.disabled = true;
  $("sweep-status").textContent = "starting…";
  $("sweep-progress").style.width = "0%";
  try {
    const r = await fetch("/api/sweep", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...params(), kind: $("sweep_kind").value,
                             frames: +$("sweep_frames").value }),
    });
    const { job } = await r.json();
    sweepTimer = setInterval(async () => {
      const s = await (await fetch(`/api/sweep/${job}`)).json();
      if (s.error) {
        clearInterval(sweepTimer); btn.disabled = false;
        $("sweep-status").textContent = "✗ " + s.error;
        return;
      }
      $("sweep-progress").style.width = `${(s.progress * 100).toFixed(0)}%`;
      $("sweep-status").textContent = s.done ? "done" :
        `${(s.progress * 100).toFixed(0)} %`;
      if (s.done) {
        clearInterval(sweepTimer); btn.disabled = false;
        plotSweep(s.result);
      }
    }, 700);
  } catch (e) {
    $("sweep-status").textContent = "✗ " + e.message;
    btn.disabled = false;
  }
}
$("run-sweep").addEventListener("click", runSweep);

function plotSweep(res) {
  const floor = 1e-6;
  const traces = [
    ["ofdm", "CP-OFDM (ideal one-tap CSI)", "#fb7185", "solid", "circle"],
    ["otfs-est", "Zak-OTFS (pilot-estimated CSI)", "#22d3ee", "dash", "triangle-up"],
    ["otfs-perfect", "Zak-OTFS (perfect CSI)", "#34d399", "solid", "square"],
  ].map(([k, name, color, dash, sym]) => ({
    x: res.x, y: res.curves[k].map((v) => Math.max(v, floor)),
    name, mode: "lines+markers", line: { color, dash, width: 2 },
    marker: { symbol: sym, size: 8, color },
  }));
  Plotly.react("plot-sweep", traces, {
    ...LAYOUT, height: 480,
    xaxis: { title: res.xlabel, type: res.kind === "ntn" ? "log" : "linear", ...grid },
    yaxis: { title: "uncoded BER", type: "log",
             tickformat: ".0e", ...grid },
    legend: { x: 0.02, y: 0.04, bgcolor: "rgba(14,20,38,.8)",
              bordercolor: "#1e2a4a", borderwidth: 1 },
    margin: { l: 60, r: 16, t: 12, b: 46 },
  }, CFG);
}

/* ------------------------------------ self-update (oneclick-style) */
async function runUpdate() {
  const btn = $("update-btn"), lbl = $("update-label");
  btn.disabled = true;
  btn.classList.add("spin");
  lbl.textContent = "Updating…";
  try {
    const r = await fetch("/api/update", { method: "POST", cache: "no-store" });
    const j = await r.json();
    if (!j.ok) {
      lbl.textContent = "Update failed";
      console.error(j.log);
      setTimeout(() => { lbl.textContent = "Update"; btn.disabled = false;
                         btn.classList.remove("spin"); }, 4000);
      return;
    }
    lbl.textContent = "Restarting…";
    setTimeout(() => location.reload(), 4000);
  } catch (e) {
    // service restarted under us mid-response — expected; reload picks up
    // the new build
    lbl.textContent = "Restarting…";
    setTimeout(() => location.reload(), 4000);
  }
}
$("update-btn").addEventListener("click", runUpdate);

fetch("/api/version").then((r) => r.json())
  .then((j) => ($("version-chip").textContent = "v" + j.version))
  .catch(() => {});

/* boot: run once with defaults */
runFrame();
