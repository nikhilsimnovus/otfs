"""Zak-OTFS Waveform Studio — web UI backend.

Thin Flask layer over the validated `otfs` package (the DSP single source
of truth). Serves the dashboard, runs single frames synchronously and BER
sweeps in a background thread, and streams SDR-ready IQ downloads.
Self-update from GitHub follows the oneclick/simdoc layout: the Update
button POSTs /api/update, which runs /usr/local/sbin/otfs-update via
sudo -n (wrapper + narrow sudoers entry planted by scripts/install.sh).

Run:  python webui/app.py   (default http://0.0.0.0:8050, OTFS_PORT to
override). Deployed as the `otfs` systemd service by scripts/install.sh.
"""

import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import Flask, Response, jsonify, request, send_from_directory

from otfs import (ZakOTFSConfig, ZakOTFSModem, OFDMConfig, OFDMModem,
                  etu_channel, eva_channel, ntn_leo_channel, metrics)
from otfs.channel import DDChannel, DDPath, awgn
from otfs.qam import qam_map, qam_constellation

# Studio version. Bump on every push to the otfs repo so operators can
# confirm the Update button actually applied — the new number shows up in
# the topbar after the page reloads.
VERSION = "1.0.1"

HOST = os.environ.get("OTFS_HOST", "0.0.0.0")
PORT = int(os.environ.get("OTFS_PORT", "8050"))

app = Flask(__name__, static_folder="static", static_url_path="")


# The dashboard is a static HTML page; force-fresh so a self-update never
# leaves the browser running old JS against the new backend.
@app.after_request
def _no_cache_html(resp):
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if ctype.startswith("text/html"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

C_LIGHT = 299_792_458.0
PROFILE_MAX_DELAY_S = {"eva": 2.51e-6, "etu": 5.0e-6, "ntn": 1.0e-6,
                       "custom": None}

LAST_IQ = {}          # fmt handled at download time: {'s': ..., 'fs': ..., 'meta': ...}
SWEEPS = {}           # job id -> {progress, done, result, error}


# ------------------------------------------------------------ helpers
def build_channel(p, fs, rng):
    preset = p["channel"]
    if preset == "eva":
        return eva_channel(fs, p["fc_hz"], p["speed_kmh"], rng)
    if preset == "etu":
        return etu_channel(fs, p["fc_hz"], p["speed_kmh"], rng)
    if preset == "ntn":
        return ntn_leo_channel(fs, p["resid_doppler_hz"], rng,
                               k_factor_db=p.get("k_factor_db", 10.0))
    if preset == "custom":
        paths = [DDPath(0, p["cust_doppler_hz"] / fs, 0.85)]
        if p.get("cust_echo", True):
            d = int(round(p.get("cust_echo_delay_us", 4.0) * 1e-6 * fs))
            g = 10 ** (p.get("cust_echo_gain_db", -6.0) / 20)
            paths.append(DDPath(d, -0.6 * p["cust_doppler_hz"] / fs,
                                g * np.exp(1.1j)))
        pw = np.sqrt(sum(abs(q.gain) ** 2 for q in paths))
        for q in paths:
            q.gain /= pw
        return DDChannel(paths=paths, fs=fs)
    raise ValueError(f"unknown channel {preset}")


def doppler_budget_hz(p):
    if p["channel"] in ("eva", "etu"):
        return p["fc_hz"] * (p["speed_kmh"] / 3.6) / C_LIGHT
    if p["channel"] == "ntn":
        return 1.15 * p["resid_doppler_hz"]
    return 1.15 * abs(p["cust_doppler_hz"])


def make_modems(p):
    """Size the DD frame guard (crystallization budget) from the requested
    channel, then build matched OTFS + OFDM modems."""
    M, N, scs = p["M"], p["N"], p["scs_hz"]
    fs = M * scs
    dres = scs / N                     # Doppler resolution = 1/Tf
    fd = doppler_budget_hz(p)
    k_max = int(np.clip(np.ceil(fd / dres) + 1, 2, N // 2 - 1))
    md = PROFILE_MAX_DELAY_S[p["channel"]]
    if md is None:
        md = p.get("cust_echo_delay_us", 4.0) * 1e-6
    l_max = int(np.clip(np.ceil(md * fs) + 1, 2, M // 2 - 1))
    cp = max(8, l_max + 2)
    cfg = ZakOTFSConfig(M=M, N=N, scs_hz=scs, qam_order=p["qam_order"],
                        cp_len=cp, l_max=l_max, k_max=k_max,
                        pilot_boost_db=p.get("pilot_boost_db", 30.0))
    ofdm = OFDMModem(OFDMConfig(M=M, N=N, scs_hz=scs,
                                qam_order=p["qam_order"], cp_len=cp))
    return ZakOTFSModem(cfg), ofdm, {"k_max": k_max, "l_max": l_max, "cp": cp}


def parse_params(d):
    return {
        "M": int(d.get("M", 32)), "N": int(d.get("N", 16)),
        "scs_hz": float(d.get("scs_khz", 15)) * 1e3,
        "qam_order": int(d.get("qam_order", 16)),
        "pilot_boost_db": float(d.get("pilot_boost_db", 30)),
        "snr_db": float(d.get("snr_db", 20)),
        "channel": d.get("channel", "ntn"),
        "fc_hz": float(d.get("fc_ghz", 3.5)) * 1e9,
        "speed_kmh": float(d.get("speed_kmh", 500)),
        "resid_doppler_hz": float(d.get("resid_doppler_hz", 5000)),
        "k_factor_db": float(d.get("k_factor_db", 10)),
        "cust_doppler_hz": float(d.get("cust_doppler_hz", 3000)),
        "cust_echo": bool(d.get("cust_echo", True)),
        "cust_echo_delay_us": float(d.get("cust_echo_delay_us", 4)),
        "cust_echo_gain_db": float(d.get("cust_echo_gain_db", -6)),
        "csi": d.get("csi", "est"),
        "detector": d.get("detector", "lmmse"),
        "seed": int(d.get("seed", 0)) or np.random.randint(1, 2**31),
    }


# ------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/version")
def api_version():
    return jsonify({"version": VERSION})


# Self-update from GitHub (same layout as oneclick/simdoc): the Update
# button hits /api/update which runs /usr/local/sbin/otfs-update via
# sudo -n. That wrapper (planted by install.sh, with a matching narrow
# sudoers entry) downloads the latest main tarball and re-runs install.sh;
# the systemctl restart inside install.sh may cut this response off
# mid-stream — the client treats that as expected and reloads.
@app.post("/api/update")
def api_update():
    updater = "/usr/local/sbin/otfs-update"
    if not Path(updater).exists():
        return jsonify({
            "ok": False,
            "log": (f"[update] {updater} missing — run scripts/install.sh once "
                    f"locally to plant the wrapper + sudoers entry."),
        }), 500
    try:
        rc = subprocess.run(
            ["sudo", "-n", updater],
            capture_output=True, text=True, timeout=600,
        )
        out = (rc.stdout or "")[-4000:]
        if rc.stderr:
            out += "\n--- stderr ---\n" + rc.stderr[-2000:]
        if rc.returncode != 0:
            return jsonify({"ok": False,
                            "log": out + f"\n[update] exited {rc.returncode}"}), 500
        return jsonify({"ok": True, "log": out + "\n[update] done"})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "log": "[update] timed out after 600s"}), 504
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "log": f"[update] FAILED: {exc}"}), 500


@app.post("/api/run_frame")
def run_frame():
    p = parse_params(request.get_json(force=True))
    rng = np.random.default_rng(p["seed"])
    md, md_ofdm, sizing = make_modems(p)
    cfg = md.cfg
    chan = build_channel(p, cfg.fs, rng)
    noise_var = 10 ** (-p["snr_db"] / 10)

    # ---- OTFS
    bits = rng.integers(0, 2, md.bits_per_frame)
    x_dd, s_cp = md.modulate(bits)
    r_cp = awgn(chan.apply_linear(s_cp), noise_var, rng)
    y_dd = md.to_dd(md.strip_cp(r_cp))
    est_paths = []
    if p["csi"] == "perfect":
        H = md.build_h_true(chan)
    else:
        if p["csi"] == "est-taps":
            H, _ = md.estimate_h(y_dd, noise_var)
        else:
            H, est_chan = md.estimate_h_parametric(y_dd, noise_var)
            est_paths = [{"l": q.delay_samp,
                          "fd_hz": round(q.doppler_cps * cfg.fs, 1),
                          "gain": round(float(abs(q.gain)), 3)}
                         for q in est_chan.paths]
    rx_bits, x_hat = md.demodulate(y_dd, H, noise_var, p["detector"])
    ref = qam_map(bits, cfg.qam_order)

    # ---- OFDM baseline (same channel realisation)
    bits_o = rng.integers(0, 2, md_ofdm.bits_per_frame)
    _, s_o = md_ofdm.modulate(bits_o)
    r_o = awgn(chan.apply_linear(s_o), noise_var, rng)
    rx_o, xh_o = md_ofdm.demodulate(r_o, chan, noise_var)
    ref_o = qam_map(bits_o, cfg.qam_order)

    # ---- analysis extras
    s_ovs = metrics.oversample_rrc(s_cp, 4)
    f_psd, psd_db = metrics.psd(s_ovs, 4 * cfg.fs, nfft=2048)
    LAST_IQ.clear()
    LAST_IQ.update({"s": s_cp, "s_ovs": s_ovs, "fs": cfg.fs,
                    "meta": {"waveform": "zak-otfs", "M": cfg.M, "N": cfg.N,
                             "scs_hz": cfg.scs_hz, "cp_len": cfg.cp_len,
                             "qam_order": cfg.qam_order, "seed": p["seed"]}})

    true_paths = [{"l": q.delay_samp,
                   "fd_hz": round(q.doppler_cps * cfg.fs, 1),
                   "gain": round(float(abs(q.gain)), 3)} for q in chan.paths]

    return jsonify({
        "frame": {
            "fs_khz": cfg.fs / 1e3, "frame_ms": round(cfg.frame_duration * 1e3, 3),
            "delay_res_us": round(cfg.delay_res_s * 1e6, 2),
            "doppler_res_hz": round(cfg.doppler_res_hz, 1),
            "n_data": md.layout.n_data, "n_cells": cfg.frame_len,
            "bits": md.bits_per_frame, **sizing, "seed": p["seed"],
        },
        "otfs": {
            "ber": float(metrics.ber(bits, rx_bits)),
            "evm_pct": round(metrics.evm_rms(ref, x_hat), 2),
            "papr_db": round(metrics.papr_db(s_cp), 2),
            "const": [[round(float(v.real), 4), round(float(v.imag), 4)]
                      for v in x_hat],
        },
        "ofdm": {
            "ber": float(metrics.ber(bits_o, rx_o)),
            "evm_pct": round(metrics.evm_rms(ref_o, xh_o), 2),
            "papr_db": round(metrics.papr_db(s_o), 2),
            "const": [[round(float(v.real), 4), round(float(v.imag), 4)]
                      for v in xh_o],
        },
        "ideal_const": [[round(float(v.real), 4), round(float(v.imag), 4)]
                        for v in qam_constellation(cfg.qam_order)],
        "tx_grid": np.abs(x_dd).round(3).tolist(),
        "rx_grid": np.abs(y_dd).round(3).tolist(),
        "paths": {"true": true_paths, "est": est_paths},
        "envelope": np.abs(s_cp).round(4).tolist(),
        "psd": {"f_khz": (f_psd / 1e3).round(2).tolist(),
                "db": psd_db.round(2).tolist()},
    })


SWEEP_DEFS = {
    "speed": {"x": [3, 30, 120, 250, 500, 1000], "xlabel": "UE speed (km/h)"},
    "ntn":   {"x": [200, 500, 1000, 2000, 3500, 5000],
              "xlabel": "residual Doppler (Hz)"},
    "snr":   {"x": [8, 12, 16, 20, 24, 28], "xlabel": "SNR (dB)"},
}


def sweep_worker(job_id, p, kind, n_frames):
    try:
        xs = SWEEP_DEFS[kind]["x"]
        curves = {"ofdm": [], "otfs-est": [], "otfs-perfect": []}
        rng = np.random.default_rng(p["seed"])
        total = len(xs) * n_frames
        done = 0
        for xv in xs:
            q = dict(p)
            if kind == "speed":
                q.update(channel="eva", speed_kmh=xv)
            elif kind == "ntn":
                q.update(channel="ntn", resid_doppler_hz=max(xv, 200))
            else:
                q.update(snr_db=xv, channel="eva", speed_kmh=500)
            md, md_ofdm, _ = make_modems(q)
            noise_var = 10 ** (-q["snr_db"] / 10)
            errs = {k: 0 for k in curves}
            tot = {k: 0 for k in curves}
            for _ in range(n_frames):
                chan = build_channel(q, md.cfg.fs, rng)
                b = rng.integers(0, 2, md_ofdm.bits_per_frame)
                rx, _x = md_ofdm.run_frame(b, chan, noise_var, rng)
                errs["ofdm"] += int(np.sum(rx != b)); tot["ofdm"] += b.size
                for mode, csi in (("otfs-est", "est"),
                                  ("otfs-perfect", "perfect")):
                    b = rng.integers(0, 2, md.bits_per_frame)
                    rx, _x = md.run_frame(b, chan, noise_var, rng, csi=csi)
                    errs[mode] += int(np.sum(rx != b)); tot[mode] += b.size
                done += 1
                SWEEPS[job_id]["progress"] = done / total
            for k in curves:
                curves[k].append(errs[k] / max(tot[k], 1))
        SWEEPS[job_id].update(done=True, progress=1.0, result={
            "x": xs, "xlabel": SWEEP_DEFS[kind]["xlabel"], "curves": curves,
            "kind": kind})
    except Exception as e:                      # surface to the UI
        SWEEPS[job_id].update(done=True, error=str(e))


@app.post("/api/sweep")
def sweep_start():
    d = request.get_json(force=True)
    p = parse_params(d)
    kind = d.get("kind", "speed")
    n_frames = int(np.clip(int(d.get("frames", 8)), 2, 40))
    job_id = uuid.uuid4().hex[:12]
    SWEEPS[job_id] = {"progress": 0.0, "done": False, "result": None,
                      "error": None, "t0": time.time()}
    threading.Thread(target=sweep_worker, args=(job_id, p, kind, n_frames),
                     daemon=True).start()
    return jsonify({"job": job_id})


@app.get("/api/sweep/<job_id>")
def sweep_status(job_id):
    j = SWEEPS.get(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({k: j[k] for k in ("progress", "done", "result", "error")})


@app.get("/api/download/<fmt>")
def download(fmt):
    if not LAST_IQ:
        return jsonify({"error": "run a frame first"}), 400
    if fmt == "cf32":
        s, fs = LAST_IQ["s"], LAST_IQ["fs"]
    elif fmt == "cf32x4":
        s, fs = LAST_IQ["s_ovs"], 4 * LAST_IQ["fs"]
    elif fmt == "sc16":
        s, fs = LAST_IQ["s"], LAST_IQ["fs"]
    else:
        return jsonify({"error": "fmt must be cf32|cf32x4|sc16"}), 400
    if fmt == "sc16":
        peak = float(np.abs(np.concatenate([s.real, s.imag])).max())
        scale = peak / (32767 * 10 ** (-3 / 20))
        iq = np.empty(2 * len(s), dtype=np.int16)
        iq[0::2] = np.round(s.real / scale)
        iq[1::2] = np.round(s.imag / scale)
    else:
        iq = np.empty(2 * len(s), dtype=np.float32)
        iq[0::2] = s.real
        iq[1::2] = s.imag
    name = f"zak_otfs_{int(fs/1e3)}kHz.{fmt.replace('x4','')}"
    return Response(iq.tobytes(), mimetype="application/octet-stream",
                    headers={"Content-Disposition":
                             f"attachment; filename={name}"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
