"""
Post-processing for the Lumerical varFDTD ring-resonator runs.

1. Loads the through/drop spectra exported by lumerical/ring_varfdtd.py
2. Fits one resonance with the exact add-drop ring model -> self-coupling r1, r2,
   round-trip amplitude a, optical path length, resonance wavelength
3. Measures the resonance shift between the baseline and index-shifted runs
4. Converts the fit to temporal coupled-mode-theory (CMT) rates and computes the
   time-domain response of the ring when that shift is switched on and off
5. Writes figures to ../images and numbers to ../data/summary.json

    python fit_and_transient.py            # use data/spectra_*.npz
    python fit_and_transient.py --selftest # validate the fitter on synthetic data
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks

C = 299792458.0
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, RED = "#2a6fbb", "#e07a1f", "#c0392b"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


# ---------------------------------------------------------------- ring model
def addrop(lam, r1, r2, a, opl, lam0, s=1.0):
    """Exact steady-state add-drop ring (power transmission, through and drop)."""
    phi = 2 * np.pi * opl * (1 / lam - 1 / lam0)
    e = a * np.exp(-1j * phi)
    den = 1 - r1 * r2 * e
    thru = np.abs((r1 - r2 * e) / den) ** 2
    drop = (1 - r1**2) * (1 - r2**2) * a / np.abs(den) ** 2
    return s * thru, s * drop


def pick_resonance(lam, drop, target):
    smooth = np.convolve(drop, np.ones(7) / 7, mode="same")
    dlam = float(np.median(np.diff(lam)))
    peaks, _ = find_peaks(smooth, prominence=0.3 * smooth.max(), distance=max(1, int(2e-9 / dlam)))
    if peaks.size == 0:
        raise RuntimeError("No resonance found in the drop-port spectrum.")
    k = peaks[np.argmin(np.abs(lam[peaks] - target))]
    if peaks.size >= 2:
        fsr = float(np.median(np.diff(lam[peaks])))
    else:
        fsr = 20e-9
    return lam[k], fsr


def fit_resonance(lam, thru, drop, target=1.55e-6):
    lam0, fsr = pick_resonance(lam, drop, target)
    win = np.abs(lam - lam0) < 0.45 * fsr
    lw, tw, dw = lam[win], thru[win], drop[win]
    opl0 = lam0**2 / fsr

    def unpack(p):
        r1, r2, a, opl_um, dl_nm, s = p
        return r1, r2, a, opl_um * 1e-6, lam0 + dl_nm * 1e-9, s

    def resid(p):
        mt, md = addrop(lw, *unpack(p))
        return np.concatenate([mt - tw, md - dw])

    lo = [0.5, 0.5, 0.80, 0.5 * opl0 * 1e6, -5.0, 0.5]
    hi = [0.9999, 0.9999, 1.0, 2.0 * opl0 * 1e6, 5.0, 1.05]
    best = None
    for r1, r2 in [(0.98, 0.98), (0.95, 0.95), (0.9, 0.97), (0.97, 0.9), (0.99, 0.99)]:
        p0 = [r1, r2, 0.99, opl0 * 1e6, 0.0, max(thru.max(), 0.5)]
        sol = least_squares(resid, p0, bounds=(lo, hi), x_scale=[0.01, 0.01, 0.01, 1.0, 0.05, 0.05])
        if best is None or sol.cost < best.cost:
            best = sol
    r1, r2, a, opl, lam_res, s = unpack(best.x)
    return dict(r1=r1, r2=r2, a=a, opl=opl, lam_res=lam_res, s=s,
                fsr=fsr, lam0_guess=lam0, window=win, rms=float(np.sqrt(2 * best.cost / best.fun.size)))


def fit_shift(lam, thru, drop, base):
    """Fit only the resonance wavelength (other parameters fixed) on a shifted run."""
    win = np.abs(lam - base["lam_res"]) < 0.45 * base["fsr"]
    lw, tw, dw = lam[win], thru[win], drop[win]

    def resid(p):
        mt, md = addrop(lw, base["r1"], base["r2"], base["a"], base["opl"],
                        base["lam_res"] + p[0] * 1e-9, base["s"])
        return np.concatenate([mt - tw, md - dw])

    best = None
    for start in (-1.0, -0.2, 0.0, 0.2, 1.0):
        sol = least_squares(resid, [start], x_scale=[0.1])
        if best is None or sol.cost < best.cost:
            best = sol
    return float(best.x[0] * 1e-9)       # shift in metres


# ------------------------------------------------------------------ CMT time domain
def cmt_params(fit):
    t_rt = fit["opl"] / C
    g1 = (1 - fit["r1"] ** 2) / t_rt
    g2 = (1 - fit["r2"] ** 2) / t_rt
    g0 = (1 - fit["a"] ** 2) / t_rt
    g = g0 + g1 + g2
    omega0 = 2 * np.pi * C / fit["lam_res"]
    return dict(t_rt=t_rt, g0=g0, g1=g1, g2=g2, g=g, Q=omega0 / g, omega0=omega0)


def transient(t, delta_fn, p, s=1.0):
    dt = t[1] - t[0]
    a_t = np.zeros_like(t, dtype=complex)
    for k in range(1, len(t)):
        D = delta_fn(t[k - 1])
        M = -1j * D - p["g"] / 2
        E = np.exp(M * dt)
        a_t[k] = a_t[k - 1] * E + (np.sqrt(p["g1"]) / M) * (E - 1)
    thru = s * np.abs(1 - np.sqrt(p["g1"]) * a_t) ** 2
    drop = s * np.abs(np.sqrt(p["g2"]) * a_t) ** 2
    return thru, drop


# ----------------------------------------------------------------------- driver
def synthetic(dn_shift_nm=0.15):
    """Fake 'Lumerical' spectra from the model itself (self-test only)."""
    rng = np.random.default_rng(1)
    lam = np.linspace(1.52e-6, 1.58e-6, 3000)
    truth = dict(r1=0.965, r2=0.972, a=0.985, opl=100e-6, lam_res=1.5503e-6, s=0.97)
    out = {}
    for tag, shift in (("baseline", 0.0), ("shifted", dn_shift_nm * 1e-9)):
        t, d = addrop(lam, truth["r1"], truth["r2"], truth["a"], truth["opl"],
                      truth["lam_res"] + shift, truth["s"])
        out[tag] = dict(lam=lam, thru=t + rng.normal(0, 2e-3, lam.size), drop=d + rng.normal(0, 2e-3, lam.size))
    return out, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "images"))
    ap.add_argument("--target", type=float, default=1.55e-6, help="resonance wavelength to fit (m)")
    ap.add_argument("--shift-linewidths", type=float, default=None,
                    help="override the simulated step size (in linewidths) instead of using the measured shift")
    args = ap.parse_args()

    if args.selftest:
        runs, truth = synthetic()
        args.out = os.path.join(HERE, "_selftest_images")
        args.data = os.path.join(HERE, "_selftest_data")
        dn = 5e-4
    else:
        runs = {k: dict(np.load(os.path.join(args.data, f"spectra_{k}.npz"))) for k in ("baseline", "shifted")}
        dn = float(np.loadtxt(os.path.join(args.data, "dn.txt")))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.data, exist_ok=True)

    b = runs["baseline"]
    fit = fit_resonance(b["lam"], b["thru"], b["drop"], args.target)
    shift = fit_shift(runs["shifted"]["lam"], runs["shifted"]["thru"], runs["shifted"]["drop"], fit)
    p = cmt_params(fit)
    fwhm = fit["lam_res"] ** 2 * p["g"] / (2 * np.pi * C)
    sens = shift / dn                           # nm of shift per unit index step (m per RIU)
    shift_lw = abs(shift) / fwhm                # shift in linewidths

    # ---- Fig 1: fitted spectra ----
    lam_nm = b["lam"] * 1e9
    win = fit["window"]
    mt, md = addrop(b["lam"], fit["r1"], fit["r2"], fit["a"], fit["opl"], fit["lam_res"], fit["s"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(lam_nm, 10 * np.log10(np.clip(b["thru"], 0, None) + 1e-6), color=BLUE, lw=1, label="Through (varFDTD)")
    ax.plot(lam_nm, 10 * np.log10(np.clip(b["drop"], 0, None) + 1e-6), color=ORANGE, lw=1, label="Drop (varFDTD)")
    ax.plot(lam_nm[win], 10 * np.log10(mt[win] + 1e-6), "k--", lw=1.2, label="Add-drop model fit")
    ax.plot(lam_nm[win], 10 * np.log10(md[win] + 1e-6), "k--", lw=1.2)
    ax.set_ylim(-40, 3); ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Transmission (dB)")
    ax.set_title(f"Ring spectrum from varFDTD:  Q$_{{loaded}}$ ≈ {p['Q']:,.0f},  FSR ≈ {fit['fsr']*1e9:.1f} nm")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(args.out, "01_varfdtd_spectrum_fit.png"), dpi=200); plt.close(fig)

    # ---- Fig 2: resonance shift ----
    s2 = runs["shifted"]
    zoom = np.abs(b["lam"] - fit["lam_res"]) < 4 * fwhm
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(b["lam"][zoom] * 1e9, b["drop"][zoom], color=BLUE, label="baseline")
    ax.plot(s2["lam"][zoom] * 1e9, s2["drop"][zoom], color=RED, label=f"ring index +{dn:g}")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Drop-port transmission")
    ax.set_title(f"Resonance shift: {shift*1e9:+.3f} nm  ({shift_lw:.2f} linewidths)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "02_resonance_shift.png"), dpi=200); plt.close(fig)

    # ---- time domain ----
    step_lw = args.shift_linewidths if args.shift_linewidths is not None else shift_lw
    d_omega = -np.sign(shift if shift != 0 else 1.0) * step_lw * p["g"]   # resonance moves by -2*pi*c*dlam/lam^2
    tau = 2 / p["g"]
    t_on, t_off = 12 * tau, 24 * tau
    t = np.arange(0, 36 * tau, tau / 400)
    prof = lambda tt: d_omega if (t_on <= tt < t_off) else 0.0
    thru_t, drop_t = transient(t, prof, p, fit["s"])
    thru_0, drop_0 = transient(t, lambda _t: 0.0, p, fit["s"])

    fig, axs = plt.subplots(3, 1, figsize=(8.5, 8), sharex=True, gridspec_kw={"height_ratios": [0.6, 1.4, 1.4]})
    axs[0].plot(t * 1e12, [prof(x) / p["g"] for x in t], "k"); axs[0].set_ylabel("Detuning\n(units of γ)")
    axs[0].set_title("Time-domain response to the simulated index step (CMT from varFDTD fit)")
    axs[1].plot(t * 1e12, thru_t, color=BLUE, label="Through"); axs[1].plot(t * 1e12, thru_0, "--", color="gray", lw=1, label="no step")
    axs[2].plot(t * 1e12, drop_t, color=ORANGE, label="Drop"); axs[2].plot(t * 1e12, drop_0, "--", color="gray", lw=1, label="no step")
    for ax, lab in zip(axs[1:], ("Through power", "Drop power")):
        ax.set_ylabel(lab); ax.legend(loc="upper right")
    axs[2].set_xlabel("Time (ps)")
    for ax in axs: ax.axvspan(t_on * 1e12, t_off * 1e12, color="gold", alpha=0.15)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "03_transient_response.png"), dpi=200); plt.close(fig)

    # ---- overshoot vs step size ----
    sizes = np.linspace(0.25, 4.0, 16)
    pk, st = [], []
    t_short = np.arange(0, 20 * tau, tau / 300)
    for sz in sizes:
        th, _ = transient(t_short, lambda tt, sz=sz: sz * p["g"] if tt >= 4 * tau else 0.0, p, fit["s"])
        w = t_short >= 4 * tau
        pk.append(th[w].max()); st.append(th[w][-100:].mean())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sizes, pk, "o-", color=RED, label="Peak through power")
    ax.plot(sizes, st, "s-", color=BLUE, label="Settled through power")
    ax.axvline(step_lw, color="gray", ls=":", label="simulated step")
    ax.set_xlabel("Resonance shift (linewidths)"); ax.set_ylabel("Through power"); ax.legend()
    ax.set_title("Overshoot vs step size (fitted ring)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "04_overshoot_vs_shift.png"), dpi=200); plt.close(fig)

    summary = dict(
        lam_res_nm=fit["lam_res"] * 1e9, fsr_nm=fit["fsr"] * 1e9, Q_loaded=p["Q"], fwhm_pm=fwhm * 1e12,
        r1=fit["r1"], r2=fit["r2"], round_trip_amplitude=fit["a"], optical_path_um=fit["opl"] * 1e6,
        fit_rms=fit["rms"], photon_lifetime_ps=tau * 1e12,
        index_step=dn, resonance_shift_pm=shift * 1e12, shift_in_linewidths=shift_lw,
        sensitivity_nm_per_RIU=sens * 1e9,
        peak_through_overshoot=float(thru_t[(t >= t_on) & (t < t_off)].max()),
    )
    with open(os.path.join(args.data, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if args.selftest:
        print("\nTRUTH:", {k: truth[k] for k in ("r1", "r2", "a", "opl", "lam_res", "s")})
        print("expected shift: 150 pm")


if __name__ == "__main__":
    main()
