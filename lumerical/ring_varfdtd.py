"""
Add-drop silicon ring resonator in Ansys Lumerical MODE (2.5D varFDTD), driven by lumapi.

Runs two simulations of the same ring:
  1. baseline ring index
  2. ring index shifted by +DN (the static "index step" a modulator would apply)

and exports the through/drop spectra to ../data/ for analysis/fit_and_transient.py.

    python ring_varfdtd.py [--dn 5e-4] [--hide] [--sim-time 20000]

Geometry (SOI, TE): 220 nm Si core in SiO2, 500 nm wide ring and buses.
The core is modelled as a non-dispersive object-defined dielectric so that the
ring index can be perturbed in a controlled way (see README, Limitations).
"""
import argparse
import os
import numpy as np
from lum_utils import import_lumapi, set_props, try_values

lumapi = import_lumapi()
C = 299792458.0
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# ---- geometry (m) ----------------------------------------------------------
R = 4.0e-6           # ring centre-line radius
W = 0.5e-6           # waveguide width
GAP = 0.2e-6         # ring-bus gap
H = 0.22e-6          # silicon thickness
N_SI = 3.476
N_SIO2 = 1.444
BUS_Y = R + W + GAP  # bus-waveguide centre offset from the ring centre
X_HALF = R + 2.0e-6
Y_HALF = BUS_Y + 1.3e-6
LAM_START, LAM_STOP = 1.52e-6, 1.58e-6
N_FREQ = 3000


def build(sim, sim_time_fs):
    sim.newproject()

    # --- varFDTD region ---
    sim.addvarfdtd()
    set_props(sim, {
        "name": "varFDTD",
        "x": 0, "x span": 2 * X_HALF,
        "y": 0, "y span": 2 * Y_HALF,
        "z": H / 2, "z span": 1.0e-6,
        "background index": N_SIO2,
        "simulation time": sim_time_fs * 1e-15,
        "bandwidth": "broadband",
    }, "varFDTD region")
    try:
        sim.setactivesolver("varFDTD")
    except Exception:
        pass

    # --- structures ---
    sim.addring()
    set_props(sim, {
        "name": "ring", "x": 0, "y": 0, "z": H / 2, "z span": H,
        "inner radius": R - W / 2, "outer radius": R + W / 2,
        "theta start": 0, "theta stop": 360,
        "material": "<Object defined dielectric>", "index": N_SI,
    }, "ring")
    for name, y in (("bus_through", -BUS_Y), ("bus_drop", +BUS_Y)):
        sim.addrect()
        set_props(sim, {
            "name": name, "x": 0, "x span": 2 * X_HALF + 2e-6,
            "y": y, "y span": W, "z": H / 2, "z span": H,
            "material": "<Object defined dielectric>", "index": N_SI,
        }, name)

    # --- source: fundamental mode of the input bus ---
    sim.addmodesource()
    set_props(sim, {
        "name": "source", "injection axis": "x",
        "x": -(X_HALF - 0.6e-6), "y": -BUS_Y, "y span": 2.5e-6,
        "wavelength start": LAM_START, "wavelength stop": LAM_STOP,
    }, "source")
    try_values(sim, "mode selection", ["fundamental TE mode", "fundamental mode"], "source")

    # --- monitors ---
    # Through port: same bus, right-hand side. Drop port: upper bus, left-hand side
    # (the drop signal counter-propagates on the second bus).
    for name, x, y in (("thru", +(X_HALF - 0.6e-6), -BUS_Y),
                       ("drop", -(X_HALF - 0.6e-6), +BUS_Y)):
        sim.addpower()
        set_props(sim, {
            "name": name, "monitor type": "Linear Y",
            "x": x, "y": y, "y span": 2.5e-6,
            "override global monitor settings": 1, "frequency points": N_FREQ,
        }, name)


def run_and_export(sim, tag):
    sim.run()
    out = {}
    for name in ("thru", "drop"):
        f = np.array(sim.getdata(name, "f")).flatten()
        t = np.abs(np.array(sim.transmission(name))).flatten()
        out[name] = (C / f, t)
    lam, thru = out["thru"]
    _, drop = out["drop"]
    order = np.argsort(lam)
    path = os.path.join(DATA, f"spectra_{tag}.npz")
    np.savez(path, lam=lam[order], thru=thru[order], drop=drop[order])
    print(f"saved {path}  ({lam.size} points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dn", type=float, default=5e-4, help="ring core index step for run 2")
    ap.add_argument("--sim-time", type=float, default=20000.0, help="simulation time in fs")
    ap.add_argument("--hide", action="store_true", help="run without the Lumerical GUI")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    with lumapi.MODE(hide=args.hide) as sim:
        build(sim, args.sim_time)
        sim.save(os.path.join(DATA, "ring_varfdtd.lms"))

        print("Running baseline ...")
        run_and_export(sim, "baseline")

        print(f"Running with ring index +{args.dn:g} ...")
        sim.switchtolayout()
        sim.setnamed("ring", "index", N_SI + args.dn)
        run_and_export(sim, "shifted")
        np.savetxt(os.path.join(DATA, "dn.txt"), [args.dn])


if __name__ == "__main__":
    main()
