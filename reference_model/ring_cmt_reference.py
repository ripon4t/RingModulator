"""
Transient response of an add-drop silicon ring modulator
using temporal coupled-mode theory (CMT).

Generates all figures in ../images/.
Run:  python ring_transient.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Device parameters (typical SOI add-drop ring, illustrative values)
# ----------------------------------------------------------------------------
c = 299792458.0
R = 10e-6                  # ring radius [m]
L = 2 * np.pi * R          # round-trip length [m]
n_g = 4.2                  # group index
lam_res = 1550.7e-9        # resonance wavelength [m]
r1, r2, a = None, 0.98, 0.995   # self-coupling (input), self-coupling (drop), round-trip amplitude loss
r1 = r2 * a                # choose r1 = r2*a -> critical coupling (through-port null)

T_rt = n_g * L / c                         # round-trip time [s]
g1 = (1 - r1**2) / T_rt                    # input coupling rate  [1/s]
g2 = (1 - r2**2) / T_rt                    # drop coupling rate   [1/s]
g0 = (1 - a**2) / T_rt                     # intrinsic loss rate  [1/s]
g = g0 + g1 + g2                           # total energy decay rate
omega0 = 2 * np.pi * c / lam_res
Q_loaded = omega0 / g
FSR = lam_res**2 / (n_g * L)


# ----------------------------------------------------------------------------
# 1. Steady-state spectrum (exact add-drop ring transfer functions)
# ----------------------------------------------------------------------------
def spectrum(lam):
    phi = 2 * np.pi * n_g * L * (1 / lam - 1 / lam_res)
    e = a * np.exp(-1j * phi)
    den = 1 - r1 * r2 * e
    t_thru = (r1 - r2 * e) / den
    t_drop = -np.sqrt((1 - r1**2) * (1 - r2**2)) * np.sqrt(a) * np.exp(-1j * phi / 2) / den
    return np.abs(t_thru) ** 2, np.abs(t_drop) ** 2


# ----------------------------------------------------------------------------
# 2. Transient response (CMT, exact piecewise-constant integration)
#    da/dt = (-j*Delta(t) - g/2) a + sqrt(g1) s_in
#    s_thru = s_in - sqrt(g1) a ;  s_drop = sqrt(g2) a
#    Delta(t) = (resonance frequency) - (laser frequency)
# ----------------------------------------------------------------------------
def transient(t, delta_fn):
    dt = t[1] - t[0]
    a_t = np.zeros_like(t, dtype=complex)
    s_in = 1.0
    for k in range(1, len(t)):
        D = delta_fn(t[k - 1])
        M = -1j * D - g / 2
        E = np.exp(M * dt)
        a_t[k] = a_t[k - 1] * E + (np.sqrt(g1) * s_in / M) * (E - 1)
    thru = np.abs(s_in - np.sqrt(g1) * a_t) ** 2
    drop = np.abs(np.sqrt(g2) * a_t) ** 2
    return thru, drop, np.abs(a_t) ** 2


def step_delta(shift, t_on, t_off):
    return lambda tt: shift if (t_on <= tt < t_off) else 0.0


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# --- Fig 1: schematic -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=6, color="#2a6fbb"))
ax.add_patch(Rectangle((-2.6, -1.42), 5.2, 0.18, color="#2a6fbb"))   # bus (input/through)
ax.add_patch(Rectangle((-2.6, 1.24), 5.2, 0.18, color="#2a6fbb"))    # drop bus
ax.add_patch(Rectangle((-0.45, 0.78), 0.9, 0.42, color="#e07a1f", alpha=0.85))  # modulated section
ax.text(0, 0.99, "index\nmodulation", ha="center", va="center", fontsize=8, color="white", weight="bold")
for (x0, y0, x1, y1, col) in [(-3.2, -1.33, -2.65, -1.33, "k"), (2.65, -1.33, 3.3, -1.33, "k"),
                              (-2.65, 1.33, -3.3, 1.33, "k")]:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15, color=col, lw=2))
ax.text(-3.25, -1.0, "Input", ha="center"); ax.text(3.35, -1.0, "Through", ha="center")
ax.text(-3.35, 1.0, "Drop", ha="center")
ax.text(0, -0.1, f"R = {R*1e6:.0f} µm\nn$_g$ = {n_g}", ha="center", fontsize=10)
ax.set_xlim(-3.8, 3.9); ax.set_ylim(-1.8, 1.9)
ax.set_title("Add-drop ring modulator (schematic)")
fig.tight_layout(); fig.savefig(f"{OUT}/01_device_schematic.png", dpi=200); plt.close(fig)

# --- Fig 2: spectrum --------------------------------------------------------
lam = np.linspace(1545e-9, 1575e-9, 20000)
th, dr = spectrum(lam)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(lam * 1e9, 10 * np.log10(th + 1e-12), label="Through", color="#2a6fbb")
ax.plot(lam * 1e9, 10 * np.log10(dr + 1e-12), label="Drop", color="#e07a1f")
ax.axvline(lam_res * 1e9, ls="--", color="gray", lw=1)
ax.set_ylim(-40, 3); ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Transmission (dB)")
ax.set_title(f"Steady-state spectrum  (FSR ≈ {FSR*1e9:.1f} nm, loaded Q ≈ {Q_loaded:,.0f})")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/02_spectrum.png", dpi=200); plt.close(fig)

# --- Fig 3: transient response ---------------------------------------------
t = np.arange(0, 700e-12, 0.1e-12)
t_on, t_off = 250e-12, 450e-12
shift = 1.5 * g                      # resonance pushed away by 1.5 linewidths
thru, drop, ucav = transient(t, step_delta(shift, t_on, t_off))
thru0, drop0, _ = transient(t, lambda _t: 0.0)

fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True,
                        gridspec_kw={"height_ratios": [0.6, 1.4, 1.4]})
dprof = np.array([step_delta(shift, t_on, t_off)(x) for x in t])
axs[0].plot(t * 1e12, dprof / g, color="k")
axs[0].set_ylabel("Detuning\n(units of γ)")
axs[0].set_title("Transient response to a step in ring refractive index")
axs[1].plot(t * 1e12, thru, color="#2a6fbb", label="Through (modulated)")
axs[1].plot(t * 1e12, thru0, color="gray", ls="--", lw=1, label="Through (no modulation)")
axs[1].set_ylabel("Power (norm.)"); axs[1].legend(loc="upper right")
axs[2].plot(t * 1e12, drop, color="#e07a1f", label="Drop (modulated)")
axs[2].plot(t * 1e12, drop0, color="gray", ls="--", lw=1, label="Drop (no modulation)")
axs[2].set_ylabel("Power (norm.)"); axs[2].set_xlabel("Time (ps)"); axs[2].legend(loc="upper right")
for ax in axs:
    ax.axvspan(t_on * 1e12, t_off * 1e12, color="gold", alpha=0.15)
fig.tight_layout(); fig.savefig(f"{OUT}/03_transient_response.png", dpi=200); plt.close(fig)

# --- Fig 4: sweep of step size ---------------------------------------------
shifts = np.linspace(0.2, 4.0, 20) * g
peak, steady = [], []
for s in shifts:
    th_s, _, _ = transient(t, step_delta(s, t_on, t_off))
    window = th_s[(t >= t_on) & (t < t_off)]
    peak.append(window.max()); steady.append(window[-200:].mean())
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(shifts / g, peak, "o-", label="Peak through power", color="#c0392b")
ax.plot(shifts / g, steady, "s-", label="Settled through power", color="#2a6fbb")
ax.axhline(1, color="gray", ls=":")
ax.set_xlabel("Resonance shift (units of linewidth γ)"); ax.set_ylabel("Through-port power (norm.)")
ax.set_title("Overshoot vs. size of index step"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/04_overshoot_vs_shift.png", dpi=200); plt.close(fig)

# --- Summary printed for the README ----------------------------------------
w = (t >= t_on) & (t < t_off)
print(f"T_round_trip = {T_rt*1e12:.2f} ps")
print(f"Loaded Q     = {Q_loaded:.0f}")
print(f"Photon life  = {2/g*1e12:.2f} ps  (energy 1/e)")
print(f"FSR          = {FSR*1e9:.2f} nm")
print(f"Through power on resonance (steady) = {thru[(t>=t_on-20e-12)&(t<t_on)].mean():.4f}")
print(f"Through power peak during step      = {thru[w].max():.3f}")
print(f"Through power settled during step   = {thru[w][-200:].mean():.3f}")
print(f"Shift used = {shift/g:.2f} linewidths = {shift/(2*np.pi)/1e9:.1f} GHz")
