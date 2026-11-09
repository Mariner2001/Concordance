import numpy as np
from math import comb
from scipy import special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
                     "axes.linewidth": 0.8, "legend.frameon": False})
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + os.sep
import os; os.makedirs(OUT, exist_ok=True)

# ---------------- Figure 1: attainable (tau_2, tau_4 - tau_2) ----------------
t2 = np.linspace(-1/3, 1, 801)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
lo = (8 * np.maximum(t2, 0) - 1) / 7 - t2
up = (1 + 6 * t2) / 7 - t2
ax.fill_between(t2, lo, up, color="0.88", lw=0, label="exchangeable 4-copulas")
g = np.linspace(0, 1, 401)
ax.fill_between(g, -g * (1 - g) / 7, 0 * g, color="0.35", lw=0, label="global copulas")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"$\tau_2$"); ax.set_ylabel(r"$\tau_4-\tau_2$")
ax.set_xlim(-0.36, 1.02); ax.set_ylim(-0.16, 0.205)
ax.annotate(r"$\tau_4=\tau_2$:  $f(t)=\min(t/b,1)$", xy=(0.3, 0.0), xytext=(0.12, 0.085),
            arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=8.5)
ax.annotate(r"$\tau_4=\tau_2(6+\tau_2)/7$:  $f(t)=\max(t,c)$", xy=(0.5, -0.25/7), xytext=(0.12, -0.125),
            arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=8.5)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT + "fig_region.pdf"); plt.close(fig)

# ---------------- Figure 2: tau_n for DQU07 generators and Spearman for Clayton ----------------
def tau_from_mom(n, mom):
    return sum(comb(n, 2 * k) * mom(k) for k in range(1, n // 2 + 1)) / (2 ** (n - 1) - 1)

ns = np.arange(2, 41)
gens = [
    (r"$f(t)=\min(2t,1)$", lambda k: 0.75),
    (r"$f(t)=t^{1/4}$", lambda k: 0.75 / (0.75 + 0.5 * k)),
    (r"$f(t)=0.4t+0.6$", None),
    (r"$f(t)=t^{1/2}$", lambda k: 1 / (1 + 2 * k)),
    (r"$f(t)=\max(t,0.8)$", lambda k: 0.64 ** (2 * k)),
]
def mom_frechet(k, al=0.4):
    # E[R^{2k}] = 1 - 2k/(2k-1) (1 - 2 int_0^1 t f^{4k-2})
    t = np.linspace(0, 1, 200001); f = al * t + 1 - al
    I = np.trapezoid(t * f ** (4 * k - 2), t)
    return 1 - 2 * k / (2 * k - 1) * (1 - 2 * I)
cache = {}
def mF(k):
    if k not in cache: cache[k] = mom_frechet(k)
    return cache[k]
gens[2] = (gens[2][0], mF)

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
ax = axs[0]
styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
for (lab, mom), st in zip(gens, styles):
    ax.plot(ns, [tau_from_mom(int(n), mom) for n in ns], linestyle=st, color="k", lw=1.0, label=lab)
ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$\tau_n(C_{f,n})$"); ax.set_ylim(0, 0.85)
ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.85))
ax.set_title("(a)", fontsize=10)

# Clayton rho^- for theta = 1, 2, 4
def rho_from_E(E, n):
    c = (n + 1) / (1 - (n + 1) * 2.0 ** (-n)); d = (n + 1) * 2.0 ** (-n) / (1 - (n + 1) * 2.0 ** (-n))
    return c * E - d
ys = np.linspace(-60, 6.5, 300001); vs = np.exp(ys)
def m_clayton(v, a):
    if a < 1:
        return np.exp(v) * v ** a * special.gammaincc(1 - a, v) * special.gamma(1 - a)
    return v * np.exp(v) * special.exp1(v)          # a = 1
nn = np.unique(np.round(np.logspace(np.log10(2), 4, 60)).astype(int))
ax = axs[1]
for theta, st in [(4.0, "-"), (2.0, "--"), (1.0, ":")]:
    a = 1 / theta
    m = m_clayton(vs, a)
    dens = np.exp(a * ys - vs - special.gammaln(a))
    vals_minus = [rho_from_E(np.trapezoid(np.exp(n * np.log(np.clip(1 - m, 1e-300, 1))) * dens, ys)
                             + (special.gammainc(a, np.exp(-60))), n) for n in nn]
    ax.semilogx(nn, vals_minus, linestyle=st, color="k", lw=1.0, label=rf"$\rho_n^-$, $\theta={theta:g}$")
    if theta == 2.0:
        vals_plus = [rho_from_E(np.trapezoid(np.exp(n * np.log(np.clip(m, 1e-300, 1))) * dens, ys), n) for n in nn]
        ax.semilogx(nn, vals_plus, linestyle="-.", color="0.5", lw=1.0, label=r"$\rho_n^+$, $\theta=2$")
for theta in [4.0, 2.0]:
    a = 1 / theta
    ax.axhline(np.sin(np.pi * a) / (np.pi * a), color="0.6", lw=0.6)
ax.set_xlabel(r"$n$"); ax.set_ylabel("Spearman's rho (Clayton)"); ax.set_ylim(-0.02, 1.0)
ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(0.99, 0.60))
ax.text(1.6e3, np.sin(np.pi/4)/(np.pi/4)+0.02, r"$2\sqrt{2}/\pi$", fontsize=8)
ax.text(1.6e3, 2/np.pi+0.02, r"$2/\pi$", fontsize=8)
ax.set_title("(b)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT + "fig_limits.pdf"); plt.close(fig)
print("figures written")
