"""Values of tau_n, beta_n, rho^-_n, rho^+_n for four global copulas (for the paper's table).
(A) DQU07 with f(t)=sqrt(t)   (B) DQU07 with f(t)=min(2t,1)
(C) Clayton theta=2           (D) Gumbel theta=2
"""
import numpy as np
from math import comb
from scipy import special, integrate

NS = [2, 3, 4, 5, 10, 20, 50]

def kappa_from_moments(n, mom):          # mom(k) = E[R^{2k}]
    return sum(comb(n, 2 * k) * mom(k) for k in range(1, n // 2 + 1)) / (2 ** (n - 1) - 1)

def rho_from_E(E, n):
    c = (n + 1) / (1 - (n + 1) * 2.0 ** (-n)); d = (n + 1) * 2.0 ** (-n) / (1 - (n + 1) * 2.0 ** (-n))
    return c * E - d

# ---------- (A) and (B): DQU07 family -------------------------------------------------
def dqu_values(f, G, mom_tau, n):
    tau = kappa_from_moments(n, mom_tau)
    x = f(np.array([0.5]))[0]; g = 0.5 / x
    beta = kappa_from_moments(n, lambda k: g * (2 * x - 1) ** (2 * k) + 1 - g)
    z = np.linspace(0, 1, 400001); fz = f(z)
    cum = np.concatenate([[0], np.cumsum(0.5 * (fz[1:] + fz[:-1]) * np.diff(z))]); Fz = cum[-1] - cum
    Gz = G(z); atom = G(np.array([1e-14]))[0]; Gz[0] = atom
    dG = np.diff(Gz); mid = 0.5 * (Fz[1:] + Fz[:-1])
    Em = np.sum((1 - mid) ** n * dG) + atom * (1 - Fz[0]) ** n
    E1m = np.sum(mid ** n * dG) + atom * Fz[0] ** n
    return tau, beta, rho_from_E(E1m, n), rho_from_E(Em, n)

fA = lambda t: np.sqrt(t); GA = lambda t: np.sqrt(t)
momA = lambda k: 1.0 / (1 + 2 * k)                      # E[R^{2k}] = (1-a)/(1-a+2ka), a=1/2
fB = lambda t: np.minimum(2 * t, 1.0); GB = lambda t: np.where(t <= 0.5, 0.5, t)
momB = lambda k: 0.75                                   # R in {0,1}, P(R=1)=1-b^2=3/4

# ---------- (C) Clayton theta=2 and (D) Gumbel theta=2 ---------------------------------
mom_arcsine = lambda k: comb(2 * k, k) / 4 ** k          # E[Y^{2k}], Y arcsine on [-1,1]

def clayton_m(v, a=0.5):
    v = np.asarray(v, dtype=float)
    out = np.empty_like(v)
    small = v < 600
    vs_ = v[small]
    out[small] = np.exp(vs_) * vs_ ** a * special.gammaincc(1 - a, vs_) * special.gamma(1 - a)
    vl = v[~small]
    out[~small] = 1 - a / vl + a * (a + 1) / vl ** 2
    return out

def gumbel_one_minus_m(v):
    return 0.5 * np.sqrt(np.pi / v) * special.erfcx(1.0 / (2.0 * np.sqrt(v)))

ys = np.linspace(-60, 60, 400001); vs = np.exp(ys)
dens_C = np.exp(0.5 * ys - vs - special.gammaln(0.5))                  # Gamma(1/2) in log-v scale
dens_D = np.exp(-0.5 * ys - 1.0 / (4 * vs)) / (2 * np.sqrt(np.pi))     # Levy in log-v scale: v*p(v)
mC = clayton_m(vs); omD = gumbel_one_minus_m(vs); mD = 1 - omD

def E_pow(base, dens, n):
    b = np.clip(base, 1e-300, 1.0)
    return np.trapezoid(np.exp(n * np.log(b)) * dens, ys)

def arch_values(dens, m, c, n):
    tau = kappa_from_moments(n, mom_arcsine)
    p = np.exp(-vs * c)                                   # F_V(1/2)
    qn = E_pow(p, dens, n) + E_pow(1 - p, dens, n)
    beta = (2 ** (n - 1) * qn - 1) / (2 ** (n - 1) - 1)
    return tau, beta, rho_from_E(E_pow(1 - m, dens, n), n), rho_from_E(E_pow(m, dens, n), n)

cC = 2 ** 2 - 1            # Clayton: psi^{-1}(1/2) = 2^theta - 1
cD = np.log(2) ** 2        # Gumbel:  psi^{-1}(1/2) = (ln 2)^theta

rows = {}
for name, fun in [("A", lambda n: dqu_values(fA, GA, momA, n)), ("B", lambda n: dqu_values(fB, GB, momB, n)),
                  ("C", lambda n: arch_values(dens_C, mC, cC, n)), ("D", lambda n: arch_values(dens_D, mD, cD, n))]:
    rows[name] = [fun(n) for n in NS]

lim = {"A": (0.0, 1 - 1 / np.sqrt(2), 0.0, 0.5), "B": (0.75, 1.0, 0.0, 1.0),
       "C": (0.0, 0.0, 2 / np.pi, 0.0), "D": (0.0, 0.0, 0.0, 2 / np.pi)}
names = ["tau", "beta", "rho-", "rho+"]
for key in "ABCD":
    print("Family", key)
    for j, nm in enumerate(names):
        vals = " & ".join(f"{rows[key][i][j]:.4f}" for i in range(len(NS)))
        print(f"  {nm:5s}: {vals} & {lim[key][j]:.4f}")
# sanity: check Clayton beta via closed form for small n: C_n(1/2)=psi(n c)=(1+3n)^(-1/2)
for n in [2, 3, 4]:
    Cn = (1 + 3 * n) ** -0.5
    Cbar = sum(comb(n, j) * (-1) ** j * (1 + 3 * j) ** -0.5 for j in range(0, n + 1))
    print("Clayton beta check n=", n, (2 ** (n - 1) * (Cn + Cbar) - 1) / (2 ** (n - 1) - 1), rows["C"][NS.index(n)][1])
    Dn = 2 ** (-np.sqrt(n)); Dbar = sum(comb(n, j) * (-1) ** j * 2 ** (-np.sqrt(j)) for j in range(0, n + 1))
    print("Gumbel  beta check n=", n, (2 ** (n - 1) * (Dn + Dbar) - 1) / (2 ** (n - 1) - 1), rows["D"][NS.index(n)][1])
