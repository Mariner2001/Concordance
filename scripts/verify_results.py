"""Numerical verification of the formulas stated in the manuscript
"Multivariate concordance of global copulas as the number of inputs grows".

Each check prints the value given by the paper next to an independent
computation (exact formula, numerical integration or Monte Carlo).
Run from the scripts folder:  python3 verify_results.py   (one or two minutes)
"""
import os
import re
import subprocess
import sys
from math import comb, pi, sqrt

import numpy as np
from scipy import special, stats

rng = np.random.default_rng(2026)
GRID = np.linspace(0.0, 1.0, 1_000_001)
ALL_OK = True


def w(n, k):
    return comb(n, 2 * k) / (2 ** (n - 1) - 1) if 2 * k <= n else 0.0


def kappa(n, mom):
    """sum_k w_{n,k} mom(k), Theorem 4.2."""
    return sum(w(n, k) * mom(k) for k in range(1, n // 2 + 1))


def h(n):
    return (n + 1) / (2 ** n - n - 1)


def report(label, paper, check, tol):
    global ALL_OK
    ok = abs(paper - check) <= tol
    ALL_OK &= ok
    print(f"  [{'OK ' if ok else 'BAD'}] {label}: paper={paper:.6f}  check={check:.6f}")


def report_mc(label, paper, samples, scale=1.0, shift=0.0):
    """Monte Carlo check: tolerance of four standard errors."""
    est = scale * np.mean(samples) + shift
    se = abs(scale) * np.std(samples) / np.sqrt(samples.size)
    report(label + " (MC)", paper, est, max(4 * se, 1e-6))


def flag(label, ok):
    global ALL_OK
    ALL_OK &= bool(ok)
    print(f"  [{'OK ' if ok else 'BAD'}] {label}")


# ---------------------------------------------------------------------------
# Family C_{f,n}(u) = u_[1] prod_{i>=2} f(u_[i])
# ---------------------------------------------------------------------------
GENS = {
    "sqrt(t)": lambda t: np.sqrt(t),
    "0.4t+0.6": lambda t: 0.4 * t + 0.6,
    "min(2t,1)": lambda t: np.minimum(2 * t, 1.0),
    "max(t,0.6)": lambda t: np.maximum(t, 0.6),
    "2t/(1+t)": lambda t: 2 * t / (1 + t),
}


def tables(f):
    fg = f(GRID)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(GRID > 0, GRID / fg, np.nan)
    G[0] = G[1]                                   # atom G(0) = lim t/f(t)
    F = np.concatenate([[0], np.cumsum(0.5 * (fg[1:] + fg[:-1]) * np.diff(GRID))])
    return fg, np.maximum.accumulate(G), F[-1] - F   # f, G, F_f(t) = int_t^1 f


def sample(f, n, N):
    """U_i = max(W_i, Z) with W_i ~ f and Z ~ G (Section 2.3)."""
    fg, G, _ = tables(f)
    last = len(GRID) - 1
    W = GRID[np.minimum(np.searchsorted(fg, rng.random((N, n))), last)]
    Z = GRID[np.minimum(np.searchsorted(G, rng.random(N)), last)]
    return np.maximum(W, Z[:, None])


def C_family(f, U):
    Us = np.sort(U, axis=1)
    return Us[:, 0] * np.prod(f(Us[:, 1:]), axis=1)


def tau_a(f, n):
    """Corollary 5.2(a), first expression."""
    mu = lambda k: 1 - 2 * k / (2 * k - 1) * (1 - 2 * np.trapezoid(GRID * f(GRID) ** (4 * k - 2), GRID))
    return kappa(n, mu)


def tau_b(f, n):
    """Corollary 5.2(a), second expression."""
    s = sum(comb(n - 1, j) / j * (1 - 2 * np.trapezoid(GRID * f(GRID) ** (2 * j), GRID))
            for j in range(1, n, 2))
    return 1 - n / (2 ** (n - 1) - 1) * s


def beta_15(f, n):
    """Eq. (15)."""
    x = float(f(np.array([0.5]))[0])
    return (2 ** (n - 2) * (2 + x ** (n - 1) - (1 - (1 - x) ** n) / x) - 1) / (2 ** (n - 1) - 1)


def beta_direct(f, n):
    """Definition (2) with C(t1) = t f(t)^{n-1} and inclusion-exclusion for the survival function."""
    x = float(f(np.array([0.5]))[0])
    Cn = 0.5 * x ** (n - 1)
    Cbar = 1 + sum((-1) ** k * comb(n, k) * 0.5 * x ** (k - 1) for k in range(1, n + 1))
    return (2 ** (n - 1) * (Cn + Cbar) - 1) / (2 ** (n - 1) - 1)


def phi_direct(f, n):
    """Definition (5): diagonal t f(t)^{n-1}; survival diagonal 1 - t(1-(1-f)^n)/f."""
    t = GRID[1:]; fg = f(t)
    delta = t * fg ** (n - 1)
    cbar = 1 - t * (1 - (1 - fg) ** n) / fg
    return (n + 1) / (n - 1) * (np.trapezoid(delta, t) + np.trapezoid(cbar, t)) - 2 / (n - 1)


print("1. Corrections to [7] (Section 3)")
f = GENS["sqrt(t)"]
pub = (4 - 8 * sum(k / 2 ** (k - 1) * np.trapezoid(GRID * (1 - f(GRID) ** 2) ** (k - 1), GRID)
                   for k in (2, 3)) - 1) / 3
report("published tau_3 for f=sqrt(t) is 7/18", 7 / 18, pub, 1e-6)
report("correct tau_3 for f=sqrt(t) is 1/3", 1 / 3, tau_a(f, 3), 1e-6)
x = 2 ** -0.5
report("published beta_2 equals 3 for every f", 3.0, 2 + x + (1 - (1 - x) ** 2) / x - 1, 1e-12)
V = rng.random(1_000_000)
report_mc("int C dC = 1/4 for C = M_2(u1,u2) u3", 0.25, V * rng.random(V.size))

print("2. Corollary 5.2 against independent computations")
for name, f in GENS.items():
    _, _, Ff = tables(f)
    for n in (2, 3, 4, 6):
        U = sample(f, n, 400_000)
        a = tau_a(f, n)
        report(f"{name:10s} n={n}: tau, the two expressions of (a)", a, tau_b(f, n), 1e-6)
        report_mc(f"{name:10s} n={n}: tau vs E[C(U)]", a, C_family(f, U),
                  2 ** n / (2 ** (n - 1) - 1), -1 / (2 ** (n - 1) - 1))
        report(f"{name:10s} n={n}: beta, Eq. (15) vs definition", beta_15(f, n), beta_direct(f, n), 1e-12)
        report_mc(f"{name:10s} n={n}: int C dPi", n * np.trapezoid(GRID * Ff ** (n - 1), GRID),
                  C_family(f, rng.random((400_000, n))))
        report_mc(f"{name:10s} n={n}: int Pi dC", 1 - n * np.trapezoid(GRID * (1 - Ff) ** (n - 1), GRID),
                  np.prod(U, axis=1))
    taus = [tau_a(f, n) for n in range(2, 13)]
    flag(f"{name:10s} tau_n non-increasing and tau_2 = tau_3",
         all(b <= a + 1e-12 for a, b in zip(taus, taus[1:])) and abs(taus[0] - taus[1]) < 1e-9)

print("3. Limits (Corollary 5.2(d)) and Example 5.6")
al = 0.5; f = lambda t: t ** al
# f(t)=sqrt(t): Z = V^2 with V uniform and m = 1 - F_f(Z) = 1/3 + (2/3) V^3 (Proposition 5.1(c))
from scipy import integrate
for n in (1000, 10000):
    Emn = integrate.quad(lambda v: (1 / 3 + 2 / 3 * v ** 3) ** n, 0, 1, points=[1 - 20 / n], limit=500)[0]
    report(f"CA alpha=1/2: rho^+_n ~ (n+1)E[m^n] -> 1-alpha = lambda_U (n={n})", 1 - al, (n + 1) * Emn, 2e-3)
report("CA alpha=1/2: beta_n -> 1-2^(alpha-1) (n=200)", 1 - 2 ** (al - 1), beta_15(f, 200), 1e-9)
report("CA alpha=1/2: mu_3 closed form", (1 - al) / (1 + 5 * al),
       1 - 6 / 5 * (1 - 2 * np.trapezoid(GRID * f(GRID) ** 10, GRID)), 1e-6)
report("CA alpha=1/2: phi_n -> (1-a)/(2-a) (n=400)", (1 - al) / (2 - al), phi_direct(f, 400), 1e-2)
for b in (0.3, 0.7):
    f = lambda t, b=b: np.minimum(t / b, 1.0)
    report(f"ordinal sum b={b}: tau_7 = 1-b^2", 1 - b ** 2, tau_a(f, 7), 1e-5)
    report(f"ordinal sum b={b}: beta_n limit (n=400)", 1.0 if b <= 0.5 else 1 - b, beta_15(f, 400), 1e-6)
    report(f"ordinal sum b={b}: phi_n -> 1-b^2 (n=400)", 1 - b ** 2, phi_direct(f, 400), 1e-2)
c = 0.6; f = lambda t: np.maximum(t, c)
for n in (3, 5, 8):
    report(f"max(t,0.6) n={n}: closed form of tau_n", ((1 + c * c) ** n + (1 - c * c) ** n - 2) / (2 ** n - 2),
           tau_a(f, n), 1e-5)

print("4. Theorem 5.4 (universality) and Corollary 5.5")
cases = [("|arcsine|", lambda x: 2 / pi * np.arcsin(np.clip(x, 0, 1)), lambda k: comb(2 * k, k) / 4 ** k),
         ("Beta(2,3)", lambda x: stats.beta.cdf(x, 2, 3), lambda k: special.beta(2 + 2 * k, 3) / special.beta(2, 3))]
for label, K, mom in cases:
    s = np.linspace(0, 1, 2_000_001)
    q = s * np.sqrt(K(s ** 2))                        # q_nu, strictly increasing here
    f = lambda t, q=q, s=s: np.interp(t, q, s)       # f_nu = generalized inverse of q_nu
    tt = np.linspace(1e-3, 1, 5001); fv = f(tt)
    flag(f"{label}: f_nu is a generator (monotone, f(t)/t non-increasing, f(1)=1)",
         np.all(np.diff(fv) >= 0) and np.all(np.diff(fv / tt) <= 1e-9) and abs(fv[-1] - 1) < 1e-9)
    for n in (2, 4, 6):
        report(f"{label}: tau_n of C_(f_nu) equals sum_k w_nk E[R^2k], n={n}", kappa(n, mom), tau_a(f, n), 2e-3)
for b in (0.4, 0.8):
    f = lambda t, b=b: np.minimum(t / b, 1.0)
    report(f"upper boundary tau_4 = tau_2, min(t/b,1), b={b}", tau_a(f, 2), tau_a(f, 4), 1e-5)
for c in (0.4, 0.8):
    f = lambda t, c=c: np.maximum(t, c); t2 = tau_a(f, 2)
    report(f"lower boundary tau_4 = tau_2(6+tau_2)/7, max(t,c), c={c}", t2 * (6 + t2) / 7, tau_a(f, 4), 1e-5)

print("5. Remark 4.3(a): exchangeable 4-copula with tau_2 = -1/3 and tau_4 = -1/7")


def sample_mix(N):
    pats = np.array([(1, 1, -1, -1), (1, -1, 1, -1), (1, -1, -1, 1)])
    P = pats[rng.integers(3, size=N)]; Vm = rng.random(N)
    return np.where(P > 0, Vm[:, None], 1 - Vm[:, None])


U, Up = sample_mix(1_000_000), sample_mix(1_000_000)
report_mc("tau_4", -1 / 7, np.all(Up < U, axis=1).astype(float), 16 / 7, -1 / 7)
report_mc("tau_2", -1 / 3, np.all(Up[:, :2] < U[:, :2], axis=1).astype(float), 4.0, -1.0)

print("6. Archimedean global copulas (Section 6)")
N = 1_000_000
for n in (2, 3, 4):
    a = 0.5
    Vc = rng.gamma(a, 1.0, size=N)
    Uc = (1 + rng.exponential(size=(N, n)) / Vc[:, None]) ** (-a)
    closed = special.poch(a, n) / special.poch(2 * a, n)
    report_mc(f"Clayton theta=2, n={n}: int C dC = (1/theta)_n/(2/theta)_n", closed,
              (np.sum(Uc ** -2.0, axis=1) - n + 1) ** -0.5)
    report(f"Clayton theta=2, n={n}: (1/2)_n/(1)_n = binom(2n,n)/4^n", comb(2 * n, n) / 4 ** n, closed, 1e-12)
    Vg = 1 / (2 * rng.standard_normal(N) ** 2)        # Laplace transform exp(-sqrt(s))
    Ug = np.exp(-np.sqrt(rng.exponential(size=(N, n)) / Vg[:, None]))
    report_mc(f"Gumbel theta=2, n={n}: int C dC = binom(2n,n)/4^n", comb(2 * n, n) / 4 ** n,
              np.exp(-np.sqrt(np.sum(np.log(Ug) ** 2, axis=1))))

ys = np.linspace(-200, 60, 1_300_001); vs = np.exp(ys)


def clayton_m(v, a):
    out = np.empty_like(v); sm = v < 600
    out[sm] = np.exp(v[sm]) * v[sm] ** a * special.gammaincc(1 - a, v[sm]) * special.gamma(1 - a)
    vl = v[~sm]; out[~sm] = 1 - a / vl + a * (a + 1) / vl ** 2
    return out


def E_pow(base, dens, n):
    return np.trapezoid(np.exp(n * np.log(np.clip(base, 1e-300, 1.0))) * dens, ys)


for th in (2.0, 4.0):
    a = 1 / th
    dens = np.exp(a * ys - vs - special.gammaln(a))   # Gamma(a) density in log scale
    m = clayton_m(vs, a)
    n = 100_000
    report(f"Clayton theta={th:g}: rho^-_n close to (theta/pi)sin(pi/theta) at n=1e5", np.sin(pi * a) / (pi * a),
           (n + 1) * E_pow(1 - m, dens, n), 1.5e-2)
densG = np.exp(-0.5 * ys - 1 / (4 * vs)) / (2 * np.sqrt(pi))            # Levy density in log scale
mG = 1 - 0.5 * np.sqrt(pi / vs) * special.erfcx(1 / (2 * np.sqrt(vs)))
n = 100_000
report("Gumbel theta=2: rho^+_n close to 2/pi at n=1e5", 2 / pi, (n + 1) * E_pow(mG, densG, n), 1.5e-2)
densC = np.exp(0.5 * ys - vs - special.gammaln(0.5)); mC = clayton_m(vs, 0.5)
for n in (2, 5, 20):
    report(f"Proposition 6.2, n={n}: rho^-(Clayton 2) = rho^+(Gumbel 2)",
           h(n) * (2 ** n * E_pow(1 - mC, densC, n) - 1), h(n) * (2 ** n * E_pow(mG, densG, n) - 1), 2e-4)

print("7. Gaussian global copulas (Proposition 6.5)")
rc = 0.5; gam = sqrt(rc / (2 - rc)); z = 1.3; xi = np.linspace(-9, 9, 200_001)
report("E[Phi(X_1) | Z=z] = Phi(gamma z)", stats.norm.cdf(gam * z),
       np.trapezoid(stats.norm.cdf(sqrt(rc) * z + sqrt(1 - rc) * xi) * stats.norm.pdf(xi), xi), 1e-8)
zz = np.linspace(-12, 12, 400_001); vals = []
for n in (2, 10, 100, 1000, 10000):
    Em = np.trapezoid(stats.norm.cdf(gam * zz) ** n * stats.norm.pdf(zz), zz)
    vals.append(h(n) * (2 ** n * Em - 1) if n <= 60 else (n + 1) * Em)
print("  rho^+_n (rho=0.5) for n = 2, 10, 100, 1000, 10000:", np.round(vals, 4))
flag("Gaussian rho^+_n tends to 0", vals[-1] < 0.02 and all(b < a for a, b in zip(vals[1:], vals[2:])))

print("8. Table 1 (recomputed with table_values.py and compared with the manuscript)")
here = os.path.dirname(os.path.abspath(__file__))
tex_path = os.path.join(here, "..", "global_copulas_concordance.tex")
if not os.path.exists(tex_path):
    print("  manuscript not found next to the scripts folder: check skipped")
else:
    out = subprocess.run([sys.executable, os.path.join(here, "table_values.py")],
                         capture_output=True, text=True).stdout
    tex = open(tex_path).read()
    rows_tex = re.findall(r"\$(\\tau_n|\\beta_n|\\rho\^-_n|\\rho\^\+_n)\$\s*&([^\\]+)\\\\", tex)
    rows_py = re.findall(r"(tau|beta|rho-|rho\+)\s*:\s*([^\n]+)", out)
    bad = 0
    for (_, vt), (_, vp) in zip(rows_tex, rows_py):
        for a_, b_ in zip([float(v) for v in vt.split("&")], [float(v) for v in vp.split("&")]):
            if abs(a_ - b_) > 1.5e-4:
                bad += 1
    flag(f"{len(rows_tex)} rows of Table 1 compared, {bad} mismatches", bad == 0 and len(rows_tex) == 16 == len(rows_py))

print("\nALL CHECKS PASSED" if ALL_OK else "\nSOME CHECKS FAILED")
