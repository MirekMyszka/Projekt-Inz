"""PILOT (prototyp, 2026-10-13): czy metody z docs/plan_etap_1.md wykrywają anomalie w scenariuszach S1, S2, S3.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.symulacje.pilot_etap1 S1 50 30     # scenariusz, długość anomalii L, liczba szeregów
    python -m src.symulacje.pilot_etap1 S2 50 30
    python -m src.symulacje.pilot_etap1 S3 100 30

To jest szybki, uproszczony prototyp do sprawdzenia sensu doboru metod - NIE docelowa implementacja.
Wyniki i wnioski: wyniki/pilot_etap1_wyniki.txt oraz docs/plan_etap_1.md, rozdz. 8.
Parametry pilota: S1 Δ=1σ, c=2, φ=0,7; S2 szok 5σ, φ 0,7→0, równowaga +2σx; S3 faza π/2, amplituda ×0,5, okres ×0,5.
"""
import sys, warnings
import numpy as np
import ruptures as rpt
import stumpy
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")
N, W, STRIDE, P, A_SIN = 3000, 50, 5, 50, 3.0

# ---------------------------------------------------------------- generatory
def ar1(n, phi, sigma, rng, mu=None, x0=0.0):
    x = np.empty(n); prev = x0; e = rng.normal(0, sigma, n)
    mu = np.zeros(n) if mu is None else mu
    for i in range(n):
        prev = mu[i] + phi * (prev - (mu[i - 1] if i else mu[i])) + e[i]; x[i] = prev
    return x

def miejsca(rng, L, n=N, k=3):
    seg = n // k
    return [s * seg + rng.integers(100, seg - L - 100) for s in range(k)]

def gen(scen, rng, L, z_anomaliami=True):
    lab = np.zeros(N, int)
    if scen == "S1":
        x = rng.normal(0, 1, N)
        if z_anomaliami:
            for typ, st in zip(rng.permutation([1, 2, 3]), miejsca(rng, L)):
                sl = slice(st, st + L); lab[sl] = typ
                if typ == 1: x[sl] += 1.0                       # S1-A poziom, Δ = 1σ
                if typ == 2: x[sl] *= 2.0                       # S1-B zmienność, c = 2
                if typ == 3:                                    # S1-C zależność, φ = 0.7, ta sama wariancja
                    x[sl] = ar1(L, 0.7, np.sqrt(1 - 0.49), rng, x0=rng.normal())
    elif scen == "S2":
        phi, s_e = 0.7, 1.0; s_x = s_e / np.sqrt(1 - phi**2)
        e = rng.normal(0, s_e, N); mu = np.zeros(N); ph = np.full(N, phi); se = np.full(N, s_e)
        if z_anomaliami:
            for typ, st in zip(rng.permutation([1, 2, 3]), miejsca(rng, L)):
                if typ == 1:                                    # S2-A szok innowacyjny 5σ, wygasa
                    e[st] += 5 * s_e; dl = int(np.ceil(np.log(0.05 / 5) / np.log(phi))) + 1
                    lab[st:st + dl] = 1
                if typ == 2:                                    # S2-B φ: 0.7 -> 0, wariancja zachowana
                    ph[st:st + L] = 0.0; se[st:st + L] = s_x; lab[st:st + L] = 2
                if typ == 3:                                    # S2-C poziom równowagi +2σx
                    mu[st:st + L] = 2 * s_x; lab[st:st + L] = 3
        x = np.empty(N); prev = 0.0
        for i in range(N):
            eps = e[i] / s_e * se[i]
            prev = mu[i] + ph[i] * (prev - mu[i]) + eps; x[i] = prev
    else:  # S3
        t = np.arange(N); faza = np.zeros(N); amp = np.full(N, A_SIN); per = np.full(N, float(P))
        if z_anomaliami:
            for typ, st in zip(rng.permutation([1, 2, 3]), miejsca(rng, L)):
                sl = slice(st, st + L); lab[sl] = typ
                if typ == 1: faza[sl] = np.pi / 2                # S3-A faza
                if typ == 2: amp[sl] = 0.5 * A_SIN               # S3-B amplituda ×0.5
                if typ == 3: per[sl] = 0.5 * P                   # S3-C okres ×0.5
        x = amp * np.sin(2 * np.pi * t / per + faza) + rng.normal(0, 1, N)
    return x, lab

# ---------------------------------------------------------------- narzędzia
def okna(x, w=W, stride=STRIDE):
    starty = np.arange(0, len(x) - w + 1, stride)
    return starty, np.stack([x[s:s + w] for s in starty])

def do_punktow(starty, wyn, w=W, n=N):
    """wynik okna -> punkty (maksimum z okien pokrywających punkt)"""
    out = np.full(n, -np.inf)
    for s, v in zip(starty, wyn): out[s:s + w] = np.maximum(out[s:s + w], v)
    out[np.isinf(out)] = np.nanmin(wyn)
    return out

def rz(v):  # robust standaryzacja cech
    med = np.median(v, 0); mad = np.median(np.abs(v - med), 0) * 1.4826 + 1e-9
    return (v - med) / mad

def ac(m, lag=1):
    m = m - m.mean(1, keepdims=True)
    return (m[:, lag:] * m[:, :-lag]).sum(1) / ((m**2).sum(1) + 1e-12)

def knn_score(F, starty, k=10, w=W):
    D = np.sqrt(((F[:, None, :] - F[None, :, :])**2).sum(-1))
    D[np.abs(starty[:, None] - starty[None, :]) < w] = np.inf   # bez „trywialnych” sąsiadów (nakładające się okna)
    return np.sort(D, 1)[:, :k].mean(1), D

def lof_score(D, k=10):
    Dc = np.where(np.isinf(D), 1e6, D); np.fill_diagonal(Dc, 0)
    lof = LocalOutlierFactor(n_neighbors=k, metric="precomputed").fit(Dc)
    return -lof.negative_outlier_factor_

def cusum(z, k=0.5):
    sp = np.zeros(len(z)); sm = np.zeros(len(z)); a = b = 0.0
    for i, v in enumerate(z): a = max(0, a + v - k); b = max(0, b - v - k); sp[i] = a; sm[i] = b
    # offline: wygładzamy w obie strony (forward + backward), żeby alarm obejmował cały fragment
    sp2 = np.zeros(len(z)); sm2 = np.zeros(len(z)); a = b = 0.0
    for i in range(len(z) - 1, -1, -1): a = max(0, a + z[i] - k); b = max(0, b - z[i] - k); sp2[i] = a; sm2[i] = b
    return np.maximum.reduce([sp, sm, sp2, sm2])

def st(v):
    med = np.median(v); s = np.median(np.abs(v - med)) * 1.4826 + 1e-12
    return (v - med) / s

def pelt_score(y, model, pen, x_dla_stat):
    try: bk = rpt.Pelt(model=model, min_size=10, jump=10, params={"order": 1} if model == "ar" else None).fit(y).predict(pen=pen)
    except TypeError: bk = rpt.Pelt(model=model, min_size=10, jump=10).fit(y).predict(pen=pen)
    out = np.zeros(len(y)); start = 0
    mu, var = np.median(x_dla_stat), np.var(x_dla_stat)
    for end in bk:
        seg = x_dla_stat[start:end]; n = len(seg)
        zs = [abs(seg.mean() - mu) * np.sqrt(n / var), abs(np.log(seg.var() / var + 1e-9)) * np.sqrt(n / 2),
              abs(ac(seg[None])[0] - ac(x_dla_stat[None])[0]) * np.sqrt(n)]
        out[start:end] = max(zs); start = end
    return out

def harm(m, starty, w):
    t = (starty[:, None] + np.arange(w)[None, :]) * 2 * np.pi / P
    S, C = np.sin(t), np.cos(t)
    a = (m * S).sum(1) / (S * S).sum(1); b = (m * C).sum(1) / (C * C).sum(1)
    res = m - a[:, None] * S - b[:, None] * C
    spec = np.abs(np.fft.rfft(m - m.mean(1, keepdims=True), axis=1)); dom = spec.argmax(1)
    return np.c_[np.hypot(a, b), np.arctan2(b, a), dom, res.std(1)]

# ---------------------------------------------------------------- metody
def metody(scen, x):
    out = {}
    if scen == "S1":
        zs = lambda v: (v - v.mean()) / v.std()          # CUSUM: standaryzacja średnią i odch. std (x² jest skośne - mediana przesuwa środek)
        z1 = zs(x); z2 = zs(x**2); z3 = zs(np.r_[0, x[1:] * x[:-1]])
        out["CUSUM (x, x², x·x₋₁)"] = np.maximum.reduce([cusum(z1), cusum(z2), cusum(z3)])
        pen = 2 * np.log(N) * 2
        out["PELT"] = pelt_score(np.c_[x, np.r_[0, x[1:] * x[:-1]]], "normal", pen, x)   # x i iloczyn sąsiadów naraz
        mp = stumpy.aamp(x, m=W)[:, 0].astype(float); out["Matrix Profile (bez norm.)"] = do_punktow(np.arange(len(mp)), mp, w=W)
        mpn = stumpy.stump(x, m=W)[:, 0].astype(float); out["  MP z normalizacją (porówn.)"] = do_punktow(np.arange(len(mpn)), mpn, w=W)
        s, m = okna(x); F = rz(np.c_[m.mean(1), m.std(1), ac(m), kurtosis(m, 1), np.ptp(m, 1)])
        kn, D = knn_score(F, s); out["cechy okien + kNN"] = do_punktow(s, kn)
        out["Isolation Forest"] = do_punktow(s, -IsolationForest(n_estimators=200, random_state=0).fit(F).score_samples(F))
        out["LOF"] = do_punktow(s, lof_score(D))
    elif scen == "S2":
        phi = np.dot(x[1:] - x.mean(), x[:-1] - x.mean()) / np.dot(x[:-1] - x.mean(), x[:-1] - x.mean())
        r = np.r_[0, (x[1:] - x.mean()) - phi * (x[:-1] - x.mean())]
        zs = lambda v: (v - v.mean()) / v.std()
        zr = zs(r)
        out["reszty AR + CUSUM"] = np.maximum.reduce([cusum(zr), cusum(zs(r**2)), cusum(zs(np.r_[0, r[1:] * r[:-1]])), np.abs(zr)])
        out["PELT (koszt AR)"] = pelt_score(x, "ar", 2 * np.log(N), x)
        mpn = stumpy.stump(x, m=W)[:, 0].astype(float); out["Matrix Profile (z norm.)"] = do_punktow(np.arange(len(mpn)), mpn)
        s, m = okna(x); _, mr = okna(r)
        F = rz(np.c_[m.mean(1), m.std(1), ac(m, 1), ac(m, 2), ac(m, 3), np.abs(mr).max(1)])
        kn, D = knn_score(F, s); out["cechy okien + kNN"] = do_punktow(s, kn)
        out["Isolation Forest"] = do_punktow(s, -IsolationForest(n_estimators=200, random_state=0).fit(F).score_samples(F))
        out["One-Class SVM"] = do_punktow(s, -OneClassSVM(nu=0.05, gamma="scale").fit(F).score_samples(F))
    else:
        res = STL(x, period=P, robust=True).fit().resid
        e = st(res**2); ew = np.zeros(N); v = 0.0
        for i in range(N): v = 0.05 * e[i] + 0.95 * v; ew[i] = v
        ew2 = np.zeros(N); v = 0.0
        for i in range(N - 1, -1, -1): v = 0.05 * e[i] + 0.95 * v; ew2[i] = v
        out["STL + EWMA"] = np.maximum(ew, ew2)
        out["PELT (reszty STL)"] = pelt_score(res, "normal", 2 * np.log(N), res)
        mpn = stumpy.stump(x, m=P)[:, 0].astype(float); out["Matrix Profile (z norm., okno=P)"] = do_punktow(np.arange(len(mpn)), mpn, w=P)
        mpa = stumpy.aamp(x, m=P)[:, 0].astype(float); out["  MP bez normalizacji (porówn.)"] = do_punktow(np.arange(len(mpa)), mpa, w=P)
        s, m = okna(x, w=P); H = harm(m, s, P)
        F = rz(np.c_[H[:, 0], np.cos(H[:, 1]), np.sin(H[:, 1]), H[:, 2], H[:, 3]])
        kn, D = knn_score(F, s, w=P); out["cechy cykli + kNN"] = do_punktow(s, kn, w=P)
        out["Isolation Forest"] = do_punktow(s, -IsolationForest(n_estimators=200, random_state=0).fit(F).score_samples(F), w=P)
        mm = m / A_SIN
        ae = MLPRegressor(hidden_layer_sizes=(16, 4, 16), max_iter=400, random_state=0).fit(mm, mm)
        out["autoenkoder"] = do_punktow(s, ((ae.predict(mm) - mm)**2).mean(1), w=P)
    return out

# ---------------------------------------------------------------- ocena
def wykryte(alarm, lab, typ, tol):
    idx = np.where(lab == typ)[0]
    if len(idx) == 0: return np.nan
    a, b = max(0, idx[0] - tol), min(N, idx[-1] + tol + 1)
    return float(alarm[a:b].any())

def fa_epizody(alarm, lab, tol):
    norm = alarm.copy()
    for typ in (1, 2, 3):
        idx = np.where(lab == typ)[0]
        if len(idx): norm[max(0, idx[0] - tol):idx[-1] + tol + 1] = False
    return int((np.diff(np.r_[0, norm.astype(int)]) == 1).sum())   # liczba osobnych epizodów fałszywych alarmów

def uruchom(scen, L, R=30, seed=1):
    rng = np.random.default_rng(seed); tol = (P if scen == "S3" else W) // 2
    czyste = [metody(scen, gen(scen, rng, L, False)[0]) for _ in range(R)]
    nazwy = list(czyste[0])
    # próg: kwantyl 99,5% wyników (po standaryzacji w szeregu) na szeregach BEZ anomalii
    def norm(v): return v
    prog = {k: np.quantile([c[k].max() for c in czyste], 0.80) + 1e-9 for k in nazwy}
    fa_czyste = {k: np.mean([fa_epizody(norm(c[k]) > prog[k], np.zeros(N, int), tol) for c in czyste]) for k in nazwy}
    wyn = {k: {"rec": [[], [], []], "auc": [[], [], []], "fa": []} for k in nazwy}
    for _ in range(R):
        x, lab = gen(scen, rng, L, True); sc = metody(scen, x)
        for k in nazwy:
            v = norm(sc[k]); al = v > prog[k]
            for j, typ in enumerate((1, 2, 3)):
                wyn[k]["rec"][j].append(wykryte(al, lab, typ, tol))
                msk = (lab == 0) | (lab == typ)
                wyn[k]["auc"][j].append(roc_auc_score(lab[msk] == typ, v[msk]))
            wyn[k]["fa"].append(fa_epizody(al, lab, tol))
    return nazwy, wyn, fa_czyste

TYPY = {"S1": ["A poziom", "B zmienność", "C zależność"], "S2": ["A szok", "B zmiana φ", "C równowaga"],
        "S3": ["A faza", "B amplituda", "C okres"]}
if __name__ == "__main__":
    scen = sys.argv[1]; L = int(sys.argv[2]); R = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    nazwy, wyn, fa = uruchom(scen, L, R)
    t = TYPY[scen]
    print(f"\n=== {scen}, L = {L}, {R} szeregów z anomaliami + {R} bez; próg: ten sam poziom fałszywych alarmów dla każdej metody (20% czystych szeregów ma ≥1 fałszywy alarm) ===")
    print(f"{'metoda':34s} | " + " | ".join(f"{x:>13s}" for x in t) + " | szeregi z fałsz. alarmem")
    print(f"{'':34s} | " + " | ".join(f"{'wykr.  AUC':>13s}" for _ in t) + " |")
    for k in nazwy:
        r = wyn[k]
        print(f"{k:34s} | " + " | ".join(f"{np.nanmean(r['rec'][j])*100:4.0f}%  {np.mean(r['auc'][j]):.2f}" for j in range(3))
              + f" | {np.mean(np.array(r['fa']) > 0)*100:.0f}%")
