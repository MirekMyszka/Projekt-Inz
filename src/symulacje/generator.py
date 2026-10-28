"""Generatory szeregów tła (scenariusze z docs/plan_etap_1.md, rozdz. 2).

S1 - biały szum, S2 - AR(1) (także o parametrach zmiennych w czasie), S3 - szum + sinus.
Każda funkcja dostaje generator liczb losowych `rng` (numpy.random.Generator), dzięki czemu każdy szereg
da się odtworzyć z zapisanego ziarna.
"""
import numpy as np

N_DOMYSLNE = 3000  # długość szeregu (plan_etap_1.md, rozdz. 2)


def bialy_szum(n: int = N_DOMYSLNE, sigma: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """S1: x_t = ε_t, ε_t ~ N(0, σ²) niezależne."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(0.0, sigma, n)


def ar1(n: int, phi: float, sigma_x: float, rng: np.random.Generator, x0: float | None = None) -> np.ndarray:
    """Stacjonarny AR(1) o zadanej wariancji szeregu σ_x²: x_t = φ·x_{t−1} + e_t, e_t ~ N(0, σ_x²(1 − φ²)).

    Punkt startowy losowany z rozkładu stacjonarnego N(0, σ_x²), więc cały fragment ma od początku
    tę samą wariancję co szereg tła. Używane m.in. do anomalii S1-C (pojawienie się zależności).
    """
    if not -1 < phi < 1:
        raise ValueError("|φ| musi być < 1 (proces stacjonarny)")
    e = rng.normal(0.0, sigma_x * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma_x) if x0 is None else x0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def ar1_zmienne(n: int, phi: np.ndarray, mu: np.ndarray, sigma_e: np.ndarray, rng: np.random.Generator,
                impuls: np.ndarray | None = None) -> np.ndarray:
    """AR(1) o parametrach zmiennych w czasie (S2):

        x_t = μ_t + φ_t·(x_{t−1} − μ_t) + σ_{ε,t}·z_t + impuls_t,     z_t ~ N(0, 1)

    Przy stałych parametrach to zwykły AR(1) z poziomem równowagi μ. Zmiana μ_t na fragmencie powoduje, że szereg
    STOPNIOWO dochodzi do nowego poziomu (geometrycznie, w tempie φ) i stopniowo wraca. Punkt startowy - z rozkładu
    stacjonarnego dla parametrów z t = 0.
    """
    z = rng.standard_normal(n)
    imp = np.zeros(n) if impuls is None else impuls
    x = np.empty(n)
    var0 = sigma_e[0] ** 2 / (1 - phi[0] ** 2)
    prev = mu[0] + np.sqrt(var0) * rng.standard_normal()
    for t in range(n):
        prev = mu[t] + phi[t] * (prev - mu[t]) + sigma_e[t] * z[t] + imp[t]
        x[t] = prev
    return x


def sinus_zmienny(amplituda: np.ndarray, faza: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """S3: x_t = a_t·sin(ψ_t) + ε_t, ε_t ~ N(0, σ²) - cykl o amplitudzie a_t i fazie ψ_t zadanych dla każdego punktu.
    W normie a_t = A i ψ_t = 2πt/P + θ (zwykły sinus); anomalie S3 zmieniają a_t lub ψ_t na fragmencie."""
    return amplituda * np.sin(faza) + rng.normal(0.0, sigma, len(faza))
