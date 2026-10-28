"""Wstawianie anomalii zbiorowych i tworzenie kompletnych szeregów testowych (docs/plan_etap_1.md, rozdz. 3).

Scenariusze: S1 - biały szum (szereg_s1), S2 - AR(1) (szereg_s2), S3 - szum + sinus (szereg_s3).

Scenariusz S1 - biały szum - ma trzy własne rodzaje anomalii:
    S1-A  krótkotrwała zmiana poziomu       x[fragment] += Δ·σ
    S1-B  krótkotrwała zmiana zmienności    x[fragment] *= c
    S1-C  pojawienie się zależności         x[fragment] = AR(1) o współczynniku φ i TEJ SAMEJ wariancji σ²

W każdym szeregu są wszystkie trzy anomalie (po jednej z każdego rodzaju). Szereg dzielimy na 3 równe części;
w każdej części jedna anomalia w losowym miejscu, kolejność rodzajów losowa, co najmniej `MARGINES` punktów
odstępu od brzegów części (a więc i od sąsiednich anomalii oraz od początku/końca szeregu).

Etykiety: tablica int długości n - 0 = norma, 1 = A, 2 = B, 3 = C.
"""
from dataclasses import dataclass, field

import numpy as np

from src.symulacje.generator import N_DOMYSLNE, ar1, ar1_zmienne, bialy_szum, sinus_zmienny

MARGINES = 100
RODZAJE_S1 = {1: "S1-A zmiana poziomu", 2: "S1-B zmiana zmienności", 3: "S1-C pojawienie się zależności"}

# Poziomy intensywności 1 (najsłabszy) ... 4 (najsilniejszy) - jeden wspólny „suwak” dla trzech rodzajów.
# Δ i c zgodnie z planem; dla φ plan podaje {0,5; 0,7; 0,9} - dodano 0,3 jako najsłabszy poziom, żeby każdy
# rodzaj miał 4 poziomy.
POZIOMY_S1 = {
    1: {"delta": 0.5, "c": 1.5, "phi": 0.3},
    2: {"delta": 1.0, "c": 2.0, "phi": 0.5},
    3: {"delta": 2.0, "c": 3.0, "phi": 0.7},
    4: {"delta": 3.0, "c": 4.0, "phi": 0.9},
}


@dataclass
class Anomalia:
    rodzaj: int          # 1 = A, 2 = B, 3 = C
    start: int           # pierwszy indeks fragmentu
    koniec: int          # indeks ZA ostatnim punktem (fragment = x[start:koniec])
    parametry: dict = field(default_factory=dict)

    @property
    def dlugosc(self) -> int:
        return self.koniec - self.start


@dataclass
class SzeregTestowy:
    x: np.ndarray
    etykiety: np.ndarray
    anomalie: list
    scenariusz: str
    parametry: dict
    ziarno: int


def rozmiesc(n: int, L: int | list, rng: np.random.Generator, k: int = 3, margines: int = MARGINES) -> list[int]:
    """Początki k fragmentów: po jednym w każdej z k równych części szeregu, z marginesem.
    L - wspólna długość albo lista długości kolejnych fragmentów."""
    czesc = n // k
    dl = [L] * k if isinstance(L, int) else list(L)
    if max(dl) + 2 * margines > czesc:
        raise ValueError(f"L = {max(dl)} za długie: część szeregu ma {czesc} punktów, a potrzeba L + 2·{margines}")
    return [i * czesc + int(rng.integers(margines, czesc - margines - d + 1)) for i, d in enumerate(dl)]


def wstaw_s1(x: np.ndarray, rodzaj: int, start: int, L: int, poziom: dict, sigma: float,
             rng: np.random.Generator) -> dict:
    """Wstawia anomalię S1 danego rodzaju do x (w miejscu) i zwraca jej parametry."""
    sl = slice(start, start + L)
    if rodzaj == 1:
        x[sl] += poziom["delta"] * sigma
        return {"delta": poziom["delta"]}
    if rodzaj == 2:
        x[sl] *= poziom["c"]
        return {"c": poziom["c"]}
    if rodzaj == 3:
        x[sl] = ar1(L, poziom["phi"], sigma, rng)
        return {"phi": poziom["phi"]}
    raise ValueError(f"Nieznany rodzaj anomalii S1: {rodzaj}")


def szereg_s1(ziarno: int, poziom: int | dict = 2, L: int = 50, n: int = N_DOMYSLNE, sigma: float = 1.0,
              z_anomaliami: bool = True) -> SzeregTestowy:
    """Kompletny szereg testowy S1: biały szum + (opcjonalnie) trzy anomalie S1-A, S1-B, S1-C.

    ziarno       - całkowicie wyznacza szereg (te same argumenty + ziarno = ten sam szereg),
    poziom       - poziom intensywności 1..4 (POZIOMY_S1) albo własny słownik {"delta", "c", "phi"}
                   (Δ w jednostkach σ),
    L            - długość każdej anomalii,
    sigma        - odchylenie standardowe szumu (poziom szumu),
    z_anomaliami - False daje szereg bez anomalii (do kalibracji progów i pomiaru fałszywych alarmów).
    """
    par_poziomu = POZIOMY_S1[poziom] if isinstance(poziom, int) else poziom
    rng = np.random.default_rng(ziarno)
    x = bialy_szum(n, sigma, rng)
    etykiety = np.zeros(n, dtype=int)
    anomalie = []
    if z_anomaliami:
        starty = rozmiesc(n, L, rng)
        rodzaje = rng.permutation([1, 2, 3])
        for rodzaj, start in zip(rodzaje, starty):
            par = wstaw_s1(x, int(rodzaj), start, L, par_poziomu, sigma, rng)
            etykiety[start:start + L] = rodzaj
            anomalie.append(Anomalia(int(rodzaj), start, start + L, par))
    return SzeregTestowy(x, etykiety, anomalie, "S1",
                         {"poziom": poziom, "L": L, "n": n, "sigma": sigma, "z_anomaliami": z_anomaliami}, ziarno)


# ============================================================================================ S2 - AR(1)
RODZAJE_S2 = {1: "S2-A szok wygasający", 2: "S2-B zmiana siły zależności", 3: "S2-C przesunięcie poziomu równowagi"}
PHI_S2 = 0.7          # zależność tła (plan_etap_1.md, rozdz. 2)

# Poziomy intensywności 1..4. Względem planu ({3; 5; 8}, {0; 0,3; 0,95}, {1; 2; 3}) dodano po jednym słabszym poziomie,
# a zmianę φ uporządkowano jako SŁABNIĘCIE zależności (0,5 → 0,3 → 0 → −0,5 = odwrócenie), żeby intensywność rosła
# monotonicznie. Wzrost zależności (φ' = 0,95) sprawdzany osobno (część dodatkowa eksperymentu S2).
POZIOMY_S2 = {
    1: {"impuls": 2.0, "phi_prim": 0.5, "delta": 0.5},
    2: {"impuls": 3.0, "phi_prim": 0.3, "delta": 1.0},
    3: {"impuls": 5.0, "phi_prim": 0.0, "delta": 2.0},
    4: {"impuls": 8.0, "phi_prim": -0.5, "delta": 3.0},
}


def dlugosc_szoku(phi: float, prog: float = 0.05) -> int:
    """Liczba punktów, po której ślad szoku spada poniżej `prog` (5%) jego wielkości: φ^k < prog."""
    return int(np.ceil(np.log(prog) / np.log(abs(phi)))) + 1


def szereg_s2(ziarno: int, poziom: int | dict = 2, L: int = 50, n: int = N_DOMYSLNE, phi: float = PHI_S2,
              sigma_x: float = 1.0, z_anomaliami: bool = True) -> SzeregTestowy:
    """Kompletny szereg testowy S2: AR(1) z φ = 0,7 (wariancja szeregu σ_x² = 1) + trzy anomalie:

    S2-A  szok innowacyjny: do jednej innowacji dodajemy impuls·σ_ε; szereg wraca do normy zgodnie z dynamiką AR
          (ślad maleje jak φ^k). Etykieta obejmuje czas wygaśnięcia do 5% (dla φ = 0,7: 10 punktów) - długość tej
          anomalii wynika z dynamiki szeregu, a nie z L.
    S2-B  zmiana siły zależności: na fragmencie długości L φ → φ', przy czym σ_ε dobrane tak, żeby wariancja szeregu
          σ_x² się NIE zmieniła (σ_ε' = σ_x·√(1 − φ'²)).
    S2-C  przesunięcie poziomu równowagi: na fragmencie długości L μ → Δ·σ_x; szereg dochodzi do nowego poziomu
          stopniowo (w tempie φ) i po fragmencie stopniowo wraca.
    """
    par = POZIOMY_S2[poziom] if isinstance(poziom, int) else poziom
    rng = np.random.default_rng(ziarno)
    sigma_e = sigma_x * np.sqrt(1 - phi ** 2)
    phi_t, mu_t, se_t, imp = np.full(n, phi), np.zeros(n), np.full(n, sigma_e), np.zeros(n)
    etykiety = np.zeros(n, dtype=int)
    anomalie = []
    if z_anomaliami:
        rodzaje = [int(r) for r in rng.permutation([1, 2, 3])]
        L_szok = dlugosc_szoku(phi)
        dl = [L_szok if r == 1 else L for r in rodzaje]
        for rodzaj, start, d in zip(rodzaje, rozmiesc(n, dl, rng), dl):
            if rodzaj == 1:
                imp[start] = par["impuls"] * sigma_e
                p = {"impuls": par["impuls"]}
            elif rodzaj == 2:
                phi_t[start:start + d] = par["phi_prim"]
                se_t[start:start + d] = sigma_x * np.sqrt(1 - par["phi_prim"] ** 2)
                p = {"phi_prim": par["phi_prim"]}
            else:
                mu_t[start:start + d] = par["delta"] * sigma_x
                p = {"delta": par["delta"]}
            etykiety[start:start + d] = rodzaj
            anomalie.append(Anomalia(rodzaj, start, start + d, p))
    x = ar1_zmienne(n, phi_t, mu_t, se_t, rng, imp)
    return SzeregTestowy(x, etykiety, anomalie, "S2",
                         {"poziom": poziom, "L": L, "n": n, "phi": phi, "sigma_x": sigma_x, "z_anomaliami": z_anomaliami},
                         ziarno)


# ============================================================================================ S3 - szum + sinus
RODZAJE_S3 = {1: "S3-A przesunięcie fazy", 2: "S3-B zmiana amplitudy", 3: "S3-C zmiana okresu"}
A_S3, P_S3 = 3.0, 50          # amplituda i okres cyklu tła (plan_etap_1.md, rozdz. 2)
LS_S3 = (50, 100, 200)        # długości anomalii = 1, 2 i 4 cykle (zmiana po pilocie, plan rozdz. 8)

# Poziomy intensywności 1..4. Względem planu ({π/4; π/2; π}, {0; 0,5; 1,5; 2}, {0,5; 0,75; 1,5}):
#   - faza: dodano najsłabszy poziom π/8,
#   - amplituda: uporządkowana jako SŁABNIĘCIE cyklu (×0,75 → ×0,5 → ×0,25 → zanik ×0), żeby intensywność rosła
#     monotonicznie; wzmocnienie (×1,5; ×2) sprawdzane osobno (część dodatkowa),
#   - okres: uporządkowany jako PRZYSPIESZENIE cyklu (×0,9 → ×0,75 → ×0,6 → ×0,5); spowolnienie (×1,25; ×1,5) osobno.
POZIOMY_S3 = {
    1: {"faza_pi": 0.125, "k": 0.75, "r": 0.9},
    2: {"faza_pi": 0.25, "k": 0.5, "r": 0.75},
    3: {"faza_pi": 0.5, "k": 0.25, "r": 0.6},
    4: {"faza_pi": 1.0, "k": 0.0, "r": 0.5},
}


def szereg_s3(ziarno: int, poziom: int | dict = 2, L: int = 100, n: int = N_DOMYSLNE, A: float = A_S3,
              P: int = P_S3, sigma: float = 1.0, z_anomaliami: bool = True) -> SzeregTestowy:
    """Kompletny szereg testowy S3: x_t = a_t·sin(ψ_t) + ε_t, w normie a_t = A, ψ_t = 2πt/P + θ, ε ~ N(0, σ²).
    Faza początkowa θ losowana z [0, 2π) (metody nie mogą zakładać θ = 0). Trzy anomalie okresowości:

    S3-A  przesunięcie fazy: na fragmencie ψ_t → ψ_t + Δθ (cykl „spóźniony”); po fragmencie cykl wraca na swoje
          miejsce - na obu końcach fragmentu przebieg ma skok.
    S3-B  zmiana amplitudy: na fragmencie a_t = k·A (słabnięcie / zanik cyklu); faza i okres bez zmian.
    S3-C  zmiana okresu: od początku fragmentu cykl biegnie z okresem r·P - faza CIĄGŁA na początku
          (ψ_t = ψ_start + 2π(t − start)/(r·P)); po fragmencie cykl wraca do normalnej fazy (skok na końcu - inaczej
          cały dalszy szereg byłby trwale przesunięty, a to już nie anomalia zbiorowa, tylko punkt zmiany).
    """
    par = POZIOMY_S3[poziom] if isinstance(poziom, int) else poziom
    rng = np.random.default_rng(ziarno)
    theta = float(rng.uniform(0, 2 * np.pi))
    t = np.arange(n)
    psi = 2 * np.pi * t / P + theta
    amp = np.full(n, float(A))
    etykiety = np.zeros(n, dtype=int)
    anomalie = []
    if z_anomaliami:
        rodzaje = [int(r) for r in rng.permutation([1, 2, 3])]
        for rodzaj, start in zip(rodzaje, rozmiesc(n, L, rng)):
            sl = slice(start, start + L)
            if rodzaj == 1:
                psi[sl] += par["faza_pi"] * np.pi
                p = {"faza_pi": par["faza_pi"]}
            elif rodzaj == 2:
                amp[sl] = par["k"] * A
                p = {"k": par["k"]}
            else:
                psi[sl] = psi[start] + 2 * np.pi * np.arange(L) / (par["r"] * P)
                p = {"r": par["r"]}
            etykiety[sl] = rodzaj
            anomalie.append(Anomalia(rodzaj, start, start + L, p))
    x = sinus_zmienny(amp, psi, sigma, rng)
    return SzeregTestowy(x, etykiety, anomalie, "S3",
                         {"poziom": poziom, "L": L, "n": n, "A": A, "P": P, "sigma": sigma, "theta": theta,
                          "z_anomaliami": z_anomaliami}, ziarno)
