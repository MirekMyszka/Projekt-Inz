"""Eksperyment symulacyjny dla scenariusza S2 - AR(1) (docs/plan_etap_1.md, rozdz. 5–6).

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.ewaluacja.eksperyment_s2                 # wszystkie części (ok. 1 h na 2 rdzeniach)
    python -m src.ewaluacja.eksperyment_s2 glowny          # wybrane części: glowny, okno, tlo, dodatkowa, hybryda
    python -m src.ewaluacja.eksperyment_s2 --szybko        # mała wersja do sprawdzenia całego łańcucha

Części:
  glowny    - intensywność (4 poziomy) × długość L ∈ {20, 50, 100} × 100 szeregów (każdy z 3 anomaliami)
              × 6 metod (kNN, Isolation Forest i One-Class SVM w wariantach 5 i 3 cech) + Isolation Forest
              i One-Class SVM uczone na 20 szeregach bez anomalii; φ tła = 0,7, σ_x = 1, okno w = 50. Szok (S2-A) ma zawsze długość 10 (czas wygaśnięcia).
  okno      - długość okna w ∈ {20, 50, 100} dla metod okienkowych; poziomy 2 i 3, wszystkie L, 50 szeregów.
  tlo       - WŁAŚCIWOŚCI SZEREGU: siła zależności tła φ ∈ {0,3; 0,7; 0,9} przy stałych anomaliach
              (szok 5σ_ε, φ' = 0, Δ = 2σ_x; L = 50), 100 szeregów na φ.
  dodatkowa - analizy dodatkowe (L = 50, 100 szeregów na konfigurację):
              (a) metody z S1 zastosowane wprost do S2, bez modelu AR - czy modelowanie zależności jest potrzebne,
              (b) wzrost zależności φ' = 0,95 zamiast spadku (poziomy 2 i 3 dla A i C),
              (c) One-Class SVM (5 cech): szerokość jądra gamma = „scale” / 0,01 / 0,002.
  hybryda   - analiza dodatkowa (d), zaproponowana PO wynikach (a): CUSUM mieszany (poziom na x, zmienność
              i zależność na resztach AR) vs „reszty AR + CUSUM” vs CUSUM z S1; poziomy 1–4, L = 50, 100 szeregów.
Wyniki (CSV): wyniki/s2/<czesc>_{anomalie,szeregi,czyste,kalibracja}.csv
"""
import os
import sys
import time

from src.config import WYNIKI
from src.ewaluacja.eksperyment import Eksperyment
from src.metody import CUSUM, CechyKNN, CUSUMHybryda, MatrixProfile, OneClassSVM, ResztyARCUSUM, metody_s2
from src.symulacje.anomalie import POZIOMY_S2, RODZAJE_S2, szereg_s2

SCEN = {"generuj": szereg_s2, "rodzaje": RODZAJE_S2, "parametr": {1: "impuls", 2: "phi_prim", 3: "delta"}}
LS = (20, 50, 100)


def czesc_glowna(e, R):
    zad = [(3_100_000 + 100_000 * p + 10_000 * i + r, p, L) for p in POZIOMY_S2 for i, L in enumerate(LS) for r in range(R)]
    e.czesc("glowny", [(metody_s2(w=50, uczone=True), zad, {}, {"w": 50, "phi_tla": 0.7})])


def czesc_okno(e, R):
    konf = []
    for w in (20, 50, 100):
        metody = [m for m in metody_s2(w=w) if hasattr(m, "w")]
        zad = [(3_600_000 + 100_000 * p + 10_000 * i + r, p, L) for p in (2, 3) for i, L in enumerate(LS) for r in range(R)]
        konf.append((metody, zad, {}, {"w": w, "phi_tla": 0.7}))
    e.czesc("okno", konf)


def czesc_tlo(e, R):
    poziom = {"impuls": 5.0, "phi_prim": 0.0, "delta": 2.0}
    konf = [(metody_s2(w=50), [(4_100_000 + r, poziom, 50) for r in range(R)], {"phi": phi}, {"w": 50, "phi_tla": phi})
            for phi in (0.3, 0.7, 0.9)]
    e.czesc("tlo", konf)


def czesc_dodatkowa(e, R):
    zad23 = [(4_600_000 + 100_000 * p + r, p, 50) for p in (2, 3) for r in range(R)]
    metody_s1_wprost = [CUSUM(nazwa="CUSUM z S1 (bez modelu AR)"),
                        MatrixProfile(w=50),
                        CechyKNN(w=50, zestaw="s1", nazwa="kNN z cechami S1 (bez modelu AR)")]
    wzrost = [(4_900_000 + 100_000 * p + r, {**POZIOMY_S2[p], "phi_prim": 0.95, "_poziom": p}, 50)
              for p in (2, 3) for r in range(R)]
    gamma = [OneClassSVM(gamma="scale", nazwa="One-Class SVM gamma=scale"),
             OneClassSVM(gamma=0.01, nazwa="One-Class SVM gamma=0,01"),
             OneClassSVM(gamma=0.002, nazwa="One-Class SVM gamma=0,002")]
    e.czesc("dodatkowa", [
        (metody_s1_wprost, zad23, {}, {"w": 50, "phi_tla": 0.7, "analiza": "a_metody_S1_wprost"}),
        (metody_s2(w=50), wzrost, {}, {"w": 50, "phi_tla": 0.7, "analiza": "b_wzrost_phi_0.95"}),
        (gamma, zad23, {}, {"w": 50, "phi_tla": 0.7, "analiza": "c_gamma_OCSVM"}),
    ])


def czesc_hybryda(e, R):
    zad = [(5_300_000 + 100_000 * p + r, p, 50) for p in POZIOMY_S2 for r in range(R)]
    e.czesc("hybryda", [([ResztyARCUSUM(), CUSUM(nazwa="CUSUM z S1 (bez modelu AR)"), CUSUMHybryda()], zad, {},
                         {"w": 50, "phi_tla": 0.7})])


if __name__ == "__main__":
    arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    szybko = "--szybko" in sys.argv
    czesci = arg or ["glowny", "okno", "tlo", "dodatkowa", "hybryda"]
    R = {"glowny": 100, "okno": 50, "tlo": 100, "dodatkowa": 100, "hybryda": 100}
    n_kal = n_wal = 200
    katalog = WYNIKI / "s2"
    if szybko:
        R = {k: 4 for k in R}
        n_kal = n_wal = 30
        katalog = WYNIKI / "s2_szybki"
    t = time.time()
    e = Eksperyment(SCEN, katalog, max(1, os.cpu_count() or 1), n_kal, n_wal)
    for c in czesci:
        print(f"== Część: {c} ==", flush=True)
        {"glowny": czesc_glowna, "okno": czesc_okno, "tlo": czesc_tlo, "dodatkowa": czesc_dodatkowa,
         "hybryda": czesc_hybryda}[c](e, R[c])
    e.zamknij()
    print(f"Gotowe w {(time.time() - t) / 60:.1f} min")
