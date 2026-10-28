"""Eksperyment symulacyjny dla scenariusza S3 - biały szum + sinus (docs/plan_etap_1.md, rozdz. 5–6).

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.ewaluacja.eksperyment_s3                 # wszystkie części
    python -m src.ewaluacja.eksperyment_s3 glowny          # wybrane części: glowny, okno, tlo, dodatkowa
    python -m src.ewaluacja.eksperyment_s3 --szybko        # mała wersja do sprawdzenia całego łańcucha

Części:
  glowny    - intensywność (4 poziomy) × długość L ∈ {50, 100, 200} (1, 2, 4 cykle) × 100 szeregów (każdy z 3 anomaliami)
              × 6 metod (kNN i Isolation Forest w wariantach 5 i 3 cech) + Isolation Forest i autoenkoder uczone
              na 20 szeregach bez anomalii + Matrix Profile bez normalizacji (porównawczo); A = 3, P = 50, σ = 1, okno w = P = 50.
  okno      - długość okna w ∈ {25, 50, 100} (pół cyklu, cykl, dwa cykle) dla metod okienkowych; poziomy 2 i 3,
              wszystkie L, 50 szeregów.
  tlo       - WŁAŚCIWOŚCI SZEREGU: amplituda cyklu A ∈ {1, 3, 5} przy σ = 1 (stosunek sygnału do szumu) i stałych
              anomaliach (poziom 3: faza +π/2, amplituda ×0,25, okres ×0,6; L = 100), 100 szeregów na A.
  dodatkowa - analizy dodatkowe (poziomy 2 i 3):
              (a) model cyklu: STL z domyślnym oknem sezonowości (7 cykli) vs STL z oknem dłuższym niż szereg (wersja
                  główna) vs sinus dopasowany globalnie - czy model cyklu „wchłania” anomalię; L ∈ {50, 100, 200},
                  50 szeregów na kombinację,
              (b) odwrotny kierunek anomalii: faza −Δθ, wzmocnienie cyklu (×1,5; ×2), spowolnienie (okres ×1,25; ×1,5);
                  L = 100, 100 szeregów na poziom,
              (c) autoenkoder: nauka na tym samym szeregu bez odrzucania / z odrzucaniem 10% okien (wersja główna) /
                  na szeregach bez anomalii; L = 100, 100 szeregów na poziom.
Wyniki (CSV): wyniki/s3/<czesc>_{anomalie,szeregi,czyste,kalibracja}.csv
"""
import os
import sys
import time

from src.config import WYNIKI
from src.ewaluacja.eksperyment import Eksperyment
from src.metody import PELTSTL, STLEWMA, Autoenkoder, metody_s3
from src.symulacje.anomalie import LS_S3, POZIOMY_S3, RODZAJE_S3, szereg_s3

SCEN = {"generuj": szereg_s3, "rodzaje": RODZAJE_S3, "parametr": {1: "faza_pi", 2: "k", 3: "r"}}


def czesc_glowna(e, R):
    zad = [(6_100_000 + 100_000 * p + 10_000 * i + r, p, L) for p in POZIOMY_S3 for i, L in enumerate(LS_S3)
           for r in range(R)]
    e.czesc("glowny", [(metody_s3(w=50, uczone=True), zad, {}, {"w": 50, "A": 3.0})])


def czesc_okno(e, R):
    konf = []
    for w in (25, 50, 100):
        metody = [m for m in metody_s3(w=w) if hasattr(m, "w")]
        zad = [(6_600_000 + 100_000 * p + 10_000 * i + r, p, L) for p in (2, 3) for i, L in enumerate(LS_S3)
               for r in range(R)]
        konf.append((metody, zad, {}, {"w": w, "A": 3.0}))
    e.czesc("okno", konf)


def czesc_tlo(e, R):
    konf = [(metody_s3(w=50), [(7_100_000 + r, 3, 100) for r in range(R)], {"A": A}, {"w": 50, "A": A})
            for A in (1.0, 3.0, 5.0)]
    e.czesc("tlo", konf)


def czesc_dodatkowa(e, R):
    modele = [STLEWMA(sezon=7), STLEWMA(), STLEWMA(model="harmoniczny"),
              PELTSTL(sezon=7), PELTSTL(), PELTSTL(model="harmoniczny")]
    zad_a = [(7_600_000 + 100_000 * p + 10_000 * i + r, p, L) for p in (2, 3) for i, L in enumerate(LS_S3)
             for r in range(max(1, R // 2))]
    odwrotne = {2: {"faza_pi": -0.25, "k": 1.5, "r": 1.25, "_poziom": 2},
                3: {"faza_pi": -0.5, "k": 2.0, "r": 1.5, "_poziom": 3}}
    zad_b = [(8_100_000 + 100_000 * p + r, odwrotne[p], 100) for p in (2, 3) for r in range(R)]
    ae = [Autoenkoder(odrzuc=0.0, nazwa="autoenkoder (bez odrzucania okien)"),
          Autoenkoder(nazwa="autoenkoder (odrzucanie 10% okien)"),
          Autoenkoder(nauka="czyste", nazwa="autoenkoder (uczony na szeregach bez anomalii)")]
    zad_c = [(8_600_000 + 100_000 * p + r, p, 100) for p in (2, 3) for r in range(R)]
    e.czesc("dodatkowa", [
        (modele, zad_a, {}, {"w": 50, "A": 3.0, "analiza": "a_model_cyklu"}),
        (metody_s3(w=50), zad_b, {}, {"w": 50, "A": 3.0, "analiza": "b_odwrotny_kierunek"}),
        (ae, zad_c, {}, {"w": 50, "A": 3.0, "analiza": "c_autoenkoder"}),
    ])


if __name__ == "__main__":
    arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    szybko = "--szybko" in sys.argv
    czesci = arg or ["glowny", "okno", "tlo", "dodatkowa"]
    R = {"glowny": 100, "okno": 50, "tlo": 100, "dodatkowa": 100}
    n_kal = n_wal = 200
    katalog = WYNIKI / "s3"
    if szybko:
        R = {k: 4 for k in R}
        n_kal = n_wal = 30
        katalog = WYNIKI / "s3_szybki"
    t = time.time()
    e = Eksperyment(SCEN, katalog, max(1, os.cpu_count() or 1), n_kal, n_wal)
    for c in czesci:
        print(f"== Część: {c} ==", flush=True)
        {"glowny": czesc_glowna, "okno": czesc_okno, "tlo": czesc_tlo, "dodatkowa": czesc_dodatkowa}[c](e, R[c])
    e.zamknij()
    print(f"Gotowe w {(time.time() - t) / 60:.1f} min")
