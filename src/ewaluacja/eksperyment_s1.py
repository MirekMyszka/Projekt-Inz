"""Eksperyment symulacyjny dla scenariusza S1 - biały szum (docs/plan_etap_1.md, rozdz. 5–6).

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.ewaluacja.eksperyment_s1                 # wszystkie części (ok. 1 h na 2 rdzeniach)
    python -m src.ewaluacja.eksperyment_s1 glowny          # tylko część główna
    python -m src.ewaluacja.eksperyment_s1 okno szum       # wybrane części
    python -m src.ewaluacja.eksperyment_s1 --szybko        # mała wersja do sprawdzenia, czy wszystko działa

Części:
  glowny - intensywność (4 poziomy) × długość anomalii L ∈ {20, 50, 100} × 100 szeregów (każdy z 3 anomaliami)
           × 6 metod (kNN, Isolation Forest i LOF w wariantach 5 i 3 cech) + Isolation Forest i LOF uczone na
           20 szeregach bez anomalii + Matrix Profile z normalizacją porównawczo; σ = 1, okno w = 50.
  okno   - wpływ długości okna w ∈ {20, 50, 100} na metody okienkowe (metoda nie zna prawdziwego L);
           poziomy 2 i 3, L ∈ {20, 50, 100}, 50 szeregów na konfigurację.
  szum   - wpływ poziomu szumu σ ∈ {0,5; 1; 2} przy STAŁEJ bezwzględnej wielkości anomalii
           (Δ = 1, c = 2, φ = 0,5; L = 50), 100 szeregów na σ.
  ablacja - analiza dodatkowa: dlaczego S1-C (zależność) jest trudna - CUSUM i kNN w wersji pełnej oraz patrzące
           TYLKO na zależność (karta zależności CUSUM, cecha autokorelacji); 4 poziomy, L = 50, 100 szeregów na poziom.

Procedura (dla każdej części i każdej konfiguracji metod):
  1. kalibracja na 200 szeregach BEZ anomalii: stałe metod (skale CUSUM, kara PELT) i próg alarmu każdej metody
     (wspólna zasada: fałszywy alarm w 10% szeregów bez anomalii - src/ewaluacja/metryki.py),
  2. walidacja progu na 200 NOWYCH szeregach bez anomalii (rzeczywisty odsetek fałszywych alarmów),
  3. ocena na szeregach z anomaliami.
Ziarna są stałe - ponowne uruchomienie daje identyczne wyniki.

Wyniki (CSV) w wyniki/s1/:
  <czesc>_anomalie.csv  - wiersz = metoda × szereg × anomalia (wykryta, AUC, AP)
  <czesc>_szeregi.csv   - wiersz = metoda × szereg (fałszywe alarmy, precyzja, czułość punktowa, F1)
  <czesc>_czyste.csv    - wiersz = metoda × szereg bez anomalii z walidacji (fałszywe alarmy)
  <czesc>_kalibracja.csv - progi i skalibrowane stałe metod
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.config import WYNIKI
from src.ewaluacja.metryki import ALFA_DOMYSLNA, fa_na_szeregu_czystym, kalibruj_prog, ocen_szereg
from src.metody import metody_s1
from src.symulacje.anomalie import POZIOMY_S1, RODZAJE_S1, szereg_s1

KATALOG = WYNIKI / "s1"
TOL = 25                     # strefa tolerancji (połowa domyślnego okna) - ta sama dla wszystkich metod i części
N_KAL, N_WAL = 200, 200      # szeregi bez anomalii: kalibracja / walidacja
ZIARNO_KAL, ZIARNO_WAL = 1_000_000, 2_000_000
N_UCZ, ZIARNO_UCZ = 20, 9_500_000   # szeregi uczące wariantów „uczony na szeregach bez anomalii” (osobne ziarna)
NAZWA_PARAMETRU = {1: "delta", 2: "c", 3: "phi"}

_PULA = None                 # jedna pula procesów na cały przebieg (biblioteka stumpy kompiluje się raz na proces)


def _wyniki_czyste(zadanie):
    metody, ziarno, sigma = zadanie
    x = szereg_s1(ziarno, sigma=sigma, z_anomaliami=False).x
    return ziarno, [m.wynik(x) for m in metody]


def _ocen(zadanie):
    """Zadanie = (metody, ziarno, poziom, L, sigma, progi, etykieta) → wiersze tabel anomalii i szeregów."""
    metody, ziarno, poziom, L, sigma, progi, etykieta = zadanie
    s = szereg_s1(ziarno, poziom=poziom, L=L, sigma=sigma)
    an_rows, sz_rows = [], []
    for m, prog in zip(metody, progi):
        r = ocen_szereg(m.wynik(s.x), s, prog, TOL)
        baza = {**etykieta, "metoda": m.nazwa, "grupa": m.grupa, "ziarno": ziarno, "L": L, "sigma": sigma}
        for a in s.anomalie:
            an_rows.append({**baza, "rodzaj": a.rodzaj, "rodzaj_nazwa": RODZAJE_S1[a.rodzaj],
                            "parametr": NAZWA_PARAMETRU[a.rodzaj], "wartosc": list(a.parametry.values())[0],
                            "start": a.start, "wykryta": r["wykryte"][a.rodzaj],
                            "auc": r["auc"][a.rodzaj], "ap": r["ap"][a.rodzaj]})
        sz_rows.append({**baza, "fa_epizody": r["fa_epizody"], "precyzja": r["precyzja"],
                        "czulosc_pkt": r["czulosc_pkt"], "f1": r["f1"]})
    return an_rows, sz_rows


def kalibruj(metody, sigma, n_kal):
    """Kalibracja stałych metod i progów na szeregach bez anomalii. Zwraca (progi, tabela kalibracji)."""
    if any(getattr(m, "nauka", "szereg") == "czyste" for m in metody):
        uczace = [szereg_s1(ZIARNO_UCZ + i, sigma=sigma, z_anomaliami=False).x for i in range(N_UCZ)]
        for m in metody:
            m.ucz(uczace)
    czyste = [szereg_s1(ZIARNO_KAL + i, sigma=sigma, z_anomaliami=False).x for i in range(n_kal)]
    for m in metody:
        m.kalibruj(czyste)                     # tanie (CUSUM, PELT) - w procesie głównym
    wyniki = dict(_PULA.map(_wyniki_czyste, [(metody, ZIARNO_KAL + i, sigma) for i in range(n_kal)], chunksize=4))
    progi = [kalibruj_prog([wyniki[ZIARNO_KAL + i][j] for i in range(n_kal)], ALFA_DOMYSLNA)
             for j in range(len(metody))]
    tab = pd.DataFrame({"metoda": [m.nazwa for m in metody], "prog": progi,
                        "kara_pelt": [getattr(m, "kara", np.nan) for m in metody],
                        "skale_cusum": [str(np.round(m.skale, 3).tolist()) if hasattr(m, "skale") else "" for m in metody]})
    return progi, tab


def waliduj(metody, progi, sigma, n_wal, etykieta):
    wyn = _PULA.map(_wyniki_czyste, [(metody, ZIARNO_WAL + i, sigma) for i in range(n_wal)], chunksize=4)
    return [{**etykieta, "metoda": m.nazwa, "ziarno": z, "sigma": sigma,
             "fa_epizody": fa_na_szeregu_czystym(w[j], progi[j])}
            for z, w in wyn for j, m in enumerate(metody)]


def uruchom_konfiguracje(metody, sigma, zadania_bez_progow, n_kal, n_wal, etykieta):
    t0 = time.time()
    progi, kal = kalibruj(metody, sigma, n_kal)
    kal = kal.assign(**etykieta, sigma=sigma)
    czyste = waliduj(metody, progi, sigma, n_wal, etykieta)
    zadania = [(metody, z, p, L, sigma, progi, etykieta) for z, p, L in zadania_bez_progow]
    an, sz = [], []
    for a, s in _PULA.imap_unordered(_ocen, zadania, chunksize=4):
        an += a
        sz += s
    print(f"  {etykieta} σ={sigma}: {len(zadania)} szeregów, {time.time() - t0:.0f} s", flush=True)
    return an, sz, czyste, kal


def zapisz(nazwa, an, sz, czyste, kal):
    KATALOG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(an).to_csv(KATALOG / f"{nazwa}_anomalie.csv", index=False)
    pd.DataFrame(sz).to_csv(KATALOG / f"{nazwa}_szeregi.csv", index=False)
    pd.DataFrame(czyste).to_csv(KATALOG / f"{nazwa}_czyste.csv", index=False)
    pd.concat(kal).to_csv(KATALOG / f"{nazwa}_kalibracja.csv", index=False)
    print(f"Zapisano {KATALOG.relative_to(WYNIKI.parent)}/{nazwa}_*.csv", flush=True)


def czesc_glowna(R, n_kal, n_wal):
    print("== Część główna: intensywność × L ==", flush=True)
    zad = [(100_000 * poziom + 10_000 * iL + r, poziom, L)
           for poziom in POZIOMY_S1 for iL, L in enumerate((20, 50, 100)) for r in range(R)]
    wyn = uruchom_konfiguracje(metody_s1(w=50, uczone=True), 1.0, zad, n_kal, n_wal, {"czesc": "glowny", "w": 50})
    zapisz("glowny", *wyn[:3], [wyn[3]])


def czesc_okno(R, n_kal, n_wal):
    print("== Część: długość okna w ==", flush=True)
    AN, SZ, CZ, KAL = [], [], [], []
    for w in (20, 50, 100):
        metody = [m for m in metody_s1(w=w, z_porownawczymi=False) if hasattr(m, "w")]
        zad = [(500_000 + 100_000 * poziom + 10_000 * iL + r, poziom, L)
               for poziom in (2, 3) for iL, L in enumerate((20, 50, 100)) for r in range(R)]
        an, sz, cz, kal = uruchom_konfiguracje(metody, 1.0, zad, n_kal, n_wal, {"czesc": "okno", "w": w})
        AN += an; SZ += sz; CZ += cz; KAL.append(kal)
    zapisz("okno", AN, SZ, CZ, KAL)


def czesc_szum(R, n_kal, n_wal):
    print("== Część: poziom szumu σ ==", flush=True)
    AN, SZ, CZ, KAL = [], [], [], []
    for sigma in (0.5, 1.0, 2.0):
        poziom = {"delta": 1.0 / sigma, "c": 2.0, "phi": 0.5}     # Δ = 1 w jednostkach bezwzględnych
        zad = [(900_000 + r, poziom, 50) for r in range(R)]
        an, sz, cz, kal = uruchom_konfiguracje(metody_s1(w=50), sigma, zad, n_kal, n_wal, {"czesc": "szum", "w": 50})
        AN += an; SZ += sz; CZ += cz; KAL.append(kal)
    zapisz("szum", AN, SZ, CZ, KAL)


def czesc_ablacja(R, n_kal, n_wal):
    print("== Analiza dodatkowa: ablacja cech dla S1-C ==", flush=True)
    from src.metody import CUSUM, CechyKNN
    metody = [CUSUM(), CUSUM(karty=(2,), nazwa="CUSUM: tylko karta zależności"),
              CechyKNN(), CechyKNN(cechy=3, nazwa="kNN: średnia, odch. std, autokorelacja"),
              CechyKNN(kolumny=[2], nazwa="kNN: tylko autokorelacja")]
    zad = [(1_300_000 + 100_000 * poziom + r, poziom, 50) for poziom in POZIOMY_S1 for r in range(R)]
    wyn = uruchom_konfiguracje(metody, 1.0, zad, n_kal, n_wal, {"czesc": "ablacja", "w": 50})
    zapisz("ablacja", *wyn[:3], [wyn[3]])


if __name__ == "__main__":
    arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    szybko = "--szybko" in sys.argv
    czesci = arg or ["glowny", "okno", "szum", "ablacja"]
    pula = max(1, os.cpu_count() or 1)
    R_gl, R_ok, R_sz, n_kal, n_wal = (5, 3, 5, 30, 30) if szybko else (100, 50, 100, N_KAL, N_WAL)
    if szybko:
        KATALOG = WYNIKI / "s1_szybki"
    t = time.time()
    _PULA = Pool(pula)
    for c in czesci:
        {"glowny": lambda: czesc_glowna(R_gl, n_kal, n_wal),
         "okno": lambda: czesc_okno(R_ok, n_kal, n_wal),
         "szum": lambda: czesc_szum(R_sz, n_kal, n_wal),
         "ablacja": lambda: czesc_ablacja(R_sz, n_kal, n_wal)}[c]()
    _PULA.close()
    print(f"Gotowe w {(time.time() - t) / 60:.1f} min")
