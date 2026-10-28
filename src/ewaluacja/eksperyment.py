"""Uniwersalny silnik eksperymentu symulacyjnego - ta sama procedura dla każdego scenariusza (S2, S3, ...).

(S1 ma własny, wcześniejszy plik src/ewaluacja/eksperyment_s1.py z identyczną procedurą - zostawiony bez zmian,
żeby jego wyniki dało się odtworzyć bit w bit.)

Procedura dla jednej konfiguracji (zestaw metod + parametry szeregu tła):
  0. (tylko warianty ML uczone na szeregach bez anomalii) uczenie modelu na N_UCZ OSOBNYCH szeregach bez anomalii,
  1. kalibracja na N_KAL szeregach BEZ anomalii: stałe metod (np. skale kart CUSUM, kara PELT) i próg alarmu każdej
     metody - wspólna zasada: fałszywy alarm w 10% szeregów bez anomalii (src/ewaluacja/metryki.py),
  2. walidacja progu na N_WAL NOWYCH szeregach bez anomalii (rzeczywisty odsetek fałszywych alarmów),
  3. ocena na szeregach z anomaliami.
Ziarna są stałe - ponowne uruchomienie daje identyczne wyniki.

Scenariusz opisuje słownik:
  {"generuj": funkcja(ziarno, poziom, L, z_anomaliami, **kw) -> SzeregTestowy,
   "rodzaje": {1: "...", 2: "...", 3: "..."}, "parametr": {1: "...", 2: "...", 3: "..."}}
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.ewaluacja.metryki import ALFA_DOMYSLNA, fa_na_szeregu_czystym, kalibruj_prog, ocen_szereg

TOL = 25                      # strefa tolerancji (połowa domyślnego okna) - ta sama dla wszystkich metod
N_KAL, N_WAL = 200, 200
ZIARNO_KAL, ZIARNO_WAL = 1_000_000, 2_000_000
N_UCZ, ZIARNO_UCZ = 20, 9_500_000   # szeregi uczące wariantów „uczony na szeregach bez anomalii” (osobne ziarna)


def _wyniki_czyste(zadanie):
    scen, metody, ziarno, kw = zadanie
    x = scen["generuj"](ziarno, z_anomaliami=False, **kw).x
    return ziarno, [m.wynik(x) for m in metody]


def _ocen(zadanie):
    scen, metody, ziarno, poziom, L, kw, progi, etykieta = zadanie
    s = scen["generuj"](ziarno, poziom=poziom, L=L, **kw)
    an_rows, sz_rows = [], []
    for m, prog in zip(metody, progi):
        r = ocen_szereg(m.wynik(s.x), s, prog, TOL)
        baza = {**etykieta, "metoda": m.nazwa, "grupa": m.grupa, "ziarno": ziarno, "L": L,
                "poziom": poziom if isinstance(poziom, int) else poziom.get("_poziom", 0)}
        for a in s.anomalie:
            an_rows.append({**baza, "rodzaj": a.rodzaj, "rodzaj_nazwa": scen["rodzaje"][a.rodzaj],
                            "parametr": scen["parametr"][a.rodzaj], "wartosc": list(a.parametry.values())[0],
                            "start": a.start, "dlugosc": a.dlugosc, "wykryta": r["wykryte"][a.rodzaj],
                            "auc": r["auc"][a.rodzaj], "ap": r["ap"][a.rodzaj]})
        sz_rows.append({**baza, "fa_epizody": r["fa_epizody"], "precyzja": r["precyzja"],
                        "czulosc_pkt": r["czulosc_pkt"], "f1": r["f1"]})
    return an_rows, sz_rows


class Eksperyment:
    """Jedna pula procesów na cały przebieg (biblioteka stumpy kompiluje się raz na proces)."""

    def __init__(self, scen: dict, katalog, procesy: int, n_kal: int = N_KAL, n_wal: int = N_WAL):
        self.scen, self.katalog, self.n_kal, self.n_wal = scen, katalog, n_kal, n_wal
        self.pula = Pool(procesy)

    def zamknij(self):
        self.pula.close()

    def kalibruj(self, metody, kw):
        if any(getattr(m, "nauka", "szereg") == "czyste" for m in metody):
            uczace = [self.scen["generuj"](ZIARNO_UCZ + i, z_anomaliami=False, **kw).x for i in range(N_UCZ)]
            for m in metody:
                m.ucz(uczace)
        czyste = [self.scen["generuj"](ZIARNO_KAL + i, z_anomaliami=False, **kw).x for i in range(self.n_kal)]
        for m in metody:
            m.kalibruj(czyste)                 # tanie kalibracje stałych - w procesie głównym
        wyn = dict(self.pula.map(_wyniki_czyste, [(self.scen, metody, ZIARNO_KAL + i, kw) for i in range(self.n_kal)],
                                 chunksize=4))
        progi = [kalibruj_prog([wyn[ZIARNO_KAL + i][j] for i in range(self.n_kal)], ALFA_DOMYSLNA)
                 for j in range(len(metody))]
        tab = pd.DataFrame({"metoda": [m.nazwa for m in metody], "prog": progi,
                            "kara_pelt": [getattr(m, "kara", np.nan) for m in metody]})
        return progi, tab

    def waliduj(self, metody, progi, kw, etykieta):
        wyn = self.pula.map(_wyniki_czyste, [(self.scen, metody, ZIARNO_WAL + i, kw) for i in range(self.n_wal)],
                            chunksize=4)
        return [{**etykieta, "metoda": m.nazwa, "ziarno": z, "fa_epizody": fa_na_szeregu_czystym(w[j], progi[j])}
                for z, w in wyn for j, m in enumerate(metody)]

    def konfiguracja(self, metody, zadania, kw, etykieta):
        """zadania: lista (ziarno, poziom, L); kw: parametry szeregu tła; etykieta: kolumny opisujące część."""
        t0 = time.time()
        progi, kal = self.kalibruj(metody, kw)
        kal = kal.assign(**etykieta)
        czyste = self.waliduj(metody, progi, kw, etykieta)
        an, sz = [], []
        for a, s in self.pula.imap_unordered(
                _ocen, [(self.scen, metody, z, p, L, kw, progi, etykieta) for z, p, L in zadania], chunksize=4):
            an += a
            sz += s
        print(f"  {etykieta}: {len(zadania)} szeregów, {time.time() - t0:.0f} s", flush=True)
        return an, sz, czyste, kal

    def zapisz(self, nazwa, an, sz, czyste, kal_lista):
        self.katalog.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(an).to_csv(self.katalog / f"{nazwa}_anomalie.csv", index=False)
        pd.DataFrame(sz).to_csv(self.katalog / f"{nazwa}_szeregi.csv", index=False)
        pd.DataFrame(czyste).to_csv(self.katalog / f"{nazwa}_czyste.csv", index=False)
        pd.concat(kal_lista).to_csv(self.katalog / f"{nazwa}_kalibracja.csv", index=False)
        print(f"Zapisano {self.katalog.name}/{nazwa}_*.csv", flush=True)

    def czesc(self, nazwa, konfiguracje):
        """konfiguracje: lista (metody, zadania, kw, etykieta) - wyniki łączone w jeden zestaw plików."""
        AN, SZ, CZ, KAL = [], [], [], []
        for metody, zadania, kw, etykieta in konfiguracje:
            an, sz, cz, kal = self.konfiguracja(metody, zadania, kw, {"czesc": nazwa, **etykieta})
            AN += an; SZ += sz; CZ += cz; KAL.append(kal)
        self.zapisz(nazwa, AN, SZ, CZ, KAL)
