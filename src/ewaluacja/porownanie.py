"""Porównanie scenariuszy S1–S3 (docs/plan_etap_1.md, krok 9) - eksperyment PRZENIESIENIA metod.

Pytanie: ile skuteczności metody zawdzięczają dopasowaniu do właściwości szeregu? Zestaw metod skonfigurowany dla
scenariusza X (metody_s1 / metody_s2 / metody_s3, okno w = 50, warianty 5 i 3 cech) uruchamiamy na szeregach
scenariusza Y. Daje to macierz 3 × 3: na przekątnej metody „u siebie” (wyniki z części głównej każdego scenariusza),
poza przekątną - metody przeniesione do szeregu o innych właściwościach:
    metody S1 (bez modelu tła)         → S2 (zależność AR), S3 (cykl),
    metody S2 (reszty modelu AR(1))    → S1, S3,
    metody S3 (reszty STL, cechy cykli) → S1, S2.

Procedura jak w każdym scenariuszu (src/ewaluacja/eksperyment.py): stałe metod i progi kalibrowane na 200 szeregach
bez anomalii SCENARIUSZA DOCELOWEGO (fałszywy alarm w 10% szeregów), walidacja na 200 nowych, ocena na szeregach
z anomaliami. Szeregi oceny to DOKŁADNIE te same szeregi co w części głównej scenariusza docelowego (te same ziarna):
poziomy intensywności 2 i 3, wszystkie długości L scenariusza, pierwsze 50 ze 100 szeregów na (poziom, L) - porównanie z przekątną
jest więc sparowane (ta sama anomalia, inna metoda).

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.ewaluacja.porownanie              # wszystkie 6 komórek, ok. 2 h na 2 rdzeniach
    python -m src.ewaluacja.porownanie "S3>S1"      # wybrane komórki (metody S3 na szeregach S1)
    python -m src.ewaluacja.porownanie --szybko     # mała wersja do sprawdzenia łańcucha
Wyniki (CSV): wyniki/porownanie/przeniesienie_{anomalie,szeregi,czyste,kalibracja}.csv (kolumny metody_z, szereg -
    skąd metody, na jakich szeregach); każda policzona komórka zastępuje swoje wcześniejsze wiersze.
"""
import os
import sys
import time

from src.config import WYNIKI
from src.ewaluacja.eksperyment import Eksperyment
from src.metody import metody_s1, metody_s2, metody_s3
from src.symulacje.anomalie import LS_S3, RODZAJE_S1, RODZAJE_S2, RODZAJE_S3, szereg_s1, szereg_s2, szereg_s3

SCENARIUSZE = {
    "S1": {"generuj": szereg_s1, "rodzaje": RODZAJE_S1, "parametr": {1: "delta", 2: "c", 3: "phi"}},
    "S2": {"generuj": szereg_s2, "rodzaje": RODZAJE_S2, "parametr": {1: "impuls", 2: "phi_prim", 3: "delta"}},
    "S3": {"generuj": szereg_s3, "rodzaje": RODZAJE_S3, "parametr": {1: "faza_pi", 2: "k", 3: "r"}},
}
DLUGOSCI = {"S1": (20, 50, 100), "S2": (20, 50, 100), "S3": LS_S3}
# Ziarna części głównej: ZIARNO_0 + 100 000·poziom + 10 000·(indeks L) + nr szeregu (eksperyment_s1/s2/s3.py).
ZIARNO_0 = {"S1": 0, "S2": 3_100_000, "S3": 6_100_000}
METODY = {"S1": lambda: metody_s1(w=50, z_porownawczymi=False),
          "S2": lambda: metody_s2(w=50),
          "S3": lambda: metody_s3(w=50, z_porownawczymi=False)}
POZIOMY = (2, 3)
R_OCENA = 50          # szeregów na (poziom, L): pierwsze 50 ze 100 w części głównej


def zadania(szereg: str, R: int) -> list:
    """(ziarno, poziom, L) - te same szeregi co w części głównej scenariusza `szereg`."""
    return [(ZIARNO_0[szereg] + 100_000 * p + 10_000 * i + r, p, L)
            for p in POZIOMY for i, L in enumerate(DLUGOSCI[szereg]) for r in range(R)]


def pary():
    return [(z, s) for z in METODY for s in SCENARIUSZE if z != s]


PLIKI = ("anomalie", "szeregi", "czyste", "kalibracja")


def policz(z: str, s: str, katalog, R: int, n_kal: int, n_wal: int):
    """Jedna komórka macierzy (metody scenariusza z na szeregach scenariusza s). Wynik dopisywany do
    katalog/przeniesienie_*.csv w miejsce wcześniejszych wierszy tej komórki - komórki można liczyć osobno."""
    import pandas as pd
    e = Eksperyment(SCENARIUSZE[s], katalog, max(1, os.cpu_count() or 1), n_kal, n_wal)
    an, sz, cz, kal = e.konfiguracja(METODY[z](), zadania(s, R), {},
                                     {"czesc": "przeniesienie", "metody_z": z, "szereg": s, "w": 50})
    e.zamknij()
    katalog.mkdir(parents=True, exist_ok=True)
    kolejnosc = {p: i for i, p in enumerate(pary())}
    for f, nowe in zip(PLIKI, (pd.DataFrame(an), pd.DataFrame(sz), pd.DataFrame(cz), kal)):
        plik = katalog / f"przeniesienie_{f}.csv"
        stare = pd.read_csv(plik) if plik.exists() else pd.DataFrame(columns=nowe.columns)
        stare = stare[~((stare.metody_z == z) & (stare.szereg == s))]
        t = pd.concat([stare, nowe], ignore_index=True)
        t = t.iloc[sorted(range(len(t)), key=lambda i: kolejnosc[(t.metody_z.iloc[i], t.szereg.iloc[i])])]
        t.to_csv(plik, index=False)
    print(f"Zapisano komórkę {z} → {s} w {katalog.name}/przeniesienie_*.csv", flush=True)


if __name__ == "__main__":
    szybko = "--szybko" in sys.argv
    R, n_kal, n_wal = (4, 30, 30) if szybko else (R_OCENA, 200, 200)
    katalog = WYNIKI / ("porownanie_szybki" if szybko else "porownanie")
    wybrane = [tuple(a.split(">")) for a in sys.argv[1:] if ">" in a] or pary()
    t = time.time()
    for z, s in wybrane:
        policz(z, s, katalog, R, n_kal, n_wal)
    print(f"Gotowe w {(time.time() - t) / 60:.1f} min")
