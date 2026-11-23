"""Wspólne narzędzia oceny na danych rzeczywistych (etap 2).

Na danych rzeczywistych nie ma osobnych szeregów bez anomalii, na których w etapie 1 ustalaliśmy próg. Odpowiednik:
JEDNOSTKA CZASU (doba) zamiast szeregu - próg ustalamy tak, żeby alarm pojawił się w odsetku α dni wybranego zbioru
dni „normalnych”:
  - wariant KALIBRACJA (główny, jak w praktyce): dni okresu kalibracyjnego - początkowego fragmentu szeregu, który
    traktujemy jak dane historyczne; próg = kwantyl (1 − α) dziennych maksimów wyniku (etap 1: kwantyl maksimów
    na szeregach bez anomalii),
  - wariant RÓWNY BUDŻET (porównawczy): dni bez anomalii w okresie oceny - wszystkie metody mają wtedy dokładnie
    ten sam odsetek dni z fałszywym alarmem, więc różnice w wykrywaniu nie wynikają z różnej surowości progu.
PELT zwraca wynik 0 poza odcinkami istotnie innymi od tła, więc dla niego tę samą zasadę realizuje kara (najmniejsza
kara, przy której alarm pojawia się w ≤ α dni) - jak w kalibracji PELT w etapie 1.
"""
import numpy as np
import pandas as pd


def dzienne_maksima(wynik: np.ndarray, dzien: np.ndarray) -> pd.Series:
    return pd.Series(wynik).groupby(dzien).max()


def prog_dni(wynik: np.ndarray, dzien: np.ndarray, dni: np.ndarray, alfa: float) -> float:
    """Próg = kwantyl (1 − α) maksimów wyniku w wybranych dniach - alarm w ok. α tych dni."""
    m = dzienne_maksima(wynik, dzien)
    return float(np.quantile(m.loc[dni].values, 1 - alfa))


def odsetek_dni_z_alarmem(alarm: np.ndarray, dzien: np.ndarray, dni: np.ndarray, maska_pkt=None) -> float:
    """Odsetek wybranych dni z co najmniej jednym punktem alarmu (maska_pkt - które punkty dnia brać pod uwagę)."""
    a = pd.Series(alarm if maska_pkt is None else alarm & maska_pkt).groupby(dzien).any()
    return float(a.loc[dni].mean())


KARY = np.unique(np.round(np.geomspace(3, 3000, 31)).astype(int))


def kara_pelt(wynik_dla_kary, dzien: np.ndarray, dni: np.ndarray, alfa: float, maska_pkt=None) -> int:
    """Kara całkowita z siatki KARY (3…3000, geometrycznie), która WYKORZYSTUJE budżet fałszywych alarmów: największy
    odsetek wybranych dni z alarmem (wynik > 0), nie większy niż α (przy remisie - mniejsza kara). Odpowiednik
    progu-kwantyla innych metod: próg też „zużywa” cały budżet α. Siatka zamiast bisekcji z etapu 1: na danych
    rzeczywistych odsetek dni z alarmem NIE maleje monotonicznie z karą (duża kara = długie odcinki, a statystyka z
    rośnie z długością odcinka). wynik_dla_kary(kara) -> wynik punktów. Gdy żadna kara nie spełnia warunku -
    kara o najmniejszym odsetku."""
    odsetki = np.array([odsetek_dni_z_alarmem(wynik_dla_kary(k) > 0, dzien, dni, maska_pkt) for k in KARY])
    ok = odsetki <= alfa
    if not ok.any():
        return int(KARY[int(np.argmin(odsetki))])
    return int(KARY[ok][int(np.argmax(odsetki[ok]))])
