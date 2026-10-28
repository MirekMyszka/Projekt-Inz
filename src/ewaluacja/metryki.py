"""Ocena metod wykrywania anomalii zbiorowych (docs/plan_etap_1.md, rozdz. 6 + poprawki po pilocie, rozdz. 8).

Zasady:
1. Metoda zwraca wynik (score) dla każdego punktu; alarm = wynik > próg.
2. Próg jest ustalany JEDNAKOWO dla wszystkich metod na szeregach BEZ anomalii: tak, żeby tylko w odsetku α
   takich szeregów pojawił się choć jeden alarm (domyślnie α = 10%). Wtedy wszystkie metody mają ten sam poziom
   fałszywych alarmów i ich czułość można uczciwie porównać.
3. Alarmy grupujemy w EPIZODY (ciągłe serie punktów z alarmem).
4. Każda anomalia ma STREFĘ TOLERANCJI: [start − tol, koniec + tol). Metody okienkowe i kumulacyjne „rozmywają”
   alarm na kilkadziesiąt punktów wokół anomalii - alarm tuż przed lub tuż po anomalii nie jest fałszywy.
   - anomalia WYKRYTA  = choć jeden punkt z alarmem w jej strefie,
   - FAŁSZYWY ALARM    = epizod, który nie dotyka żadnej strefy.
5. Miary punktowe (precyzja, czułość punktowa, F1): punkty w strefie, ale poza samą anomalią są pomijane
   (ani trafienie, ani błąd).
6. Miary niezależne od progu, osobno dla każdego rodzaju anomalii:
   - AUC (ROC)  - prawdopodobieństwo, że losowy punkt anomalii ma wyższy wynik niż losowy punkt normy,
   - AP (PR-AUC) - średnia precyzja; czuła na rzadkie anomalie.
   Negatywy = punkty normy poza strefami tolerancji.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ALFA_DOMYSLNA = 0.10


def epizody(alarm: np.ndarray) -> list[tuple[int, int]]:
    """Ciągłe serie True jako lista (start, koniec) z końcem wyłącznym."""
    a = np.r_[False, np.asarray(alarm, bool), False]
    zmiany = np.flatnonzero(a[1:] != a[:-1])
    return list(zip(zmiany[::2].tolist(), zmiany[1::2].tolist()))


def maska_stref(anomalie, n: int, tol: int) -> np.ndarray:
    """True dla punktów należących do którejkolwiek strefy tolerancji [start − tol, koniec + tol)."""
    m = np.zeros(n, bool)
    for an in anomalie:
        m[max(0, an.start - tol):min(n, an.koniec + tol)] = True
    return m


def kalibruj_prog(wyniki_czyste: list[np.ndarray], alfa: float = ALFA_DOMYSLNA) -> float:
    """Próg = kwantyl (1 − α) maksimów wyniku na szeregach bez anomalii.

    Przy takim progu w ok. α·100% szeregów bez anomalii pojawi się co najmniej jeden fałszywy alarm.
    """
    maksima = np.array([np.max(w) for w in wyniki_czyste])
    return float(np.quantile(maksima, 1 - alfa))


def ocen_szereg(wynik: np.ndarray, szereg, prog: float, tol: int) -> dict:
    """Ocena jednej metody na jednym szeregu testowym przy ustalonym progu.

    Zwraca: wykryte {rodzaj: bool}, auc/ap {rodzaj: float}, liczba fałszywych epizodów,
    precyzja / czułość / F1 punktowe.
    """
    n = len(wynik)
    alarm = wynik > prog
    etyk = szereg.etykiety
    strefy = maska_stref(szereg.anomalie, n, tol)

    wykryte = {}
    for an in szereg.anomalie:
        a, b = max(0, an.start - tol), min(n, an.koniec + tol)
        wykryte[an.rodzaj] = bool(alarm[a:b].any())

    fa = sum(1 for s, k in epizody(alarm) if not strefy[s:k].any())

    anom = etyk > 0
    pomin = strefy & ~anom                       # margines tolerancji: ani TP, ani FP
    tp = int((alarm & anom).sum())
    fp = int((alarm & ~strefy).sum())
    fn = int((~alarm & anom).sum())
    precyzja = tp / (tp + fp) if tp + fp else np.nan
    czulosc_pkt = tp / (tp + fn) if tp + fn else np.nan
    f1 = (2 * precyzja * czulosc_pkt / (precyzja + czulosc_pkt)
          if tp and not np.isnan(precyzja) and not np.isnan(czulosc_pkt) else 0.0)

    auc, ap = {}, {}
    negatywy = (etyk == 0) & ~strefy
    for an in szereg.anomalie:
        poz = etyk == an.rodzaj
        msk = poz | negatywy
        y = poz[msk]
        s = wynik[msk]
        auc[an.rodzaj] = float(roc_auc_score(y, s)) if np.ptp(s) > 0 else 0.5
        ap[an.rodzaj] = float(average_precision_score(y, s)) if np.ptp(s) > 0 else float(y.mean())

    return {"wykryte": wykryte, "auc": auc, "ap": ap, "fa_epizody": fa,
            "precyzja": precyzja, "czulosc_pkt": czulosc_pkt, "f1": f1, "pominiete_pkt": int(pomin.sum())}


def fa_na_szeregu_czystym(wynik: np.ndarray, prog: float) -> int:
    """Liczba epizodów alarmu na szeregu bez anomalii (każdy jest fałszywy)."""
    return len(epizody(wynik > prog))
