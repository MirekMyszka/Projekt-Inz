"""Etap 2 - SPY i BTCUSDT: szeregi dzienne, ocena metod na katalogu znanych zdarzeń (etykiety częściowe).

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.rzeczywiste.rynki            # oba zbiory
    python -m src.rzeczywiste.rynki spy        # jeden zbiór

Dlaczego szereg dzienny: metody nie dostają ceny (trend), tylko jej przekształcenia (docs/plan_metod.md, rozdz. 1),
a dane 5-min (210 tys. / 958 tys. świec) są za duże dla metod O(n²) (Matrix Profile, kNN). Z danych 5-min liczymy
dla każdego dnia (SPY - sesja, BTC - doba UTC):
    zmienność  log RV  - log zrealizowanej zmienności: ½·log Σ r², r - 5-min log-zwroty bez luk (SPY bez przejścia przez
               noc, BTC bez świec z zerowym wolumenem), przeskalowana do pełnego dnia (78 / 288 świec) - skrócone
               sesje i dni z przerwą techniczną nie wyglądają przez to na „spokojne”,
    wolumen    log wolumenu dziennego (też przeskalowany do pełnego dnia) minus roczna (250 dni) mediana krocząca
               wyśrodkowana na danym dniu - usunięcie wieloletniego trendu (zmiany struktury rynku) przy zachowaniu
               kilkutygodniowych reżimów; mediana wyśrodkowana, bo cała analiza jest offline (pierwsza wersja -
               mediana z 250 dni wstecz - na początku BTC, przy szybko rosnącym rynku 2017, dawała sztuczne skoki).
Dni, w których zostało < 50% świec, są pomijane. Długość: SPY ok. 2 700, BTC ok. 3 300 dni - jak szeregi
symulacyjne (3 000). Właściwość szeregu: log RV ma autokorelację 0,8 (SPY) / 0,76 (BTC) - szereg zależny jak w S2
(φ = 0,7), więc metody = zestaw S2 (model AR(1)), okno w = 10 dni (anomalie od 1 dnia do kilku tygodni).

Podział czasu: kalibracja = do 2019-12-31 (SPY 2016–2019, BTC 2017-08 – 2019), ocena = od 2020-01-01 (m.in. COVID,
2022, 2024-08-05, kwiecień 2025, 2025-10-10). Na kalibracji: uczenie wariantów ML „uczonych na okresie
kalibracyjnym”, skale kart CUSUM (półroczne fragmenty), kara PELT i próg - alarm w α dni okresu kalibracyjnego.

Ocena (katalog.py - etykiety częściowe, więc bez precyzji):
    recall@N  - odsetek zdarzeń katalogu, w których (± 1 dzień) leży choć jeden z N dni o najwyższym wyniku metody
                (N = 1%, 2%, 5% dni całego szeregu) - miara niezależna od progu,
    wykrycie przy progu z kalibracji (α = 1%, 2%, 5% dni) - zdarzenia z okresu oceny, oraz liczba dni z alarmem na rok,
    zgodność metod - wspólna część N = 2% najwyższych dni (indeks Jaccarda) i dni wskazane przez większość metod.
Zmienność oceniana na zdarzeniach typów A–D, wolumen - na typie F (typ E - tylko w studiach przypadków 5-min).

Wyniki: wyniki/rzeczywiste/{spy,btc}_{dni,zdarzenia,metody,zgodnosc,pelt_kary}.csv (pelt_kary - odsetek dni z alarmem
PELT w funkcji kary, diagnostyka).
"""
import sys

import numpy as np
import pandas as pd

from src.config import BTCUSDT, SPY, WYNIKI
from src.metody import CUSUM, PELTAR, CechyKNN, IsolationForest, MatrixProfile, OneClassSVM, ResztyARCUSUM
from src.rzeczywiste.katalog import katalog
from src.rzeczywiste.wspolne import KARY, kara_pelt, odsetek_dni_z_alarmem

KATALOG = WYNIKI / "rzeczywiste"
ZBIORY = {"spy": ("SPY", SPY, 78), "btc": ("BTC", BTCUSDT, 288)}
KONIEC_KAL = pd.Timestamp("2019-12-31").date()
W = 10
ALFY = (0.01, 0.02, 0.05)
NY = (0.01, 0.02, 0.05)
UCZ = "uczony na okresie kalibracyjnym"
SZEREGI = {"zmiennosc": ("A", "B", "C", "D"), "wolumen": ("F",)}
POROWNAWCZE = ("CUSUM z S1 (bez modelu AR)",)


def dzienne(klucz: str) -> pd.DataFrame:
    nazwa, cfg, na_dzien = ZBIORY[klucz]
    df = pd.read_parquet(cfg["clean"])
    dzien = pd.Series(df.index.date, index=df.index)
    r = np.log(df["c"]).diff()
    ok = (df["luka_przed"] == 0) & (dzien == dzien.shift())
    if "zerowy_wolumen" in df:
        ok &= ~df["zerowy_wolumen"]
    r = r[ok]
    g = r.groupby(r.index.date)
    n = df["c"].groupby(df.index.date).count()
    rv = np.sqrt((r ** 2).groupby(r.index.date).sum() * na_dzien / g.count())
    v = df["v"].groupby(df.index.date).sum() * na_dzien / n
    d = pd.DataFrame({"swiece": n, "log_rv": np.log(rv), "log_v": np.log(v.where(v > 0))})
    d = d[(d.swiece >= 0.5 * na_dzien) & np.isfinite(d.log_rv) & np.isfinite(d.log_v)]
    d["wolumen"] = d.log_v - d.log_v.rolling(250, center=True, min_periods=125).median()
    d["zmiennosc"] = d["log_rv"]
    d.index = pd.Index(d.index, name="data")
    return d


def metody():
    m = [ResztyARCUSUM(), PELTAR(korekta_zaleznosci=True), MatrixProfile(w=W, na_resztach=True)]
    m += [CechyKNN(w=W, krok=1, zestaw="s2", cechy=c) for c in (5, 3)]
    m += [IsolationForest(w=W, krok=1, zestaw="s2", cechy=c) for c in (5, 3)]
    m += [OneClassSVM(w=W, krok=1, zestaw="s2", cechy=c) for c in (5, 3)]
    c3 = lambda c: "3 cechy" if c == 3 else "5 cech"
    m += [IsolationForest(w=W, krok=1, zestaw="s2", cechy=c, nauka="czyste", nazwa=f"Isolation Forest ({c3(c)}, {UCZ})")
          for c in (5, 3)]
    m += [OneClassSVM(w=W, krok=1, zestaw="s2", cechy=c, nauka="czyste", nazwa=f"One-Class SVM ({c3(c)}, {UCZ})")
          for c in (5, 3)]
    m.append(CUSUM(nazwa="CUSUM z S1 (bez modelu AR)"))
    return m


def fragmenty(x: np.ndarray, dl: int) -> list[np.ndarray]:
    return [x[i:i + dl] for i in range(0, len(x) - dl + 1, dl)]


def oblicz_wyniki(x: np.ndarray, kal: np.ndarray, dni_kal: np.ndarray, dzien: np.ndarray):
    """Wynik każdej metody dla każdego dnia; PELT - z karą z kalibracji dla każdego α."""
    wyniki, kary, diag = {}, {}, []
    for m in metody():
        m.ucz([x[kal]])
        if hasattr(m, "kalibruj") and not isinstance(m, PELTAR):
            m.kalibruj(fragmenty(x[kal], 126))           # półroczne fragmenty okresu kalibracyjnego
        if isinstance(m, PELTAR):
            wk = lambda k, m=m: np.where((z := m._z(x, k)) > m.Z_KRYT, z, 0.0)
            for k in KARY:                                   # diagnostyka: odsetek dni z alarmem w funkcji kary
                al = wk(k) > 0
                diag.append({"kara": int(k), "odcinki": len(m.granice(x, k)), "dni_alarmu_kal": float(al[kal].mean()),
                             "dni_alarmu_ocena": float(al[~kal].mean())})
            for a in ALFY:
                kary[a] = kara_pelt(wk, dzien, dni_kal, a)
                wyniki[(m.nazwa, a)] = wk(kary[a])
            # ranking (recall@N): wynik z karą dla α = 2% plus drobna składowa z, żeby dni z wynikiem 0 dało się
            # uporządkować (ranking tylko po odcinkach istotnych byłby pełen remisów)
            z = m._z(x, kary[0.02])
            wyniki[m.nazwa] = wyniki[(m.nazwa, 0.02)] + 1e-3 * z
        else:
            wyniki[m.nazwa] = m.wynik(x)
    return wyniki, kary, diag


def ocen(klucz: str):
    nazwa, _, _ = ZBIORY[klucz]
    d = dzienne(klucz)
    daty = np.array(d.index)
    kal = daty <= KONIEC_KAL
    dzien = np.arange(len(d))
    dni_kal = dzien[kal]
    lat_oceny = (~kal).sum() / 252 if klucz == "spy" else (~kal).sum() / 365.25
    kat = katalog()
    kat = kat[kat.zbior == nazwa].reset_index(drop=True)
    wiersze_z, wiersze_m, zgod, dni_out = [], [], [], d[["swiece", "log_rv", "wolumen"]].copy()
    diag_all = []
    for szereg, typy in SZEREGI.items():
        x = d[szereg].values.astype(float)
        wyniki, kary, diag = oblicz_wyniki(x, kal, dni_kal, dzien)
        diag_all += [{"szereg": szereg, **r} for r in diag]
        ev = kat[kat.typ.isin(typy)]
        maski = [(daty >= (pd.Timestamp(o) - pd.Timedelta(days=1)).date()) &
                 (daty <= (pd.Timestamp(e) + pd.Timedelta(days=1)).date()) for o, e in zip(ev.od, ev["do"])]
        top = {}
        for nm, s in wyniki.items():
            if isinstance(nm, tuple):
                continue
            dni_out[f"{szereg}: {nm}"] = s
            rank = pd.Series(s).rank(ascending=False, method="first").values
            for N in NY:
                top[(nm, N)] = rank <= int(round(N * len(s)))
            for a in ALFY:
                sa = wyniki.get((nm, a), s)
                prog = 0.0 if (nm, a) in wyniki else float(np.quantile(sa[kal], 1 - a))
                alarm = sa > prog
                alarmy_rok = alarm[~kal].sum() / lat_oceny
                wiersze_m.append({"zbior": nazwa, "szereg": szereg, "metoda": nm, "alfa": a, "prog": prog,
                                  "kara_pelt": kary.get(a, np.nan) if (nm, a) in wyniki else np.nan,
                                  "dni_alarmu_kal": float(alarm[kal].mean()), "dni_alarmu_ocena": float(alarm[~kal].mean()),
                                  "alarmy_na_rok": alarmy_rok})
                for (_, e), msk in zip(ev.iterrows(), maski):
                    wiersze_z.append({"zbior": nazwa, "szereg": szereg, "metoda": nm, "miara": f"prog_{a}",
                                      "typ": e.typ, "od": e.od, "do": e["do"], "opis": e.opis,
                                      "okres": "kalibracja" if e["do"] <= KONIEC_KAL else "ocena",
                                      "wykryte": bool(alarm[msk].any())})
            for N in NY:
                for (_, e), msk in zip(ev.iterrows(), maski):
                    wiersze_z.append({"zbior": nazwa, "szereg": szereg, "metoda": nm, "miara": f"top_{N}",
                                      "typ": e.typ, "od": e.od, "do": e["do"], "opis": e.opis,
                                      "okres": "kalibracja" if e["do"] <= KONIEC_KAL else "ocena",
                                      "wykryte": bool(top[(nm, N)][msk].any())})
        nazwy = [nm for nm in wyniki if not isinstance(nm, tuple)]
        for i, a in enumerate(nazwy):
            for b in nazwy[i + 1:]:
                ta, tb = top[(a, 0.02)], top[(b, 0.02)]
                zgod.append({"zbior": nazwa, "szereg": szereg, "metoda_a": a, "metoda_b": b,
                             "jaccard": (ta & tb).sum() / (ta | tb).sum()})
        glowne = [nm for nm in nazwy if nm not in POROWNAWCZE]
        dni_out[f"{szereg}: liczba metod w top 2%"] = np.sum([top[(nm, 0.02)] for nm in glowne], axis=0)
        print(f"  {nazwa} {szereg}: " + ", ".join(
            f"{nm.split(' (')[0][:18]} {np.mean([r['wykryte'] for r in wiersze_z if r['metoda'] == nm and r['szereg'] == szereg and r['miara'] == 'top_0.02']):.0%}"
            for nm in nazwy), flush=True)
    KATALOG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(wiersze_z).to_csv(KATALOG / f"{klucz}_zdarzenia.csv", index=False)
    pd.DataFrame(wiersze_m).to_csv(KATALOG / f"{klucz}_metody.csv", index=False)
    pd.DataFrame(zgod).to_csv(KATALOG / f"{klucz}_zgodnosc.csv", index=False)
    dni_out.to_csv(KATALOG / f"{klucz}_dni.csv", float_format="%.5g")
    pd.DataFrame(diag_all).to_csv(KATALOG / f"{klucz}_pelt_kary.csv", index=False, float_format="%.4g")
    print("Zapisano", KATALOG / f"{klucz}_*.csv")


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(ZBIORY)):
        ocen(k)
