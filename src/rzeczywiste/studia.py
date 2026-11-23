"""Etap 2 - studia przypadków na danych 5-min (SPY, BTCUSDT): wspólne zdarzenia obu rynków i struktura dobowa (FOMC).

Uruchomienie (z katalogu głównego repozytorium, po src.rzeczywiste.rynki):
    python -m src.rzeczywiste.studia

Dla każdego zdarzenia z katalog.STUDIA: 5-min log-zwroty z usuniętą sezonowością dobową, |r| / s(pora dnia), gdzie
s = mediana |r| o tej porze dnia w całym zbiorze (SPY: godzina sesji, BTC: godzina i minuta UTC) - bez tego każde
otwarcie sesji SPY wyglądałoby na anomalię (docs/plan_metod.md, rozdz. 2). Zapisywane:
    wyniki/rzeczywiste/studia_5min.csv - dla każdego dnia studiów: maks. i mediana |r| po usunięciu sezonowości,
        największy ruch (godzina), wynik dzienny metod (z rynki.py) i liczba metod z danym dniem w top 2%,
    wyniki/rzeczywiste/fomc.csv - typ E: dni FOMC w SPY - |r| po usunięciu sezonowości w 14:00–14:55 (komunikat
        o 14:00 i konferencja prasowa o 14:30)
        wobec reszty dnia i wobec tej samej pory w zwykłe dni; czy metody dzienne wskazują te dni.
"""
import numpy as np
import pandas as pd

from src.config import BTCUSDT, SPY, WYNIKI
from src.rzeczywiste.katalog import STUDIA, katalog

KATALOG = WYNIKI / "rzeczywiste"
ZBIORY = {"spy": ("SPY", SPY), "btc": ("BTC", BTCUSDT)}


def zwroty_bez_sezonowosci(klucz: str) -> pd.DataFrame:
    _, cfg = ZBIORY[klucz]
    df = pd.read_parquet(cfg["clean"])
    dzien = pd.Series(df.index.date, index=df.index)
    r = np.log(df["c"]).diff()
    ok = (df["luka_przed"] == 0) & (dzien == dzien.shift())
    if "zerowy_wolumen" in df:
        ok &= ~df["zerowy_wolumen"]
    r = r[ok]
    pora = r.index.strftime("%H:%M")
    s = r.abs().groupby(pora).transform("median")
    return pd.DataFrame({"c": df["c"].reindex(r.index), "r": r, "z": r.abs() / s, "pora": pora})


def policz():
    wiersze, fomc = [], []
    for klucz, (nazwa, _) in ZBIORY.items():
        z = zwroty_bez_sezonowosci(klucz)
        dni = pd.read_csv(KATALOG / f"{klucz}_dni.csv", index_col=0, parse_dates=False)
        kol = [c for c in dni.columns if c.startswith("zmiennosc: ") and "liczba metod" not in c]
        for opis, od, do in STUDIA:
            w = z[(z.index.date >= pd.Timestamp(od).date()) & (z.index.date <= pd.Timestamp(do).date())]
            for d, g in w.groupby(w.index.date):
                i = g.z.idxmax()
                rek = {"zbior": nazwa, "studium": opis, "data": d, "maks_z": g.z.max(), "mediana_z": g.z.median(),
                       "godzina_maks": i.strftime("%H:%M"), "zwrot_maks_%": 100 * g.r.loc[i],
                       "zwrot_dnia_%": 100 * (np.exp(g.r.sum()) - 1)}
                if str(d) in dni.index:
                    rek["metody_top2%"] = int(dni.loc[str(d), "zmiennosc: liczba metod w top 2%"])
                    rek["log_rv"] = float(dni.loc[str(d), "log_rv"])
                wiersze.append(rek)
        if klucz == "spy":
            kat = katalog()
            daty_fomc = set(kat[(kat.zbior == "SPY") & (kat.typ == "E")].od)
            zw = z.copy()
            zw["data"] = zw.index.date
            zw["ogloszenie"] = zw.pora.str.startswith("14:")          # 14:00 komunikat, 14:30 konferencja prasowa
            zwykle = zw[~zw.data.isin(daty_fomc) & zw.ogloszenie].z.median()
            for d in sorted(daty_fomc):
                g = zw[zw.data == d]
                if g.empty:
                    continue
                rek = {"data": d, "z_14_maks": g[g.ogloszenie].z.max(),
                       "z_14_mediana": g[g.ogloszenie].z.median(),
                       "z_reszta_dnia_mediana": g[~g.ogloszenie].z.median(), "z_14_mediana_zwykle_dni": zwykle}
                if str(d) in dni.index:
                    rek["metody_top2%"] = int(dni.loc[str(d), "zmiennosc: liczba metod w top 2%"])
                    rek["pozycja_log_rv_%"] = float((dni.log_rv > dni.loc[str(d), "log_rv"]).mean() * 100)
                fomc.append(rek)
    pd.DataFrame(wiersze).to_csv(KATALOG / "studia_5min.csv", index=False, float_format="%.4g")
    pd.DataFrame(fomc).to_csv(KATALOG / "fomc.csv", index=False, float_format="%.4g")
    print("Zapisano", KATALOG / "studia_5min.csv", "i fomc.csv")


if __name__ == "__main__":
    policz()
