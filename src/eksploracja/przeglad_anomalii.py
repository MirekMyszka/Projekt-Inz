"""Wstępny przegląd anomalii w danych rzeczywistych (SPY, BTCUSDT) - podstawa do docs/plan_metod.md.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.eksploracja.przeglad_anomalii            # oba zbiory
    python -m src.eksploracja.przeglad_anomalii btcusdt    # jeden zbiór

Wypisuje (bez zapisywania plików):
- właściwości szeregu: kurtoza zwrotów, autokorelacja |r| (skupianie zmienności), profil dobowy/tygodniowy,
- anomalie punktowe: największe 5-min log-zwroty wg globalnego i lokalnego (kroczącego) robust z-score,
- reżimy zmienności: dni i miesiące o największej / najmniejszej zrealizowanej zmienności (RV),
- anomalie wolumenu: skoki względem typowego wolumenu o tej porze dnia, dni o skrajnym wolumenie,
- SPY: luki overnight; BTC: największe ruchy 1-godzinne i 1-dniowe.
Zwroty liczone tylko między sąsiednimi świecami (bez luk; dla SPY bez przejść przez noc).
"""
import sys

import numpy as np
import pandas as pd

from src.config import BTCUSDT, SPY

ZBIORY = {"spy": SPY, "btcusdt": BTCUSDT}


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    return (x - med) / (1.4826 * (x - med).abs().median())


def przeglad(nazwa: str):
    cfg = ZBIORY[nazwa]
    df = pd.read_parquet(cfg["clean"])
    spy = nazwa == "spy"
    na_dzien = 78 if spy else 288
    print(f"\n{'#' * 20} {nazwa.upper()} {'#' * 20}")

    dzien = pd.Series(df.index.date, index=df.index)
    r = np.log(df["c"]).diff()
    ok = (df["luka_przed"] == 0) & ((dzien == dzien.shift()) if spy else ~df["zerowy_wolumen"])
    r = r[ok].dropna()

    # --- właściwości szeregu
    print(f"Zwrotów: {len(r)} | kurtoza: {r.kurt():.1f} | ACF r(1): {r.autocorr(1):.3f} | "
          f"ACF |r|(1): {r.abs().autocorr(1):.3f} | ACF |r|(1 dzień): {r.abs().autocorr(na_dzien):.3f}")
    a = r.abs()
    if spy:
        prof = a.groupby(a.index.strftime("%H:%M")).mean()
        srodek = prof.iloc[30:50].mean()
        print(f"Profil U: początek sesji {prof.iloc[:3].mean() / srodek:.2f}x, koniec {prof.iloc[-3:].mean() / srodek:.2f}x środka dnia")
    else:
        godz = a.groupby(a.index.hour).mean()
        tydz = a.groupby(a.index.dayofweek).mean()
        print("Profil godzinowy |r| (UTC, względem średniej):", (godz / godz.mean()).round(2).to_dict())
        print("Profil tygodniowy (0=pon):", (tydz / tydz.mean()).round(2).to_dict())

    # --- anomalie punktowe
    z = robust_z(r)
    lok = r.abs().rolling(na_dzien * (5 if spy else 1), min_periods=50).median()
    zl = r / (1.4826 * lok)
    print(f"\n|z| > 10: {(z.abs() > 10).sum()}, |z| > 20: {(z.abs() > 20).sum()}, |z_lokalny| > 10: {(zl.abs() > 10).sum()}")
    for nazwa_z, zz in [("globalny robust z", z), ("lokalny (kroczący) robust z", zl)]:
        top = zz.abs().nlargest(12).index
        print(f"Największe zwroty wg: {nazwa_z}")
        print(pd.DataFrame({"zwrot_%": (r[top] * 100).round(2), "z": zz[top].round(1)}).to_string())

    # --- reżimy zmienności
    rv = (r ** 2).groupby(r.index.date).sum() ** 0.5
    mies = rv.groupby(pd.to_datetime(rv.index).to_period("M")).median()
    print(f"\nDzienna RV: mediana {rv.median() * 100:.2f}%")
    print("Dni o największej RV [%]:\n" + (rv.nlargest(10) * 100).round(2).to_string())
    print("Miesiące o największej medianie RV [%]:\n" + (mies.nlargest(8) * 100).round(2).to_string())
    print("Miesiące o najmniejszej medianie RV [%]:\n" + (mies.nsmallest(5) * 100).round(2).to_string())

    # --- anomalie wolumenu
    lv = np.log(df["v"].where(df["v"] > 0))
    dv = lv - lv.groupby(df.index.strftime("%H:%M")).transform("median")
    vz = dv - dv.rolling(2000, min_periods=200).median()
    print("\nSkoki wolumenu 5-min (log względem pory dnia i trendu):\n" + vz.nlargest(10).round(2).to_string())
    dvol = df["v"].groupby(df.index.date).sum()
    rel = dvol / dvol.rolling(60, center=True, min_periods=20).median()
    print("Dni o największym wolumenie względnym:\n" + rel.nlargest(10).round(2).to_string())
    print("Dni o najmniejszym wolumenie względnym:\n" + rel.nsmallest(8).round(2).to_string())

    # --- ruchy na dłuższych skalach
    if spy:
        on = np.log(df["o"]).groupby(dzien).first() - np.log(df["c"]).groupby(dzien).last().shift()
        top = on.abs().nlargest(10).index
        print("\nNajwiększe luki overnight [%]:\n" + ((np.exp(on[top]) - 1) * 100).round(2).to_string())
    else:
        for okres in ["1h", "1D"]:
            m = np.log(df["c"]).resample(okres).last().diff()
            top = m.abs().nlargest(10).index
            print(f"\nNajwiększe ruchy {okres} [%]:\n" + ((np.exp(m[top]) - 1) * 100).round(1).to_string())


if __name__ == "__main__":
    for n in (sys.argv[1:] or ZBIORY):
        przeglad(n.lower())
