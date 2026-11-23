"""Kontrola jakości i właściwości danych rzeczywistych (SPY, BTCUSDT, NYC Taxi) - podstawa raportu z części danych.

Uruchomienie (z katalogu głównego repozytorium, po czyszczeniu):
    python -m src.eksploracja.raport_danych

Kontrola jest NIEZALEŻNA od skryptów czyszczenia: siatka oczekiwanych świec liczona od nowa (SPY - kalendarz NYSE
z pandas_market_calendars, BTC - pełna siatka 5-min, NYC - siatka 30-min), spójność OHLC i VWAP, duplikaty, braki.
Właściwości szeregów: rozkład i autokorelacja 5-min log-zwrotów (bez luk; SPY bez przejść przez noc), zmienność
dzienna, profil dobowy i tygodniowy; NYC - autokorelacja przy dobie i tygodniu, stosunek ruchu dzień / noc.

Zapisuje:
    wyniki/dane/jakosc.csv          - zbiór, miara, wartość (kontrola jakości)
    wyniki/dane/przerwy_btc.csv     - ciągłe bloki brakujących świec BTC (początek, koniec, liczba świec)
    wyniki/dane/wlasciwosci.csv     - zbiór, miara, wartość (właściwości szeregów)
    wyniki/dane/anomalie.csv        - przykłady anomalii (jak w przeglad_anomalii.py): największe zwroty wg globalnego
                                      i lokalnego robust z, dni i miesiące o skrajnej RV, wolumen, luki / ruchy dobowe
    wyniki/dane/profile.csv         - profil dobowy |r| (SPY: pora sesji, BTC: godzina UTC) i tygodniowy (BTC)
    wyniki/wykresy/dane_szeregi.png    - trzy zbiory: cena SPY, cena BTC (skala log), NYC Taxi z oknami NAB
    wyniki/wykresy/dane_wlasciwosci.png - profile dobowe, ogony rozkładu zwrotów, autokorelacja |r|
    wyniki/wykresy/dane_jakosc.png     - braki i świece bez transakcji w czasie; tydzień NYC Taxi
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import BTCUSDT, NYC_TAXI, SPY, WYKRESY, WYNIKI
from src.ewaluacja.wykresy_s1 import SIATKA, TEKST, TEKST2

KATALOG = WYNIKI / "dane"
NIEB, POM, ZIEL, SZARY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"


def _ohlc_ok(df):
    return ((df.l <= df[["o", "c"]].min(axis=1)) & (df.h >= df[["o", "c"]].max(axis=1)) & (df.l <= df.h))


def zwroty(df, spy):
    dzien = pd.Series(df.index.date, index=df.index)
    r = np.log(df["c"]).diff()
    ok = (df["luka_przed"] == 0) & (dzien == dzien.shift() if spy else ~df["zerowy_wolumen"])
    return r[ok].dropna()


# ------------------------------------------------------------------------------------------------ jakość
def jakosc_spy():
    import pandas_market_calendars as mcal
    df = pd.read_parquet(SPY["clean"])
    kal = mcal.get_calendar("NYSE").schedule(start_date=df.index[0].date(), end_date=df.index[-1].date())
    siatka = []
    for _, s in kal.iterrows():
        siatka.append(pd.date_range(s.market_open, s.market_close, freq="5min", inclusive="left"))
    siatka = pd.DatetimeIndex(np.concatenate(siatka)).tz_convert("America/New_York")
    brak = siatka.difference(df.index)
    poza = df.index.difference(siatka)
    n = df.groupby(df.index.date).size()
    oczek = pd.Series(siatka.date).value_counts().sort_index()
    w = {
        "świece": len(df), "pierwsza": df.index[0], "ostatnia": df.index[-1], "dni sesji w kalendarzu NYSE": len(kal),
        "dni w danych": n.size, "dni z 78 świecami": int((n == 78).sum()),
        "sesje skrócone w kalendarzu (42 świece)": int((oczek == 42).sum()),
        "sesje skrócone w danych z 42 świecami": int(((oczek == 42) & (n.reindex(oczek.index) == 42)).sum()),
        "świece poza godzinami sesji": len(poza), "brakujące świece": len(brak),
        "dni z brakami": len(set(brak.date)), "duplikaty": int(df.index.duplicated().sum()),
        "wartości NaN": int(df[["o", "h", "l", "c", "v"]].isna().sum().sum()),
        "świece z niespójnym OHLC": int((~_ohlc_ok(df)).sum()),
        "świece z VWAP poza [l, h]": int(((df.vw < df.l - 1e-9) | (df.vw > df.h + 1e-9)).sum()),
        "świece z zerowym wolumenem": int((df.v == 0).sum()),
        "dni ze wstrzymaniem handlu": int(df.groupby(df.index.date).wstrzymanie_handlu.any().sum()),
        "brakujące świece w dniach wstrzymania handlu": int(pd.Series(brak.date).isin(
            set(df.index.date[df.wstrzymanie_handlu])).sum()),
        "zgodność z listą brakujących świec ze skryptu czyszczenia": bool(
            len(brak) == len(pd.read_csv(SPY["clean"].parent / "spy_brakujace_swiece.csv"))),
    }
    return w, brak


def jakosc_btc():
    df = pd.read_parquet(BTCUSDT["clean"])
    siatka = pd.date_range(df.index[0], df.index[-1], freq="5min")
    brak = siatka.difference(df.index)
    n = df.groupby(df.index.date).size()
    bloki = pd.Series(brak).diff().ne(pd.Timedelta("5min")).cumsum()
    przerwy = pd.DataFrame({"od": pd.Series(brak).groupby(bloki.values).min(),
                            "do": pd.Series(brak).groupby(bloki.values).max(),
                            "swiece": pd.Series(brak).groupby(bloki.values).size()}).sort_values("swiece", ascending=False)
    zv = df[df.v == 0]
    w = {
        "świece": len(df), "pierwsza": df.index[0], "ostatnia": df.index[-1], "świece w pełnej siatce 5-min": len(siatka),
        "dni (UTC)": n.size, "dni z 288 świecami": int((n == 288).sum()), "świece pierwszego dnia": int(n.iloc[0]),
        "brakujące świece": len(brak), "przerwy (ciągłe bloki braków)": len(przerwy),
        "dni z przerwą": len(set(brak.date)), "najdłuższa przerwa [świece]": int(przerwy.swiece.max()),
        "świece poza siatką 5-min": int((df.index != df.index.floor("5min")).sum()),
        "duplikaty": int(df.index.duplicated().sum()),
        "wartości NaN": int(df[["o", "h", "l", "c", "v"]].isna().sum().sum()),
        "świece z niespójnym OHLC": int((~_ohlc_ok(df)).sum()),
        "świece z VWAP poza [l, h]": int(((df.vw < df.l - 1e-9) | (df.vw > df.h + 1e-9)).sum()),
        "świece z zerowym wolumenem": len(zv),
        "  w tym płaskie (o = h = l = c)": int(((zv.o == zv.h) & (zv.h == zv.l) & (zv.l == zv.c)).sum()),
        "  w tym w 2017 r.": int((zv.index.year == 2017).sum()),
        "świece ze skorygowanym czasem": int(df.czas_skorygowany.sum()),
        "zgodność z listą brakujących świec ze skryptu czyszczenia": bool(
            len(brak) == len(pd.read_csv(BTCUSDT["clean"].parent / "btcusdt_brakujace_swiece.csv"))),
    }
    return w, przerwy, df


def jakosc_nyc():
    df = pd.read_parquet(NYC_TAXI["clean"])
    siatka = pd.date_range(df.index[0], df.index[-1], freq="30min")
    okna = df[df.anomalia_nab].groupby((~df.anomalia_nab).cumsum()[df.anomalia_nab]).size()
    w = {
        "punkty": len(df), "pierwszy": df.index[0], "ostatni": df.index[-1], "punkty w siatce 30-min": len(siatka),
        "dni": df.index.normalize().nunique(), "brakujące punkty": len(siatka.difference(df.index)),
        "duplikaty": int(df.index.duplicated().sum()), "wartości NaN": int(df.pasazerowie.isna().sum()),
        "okna NAB": len(okna), "punkty w oknie NAB (każde okno)": ", ".join(map(str, sorted(set(okna)))),
        "punkty w oknach NAB": int(df.anomalia_nab.sum()), "odsetek punktów w oknach": df.anomalia_nab.mean(),
        "punkty anomalii NAB (środki okien)": int(df.punkt_nab.sum()),
        "punkty zmiany czasu": int(df.zmiana_czasu.sum()), "maksimum": int(df.pasazerowie.max()),
        "czas maksimum": df.pasazerowie.idxmax(), "minimum": int(df.pasazerowie.min()),
        "czas minimum": df.pasazerowie.idxmin(),
    }
    return w, df


# ------------------------------------------------------------------------------------------------ właściwości
def wlasciwosci_rynek(df, spy):
    r = zwroty(df, spy)
    rv = np.sqrt((r ** 2).groupby(r.index.date).sum())
    zs = (r - r.median()) / (1.4826 * (r - r.median()).abs().median())
    a = r.abs()
    w = {"zwroty 5-min": len(r), "kurtoza (nadwyżkowa)": r.kurt(), "autokorelacja r (1)": r.autocorr(1),
         "autokorelacja |r| (1)": a.autocorr(1), "autokorelacja |r| (1 dzień)": a.autocorr(78 if spy else 288),
         "mediana dziennej RV [%]": 100 * rv.median(), "|robust z| > 10": int((zs.abs() > 10).sum()),
         "|robust z| > 10 przy rozkładzie normalnym (oczekiwane)": 2 * len(r) * norm.sf(10)}
    if spy:
        prof = a.groupby(a.index.strftime("%H:%M")).mean()
        srodek = prof.iloc[30:50].mean()     # jak w przeglad_anomalii.py
        w["początek sesji / środek dnia"] = prof.iloc[:3].mean() / srodek
        w["koniec sesji / środek dnia"] = prof.iloc[-3:].mean() / srodek
        profile = {"dobowy": prof / prof.mean()}
    else:
        godz = a.groupby(a.index.hour).mean()
        tydz = a.groupby(a.index.dayofweek).mean()
        w["godzina maks. / średnia"] = godz.max() / godz.mean()
        w["godzina maks. (UTC)"] = int(godz.idxmax())
        w["godzina min. / średnia"] = godz.min() / godz.mean()
        w["weekend / dni robocze"] = tydz.iloc[5:].mean() / tydz.iloc[:5].mean()
        profile = {"dobowy": godz / godz.mean(), "tygodniowy": tydz / tydz.mean()}
    acf = {l: a.autocorr(l) for l in (1, 2, 3, 6, 12, 24, 48, 78, 144, 288, 576)}
    return w, profile, acf, zs


def wlasciwosci_nyc(df):
    y = df.pasazerowie.astype(float)
    godz = y.groupby(y.index.hour).mean()
    dzien_tyg = y.groupby(y.index.dayofweek).mean()
    return {"średnia liczba pasażerów / 30 min": y.mean(),
            "autokorelacja (1 doba = 48)": y.autocorr(48), "autokorelacja (1 tydzień = 336)": y.autocorr(336),
            "autokorelacja (pół doby = 24)": y.autocorr(24),
            "godzina maks. / godzina min. (średnio)": godz.max() / godz.min(),
            "godzina maks.": int(godz.idxmax()), "godzina min.": int(godz.idxmin()),
            "dzień tygodnia maks. / min.": dzien_tyg.max() / dzien_tyg.min()}


def anomalie(df, spy, zbior):
    """Te same miary co w przeglad_anomalii.py, zapisane do tabeli (kategoria, czas, wartość, z)."""
    r = zwroty(df, spy)
    na_dzien = 78 if spy else 288
    z = (r - r.median()) / (1.4826 * (r - r.median()).abs().median())
    zl = r / (1.4826 * r.abs().rolling(na_dzien * (5 if spy else 1), min_periods=50).median())
    w = []
    dodaj = lambda kat, s, z_=None: w.extend({"zbior": zbior, "kategoria": kat, "czas": t, "wartosc": v,
                                              "z": None if z_ is None else z_[t]} for t, v in s.items())
    for kat, zz in (("zwrot 5-min [%], globalny z", z), ("zwrot 5-min [%], lokalny z", zl)):
        top = zz.abs().nlargest(12).index
        dodaj(kat, 100 * r[top], zz)
    rv = (r ** 2).groupby(r.index.date).sum() ** 0.5
    mies = rv.groupby(pd.to_datetime(rv.index).to_period("M")).median()
    dodaj("dzienna RV [%] - największe", 100 * rv.nlargest(10))
    dodaj("miesięczna mediana RV [%] - największe", 100 * mies.nlargest(8))
    dodaj("miesięczna mediana RV [%] - najmniejsze", 100 * mies.nsmallest(5))
    w.append({"zbior": zbior, "kategoria": "mediana dziennej RV [%]", "czas": "", "wartosc": 100 * rv.median()})
    dvol = df["v"].groupby(df.index.date).sum()
    rel = dvol / dvol.rolling(60, center=True, min_periods=20).median()
    dodaj("wolumen dzienny / mediana 60 dni - największe", rel.nlargest(10))
    dodaj("wolumen dzienny / mediana 60 dni - najmniejsze", rel.nsmallest(8))
    if spy:
        dzien = pd.Series(df.index.date, index=df.index)
        on = np.log(df["o"]).groupby(dzien).first() - np.log(df["c"]).groupby(dzien).last().shift()
        dodaj("luka overnight [%]", 100 * (np.exp(on[on.abs().nlargest(10).index]) - 1))
    else:
        m = np.log(df["c"]).resample("1D").last().diff()
        dodaj("ruch dobowy [%]", 100 * (np.exp(m[m.abs().nlargest(10).index]) - 1))
    w.append({"zbior": zbior, "kategoria": "|z| > 10 (globalny)", "czas": "", "wartosc": int((z.abs() > 10).sum())})
    w.append({"zbior": zbior, "kategoria": "|z| > 20 (globalny)", "czas": "", "wartosc": int((z.abs() > 20).sum())})
    w.append({"zbior": zbior, "kategoria": "|z| > 10 (lokalny)", "czas": "", "wartosc": int((zl.abs() > 10).sum())})
    return w


# ------------------------------------------------------------------------------------------------ wykresy
def _styl(ax):
    ax.grid(color=SIATKA, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def wykresy(spy, btc, nyc, prof_spy, prof_btc, acf_spy, acf_btc, z_spy, z_btc, brak_spy, przerwy):
    # 1. szeregi
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.2))
    c = spy.c.groupby(spy.index.date).last()
    ax[0].plot(pd.to_datetime(c.index), c.values, color=NIEB, lw=0.9)
    ax[0].set_title("SPY – cena zamknięcia sesji (z danych 5-min, 2016-01 – 2026-09)", loc="left", fontsize=9.5)
    ax[0].set_ylabel("USD")
    b = btc.c.groupby(btc.index.date).last()
    ax[1].semilogy(pd.to_datetime(b.index), b.values, color=POM, lw=0.9)
    ax[1].set_title("BTCUSDT – cena na koniec doby UTC (skala logarytmiczna, 2017-08 – 2026-09)", loc="left", fontsize=9.5)
    ax[1].set_ylabel("USDT")
    ax[2].plot(nyc.index, nyc.pasazerowie, color=TEKST2, lw=0.5)
    for i, (_, g) in enumerate(nyc[nyc.anomalia_nab].groupby((~nyc.anomalia_nab).cumsum()[nyc.anomalia_nab])):
        ax[2].axvspan(g.index[0], g.index[-1], color=POM, alpha=0.18, lw=0)
        ax[2].text(g.index[len(g) // 2], nyc.pasazerowie.max() * (1.12 if i % 2 else 1.03), g.zdarzenie.iloc[0],
                   ha="center", va="top", fontsize=7, color=POM)
    ax[2].set_ylim(0, nyc.pasazerowie.max() * 1.14)
    ax[2].set_title("NYC Taxi – liczba pasażerów co 30 min; zaznaczone okna anomalii NAB", loc="left", fontsize=9.5)
    for a in ax:
        _styl(a)
    fig.tight_layout()
    fig.savefig(WYKRESY / "dane_szeregi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. właściwości
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.4))
    p = prof_spy["dobowy"]
    ax[0, 0].bar(range(len(p)), p.values, color=NIEB, width=0.85)
    ax[0, 0].set_xticks(range(0, len(p), 6), p.index[::6], rotation=0, fontsize=7.5)
    ax[0, 0].axhline(1, color=TEKST2, lw=0.8, ls="--")
    ax[0, 0].set_title("SPY – średnie |r| o danej porze sesji (względem średniej) – profil „U”", loc="left", fontsize=9.5)
    g, t = prof_btc["dobowy"], prof_btc["tygodniowy"]
    ax[0, 1].bar(g.index, g.values, color=POM, width=0.85)
    ax[0, 1].axhline(1, color=TEKST2, lw=0.8, ls="--")
    ax[0, 1].set_xticks(range(0, 24, 2))
    ax[0, 1].set_xlabel("godzina UTC")
    dni = ["pn", "wt", "śr", "cz", "pt", "so", "nd"]
    ax[0, 1].set_title("BTC – średnie |r| o danej godzinie; dni tygodnia: " +
                       ", ".join(f"{d} {v:.2f}".replace(".", ",") for d, v in zip(dni, t.values)), loc="left", fontsize=8.6)
    for z, kol, nazwa in ((z_spy, NIEB, "SPY"), (z_btc, POM, "BTC")):
        x = np.sort(np.abs(z.values))
        ogon = 1 - np.arange(len(x)) / len(x)
        ax[1, 0].loglog(x[::-1][:200000][::-1], ogon[::-1][:200000][::-1], color=kol, lw=1.2, label=nazwa)
    xx = np.logspace(-1, 1.2, 60)
    ax[1, 0].loglog(xx, 2 * norm.sf(xx), color=TEKST2, ls="--", lw=1, label="rozkład normalny")
    ax[1, 0].set_ylim(1e-6, 1.2)
    ax[1, 0].set_xlabel("|robust z| zwrotu 5-min")
    ax[1, 0].set_ylabel("P(|z| > x)")
    ax[1, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].set_title("Ogony rozkładu zwrotów 5-min – znacznie cięższe niż normalne", loc="left", fontsize=9.5)
    for acf, kol, nazwa, k in ((acf_spy, NIEB, "SPY (1 dzień = 78)", 78), (acf_btc, POM, "BTC (1 doba = 288)", 288)):
        l = np.array(list(acf))
        ax[1, 1].semilogx(l, list(acf.values()), "o-", color=kol, ms=3.5, lw=1.2, label=nazwa)
    ax[1, 1].axhline(0, color=TEKST2, lw=0.8)
    ax[1, 1].set_xlabel("przesunięcie [świece 5-min]")
    ax[1, 1].legend(frameon=False, fontsize=8)
    ax[1, 1].set_title("Autokorelacja |r| – skupianie zmienności", loc="left", fontsize=9.5)
    for a in ax.flat:
        _styl(a)
    fig.tight_layout()
    fig.savefig(WYKRESY / "dane_wlasciwosci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. jakość + tydzień NYC
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    mies = pd.period_range("2017-08", "2026-09", freq="M")
    brak_m = pd.Series(1, index=pd.DatetimeIndex(np.repeat(przerwy.od.values, przerwy.swiece.values))).groupby(
        lambda t: t.to_period("M")).sum().reindex(mies, fill_value=0)
    zv = btc[btc.v == 0]
    zv_m = zv.groupby(zv.index.tz_localize(None).to_period("M")).size().reindex(mies, fill_value=0)
    x = mies.to_timestamp()
    ax[0].bar(x, brak_m.values, width=25, color=POM, label="brakujące świece BTC")
    ax[0].bar(x, zv_m.values, width=25, bottom=brak_m.values, color=SZARY, label="świece BTC bez transakcji")
    sm = pd.Series(1, index=brak_spy.tz_localize(None)).groupby(lambda t: t.to_period("M")).sum()
    ax[0].bar(sm.index.to_timestamp(), sm.values, width=25, color=NIEB, label="brakujące świece SPY")
    ax[0].set_yscale("symlog", linthresh=10)
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Braki danych i świece bez transakcji – liczba w miesiącu", loc="left", fontsize=9.5)
    y = nyc.pasazerowie.astype(float)
    tyg = y.groupby([y.index.dayofweek, y.index.hour + y.index.minute / 60]).agg(["median", lambda v: v.quantile(.1),
                                                                                lambda v: v.quantile(.9)])
    xs = np.arange(len(tyg)) / 48
    ax[1].fill_between(xs, tyg.iloc[:, 1], tyg.iloc[:, 2], color=NIEB, alpha=0.2, lw=0, label="10–90% tygodni")
    ax[1].plot(xs, tyg.iloc[:, 0], color=NIEB, lw=1.1, label="mediana")
    ax[1].set_xticks(np.arange(7) + 0.5, dni)
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")
    ax[1].set_title("NYC Taxi – typowy tydzień (pasażerowie / 30 min)", loc="left", fontsize=9.5)
    for a in ax:
        _styl(a)
    fig.tight_layout()
    fig.savefig(WYKRESY / "dane_jakosc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    KATALOG.mkdir(parents=True, exist_ok=True)
    WYKRESY.mkdir(parents=True, exist_ok=True)
    js, brak_spy = jakosc_spy()
    jb, przerwy, btc = jakosc_btc()
    jn, nyc = jakosc_nyc()
    spy = pd.read_parquet(SPY["clean"])
    ws, ps, acf_s, zs = wlasciwosci_rynek(spy, True)
    wb, pb, acf_b, zb = wlasciwosci_rynek(btc, False)
    wn = wlasciwosci_nyc(nyc)
    rek = lambda zb_, d: [{"zbior": zb_, "miara": k, "wartosc": v} for k, v in d.items()]
    pd.DataFrame(rek("SPY", js) + rek("BTC", jb) + rek("NYC", jn)).to_csv(KATALOG / "jakosc.csv", index=False)
    przerwy.to_csv(KATALOG / "przerwy_btc.csv", index=False)
    pd.DataFrame(rek("SPY", ws) + rek("BTC", wb) + rek("NYC", wn)
                 + [{"zbior": z_, "miara": f"autokorelacja |r| ({l})", "wartosc": v}
                    for z_, acf in (("SPY", acf_s), ("BTC", acf_b)) for l, v in acf.items()]).to_csv(
        KATALOG / "wlasciwosci.csv", index=False, float_format="%.5g")
    pd.concat([ps["dobowy"].rename("SPY dobowy").reset_index(drop=False).rename(columns={"index": "pora"}),
               pb["dobowy"].rename("BTC dobowy").reset_index().rename(columns={"index": "pora"}),
               pb["tygodniowy"].rename("BTC tygodniowy").reset_index().rename(columns={"index": "pora"})],
              ignore_index=True).to_csv(KATALOG / "profile.csv", index=False, float_format="%.4g")
    pd.DataFrame(anomalie(spy, True, "SPY") + anomalie(btc, False, "BTC")).to_csv(
        KATALOG / "anomalie.csv", index=False, float_format="%.4g")
    wykresy(spy, btc, nyc, ps, pb, acf_s, acf_b, zs, zb, brak_spy, przerwy)
    for z_, d in (("SPY", js), ("BTC", jb), ("NYC", jn), ("SPY", ws), ("BTC", wb), ("NYC", wn)):
        print(z_, d)
    print("Zapisano", KATALOG, "i wykresy dane_*.png")


if __name__ == "__main__":
    main()
