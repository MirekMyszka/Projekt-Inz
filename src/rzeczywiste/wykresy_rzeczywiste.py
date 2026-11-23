"""Etap 2 - wykresy i tabele dla danych rzeczywistych (po nyc_taxi.py, rynki.py, studia.py).

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.rzeczywiste.wykresy_rzeczywiste

Tworzy:
  wyniki/wykresy/rz_nyc_szereg.png    - NYC Taxi: szereg, okna NAB, święta nieoznaczone i wynik czterech metod z progiem
  wyniki/wykresy/rz_nyc_auc.png       - NYC Taxi: AUC każdej metody dla każdego okna NAB
  wyniki/wykresy/rz_rynki_szeregi.png - SPY i BTC: dzienny log RV i wolumen, zdarzenia katalogu, liczba metod z alarmem
  wyniki/wykresy/rz_rynki_recall.png  - SPY i BTC: recall@2% według typu zdarzenia
  wyniki/wykresy/rz_studia.png        - studia przypadków 5-min: |r| po usunięciu sezonowości dobowej
  wyniki/rzeczywiste/podsumowanie.md  - wszystkie tabele
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import WYKRESY
from src.ewaluacja.wykresy_porownanie import tabela
from src.ewaluacja.wykresy_s1 import SIATKA, TEKST, TEKST2
from src.rzeczywiste import nyc_taxi as nyc
from src.rzeczywiste.katalog import OPIS_TYPU, STUDIA, katalog
from src.rzeczywiste.rynki import KONIEC_KAL
from src.rzeczywiste.studia import zwroty_bez_sezonowosci

K = nyc.KATALOG
KOL_ZD = {"A": "#e87ba4", "B": "#8a5cc2", "C": "#eb6834", "D": "#2a78d6", "E": "#1baf7a", "F": "#eda100"}
NYC_POKAZ = ["STL + EWMA", "Matrix Profile bez normalizacji", "cechy cykli + kNN (5 cech)",
             f"Isolation Forest (5 cech, {nyc.UCZ})"]


def pct(v):
    return "–" if pd.isna(v) else f"{round(100 * v)}%"


def f2(v):
    return "–" if pd.isna(v) else f"{v:.2f}".replace(".", ",")


def kolejnosc_nyc(nazwy):
    pierwsze = ["STL + EWMA", "PELT (reszty STL)", "Matrix Profile z normalizacją", "cechy cykli + kNN (5 cech)",
                "cechy cykli + kNN (3 cechy)", "Isolation Forest (5 cech)", "Isolation Forest (3 cechy)", "autoenkoder"]
    return [n for n in pierwsze if n in nazwy] + [n for n in nazwy if n not in pierwsze]


# ------------------------------------------------------------------------------------------------ NYC Taxi
def wykres_nyc():
    df, y, dzien = nyc.wczytaj()
    w = pd.read_csv(K / "nyc_taxi_wyniki_punktow.csv", index_col=0, parse_dates=True)
    met = pd.read_csv(K / "nyc_taxi_metody.csv")
    okno, strefa, kal = nyc.maski(df)
    t = df.index
    fig, axes = plt.subplots(1 + len(NYC_POKAZ), 1, figsize=(12.5, 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.6] + [1] * len(NYC_POKAZ)})

    def tlo(ax, opis=False):
        for i, z in enumerate(nyc.ZDARZENIA, 1):
            idx = np.flatnonzero(okno == i)
            ax.axvspan(t[idx[0]], t[idx[-1]], color="#eb6834", alpha=0.16, lw=0, label="okno NAB" if opis and i == 1 else None)
        for j, (d, nz) in enumerate(nyc.SWIETA.items()):
            ax.axvspan(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1), color="#8a5cc2", alpha=0.18, lw=0,
                       label="święto nieoznaczone w NAB" if opis and j == 0 else None)
        ax.axvline(t[nyc.DNI_KAL * nyc.DOBA], color=TEKST2, lw=0.9, ls="--")

    axes[0].plot(t, df["pasazerowie"] / 1000, lw=0.35, color=TEKST2)
    tlo(axes[0], True)
    axes[0].set_ylabel("pasażerowie\n(tys. / 30 min)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[0].set_title("NYC Taxi – szereg, okna NAB i wynik metod; linia przerywana pionowa – koniec okresu kalibracyjnego, "
                      "pozioma – próg (alarm w 10% dni kalibracji)", loc="left", fontsize=10, color=TEKST)
    for ax, m in zip(axes[1:], NYC_POKAZ):
        tlo(ax)
        ax.plot(t, w[m], lw=0.6, color="#2a78d6")
        prog = met[(met.metoda == m) & (met.wariant == "kalibracja") & (met.alfa == 0.10)].prog.iloc[0]
        ax.axhline(prog, color=TEKST, lw=0.9, ls="--")
        ax.set_ylabel(m.replace(" (5 cech, uczony", "\n(5 cech, uczony").replace(" bez normalizacji", "\nbez normalizacji")
                      .replace(" (5 cech)", "\n(5 cech)"), rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([])
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(WYKRESY / "rz_nyc_szereg.png", bbox_inches="tight")
    plt.close(fig)

    o = pd.read_csv(K / "nyc_taxi_okna.csv")
    o = o[(o.wariant == "kalibracja") & (o.alfa == 0.10)]
    nazwy = kolejnosc_nyc(list(dict.fromkeys(o.metoda)))
    M = np.array([[o[(o.metoda == m) & (o.zdarzenie == z)].auc.iloc[0] for z in nyc.ZDARZENIA] for m in nazwy])
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(nazwy) + 1.4))
    ax.imshow(M, cmap="Blues", vmin=0.4, vmax=1, aspect="auto")
    for i in range(len(nazwy)):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] > 0.82 else TEKST)
    ax.set_xticks(range(5))
    ax.set_xticklabels(nyc.ZDARZENIA, fontsize=8.5)
    ax.set_yticks(range(len(nazwy)))
    ax.set_yticklabels(nazwy, fontsize=8.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("NYC Taxi – AUC dla każdego okna NAB (punkty okna wobec punktów normy okresu oceny)", loc="left",
                 fontsize=10, color=TEKST)
    fig.tight_layout()
    fig.savefig(WYKRESY / "rz_nyc_auc.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ SPY, BTC
def wykres_rynki():
    kat = katalog()
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11), gridspec_kw={"height_ratios": [1.2, 0.6, 1.2, 0.6]})
    for j, (klucz, nazwa) in enumerate((("spy", "SPY"), ("btc", "BTC"))):
        d = pd.read_csv(K / f"{klucz}_dni.csv", index_col=0, parse_dates=True)
        ax, ax2 = axes[2 * j], axes[2 * j + 1]
        ax.plot(d.index, d.log_rv, lw=0.5, color=TEKST2)
        for _, e in kat[(kat.zbior == nazwa) & kat.typ.isin(list("ABCDF"))].iterrows():
            ax.axvspan(pd.Timestamp(e.od), pd.Timestamp(e["do"]) + pd.Timedelta(days=1), color=KOL_ZD[e.typ],
                       alpha=0.25 if e.typ == "D" else 0.8, lw=0 if e.typ == "D" else 1.2)
        ax.axvline(pd.Timestamp(KONIEC_KAL), color=TEKST2, lw=0.9, ls="--")
        ax.set_ylabel(f"{nazwa}\nlog RV dzienna")
        ax.set_title(f"{nazwa} – dzienna zrealizowana zmienność (log), zdarzenia katalogu (kolor = typ) i liczba metod, "
                     "dla których dzień jest w 2% najwyższych", loc="left", fontsize=9.5, color=TEKST)
        ax2.bar(d.index, d["zmiennosc: liczba metod w top 2%"], width=2, color="#2a78d6", label="zmienność")
        ax2.bar(d.index, -d["wolumen: liczba metod w top 2%"], width=2, color="#eda100", label="wolumen (w dół)")
        ax2.axhline(0, color=TEKST2, lw=0.5)
        ax2.set_ylabel("liczba metod")
        ax2.legend(loc="upper right", fontsize=7.5, ncol=2)
        for a in (ax, ax2):
            a.xaxis.set_major_locator(mdates.YearLocator())
            a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    h = [plt.Rectangle((0, 0), 1, 1, color=KOL_ZD[t], alpha=0.8) for t in "ABCDF"]
    fig.legend(h, [f"{t} – {OPIS_TYPU[t]}" for t in "ABCDF"], loc="lower center", ncol=5, fontsize=8.3,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(WYKRESY / "rz_rynki_szeregi.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True)
    for ax, (klucz, nazwa) in zip(axes, (("spy", "SPY"), ("btc", "BTC"))):
        z = pd.read_csv(K / f"{klucz}_zdarzenia.csv")
        z = z[z.miara == "top_0.02"]
        nazwy = list(dict.fromkeys(z.metoda))
        typy = [t for t in "ABCDF" if t in set(z.typ)]
        M = np.array([[z[(z.metoda == m) & (z.typ == t)].wykryte.mean() for t in typy] for m in nazwy])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(nazwy)):
            for k_ in range(len(typy)):
                n = len(z[(z.metoda == nazwy[i]) & (z.typ == typy[k_])])
                ax.text(k_, i, f"{round(100 * M[i, k_])}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, k_] > 0.6 else TEKST)
        ax.set_xticks(range(len(typy)))
        ax.set_xticklabels([f"{t}\n{OPIS_TYPU[t]}\n(n = {len(z[(z.metoda == nazwy[0]) & (z.typ == t)])})" for t in typy],
                           fontsize=7.8)
        ax.set_yticks(range(len(nazwy)))
        ax.set_yticklabels(nazwy, fontsize=8)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(nazwa, loc="left", fontsize=10, color=TEKST)
    fig.suptitle("Odsetek zdarzeń katalogu (%), w których leży choć jeden z 2% dni o najwyższym wyniku metody "
                 "(A–D: szereg zmienności, F: wolumenu)", x=0.01, ha="left", fontsize=10, color=TEKST)
    fig.tight_layout()
    fig.savefig(WYKRESY / "rz_rynki_recall.png", bbox_inches="tight")
    plt.close(fig)


def wykres_studia():
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 10.5))
    for j, (klucz, nazwa) in enumerate((("spy", "SPY"), ("btc", "BTC"))):
        z = zwroty_bez_sezonowosci(klucz)
        for i, (opis, od, do) in enumerate(STUDIA):
            ax = axes[i, j]
            w = z[(z.index.date >= pd.Timestamp(od).date()) & (z.index.date <= pd.Timestamp(do).date())]
            x = np.arange(len(w))
            ax.vlines(x, 0, w.z.values, color="#2a78d6", lw=0.7)      # pionowe kreski - przy tysiącach świec
            ax.set_ylim(0, w.z.max() * 1.05)                         # słupki bar() gubią pojedyncze skoki
            ax2 = ax.twinx()
            ax2.plot(x, w.c, color=TEKST2, lw=0.8)
            ax2.set_yticks([])
            dni = pd.Series(w.index.date)
            granice = np.flatnonzero(dni.values[1:] != dni.values[:-1]) + 1
            for g in granice:
                ax.axvline(g, color=SIATKA, lw=0.8)
            srodki = np.r_[0, granice]
            ax.set_xticks(srodki)
            ax.set_xticklabels([pd.Timestamp(dni.iloc[s]).strftime("%d.%m") for s in srodki], fontsize=7)
            ax.set_title(f"{nazwa} – {opis}", loc="left", fontsize=9, color=TEKST)
            if j == 0:
                ax.set_ylabel("|r| / typowe |r|\no tej porze dnia", fontsize=8)
            ax.grid(False)
    fig.suptitle("Studia przypadków (5 min): słupki – |zwrot| względem typowego o tej porze dnia, linia – cena",
                 x=0.01, ha="left", fontsize=10.5, color=TEKST)
    fig.tight_layout()
    fig.savefig(WYKRESY / "rz_studia.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tabele
def podsumowanie():
    out = ["# Etap 2 – dane rzeczywiste: tabele (generowane przez src/rzeczywiste/wykresy_rzeczywiste.py)", ""]
    met = pd.read_csv(K / "nyc_taxi_metody.csv")
    o = pd.read_csv(K / "nyc_taxi_okna.csv")
    nazwy = kolejnosc_nyc(list(dict.fromkeys(met.metoda)))
    for war, opis in (("kalibracja", "próg z okresu kalibracyjnego (alarm w α dni kalibracji)"),
                      ("rowny_budzet", "równy budżet (alarm w α dni bez anomalii okresu oceny)")):
        w = []
        for m in nazwy:
            x = met[(met.metoda == m) & (met.wariant == war)].set_index("alfa")
            w.append([m] + [f"{int(x.loc[a, 'wykryte_okna'])}/5 · {pct(x.loc[a, 'fa_dni_ocena'])}" for a in nyc.ALFY]
                     + [f2(x.iloc[0].auc_wszystkie_okna), f2(x.iloc[0].ap)])
        out += [f"## NYC Taxi – wykryte okna NAB i dni z fałszywym alarmem w okresie oceny; {opis}", "",
                tabela(["metoda"] + [f"α = {round(100 * a)}%" for a in nyc.ALFY] + ["AUC", "AP"], w), ""]
    oo = o[(o.wariant == "kalibracja") & (o.alfa == 0.10)]
    out += ["## NYC Taxi – AUC dla każdego okna", "",
            tabela(["metoda"] + nyc.ZDARZENIA, [[m] + [f2(oo[(oo.metoda == m) & (oo.zdarzenie == z)].auc.iloc[0])
                                                    for z in nyc.ZDARZENIA] for m in nazwy]), ""]
    d = pd.read_csv(K / "nyc_taxi_dni.csv")
    w = []
    for dat, nz in nyc.SWIETA.items():
        x = d[d.data == dat].iloc[0]
        dd = d[d.okres == x.okres]
        w.append([f"{dat} {nz} ({x.okres})"] + [str(int((dd[m] > x[m]).sum() + 1)) for m in nazwy])
    out += ["## NYC Taxi – święta nieoznaczone w NAB: miejsce dnia w rankingu dziennych maksimów wyniku "
            "(w swoim okresie: kalibracja 105 dni, ocena 110 dni)", "", tabela(["dzień"] + nazwy, w), ""]

    for klucz, nazwa in (("spy", "SPY"), ("btc", "BTC")):
        z = pd.read_csv(K / f"{klucz}_zdarzenia.csv")
        m = pd.read_csv(K / f"{klucz}_metody.csv")
        nz = list(dict.fromkeys(z.metoda))
        for szereg, typy in (("zmiennosc", "ABCD"), ("wolumen", "F")):
            zz = z[z.szereg == szereg]
            typy = [t for t in typy if t in set(zz.typ)]
            w = []
            for me in nz:
                kom = [me]
                for t in typy:
                    x = zz[(zz.metoda == me) & (zz.typ == t)]
                    kom.append(" / ".join(pct(x[x.miara == f"top_{N}"].wykryte.mean()) for N in (0.01, 0.02, 0.05)))
                xo = zz[(zz.metoda == me) & (zz.miara == "prog_0.02") & (zz.okres == "ocena")]
                mm = m[(m.metoda == me) & (m.szereg == szereg) & (m.alfa == 0.02)].iloc[0]
                kom += [pct(xo.wykryte.mean()), f"{mm.alarmy_na_rok:.1f}".replace(".", ",")]
                w.append(kom)
            n_typ = {t: len(zz[(zz.metoda == nz[0]) & (zz.typ == t) & (zz.miara == "top_0.01")]) for t in typy}
            out += [f"## {nazwa} – szereg: {szereg}; recall@N (N = 1 / 2 / 5% dni) według typu zdarzenia; wykrycie przy "
                    "progu z kalibracji (α = 2%) w okresie oceny i liczba dni z alarmem na rok", "",
                    tabela(["metoda"] + [f"{t} {OPIS_TYPU[t]} (n = {n_typ[t]})" for t in typy] +
                           ["wykryte w okresie oceny (próg α = 2%)", "dni z alarmem na rok (ocena)"], w), ""]
        g = pd.read_csv(K / f"{klucz}_zgodnosc.csv")
        g = g[(~g.metoda_a.str.startswith("CUSUM z S1")) & (~g.metoda_b.str.startswith("CUSUM z S1"))]
        out += [f"## {nazwa} – zgodność metod (indeks Jaccarda zbiorów 2% najwyższych dni; bez CUSUM z S1)", "",
                tabela(["szereg", "mediana", "min", "max"],
                       [[s, f2(x.jaccard.median()), f2(x.jaccard.min()), f2(x.jaccard.max())] for s, x in g.groupby("szereg")]), ""]
        dni = pd.read_csv(K / f"{klucz}_dni.csv", index_col=0)
        top = dni.sort_values("zmiennosc: liczba metod w top 2%", ascending=False).head(15)
        kat = katalog()
        kat = kat[kat.zbior == nazwa]
        def opis_dnia(d):
            dd = pd.Timestamp(d).date()
            x = kat[(kat.od <= dd) & (kat["do"] >= dd)]
            return "; ".join(f"{r.typ}: {r.opis}" for _, r in x.iterrows()) or "–"
        out += [f"## {nazwa} – dni wskazane przez najwięcej metod (szereg zmienności) i zdarzenia katalogu", "",
                tabela(["dzień", "liczba metod (z 13)", "log RV", "zdarzenie w katalogu"],
                       [[d, int(r["zmiennosc: liczba metod w top 2%"]), f2(r.log_rv), opis_dnia(d)] for d, r in top.iterrows()]),
                ""]
    st = pd.read_csv(K / "studia_5min.csv")
    out += ["## Studia przypadków 5-min (z – |r| względem typowego o tej porze dnia)", "",
            tabela(["zbiór", "studium", "dzień", "maks. z", "mediana z", "godzina maks.", "zwrot maks. %", "zwrot dnia %",
                    "metody w top 2%"],
                   [[r.zbior, r.studium, r.data, f"{r.maks_z:.1f}", f"{r.mediana_z:.1f}", r.godzina_maks,
                     f"{r['zwrot_maks_%']:.2f}", f"{r['zwrot_dnia_%']:.2f}", "–" if pd.isna(r.get("metody_top2%")) else
                     int(r["metody_top2%"])] for _, r in st.iterrows()]), ""]
    fo = pd.read_csv(K / "fomc.csv")
    out += ["## Typ E – dni FOMC w SPY (z o 14:00–14:55 i w pozostałej części dnia; zwykłe dni o 14: mediana 1)", "",
            tabela(["dzień", "maks. z 14:00–14:55", "mediana z 14:00–14:55", "mediana z reszta dnia",
                    "metody w top 2%", "dni o wyższym log RV (%)"],
                   [[r.data, f"{r.z_14_maks:.1f}", f"{r.z_14_mediana:.1f}", f"{r.z_reszta_dnia_mediana:.1f}",
                     int(r["metody_top2%"]), f"{r['pozycja_log_rv_%']:.0f}%"] for _, r in fo.iterrows()]), ""]
    (K / "podsumowanie.md").write_text("\n".join(out), encoding="utf-8")


def main():
    WYKRESY.mkdir(parents=True, exist_ok=True)
    wykres_nyc()
    wykres_rynki()
    wykres_studia()
    podsumowanie()
    print("Zapisano wykresy rz_*.png i", K / "podsumowanie.md")


if __name__ == "__main__":
    main()
