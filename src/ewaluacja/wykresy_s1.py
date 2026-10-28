"""Wykresy i tabele podsumowujące eksperyment S1 (wyniki z src/ewaluacja/eksperyment_s1.py).

Uruchomienie (z katalogu głównego repozytorium, po eksperymencie):
    python -m src.ewaluacja.wykresy_s1               # wyniki z wyniki/s1/
    python -m src.ewaluacja.wykresy_s1 --szybko      # wyniki z wyniki/s1_szybki/

Tworzy:
  wyniki/wykresy/s1_przyklad.png              - przykładowy szereg i wynik każdej metody z progiem alarmu
  wyniki/wykresy/s1_czulosc_intensywnosc.png  - odsetek wykrytych anomalii vs intensywność (rodzaj A, B, C × długość L)
  wyniki/wykresy/s1_auc_intensywnosc.png      - AUC vs intensywność (rodzaj × L)
  wyniki/wykresy/s1_uczenie.png               - metody ML uczone na tym samym szeregu vs na szeregach bez anomalii
  wyniki/wykresy/s1_czulosc_dlugosc.png       - odsetek wykrytych vs długość anomalii L
  wyniki/wykresy/s1_okno.png                  - wpływ długości okna w
  wyniki/wykresy/s1_szum.png                  - wpływ poziomu szumu σ
  wyniki/wykresy/s1_ablacja_zaleznosc.png     - analiza dodatkowa: S1-C dla metod patrzących tylko na zależność
  wyniki/s1/podsumowanie.md                   - tabele liczbowe (wszystkie wartości użyte we wnioskach)
Przedziały ufności: 95%, Wilsona dla odsetków, ±1,96·SE dla średnich AUC.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import WYKRESY, WYNIKI
from src.ewaluacja.warianty_metod import liczba_cech, rozwin, styl, uczony

# Stały kolor każdej RODZINY metod na wszystkich wykresach (paleta kategoryczna, kolejność stała); warianty:
# linia ciągła = 5 cech, przerywana = 3 cechy, kropkowana = uczony na szeregach bez anomalii (warianty_metod.py)
KOLORY = {
    "CUSUM (poziom, zmienność, zależność)": "#2a78d6",
    "PELT": "#eb6834",
    "Matrix Profile": "#1baf7a",
    "cechy okien + kNN": "#eda100",
    "Isolation Forest": "#e87ba4",
    "LOF": "#008300",
    "Matrix Profile z normalizacją": "#8a8985",
}
ZNACZNIKI = {"CUSUM (poziom, zmienność, zależność)": "o", "PELT": "s", "Matrix Profile": "^", "cechy okien + kNN": "D",
             "Isolation Forest": "v", "LOF": "P", "Matrix Profile z normalizacją": "x"}
POROWNAWCZE = ("Matrix Profile z normalizacją",)
KOLEJNOSC = rozwin(list(KOLORY), {"cechy okien + kNN", "Isolation Forest", "LOF"}, {"Isolation Forest", "LOF"})
KOLEJNOSC_GL = [m for m in KOLEJNOSC if not uczony(m)]          # wykresy główne - bez wariantów uczonych
ML_S1 = ["Isolation Forest", "LOF"]
RODZAJ = {1: "S1-A: zmiana poziomu", 2: "S1-B: zmiana zmienności", 3: "S1-C: pojawienie się zależności"}
OS_X = {1: "Δ (w jednostkach σ)", 2: "c (mnożnik odch. std)", 3: "φ (siła zależności)"}
TEKST, TEKST2, SIATKA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": TEKST2, "axes.labelcolor": TEKST, "xtick.color": TEKST2,
                     "ytick.color": TEKST2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": SIATKA, "grid.linewidth": 0.6, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160})


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def agreguj(df, grupy):
    def f(g):
        k, n = int(g["wykryta"].sum()), len(g)
        lo, hi = wilson(k, n)
        return pd.Series({"czulosc": k / n, "czulosc_lo": lo, "czulosc_hi": hi, "n": n,
                          "auc": g["auc"].mean(), "auc_se": g["auc"].std(ddof=1) / np.sqrt(n), "ap": g["ap"].mean()})
    return df.groupby(grupy).apply(f, include_groups=False).reset_index()


def linie_metod(ax, t, x, y, lo, hi, metody, kolory, znaczniki, porownawcze=(), pasma=True):
    """Wspólne dla S1–S3: jedna linia na metodę (styl wariantu - warianty_metod.styl)."""
    for m in metody:
        d = t[t.metoda == m].sort_values(x)
        if d.empty:
            continue
        st = styl(m, kolory, znaczniki, porownawcze)
        ax.plot(d[x], d[y], ms=4.5, label=m + (" (porównawczo)" if m in porownawcze else ""), **st)
        if lo is not None and pasma and liczba_cech(m) != 3 and not uczony(m):
            ax.fill_between(d[x], d[lo], d[hi], color=st["color"], alpha=0.08, lw=0)


def _linie(ax, t, x, y, lo=None, hi=None, metody=None):
    linie_metod(ax, t, x, y, lo, hi, metody or KOLEJNOSC_GL, KOLORY, ZNACZNIKI, POROWNAWCZE)


def _legenda(fig, axes, ncol=4):
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=8.3, bbox_to_anchor=(0.5, -0.02))


def wykres_siatka_L(an, miara, x, Ls, rodzaje, os_x, etykiety_x, linie, plik, tytul, opis_L=None, ncol=4):
    """Wspólne dla S1–S3: odsetek wykrytych (albo AUC) a intensywność - wiersz = długość anomalii L,
    kolumna = rodzaj anomalii (ustalenie z konsultacji: wyniki osobno dla każdego L, nie średnio po L)."""
    t = agreguj(an, ["metoda", "rodzaj", "L", x])
    fig, axes = plt.subplots(len(Ls), 3, figsize=(12.5, 3.0 * len(Ls) + 0.9), sharey=True, squeeze=False)
    for i, L in enumerate(Ls):
        for j, r in enumerate((1, 2, 3)):
            ax = axes[i, j]
            d = t[(t.rodzaj == r) & (t.L == L)]
            if miara == "czulosc":
                linie(ax, d, x, "czulosc", "czulosc_lo", "czulosc_hi")
                ax.set_ylim(-0.03, 1.03)
            else:
                d = d.assign(lo=d.auc - 1.96 * d.auc_se, hi=d.auc + 1.96 * d.auc_se)
                linie(ax, d, x, "auc", "lo", "hi")
                ax.axhline(0.5, color=TEKST2, lw=0.8, ls=":")
                ax.set_ylim(0.3, 1.02)
            w = sorted(d[x].unique())
            ax.set_xticks(w)
            ax.set_xticklabels(etykiety_x(r, w))
            if i == 0:
                ax.set_title(rodzaje[r], fontsize=10, color=TEKST, loc="left")
            if i == len(Ls) - 1:
                ax.set_xlabel(os_x[r])
            if j == 0:
                ax.set_ylabel(f"{(opis_L or {}).get(L, f'L = {L}')}\n" +
                              ("odsetek wykrytych" if miara == "czulosc" else "AUC"))
    fig.suptitle(tytul, x=0.01, ha="left", fontsize=11, color=TEKST)
    _legenda(fig, axes, ncol)
    fig.tight_layout(rect=(0, 0.075 if len(Ls) > 1 else 0.1, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_uczenie(an, rodziny, Ls, rodzaje, kolory, znaczniki, plik, tytul, ae=None, etyk_L=None):
    """Wspólne dla S1–S3: metody ML uczone na tym samym szeregu (linia ciągła) i na szeregach bez anomalii
    (kropkowana); wiersz = zestaw cech (5 / 3), x = długość anomalii L, średnio po poziomach."""
    t = agreguj(an, ["metoda", "rodzaj", "L"])
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.9), sharey=True, sharex=True)
    for i, c in enumerate(("5 cech", "3 cechy")):
        for j, r in enumerate((1, 2, 3)):
            ax = axes[i, j]
            metody = [f"{b} ({c})" for b in rodziny] + [f"{b} ({c}, uczony na szeregach bez anomalii)" for b in rodziny]
            if ae:
                metody = [ae, f"{ae} (uczony na szeregach bez anomalii)"] + metody
            for m in metody:
                d = t[(t.metoda == m) & (t.rodzaj == r)].sort_values("L")
                st = styl(m, kolory, znaczniki)
                st["mfc"] = st["color"]
                st["ls"] = ":" if uczony(m) else "-"
                ax.plot(d.L, d.czulosc, ms=4.5, label=m.replace(f" ({c})", "").replace(f" ({c}, ", " ("), **st)
            ax.set_xticks(Ls)
            ax.set_xticklabels(etyk_L or [str(L) for L in Ls])
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(rodzaje[r], fontsize=10, color=TEKST, loc="left")
            else:
                ax.set_xlabel("długość anomalii L")
            if j == 0:
                ax.set_ylabel(f"{c}\nodsetek wykrytych")
    fig.suptitle(tytul, x=0.01, ha="left", fontsize=11, color=TEKST)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_dlugosc(an, plik):
    t = agreguj(an, ["metoda", "rodzaj", "L"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    for ax, r in zip(axes, (1, 2, 3)):
        _linie(ax, t[t.rodzaj == r], "L", "czulosc", "czulosc_lo", "czulosc_hi")
        ax.set_title(RODZAJ[r], fontsize=10, color=TEKST, loc="left")
        ax.set_xlabel("długość anomalii L (punkty)")
        ax.set_xticks([20, 50, 100])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("odsetek wykrytych anomalii")
    fig.suptitle("S1 – wykrywalność a długość anomalii (średnio po 4 poziomach intensywności)", x=0.01, ha="left",
                 fontsize=11, color=TEKST)
    _legenda(fig, axes)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_parametr(an, kol, etyk_x, plik, tytul, xlog=False):
    t = agreguj(an, ["metoda", "rodzaj", kol])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    for ax, r in zip(axes, (1, 2, 3)):
        _linie(ax, t[t.rodzaj == r], kol, "czulosc", "czulosc_lo", "czulosc_hi")
        ax.set_title(RODZAJ[r], fontsize=10, color=TEKST, loc="left")
        ax.set_xlabel(etyk_x)
        if xlog:
            ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(t[kol].unique()))
        ax.set_xticklabels([f"{v:g}".replace(".", ",") for v in sorted(t[kol].unique())])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("odsetek wykrytych anomalii")
    fig.suptitle(tytul, x=0.01, ha="left", fontsize=11, color=TEKST)
    _legenda(fig, axes)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_przyklad(plik, katalog):
    """Jeden szereg (poziom 3, L = 50) i wyniki metod z progami z kalibracji części głównej."""
    from src.ewaluacja.eksperyment_s1 import ZIARNO_KAL
    from src.metody import metody_s1
    from src.symulacje.anomalie import szereg_s1

    kal = pd.read_csv(katalog / "glowny_kalibracja.csv").set_index("metoda")
    s = szereg_s1(310_007, poziom=3, L=50)
    czyste = [szereg_s1(ZIARNO_KAL + i, z_anomaliami=False).x for i in range(200)]
    metody = [m for m in metody_s1() if liczba_cech(m.nazwa) != 3]      # warianty 3-cechowe - w tabelach
    fig, axes = plt.subplots(len(metody) + 1, 1, figsize=(12.5, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1.6] + [1] * len(metody)})
    kol_an = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

    def zaznacz(ax, z_opisem=False):
        for a in s.anomalie:
            ax.axvspan(a.start, a.koniec, color=kol_an[a.rodzaj], alpha=0.13, lw=0,
                       label=RODZAJ[a.rodzaj] if z_opisem else None)

    axes[0].plot(s.x, lw=0.55, color=TEKST2)
    zaznacz(axes[0], True)
    axes[0].set_title("Szereg S1 (poziom intensywności 3: Δ = 2σ, c = 3, φ = 0,7; L = 50) i wyniki metod; "
                      "linia przerywana = próg alarmu", loc="left", fontsize=10, color=TEKST)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    for ax, m in zip(axes[1:], metody):
        if m.nazwa in ("CUSUM (poziom, zmienność, zależność)", "PELT"):
            m.kalibruj(czyste)                      # te same szeregi co w eksperymencie → te same stałe
        w = m.wynik(s.x)
        zaznacz(ax)
        ax.plot(w, color=styl(m.nazwa, KOLORY, ZNACZNIKI)["color"], lw=1.1)
        ax.axhline(kal.loc[m.nazwa, "prog"], color=TEKST, lw=0.9, ls="--")
        ax.set_ylabel(m.nazwa.replace(" z normalizacją", "\nz normalizacją").replace(" (", "\n("),
                      rotation=0, ha="right", va="center", fontsize=8.5)
        ax.set_yticks([])
    axes[-1].set_xlabel("indeks punktu")
    fig.tight_layout()
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


ABLACJA = {  # wariant: (kolor jak metoda bazowa, styl linii)
    "CUSUM (poziom, zmienność, zależność)": ("#2a78d6", "-", "o"),
    "CUSUM: tylko karta zależności": ("#2a78d6", "--", "o"),
    "cechy okien + kNN (5 cech)": ("#eda100", "-", "D"),
    "kNN: średnia, odch. std, autokorelacja": ("#eda100", "-.", "D"),
    "kNN: tylko autokorelacja": ("#eda100", ":", "D"),
}


def wykres_ablacja(ab, plik):
    t = agreguj(ab[ab.rodzaj == 3], ["metoda", "wartosc"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m, (kol, ls, mk) in ABLACJA.items():
        d = t[t.metoda == m].sort_values("wartosc")
        axes[0].plot(d.wartosc, d.czulosc, color=kol, ls=ls, marker=mk, ms=5, lw=2, label=m)
        axes[0].fill_between(d.wartosc, d.czulosc_lo, d.czulosc_hi, color=kol, alpha=0.08, lw=0)
        axes[1].plot(d.wartosc, d.auc, color=kol, ls=ls, marker=mk, ms=5, lw=2, label=m)
    axes[0].set_ylabel("odsetek wykrytych anomalii S1-C")
    axes[1].set_ylabel("AUC dla S1-C")
    axes[1].axhline(0.5, color=TEKST2, lw=0.8, ls=":")
    for ax in axes:
        ax.set_xlabel("φ (siła zależności)")
        ax.set_xticks(sorted(t.wartosc.unique()))
    axes[0].set_ylim(-0.03, 1.03)
    fig.suptitle("S1-C – czy metody wykrywają zależność lepiej, gdy patrzą TYLKO na nią? (L = 50; linia ciągła = "
                 "wersja z eksperymentu głównego)", x=0.01, ha="left", fontsize=10.5, color=TEKST)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def _klucz_naturalny(c):
    """Sortowanie kolumn z liczbami po wartości, nie alfabetycznie (L20 < L50 < L100)."""
    import re
    return [float(t) if re.fullmatch(r"\d+(\.\d+)?", t) else t for t in re.split(r"(\d+(?:\.\d+)?)", str(c))]


def _tab(df, kol_wier, kol_kol, wart, fmt="{:.0%}", kolejnosc=None):
    kol = kolejnosc or KOLEJNOSC
    p = df.pivot_table(index=kol_wier, columns=kol_kol, values=wart).reindex([m for m in kol if m in df[kol_wier].unique()])
    p = p[sorted(p.columns, key=_klucz_naturalny)]
    naglowek = "| " + " | ".join([kol_wier] + [f"{c}" for c in p.columns]) + " |"
    linie = [naglowek, "|" + "---|" * (len(p.columns) + 1)]
    for m, w in p.iterrows():
        linie.append("| " + " | ".join([m] + [fmt.format(v).replace(".", ",") if pd.notna(v) else "–" for v in w]) + " |")
    return "\n".join(linie)


def podsumowanie(katalog):
    an = pd.read_csv(katalog / "glowny_anomalie.csv")
    sz = pd.read_csv(katalog / "glowny_szeregi.csv")
    cz = pd.read_csv(katalog / "glowny_czyste.csv")
    kal = pd.read_csv(katalog / "glowny_kalibracja.csv")
    out = ["# S1 – tabele wyników (generowane automatycznie przez src/ewaluacja/wykresy_s1.py)", ""]
    out += [f"Część główna: {an.ziarno.nunique()} szeregów z anomaliami ({an.groupby('metoda').size().iloc[0]} anomalii "
            f"na metodę), {cz.ziarno.nunique()} szeregów walidacyjnych bez anomalii. Próg: fałszywy alarm w 10% "
            "szeregów kalibracyjnych bez anomalii.", ""]

    out += ["## Kalibracja", "", kal[["metoda", "prog", "kara_pelt", "skale_cusum"]].to_markdown(index=False), ""]

    fa = cz.groupby("metoda").agg(odsetek_z_fa=("fa_epizody", lambda v: (v > 0).mean()),
                                  fa_na_1000_pkt=("fa_epizody", lambda v: v.mean() / 3)).reset_index()
    fa2 = sz.groupby("metoda").agg(fa_w_szeregach_z_anomaliami=("fa_epizody", "mean"),
                                   precyzja=("precyzja", "mean"), czulosc_pkt=("czulosc_pkt", "mean"),
                                   f1=("f1", "mean")).reset_index()
    t = fa.merge(fa2, on="metoda")
    t["k"] = t.metoda.map({m: i for i, m in enumerate(KOLEJNOSC)})
    out += ["## Fałszywe alarmy (walidacja na nowych szeregach bez anomalii) i miary punktowe", "",
            t.sort_values("k").drop(columns="k").round(3).to_markdown(index=False), ""]

    an["rodzaj_kr"] = an.rodzaj.map({1: "A", 2: "B", 3: "C"})
    an["poziom"] = an.groupby("rodzaj").wartosc.rank(method="dense").astype(int)
    k = an.rodzaj_kr + an.poziom.astype(str)
    out += ["Poziomy: A Δ = 0,5 / 1 / 2 / 3 σ; B c = 1,5 / 2 / 3 / 4; C φ = 0,3 / 0,5 / 0,7 / 0,9.", ""]
    for L in (20, 50, 100):
        d = an[an.L == L]
        out += [f"## Odsetek wykrytych anomalii – rodzaj × poziom intensywności, L = {L}", "",
                _tab(d.assign(kol=k[an.L == L]), "metoda", "kol", "wykryta"), "",
                f"## AUC – rodzaj × poziom intensywności, L = {L}", "",
                _tab(d.assign(kol=k[an.L == L]), "metoda", "kol", "auc", "{:.2f}"), ""]
    out += ["## Odsetek wykrytych anomalii – rodzaj × poziom intensywności (średnio po L)", "",
            _tab(an.assign(kol=k), "metoda", "kol", "wykryta"), ""]
    out += ["## AUC – rodzaj × poziom intensywności (średnio po L)", "",
            _tab(an.assign(kol=k), "metoda", "kol", "auc", "{:.2f}"), ""]
    out += ["## Odsetek wykrytych – rodzaj × długość L (średnio po poziomach)", "",
            _tab(an.assign(kol=an.rodzaj_kr + "-L" + an.L.astype(str)), "metoda", "kol", "wykryta"), ""]
    out += ["## Średnio dla rodzaju (wszystkie poziomy i L)", "",
            _tab(an, "metoda", "rodzaj_kr", "wykryta"), "", _tab(an, "metoda", "rodzaj_kr", "auc", "{:.2f}"), "",
            "## AP (PR-AUC) średnio dla rodzaju", "", _tab(an, "metoda", "rodzaj_kr", "ap", "{:.2f}"), ""]

    if (katalog / "okno_anomalie.csv").exists():
        ok = pd.read_csv(katalog / "okno_anomalie.csv")
        ok["rodzaj_kr"] = ok.rodzaj.map({1: "A", 2: "B", 3: "C"})
        out += ["## Długość okna w (poziomy 2 i 3, wszystkie L)", "",
                _tab(ok.assign(kol=ok.rodzaj_kr + "-w" + ok.w.astype(str)), "metoda", "kol", "wykryta"), "",
                _tab(ok.assign(kol=ok.rodzaj_kr + "-w" + ok.w.astype(str) + "-L" + ok.L.astype(str)),
                     "metoda", "kol", "wykryta"), ""]
    if (katalog / "szum_anomalie.csv").exists():
        su = pd.read_csv(katalog / "szum_anomalie.csv")
        su["rodzaj_kr"] = su.rodzaj.map({1: "A", 2: "B", 3: "C"})
        out += ["## Poziom szumu σ (Δ = 1 bezwzględnie, c = 2, φ = 0,5, L = 50)", "",
                _tab(su.assign(kol=su.rodzaj_kr + "-σ" + su.sigma.astype(str)), "metoda", "kol", "wykryta"), "",
                _tab(su.assign(kol=su.rodzaj_kr + "-σ" + su.sigma.astype(str)), "metoda", "kol", "auc", "{:.2f}"), ""]
        czs = pd.read_csv(katalog / "szum_czyste.csv")
        out += ["Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem) dla każdego σ:", "",
                _tab(czs.assign(fa=czs.fa_epizody > 0), "metoda", "sigma", "fa"), ""]
    if (katalog / "ablacja_anomalie.csv").exists():
        ab = pd.read_csv(katalog / "ablacja_anomalie.csv")
        ab["rodzaj_kr"] = ab.rodzaj.map({1: "A", 2: "B", 3: "C"})
        ab["poziom"] = ab.groupby("rodzaj").wartosc.rank(method="dense").astype(int)
        k = ab.rodzaj_kr + ab.poziom.astype(str)
        out += ["## Analiza dodatkowa: metody patrzące tylko na zależność (L = 50)", "",
                _tab(ab.assign(kol=k), "metoda", "kol", "wykryta", kolejnosc=list(ABLACJA)), "",
                _tab(ab.assign(kol=k), "metoda", "kol", "auc", "{:.2f}", kolejnosc=list(ABLACJA)), ""]
        cza = pd.read_csv(katalog / "ablacja_czyste.csv")
        out += ["Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem):", "",
                cza.groupby("metoda").fa_epizody.apply(lambda v: (v > 0).mean()).round(3).to_frame("odsetek").to_markdown(), ""]
    (katalog / "podsumowanie.md").write_text("\n".join(out), encoding="utf-8")


def main(katalog):
    WYKRESY.mkdir(parents=True, exist_ok=True)
    an = pd.read_csv(katalog / "glowny_anomalie.csv")
    etyk = lambda r, w: [f"{v:g}".replace(".", ",") for v in w]
    wykres_siatka_L(an, "czulosc", "wartosc", (20, 50, 100), RODZAJ, OS_X, etyk, _linie, "s1_czulosc_intensywnosc.png",
                    "S1 – odsetek wykrytych anomalii a intensywność, osobno dla każdej długości L "
                    "(fałszywy alarm w 10% szeregów bez anomalii)")
    wykres_siatka_L(an, "auc", "wartosc", (20, 50, 100), RODZAJ, OS_X, etyk, _linie, "s1_auc_intensywnosc.png",
                    "S1 – AUC (miara niezależna od progu) a intensywność anomalii, osobno dla każdej długości L")
    wykres_uczenie(an, ML_S1, [20, 50, 100], RODZAJ, KOLORY, ZNACZNIKI, "s1_uczenie.png",
                   "S1 – metody ML: uczenie na tym samym szeregu (linia ciągła) i na 20 szeregach bez anomalii "
                   "(kropkowana); średnio po poziomach")
    wykres_dlugosc(an, "s1_czulosc_dlugosc.png")
    if (katalog / "okno_anomalie.csv").exists():
        wykres_parametr(pd.read_csv(katalog / "okno_anomalie.csv"), "w", "długość okna w (punkty)", "s1_okno.png",
                        "S1 – wpływ długości okna w na metody okienkowe (poziomy 2–3, L = 20/50/100 łącznie)")
    if (katalog / "szum_anomalie.csv").exists():
        wykres_parametr(pd.read_csv(katalog / "szum_anomalie.csv"), "sigma", "poziom szumu σ", "s1_szum.png",
                        "S1 – wpływ poziomu szumu σ przy stałej wielkości anomalii (Δ = 1, c = 2, φ = 0,5; L = 50)",
                        xlog=True)
    if (katalog / "ablacja_anomalie.csv").exists():
        wykres_ablacja(pd.read_csv(katalog / "ablacja_anomalie.csv"), "s1_ablacja_zaleznosc.png")
    wykres_przyklad("s1_przyklad.png", katalog)
    podsumowanie(katalog)
    print("Zapisano wykresy w", WYKRESY, "i", katalog / "podsumowanie.md")


if __name__ == "__main__":
    main(WYNIKI / ("s1_szybki" if "--szybko" in sys.argv else "s1"))
