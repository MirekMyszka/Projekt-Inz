"""Wykresy i tabele podsumowujące eksperyment S3 (wyniki z src/ewaluacja/eksperyment_s3.py).

Uruchomienie (z katalogu głównego repozytorium, po eksperymencie):
    python -m src.ewaluacja.wykresy_s3               # wyniki z wyniki/s3/
    python -m src.ewaluacja.wykresy_s3 --szybko      # wyniki z wyniki/s3_szybki/

Tworzy:
  wyniki/wykresy/s3_przyklad.png              - przykładowy szereg (szum + sinus) i wynik każdej metody z progiem alarmu
  wyniki/wykresy/s3_czulosc_intensywnosc.png  - odsetek wykrytych anomalii vs poziom intensywności (osobno A, B, C)
  wyniki/wykresy/s3_uczenie.png               - metody ML uczone na tym samym szeregu vs na szeregach bez anomalii
  wyniki/wykresy/s3_auc_intensywnosc.png      - AUC vs poziom intensywności
  wyniki/wykresy/s3_czulosc_dlugosc.png       - odsetek wykrytych vs długość L (1, 2, 4 cykle)
  wyniki/wykresy/s3_okno.png                  - wpływ długości okna w
  wyniki/wykresy/s3_tlo.png                   - wpływ amplitudy cyklu A (stosunek sygnału do szumu; właściwości szeregu)
  wyniki/wykresy/s3_dodatkowa.png             - analizy dodatkowe: model cyklu, odwrotny kierunek, autoenkoder
  wyniki/s3/podsumowanie.md                   - tabele liczbowe (wszystkie wartości użyte we wnioskach)
Przedziały ufności: 95%, Wilsona dla odsetków, ±1,96·SE dla średnich AUC.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import WYKRESY, WYNIKI
from src.ewaluacja.warianty_metod import baza, liczba_cech, rozwin, uczony
from src.ewaluacja.wykresy_s1 import (SIATKA, TEKST, TEKST2, _tab, agreguj, linie_metod,  # noqa: F401  (wspólny styl)
                                      wykres_siatka_L, wykres_uczenie)

# Kolor = rodzina metody (te same miejsca w palecie co w S1 i S2); autoenkoder - nowy, ósmy kolor;
# Matrix Profile bez normalizacji - szary, przerywany (wersja porównawcza, jak w S1 wersja z normalizacją).
KOLORY = {
    "STL + EWMA": "#2a78d6",
    "PELT (reszty STL)": "#eb6834",
    "Matrix Profile z normalizacją": "#1baf7a",
    "cechy cykli + kNN": "#eda100",
    "Isolation Forest": "#e87ba4",
    "autoenkoder": "#e34948",
    "Matrix Profile bez normalizacji": "#8a8985",
}
ZNACZNIKI = dict(zip(KOLORY, ["o", "s", "^", "D", "v", "P", "x"]))
POROWNAWCZA = "Matrix Profile bez normalizacji"
ML_S3 = ["Isolation Forest"]
KOLEJNOSC = rozwin(list(KOLORY), {"cechy cykli + kNN", "Isolation Forest"}, set(ML_S3), ae="autoenkoder")
KOLEJNOSC_GL = [m for m in KOLEJNOSC if not uczony(m)]
RODZAJ = {1: "S3-A: przesunięcie fazy", 2: "S3-B: zmiana amplitudy", 3: "S3-C: zmiana okresu"}
OPIS_POZIOMU = {1: ["π/8", "π/4", "π/2", "π"], 2: ["×0,75", "×0,5", "×0,25", "×0"],
                3: ["×0,9", "×0,75", "×0,6", "×0,5"]}
OS_X = {1: "przesunięcie fazy Δθ", 2: "amplituda cyklu na fragmencie (k·A)", 3: "okres cyklu na fragmencie (r·P)"}
KROTKIE = {"STL + EWMA": "STL+\nEWMA", "PELT (reszty STL)": "PELT\n(STL)", "Matrix Profile z normalizacją": "MP\n(z norm.)",
           "cechy cykli + kNN": "kNN", "Isolation Forest": "IF", "autoenkoder": "AE", POROWNAWCZA: "MP\n(bez norm.)"}


def krotka_m(m):
    return KROTKIE[baza(m)] + ("\n3 cechy" if liczba_cech(m) == 3 else "\n5 cech" if liczba_cech(m) == 5 else "")


def _linie(ax, t, x, y, lo=None, hi=None):
    linie_metod(ax, t, x, y, lo, hi, KOLEJNOSC_GL, KOLORY, ZNACZNIKI, (POROWNAWCZA,))


def _zapisz(fig, axes, plik, tytul):
    fig.suptitle(tytul, x=0.01, ha="left", fontsize=11, color=TEKST)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_parametr(an, kol, etyk_x, plik, tytul, etykiety=None):
    t = agreguj(an, ["metoda", "rodzaj", kol])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    for ax, r in zip(axes, (1, 2, 3)):
        _linie(ax, t[t.rodzaj == r], kol, "czulosc", "czulosc_lo", "czulosc_hi")
        ax.set_title(RODZAJ[r], fontsize=10, color=TEKST, loc="left")
        ax.set_xlabel(etyk_x)
        w = sorted(t[kol].unique())
        ax.set_xticks(w)
        ax.set_xticklabels(etykiety or [f"{v:g}".replace(".", ",") for v in w])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("odsetek wykrytych anomalii")
    _zapisz(fig, axes, plik, tytul)


def wykres_dodatkowa(katalog, plik):
    dod = pd.read_csv(katalog / "dodatkowa_anomalie.csv")
    gl = pd.read_csv(katalog / "glowny_anomalie.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    # (a) model cyklu - odsetek wykrytych vs L dla trzech modeli; STL+EWMA linia ciągła, PELT przerywana
    a = dod[dod.analiza == "a_model_cyklu"]
    modele = [(" [STL, okno sezonowości 7 cykli]", "STL, okno 7 cykli (domyślne)", "#86b6ef"),
              ("", "STL, okno dłuższe niż szereg (główna)", "#2a78d6"),
              (" [model: sinus dopasowany globalnie]", "sinus dopasowany globalnie", "#0d366b")]
    for baza, ls in (("STL + EWMA", "-"), ("PELT (reszty STL)", "--")):
        for suf, opis, kol in modele:
            d = a[a.metoda == baza + suf].groupby("L").wykryta.mean()
            axes[0].plot(d.index, d.values, color=kol, ls=ls, marker="o" if ls == "-" else "s", ms=4.5, lw=1.8,
                         label=f"{'EWMA' if ls == '-' else 'PELT'}: {opis}")
    axes[0].set_xticks([50, 100, 200])
    axes[0].set_xticklabels(["50\n(1 cykl)", "100\n(2 cykle)", "200\n(4 cykle)"])
    axes[0].set_xlabel("długość anomalii L")
    axes[0].set_ylim(0, 1.32)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("odsetek wykrytych anomalii (A, B, C łącznie)")
    axes[0].set_title("(a) Czy model cyklu „wchłania” anomalię? (poziomy 2–3)", loc="left", fontsize=9.5)
    axes[0].legend(fontsize=6.6, ncol=2, loc="upper left")

    # (b) odwrotny kierunek - dla każdego rodzaju: kierunek z części głównej vs odwrotny (poziomy 2-3, L = 100)
    b = dod[dod.analiza == "b_odwrotny_kierunek"]
    g = gl[(gl.L == 100) & gl.poziom.isin([2, 3])]
    xs = np.arange(len(KOLEJNOSC_GL))
    szer = 0.13
    kol_r = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
    opis_r = {1: ("faza +", "faza −"), 2: ("osłabienie", "wzmocnienie"), 3: ("przyspieszenie", "spowolnienie")}
    for j, r in enumerate((1, 2, 3)):
        v1 = [g[(g.metoda == m) & (g.rodzaj == r)].wykryta.mean() for m in KOLEJNOSC_GL]
        v2 = [b[(b.metoda == m) & (b.rodzaj == r)].wykryta.mean() for m in KOLEJNOSC_GL]
        off = (j - 1) * 2 * szer
        axes[1].bar(xs + off - szer / 2, v1, szer, color=kol_r[r], label=f"{RODZAJ[r][:4]} {opis_r[r][0]}")
        axes[1].bar(xs + off + szer / 2, v2, szer, color="white", edgecolor=kol_r[r], hatch="///", lw=1.0,
                    label=f"{RODZAJ[r][:4]} {opis_r[r][1]}")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([krotka_m(m) for m in KOLEJNOSC_GL], fontsize=6.3)
    axes[1].set_ylim(0, 1.32)
    axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1].set_title("(b) Kierunek zmiany (poziomy 2–3, L = 100)", loc="left", fontsize=9.5)
    axes[1].legend(fontsize=6.6, ncol=3, loc="upper left")

    # (c) autoenkoder - sposób uczenia
    c = dod[dod.analiza == "c_autoenkoder"]
    warianty = list(dict.fromkeys(c.metoda))
    xs = np.arange(len(warianty))
    for i, r in enumerate((1, 2, 3)):
        v = [c[(c.metoda == m) & (c.rodzaj == r)].wykryta.mean() for m in warianty]
        auc = [c[(c.metoda == m) & (c.rodzaj == r)].auc.mean() for m in warianty]
        axes[2].bar(xs + (i - 1) * 0.27, v, 0.25, color=["#f4a3a2", "#e34948", "#9e2b2a"][i], label=RODZAJ[r].split(":")[0])
        for xx, vv, aa in zip(xs + (i - 1) * 0.27, v, auc):
            axes[2].text(xx, vv + 0.02, f"{aa:.2f}".replace(".", ","), ha="center", fontsize=6.5, color=TEKST2)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([m.replace("autoenkoder (", "").rstrip(")").replace(" bez anomalii", "\nbez anomalii")
                             .replace(" 10% okien", "\n10% okien (główna)").replace(" okien", "\nokien") for m in warianty],
                            fontsize=7.5)
    axes[2].set_ylim(0, 1.32)
    axes[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[2].set_title("(c) Autoenkoder: na czym się uczy (nad słupkami: AUC)", loc="left", fontsize=9.5)
    axes[2].legend(fontsize=7.5, loc="upper left", ncol=3)
    fig.suptitle("S3 – analizy dodatkowe", x=0.01, ha="left", fontsize=11, color=TEKST)
    fig.tight_layout()
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_przyklad(plik, katalog):
    from src.ewaluacja.eksperyment import ZIARNO_KAL
    from src.metody import metody_s3
    from src.symulacje.anomalie import szereg_s3

    kal = pd.read_csv(katalog / "glowny_kalibracja.csv").set_index("metoda")
    s = szereg_s3(6_100_000 + 200_000 + 10_000 + 7, poziom=2, L=100)
    czyste = [szereg_s3(ZIARNO_KAL + i, z_anomaliami=False).x for i in range(200)]
    metody = [m for m in metody_s3() if liczba_cech(m.nazwa) != 3]      # warianty 3-cechowe - w tabelach
    fig, axes = plt.subplots(len(metody) + 1, 1, figsize=(12.5, 12.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.6] + [1] * len(metody)})
    kol_an = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

    def zaznacz(ax, z_opisem=False):
        for a in s.anomalie:
            ax.axvspan(a.start, a.koniec, color=kol_an[a.rodzaj], alpha=0.15, lw=0,
                       label=RODZAJ[a.rodzaj] if z_opisem else None)

    axes[0].plot(s.x, lw=0.5, color=TEKST2)
    zaznacz(axes[0], True)
    axes[0].set_title("Szereg S3 – szum + sinus, A = 3, P = 50 (poziom 2: faza +π/4, amplituda ×0,5, okres ×0,75; "
                      "L = 100) i wyniki metod; linia przerywana = próg alarmu", loc="left", fontsize=10, color=TEKST)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    for ax, m in zip(axes[1:], metody):
        m.kalibruj(czyste)                     # te same szeregi co w eksperymencie → te same stałe
        w = m.wynik(s.x)
        zaznacz(ax)
        ax.plot(w, color=KOLORY[baza(m.nazwa)], lw=1.1)
        ax.axhline(kal.loc[m.nazwa, "prog"], color=TEKST, lw=0.9, ls="--")
        ax.set_ylabel(m.nazwa.replace(" (", "\n(").replace(" + ", "\n+ ").replace(" z ", "\nz ").replace(" bez ", "\nbez "),
                      rotation=0, ha="right", va="center", fontsize=8.5)
        ax.set_yticks([])
    axes[-1].set_xlabel("indeks punktu")
    fig.tight_layout()
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def _tabele(d, kol, metody, z_auc=True):
    out = [_tab(d.assign(kol=kol), "metoda", "kol", "wykryta", kolejnosc=metody), ""]
    if z_auc:
        out += [_tab(d.assign(kol=kol), "metoda", "kol", "auc", "{:.2f}", kolejnosc=metody), ""]
    return out


def podsumowanie(katalog):
    an = pd.read_csv(katalog / "glowny_anomalie.csv")
    sz = pd.read_csv(katalog / "glowny_szeregi.csv")
    cz = pd.read_csv(katalog / "glowny_czyste.csv")
    kal = pd.read_csv(katalog / "glowny_kalibracja.csv")
    an["rodzaj_kr"] = an.rodzaj.map({1: "A", 2: "B", 3: "C"})
    out = ["# S3 – tabele wyników (generowane automatycznie przez src/ewaluacja/wykresy_s3.py)", "",
           f"Część główna: {an.ziarno.nunique()} szeregów z anomaliami ({an.groupby('metoda').size().iloc[0]} anomalii na "
           f"metodę), {cz.ziarno.nunique()} szeregów walidacyjnych bez anomalii. Próg: fałszywy alarm w 10% szeregów "
           "kalibracyjnych bez anomalii.", "",
           "## Kalibracja", "", kal.assign(k=kal.metoda.map({m: i for i, m in enumerate(KOLEJNOSC)})).sort_values("k")
           [["metoda", "prog", "kara_pelt"]].to_markdown(index=False), ""]
    fa = cz.groupby("metoda").agg(odsetek_z_fa=("fa_epizody", lambda v: (v > 0).mean()),
                                  fa_na_1000_pkt=("fa_epizody", lambda v: v.mean() / 3)).reset_index()
    fa2 = sz.groupby("metoda").agg(fa_w_szeregach_z_anomaliami=("fa_epizody", "mean"), precyzja=("precyzja", "mean"),
                                   czulosc_pkt=("czulosc_pkt", "mean"), f1=("f1", "mean")).reset_index()
    t = fa.merge(fa2, on="metoda")
    t["k"] = t.metoda.map({m: i for i, m in enumerate(KOLEJNOSC)})
    out += ["## Fałszywe alarmy (walidacja na nowych szeregach bez anomalii) i miary punktowe", "",
            t.sort_values("k").drop(columns="k").round(3).to_markdown(index=False), ""]
    k = an.rodzaj_kr + an.poziom.astype(str)
    out += ["Poziomy: A Δθ = π/8 / π/4 / π/2 / π; B k = 0,75 / 0,5 / 0,25 / 0 (amplituda k·A); "
            "C r = 0,9 / 0,75 / 0,6 / 0,5 (okres r·P).", ""]
    for L in (50, 100, 200):
        out += [f"## Odsetek wykrytych i AUC – poziom intensywności przy L = {L} ({L // 50} cykl/e)", ""] + \
               _tabele(an[an.L == L], (an.rodzaj_kr + an.poziom.astype(str))[an.L == L], KOLEJNOSC)
    out += ["## Odsetek wykrytych anomalii i AUC – rodzaj × poziom intensywności (średnio po L)", ""] + \
        _tabele(an, k, KOLEJNOSC)
    out += ["## Odsetek wykrytych i AUC – rodzaj × długość L (średnio po poziomach)", ""] + \
           _tabele(an, an.rodzaj_kr + "-L" + an.L.astype(str), KOLEJNOSC)
    out += ["## Średnio dla rodzaju (wszystkie poziomy i L)", ""] + _tabele(an, an.rodzaj_kr, KOLEJNOSC)
    out += ["## AP (PR-AUC) średnio dla rodzaju", "", _tab(an, "metoda", "rodzaj_kr", "ap", "{:.2f}", kolejnosc=KOLEJNOSC), ""]
    if (katalog / "okno_anomalie.csv").exists():
        ok = pd.read_csv(katalog / "okno_anomalie.csv")
        ok["rodzaj_kr"] = ok.rodzaj.map({1: "A", 2: "B", 3: "C"})
        out += ["## Długość okna w (poziomy 2 i 3, wszystkie L)", ""] + \
               _tabele(ok, ok.rodzaj_kr + "-w" + ok.w.astype(str), KOLEJNOSC) + \
               [_tab(ok.assign(kol=ok.rodzaj_kr + "-w" + ok.w.astype(str) + "-L" + ok.L.astype(str)), "metoda", "kol",
                     "wykryta", kolejnosc=KOLEJNOSC), ""]
    if (katalog / "tlo_anomalie.csv").exists():
        tl = pd.read_csv(katalog / "tlo_anomalie.csv")
        tl["rodzaj_kr"] = tl.rodzaj.map({1: "A", 2: "B", 3: "C"})
        out += ["## Amplituda cyklu tła A przy σ = 1 (poziom 3: faza +π/2, amplituda ×0,25, okres ×0,6; L = 100)", ""] + \
               _tabele(tl, tl.rodzaj_kr + "-A" + tl.A.astype(str), KOLEJNOSC)
        ctl = pd.read_csv(katalog / "tlo_czyste.csv")
        out += ["Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem) dla każdego A:", "",
                _tab(ctl.assign(fa=ctl.fa_epizody > 0), "metoda", "A", "fa", kolejnosc=KOLEJNOSC), ""]
        ktl = pd.read_csv(katalog / "tlo_kalibracja.csv")
        out += ["Kara PELT dla każdego A:", "", ktl[ktl.metoda == "PELT (reszty STL)"][["A", "kara_pelt"]]
                .to_markdown(index=False), ""]
    if (katalog / "dodatkowa_anomalie.csv").exists():
        dd = pd.read_csv(katalog / "dodatkowa_anomalie.csv")
        dd["rodzaj_kr"] = dd.rodzaj.map({1: "A", 2: "B", 3: "C"})
        a = dd[dd.analiza == "a_model_cyklu"]
        kol = list(dict.fromkeys(a.metoda))
        out += ["## Analiza dodatkowa (a) Model cyklu: STL z oknem 7 cykli / STL z oknem dłuższym niż szereg / "
                "sinus dopasowany globalnie (poziomy 2–3)", ""] + \
               _tabele(a, a.rodzaj_kr + "-L" + a.L.astype(str), kol) + \
               [_tab(a.assign(kol="L" + a.L.astype(str)), "metoda", "kol", "wykryta", kolejnosc=kol), "",
                _tab(a.assign(kol=a.rodzaj_kr + a.poziom.astype(str)), "metoda", "kol", "wykryta", kolejnosc=kol), ""]
        b = dd[dd.analiza == "b_odwrotny_kierunek"]
        out += ["## Analiza dodatkowa (b) Odwrotny kierunek anomalii, L = 100: poziom 2 = faza −π/4, amplituda ×1,5, "
                "okres ×1,25; poziom 3 = faza −π/2, amplituda ×2, okres ×1,5", ""] + \
               _tabele(b, b.rodzaj_kr + b.poziom.astype(str), KOLEJNOSC)
        g = an[(an.L == 100) & an.poziom.isin([2, 3])]
        out += ["Dla porównania – kierunek z części głównej (faza +, osłabienie, przyspieszenie), poziomy 2–3, L = 100:", ""] + \
               _tabele(g, g.rodzaj_kr + g.poziom.astype(str), KOLEJNOSC)
        c = dd[dd.analiza == "c_autoenkoder"]
        kol = list(dict.fromkeys(c.metoda))
        out += ["## Analiza dodatkowa (c) Autoenkoder – sposób uczenia (poziomy 2–3, L = 100)", ""] + \
               _tabele(c, c.rodzaj_kr + c.poziom.astype(str), kol)
        cd = pd.read_csv(katalog / "dodatkowa_czyste.csv")
        out += ["Fałszywe alarmy na walidacji w analizach dodatkowych (odsetek szeregów z ≥1 alarmem):", "",
                cd.groupby(["analiza", "metoda"]).fa_epizody.apply(lambda v: (v > 0).mean()).round(3)
                .to_frame("odsetek").to_markdown(), ""]
        kd = pd.read_csv(katalog / "dodatkowa_kalibracja.csv")
        out += ["Kary PELT w analizach dodatkowych:", "",
                kd[kd.kara_pelt.notna()][["analiza", "metoda", "kara_pelt"]].to_markdown(index=False), ""]
    (katalog / "podsumowanie.md").write_text("\n".join(out), encoding="utf-8")


def main(katalog):
    WYKRESY.mkdir(parents=True, exist_ok=True)
    an = pd.read_csv(katalog / "glowny_anomalie.csv")
    etyk = lambda r, w: [OPIS_POZIOMU[r][int(v) - 1] for v in w]
    opis_L = {50: "L = 50 (1 cykl)", 100: "L = 100 (2 cykle)", 200: "L = 200 (4 cykle)"}
    wykres_siatka_L(an, "czulosc", "poziom", (50, 100, 200), RODZAJ, OS_X, etyk, _linie, "s3_czulosc_intensywnosc.png",
                    "S3 – odsetek wykrytych anomalii a intensywność, osobno dla każdej długości L "
                    "(fałszywy alarm w 10% szeregów bez anomalii)", opis_L)
    wykres_siatka_L(an, "auc", "poziom", (50, 100, 200), RODZAJ, OS_X, etyk, _linie, "s3_auc_intensywnosc.png",
                    "S3 – AUC (miara niezależna od progu) a intensywność anomalii, osobno dla każdej długości L", opis_L)
    wykres_uczenie(an, ML_S3, [50, 100, 200], RODZAJ, KOLORY, ZNACZNIKI, "s3_uczenie.png",
                   "S3 – metody ML: uczenie na tym samym szeregu (linia ciągła) i na 20 szeregach bez anomalii "
                   "(kropkowana); średnio po poziomach", ae="autoenkoder",
                   etyk_L=["50\n(1 cykl)", "100\n(2 cykle)", "200\n(4 cykle)"])
    wykres_parametr(an, "L", "długość anomalii L", "s3_czulosc_dlugosc.png",
                    "S3 – wykrywalność a długość anomalii (średnio po 4 poziomach)",
                    ["50\n(1 cykl)", "100\n(2 cykle)", "200\n(4 cykle)"])
    if (katalog / "okno_anomalie.csv").exists():
        wykres_parametr(pd.read_csv(katalog / "okno_anomalie.csv"), "w", "długość okna w", "s3_okno.png",
                        "S3 – wpływ długości okna w na metody okienkowe (poziomy 2–3, L = 50/100/200 łącznie)",
                        ["25\n(½ cyklu)", "50\n(1 cykl)", "100\n(2 cykle)"])
    if (katalog / "tlo_anomalie.csv").exists():
        wykres_parametr(pd.read_csv(katalog / "tlo_anomalie.csv"), "A", "amplituda cyklu tła A (szum σ = 1)", "s3_tlo.png",
                        "S3 – wpływ amplitudy cyklu tła A, czyli stosunku sygnału do szumu (poziom 3; L = 100)")
    if (katalog / "dodatkowa_anomalie.csv").exists():
        wykres_dodatkowa(katalog, "s3_dodatkowa.png")
    wykres_przyklad("s3_przyklad.png", katalog)
    podsumowanie(katalog)
    print("Zapisano wykresy w", WYKRESY, "i", katalog / "podsumowanie.md")


if __name__ == "__main__":
    main(WYNIKI / ("s3_szybki" if "--szybko" in sys.argv else "s3"))
