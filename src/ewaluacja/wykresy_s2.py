"""Wykresy i tabele podsumowujące eksperyment S2 (wyniki z src/ewaluacja/eksperyment_s2.py).

Uruchomienie (z katalogu głównego repozytorium, po eksperymencie):
    python -m src.ewaluacja.wykresy_s2               # wyniki z wyniki/s2/
    python -m src.ewaluacja.wykresy_s2 --szybko      # wyniki z wyniki/s2_szybki/

Tworzy:
  wyniki/wykresy/s2_przyklad.png              - przykładowy szereg AR(1) i wynik każdej metody z progiem alarmu
  wyniki/wykresy/s2_czulosc_intensywnosc.png  - odsetek wykrytych anomalii vs poziom intensywności (osobno A, B, C)
  wyniki/wykresy/s2_uczenie.png               - metody ML uczone na tym samym szeregu vs na szeregach bez anomalii
  wyniki/wykresy/s2_auc_intensywnosc.png      - AUC vs poziom intensywności
  wyniki/wykresy/s2_czulosc_dlugosc.png       - odsetek wykrytych vs długość L (B i C; szok ma stałą długość)
  wyniki/wykresy/s2_okno.png                  - wpływ długości okna w
  wyniki/wykresy/s2_tlo.png                   - wpływ siły zależności tła φ (właściwości szeregu)
  wyniki/wykresy/s2_dodatkowa.png             - analizy dodatkowe: metody S1 bez modelu AR, wzrost φ, gamma OC-SVM
  wyniki/s2/podsumowanie.md                   - tabele liczbowe (wszystkie wartości użyte we wnioskach)
Przedziały ufności: 95%, Wilsona dla odsetków, ±1,96·SE dla średnich AUC.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import WYKRESY, WYNIKI
from src.ewaluacja.warianty_metod import baza, liczba_cech, rozwin, styl, uczony
from src.ewaluacja.wykresy_s1 import (SIATKA, TEKST, TEKST2, _tab, agreguj, linie_metod,  # noqa: F401  (wspólny styl)
                                      wykres_siatka_L, wykres_uczenie)

# Kolor = metoda (te same rodziny metod co w S1 mają te same kolory; One-Class SVM - nowy, siódmy kolor)
KOLORY = {
    "reszty AR + CUSUM": "#2a78d6",
    "PELT (koszt AR)": "#eb6834",
    "Matrix Profile (reszty AR)": "#1baf7a",
    "cechy okien + kNN": "#eda100",
    "Isolation Forest": "#e87ba4",
    "One-Class SVM": "#6250d6",
}
ZNACZNIKI = dict(zip(KOLORY, ["o", "s", "^", "D", "v", "P"]))
ML_S2 = ["Isolation Forest", "One-Class SVM"]
KOLEJNOSC = rozwin(list(KOLORY), {"cechy okien + kNN", "Isolation Forest", "One-Class SVM"}, set(ML_S2))
KOLEJNOSC_GL = [m for m in KOLEJNOSC if not uczony(m)]
KROTKIE = {"reszty AR + CUSUM": "CUSUM\n(reszty)", "PELT (koszt AR)": "PELT\n(AR)", "Matrix Profile (reszty AR)": "MP\n(reszty)",
           "cechy okien + kNN": "kNN", "Isolation Forest": "IF", "One-Class SVM": "OC-SVM"}


def krotka_m(m):
    return KROTKIE[baza(m)] + ("\n3 cechy" if liczba_cech(m) == 3 else "\n5 cech" if liczba_cech(m) == 5 else "")
RODZAJ = {1: "S2-A: szok wygasający", 2: "S2-B: zmiana siły zależności", 3: "S2-C: przesunięcie poziomu równowagi"}
OPIS_POZIOMU = {1: ["2σε", "3σε", "5σε", "8σε"], 2: ["φ'=0,5", "φ'=0,3", "φ'=0", "φ'=−0,5"],
                3: ["Δ=0,5", "Δ=1", "Δ=2", "Δ=3"]}
OS_X = {1: "impuls szoku", 2: "nowa siła zależności φ' (tło φ = 0,7)", 3: "przesunięcie równowagi (σ_x)"}


def _linie(ax, t, x, y, lo=None, hi=None):
    linie_metod(ax, t, x, y, lo, hi, KOLEJNOSC_GL, KOLORY, ZNACZNIKI)


def _zapisz(fig, axes, plik, tytul):
    fig.suptitle(tytul, x=0.01, ha="left", fontsize=11, color=TEKST)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_dlugosc(an, plik):
    t = agreguj(an[an.rodzaj > 1], ["metoda", "rodzaj", "L"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), sharey=True)
    for ax, r in zip(axes, (2, 3)):
        _linie(ax, t[t.rodzaj == r], "L", "czulosc", "czulosc_lo", "czulosc_hi")
        ax.set_title(RODZAJ[r], fontsize=10, color=TEKST, loc="left")
        ax.set_xlabel("długość anomalii L (punkty)")
        ax.set_xticks([20, 50, 100])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("odsetek wykrytych anomalii")
    _zapisz(fig, axes, plik, "S2 – wykrywalność a długość anomalii (średnio po 4 poziomach; szok S2-A ma zawsze 10 punktów)")


def wykres_parametr(an, kol, etyk_x, plik, tytul):
    t = agreguj(an, ["metoda", "rodzaj", kol])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    for ax, r in zip(axes, (1, 2, 3)):
        _linie(ax, t[t.rodzaj == r], kol, "czulosc", "czulosc_lo", "czulosc_hi")
        ax.set_title(RODZAJ[r], fontsize=10, color=TEKST, loc="left")
        ax.set_xlabel(etyk_x)
        w = sorted(t[kol].unique())
        ax.set_xticks(w)
        ax.set_xticklabels([f"{v:g}".replace(".", ",") for v in w])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("odsetek wykrytych anomalii")
    _zapisz(fig, axes, plik, tytul)


def _dodatkowa(katalog):
    """Wyniki analiz dodatkowych (poziom w analizie (b) zapisany w kolumnie `poziom` przez klucz „_poziom”)."""
    return pd.read_csv(katalog / "dodatkowa_anomalie.csv")


def wykres_dodatkowa(katalog, plik):
    dod = _dodatkowa(katalog)
    gl = pd.read_csv(katalog / "glowny_anomalie.csv")
    gl23 = gl[(gl.L == 50) & gl.poziom.isin([2, 3])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    # (a) metody z S1 bez modelu AR vs ich odpowiedniki z S2 (poziomy 2-3, L = 50)
    pary = [("CUSUM z S1 (bez modelu AR)", "reszty AR + CUSUM", "#2a78d6", "CUSUM"),
            ("Matrix Profile", "Matrix Profile (reszty AR)", "#1baf7a", "Matrix Profile"),
            ("kNN z cechami S1 (bez modelu AR)", "cechy okien + kNN (5 cech)", "#eda100", "kNN")]
    a = dod[dod.analiza == "a_metody_S1_wprost"]
    xs = np.arange(3)
    for i, (m1, m2, kol, krotka) in enumerate(pary):
        v1 = [a[(a.metoda == m1) & (a.rodzaj == r)].wykryta.mean() for r in (1, 2, 3)]
        v2 = [gl23[(gl23.metoda == m2) & (gl23.rodzaj == r)].wykryta.mean() for r in (1, 2, 3)]
        axes[0].bar(xs + i * 0.27 - 0.27, v1, 0.12, color="white", edgecolor=kol, hatch="///", lw=1.2,
                    label=f"{krotka}: bez modelu AR (z S1)")
        axes[0].bar(xs + i * 0.27 - 0.27 + 0.13, v2, 0.12, color=kol, label=f"{krotka}: z modelem AR (S2)")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["A szok", "B zmiana φ", "C równowaga"])
    axes[0].set_ylim(0, 1.45)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("odsetek wykrytych anomalii")
    axes[0].set_title("(a) Czy model AR jest potrzebny? (poziomy 2–3, L = 50)", loc="left", fontsize=9.5)
    axes[0].legend(fontsize=6.8, ncol=2, loc="upper left")

    # (b) wzrost zależności φ' = 0,95 vs spadek φ' = 0 (zmiana φ - rodzaj B)
    b = dod[(dod.analiza == "b_wzrost_phi_0.95") & (dod.rodzaj == 2)]
    gl_b = gl[(gl.L == 50) & (gl.rodzaj == 2)]
    xs = np.arange(len(KOLEJNOSC_GL))
    wz = [b[b.metoda == m].wykryta.mean() for m in KOLEJNOSC_GL]
    sp = [gl_b[(gl_b.metoda == m) & (gl_b.poziom == 3)].wykryta.mean() for m in KOLEJNOSC_GL]
    kol = [KOLORY[baza(m)] for m in KOLEJNOSC_GL]
    axes[1].bar(xs - 0.2, sp, 0.38, color=kol, label="spadek: φ' = 0")
    axes[1].bar(xs + 0.2, wz, 0.38, color="white", edgecolor=kol, hatch="///", lw=1.2, label="wzrost: φ' = 0,95")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([krotka_m(m) for m in KOLEJNOSC_GL], fontsize=6.5)
    axes[1].set_ylim(0, 1.3)
    axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1].set_title("(b) Zmiana φ w dół i w górę (S2-B, L = 50)", loc="left", fontsize=9.5)
    axes[1].legend(fontsize=7.5, loc="upper left", ncol=2)

    # (c) One-Class SVM - szerokość jądra
    c = dod[dod.analiza == "c_gamma_OCSVM"]
    warianty = ["One-Class SVM gamma=scale", "One-Class SVM gamma=0,01", "One-Class SVM gamma=0,002"]
    xs = np.arange(3)
    for i, (r, kol) in enumerate(zip((1, 2, 3), ("#cdbfff", "#6250d6", "#3d2f9e"))):
        v = [c[(c.metoda == m) & (c.rodzaj == r)].wykryta.mean() for m in warianty]
        axes[2].bar(xs + (i - 1) * 0.27, v, 0.25, color=kol, label=RODZAJ[r].split(":")[0])
        auc = [c[(c.metoda == m) & (c.rodzaj == r)].auc.mean() for m in warianty]
        for xx, vv, aa in zip(xs + (i - 1) * 0.27, v, auc):
            axes[2].text(xx, vv + 0.02, f"{aa:.2f}".replace(".", ","), ha="center", fontsize=6.5, color=TEKST2)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(["gamma = scale\n(≈0,26)", "gamma = 0,01\n(wybrane)", "gamma = 0,002"])
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title("(c) One-Class SVM (5 cech): szerokość jądra (nad słupkami: AUC)", loc="left", fontsize=9.5)
    axes[2].legend(fontsize=7.5)
    fig.suptitle("S2 – analizy dodatkowe", x=0.01, ha="left", fontsize=11, color=TEKST)
    fig.tight_layout()
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_przyklad(plik, katalog):
    from src.ewaluacja.eksperyment import ZIARNO_KAL
    from src.metody import metody_s2
    from src.symulacje.anomalie import szereg_s2

    kal = pd.read_csv(katalog / "glowny_kalibracja.csv").set_index("metoda")
    s = szereg_s2(3_100_000 + 300_000 + 10_000 + 7, poziom=3, L=50)
    czyste = [szereg_s2(ZIARNO_KAL + i, z_anomaliami=False).x for i in range(200)]
    metody = [m for m in metody_s2() if liczba_cech(m.nazwa) != 3]      # warianty 3-cechowe - w tabelach
    fig, axes = plt.subplots(len(metody) + 1, 1, figsize=(12.5, 11.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.6] + [1] * len(metody)})
    kol_an = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

    def zaznacz(ax, z_opisem=False):
        for a in s.anomalie:
            ax.axvspan(a.start, a.koniec, color=kol_an[a.rodzaj], alpha=0.15 if a.rodzaj > 1 else 0.35, lw=0,
                       label=RODZAJ[a.rodzaj] if z_opisem else None)

    axes[0].plot(s.x, lw=0.55, color=TEKST2)
    zaznacz(axes[0], True)
    axes[0].set_title("Szereg S2 – AR(1), φ = 0,7 (poziom 3: szok 5σε, φ' = 0, Δ = 2σ_x; L = 50) i wyniki metod; "
                      "linia przerywana = próg alarmu", loc="left", fontsize=10, color=TEKST)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    for ax, m in zip(axes[1:], metody):
        m.kalibruj(czyste)                     # te same szeregi co w eksperymencie → te same stałe
        w = m.wynik(s.x)
        zaznacz(ax)
        ax.plot(w, color=KOLORY[baza(m.nazwa)], lw=1.1)
        ax.axhline(kal.loc[m.nazwa, "prog"], color=TEKST, lw=0.9, ls="--")
        ax.set_ylabel(m.nazwa.replace(" (", "\n(").replace(" + ", "\n+ "), rotation=0, ha="right", va="center",
                      fontsize=8.5)
        ax.set_yticks([])
    axes[-1].set_xlabel("indeks punktu")
    fig.tight_layout()
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def podsumowanie(katalog):
    an = pd.read_csv(katalog / "glowny_anomalie.csv")
    sz = pd.read_csv(katalog / "glowny_szeregi.csv")
    cz = pd.read_csv(katalog / "glowny_czyste.csv")
    kal = pd.read_csv(katalog / "glowny_kalibracja.csv")
    an["rodzaj_kr"] = an.rodzaj.map({1: "A", 2: "B", 3: "C"})
    out = ["# S2 – tabele wyników (generowane automatycznie przez src/ewaluacja/wykresy_s2.py)", "",
           f"Część główna: {an.ziarno.nunique()} szeregów z anomaliami ({an.groupby('metoda').size().iloc[0]} anomalii na "
           f"metodę), {cz.ziarno.nunique()} szeregów walidacyjnych bez anomalii. Próg: fałszywy alarm w 10% szeregów "
           "kalibracyjnych bez anomalii.", "",
           "## Kalibracja", "", kal[["metoda", "prog", "kara_pelt"]].to_markdown(index=False), ""]
    fa = cz.groupby("metoda").agg(odsetek_z_fa=("fa_epizody", lambda v: (v > 0).mean()),
                                  fa_na_1000_pkt=("fa_epizody", lambda v: v.mean() / 3)).reset_index()
    fa2 = sz.groupby("metoda").agg(fa_w_szeregach_z_anomaliami=("fa_epizody", "mean"), precyzja=("precyzja", "mean"),
                                   czulosc_pkt=("czulosc_pkt", "mean"), f1=("f1", "mean")).reset_index()
    t = fa.merge(fa2, on="metoda")
    t["k"] = t.metoda.map({m: i for i, m in enumerate(KOLEJNOSC)})
    out += ["## Fałszywe alarmy (walidacja na nowych szeregach bez anomalii) i miary punktowe", "",
            t.sort_values("k").drop(columns="k").round(3).to_markdown(index=False), ""]
    k = an.rodzaj_kr + an.poziom.astype(str)
    out += ["Poziomy: A impuls = 2 / 3 / 5 / 8 σε; B φ' = 0,5 / 0,3 / 0 / −0,5 (tło φ = 0,7); C Δ = 0,5 / 1 / 2 / 3 σ_x. "
            "L = długość anomalii B i C; szok S2-A ma zawsze 10 punktów.", ""]
    for L in (20, 50, 100):
        d = an[an.L == L]
        out += [f"## Odsetek wykrytych anomalii – rodzaj × poziom intensywności, L = {L}", "",
                _tab(d.assign(kol=k[an.L == L]), "metoda", "kol", "wykryta", kolejnosc=KOLEJNOSC), "",
                f"## AUC – rodzaj × poziom intensywności, L = {L}", "",
                _tab(d.assign(kol=k[an.L == L]), "metoda", "kol", "auc", "{:.2f}", kolejnosc=KOLEJNOSC), ""]
    out += ["## Odsetek wykrytych anomalii – rodzaj × poziom intensywności (średnio po L)", "",
            _tab(an.assign(kol=k), "metoda", "kol", "wykryta", kolejnosc=KOLEJNOSC), "",
            "## AUC – rodzaj × poziom intensywności (średnio po L)", "",
            _tab(an.assign(kol=k), "metoda", "kol", "auc", "{:.2f}", kolejnosc=KOLEJNOSC), "",
            "## Odsetek wykrytych – rodzaj × długość L (średnio po poziomach; dla A długość zawsze 10)", "",
            _tab(an.assign(kol=an.rodzaj_kr + "-L" + an.L.astype(str)), "metoda", "kol", "wykryta", kolejnosc=KOLEJNOSC), "",
            "## Średnio dla rodzaju (wszystkie poziomy i L)", "",
            _tab(an, "metoda", "rodzaj_kr", "wykryta", kolejnosc=KOLEJNOSC), "",
            _tab(an, "metoda", "rodzaj_kr", "auc", "{:.2f}", kolejnosc=KOLEJNOSC), "",
            "## AP (PR-AUC) średnio dla rodzaju", "", _tab(an, "metoda", "rodzaj_kr", "ap", "{:.2f}", kolejnosc=KOLEJNOSC), ""]
    if (katalog / "okno_anomalie.csv").exists():
        ok = pd.read_csv(katalog / "okno_anomalie.csv")
        ok["rodzaj_kr"] = ok.rodzaj.map({1: "A", 2: "B", 3: "C"})
        out += ["## Długość okna w (poziomy 2 i 3, wszystkie L)", "",
                _tab(ok.assign(kol=ok.rodzaj_kr + "-w" + ok.w.astype(str)), "metoda", "kol", "wykryta", kolejnosc=KOLEJNOSC), "",
                _tab(ok.assign(kol=ok.rodzaj_kr + "-w" + ok.w.astype(str) + "-L" + ok.L.astype(str)), "metoda", "kol",
                     "wykryta", kolejnosc=KOLEJNOSC), ""]
    if (katalog / "tlo_anomalie.csv").exists():
        tl = pd.read_csv(katalog / "tlo_anomalie.csv")
        tl["rodzaj_kr"] = tl.rodzaj.map({1: "A", 2: "B", 3: "C"})
        k = tl.rodzaj_kr + "-φ" + tl.phi_tla.astype(str)
        out += ["## Siła zależności tła φ (szok 5σε, φ' = 0, Δ = 2σ_x, L = 50)", "",
                _tab(tl.assign(kol=k), "metoda", "kol", "wykryta", kolejnosc=KOLEJNOSC), "",
                _tab(tl.assign(kol=k), "metoda", "kol", "auc", "{:.2f}", kolejnosc=KOLEJNOSC), ""]
        ctl = pd.read_csv(katalog / "tlo_czyste.csv")
        out += ["Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem) dla każdego φ:", "",
                _tab(ctl.assign(fa=ctl.fa_epizody > 0), "metoda", "phi_tla", "fa", kolejnosc=KOLEJNOSC), ""]
    if (katalog / "dodatkowa_anomalie.csv").exists():
        dd = _dodatkowa(katalog)
        dd["rodzaj_kr"] = dd.rodzaj.map({1: "A", 2: "B", 3: "C"})
        for analiza, tytul in [("a_metody_S1_wprost", "(a) Metody z S1 zastosowane wprost (bez modelu AR), poziomy 2–3, L = 50"),
                               ("b_wzrost_phi_0.95", "(b) Wzrost zależności φ' = 0,95 (B); A i C jak poziomy 2–3, L = 50"),
                               ("c_gamma_OCSVM", "(c) One-Class SVM – szerokość jądra, poziomy 2–3, L = 50")]:
            d = dd[dd.analiza == analiza]
            kol = list(dict.fromkeys(d.metoda))
            out += [f"## Analiza dodatkowa {tytul}", "",
                    _tab(d.assign(kol=d.rodzaj_kr + d.poziom.astype(str)), "metoda", "kol", "wykryta", kolejnosc=kol), "",
                    _tab(d.assign(kol=d.rodzaj_kr + d.poziom.astype(str)), "metoda", "kol", "auc", "{:.2f}", kolejnosc=kol), ""]
        gl = an[(an.L == 50) & an.poziom.isin([2, 3])]
        out += ["Dla porównania – metody S2 z części głównej, poziomy 2–3, L = 50:", "",
                _tab(gl.assign(kol=gl.rodzaj_kr + gl.poziom.astype(str)), "metoda", "kol", "wykryta", kolejnosc=KOLEJNOSC), "",
                _tab(gl.assign(kol=gl.rodzaj_kr + gl.poziom.astype(str)), "metoda", "kol", "auc", "{:.2f}", kolejnosc=KOLEJNOSC), ""]
        cd = pd.read_csv(katalog / "dodatkowa_czyste.csv")
        out += ["Fałszywe alarmy na walidacji w analizach dodatkowych (odsetek szeregów z ≥1 alarmem):", "",
                cd.groupby(["analiza", "metoda"]).fa_epizody.apply(lambda v: (v > 0).mean()).round(3)
                .to_frame("odsetek").to_markdown(), ""]
    if (katalog / "hybryda_anomalie.csv").exists():
        hy = pd.read_csv(katalog / "hybryda_anomalie.csv")
        hy["rodzaj_kr"] = hy.rodzaj.map({1: "A", 2: "B", 3: "C"})
        kol = list(dict.fromkeys(hy.metoda))
        k = hy.rodzaj_kr + hy.poziom.astype(str)
        out += ["## Analiza dodatkowa (d) CUSUM mieszany – wariant zaproponowany po wynikach (a); poziomy 1–4, L = 50", "",
                _tab(hy.assign(kol=k), "metoda", "kol", "wykryta", kolejnosc=kol), "",
                _tab(hy, "metoda", "rodzaj_kr", "wykryta", kolejnosc=kol), "",
                _tab(hy, "metoda", "rodzaj_kr", "auc", "{:.2f}", kolejnosc=kol), ""]
        ch = pd.read_csv(katalog / "hybryda_czyste.csv")
        out += ["Fałszywe alarmy na walidacji (odsetek szeregów z ≥1 alarmem):", "",
                ch.groupby("metoda").fa_epizody.apply(lambda v: (v > 0).mean()).round(3).to_frame("odsetek").to_markdown(), ""]
    (katalog / "podsumowanie.md").write_text("\n".join(out), encoding="utf-8")


def main(katalog):
    WYKRESY.mkdir(parents=True, exist_ok=True)
    an = pd.read_csv(katalog / "glowny_anomalie.csv")
    etyk = lambda r, w: [OPIS_POZIOMU[r][int(v) - 1] for v in w]
    opis_L = {L: f"L = {L} (B, C)" for L in (20, 50, 100)}
    wykres_siatka_L(an, "czulosc", "poziom", (20, 50, 100), RODZAJ, OS_X, etyk, _linie, "s2_czulosc_intensywnosc.png",
                    "S2 – odsetek wykrytych anomalii a intensywność, osobno dla każdej długości L (szok zawsze 10 pkt; "
                    "fałszywy alarm w 10% szeregów bez anomalii)", opis_L, ncol=3)
    wykres_siatka_L(an, "auc", "poziom", (20, 50, 100), RODZAJ, OS_X, etyk, _linie, "s2_auc_intensywnosc.png",
                    "S2 – AUC (miara niezależna od progu) a intensywność anomalii, osobno dla każdej długości L", opis_L,
                    ncol=3)
    wykres_uczenie(an, ML_S2, [20, 50, 100], RODZAJ, KOLORY, ZNACZNIKI, "s2_uczenie.png",
                   "S2 – metody ML: uczenie na tym samym szeregu (linia ciągła) i na 20 szeregach bez anomalii "
                   "(kropkowana); średnio po poziomach")
    wykres_dlugosc(an, "s2_czulosc_dlugosc.png")
    if (katalog / "okno_anomalie.csv").exists():
        wykres_parametr(pd.read_csv(katalog / "okno_anomalie.csv"), "w", "długość okna w (punkty)", "s2_okno.png",
                        "S2 – wpływ długości okna w na metody okienkowe (poziomy 2–3, L = 20/50/100 łącznie)")
    if (katalog / "tlo_anomalie.csv").exists():
        wykres_parametr(pd.read_csv(katalog / "tlo_anomalie.csv"), "phi_tla", "siła zależności tła φ", "s2_tlo.png",
                        "S2 – wpływ siły zależności szeregu tła φ (szok 5σε, φ' = 0, Δ = 2σ_x; L = 50)")
    if (katalog / "dodatkowa_anomalie.csv").exists():
        wykres_dodatkowa(katalog, "s2_dodatkowa.png")
    wykres_przyklad("s2_przyklad.png", katalog)
    podsumowanie(katalog)
    print("Zapisano wykresy w", WYKRESY, "i", katalog / "podsumowanie.md")


if __name__ == "__main__":
    main(WYNIKI / ("s2_szybki" if "--szybko" in sys.argv else "s2"))
