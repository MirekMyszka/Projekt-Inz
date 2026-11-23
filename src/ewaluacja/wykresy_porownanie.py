"""Porównanie scenariuszy S1–S3 (docs/plan_etap_1.md, krok 9): wykresy i tabele z wyników wszystkich trzech
eksperymentów oraz eksperymentu przeniesienia metod (src/ewaluacja/porownanie.py).

Uruchomienie (z katalogu głównego repozytorium, po eksperymentach S1–S3 i porównaniu):
    python -m src.ewaluacja.wykresy_porownanie

Tworzy:
  wyniki/wykresy/por_mapa_wykrywalnosci.png - odsetek wykrytych: wariant metody × 9 rodzajów anomalii (S1–S3)
  wyniki/wykresy/por_przeniesienie.png      - macierz przeniesienia: metody scenariusza X na szeregach scenariusza Y
  wyniki/wykresy/por_wlasciwosci.png        - wpływ właściwości szeregu: σ (S1), φ tła (S2), siła cyklu A (S3)
  wyniki/wykresy/por_okno_dlugosc.png       - długość okna w i długość anomalii L we wszystkich scenariuszach
  wyniki/porownanie/podsumowanie.md         - wszystkie tabele liczbowe porównania

Metody porównujemy w RODZINACH - ta sama rola w każdym scenariuszu, konfiguracja dopasowana do szeregu
(docs/plan_etap_1.md, rozdz. 4.2): karta kontrolna (CUSUM / reszty AR + CUSUM / STL + EWMA), PELT, Matrix Profile,
kNN na cechach okien, Isolation Forest, druga metoda ML (LOF / One-Class SVM / autoenkoder).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import WYKRESY, WYNIKI
from src.ewaluacja.porownanie import POZIOMY, R_OCENA
from src.ewaluacja.warianty_metod import UCZONY
from src.ewaluacja.wykresy_s1 import SIATKA, TEKST, TEKST2, wilson

SC = ("S1", "S2", "S3")
KAT = WYNIKI / "porownanie"
OPIS_SC = {"S1": "S1 – biały szum", "S2": "S2 – AR(1), φ = 0,7", "S3": "S3 – szum + sinus (A = 3, P = 50)"}
RODZAJ = {"S1": {1: "S1-A poziom", 2: "S1-B zmienność", 3: "S1-C zależność"},
          "S2": {1: "S2-A szok", 2: "S2-B zmiana φ", 3: "S2-C równowaga"},
          "S3": {1: "S3-A faza", 2: "S3-B amplituda", 3: "S3-C okres"}}
# Rodzaj anomalii (docs/plan_etap_1.md, rozdz. 3): co zmienia się we fragmencie.
KATEGORIA = {("S1", 1): "wartości", ("S1", 2): "wartości", ("S3", 2): "wartości",
             ("S1", 3): "struktury", ("S2", 2): "struktury", ("S3", 1): "struktury", ("S3", 3): "struktury",
             ("S2", 1): "dynamiki", ("S2", 3): "dynamiki"}
OPIS_KAT = {"wartości": "anomalie wartości – zmienia się poziom albo rozrzut",
            "struktury": "anomalie struktury – średnia i wariancja jak w normie, zmienia się przebieg w czasie",
            "dynamiki": "anomalie o kształcie wynikającym z dynamiki tła – bez ostrej granicy"}
RODZINY = {
    "karta kontrolna": {"S1": "CUSUM (poziom, zmienność, zależność)", "S2": "reszty AR + CUSUM", "S3": "STL + EWMA"},
    "PELT": {"S1": "PELT", "S2": "PELT (koszt AR)", "S3": "PELT (reszty STL)"},
    "Matrix Profile": {"S1": "Matrix Profile", "S2": "Matrix Profile (reszty AR)", "S3": "Matrix Profile z normalizacją"},
    "kNN": {"S1": "cechy okien + kNN", "S2": "cechy okien + kNN", "S3": "cechy cykli + kNN"},
    "Isolation Forest": {"S1": "Isolation Forest", "S2": "Isolation Forest", "S3": "Isolation Forest"},
    "druga metoda ML": {"S1": "LOF", "S2": "One-Class SVM", "S3": "autoenkoder"},
}
Z_CECHAMI = {"kNN", "Isolation Forest", "druga metoda ML"}
ML = {"Isolation Forest", "druga metoda ML"}
KOLORY = {"karta kontrolna": "#2a78d6", "PELT": "#eb6834", "Matrix Profile": "#1baf7a", "kNN": "#eda100",
          "Isolation Forest": "#e87ba4", "druga metoda ML": "#008300"}
ZNACZNIKI = {"karta kontrolna": "o", "PELT": "s", "Matrix Profile": "^", "kNN": "D", "Isolation Forest": "v",
             "druga metoda ML": "P"}
# warianty: "" = bez cech / 5 cech, "3" = 3 cechy, "U" = 5 cech uczony na szeregach bez anomalii, "3U"
WARIANTY = ("", "3", "U", "3U")
OPIS_W = {"": "", "3": " – 3 cechy", "U": " – uczony bez anomalii", "3U": " – 3 cechy, uczony bez anomalii"}


def nazwa(rodzina: str, sc: str, wariant: str = "") -> str | None:
    """Pełna nazwa metody w CSV scenariusza sc (None - wariant nie istnieje)."""
    b = RODZINY[rodzina][sc]
    if rodzina not in Z_CECHAMI:
        return b if wariant == "" else None
    if b == "autoenkoder":
        return {"": b, "U": f"{b} ({UCZONY})"}.get(wariant)
    if wariant in ("U", "3U") and rodzina not in ML:
        return None
    return {"": f"{b} (5 cech)", "3": f"{b} (3 cechy)", "U": f"{b} (5 cech, {UCZONY})",
            "3U": f"{b} (3 cechy, {UCZONY})"}[wariant]


def wiersze(uczone=True):
    """(rodzina, wariant, etykieta) w stałej kolejności tabel."""
    out = []
    for r in RODZINY:
        for w in WARIANTY:
            if (w in ("U", "3U") and not uczone) or all(nazwa(r, s, w) is None for s in SC):
                continue
            et = r + ({"": " – 5 cech", "3": " – 3 cechy", "U": " – 5 cech, uczony bez anomalii",
                       "3U": " – 3 cechy, uczony bez anomalii"}[w] if r in Z_CECHAMI else "")
            out.append((r, w, et))
    return out


def wczytaj(sc: str, czesc: str, plik: str = "anomalie") -> pd.DataFrame:
    d = pd.read_csv(WYNIKI / sc.lower() / f"{czesc}_{plik}.csv")
    if plik == "anomalie" and "poziom" not in d.columns:      # S1: poziom zapisany w ziarnie (100 000·poziom + ...)
        d["poziom"] = d.ziarno // 100_000
    return d


def pct(v, znak=True):
    return "–" if v is None or pd.isna(v) else f"{round(100 * v)}" + ("%" if znak else "")


def f2(v):
    return "–" if v is None or pd.isna(v) else f"{v:.2f}".replace(".", ",")


def tabela(nag, wiersze_tab):
    return "\n".join(["| " + " | ".join(nag) + " |", "|" + "---|" * len(nag)] +
                     ["| " + " | ".join(str(k) for k in w) + " |" for w in wiersze_tab])


# ------------------------------------------------------------------------------------------------ dane zbiorcze
def glowne() -> dict:
    return {s: wczytaj(s, "glowny") for s in SC}


def wykrywalnosc(gl: dict, uczone=True) -> pd.DataFrame:
    """Wiersz = wariant metody × scenariusz × rodzaj: odsetek wykrytych, AUC, n (wszystkie poziomy i L)."""
    rows = []
    for r, w, et in wiersze(uczone):
        for s in SC:
            m = nazwa(r, s, w)
            if m is None:
                continue
            d = gl[s][gl[s].metoda == m]
            for k in (1, 2, 3):
                x = d[d.rodzaj == k]
                rows.append({"rodzina": r, "wariant": w, "etykieta": et, "sc": s, "metoda": m, "rodzaj": k,
                             "anomalia": RODZAJ[s][k], "kategoria": KATEGORIA[(s, k)], "wykryta": x.wykryta.mean(),
                             "auc": x.auc.mean(), "n": len(x)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ wykresy
def mapa(t: pd.DataFrame, miara: str, plik: str, tytul: str):
    et = [e for _, _, e in wiersze()]
    kol = [(s, k) for s in SC for k in (1, 2, 3)]
    M = np.full((len(et), len(kol)), np.nan)
    for i, e in enumerate(et):
        for j, (s, k) in enumerate(kol):
            v = t[(t.etykieta == e) & (t.sc == s) & (t.rodzaj == k)][miara]
            if len(v):
                M[i, j] = v.iloc[0]
    fig, ax = plt.subplots(figsize=(12.5, 0.42 * len(et) + 1.6))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f3f2ee")
    lo = 0.0 if miara == "wykryta" else 0.4
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=lo, vmax=1.0, aspect="auto")
    for i in range(len(et)):
        for j in range(len(kol)):
            v = M[i, j]
            if not np.isnan(v):
                txt = f"{round(100 * v)}" if miara == "wykryta" else f"{v:.2f}".replace(".", ",")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if (v - lo) / (1 - lo) > 0.6 else TEKST)
    ax.set_xticks(range(len(kol)))
    ax.set_xticklabels([RODZAJ[s][k] for s, k in kol], rotation=0, fontsize=8)
    ax.set_yticks(range(len(et)))
    ax.set_yticklabels(et, fontsize=8.5)
    for x in (2.5, 5.5):
        ax.axvline(x, color="white", lw=3)
    ax.grid(False)
    ax.tick_params(length=0)
    for s_, sp in ax.spines.items():
        sp.set_visible(False)
    ax.set_title(tytul, loc="left", fontsize=10.5, color=TEKST)
    fig.tight_layout()
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def przeniesienie() -> pd.DataFrame | None:
    """Wiersz = wariant metody × (metody_z, szereg) × rodzaj; przekątna z części głównej - te same szeregi co
    w przeniesieniu (poziomy 2–3, pierwsze R_OCENA szeregów na (poziom, L): ziarno % 10 000 < R_OCENA)."""
    plik = KAT / "przeniesienie_anomalie.csv"
    if not plik.exists():
        return None
    pr = pd.read_csv(plik)
    gl = {s: wczytaj(s, "glowny") for s in SC}
    rows = []
    for r, w, et in wiersze(uczone=False):
        for z in SC:
            m = nazwa(r, z, w)
            if m is None:
                continue
            for s in SC:
                d = (gl[s][gl[s].poziom.isin(POZIOMY) & (gl[s].ziarno % 10_000 < R_OCENA)] if z == s
                     else pr[(pr.metody_z == z) & (pr.szereg == s)])
                d = d[d.metoda == m]
                for k in (1, 2, 3):
                    x = d[d.rodzaj == k]
                    rows.append({"rodzina": r, "wariant": w, "etykieta": et, "metody_z": z, "szereg": s, "metoda": m,
                                 "rodzaj": k, "wykryta": x.wykryta.mean(), "auc": x.auc.mean(), "n": len(x)})
    return pd.DataFrame(rows)


def wykres_przeniesienie(t: pd.DataFrame, plik: str):
    et = [e for _, _, e in wiersze(uczone=False)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.36 * len(et) + 2.2), sharey=True)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f3f2ee")
    for ax, s in zip(axes, SC):
        M = np.full((len(et), 3), np.nan)
        for i, e in enumerate(et):
            for j, z in enumerate(SC):
                v = t[(t.etykieta == e) & (t.metody_z == z) & (t.szereg == s)].wykryta
                if len(v) == 3:
                    M[i, j] = v.mean()
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(et)):
            for j in range(3):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{round(100 * M[i, j])}", ha="center", va="center", fontsize=8.5,
                            color="white" if M[i, j] > 0.6 else TEKST, fontweight="bold" if SC[j] == s else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"metody {z}" + (" (u siebie)" if z == s else "") for z in SC], fontsize=8)
        ax.set_title(f"szeregi {OPIS_SC[s]}", loc="left", fontsize=9.5, color=TEKST)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(et)))
    axes[0].set_yticklabels(et, fontsize=8.5)
    fig.suptitle("Przeniesienie metod: odsetek wykrytych anomalii (średnio A, B, C; poziomy 2–3, wszystkie L) – "
                 "kolumna = skąd pochodzi zestaw metod", x=0.01, ha="left", fontsize=10.5, color=TEKST)
    fig.tight_layout()
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def _styl(r, w):
    return {"color": KOLORY[r], "marker": ZNACZNIKI[r], "ls": "--" if "3" in w else "-", "lw": 1.8, "ms": 4.5,
            "mfc": "white" if "3" in w else KOLORY[r]}


WLASCIWOSCI = {"S1": ("szum", "sigma", "poziom szumu σ (Δ = 1, c = 2, φ = 0,5 bezwzględnie)"),
               "S2": ("tlo", "phi_tla", "siła zależności tła φ (szok 5σ_ε, φ' = 0, Δ = 2σ_x)"),
               "S3": ("tlo", "A", "amplituda cyklu A przy σ = 1 (faza π/2, ×0,25, okres ×0,6)")}


def wykres_wlasciwosci(plik: str):
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 10.2), sharey=True)
    for i, s in enumerate(SC):
        czesc, kol, opis = WLASCIWOSCI[s]
        d = wczytaj(s, czesc)
        for j, k in enumerate((1, 2, 3)):
            ax = axes[i, j]
            for r, w, et in wiersze(uczone=False):
                m = nazwa(r, s, w)
                x = d[(d.metoda == m) & (d.rodzaj == k)]
                if m is None or x.empty:
                    continue
                g = x.groupby(kol).wykryta.mean()
                ax.plot(g.index, g.values, label=et, **_styl(r, w))
            v = sorted(d[kol].unique())
            ax.set_xticks(v)
            ax.set_xticklabels([f"{u:g}".replace(".", ",") for u in v])
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(RODZAJ[s][k], loc="left", fontsize=10, color=TEKST)
            if j == 0:
                ax.set_ylabel("odsetek wykrytych")
            if j == 1:
                ax.set_xlabel(opis, fontsize=8.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Właściwości szeregu przy stałej anomalii: szum (S1), zależność tła (S2), siła cyklu (S3); "
                 "linia przerywana = 3 cechy", x=0.01, ha="left", fontsize=10.5, color=TEKST)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


def wykres_okno_dlugosc(gl: dict, plik: str):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), sharey=True)
    for j, s in enumerate(SC):
        ok = wczytaj(s, "okno")
        for r, w, et in wiersze(uczone=False):
            m = nazwa(r, s, w)
            for i, (d, kol) in enumerate(((ok, "w"), (gl[s], "L"))):
                x = d[d.metoda == m]
                if m is None or x.empty:
                    continue
                g = x.groupby(kol).wykryta.mean()
                axes[i, j].plot(g.index, g.values, label=et, **_styl(r, w))
        for i, (d, kol, op) in enumerate(((ok, "w", "długość okna w (poziomy 2–3, wszystkie L)"),
                                          (gl[s], "L", "długość anomalii L (wszystkie poziomy)"))):
            v = sorted(d[kol].unique())
            axes[i, j].set_xticks(v)
            axes[i, j].set_xlabel(op, fontsize=8.5)
            axes[i, j].set_ylim(-0.03, 1.03)
        axes[0, j].set_title(OPIS_SC[s], loc="left", fontsize=10, color=TEKST)
    for i in range(2):
        axes[i, 0].set_ylabel("odsetek wykrytych (średnio A, B, C)")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Długość okna (górny wiersz) i długość anomalii (dolny) we wszystkich scenariuszach; "
                 "linia przerywana = 3 cechy", x=0.01, ha="left", fontsize=10.5, color=TEKST)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(WYKRESY / plik, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ podsumowanie
def podsumowanie(gl: dict, t: pd.DataFrame, tp: pd.DataFrame | None):
    out = ["# Porównanie S1–S3 – tabele (generowane automatycznie przez src/ewaluacja/wykresy_porownanie.py)", ""]
    out += ["Nazwy metod w rodzinach:", "",
            tabela(["rodzina", "S1", "S2", "S3"], [[r] + [RODZINY[r][s] for s in SC] for r in RODZINY]), ""]

    # 1. mapa wykrywalności
    kol = [(s, k) for s in SC for k in (1, 2, 3)]
    for miara, fmt, tyt in (("wykryta", pct, "Odsetek wykrytych"), ("auc", f2, "AUC")):
        w = []
        for _, _, e in wiersze():
            w.append([e] + [fmt(t[(t.etykieta == e) & (t.sc == s) & (t.rodzaj == k)][miara].mean())
                            if len(t[(t.etykieta == e) & (t.sc == s) & (t.rodzaj == k)]) else "–" for s, k in kol])
        out += [f"## {tyt} – część główna (wszystkie poziomy i L)", "",
                tabela(["metoda"] + [RODZAJ[s][k] for s, k in kol], w), ""]

    # 2. najlepsze metody w każdym scenariuszu
    out += ["## Ranking metod w scenariuszu (średnio A, B, C; wszystkie poziomy i L)", ""]
    w = []
    for s in SC:
        r = t[t.sc == s].groupby("metoda").agg(w=("wykryta", "mean"), auc=("auc", "mean")).sort_values("w", ascending=False)
        for i, (m, x) in enumerate(r.iterrows()):
            w.append([s, i + 1, m, pct(x.w), f2(x.auc)])
    out += [tabela(["scenariusz", "miejsce", "metoda", "wykryte (śr.)", "AUC (śr.)"], w), ""]

    # 3. kategorie anomalii - najlepsza metoda w każdej grupie
    out += ["## Najlepsza metoda dla każdej anomalii – w każdej grupie metod (wszystkie poziomy i L)", ""]
    grupa = {"karta kontrolna": "statystyczne", "PELT": "statystyczne", "Matrix Profile": "fragmenty",
             "kNN": "fragmenty", "Isolation Forest": "ML", "druga metoda ML": "ML"}
    w = []
    for kat in OPIS_KAT:
        for s, k in [k_ for k_ in kol if KATEGORIA[k_] == kat]:
            d = t[(t.sc == s) & (t.rodzaj == k)]
            kom = [kat, RODZAJ[s][k]]
            for g in ("statystyczne", "fragmenty", "ML"):
                x = d[d.rodzina.map(grupa) == g].sort_values("wykryta", ascending=False).iloc[0]
                kom.append(f"{x.metoda} – {pct(x.wykryta)}")
            w.append(kom)
    out += [tabela(["kategoria", "anomalia", "statystyczne", "fragmenty", "ML"], w), ""]
    w = []
    for kat in OPIS_KAT:
        d = t[t.kategoria == kat]
        w.append([kat] + [pct(d[d.rodzina == r].groupby(["sc", "rodzaj"]).wykryta.max().mean()) for r in RODZINY])
    out += ["Średnio w kategorii (dla każdej rodziny najlepszy wariant):", "",
            tabela(["kategoria"] + list(RODZINY), w), ""]

    # 4. przeniesienie
    if tp is not None:
        for miara, fmt in (("wykryta", lambda v: pct(v, False)), ("auc", f2)):
            out += [f"## Przeniesienie metod – {'odsetek wykrytych' if miara == 'wykryta' else 'AUC'} "
                    "A / B / C (poziomy 2–3, wszystkie L)", ""]
            for s in SC:
                w = []
                for _, _, e in wiersze(uczone=False):
                    kom = [e]
                    for z in SC:
                        x = tp[(tp.etykieta == e) & (tp.metody_z == z) & (tp.szereg == s)].sort_values("rodzaj")
                        kom.append(" / ".join(fmt(v) for v in x[miara]) if len(x) == 3 else "–")
                    w.append(kom)
                out += [f"Szeregi {OPIS_SC[s]}:", "",
                        tabela(["metoda"] + [f"metody {z}" + (" (u siebie)" if z == s else "") for z in SC], w), ""]
        w = []
        for s in SC:
            x = tp[tp.szereg == s].groupby(["metody_z", "etykieta"]).wykryta.mean()
            kom = [OPIS_SC[s]]
            for z in SC:
                v = x.loc[z]
                kom.append(f"{pct(v.mean())} (najlepsza: {v.idxmax()} – {pct(v.max())})")
            w.append(kom)
        out += ["Średnio dla wszystkich wariantów metod (A, B, C łącznie) i najlepszy wariant:", "",
                tabela(["szeregi"] + [f"metody {z}" for z in SC], w), ""]
        pc = pd.read_csv(KAT / "przeniesienie_czyste.csv")
        f = pc.groupby(["metody_z", "szereg", "metoda"]).fa_epizody.apply(lambda v: (v > 0).mean())
        out += ["Fałszywe alarmy na walidacji (200 szeregów bez anomalii) – metody przeniesione:", "",
                tabela(["metody z", "szeregi", "metoda", "szeregi z ≥ 1 fałszywym alarmem"],
                       [[a, b, m, f"{100 * v:.1f}%".replace(".", ",")] for (a, b, m), v in f.items()]), ""]

    # 5. cechy 5 vs 3, uczenie
    w = []
    for r in ("kNN", "Isolation Forest", "druga metoda ML"):
        kom = [r]
        for s in SC:
            a, b = nazwa(r, s, ""), nazwa(r, s, "3")
            if b is None:
                kom.append("–")
                continue
            x5 = [t[(t.metoda == a) & (t.sc == s) & (t.rodzaj == k)].wykryta.iloc[0] for k in (1, 2, 3)]
            x3 = [t[(t.metoda == b) & (t.sc == s) & (t.rodzaj == k)].wykryta.iloc[0] for k in (1, 2, 3)]
            kom.append(" / ".join(pct(v, False) for v in x5) + " → " + " / ".join(pct(v, False) for v in x3))
        w.append(kom)
    out += ["## Zestaw cech: 5 cech → 3 cechy (odsetek wykrytych A / B / C)", "", tabela(["metoda"] + list(SC), w), ""]
    w = []
    for r in ML:
        for c in ("", "3"):
            kom = [r + (" – 3 cechy" if c else " – 5 cech")]
            for s in SC:
                a, b = nazwa(r, s, c), nazwa(r, s, c + "U")
                if a is None or b is None:
                    kom.append("–")
                    continue
                xa = [t[(t.metoda == a) & (t.sc == s) & (t.rodzaj == k)].wykryta.iloc[0] for k in (1, 2, 3)]
                xb = [t[(t.metoda == b) & (t.sc == s) & (t.rodzaj == k)].wykryta.iloc[0] for k in (1, 2, 3)]
                kom.append(" / ".join(pct(v, False) for v in xa) + " → " + " / ".join(pct(v, False) for v in xb))
            w.append(kom)
    out += ["## Uczenie: ten sam szereg → szeregi bez anomalii (odsetek wykrytych A / B / C)", "",
            tabela(["metoda"] + list(SC), sorted(w)), ""]

    # 6. właściwości szeregu
    for s in SC:
        czesc, kol_, opis = WLASCIWOSCI[s]
        d = wczytaj(s, czesc)
        v = sorted(d[kol_].unique())
        w = []
        for r, wa, e in wiersze(uczone=False):
            m = nazwa(r, s, wa)
            x = d[d.metoda == m]
            if m is None or x.empty:
                continue
            g = x.groupby([kol_, "rodzaj"]).wykryta.mean()
            w.append([e] + [" / ".join(pct(g.get((u, k)), False) for k in (1, 2, 3)) for u in v])
        out += [f"## Właściwość szeregu {s}: {opis} – odsetek wykrytych A / B / C", "",
                tabela(["metoda"] + [f"{kol_} = {u:g}".replace(".", ",") for u in v], w), ""]

    # 7. okno i długość anomalii
    for kol_, czesc in (("w", "okno"), ("L", "glowny")):
        w = []
        for r, wa, e in wiersze(uczone=False):
            kom = [e]
            for s in SC:
                d = gl[s] if czesc == "glowny" else wczytaj(s, "okno")
                x = d[d.metoda == nazwa(r, s, wa)]
                kom.append(" / ".join(pct(v, False) for v in x.groupby(kol_).wykryta.mean()) if len(x) else "–")
            w.append(kom)
        wart = {s: sorted((gl[s] if czesc == "glowny" else wczytaj(s, "okno"))[kol_].unique()) for s in SC}
        out += [f"## {'Długość okna w' if kol_ == 'w' else 'Długość anomalii L'} – odsetek wykrytych (średnio A, B, C)",
                "", tabela(["metoda"] + [f"{s}: {kol_} = " + " / ".join(map(str, wart[s])) for s in SC], w), ""]

    # 8. fałszywe alarmy i lokalizacja
    wal = pd.read_csv(WYNIKI / "walidacja_progow.csv").set_index(["scenariusz", "metoda"]).odsetek_z_fa
    w = []
    for r, wa, e in wiersze():
        kom = [e]
        for s in SC:
            m = nazwa(r, s, wa)
            if m is None:
                kom.append("–")
                continue
            cz = wczytaj(s, "glowny", "czyste")
            sz = wczytaj(s, "glowny", "szeregi")
            fa = (cz[cz.metoda == m].fa_epizody > 0).mean()
            kom.append(f"{100 * fa:.1f}% / {100 * wal.get((s, m), np.nan):.1f}% / {f2(sz[sz.metoda == m].f1.mean())}"
                       .replace(".", ","))
        w.append(kom)
    out += ["## Fałszywe alarmy (walidacja 200 / sprawdzenie na 1 000 szeregów) i F1", "",
            tabela(["metoda"] + list(SC), w), ""]
    KAT.mkdir(parents=True, exist_ok=True)
    (KAT / "podsumowanie.md").write_text("\n".join(out), encoding="utf-8")


def main():
    WYKRESY.mkdir(parents=True, exist_ok=True)
    gl = glowne()
    t = wykrywalnosc(gl)
    mapa(t, "wykryta", "por_mapa_wykrywalnosci.png",
         "Odsetek wykrytych anomalii (%) – wszystkie scenariusze, część główna (wszystkie poziomy i L)")
    wykres_wlasciwosci("por_wlasciwosci.png")
    wykres_okno_dlugosc(gl, "por_okno_dlugosc.png")
    tp = przeniesienie()
    if tp is not None:
        wykres_przeniesienie(tp, "por_przeniesienie.png")
    podsumowanie(gl, t, tp)
    print("Zapisano wykresy por_*.png i", KAT / "podsumowanie.md")


if __name__ == "__main__":
    main()
