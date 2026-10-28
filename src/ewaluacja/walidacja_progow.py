"""Niezależne sprawdzenie progów alarmu na 1 000 NOWYCH szeregach bez anomalii - wszystkie scenariusze (S1, S2, S3).

W eksperymentach próg każdej metody ustalany jest na 200 szeregach bez anomalii (fałszywy alarm w 10% z nich)
i sprawdzany na 200 kolejnych. 200 szeregów daje odsetek z niepewnością ±4 punkty procentowe (95%), a dla metod
z ciężkim ogonem maksimum wyniku - więcej. Tu ta sama kalibracja (te same 200 szeregów, te same stałe metod i progi
co w części głównej eksperymentu każdego scenariusza) i sprawdzenie na 1 000 szeregach o ziarnach 20 000 000 + i,
nieużywanych nigdzie indziej.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.ewaluacja.walidacja_progow            # ok. 1,5 h na 2 rdzeniach
    python -m src.ewaluacja.walidacja_progow s1 s2      # wybrane scenariusze
Wyniki: wyniki/walidacja_progow.csv, wyniki/walidacja_progow.md
"""
import os
import sys

import pandas as pd

from src.config import WYNIKI
from src.ewaluacja.eksperyment import Eksperyment, _wyniki_czyste
from src.ewaluacja.metryki import fa_na_szeregu_czystym
from src.ewaluacja.wykresy_s1 import wilson
from src.metody import metody_s1, metody_s2, metody_s3
from src.symulacje.anomalie import RODZAJE_S1, RODZAJE_S2, RODZAJE_S3, szereg_s1, szereg_s2, szereg_s3

N = 1000
ZIARNO = 20_000_000
SCENARIUSZE = {
    "s1": ({"generuj": szereg_s1, "rodzaje": RODZAJE_S1}, lambda: metody_s1(w=50, uczone=True)),
    "s2": ({"generuj": szereg_s2, "rodzaje": RODZAJE_S2}, lambda: metody_s2(w=50, uczone=True)),
    "s3": ({"generuj": szereg_s3, "rodzaje": RODZAJE_S3}, lambda: metody_s3(w=50, uczone=True)),
}


def waliduj(nazwa, procesy):
    scen, fabryka = SCENARIUSZE[nazwa]
    metody = fabryka()
    e = Eksperyment(scen, WYNIKI, procesy)
    progi, _ = e.kalibruj(metody, {})
    wyn = e.pula.map(_wyniki_czyste, [(scen, metody, ZIARNO + i, {}) for i in range(N)], chunksize=8)
    e.zamknij()
    wiersze = []
    for j, m in enumerate(metody):
        fa = [fa_na_szeregu_czystym(w[j], progi[j]) for _, w in wyn]
        k = sum(f > 0 for f in fa)
        lo, hi = wilson(k, N)
        wiersze.append({"scenariusz": nazwa.upper(), "metoda": m.nazwa, "prog": progi[j], "n": N,
                        "odsetek_z_fa": k / N, "ci_lo": lo, "ci_hi": hi, "fa_na_szereg": sum(fa) / N})
        print(f"  {nazwa} {m.nazwa}: {k / N:.1%} [{lo:.1%}; {hi:.1%}]", flush=True)
    return wiersze


if __name__ == "__main__":
    wybrane = [a for a in sys.argv[1:] if a in SCENARIUSZE] or list(SCENARIUSZE)
    plik = WYNIKI / "walidacja_progow.csv"
    stare = pd.read_csv(plik) if plik.exists() else pd.DataFrame()
    nowe = pd.DataFrame([w for s in wybrane for w in waliduj(s, max(1, os.cpu_count() or 1))])
    if not stare.empty:
        stare = stare[~stare.scenariusz.isin([s.upper() for s in wybrane])]
    tab = pd.concat([stare, nowe]).sort_values("scenariusz", kind="stable")
    tab.to_csv(plik, index=False)
    linie = ["# Sprawdzenie progów alarmu na 1 000 nowych szeregach bez anomalii", "",
             "Generowane przez `src/ewaluacja/walidacja_progow.py`. Cel kalibracji: fałszywy alarm w 10% szeregów "
             "bez anomalii. Przedział ufności 95% (Wilsona).", "",
             "| scenariusz | metoda | szeregi z ≥ 1 fałszywym alarmem | 95% CI | fałszywe alarmy na szereg |",
             "|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        linie.append(f"| {r.scenariusz} | {r.metoda} | {r.odsetek_z_fa:.1%} | {r.ci_lo:.1%} – {r.ci_hi:.1%} | "
                     f"{r.fa_na_szereg:.3f} |".replace(".", ","))
    (WYNIKI / "walidacja_progow.md").write_text("\n".join(linie) + "\n", encoding="utf-8")
    print("Zapisano", plik)
