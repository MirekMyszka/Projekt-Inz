"""Testy porównania S1–S3: rodziny metod, ziarna eksperymentu przeniesienia."""
from src.ewaluacja.porownanie import DLUGOSCI, METODY, ZIARNO_0, pary, zadania
from src.ewaluacja.wykresy_porownanie import RODZINY, WARIANTY, nazwa
from src.metody import metody_s1, metody_s2, metody_s3


def test_nazwy_rodzin_istnieja_w_zestawach_metod():
    """Każdy wariant metody w tabelach porównania to metoda z zestawu danego scenariusza (część główna)."""
    zestawy = {"S1": metody_s1(uczone=True), "S2": metody_s2(uczone=True), "S3": metody_s3(uczone=True)}
    for sc, metody in zestawy.items():
        nazwy = {m.nazwa for m in metody}
        for r in RODZINY:
            for w in WARIANTY:
                m = nazwa(r, sc, w)
                assert m is None or m in nazwy, (sc, r, w, m)


def test_kazda_metoda_glowna_nalezy_do_rodziny():
    """Wszystkie metody główne (bez porównawczych) są w którejś rodzinie - porównanie niczego nie pomija."""
    zestawy = {"S1": metody_s1(uczone=True, z_porownawczymi=False), "S2": metody_s2(uczone=True),
               "S3": metody_s3(uczone=True, z_porownawczymi=False)}
    for sc, metody in zestawy.items():
        w_rodzinach = {nazwa(r, sc, w) for r in RODZINY for w in WARIANTY}
        assert {m.nazwa for m in metody} <= w_rodzinach, sc


def test_przeniesienie_na_szeregach_czesci_glownej():
    """Ziarna przeniesienia = ziarna części głównej scenariusza docelowego (poziomy 2–3) - ocena sparowana."""
    assert ZIARNO_0 == {"S1": 0, "S2": 3_100_000, "S3": 6_100_000}
    z = zadania("S2", 100)
    assert len(z) == 2 * 3 * 100
    assert z[0] == (3_100_000 + 200_000, 2, 20)
    assert zadania("S3", 2)[-1] == (6_100_000 + 300_000 + 20_000 + 1, 3, 200)
    assert DLUGOSCI["S3"] == (50, 100, 200)
    assert len(pary()) == 6 and all(a != b for a, b in pary())
    assert all(len(METODY[s]()) >= 8 for s in METODY)
