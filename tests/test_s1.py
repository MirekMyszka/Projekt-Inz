"""Testy scenariusza S1: generator, anomalie, ocena i interfejs metod.

Uruchomienie (z katalogu głównego repozytorium):
    python -m pytest tests -q
"""
import numpy as np
import pytest

from src.ewaluacja.metryki import epizody, kalibruj_prog, maska_stref, ocen_szereg
from src.metody import metody_s1
from src.metody import CUSUM
from src.metody.statystyczne import cusum_offline, pelt
from src.symulacje.anomalie import MARGINES, Anomalia, SzeregTestowy, rozmiesc, szereg_s1
from src.symulacje.generator import ar1


# ------------------------------------------------------------------------------------------- generator
def test_powtarzalnosc():
    a, b = szereg_s1(123, poziom=3, L=50), szereg_s1(123, poziom=3, L=50)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.etykiety, b.etykiety)
    assert not np.array_equal(a.x, szereg_s1(124, poziom=3, L=50).x)


@pytest.mark.parametrize("L", [20, 50, 100])
def test_trzy_rozne_anomalie_rozlaczne(L):
    for z in range(30):
        s = szereg_s1(z, L=L)
        assert sorted(a.rodzaj for a in s.anomalie) == [1, 2, 3]
        assert all(a.dlugosc == L for a in s.anomalie)
        for r in (1, 2, 3):
            assert (s.etykiety == r).sum() == L
        st = sorted(a.start for a in s.anomalie)
        assert st[0] >= MARGINES and st[-1] + L <= len(s.x) - MARGINES
        assert all(st[i + 1] - (st[i] + L) >= 2 * MARGINES for i in range(2))


def test_szereg_bez_anomalii():
    s = szereg_s1(5, z_anomaliami=False)
    assert s.anomalie == [] and s.etykiety.sum() == 0


def test_rozmiesc_za_dlugie_L():
    with pytest.raises(ValueError):
        rozmiesc(3000, 900, np.random.default_rng(0))


def test_ar1_zachowuje_wariancje_i_ma_zaleznosc():
    rng = np.random.default_rng(0)
    x = ar1(200_000, 0.7, 1.0, rng)
    assert abs(x.var() - 1.0) < 0.03
    assert abs(np.corrcoef(x[1:], x[:-1])[0, 1] - 0.7) < 0.01


def test_anomalie_maja_zamierzony_efekt():
    """Na wielu szeregach: S1-A przesuwa średnią o Δ, S1-B mnoży odch. std przez c, S1-C nie zmienia wariancji,
    ale daje autokorelację ≈ φ."""
    sr, od, war, ac = [], [], [], []
    for z in range(300):
        s = szereg_s1(z, poziom=3, L=100)          # Δ = 2, c = 3, φ = 0,7
        for a in s.anomalie:
            f = s.x[a.start:a.koniec]
            if a.rodzaj == 1:
                sr.append(f.mean())
            elif a.rodzaj == 2:
                od.append(f.std())
            else:
                war.append(f.var())
                ac.append(np.corrcoef(f[1:], f[:-1])[0, 1])
    assert abs(np.mean(sr) - 2.0) < 0.05
    assert abs(np.mean(od) - 3.0) < 0.1
    assert abs(np.mean(war) - 1.0) < 0.1           # wariancja jak w normie
    assert abs(np.mean(ac) - 0.7) < 0.05


# ----------------------------------------------------------------------------------------------- ocena
def _szereg(n, anomalie):
    et = np.zeros(n, int)
    for a in anomalie:
        et[a.start:a.koniec] = a.rodzaj
    return SzeregTestowy(np.zeros(n), et, anomalie, "test", {}, 0)


def test_epizody():
    assert epizody(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1], bool)) == [(1, 3), (5, 6), (7, 9)]
    assert epizody(np.zeros(5, bool)) == []


def test_strefa_tolerancji_i_falszywe_alarmy():
    s = _szereg(1000, [Anomalia(1, 100, 150), Anomalia(2, 500, 520), Anomalia(3, 800, 850)])
    w = np.zeros(1000)
    w[160:170] = 5        # tuż za anomalią 1, w strefie (tol = 25) → wykrycie, nie fałszywy alarm
    w[300:305] = 5        # daleko od anomalii → fałszywy alarm
    w[810:820] = 5        # w środku anomalii 3 → wykrycie
    r = ocen_szereg(w, s, prog=1.0, tol=25)
    assert r["wykryte"] == {1: True, 2: False, 3: True}
    assert r["fa_epizody"] == 1
    # punktowo: TP = 10 (810–819), FP = 5 (300–304), 160–169 pominięte (margines)
    assert r["precyzja"] == pytest.approx(10 / 15)
    assert r["czulosc_pkt"] == pytest.approx(10 / 120)
    assert maska_stref(s.anomalie, 1000, 25)[75] and not maska_stref(s.anomalie, 1000, 25)[74]


def test_auc_idealny_i_losowy():
    s = _szereg(1000, [Anomalia(1, 100, 150), Anomalia(2, 500, 550), Anomalia(3, 800, 850)])
    w = s.etykiety.astype(float)                  # idealny wynik
    r = ocen_szereg(w, s, prog=0.5, tol=25)
    assert all(v == 1.0 for v in r["auc"].values()) and r["f1"] == 1.0
    rng = np.random.default_rng(0)
    r = ocen_szereg(rng.normal(size=1000), s, prog=10, tol=25)
    assert all(0.3 < v < 0.7 for v in r["auc"].values())


def test_kalibracja_progu():
    rng = np.random.default_rng(1)
    czyste = [rng.normal(size=500) for _ in range(1000)]
    prog = kalibruj_prog(czyste, alfa=0.1)
    nowe = [rng.normal(size=500) for _ in range(2000)]
    odsetek = np.mean([w.max() > prog for w in nowe])
    assert abs(odsetek - 0.1) < 0.025


# ---------------------------------------------------------------------------------------------- metody
def test_cusum_offline_wysoki_tylko_w_anomalii():
    z = np.zeros(1000)
    z[400:500] = 2.0
    c = cusum_offline(z, 0.5)
    assert c[400] == pytest.approx(150.0) and c[499] == pytest.approx(150.0)   # (δ − k)·L = 1,5·100
    assert c[300] == 0 and c[520] == 0 and c[600] == 0                         # nic przed i za anomalią


def test_karty_cusum_nie_przeciekaja():
    """Zmiana poziomu nie może podnosić kart zmienności i zależności (liczonych na różnicach)."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=3000)
    x[1000:1050] += 3.0
    k = CUSUM().karty(x)
    tlo = CUSUM().karty(rng.normal(size=3000)).max(axis=1)
    assert k[0, 1000:1050].max() > 5 * tlo[0]            # karta poziomu reaguje
    assert k[1].max() < 3 * tlo[1] and k[2].max() < 3 * tlo[2]   # karty zmienności i zależności - nie


def test_pelt_znajduje_granice():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    y[400:500] += 3
    g = pelt([y], kara=30)
    assert any(abs(b - 400) <= 5 for b in g) and any(abs(b - 500) <= 5 for b in g)


def test_interfejs_metod():
    s = szereg_s1(7, poziom=4, L=50)
    for m in metody_s1():
        w = m.wynik(s.x)
        assert w.shape == s.x.shape and np.all(np.isfinite(w)), m.nazwa
