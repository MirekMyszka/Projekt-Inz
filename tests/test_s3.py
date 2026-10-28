"""Testy scenariusza S3 (szum + sinus): generator, anomalie okresowości, cechy cykli, reszty cyklu, interfejs metod.

Uruchomienie (z katalogu głównego repozytorium):
    python -m pytest tests -q
"""
import numpy as np
import pytest

from src.metody import Autoenkoder, PELTSTL, STLEWMA, metody_s3
from src.metody.baza import cechy_okien_s3, okna, reszty_harmoniczne, reszty_stl
from src.metody.statystyczne import ewma_dwustronna
from src.symulacje.anomalie import LS_S3, MARGINES, POZIOMY_S3, szereg_s3


def test_tlo_to_sinus_z_szumem():
    s = szereg_s3(4, z_anomaliami=False)
    t = np.arange(len(s.x))
    czysty = 3 * np.sin(2 * np.pi * t / 50 + s.parametry["theta"])
    e = s.x - czysty
    assert abs(e.mean()) < 0.1 and abs(e.std() - 1.0) < 0.05
    assert abs(s.x.var() - (1 + 9 / 2)) < 0.4                 # wariancja = σ² + A²/2


def test_powtarzalnosc_i_rozmieszczenie():
    a, b = szereg_s3(11, poziom=3, L=200), szereg_s3(11, poziom=3, L=200)
    assert np.array_equal(a.x, b.x)
    for L in LS_S3:
        for z in range(20):
            s = szereg_s3(z, L=L)
            assert sorted(an.rodzaj for an in s.anomalie) == [1, 2, 3]
            assert all(an.dlugosc == L for an in s.anomalie)
            st = sorted(s.anomalie, key=lambda an: an.start)
            assert st[0].start >= MARGINES and st[-1].koniec <= len(s.x) - MARGINES


def test_anomalie_maja_zamierzony_efekt():
    """Faza: przebieg = sinus przesunięty o Δθ; amplituda: k·A; okres: r·P z fazą ciągłą na początku fragmentu."""
    for z in range(10):
        s = szereg_s3(z, poziom=3, L=100, sigma=1e-9)      # bez szumu - porównanie z wzorem
        th = s.parametry["theta"]
        for an in s.anomalie:
            t = np.arange(an.start, an.koniec)
            if an.rodzaj == 1:
                wzor = 3 * np.sin(2 * np.pi * t / 50 + th + 0.5 * np.pi)
            elif an.rodzaj == 2:
                wzor = 0.25 * 3 * np.sin(2 * np.pi * t / 50 + th)
            else:
                wzor = 3 * np.sin(2 * np.pi * an.start / 50 + th + 2 * np.pi * (t - an.start) / (0.6 * 50))
            assert np.allclose(s.x[an.start:an.koniec], wzor, atol=1e-6)
            # poza fragmentem - zwykły cykl
            assert np.isclose(s.x[an.koniec], 3 * np.sin(2 * np.pi * an.koniec / 50 + th), atol=1e-6)


def test_cechy_cykli_odczytuja_parametry():
    rng = np.random.default_rng(0)
    t = np.arange(3000)
    x = 2.0 * np.sin(2 * np.pi * t / 50 + 1.0) + 0.1 * rng.normal(size=3000)
    x[1000:1100] = 2.0 * np.sin(2 * np.pi * np.arange(1000, 1100) / 50 + 1.0 + np.pi / 2)
    starty, m = okna(x, 50, 5)
    F = cechy_okien_s3(m, starty, 50)
    norma = (starty < 900) | (starty > 1200)
    assert np.allclose(np.median(F[norma, 0]), 2.0, atol=0.02)          # amplituda
    assert np.abs(F[norma, 1]).max() < 0.1                              # faza typowa
    assert np.allclose(np.median(F[norma, 2]), 1.0, atol=0.05)          # 1 cykl na P
    srodek = (starty >= 1010) & (starty <= 1040)
    assert np.allclose(F[srodek, 1], np.pi / 2, atol=0.05)              # przesunięcie fazy o π/2


def test_reszty_cyklu_usuwaja_cykl():
    s = szereg_s3(2, z_anomaliami=False)
    for r in (reszty_stl(s.x, 50), reszty_harmoniczne(s.x, 50)):
        assert abs(r.std() - 1.0) < 0.1                                  # zostaje tylko szum


def test_ewma_dwustronna_jest_zlokalizowana():
    """Minimum kart w przód i w tył: wysokie na fragmencie z nadwyżką, zero poza nim (bez „ogona”)."""
    z = np.zeros(400)
    z[150:250] = 1.0
    e = ewma_dwustronna(z, 0.05)
    assert e[200] > 0.9 and e[:150].max() == 0 and e[250:].max() == 0
    assert np.allclose(e[160], e[239])                                   # symetria względem środka fragmentu


@pytest.mark.parametrize("poziom", list(POZIOMY_S3))
def test_interfejs_metod_s3(poziom):
    s = szereg_s3(7, poziom=poziom, L=100)
    for m in metody_s3() + [STLEWMA(sezon=7), PELTSTL(model="harmoniczny"), Autoenkoder(odrzuc=0.0)]:
        w = m.wynik(s.x)
        assert w.shape == s.x.shape and np.all(np.isfinite(w)), m.nazwa
