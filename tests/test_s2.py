"""Testy scenariusza S2 (AR(1)): generator, anomalie, model AR, PELT z kosztem AR, interfejs metod.

Uruchomienie (z katalogu głównego repozytorium):
    python -m pytest tests -q
"""
import numpy as np
import pytest

from src.metody import CUSUMHybryda, metody_s2
from src.metody.baza import dopasuj_ar1
from src.metody.statystyczne import _koszt_ar1, pelt_ogolny
from src.symulacje.anomalie import MARGINES, POZIOMY_S2, dlugosc_szoku, szereg_s2


def test_tlo_ma_zadana_wariancje_i_zaleznosc():
    x = np.concatenate([szereg_s2(z, z_anomaliami=False).x for z in range(60)])
    assert abs(x.var() - 1.0) < 0.03
    assert abs(np.corrcoef(x[1:], x[:-1])[0, 1] - 0.7) < 0.01


def test_powtarzalnosc_i_rozmieszczenie():
    a, b = szereg_s2(11, poziom=3, L=100), szereg_s2(11, poziom=3, L=100)
    assert np.array_equal(a.x, b.x)
    for z in range(30):
        s = szereg_s2(z, L=100)
        assert sorted(an.rodzaj for an in s.anomalie) == [1, 2, 3]
        dl = {an.rodzaj: an.dlugosc for an in s.anomalie}
        assert dl[1] == dlugosc_szoku(0.7) == 10 and dl[2] == dl[3] == 100
        st = sorted(s.anomalie, key=lambda an: an.start)
        assert st[0].start >= MARGINES and st[-1].koniec <= len(s.x) - MARGINES


def test_anomalie_maja_zamierzony_efekt():
    """S2-B: na fragmencie φ' i ta sama wariancja; S2-C: średnia fragmentu dąży do Δ; S2-A: skok ≈ impuls·σ_ε."""
    ac, war, sr, skok = [], [], [], []
    se = np.sqrt(1 - 0.7 ** 2)
    for z in range(300):
        s = szereg_s2(z, poziom=3, L=100)          # impuls 5σ_ε, φ' = 0, Δ = 2
        for an in s.anomalie:
            f = s.x[an.start:an.koniec]
            if an.rodzaj == 2:
                ac.append(np.corrcoef(f[1:], f[:-1])[0, 1])
                war.append(f.var())
            elif an.rodzaj == 3:
                sr.append(f[20:].mean())               # po dojściu do nowego poziomu (φ^20 ≈ 0)
            else:
                skok.append(s.x[an.start] - 0.7 * s.x[an.start - 1])
    assert abs(np.mean(ac) - 0.0) < 0.03
    assert abs(np.mean(war) - 1.0) < 0.1
    assert abs(np.mean(sr) - 2.0) < 0.1
    assert abs(np.mean(skok) - 5 * se) < 0.1


def test_dopasowanie_ar1():
    s = szereg_s2(3, z_anomaliami=False, n=20_000)
    mu, phi, s_e, r = dopasuj_ar1(s.x)
    assert abs(phi - 0.7) < 0.02 and abs(mu) < 0.05 and abs(s_e - np.sqrt(0.51)) < 0.02
    assert abs(np.corrcoef(r[2:], r[1:-1])[0, 1]) < 0.03        # reszty ≈ biały szum


def test_pelt_ar_znajduje_zmiane_phi():
    rng = np.random.default_rng(0)
    phi = np.full(2000, 0.7)
    phi[800:1000] = -0.5
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = phi[t] * x[t - 1] + np.sqrt(1 - phi[t] ** 2) * rng.normal()
    g = pelt_ogolny(_koszt_ar1(x), len(x), kara=30)
    assert any(abs(b - 800) <= 10 for b in g) and any(abs(b - 1000) <= 10 for b in g)


@pytest.mark.parametrize("poziom", list(POZIOMY_S2))
def test_interfejs_metod_s2(poziom):
    s = szereg_s2(7, poziom=poziom, L=50)
    for m in metody_s2() + [CUSUMHybryda()]:
        w = m.wynik(s.x)
        assert w.shape == s.x.shape and np.all(np.isfinite(w)), m.nazwa
