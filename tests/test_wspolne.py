"""Testy elementów wspólnych dla scenariuszy: odporna skala szumu.

Uruchomienie (z katalogu głównego repozytorium):
    python -m pytest tests -q
"""
import numpy as np

from src.metody.baza import odporna_skala


def test_odporna_skala_bez_anomalii_jest_nieobciazona():
    rng = np.random.default_rng(0)
    s = [odporna_skala(rng.normal(0, 2.0, 3000)) for _ in range(200)]
    assert abs(np.mean(s) - 2.0) < 0.02


def test_odporna_skala_nie_rosnie_przez_anomalie_zbiorowe():
    """Trzy długie anomalie (20% punktów, zmienność ×3): MAD zawyża skalę o > 10%, mediana wariancji okien - o < 4%."""
    rng = np.random.default_rng(1)
    s_okna, s_mad = [], []
    for _ in range(100):
        x = rng.normal(0, 1.0, 3000)
        for start in (300, 1300, 2300):
            x[start:start + 200] *= 3.0
        s_okna.append(odporna_skala(x))
        s_mad.append(1.4826 * np.median(np.abs(x - np.median(x))))
    assert np.mean(s_mad) > 1.10
    assert abs(np.mean(s_okna) - 1.0) < 0.04


def test_pelt_wynik_tylko_dla_odcinka_innego_niz_tlo():
    """PELT: odcinek ze zmianą poziomu dostaje wynik > 3, punkty normy (tło i krótkie odcinki normy) - 0."""
    from src.metody.statystyczne import PELT, maska_tla
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 3000)
    x[1000:1100] += 3.0
    m = PELT(kara=60)
    w = m.wynik(x)
    assert w[1010:1090].min() > 3
    assert (w[:950] == 0).all() and (w[1150:] == 0).all()
    assert maska_tla([1000, 1100, 3000], 3000).sum() == 1900          # dwa najdłuższe odcinki normy


# ------------------------------------------------------------------ zestawy 3 / 5 cech i uczenie na szeregach bez anomalii
from src.metody import LOF, Autoenkoder, CechyKNN, IsolationForest, OneClassSVM  # noqa: E402
from src.metody.baza import KOLUMNY_3  # noqa: E402
from src.metody.fragmenty import cechy_okien  # noqa: E402
from src.symulacje.anomalie import szereg_s1, szereg_s2, szereg_s3  # noqa: E402

GEN = {"s1": szereg_s1, "s2": szereg_s2, "s3": szereg_s3}


def test_zestaw_3_cech_to_podzbior_zestawu_5_cech():
    for zestaw, gen in GEN.items():
        x = gen(3).x
        _, F5 = cechy_okien(x, 50, 5, zestaw, 50, 5)
        _, F3 = cechy_okien(x, 50, 5, zestaw, 50, 3)
        assert F5.shape[1] == 5 and F3.shape[1] == 3
        assert np.allclose(F3, F5[:, KOLUMNY_3[zestaw]])          # standaryzacja kolumnami - podzbiór się zgadza


def test_korelacja_z_wzorcem_cyklu_s3():
    """Cecha pomocnicza S3: ≈ 1 dla okna z typowym cyklem, ≈ 0 dla okna przesuniętego o π/2."""
    from src.metody.baza import cechy_okien_s3, okna
    t = np.arange(3000)
    x = 3 * np.sin(2 * np.pi * t / 50 + 0.7)
    x[1000:1100] = 3 * np.sin(2 * np.pi * np.arange(1000, 1100) / 50 + 0.7 + np.pi / 2)
    starty, m = okna(x, 50, 5)
    F = cechy_okien_s3(m, starty, 50)
    assert np.median(F[(starty < 900) | (starty > 1200), 4]) > 0.99
    assert np.abs(F[(starty >= 1010) & (starty <= 1040), 4]).max() < 0.05


def test_nazwy_wariantow():
    assert CechyKNN(cechy=3).nazwa == "cechy okien + kNN (3 cechy)"
    assert CechyKNN(zestaw="s3").nazwa == "cechy cykli + kNN (5 cech)"
    assert IsolationForest(cechy=5, nauka="czyste").nazwa == "Isolation Forest (5 cech, uczony na szeregach bez anomalii)"
    assert OneClassSVM(cechy=3).nazwa == "One-Class SVM (3 cechy)"


def test_uczenie_na_szeregach_bez_anomalii():
    """Model uczony raz na osobnych szeregach; wynik nie zależy od kolejności ocenianych szeregów, a anomalia
    (silna zmiana zmienności w S1) dostaje wyższy wynik niż norma."""
    uczace = [szereg_s1(9_500_000 + i, z_anomaliami=False).x for i in range(20)]
    s = szereg_s1(11, poziom=4, L=100)
    for m in (IsolationForest(nauka="czyste"), LOF(nauka="czyste"), OneClassSVM(zestaw="s1", nauka="czyste")):
        m.ucz(uczace)
        w1 = m.wynik(s.x)
        m.wynik(szereg_s1(12).x)
        assert np.array_equal(w1, m.wynik(s.x)), m.nazwa
        b = next(a for a in s.anomalie if a.rodzaj == 2)
        assert np.median(w1[b.start:b.koniec]) > np.quantile(w1, 0.9), m.nazwa
    ae = Autoenkoder(nauka="czyste")
    ae.ucz([szereg_s3(9_500_000 + i, z_anomaliami=False).x for i in range(5)])
    assert ae.siec is not None and ae.odrzuc == 0.1
