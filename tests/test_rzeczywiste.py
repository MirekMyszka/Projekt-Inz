"""Testy etapu 2 (dane rzeczywiste): cechy cykli względem profilu, próg i kara na dniach, katalog zdarzeń."""
import numpy as np

from src.metody.baza import cechy_okien_rz, okna, profil_okresowy
from src.rzeczywiste.katalog import OPIS_TYPU, STUDIA, katalog
from src.rzeczywiste.wspolne import KARY, kara_pelt, odsetek_dni_z_alarmem, prog_dni


def _szereg_tygodniowy(tygodnie=6, P=336, doba=48, ziarno=0):
    rng = np.random.default_rng(ziarno)
    t = np.arange(tygodnie * P)
    cykl = np.sin(2 * np.pi * t / doba) + 0.5 * (t % P >= 5 * doba)      # weekend - wyższy poziom
    return cykl + 0.05 * rng.standard_normal(len(t))


def test_profil_okresowy_to_mediana_okresow():
    x = np.tile(np.arange(10.0), 5)
    x[3] = 100.0                                   # jedna wartość odstająca nie zmienia mediany
    assert np.allclose(profil_okresowy(x, 10), np.arange(10.0))


def test_cechy_rz_doba_typowa_i_nietypowa():
    x = _szereg_tygodniowy()
    starty, m = okna(x, 48, 48)
    F = cechy_okien_rz(m, starty, x, 336)
    assert F.shape == (len(starty), 5)
    assert np.abs(F[:, 0]).max() < 0.1            # poziom: typowe doby nie odbiegają od profilu
    assert F[:, 1].min() > 0.95                   # kształt: korelacja z profilem
    assert np.abs(F[:, 2]).max() < 0.2            # faza
    y = x.copy()
    y[48 * 10:48 * 11] = y[48 * 10:48 * 11] - 1.0                            # doba o niższym poziomie
    y[48 * 20:48 * 21] = np.roll(y[48 * 20:48 * 21], 12)                    # doba przesunięta o 6 h
    G = cechy_okien_rz(okna(y, 48, 48)[1], starty, y, 336)
    assert G[10, 0] < -0.9 and abs(G[20, 2]) > 1.0


def test_prog_dni_daje_alarm_w_alfa_dni():
    rng = np.random.default_rng(1)
    wynik = rng.standard_normal(48 * 200)
    dzien = np.arange(len(wynik)) // 48
    dni = np.arange(200)
    prog = prog_dni(wynik, dzien, dni, 0.10)
    assert abs(odsetek_dni_z_alarmem(wynik > prog, dzien, dni) - 0.10) < 0.01


def test_kara_pelt_wykorzystuje_budzet_i_dziala_przy_niemonotonicznosci():
    dzien = np.arange(100) // 10
    dni = np.arange(10)

    def wynik(k):                                  # odsetek dni z alarmem: 30% dla małych kar, 10% w środku, 0% dla dużych
        w = np.zeros(100)
        dni_alarmu = 3 if k < 20 else (1 if k < 200 else 0)
        w[:dni_alarmu * 10] = 1.0
        return w
    k = kara_pelt(wynik, dzien, dni, 0.10)
    assert 20 <= k < 200 and k == min(x for x in KARY if 20 <= x < 200)


def test_katalog_spojny():
    k = katalog()
    assert set(k.zbior) == {"SPY", "BTC"}
    assert set(k.typ) <= set(OPIS_TYPU)
    assert (k.od <= k["do"]).all()
    assert len(k) == len(k.drop_duplicates(["zbior", "typ", "od"]))
    assert len(STUDIA) == 4
