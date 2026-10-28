"""Metody oparte na fragmentach szeregu: Matrix Profile i cechy okien + kNN (docs/plan_etap_1.md, rozdz. 4.3–4.5)."""
import numpy as np
import stumpy

from src.metody.baza import (KOLUMNY_3, Detektor, cechy_okien_s1, cechy_okien_s2, cechy_okien_s3, do_punktow,
                             dopasuj_ar1, odleglosci_bez_trywialnych, okna, standaryzuj_odpornie)


def opis_cech(liczba: int) -> str:
    return "3 cechy" if liczba == 3 else f"{liczba} cech"


def cechy_okien(x: np.ndarray, w: int, krok: int, zestaw: str = "s1", P: int = 50,
                liczba: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Okna szeregu i ich cechy (standaryzowane odpornie) - zestaw cech zależny od scenariusza
    (s1, s2, s3 = cechy cykli; P - okres cyklu, tylko dla s3); liczba = 5 (pełny zestaw) albo 3 (po jednej cesze
    na rodzaj anomalii - baza.KOLUMNY_3)."""
    starty, m = okna(x, w, krok)
    if zestaw == "s1":
        F = cechy_okien_s1(m)
    elif zestaw == "s2":
        _, _, _, r = dopasuj_ar1(x)
        F = cechy_okien_s2(m, okna(r, w, krok)[1])
    elif zestaw == "s3":
        F = cechy_okien_s3(m, starty, P)
    else:
        raise ValueError(f"Nieznany zestaw cech: {zestaw}")
    if liczba == 3:
        F = F[:, KOLUMNY_3[zestaw]]
    elif liczba != 5:
        raise ValueError(f"Liczba cech: 3 albo 5, nie {liczba}")
    return starty, standaryzuj_odpornie(F)


class MatrixProfile(Detektor):
    """FR1 - Matrix Profile: dla każdego fragmentu długości w odległość do NAJBARDZIEJ podobnego innego fragmentu
    szeregu. Duża odległość = fragment bez „bliźniaka” (discord). Pomijane są fragmenty przesunięte o mniej niż w/4
    (standardowa strefa wykluczenia biblioteki stumpy i literatury o Matrix Profile; sprawdzono, że strefa = w
    zmienia wyniki S1 nieznacznie).

    normalizacja=False (stumpy.aamp) - zwykła odległość euklidesowa; widzi zmiany poziomu i zmienności.
    normalizacja=True  (stumpy.stump) - każdy fragment jest standaryzowany (średnia 0, odch. 1), więc porównywany jest
    tylko KSZTAŁT - wersja porównawcza: w S1 nie powinna widzieć zmian poziomu ani zmienności.
    na_resztach=True   - (S2) Matrix Profile liczony na resztach modelu AR(1) zamiast na samym szeregu: naturalne
    wahania AR są usunięte, szok to pojedyncza ogromna reszta, zmiana φ - fragment reszt o innej wariancji.
    (S3) okno w = okres cyklu P i normalizacja=True: porównywany jest kształt jednego cyklu; wersja bez normalizacji
    porównawczo (czy normalizacja „ukrywa” zmianę amplitudy - hipoteza z planu, rozdz. 4.5).
    """
    grupa = "fragmenty szeregu"

    def __init__(self, w: int = 50, normalizacja: bool = False, na_resztach: bool = False):
        self.w, self.normalizacja, self.na_resztach = w, normalizacja, na_resztach
        self.nazwa = ("Matrix Profile z normalizacją" if normalizacja else "Matrix Profile") + \
                     (" (reszty AR)" if na_resztach else "")

    def wynik(self, x):
        if self.na_resztach:                   # S2: MP na resztach modelu AR(1) (zmiana po pilocie)
            _, _, s_e, r = dopasuj_ar1(x)
            x = r / s_e
        mp = (stumpy.stump(x, m=self.w) if self.normalizacja else stumpy.aamp(x, m=self.w))[:, 0].astype(float)
        mp[~np.isfinite(mp)] = np.nanmax(mp[np.isfinite(mp)])
        return do_punktow(np.arange(len(mp)), mp, self.w, len(x))


class CechyKNN(Detektor):
    """FR2 - cechy okien + kNN: każde okno opisane cechami dobranymi pod anomalie scenariusza (S1: średnia, odch. std,
    autokorelacja lag 1, kurtoza, rozstęp), standaryzowanymi odpornie. Wynik okna = średnia odległość do k najbliższych
    okien NIENAKŁADAJĄCYCH się w czasie. Interpretowalne - widać, która cecha odstaje.
    cechy = 5 (pełny zestaw) albo 3 (po jednej cesze na rodzaj anomalii); liczba cech trafia do nazwy metody.
    """
    nazwa = "cechy okien + kNN"
    grupa = "fragmenty szeregu"

    def __init__(self, w: int = 50, krok: int = 5, k: int = 10, kolumny=None, nazwa=None, zestaw: str = "s1",
                 P: int = 50, cechy: int = 5):
        self.w, self.krok, self.k, self.zestaw, self.P, self.liczba_cech = w, krok, k, zestaw, P, cechy
        self.kolumny = kolumny                # do analizy dodatkowej: podzbiór cech (indeksy kolumn pełnego zestawu)
        if nazwa:
            self.nazwa = nazwa
        else:
            self.nazwa = ("cechy cykli + kNN" if zestaw == "s3" else "cechy okien + kNN") + f" ({opis_cech(cechy)})"

    def cechy(self, x):
        starty, F = cechy_okien(x, self.w, self.krok, self.zestaw, self.P, 5 if self.kolumny else self.liczba_cech)
        return starty, (F if self.kolumny is None else F[:, self.kolumny])

    def wynik(self, x):
        starty, F = self.cechy(x)
        D = odleglosci_bez_trywialnych(F, starty, self.w)
        wyn = np.sort(D, axis=1)[:, :self.k].mean(axis=1)
        return do_punktow(starty, wyn, self.w, len(x))
