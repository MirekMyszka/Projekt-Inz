"""Metody statystyczne (docs/plan_etap_1.md, rozdz. 4.3–4.5):
    S1 - CUSUM na trzech kartach, PELT;  S2 - reszty AR + CUSUM, PELT z kosztem AR;  S3 - STL + EWMA, PELT na resztach STL.
"""
import numpy as np
from scipy.signal import lfilter

from src.metody.baza import Detektor, dopasuj_ar1, odporne_mu_sigma, reszty_harmoniczne, reszty_stl


# ----------------------------------------------------------------------------------------------- CUSUM
def _cusum_jednostronny(z: np.ndarray, k: float, kierunek: int) -> np.ndarray:
    """S_t = max(0, S_{t−1} + z_t − k) liczone w przód (kierunek=1) albo w tył (kierunek=−1)."""
    n = len(z)
    out = np.empty(n)
    s = 0.0
    idx = range(n) if kierunek == 1 else range(n - 1, -1, -1)
    for t in idx:
        s = s + z[t] - k
        if s < 0.0:
            s = 0.0
        out[t] = s
    return out


def _lokalizuj(F: np.ndarray) -> np.ndarray:
    """Zamienia kartę CUSUM liczoną w przód na wynik zlokalizowany w czasie.

    Karta tworzy „wycieczki” - ciągłe odcinki F > 0. W każdej wycieczce klasyczne oszacowanie zmiany to:
    początek = ostatnie zero przed wycieczką, koniec = szczyt karty. Punkty od początku do szczytu dostają wysokość
    szczytu; punkty PO szczycie (karta tylko opada) dostają 0 - alarm nie ciągnie się za anomalią.
    """
    out = np.zeros_like(F)
    dodatnie = np.r_[False, F > 0, False]
    zmiany = np.flatnonzero(dodatnie[1:] != dodatnie[:-1])
    for s, k in zip(zmiany[::2], zmiany[1::2]):
        p = s + int(np.argmax(F[s:k]))
        out[s:p + 1] = F[p]
    return out


def cusum_offline(z: np.ndarray, k: float) -> np.ndarray:
    """Dwustronna karta CUSUM w wersji offline (cały szereg dostępny), z lokalizacją zmiany.

    S⁺_t = max(0, S⁺_{t−1} + z_t − k)  (wzrost),   S⁻_t = max(0, S⁻_{t−1} − z_t − k)  (spadek).
    Każdą kartę lokalizujemy (_lokalizuj): wynik jest wysoki tylko na odcinku od początku zmiany do szczytu karty.
    Bez tego karta po silnej anomalii opada tylko o ~k na punkt i „ciągnie” alarm setki punktów za anomalię
    (w pierwszym przebiegu eksperymentu dawało to fałszywe alarmy między anomaliami).
    Dla anomalii o nadwyżce δ i długości L wynik ≈ (δ − k)·L na całym fragmencie.
    """
    return np.maximum(_lokalizuj(_cusum_jednostronny(z, k, 1)), _lokalizuj(_cusum_jednostronny(-z, k, 1)))


def cusum_gorny(z: np.ndarray, k: float) -> np.ndarray:
    """Jednostronna (tylko wzrost) karta CUSUM w wersji offline, z lokalizacją zmiany."""
    return _lokalizuj(_cusum_jednostronny(z, k, 1))


class CUSUM(Detektor):
    """ST1 - CUSUM na trzech kartach, każda odpowiada jednej anomalii S1:

        karta 1 (poziom):     z1 = (x_t − μ)/σ,                        dwustronna, k = 0,5  → S1-A
        karta 2 (zmienność):  zd = ((x_t − x_{t−1})²/(2σ²) − 1)/√2,    tylko wzrost, k = 0,5 → S1-B
        karta 3 (zależność):  −zd,                                     tylko wzrost, k = 0,15 → S1-C

    Dlaczego zmienność i zależność liczymy na RÓŻNICACH sąsiednich punktów: zmiana poziomu ich nie zmienia, więc
    karty 2 i 3 nie „przeciekają” od anomalii S1-A. Wzrost zmienności podnosi różnice (E[zd] > 0), a zależność je
    zmniejsza - w AR(1): E[(x_t − x_{t−1})²] = 2σ²(1 − φ), czyli E[−zd] = φ/√2. Jedna statystyka, dwie strony.
    (Pierwsza wersja używała x² i iloczynu x_t·x_{t−1}; przy zmianie poziomu o Δ iloczyn rośnie o ~Δ², karta po silnej
    anomalii S1-A opadała przez ~1 500 punktów i zasłaniała leżącą dalej anomalię S1-C.)

    μ, σ szacowane odpornie (mediana; σ - mediana wariancji okien, baza.odporna_skala). k ≈ połowa najmniejszej
    istotnej zmiany (dla zależności: φ ≈ 0,4).
    Karty mają różne rozkłady, więc przed połączeniem każdą dzielimy przez medianę jej maksimum na szeregach
    bez anomalii (kalibruj). Wynik = maksimum z trzech przeskalowanych kart.
    """
    nazwa = "CUSUM (poziom, zmienność, zależność)"
    grupa = "statystyczne"

    def __init__(self, k=(0.5, 0.5, 0.15), karty=(0, 1, 2), nazwa=None):
        self.k = k
        self.karty_uzyte = list(karty)        # do analizy dodatkowej: można zostawić np. tylko kartę zależności
        self.skale = np.ones(3)
        if nazwa:
            self.nazwa = nazwa

    def karty(self, x: np.ndarray) -> np.ndarray:
        mu, s = odporne_mu_sigma(x)
        z1 = (x - mu) / s
        zd = np.r_[0.0, (np.diff(x) ** 2 / (2 * s ** 2) - 1) / np.sqrt(2)]
        return np.vstack([cusum_offline(z1, self.k[0]), cusum_gorny(zd, self.k[1]), cusum_gorny(-zd, self.k[2])])

    def kalibruj(self, szeregi_czyste):
        maks = np.array([self.karty(x).max(axis=1) for x in szeregi_czyste])
        self.skale = np.median(maks, axis=0)
        self.skale = np.where(self.skale > 0, self.skale, 1.0)   # karta, która na danych kalibracyjnych nie rośnie

    def wynik(self, x):
        return (self.karty(x) / self.skale[:, None])[self.karty_uzyte].max(axis=0)


# ------------------------------------------------------------------------------------------------ PELT
def _koszt_normalny(cs1: np.ndarray, cs2: np.ndarray, a: np.ndarray, b: int) -> np.ndarray:
    """Koszt odcinków [a, b) dla modelu normalnego ze zmienną średnią i wariancją: n·ln(σ̂²) (−2·log-wiarygodność
    z dokładnością do stałej). cs1, cs2 - sumy skumulowane y i y² z zerem na początku. Wektorowo po `a`."""
    n = b - a
    s1 = cs1[b] - cs1[a]
    s2 = cs2[b] - cs2[a]
    var = np.maximum(s2 / n - (s1 / n) ** 2, 1e-8)
    return n * np.log(var)


def pelt_ogolny(koszt, n: int, kara: float, min_dl: int = 10, skok: int = 5) -> list[int]:
    """PELT (Killick i in., 2012) - dokładny optymalny podział na odcinki minimalizujący
    Σ_odcinki koszt(odcinek) + kara·(liczba odcinków), z przycinaniem kandydatów (pruning).

    koszt(a, b) - koszt odcinków [a_i, b) dla wektora początków a (wektorowo). Granice rozważane co `skok` punktów.
    Zwraca końce odcinków (ostatni = n), jak biblioteka `ruptures`.
    """
    punkty = np.r_[np.arange(0, n, skok), n] if n % skok else np.arange(0, n + 1, skok)
    F = {0: -kara}
    poprzedni = {0: 0}
    kandydaci = np.array([0])
    for b in punkty[1:]:
        dop = kandydaci[b - kandydaci >= min_dl]
        if len(dop) == 0:
            continue
        k = koszt(dop, int(b))
        Fk = np.array([F[a] for a in dop])
        calk = Fk + k + kara
        j = int(np.argmin(calk))
        F[b] = float(calk[j])
        poprzedni[b] = int(dop[j])
        # przycinanie: kandydat a odpada, jeśli F(a) + koszt(a, b) > F(b) - nigdy już nie da lepszego podziału
        zostaw = Fk + k <= F[b]
        kandydaci = np.r_[kandydaci[b - kandydaci < min_dl], dop[zostaw], b]
    granice = []
    b = int(punkty[-1])
    while b > 0:
        granice.append(b)
        b = poprzedni[b]
    return sorted(granice)


def pelt(sygnaly: list[np.ndarray], kara: float, min_dl: int = 10, skok: int = 5) -> list[int]:
    """PELT z kosztem normalnym (zmienna średnia i wariancja) - suma kosztów niezależnych sygnałów (S1)."""
    cs = [(np.r_[0.0, np.cumsum(y)], np.r_[0.0, np.cumsum(y ** 2)]) for y in sygnaly]
    return pelt_ogolny(lambda a, b: sum(_koszt_normalny(c1, c2, a, b) for c1, c2 in cs), len(sygnaly[0]), kara,
                       min_dl, skok)


def maska_tla(granice: list[int], n: int, udzial: float = 0.5) -> np.ndarray:
    """Punkty „tła” po podziale PELT: najdłuższe odcinki, które razem obejmują co najmniej `udzial` (50%) szeregu.

    Wynik odcinka PELT porównujemy z modelem dopasowanym do TŁA, a nie do całego szeregu. Anomalie są krótkie
    (≤ 200 punktów), a odcinki normy między nimi długie (~800–1 000 punktów), więc najdłuższe odcinki to norma.
    Pierwsza wersja porównywała odcinek z parametrami całego szeregu (σ̂ z MAD, model AR z całego szeregu) - te są
    zanieczyszczone przez anomalie, a statystyki typu z rosną jak √n, więc nawet mały błąd odniesienia dawał wysoki
    wynik DŁUGIM odcinkom normy (S1, L = 100: 44% punktów normy nad progiem)."""
    starty = [0] + list(granice[:-1])
    odcinki = sorted(zip(starty, granice), key=lambda o: o[1] - o[0], reverse=True)
    m = np.zeros(n, bool)
    for a, b in odcinki:
        m[a:b] = True
        if m.sum() >= udzial * n:
            break
    return m


def _ac1(seg: np.ndarray, srodek: float | None = None) -> float:
    c = seg - (seg.mean() if srodek is None else srodek)
    return float((c[1:] * c[:-1]).sum() / ((c ** 2).sum() + 1e-12))


class _PELTBaza:
    """Wspólne dla wariantów PELT: wynik i kalibracja.

    Wynik punktu = statystyka z jego odcinka względem tła (patrz maska_tla), ale tylko jeśli odcinek jest ISTOTNIE inny
    od tła: z > Z_KRYT = 3 (reguła „trzech sigm”); w przeciwnym razie 0. Bez tego odcinki normy różniące się od tła
    losowo (|z| ≈ 1–2) dostawały wynik dodatni, a na szeregach bez anomalii PELT zwykle nie dzieli szeregu wcale
    (wynik 0) - próg ustalany na takich szeregach wypadał przy zerze i alarmowały wszystkie odcinki normy.

    Kalibracja: wspólna zasada „fałszywy alarm w 10% szeregów bez anomalii” realizowana przez KARĘ - najmniejsza kara
    całkowita (3…300, bisekcja), przy której odcinek istotnie inny od tła pojawia się w ≤ 10% szeregów bez anomalii. (Pierwsza wersja:
    kara przy średnio < 0,5 fałszywej granicy na szereg.)"""
    Z_KRYT = 3.0
    KARA_MIN, KARA_MAX = 3, 300          # kara całkowita; szukana bisekcją (odsetek alarmów maleje z karą)
    ALFA = 0.10

    def _przygotuj(self, x):
        return x

    def wynik(self, x):
        z = self._z(self._przygotuj(x), self.kara)
        return np.where(z > self.Z_KRYT, z, 0.0)

    def _odsetek_alarmow(self, dane, kara):
        return float(np.mean([(self._z(d, kara) > self.Z_KRYT).any() for d in dane]))

    def kalibruj(self, szeregi_czyste):
        dane = [self._przygotuj(x) for x in szeregi_czyste]
        lo, hi = self.KARA_MIN, self.KARA_MAX              # szukamy najmniejszej kary z odsetkiem ≤ ALFA
        if self._odsetek_alarmow(dane, lo) <= self.ALFA:
            self.kara = lo
            return
        while hi - lo > 1:
            sr = (lo + hi) // 2
            if self._odsetek_alarmow(dane, sr) <= self.ALFA:
                hi = sr
            else:
                lo = sr
        self.kara = hi


def _koszt_ar1(x: np.ndarray):
    """Koszt odcinka dla modelu AR(1) z wyrazem wolnym, dopasowanego osobno na każdym odcinku:
    n·ln(σ̂²_reszt). Pary (x_{t−1}, x_t), t = 1..n−1; sumy skumulowane pozwalają liczyć regresję dla dowolnego
    odcinka w czasie stałym (wektorowo po początkach a). Odcinek [a, b) używa par o indeksach t ∈ [max(a,1), b)."""
    z, y = x[:-1], x[1:]
    S = {k: np.r_[0.0, 0.0, np.cumsum(v)] for k, v in
         {"z": z, "y": y, "zz": z * z, "yy": y * y, "zy": z * y}.items()}

    def koszt(a, b):
        a1 = np.maximum(a, 1)
        n = b - a1
        g = {k: (v[b] - v[a1]) for k, v in S.items()}
        szz = g["zz"] - g["z"] ** 2 / n
        syy = g["yy"] - g["y"] ** 2 / n
        szy = g["zy"] - g["z"] * g["y"] / n
        var = np.maximum((syy - szy ** 2 / np.maximum(szz, 1e-12)) / n, 1e-8)
        return n * np.log(var)
    return koszt


class PELT(_PELTBaza, Detektor):
    """ST2 - PELT: optymalny podział gotowego szeregu na odcinki o stałych parametrach (metoda typowo offline).

    Koszt normalny (zmienna średnia i wariancja) liczony na dwóch sygnałach: x_t oraz x_t·x_{t−1}.
    Zmiana średniej lub wariancji x → S1-A, S1-B; zmiana średniej iloczynu sąsiednich punktów → S1-C.

    PELT zwraca granice odcinków, a nie wynik dla punktów. Wynik punktu = jak bardzo JEGO ODCINEK odbiega od TŁA
    (najdłuższych odcinków obejmujących ≥ 50% szeregu - maska_tla; średnia μ_t, odch. σ_t i autokorelacja ρ_t tła) -
    maksimum trzech statystyk typu z (każda ≈ N(0,1) dla odcinka bez anomalii):
        |średnia − μ_t|·√n/σ_t,   |ln(wariancja/σ_t²)|·√(n/2),   |autokorelacja lag 1 − ρ_t|·√n.
    Wynik i kalibracja kary: _PELTBaza (odcinek istotnie inny od tła, z > 3; kara ustalana na 10% fałszywych alarmów).
    """
    nazwa = "PELT"
    grupa = "statystyczne"

    def __init__(self, kara: float = 35.0, min_dl: int = 10, skok: int = 5):
        self.kara, self.min_dl, self.skok = kara, min_dl, skok

    def granice(self, x, kara=None):
        mu, s = odporne_mu_sigma(x)
        e = (x - mu) / s
        return pelt([e, np.r_[0.0, e[1:] * e[:-1]]], self.kara if kara is None else kara, self.min_dl, self.skok)

    def _z(self, x, kara):
        gr = self.granice(x, kara)
        tlo = maska_tla(gr, len(x))
        mu, s = float(x[tlo].mean()), float(x[tlo].std())
        starty = [0] + gr[:-1]
        num = sum(((x[a:b] - mu)[1:] * (x[a:b] - mu)[:-1]).sum() for a, b in zip(starty, gr) if tlo[a])
        rho = float(num / (((x[tlo] - mu) ** 2).sum() + 1e-12))
        out = np.zeros(len(x))
        for a, b in zip(starty, gr):
            seg = x[a:b]
            n = len(seg)
            out[a:b] = max(abs(seg.mean() - mu) * np.sqrt(n) / s,
                           abs(np.log(seg.var() / s ** 2 + 1e-12)) * np.sqrt(n / 2),
                           abs(_ac1(seg) - rho) * np.sqrt(n))
        return out


# ====================================================================================== metody dla S2 - AR(1)
class ResztyARCUSUM(Detektor):
    """ST1 (S2) - reszty modelu AR(1) + CUSUM.

    Najpierw dopasowujemy AR(1) do całego szeregu (dopasuj_ar1) i bierzemy reszty e_t - to, czego model nie
    przewidział. Dla szeregu bez anomalii są one białym szumem, więc działają na nich narzędzia z S1:
      - karty CUSUM z S1 (poziom, zmienność, zależność) liczone na resztach - ze zmniejszonym k = 0,25 dla poziomu,
        bo przesunięcie równowagi o Δσ_x daje w resztach tylko (1 − φ)·Δσ_x ≈ 0,42·Δ·σ_e (S2-C);
        zmiana φ zmienia wariancję i autokorelację reszt (S2-B);
      - test szoku innowacyjnego (innovational outlier): |e_t|/σ_e - szok to jedna ogromna reszta (S2-A).
    Każdą składową dzielimy przez medianę jej maksimum na szeregach bez anomalii; wynik = maksimum składowych.
    """
    nazwa = "reszty AR + CUSUM"
    grupa = "statystyczne"

    def __init__(self):
        self.cusum = CUSUM(k=(0.25, 0.5, 0.15))
        self.skala_szoku = 1.0

    def _reszty(self, x):
        _, _, s_e, r = dopasuj_ar1(x)
        return r / s_e

    def kalibruj(self, szeregi_czyste):
        rs = [self._reszty(x) for x in szeregi_czyste]
        self.cusum.kalibruj(rs)
        self.skala_szoku = float(np.median([np.abs(r).max() for r in rs]))

    def wynik(self, x):
        r = self._reszty(x)
        return np.maximum(self.cusum.wynik(r), np.abs(r) / self.skala_szoku)


class PELTAR(_PELTBaza, Detektor):
    """ST2 (S2) - PELT z kosztem AR(1): podział szeregu na odcinki, z których każdy ma WŁASNY model AR(1)
    (własny poziom, własne φ, własną wariancję reszt). Granice odcinków wskazują początek i koniec zmiany φ (S2-B)
    i przesunięcia równowagi (S2-C); szok (S2-A) daje krótki odcinek o ogromnej wariancji reszt.

    Wynik punktu = jak bardzo jego odcinek odbiega od modelu AR(1) dopasowanego do TŁA (najdłuższych odcinków
    obejmujących ≥ 50% szeregu - maska_tla; parametry c_t, φ_t, σ_t) - maksimum trzech statystyk typu z (każda ≈ N(0,1)
    dla odcinka bez anomalii), liczonych na resztach e_t = x_t − c_t − φ_t·x_{t−1}:
        |średnia e|·√n/σ_t,   |ln(wariancja e/σ_t²)|·√(n/2),   |autokorelacja x na odcinku − φ_t|·√n/√(1 − φ_t²).
    (Pierwsza wersja używała modelu dopasowanego do całego szeregu - zanieczyszczonego przez anomalie.)
    Wynik i kalibracja kary: _PELTBaza (odcinek istotnie inny od tła, z > 3; kara ustalana na 10% fałszywych alarmów).
    """
    nazwa = "PELT (koszt AR)"
    grupa = "statystyczne"

    def __init__(self, kara: float = 40.0, min_dl: int = 10, skok: int = 5, korekta_zaleznosci: bool = False):
        self.kara, self.min_dl, self.skok = kara, min_dl, skok
        self.korekta_zaleznosci = korekta_zaleznosci       # jak w PELTSTL - tylko dane rzeczywiste (etap 2)

    def granice(self, x, kara=None):
        return pelt_ogolny(_koszt_ar1(x), len(x), self.kara if kara is None else kara, self.min_dl, self.skok)

    def _z(self, x, kara):
        gr = self.granice(x, kara)
        tlo = maska_tla(gr, len(x))
        pary = np.flatnonzero(tlo[1:] & tlo[:-1]) + 1            # t: x_t i x_{t−1} w tle
        z, y = x[pary - 1], x[pary]
        phi = float(((z - z.mean()) * (y - y.mean())).sum() / (((z - z.mean()) ** 2).sum() + 1e-12))
        phi = float(np.clip(phi, -0.99, 0.99))
        c = float(y.mean() - phi * z.mean())
        r = np.r_[0.0, x[1:] - c - phi * x[:-1]]
        s_e = float(r[pary].std())
        f = 1.0                                   # efektywna liczba obserwacji przy zależnych resztach (PELTSTL._z)
        if self.korekta_zaleznosci:
            rho = float(np.clip(np.corrcoef(r[pary][1:], r[pary][:-1])[0, 1], 0.0, 0.99))
            f = (1 - rho) / (1 + rho)
        out = np.zeros(len(x))
        start = 0
        for koniec in gr:
            seg, rs = x[start:koniec], r[max(start, 1):koniec]
            n = len(seg) * f
            out[start:koniec] = max(abs(rs.mean()) * np.sqrt(len(rs) * f) / s_e,
                                    abs(np.log(rs.var() / s_e ** 2 + 1e-12)) * np.sqrt(len(rs) * f / 2),
                                    abs(_ac1(seg) - phi) * np.sqrt(n) / np.sqrt(1 - phi ** 2))
            start = koniec
        return out


class CUSUMHybryda(Detektor):
    """WARIANT z analizy dodatkowej S2 (d) - zaproponowany PO wynikach eksperymentu głównego, nie część zestawu z planu.

    Analiza (a) pokazała, że model AR pomaga wykryć zmianę zależności (S2-B), ale osłabia sygnał szoku i przesunięcia
    równowagi (S2-A, S2-C): w resztach zostaje z przesunięcia tylko (1 − φ)·Δ, a szok to jeden punkt zamiast
    wygasającego „garbu”. Wersja mieszana bierze z obu światów:
      - karta POZIOMU liczona na samym szeregu x (standaryzowanym odpornie; k = 0,5) - jak w S1,
      - karty ZMIENNOŚCI i ZALEŻNOŚCI liczone na resztach modelu AR (jak w „reszty AR + CUSUM”),
      - test szoku na resztach (|e_t|/σ_e).
    Każda składowa dzielona przez medianę jej maksimum na szeregach bez anomalii; wynik = maksimum składowych.
    """
    nazwa = "CUSUM mieszany (poziom na x, reszta na resztach AR)"
    grupa = "statystyczne"

    def __init__(self):
        self.na_x = CUSUM(karty=(0,))                     # tylko karta poziomu
        self.na_resztach = CUSUM(k=(0.25, 0.5, 0.15), karty=(1, 2))
        self.skala_szoku = 1.0

    @staticmethod
    def _reszty(x):
        _, _, s_e, r = dopasuj_ar1(x)
        return r / s_e

    def kalibruj(self, szeregi_czyste):
        rs = [self._reszty(x) for x in szeregi_czyste]
        self.na_x.kalibruj(szeregi_czyste)
        self.na_resztach.kalibruj(rs)
        self.skala_szoku = float(np.median([np.abs(r).max() for r in rs]))

    def wynik(self, x):
        r = self._reszty(x)
        return np.maximum.reduce([self.na_x.wynik(x), self.na_resztach.wynik(r), np.abs(r) / self.skala_szoku])


# ====================================================================================== metody dla S3 - szum + sinus
def ewma_dwustronna(z: np.ndarray, lam: float) -> np.ndarray:
    """Karta EWMA w wersji offline: E_t = λ·z_t + (1 − λ)·E_{t−1} liczona w przód i w tył, wynik = MINIMUM obu.

    Karta liczona w przód jest wysoka na anomalii i jeszcze przez kilkadziesiąt punktów ZA nią (opada wykładniczo),
    karta liczona w tył - na anomalii i PRZED nią. Minimum jest wysokie tylko tam, gdzie wysokie są obie - na samej
    anomalii (lokalizacja, jak w CUSUM). Pierwsza wersja brała średnią obu kart: po silnej anomalii „ogon” karty
    przekraczał próg poza strefą tolerancji - 0,46 fałszywego alarmu na szereg z anomaliami w S3 (wersja z minimum:
    ok. 0,07 w sprawdzeniu na szeregach spoza eksperymentu; koszt: AUC niższe o 0,02–0,06)."""
    przod = lfilter([lam], [1, -(1 - lam)], z)
    tyl = lfilter([lam], [1, -(1 - lam)], z[::-1])[::-1]
    return np.minimum(przod, tyl)


class _ResztyCyklu:
    """Wspólne dla metod S3 liczonych na resztach: usunięcie cyklu tła (STL albo dopasowanie sinusa) i standaryzacja
    reszt odpornym σ (mediana wariancji okien, baza.odporna_skala), żeby same anomalie go nie zawyżały."""

    def __init__(self, P: int = 50, model: str = "stl", sezon: int | None = None):
        self.P, self.model, self.sezon = P, model, sezon

    def reszty(self, x):
        r = reszty_harmoniczne(x, self.P) if self.model == "harmoniczny" else reszty_stl(x, self.P, self.sezon)
        mu, s = odporne_mu_sigma(r)
        return (r - mu) / s

    def _opis(self):
        if self.model == "harmoniczny":
            return " [model: sinus dopasowany globalnie]"
        return "" if self.sezon is None else f" [STL, okno sezonowości {self.sezon} cykli]"


class STLEWMA(_ResztyCyklu, Detektor):
    """ST1 (S3) - STL + karta EWMA na resztach.

    STL rozkłada szereg na trend, sezonowość (okres P) i reszty (reszty_stl). Każda anomalia S3 psuje dopasowanie cyklu:
    przesunięcie fazy i zmiana okresu dają w resztach różnicę dwóch sinusów, osłabienie cyklu - resztę (k − 1)·A·sin.
    W każdym przypadku rośnie ENERGIA reszt na fragmencie. Statystyka karty: z_t = r_t² − 1 (r - reszty
    standaryzowane; w normie E[z] ≈ 0, na anomalii > 0), wygładzana dwustronną EWMA (ewma_dwustronna).
    λ = 2/(P + 1) - efektywna „pamięć” karty ≈ jeden cykl: krótsza dawałaby wynik zależny od fazy cyklu
    (reszta przesunięcia fazy jest okresowa), dłuższa rozmywałaby alarm poza anomalię.
    """
    nazwa = "STL + EWMA"
    grupa = "statystyczne"

    def __init__(self, P: int = 50, model: str = "stl", sezon: int | None = None, nazwa=None, pamiec: int | None = None):
        super().__init__(P, model, sezon)
        self.lam = 2 / ((pamiec or P) + 1)          # pamiec - efektywna pamięć karty (domyślnie jeden cykl P)
        self.nazwa = nazwa or (self.nazwa + self._opis())

    def wynik(self, x):
        r = self.reszty(x)
        return ewma_dwustronna(r ** 2 - 1, self.lam)


class PELTSTL(_PELTBaza, _ResztyCyklu, Detektor):
    """ST2 (S3) - PELT na resztach STL: podział reszt na odcinki o stałej średniej i wariancji (koszt normalny jak w S1).
    Anomalie S3 to odcinki reszt o większej wariancji (energii) - PELT wskazuje ich początek i koniec.
    Wynik punktu = jak bardzo jego odcinek odbiega od TŁA (najdłuższych odcinków obejmujących ≥ 50% szeregu -
    maska_tla; średnia μ_t i odch. σ_t reszt w tle) - maksimum statystyk typu z:
        |średnia − μ_t|·√n/σ_t,   |ln(wariancja/σ_t²)|·√(n/2).
    Wynik i kalibracja kary: _PELTBaza (odcinek istotnie inny od tła, z > 3; kara ustalana na 10% fałszywych alarmów).
    """
    nazwa = "PELT (reszty STL)"
    grupa = "statystyczne"

    def __init__(self, P: int = 50, kara: float = 40.0, min_dl: int = 10, skok: int = 5, model: str = "stl",
                 sezon: int | None = None, nazwa=None, korekta_zaleznosci: bool = False):
        super().__init__(P, model, sezon)
        self.kara, self.min_dl, self.skok = kara, min_dl, skok
        self.korekta_zaleznosci = korekta_zaleznosci
        self.nazwa = nazwa or (self.nazwa + self._opis())

    def granice(self, r, kara=None):
        return pelt([r], self.kara if kara is None else kara, self.min_dl, self.skok)

    def _przygotuj(self, x):
        return self.reszty(x)

    def _z(self, r, kara):
        gr = self.granice(r, kara)
        tlo = maska_tla(gr, len(r))
        mu, s = float(r[tlo].mean()), float(r[tlo].std())
        # korekta_zaleznosci (dane rzeczywiste, etap 2): statystyki z zakładają niezależne reszty; przy reszt
        # autokorelacji ρ efektywna liczba obserwacji odcinka to n·(1 − ρ)/(1 + ρ) (wariancja średniej procesu AR(1)).
        # Na szeregach syntetycznych reszty są prawie niezależne (ρ ≈ 0), więc etap 1 liczono bez korekty.
        f = 1.0
        if self.korekta_zaleznosci:
            rho = float(np.clip(np.corrcoef(r[1:], r[:-1])[0, 1], 0.0, 0.99))
            f = (1 - rho) / (1 + rho)
        out = np.zeros(len(r))
        start = 0
        for koniec in gr:
            seg = r[start:koniec]
            n = len(seg) * f
            out[start:koniec] = max(abs(seg.mean() - mu) * np.sqrt(n) / s,
                                    abs(np.log(seg.var() / s ** 2 + 1e-12)) * np.sqrt(n / 2))
            start = koniec
        return out
