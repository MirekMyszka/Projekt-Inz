"""Wspólny interfejs metod wykrywania anomalii i narzędzia do pracy na oknach.

Każda metoda (detektor) ma:
    nazwa, grupa                         - do tabel i wykresów,
    kalibruj(szeregi_czyste) -> None     - opcjonalne: dostrojenie stałych na szeregach BEZ anomalii
                                           (np. skale kart CUSUM, kara PELT); domyślnie nic nie robi,
    ucz(szeregi_uczace) -> None          - tylko warianty uczone na szeregach bez anomalii (nauka="czyste"),
    wynik(x) -> np.ndarray               - wynik anomalii dla KAŻDEGO punktu (im wyższy, tym bardziej nietypowy).
Próg alarmu NIE jest częścią metody - ustala go moduł oceny (src/ewaluacja/metryki.py) jednakowo dla wszystkich.

Metody okienkowe liczą wynik dla okien długości w (z krokiem `krok`) i przypisują go punktom przez interpolację
między ŚRODKAMI okien - dzięki temu alarm nie „rozlewa się” o całą długość okna poza anomalię.
"""
import numpy as np
from scipy.stats import kurtosis


class Detektor:
    nazwa = "detektor"
    grupa = "-"

    def kalibruj(self, szeregi_czyste: list[np.ndarray]) -> None:
        pass

    def ucz(self, szeregi_uczace: list[np.ndarray]) -> None:
        """Tylko warianty uczone na szeregach bez anomalii (nauka="czyste"): dopasowanie modelu na OSOBNYCH
        szeregach uczących (innych niż szeregi kalibracji progu). Silnik eksperymentu woła ucz() przed kalibracją."""
        pass

    def wynik(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.nazwa})"


def okna(x: np.ndarray, w: int, krok: int) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (początki okien, macierz okien [liczba_okien × w])."""
    starty = np.arange(0, len(x) - w + 1, krok)
    idx = starty[:, None] + np.arange(w)[None, :]
    return starty, x[idx]


def do_punktow(starty: np.ndarray, wynik_okien: np.ndarray, w: int, n: int) -> np.ndarray:
    """Wynik okien -> wynik punktów: interpolacja liniowa między środkami okien (stała na brzegach)."""
    srodki = starty + (w - 1) / 2
    return np.interp(np.arange(n), srodki, wynik_okien)


def autokorelacja(m: np.ndarray, lag: int = 1) -> np.ndarray:
    """Autokorelacja z opóźnieniem `lag` dla każdego wiersza macierzy okien."""
    m = m - m.mean(axis=1, keepdims=True)
    return (m[:, lag:] * m[:, :-lag]).sum(1) / ((m ** 2).sum(1) + 1e-12)


# Zestawy cech okien (ustalenie z konsultacji 27.10.2026): w KAŻDYM scenariuszu dwa warianty -
#   3 cechy: po jednej cesze na każdy rodzaj anomalii scenariusza,
#   5 cech:  te 3 + 2 cechy pomocnicze z planu (plan_etap_1.md, rozdz. 4.3–4.5).
# Funkcje cechy_okien_s* zwracają zawsze 5 kolumn; KOLUMNY_3[zestaw] - indeksy wariantu 3-cechowego.
KOLUMNY_3 = {"s1": [0, 1, 2], "s2": [0, 1, 3], "s3": [0, 1, 2], "rz": [0, 1, 2]}


def cechy_okien_s1(m: np.ndarray) -> np.ndarray:
    """Cechy okna dobrane pod anomalie S1: średnia (S1-A), odch. std (S1-B), autokorelacja lag 1 (S1-C),
    kurtoza i rozstęp (pomocnicze)."""
    return np.column_stack([m.mean(1), m.std(1), autokorelacja(m, 1), kurtosis(m, axis=1), np.ptp(m, axis=1)])


def standaryzuj_odpornie(F: np.ndarray) -> np.ndarray:
    """Standaryzacja kolumn medianą i MAD (odporna na anomalie obecne w danych)."""
    med = np.median(F, axis=0)
    mad = 1.4826 * np.median(np.abs(F - med), axis=0)
    return (F - med) / np.where(mad > 0, mad, 1.0)


def odporna_skala(x: np.ndarray, w: int = 50) -> float:
    """Odporne oszacowanie odch. std szumu: pierwiastek z MEDIANY wariancji kolejnych, nienakładających się okien
    długości w (z poprawką na obciążenie mediany rozkładu χ²: mediana χ²_k ≈ k·(1 − 2/(9k))³, k = w − 1).

    Dlaczego nie 1,4826·MAD (pierwsza wersja): anomalie zbiorowe zajmują do 20% punktów szeregu i choć nie „łamią”
    MAD, to go zawyżają - mediana |x − μ| przesuwa się w stronę wyższego kwantyla. Przy L = 100 w S1 wariancja normy
    po standaryzacji wynosiła ok. 0,86, przy L = 200 w S3 - 0,73. Okno z anomalią to tylko jedno z ~60 okien,
    więc mediana wariancji okien praktycznie się nie zmienia (sprawdzenie: docs/wyniki_s1.md, rozdz. 7)."""
    k = len(x) // w
    if k < 3:                                  # zbyt krótki szereg - wracamy do MAD
        return float(1.4826 * np.median(np.abs(x - np.median(x))))
    v = x[:k * w].reshape(k, w).var(axis=1, ddof=1)
    poprawka = (1 - 2 / (9 * (w - 1))) ** 3
    return float(np.sqrt(np.median(v) / poprawka))


def odporne_mu_sigma(x: np.ndarray) -> tuple[float, float]:
    """Odporne oszacowanie poziomu i odch. std szumu: mediana szeregu i odporna_skala (mediana wariancji okien)."""
    return float(np.median(x)), odporna_skala(x)


def odleglosci_bez_trywialnych(F: np.ndarray, starty: np.ndarray, w: int) -> np.ndarray:
    """Macierz odległości euklidesowych między oknami; pary okien nakładających się w czasie (|Δstart| < w)
    dostają odległość +∞ - inaczej każde okno miałoby „bliźniaka” w sąsiednim, prawie identycznym oknie
    i anomalia byłaby maskowana (tzw. trywialne dopasowania)."""
    kw = (F ** 2).sum(1)
    D = np.sqrt(np.maximum(kw[:, None] + kw[None, :] - 2 * F @ F.T, 0))
    D[np.abs(starty[:, None] - starty[None, :]) < w] = np.inf
    return D


def dopasuj_ar1(x: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Dopasowanie modelu AR(1) z wyrazem wolnym metodą najmniejszych kwadratów: x_t = c + φ·x_{t−1} + e_t.

    Zwraca (μ, φ, σ_e, reszty), gdzie μ = c/(1 − φ) to poziom równowagi, a reszty e_t = x_t − c − φ·x_{t−1}
    (pierwsza reszta = 0). σ_e szacowane odpornie (odporna_skala: mediana wariancji okien reszt), żeby szoki i anomalie
    go nie zawyżały.
    Reszty to „to, czego model nie przewidział” - dla szeregu bez anomalii są (prawie) białym szumem.
    """
    y, z = x[1:], x[:-1]
    zc, yc = z - z.mean(), y - y.mean()
    phi = float((zc * yc).sum() / (zc ** 2).sum())
    c = float(y.mean() - phi * z.mean())
    r = np.r_[0.0, y - c - phi * z]
    s_e = odporna_skala(r[1:])
    return c / (1 - phi), phi, s_e, r


def cechy_okien_s2(m: np.ndarray, mr: np.ndarray) -> np.ndarray:
    """Cechy okna dobrane pod anomalie S2 (kolumny):
        0 średnia x            → przesunięcie poziomu równowagi (S2-C),
        1 autokorelacja lag 1  → zmiana siły zależności (S2-B),
        2 odch. std reszt AR   → pomocnicza: reszty rosną, gdy model przestaje pasować (S2-B, S2-A),
        3 maks. |reszta AR|    → szok innowacyjny (jedna ogromna reszta; S2-A),
        4 autokorelacja lag 2  → pomocnicza (plan: autokorelacja z opóźnieniem 1–3).
    Wariant 3-cechowy: kolumny 0, 1, 3. m - okna szeregu, mr - te same okna reszt modelu AR(1)."""
    return np.column_stack([m.mean(1), autokorelacja(m, 1), mr.std(1), np.abs(mr).max(1), autokorelacja(m, 2)])


# ============================================================================================ S3 - szum + sinus
def reszty_stl(x: np.ndarray, P: int, sezon: int | None = None, odporny: bool = True) -> np.ndarray:
    """Reszty rozkładu STL (Cleveland i in., 1990): x = trend + sezonowość (okres P) + reszta.

    sezon - długość wygładzania składowej sezonowej W CYKLACH (nieparzysta). Domyślnie 2·(n/P) + 1, czyli dłuższa niż
    cały szereg: składowa sezonowa jest wtedy prawie stała w czasie - tak jak cykl tła S3. Krótkie okno (domyślne
    w statsmodels: 7 cykli) pozwala sezonowości „podążać” za anomalią trwającą kilka cykli i ją wchłonąć
    (sprawdzone w analizie dodatkowej S3 (a)). odporny=True - wersja odporna STL (wagi zmniejszają wpływ odstających
    punktów na trend i sezonowość)."""
    from statsmodels.tsa.seasonal import STL
    if sezon is None:
        sezon = 2 * (len(x) // P) + 1
    klucz = (x.tobytes(), P, sezon, odporny)
    if klucz not in _PAMIEC_STL:              # STL (odporny) trwa ~1 s - obie metody S3 na resztach liczą go raz
        if len(_PAMIEC_STL) >= 4:
            _PAMIEC_STL.pop(next(iter(_PAMIEC_STL)))
        _PAMIEC_STL[klucz] = STL(x, period=P, seasonal=sezon, robust=odporny).fit().resid
    return _PAMIEC_STL[klucz].copy()


_PAMIEC_STL: dict = {}


def reszty_harmoniczne(x: np.ndarray, P: int) -> np.ndarray:
    """Reszty globalnego dopasowania sinusa o znanym okresie P metodą najmniejszych kwadratów:
    x_t ≈ c + a·sin(2πt/P) + b·cos(2πt/P). Model „idealny” dla tła S3 (do analizy dodatkowej (a))."""
    t = np.arange(len(x))
    X = np.column_stack([np.ones(len(x)), np.sin(2 * np.pi * t / P), np.cos(2 * np.pi * t / P)])
    return x - X @ np.linalg.lstsq(X, x, rcond=None)[0]


def cechy_okien_s3(m: np.ndarray, starty: np.ndarray, P: int) -> np.ndarray:
    """Cechy cykli - okna opisane parametrami dopasowanego cyklu, dobranymi pod anomalie S3:
        amplituda    √(a² + b²) z dopasowania x ≈ c + a·sin(2πt/P) + b·cos(2πt/P) w oknie        → S3-B,
        faza         przesunięcie fazy okna względem fazy typowej dla szeregu (zawinięte do (−π, π]) → S3-A,
        częstotliwość dominująca częstotliwość okna (periodogram z dopełnieniem zerami, bez okna wygładzającego -
                     dla okna o długości ~1 cyklu okno Hanninga zawyżało błąd oszacowania), w cyklach na P,
                     szukana w paśmie 0,25–4 cykli na P (okres od P/4 do 4P): bez tego ograniczenia przy słabym cyklu
                     (A = 1) lub krótkim oknie maksimum trafiało czasem w szum o wysokiej częstotliwości i jedna
                     cecha dawała odchylenia rzędu 150 odpornych odch. std                                    → S3-C,
        odch. std reszt dopasowania - ile w oknie jest „nie-cyklu” (rośnie przy każdej anomalii, pomocnicza),
        korelacja z wzorcowym cyklem sin(2πt/P + faza typowa) - pomocnicza (plan, rozdz. 4.5).
    Kolumny: amplituda, faza, częstotliwość, odch. std reszt, korelacja; wariant 3-cechowy - kolumny 0–2.
    Czas t jest GLOBALNY (indeks w szeregu), więc faza okna jest porównywalna między oknami."""
    w = m.shape[1]
    t = (starty[:, None] + np.arange(w)[None, :]) * 2 * np.pi / P
    S, C = np.sin(t), np.cos(t)
    wsp = np.empty((len(m), 3))
    res = np.empty(len(m))
    for i in range(len(m)):                           # mała regresja 3-parametrowa na każde okno
        X = np.column_stack([np.ones(w), S[i], C[i]])
        b, *_ = np.linalg.lstsq(X, m[i], rcond=None)
        wsp[i] = b
        res[i] = (m[i] - X @ b).std()
    amp = np.hypot(wsp[:, 1], wsp[:, 2])
    faza = np.arctan2(wsp[:, 2], wsp[:, 1])
    ref = np.angle(np.median(np.cos(faza)) + 1j * np.median(np.sin(faza)))
    dfaza = np.angle(np.exp(1j * (faza - ref)))
    nfft = 16 * max(64, int(2 ** np.ceil(np.log2(w))))
    spec = np.abs(np.fft.rfft(m - m.mean(1, keepdims=True), n=nfft, axis=1))
    f = np.arange(spec.shape[1]) / nfft * P                       # częstotliwość w cyklach na okres P
    pasmo = (f >= 0.25) & (f <= 4.0)                              # okres od P/4 do 4P - poza tym pasmem tylko szum
    czest = f[pasmo][spec[:, pasmo].argmax(1)]
    wzor = np.sin(t + ref)                                        # wzorcowy cykl o typowej fazie szeregu
    mc, wc = m - m.mean(1, keepdims=True), wzor - wzor.mean(1, keepdims=True)
    korel = (mc * wc).sum(1) / np.sqrt((mc ** 2).sum(1) * (wc ** 2).sum(1) + 1e-12)
    return np.column_stack([amp, dfaza, czest, res, korel])


# ============================================================================================ dane rzeczywiste
def profil_okresowy(x: np.ndarray, P: int) -> np.ndarray:
    """Typowy przebieg jednego okresu P (np. tygodnia): mediana wartości w tym samym miejscu okresu ze wszystkich
    okresów szeregu (indeks t mod P liczony od początku szeregu). Mediana - kilka dni z anomaliami go nie zmienia."""
    k = int(np.ceil(len(x) / P))
    pad = np.full(k * P, np.nan)
    pad[:len(x)] = x
    return np.nanmedian(pad.reshape(k, P), axis=0)


def cechy_okien_rz(m: np.ndarray, starty: np.ndarray, x: np.ndarray, P: int) -> np.ndarray:
    """Cechy cykli dla szeregu rzeczywistego o NIEsinusoidalnym cyklu (NYC Taxi: cykl dobowy i tygodniowy) - wzorcem
    zamiast sinusa jest typowy przebieg okresu P (profil_okresowy, tydzień = 336 punktów), a okno (doba) jest
    porównywane z profilem w tym samym miejscu tygodnia. Kolumny:
        0 poziom       średnia różnica okna od profilu (log-szereg: ile razy mniej/więcej ruchu niż zwykle),
        1 kształt      korelacja okna z profilem (czy przebieg doby jest typowy dla tego dnia tygodnia),
        2 faza         przesunięcie fazy pierwszej harmonicznej okna (okres = długość okna) względem profilu,
        3 odch. std różnic od profilu - pomocnicza (ile w oknie „nie-profilu”),
        4 amplituda    log-stosunek amplitud pierwszej harmonicznej okna i profilu - pomocnicza.
    Wariant 3-cechowy - kolumny 0–2 (poziom, kształt, faza)."""
    prof = profil_okresowy(x, P)
    w = m.shape[1]
    pm = prof[(starty[:, None] + np.arange(w)[None, :]) % P]
    d = m - pm
    mc, pc = m - m.mean(1, keepdims=True), pm - pm.mean(1, keepdims=True)
    korel = (mc * pc).sum(1) / np.sqrt((mc ** 2).sum(1) * (pc ** 2).sum(1) + 1e-12)
    e = np.exp(-2j * np.pi * np.arange(w) / w)
    hm, hp = (mc * e).sum(1), (pc * e).sum(1)
    faza = np.angle(hm * np.conj(hp))
    amp = np.log((np.abs(hm) + 1e-9) / (np.abs(hp) + 1e-9))
    return np.column_stack([d.mean(1), korel, faza, d.std(1), amp])
