"""Metody uczenia maszynowego: Isolation Forest (S1, S2, S3), LOF (S1), One-Class SVM (S2) - na cechach okien,
oraz autoenkoder (S3) - na surowych oknach (docs/plan_etap_1.md, rozdz. 4.3–4.5).

Dwa sposoby uczenia (ustalenie z konsultacji 27.10.2026 - w każdym scenariuszu oba):
  nauka="szereg" (wersja główna, nienadzorowana) - model uczy się na oknach TEGO SAMEGO szeregu, w którym szuka
      anomalii (anomalie to ok. 2–10% punktów, więc większość okien jest normalna);
  nauka="czyste" (półnadzorowana, „novelty detection”) - model uczy się raz, na oknach N_UCZ = 20 OSOBNYCH szeregów
      bez anomalii (silnik eksperymentu woła ucz(); szeregi uczące mają własne ziarna, inne niż szeregi kalibracji
      progu, żeby próg nie był ustalany na danych, na których uczył się model), a potem ocenia okna badanego szeregu.
      Okna uczące bez nakładania się (krok = w) - inaczej sąsiednie, prawie identyczne okna tego samego szeregu
      zaniżałyby odległości w zbiorze uczącym (ważne dla LOF). Cechy każdego szeregu standaryzowane tak samo jak
      w wersji głównej (odpornie, w obrębie szeregu).
Cechy - te same co w „cechy okien + kNN” (3 albo 5 cech), dzięki czemu porównujemy sposób wykrywania, a nie dobór cech.
"""
import numpy as np
from sklearn.ensemble import IsolationForest as _IF
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM as _OCSVM

from src.metody.baza import Detektor, do_punktow, odleglosci_bez_trywialnych, odporne_mu_sigma, okna
from src.metody.fragmenty import cechy_okien, opis_cech


class _NaCechachOkien(Detektor):
    grupa = "uczenie maszynowe"
    nazwa_bazowa = "-"

    def __init__(self, w: int = 50, krok: int = 5, zestaw: str = "s1", P: int = 50, cechy: int = 5,
                 nauka: str = "szereg", nazwa=None):
        self.w, self.krok, self.zestaw, self.P, self.liczba_cech, self.nauka = w, krok, zestaw, P, cechy, nauka
        self.model = None
        opis = opis_cech(cechy) + (", uczony na szeregach bez anomalii" if nauka == "czyste" else "")
        self.nazwa = nazwa or f"{self.nazwa_bazowa} ({opis})"

    def cechy(self, x, krok=None):
        return cechy_okien(x, self.w, krok or self.krok, self.zestaw, self.P, self.liczba_cech)

    def _nowy_model(self):
        raise NotImplementedError

    def ucz(self, szeregi_uczace):
        if self.nauka == "czyste":
            F = np.vstack([self.cechy(x, krok=self.w)[1] for x in szeregi_uczace])
            self.model = self._nowy_model(czyste=True).fit(F)

    def _wynik_okien(self, model, F):
        raise NotImplementedError

    def wynik(self, x):
        starty, F = self.cechy(x)
        if self.nauka == "czyste":
            return do_punktow(starty, self._wynik_okien(self.model, F), self.w, len(x))
        return do_punktow(starty, self._wynik_szeregu(starty, F), self.w, len(x))

    def _wynik_szeregu(self, starty, F):
        return self._wynik_okien(self._nowy_model().fit(F), F)


class IsolationForest(_NaCechachOkien):
    """ML1 - Isolation Forest: losowe drzewa dzielą przestrzeń cech losowymi cięciami; nietypowe okna zostają
    odizolowane po niewielu cięciach. Wynik okna = −score_samples (im wyższy, tym bardziej nietypowe).
    Ziarno drzew ustalone - wynik powtarzalny."""
    nazwa_bazowa = "Isolation Forest"

    def __init__(self, w: int = 50, krok: int = 5, n_drzew: int = 200, ziarno: int = 0, zestaw: str = "s1",
                 P: int = 50, cechy: int = 5, nauka: str = "szereg", nazwa=None):
        self.n_drzew, self.ziarno = n_drzew, ziarno
        super().__init__(w, krok, zestaw, P, cechy, nauka, nazwa)

    def _nowy_model(self, czyste=False):
        return _IF(n_estimators=self.n_drzew, random_state=self.ziarno)

    def _wynik_okien(self, model, F):
        return -model.score_samples(F)


class LOF(_NaCechachOkien):
    """ML2 - Local Outlier Factor: porównuje gęstość okna w przestrzeni cech z gęstością jego k sąsiadów;
    LOF ≫ 1 = okno w rzadszym obszarze niż sąsiedzi. Sąsiedzi wybierani tylko spośród okien nienakładających się
    w czasie (jak w kNN)."""
    nazwa_bazowa = "LOF"

    def __init__(self, w: int = 50, krok: int = 5, k: int = 10, zestaw: str = "s1", cechy: int = 5,
                 nauka: str = "szereg", nazwa=None):
        self.k = k
        super().__init__(w, krok, zestaw, 50, cechy, nauka, nazwa)

    def _nowy_model(self, czyste=False):            # wariant uczony na szeregach bez anomalii: tryb „novelty”
        return LocalOutlierFactor(n_neighbors=self.k, novelty=True)

    def _wynik_okien(self, model, F):
        return -model.score_samples(F)

    def _wynik_szeregu(self, starty, F):
        D = odleglosci_bez_trywialnych(F, starty, self.w)
        D[~np.isfinite(D)] = 1e6
        np.fill_diagonal(D, 0.0)
        lof = LocalOutlierFactor(n_neighbors=self.k, metric="precomputed").fit(D)
        return -lof.negative_outlier_factor_


class OneClassSVM(_NaCechachOkien):
    """ML2 (S2) - One-Class SVM: uczy się granicy obszaru, w którym leży większość okien w przestrzeni cech
    (jądro gaussowskie RBF). Parametr ν = 0,05 to górna granica odsetka okien, które model może zostawić poza granicą
    (≈ spodziewany odsetek nietypowych okien). Wynik okna = −decision_function: im dalej na zewnątrz, tym wyżej.
    Na tle drzew (Isolation Forest) sprawdza, czy gładka granica radzi sobie lepiej (plan, rozdz. 4.4).

    Szerokość jądra gamma = 0,01 zamiast domyślnej „scale” (≈ 0,26 przy 4 standaryzowanych cechach): jądro RBF
    K = exp(−gamma·d²) dla okien odległych o d ≫ 1/√gamma jest ≈ 0, więc wynik się NASYCA - okno bardzo nietypowe
    dostaje prawie ten sam wynik co lekko nietypowe i nie przekracza progu. Nietypowe okna odstają w przestrzeni cech
    o d ≈ 5–10, więc gamma·d² powinno być rzędu 1 → gamma ≈ 0,01. Porównanie wartości gamma: analiza dodatkowa S2."""
    nazwa_bazowa = "One-Class SVM"

    def __init__(self, w: int = 50, krok: int = 5, nu: float = 0.05, gamma=0.01, zestaw: str = "s2", nazwa=None,
                 cechy: int = 5, nauka: str = "szereg"):
        self.nu, self.gamma = nu, gamma
        super().__init__(w, krok, zestaw, 50, cechy, nauka, nazwa)

    def _nowy_model(self, czyste=False):
        return _OCSVM(nu=self.nu, gamma=self.gamma)

    def _wynik_okien(self, model, F):
        return -model.decision_function(F)


class Autoenkoder(Detektor):
    """ML2 (S3) - autoenkoder: sieć neuronowa uczona ODTWARZAĆ okna szeregu (wejście = wyjście) przez wąskie
    „gardło” (warstwę ukrytą o kilku neuronach). Przez gardło przechodzi tylko to, co typowe dla większości okien -
    w S3 kształt jednego cyklu sinusa. Okno z inną fazą, amplitudą lub okresem sieć odtwarza gorzej, więc
    błąd odtworzenia (średni kwadrat) = wynik anomalii okna.

    Okna długości w = P (jeden cykl), krok 5, dzielone przez odporne σ szeregu - BEZ normalizowania każdego okna
    osobno, żeby zmiana amplitudy (S3-B) nie znikała (poprawka po pilocie, plan rozdz. 8).
    Sieć: MLPRegressor (scikit-learn), warstwy w → 32 → gardło → 32 → w, aktywacja tanh, ustalone ziarno.

    Nauka na tym samym szeregu, w którym szukamy anomalii (nienadzorowo, jak pozostałe metody), w dwóch krokach:
    po pierwszym uczeniu odrzucamy `odrzuc` (10%) najgorzej odtwarzanych okien i uczymy sieć ponownie - żeby nie
    nauczyła się odtwarzać także anomalii (poprawka po pilocie). nauka="czyste" - wariant półnadzorowany: sieć uczona
    raz (ucz()), na oknach 20 osobnych szeregów BEZ anomalii, bez odrzucania okien.
    """
    nazwa = "autoenkoder"
    grupa = "uczenie maszynowe"

    def __init__(self, w: int = 50, krok: int = 5, gardlo: int = 2, odrzuc: float = 0.1, nauka: str = "szereg",
                 ziarno: int = 0, max_iter: int = 300, aktywacja: str = "relu", nazwa=None):
        self.w, self.krok, self.gardlo, self.odrzuc, self.nauka = w, krok, gardlo, odrzuc, nauka
        self.aktywacja = aktywacja
        self.ziarno, self.max_iter = ziarno, max_iter
        self.siec = None
        self.nazwa = nazwa or self.nazwa

    def _okna(self, x):
        _, s = odporne_mu_sigma(x)
        starty, m = okna((x - np.median(x)) / s, self.w, self.krok)
        return starty, m

    def _ucz(self, M):
        siec = MLPRegressor(hidden_layer_sizes=(32, self.gardlo, 32), activation=self.aktywacja, max_iter=self.max_iter,
                            random_state=self.ziarno, tol=1e-5)
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            siec.fit(M, M)
            if self.odrzuc > 0:
                b = ((siec.predict(M) - M) ** 2).mean(1)
                zostaw = b <= np.quantile(b, 1 - self.odrzuc)
                siec.fit(M[zostaw], M[zostaw])
        return siec

    def ucz(self, szeregi_uczace):
        if self.nauka == "czyste":
            odrzuc, self.odrzuc = self.odrzuc, 0.0      # dane uczące są bez anomalii - nie ma czego odrzucać
            self.siec = self._ucz(np.vstack([self._okna(x)[1] for x in szeregi_uczace]))
            self.odrzuc = odrzuc

    def wynik(self, x):
        starty, M = self._okna(x)
        siec = self.siec if self.nauka == "czyste" else self._ucz(M)
        return do_punktow(starty, ((siec.predict(M) - M) ** 2).mean(1), self.w, len(x))
