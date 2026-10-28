"""Metody wykrywania anomalii we wspólnym interfejsie (src/metody/baza.py).

Zestawy metod dla scenariuszy (docs/plan_etap_1.md, rozdz. 4.2) - po 2 metody z każdej grupy; metody na cechach okien
(kNN, Isolation Forest, LOF / One-Class SVM) w dwóch wariantach: 5 cech i 3 cechy (po jednej na rodzaj anomalii) -
ustalenie z konsultacji 27.10.2026:
    metody_s1() - biały szum (+ Matrix Profile z normalizacją porównawczo),
    metody_s2() - AR(1),
    metody_s3() - szum + sinus (+ Matrix Profile bez normalizacji porównawczo).
uczone=True dokłada warianty metod ML uczone na szeregach bez anomalii (część główna eksperymentu).
"""
from src.metody.fragmenty import CechyKNN, MatrixProfile
from src.metody.ml import LOF, Autoenkoder, IsolationForest, OneClassSVM
from src.metody.statystyczne import CUSUM, PELT, PELTAR, PELTSTL, STLEWMA, CUSUMHybryda, ResztyARCUSUM

CECHY = (5, 3)


def metody_s1(w: int = 50, z_porownawczymi: bool = True, uczone: bool = False) -> list:
    m = [CUSUM(), PELT(), MatrixProfile(w=w)]
    m += [CechyKNN(w=w, cechy=c) for c in CECHY]
    m += [IsolationForest(w=w, cechy=c) for c in CECHY]
    m += [LOF(w=w, cechy=c) for c in CECHY]
    if uczone:
        m += [IsolationForest(w=w, cechy=c, nauka="czyste") for c in CECHY]
        m += [LOF(w=w, cechy=c, nauka="czyste") for c in CECHY]
    if z_porownawczymi:
        m.append(MatrixProfile(w=w, normalizacja=True))
    return m


def metody_s2(w: int = 50, uczone: bool = False) -> list:
    m = [ResztyARCUSUM(), PELTAR(), MatrixProfile(w=w, na_resztach=True)]
    m += [CechyKNN(w=w, zestaw="s2", cechy=c) for c in CECHY]
    m += [IsolationForest(w=w, zestaw="s2", cechy=c) for c in CECHY]
    m += [OneClassSVM(w=w, zestaw="s2", cechy=c) for c in CECHY]
    if uczone:
        m += [IsolationForest(w=w, zestaw="s2", cechy=c, nauka="czyste") for c in CECHY]
        m += [OneClassSVM(w=w, zestaw="s2", cechy=c, nauka="czyste") for c in CECHY]
    return m


def metody_s3(w: int = 50, P: int = 50, z_porownawczymi: bool = True, uczone: bool = False) -> list:
    """S3: okno metod fragmentowych i ML domyślnie = okres cyklu P (jeden cykl)."""
    m = [STLEWMA(P), PELTSTL(P), MatrixProfile(w=w, normalizacja=True)]
    m += [CechyKNN(w=w, zestaw="s3", P=P, cechy=c) for c in CECHY]
    m += [IsolationForest(w=w, zestaw="s3", P=P, cechy=c) for c in CECHY]
    m.append(Autoenkoder(w=w))
    if uczone:
        m += [IsolationForest(w=w, zestaw="s3", P=P, cechy=c, nauka="czyste") for c in CECHY]
        m.append(Autoenkoder(w=w, nauka="czyste", nazwa="autoenkoder (uczony na szeregach bez anomalii)"))
    if z_porownawczymi:
        mp = MatrixProfile(w=w)
        mp.nazwa = "Matrix Profile bez normalizacji"
        m.append(mp)
    return m
