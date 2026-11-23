"""Etap 2 - NYC Taxi (Numenta Anomaly Benchmark): ocena ilościowa metod na rzeczywistym szeregu z etykietami.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.rzeczywiste.nyc_taxi

Szereg: liczba pasażerów taksówek w Nowym Jorku co 30 min, 2014-07-01 – 2015-01-31 (10 320 punktów, 215 dni).
Właściwość szeregu: silny cykl dobowy (48 punktów) i tygodniowy (336) - szereg OKRESOWY jak w S3, ale cykl nie jest
sinusem (dni robocze i weekend mają inny przebieg) i jest multiplikatywny (w nocy ruch jest kilka razy mniejszy).
Anomalie NAB: 5 okien (po 207 punktów ≈ 4,3 doby) - maraton, Święto Dziękczynienia, Boże Narodzenie, Nowy Rok,
śnieżyca - czyli zaburzenia okresowości i krótkotrwałe spadki poziomu ruchu.

Przygotowanie: log liczby pasażerów (cykl multiplikatywny → addytywny); dwa przedziały zmiany czasu 2014-11-02
01:00 i 01:30 (zsumowane dwie godziny - maksimum szeregu, artefakt danych) dzielimy przez 2.

Metody - zestaw S3 (szereg okresowy), z okresem STL = tydzień (P = 336) i oknem = doba (w = 48):
    STL + EWMA (pamięć karty = doba), PELT na resztach STL, Matrix Profile z normalizacją (okno doba),
    cechy cykli + kNN (5 / 3 cechy względem profilu tygodnia - baza.cechy_okien_rz), Isolation Forest (te same cechy),
    autoenkoder (okna doby); warianty ML uczone na okresie kalibracyjnym (dane „historyczne” przed oknami NAB);
    porównawczo: Matrix Profile bez normalizacji i CUSUM z S1 (bez modelu cyklu).

Podział czasu: okres KALIBRACYJNY = pierwsze 15 pełnych tygodni (2014-07-01 – 2014-10-13, 105 dni, przed pierwszym
oknem NAB 2014-10-30) - uczenie wariantów ML, kara PELT, próg (alarm w 10% dni); okres OCENY = 2014-10-14 – 2015-01-31
(110 dni, wszystkie 5 okien). Ocena: okno wykryte = alarm w oknie ± 24 punkty (pół doby); fałszywy alarm = alarm
w dniu bez anomalii (dzień w całości poza oknami ± 24 punkty); AUC / AP - punkty okna wobec punktów normy w okresie
oceny. Wariant porównawczy progu: równy budżet - alarm w 10% dni bez anomalii okresu oceny (wspolne.py).

Wyniki: wyniki/rzeczywiste/nyc_taxi_{wyniki_punktow,okna,metody,dni,pelt_kary}.csv; wykresy - wykresy_rzeczywiste.py.
"""
import numpy as np
import pandas as pd

from src.config import NYC_TAXI, WYNIKI
from src.metody import CUSUM, PELTSTL, STLEWMA, Autoenkoder, CechyKNN, IsolationForest, MatrixProfile
from src.rzeczywiste.wspolne import KARY, dzienne_maksima, kara_pelt, odsetek_dni_z_alarmem, prog_dni
from sklearn.metrics import average_precision_score, roc_auc_score

KATALOG = WYNIKI / "rzeczywiste"
DOBA, TYDZIEN, TOL, ALFA = 48, 336, 24, 0.10
ALFY = (0.02, 0.05, 0.10)          # odsetek dni z alarmem: główny 10% (jak w etapie 1) i surowsze progi
DNI_KAL = 105
ZDARZENIA = ["maraton nowojorski", "Święto Dziękczynienia", "Boże Narodzenie", "Nowy Rok", "śnieżyca (zakaz ruchu)"]
# Dni świąteczne NIEoznaczone w NAB (sprawdzane osobno - czy metody je wskazują).
SWIETA = {"2014-07-04": "Dzień Niepodległości", "2014-09-01": "Labor Day", "2014-10-13": "Columbus Day",
          "2014-11-11": "Dzień Weterana", "2015-01-19": "Dzień Martina Luthera Kinga"}
POROWNAWCZE = ("Matrix Profile bez normalizacji", "CUSUM z S1 (bez modelu cyklu)")
UCZ = "uczony na okresie kalibracyjnym"


def wczytaj():
    """(ramka danych, y = log liczby pasażerów z poprawką zmiany czasu, numer dnia dla każdego punktu)."""
    df = pd.read_parquet(NYC_TAXI["clean"])
    v = df["pasazerowie"].astype(float).where(~df["zmiana_czasu"], df["pasazerowie"] / 2)
    y = np.log(v.values)
    dzien = (np.arange(len(df)) // DOBA).astype(int)
    return df, y, dzien


def metody():
    mp = MatrixProfile(w=DOBA)
    mp.nazwa = "Matrix Profile bez normalizacji"
    m = [STLEWMA(P=TYDZIEN, pamiec=DOBA), PELTSTL(P=TYDZIEN, korekta_zaleznosci=True),
         MatrixProfile(w=DOBA, normalizacja=True)]
    m += [CechyKNN(w=DOBA, krok=4, zestaw="rz", P=TYDZIEN, cechy=c) for c in (5, 3)]
    m += [IsolationForest(w=DOBA, krok=4, zestaw="rz", P=TYDZIEN, cechy=c) for c in (5, 3)]
    m.append(Autoenkoder(w=DOBA, krok=4))
    m += [IsolationForest(w=DOBA, krok=4, zestaw="rz", P=TYDZIEN, cechy=c, nauka="czyste",
                          nazwa=f"Isolation Forest ({'3 cechy' if c == 3 else '5 cech'}, {UCZ})") for c in (5, 3)]
    m.append(Autoenkoder(w=DOBA, krok=4, nauka="czyste", nazwa=f"autoenkoder ({UCZ})"))
    m += [mp, CUSUM(nazwa="CUSUM z S1 (bez modelu cyklu)")]
    return m


def maski(df):
    """Maski punktów: okna NAB (numer zdarzenia 1..5, 0 = brak), strefy tolerancji, okres kalibracji / oceny."""
    n = len(df)
    okno = np.zeros(n, int)
    for i, z in enumerate(ZDARZENIA, 1):
        okno[(df["zdarzenie"] == z).values] = i
    strefa = np.zeros(n, bool)
    for i in range(1, 6):
        idx = np.flatnonzero(okno == i)
        strefa[max(0, idx[0] - TOL):idx[-1] + 1 + TOL] = True
    kal = np.arange(n) < DNI_KAL * DOBA
    return okno, strefa, kal


def policz():
    df, y, dzien = wczytaj()
    okno, strefa, kal = maski(df)
    ocena = ~kal
    dni_kal = np.arange(DNI_KAL)
    dzien_w_strefie = pd.Series(strefa).groupby(dzien).any()
    dni_norm_ocena = np.array([d for d in range(DNI_KAL, dzien.max() + 1) if not dzien_w_strefie[d]])
    negatywy = ocena & ~strefa

    uczace = [y[kal]]
    tyg = [y[kal][i * TYDZIEN:(i + 1) * TYDZIEN] for i in range(DNI_KAL * DOBA // TYDZIEN)]
    wyniki, wiersze_okna, wiersze_metody, diag, rho = {}, [], [], [], np.nan
    for m in metody():
        m.ucz(uczace)
        if isinstance(m, CUSUM):
            m.kalibruj(tyg)                    # skale kart: tygodnie okresu kalibracyjnego
        warianty = {}
        if isinstance(m, PELTSTL):
            r = m.reszty(y)
            # diagnostyka: odsetek dni kalibracji z alarmem w funkcji kary - bez i z korektą na zależność reszt
            for kor in (False, True):
                m.korekta_zaleznosci = kor
                for k in KARY:
                    al = np.where(m._z(r, k) > m.Z_KRYT, 1.0, 0.0) > 0
                    diag.append({"korekta_zaleznosci": kor, "kara": int(k),
                                 "dni_alarmu_kal": odsetek_dni_z_alarmem(al, dzien, dni_kal),
                                 "dni_alarmu_ocena": odsetek_dni_z_alarmem(al, dzien, dni_norm_ocena, ~strefa)})
            m.korekta_zaleznosci = True
            rho = float(np.corrcoef(r[1:], r[:-1])[0, 1])
            wk = lambda k, m=m, r=r: np.where((z := m._z(r, k)) > m.Z_KRYT, z, 0.0)
            for alfa in ALFY:
                k_kal = kara_pelt(wk, dzien, dni_kal, alfa)
                k_bud = kara_pelt(wk, dzien, dni_norm_ocena, alfa, maska_pkt=~strefa)
                warianty[("kalibracja", alfa)] = (wk(k_kal), 0.0, k_kal)
                warianty[("rowny_budzet", alfa)] = (wk(k_bud), 0.0, k_bud)
            s = warianty[("kalibracja", ALFA)][0]
        else:
            s = m.wynik(y)
            for alfa in ALFY:
                warianty[("kalibracja", alfa)] = (s, prog_dni(s, dzien, dni_kal, alfa), np.nan)
                warianty[("rowny_budzet", alfa)] = (s, prog_dni(s, dzien, dni_norm_ocena, alfa), np.nan)
        wyniki[m.nazwa] = s
        ap = average_precision_score(okno[(okno > 0) | negatywy] > 0, s[(okno > 0) | negatywy])
        auc_all = roc_auc_score(okno[(okno > 0) | negatywy] > 0, s[(okno > 0) | negatywy])
        for (wariant, alfa), (sw, prog, kara) in warianty.items():
            alarm = sw > prog
            wykryte = []
            for i, z in enumerate(ZDARZENIA, 1):
                idx = np.flatnonzero(okno == i)
                a, b = max(0, idx[0] - TOL), idx[-1] + 1 + TOL
                msk = (okno == i) | negatywy
                wykryte.append(bool(alarm[a:b].any()))
                wiersze_okna.append({"metoda": m.nazwa, "wariant": wariant, "alfa": alfa, "zdarzenie": z,
                                     "wykryte": wykryte[-1],
                                     "auc": roc_auc_score(okno[msk] == i, sw[msk]),
                                     "maks_w_oknie_wzgl_progu": float(sw[idx].max() / prog) if prog > 0 else np.nan})
            fa_dni = odsetek_dni_z_alarmem(alarm, dzien, dni_norm_ocena, maska_pkt=~strefa)
            fa_kal = odsetek_dni_z_alarmem(alarm, dzien, dni_kal)
            wiersze_metody.append({"metoda": m.nazwa, "grupa": m.grupa, "wariant": wariant, "alfa": alfa, "prog": prog,
                                   "kara_pelt": kara, "wykryte_okna": int(sum(wykryte)),
                                   "fa_dni_ocena": fa_dni, "fa_dni_liczba": int(round(fa_dni * len(dni_norm_ocena))),
                                   "dni_norm_ocena": len(dni_norm_ocena), "fa_dni_kalibracja": fa_kal,
                                   "auc_wszystkie_okna": auc_all, "ap": ap,
                                   "autokorelacja_reszt": rho if isinstance(m, PELTSTL) else np.nan})
        wm = [r for r in wiersze_metody if r["metoda"] == m.nazwa and r["wariant"] == "kalibracja"]
        print(f"  {m.nazwa}: AUC {auc_all:.2f}, AP {ap:.2f} | kalibracja α = 2/5/10%: okna "
              + " / ".join(f"{r['wykryte_okna']}" for r in wm) + ", fałszywe alarmy "
              + " / ".join(f"{r['fa_dni_ocena']:.0%}" for r in wm), flush=True)

    # dzienne maksima wyniku (do tabel dni i świąt) - ranking dnia wśród dni swojego okresu
    dni = pd.DataFrame({"data": pd.to_datetime(df.index[::DOBA]).date, "okres": np.where(np.arange(dzien.max() + 1)
                        < DNI_KAL, "kalibracja", "ocena"), "zdarzenie_nab": pd.Series(okno).groupby(dzien).max().map(
                        lambda i: ZDARZENIA[i - 1] if i else "").values,
                        "w_strefie": dzien_w_strefie.values})
    for nazwa, s in wyniki.items():
        dni[nazwa] = dzienne_maksima(s, dzien).values
    KATALOG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(wyniki, index=df.index).to_csv(KATALOG / "nyc_taxi_wyniki_punktow.csv", float_format="%.5g")
    pd.DataFrame(wiersze_okna).to_csv(KATALOG / "nyc_taxi_okna.csv", index=False)
    pd.DataFrame(wiersze_metody).to_csv(KATALOG / "nyc_taxi_metody.csv", index=False)
    dni.to_csv(KATALOG / "nyc_taxi_dni.csv", index=False, float_format="%.5g")
    pd.DataFrame(diag).to_csv(KATALOG / "nyc_taxi_pelt_kary.csv", index=False, float_format="%.4g")
    print("Zapisano", KATALOG / "nyc_taxi_*.csv")


if __name__ == "__main__":
    policz()
