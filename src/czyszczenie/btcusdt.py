"""Czyszczenie danych BTCUSDT 5-min (Binance spot).

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.czyszczenie.btcusdt

Kryptowaluty są notowane 24/7, więc nie ma godzin sesji ani kalendarza:
oczekiwana siatka to wszystkie znaczniki co 5 min od pierwszej do ostatniej świecy (288 świec na dobę, UTC).

Kroki:
1. Usunięcie duplikatów, sortowanie, wyrównanie znaczników spoza siatki 5-min, kontrola spójności OHLC.
2. Porównanie z siatką 5-min i oznaczenie brakujących świec (przerwy techniczne Binance) - bez uzupełniania:
   - luka_przed      : int, ile świec brakuje bezpośrednio przed tą świecą (szereg ciągły, także przez północ),
   - dzien_niepelny  : bool, w danym dniu (UTC) brakuje co najmniej jednej świecy,
   - zerowy_wolumen  : bool, świeca bez żadnej transakcji (v == 0) - cena przepisana, nie rynkowa,
   - czas_skorygowany: bool, znacznik czasu był poza siatką 5-min i został wyrównany (241 świec z lutego 2018).
3. Pierwszy (niepełny) dzień notowań 2017-08-17 nie jest traktowany jako luka - siatka zaczyna się od pierwszej świecy.
Lista brakujących znaczników czasu: dane/clean/btcusdt/btcusdt_brakujace_swiece.csv.
"""
import pandas as pd

from src.config import BTCUSDT, FREQ, najnowszy_plik
from src.czyszczenie.wspolne import oznacz_luki, raport_luk


def prepare_clean_data():
    raw_path = najnowszy_plik(BTCUSDT["surowe_dir"], BTCUSDT["surowe_wzorzec"])
    print("Plik surowy:", raw_path.name)
    btc = pd.read_parquet(raw_path)
    btc = btc[~btc.index.duplicated(keep="first")].sort_index()

    # Wyrównanie do siatki 5-min: po przerwie 2018-02-08/09 Binance zapisał 241 świec (2018-02-09 09:58 - 2018-02-10 05:58)
    # ze znacznikami przesuniętymi o 194,789 s względem siatki. round() przesuwa je o 105,211 s do przodu,
    # czyli do przedziału 5-min, z którym świeca pokrywa się w większości (3 min 15 s z 5 min).
    aligned_index = btc.index.round(FREQ)
    shifted = btc.index != aligned_index
    btc["czas_skorygowany"] = shifted
    if shifted.any():
        shifts = (btc.index[shifted] - aligned_index[shifted]).total_seconds()
        print(
            f"Znaczniki poza siatką {FREQ}: {int(shifted.sum())}; "
            f"normalizuję do najbliższej świecy (maks. przesunięcie: {abs(shifts).max():.3f} s)."
        )
        btc.index = aligned_index
        if btc.index.has_duplicates:
            raise ValueError("Normalizacja czasu utworzyła zduplikowane świece; wymagane ręczne sprawdzenie danych.")

    # kontrola spójności OHLC
    zle = (btc.h < btc.l) | (btc.c > btc.h) | (btc.c < btc.l) | (btc.o > btc.h) | (btc.o < btc.l)
    print("Świece z niespójnym OHLC:", int(zle.sum()))
    if zle.any():
        print(btc[zle].head().to_string())

    btc["zerowy_wolumen"] = btc["v"] == 0

    siatka = pd.date_range(btc.index.min(), btc.index.max(), freq=FREQ, tz="UTC", name="t")
    btc, missing = oznacz_luki(btc, siatka, segment=None)

    liczba = btc.groupby(btc.index.date).size()
    print(f"Świec: {len(btc)} | oczekiwanych w siatce: {len(siatka)} | zakres: {btc.index.min()} - {btc.index.max()}")
    print(f"Dni: {len(liczba)}, w tym pełnych (288 świec): {(liczba == 288).sum()}")
    print("Świec z zerowym wolumenem:", int(btc["zerowy_wolumen"].sum()))
    if len(missing):
        # najdłuższe przerwy (ciągłe bloki brakujących świec)
        blok = (missing.to_series().diff() != pd.Timedelta(FREQ)).cumsum()
        przerwy = missing.to_series().groupby(blok.to_numpy()).agg(["first", "last", "count"])
        przerwy["czas"] = przerwy["count"] * pd.Timedelta(FREQ)
        print(f"Przerw (ciągłych bloków braków): {len(przerwy)}; najdłuższe:")
        print(przerwy.sort_values("count", ascending=False).head(10).to_string(index=False))
    raport_luk(missing, maks_dni=15)

    BTCUSDT["clean"].parent.mkdir(parents=True, exist_ok=True)
    btc.to_parquet(BTCUSDT["clean"])
    pd.DataFrame({"t": missing}).to_csv(BTCUSDT["brakujace"], index=False, lineterminator="\n")
    print(f"Zapisano dane do: {BTCUSDT['clean']}")
    print(f"Zapisano listę brakujących świec do: {BTCUSDT['brakujace']}")


if __name__ == "__main__":
    prepare_clean_data()
