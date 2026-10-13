"""Czyszczenie szeregu NYC Taxi (NAB) i dołączenie etykiet anomalii.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.czyszczenie.nyc_taxi

Szereg: liczba pasażerów taksówek w Nowym Jorku w przedziałach 30 min, 2014-07-01 - 2015-01-31 (10 320 punktów,
48 na dobę). Silna okresowość dobowa (48) i tygodniowa (336). Czas lokalny Nowego Jorku BEZ strefy czasowej
(tak jak w NAB) - nie da się go jednoznacznie przypisać strefie, bo 2014-11-02 godzina 1:00-2:00 wystąpiła dwa razy.

Kolumny wyniku:
- pasazerowie    : liczba pasażerów w przedziale 30 min (bez zmian względem NAB),
- anomalia_nab   : bool, punkt leży w oknie anomalii NAB (etykieta "prawdziwa" do oceny metod),
- punkt_nab      : bool, dokładny punkt anomalii wskazany przez NAB (środek okna),
- zdarzenie      : nazwa zdarzenia dla okien NAB (maraton, Święto Dziękczynienia, ...), puste poza oknami,
- zmiana_czasu   : bool, przedział zaburzony zmianą czasu (2014-11-02 01:00 i 01:30 - zsumowane dwie godziny,
                   stąd ok. 2x zawyżona wartość; to MAKSIMUM całego szeregu i nie jest prawdziwą anomalią ruchu),
- luka_przed, dzien_niepelny : jak w pozostałych zbiorach (w tej wersji danych brak luk).
"""
import json

import pandas as pd

from src.config import NYC_TAXI, najnowszy_plik
from src.czyszczenie.wspolne import oznacz_luki, raport_luk

# Nazwy zdarzeń wg opisu NAB (data/README.md); okna w kolejności chronologicznej.
ZDARZENIA = ["maraton nowojorski", "Święto Dziękczynienia", "Boże Narodzenie", "Nowy Rok", "śnieżyca (zakaz ruchu)"]

# Zmiana czasu z letniego na zimowy w nocy 2014-11-01/02: godzina 1:00-2:00 wystąpiła dwa razy.
ZMIANA_CZASU = pd.to_datetime(["2014-11-02 01:00", "2014-11-02 01:30"])


def prepare_clean_data():
    csv_path = najnowszy_plik(NYC_TAXI["surowe_dir"], NYC_TAXI["surowe_wzorzec"])
    lab_path = najnowszy_plik(NYC_TAXI["surowe_dir"], NYC_TAXI["etykiety_wzorzec"])
    print("Pliki surowe:", csv_path.name, "|", lab_path.name)

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "t", "value": NYC_TAXI["kolumna"]}).set_index("t").sort_index()
    if df.index.has_duplicates:
        raise ValueError("Zduplikowane znaczniki czasu w danych NYC Taxi.")
    print(f"Punktów: {len(df)} | zakres: {df.index.min()} - {df.index.max()} | braki NaN: {int(df.isna().sum().sum())}")

    etykiety = json.loads(lab_path.read_text(encoding="utf-8"))
    okna = sorted((pd.Timestamp(a), pd.Timestamp(b)) for a, b in etykiety["okna"])
    punkty = pd.to_datetime(etykiety["punkty"])
    if len(okna) != len(ZDARZENIA):
        raise ValueError(f"Oczekiwano {len(ZDARZENIA)} okien NAB, jest {len(okna)} - sprawdź nazwy zdarzeń.")

    df["anomalia_nab"] = False
    df["zdarzenie"] = ""
    for (od, do), nazwa in zip(okna, ZDARZENIA):
        w = (df.index >= od) & (df.index <= do)
        df.loc[w, "anomalia_nab"] = True
        df.loc[w, "zdarzenie"] = nazwa
        punkt = punkty[(punkty >= od) & (punkty <= do)]
        print(f"  okno NAB: {od} - {do} ({w.sum()} pkt) | punkt: {list(punkt.astype(str))} | {nazwa}")
    df["punkt_nab"] = df.index.isin(punkty)
    df["zmiana_czasu"] = df.index.isin(ZMIANA_CZASU)

    siatka = pd.date_range(df.index.min(), df.index.max(), freq=NYC_TAXI["freq"], name="t")
    df, missing = oznacz_luki(df, siatka, segment=None)

    kol = NYC_TAXI["kolumna"]
    print(f"Punktów w oknach anomalii: {df['anomalia_nab'].sum()} ({df['anomalia_nab'].mean():.1%})")
    print("Przedziały ze zmianą czasu:", df.loc[df["zmiana_czasu"], kol].to_dict(), "| maksimum szeregu:", df[kol].idxmax())
    raport_luk(missing)

    NYC_TAXI["clean"].parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(NYC_TAXI["clean"])
    print(f"Zapisano dane do: {NYC_TAXI['clean']}")


if __name__ == "__main__":
    prepare_clean_data()
