# Projekt inżynierski – podsumowanie

> Plik-ściągawka: co robimy, co już jest, jak to uruchomić i co dalej.
> Ostatnia weryfikacja stanu repozytorium: **2026-10-13**.

## 1. Temat i cel

**Wykrywanie anomalii w szeregach czasowych z wykorzystaniem metod statystycznych i uczenia maszynowego.**

Cel: opracować i porównać metody automatycznego wykrywania anomalii w szeregach czasowych oraz zbadać,
jak ich skuteczność zależy od:

- **rodzaju anomalii** – pojedyncze obserwacje odstające (punktowe), zmiana poziomu, zmiana zmienności (wariancji),
  zaburzenie struktury czasowej (np. okresowości),
- **intensywności** anomalii (amplituda względem szumu),
- **czasu trwania** anomalii,
- **właściwości szeregu** (trend, sezonowość, autokorelacja, heteroskedastyczność),
- **poziomu szumu**.

Weryfikacja przede wszystkim **symulacyjna** (znane położenie i typ anomalii → można liczyć skuteczność
detekcji i odsetek fałszywych alarmów). Dodatkowo analiza **rzeczywistego szeregu** – w tym projekcie: SPY.

### Grupy metod do porównania

Szczegółowy dobór metod do rodzajów anomalii występujących w danych: [docs/plan_metod.md](docs/plan_metod.md).

| Grupa | Przykładowe metody |
|---|---|
| Klasyczne statystyczne | z-score / robust z-score (mediana, MAD), reguła IQR, wykresy kontrolne (Shewhart, CUSUM, EWMA), reszty z modelu ARIMA / STL |
| Podobieństwo / charakterystyki fragmentów | okno przesuwne + cechy (średnia, wariancja, entropia, widmo), odległości między fragmentami (matrix profile / discords), kNN na oknach |
| Uczenie maszynowe | Isolation Forest, Local Outlier Factor, One-Class SVM, (opcjonalnie) autoenkoder / LSTM – błąd rekonstrukcji |

### Metryki oceny (w symulacjach)

- precyzja, czułość (recall), F1 – na poziomie punktów i na poziomie zdarzeń (anomalia wykryta, jeśli trafiony choć jeden punkt w oknie),
- odsetek fałszywych alarmów (FPR) na szeregu bez anomalii,
- opóźnienie detekcji (dla zmian poziomu / wariancji),
- ROC-AUC / PR-AUC dla metod zwracających wynik ciągły.

## 2. Repozytorium

- GitHub: https://github.com/MirekMyszka/Projekt-Inz (gałąź `main`)
- Lokalnie: `Projekt Inż/Projekt-Inz/` (środowisko Pythona: `Projekt Inż/.venv`, Python 3.10).

```
Projekt-Inz/
├── README.md, PROJEKT.md, requirements.txt, .env.example, .gitignore
├── src/
│   ├── config.py                  # ścieżki i parametry zbiorów (zakresy dat, nazwy plików)
│   ├── pobieranie/
│   │   ├── spy_alpaca.py          # SPY 5-min z Alpaca (klucze z .env)
│   │   ├── btcusdt_binance.py     # BTCUSDT 5-min z archiwum Binance (bez klucza)
│   │   └── nyc_taxi_nab.py        # NYC Taxi + etykiety anomalii z NAB (GitHub)
│   ├── czyszczenie/
│   │   ├── wspolne.py             # oznaczanie luk względem oczekiwanej siatki czasu
│   │   ├── spy.py                 # sesje wg kalendarza NYSE + luki + circuit breaker
│   │   ├── btcusdt.py             # siatka 24/7 + luki + zerowy wolumen
│   │   └── nyc_taxi.py            # etykiety anomalii NAB + artefakt zmiany czasu
│   ├── eksploracja/
│   │   ├── podglad.py             # statystyki i wykresy cen
│   │   └── przeglad_anomalii.py   # przegląd anomalii w danych (podstawa planu metod)
│   ├── symulacje/                 # [do zrobienia]
│   ├── metody/                    # [do zrobienia]
│   └── ewaluacja/                 # [do zrobienia]
├── dane/
│   ├── surowe/spy/      spy_5min_sip_2016_2026_pobrano_2026-10-13.parquet
│   ├── surowe/btcusdt/  btcusdt_5min_binance_2017_2026_pobrano_<data>.parquet
│   ├── surowe/nyc_taxi/ nyc_taxi_nab_pobrano_<data>.csv, nyc_taxi_etykiety_nab_pobrano_<data>.json
│   ├── clean/spy/       spy_5min_session_2016_2026.parquet, spy_brakujace_swiece.csv
│   ├── clean/btcusdt/   btcusdt_5min_2017_2026.parquet, btcusdt_brakujace_swiece.csv
│   ├── clean/nyc_taxi/  nyc_taxi_30min_2014_2015.parquet
│   └── cache/binance/   miesięczne ZIP-y z Binance (poza Gitem)
├── wyniki/wykresy/      spy_5min_close.png, btcusdt_5min_close.png, nyc_taxi_30min.png
└── docs/
    ├── data_pipeline.md         # pobieranie i czyszczenie danych
    ├── plan_metod.md            # anomalie w danych, dobór metod, plan eksperymentów (szeroki)
    └── plan_etap_1.md           # AKTUALNY etap: ustalenia z konsultacji, 3 scenariusze, 6 metod
```

Zasady:
- skrypty uruchamiamy **z katalogu `Projekt-Inz`** jako moduły: `python -m src.<pakiet>.<skrypt>`,
- ścieżki i parametry zmieniamy tylko w `src/config.py`,
- dane surowe nie są modyfikowane; skrypty czyszczące biorą **najnowszy** plik surowy (data pobrania w nazwie).

### Uruchomienie

```bash
pip install -r requirements.txt
copy .env.example .env                     # klucze Alpaca (tylko dla SPY)
python -m src.pobieranie.spy_alpaca        # → dane/surowe/spy/
python -m src.czyszczenie.spy              # → dane/clean/spy/
python -m src.pobieranie.btcusdt_binance   # → dane/surowe/btcusdt/  (+ cache ZIP-ów)
python -m src.czyszczenie.btcusdt          # → dane/clean/btcusdt/
python -m src.pobieranie.nyc_taxi_nab      # → dane/surowe/nyc_taxi/
python -m src.czyszczenie.nyc_taxi         # → dane/clean/nyc_taxi/
python -m src.eksploracja.podglad          # → wyniki/wykresy/
```

## 3. Dane

| | SPY | BTCUSDT | NYC Taxi |
|---|---|---|---|
| Rodzaj | ETF na S&P 500 (akcje USA) | kryptowaluta, para spot BTC/USDT | liczba pasażerów taksówek w Nowym Jorku |
| Źródło | Alpaca Market Data API (`/v2/stocks/bars`, feed `sip`, `adjustment=all`) | Binance, archiwum data.binance.vision (spot, klines) | Numenta Anomaly Benchmark (GitHub, licencja MIT) |
| Interwał | 5 min | 5 min | 30 min |
| Zakres | 2016-01-04 – 2026-09-30 | 2017-08-17 – 2026-09-30 | 2014-07-01 – 2015-01-31 |
| Godziny | sesja NYSE 09:30–16:00 (America/New_York) | 24/7 (UTC) | 24/7 (czas lokalny NY, bez strefy) |
| Punktów na dzień | 78 (skrócona sesja: 42) | 288 | 48 |
| Klucz API | tak (`.env`) | nie | nie |
| **Etykiety anomalii** | nie (tylko katalog zdarzeń) | nie (tylko katalog zdarzeń) | **tak – 5 okien NAB** |

Wspólne kolumny: `o, h, l, c` – OHLC, `v` – wolumen (SPY: akcje, BTC: BTC), `n` – liczba transakcji, `vw` – VWAP.
Indeks `t` = **początek** świecy. BTCUSDT ma dodatkowo `qv` – obrót w USDT.

Dlaczego trzy zbiory: różnią się właściwościami (zmienność BTC kilkukrotnie większa niż SPY, brak przerw nocnych
i weekendowych; NYC Taxi ma silną okresowość dobową i tygodniową), co pozwala sprawdzić, jak metody detekcji zależą
od właściwości szeregu. Tylko NYC Taxi ma etykiety anomalii – pozwala liczyć skuteczność także na danych rzeczywistych.

### 3.1 SPY

| | Surowe | Clean (sesja) |
|---|---|---|
| Wierszy | 496 241 | 209 908 |
| Zakres | 2016-01-01 00:00 UTC – 2026-09-30 23:55 UTC | 2016-01-04 09:30 – 2026-09-30 15:55 (America/New_York) |
| Braki (NaN) / duplikaty indeksu | 0 / 0 | 0 / 0 |

Kolumny dodane przy czyszczeniu:

| Kolumna | Typ | Znaczenie |
|---|---|---|
| `skrocona_sesja` | bool | dzień ze skróconą sesją wg kalendarza NYSE (zamknięcie 13:00) |
| `luka_przed` | int | ile oczekiwanych świec brakuje bezpośrednio przed tą świecą (w obrębie sesji) |
| `dzien_niepelny` | bool | w tym dniu brakuje co najmniej jednej świecy |
| `wstrzymanie_handlu` | bool | dzień z circuit breakerem: 9, 12, 16, 18 marca 2020 |

Czyszczenie (`src/czyszczenie/spy.py`):
1. Konwersja UTC → America/New_York.
2. Filtr do godzin sesji z kalendarza NYSE (`pandas_market_calendars`): `market_open <= t < market_close`.
   Handel przed- i posesyjny (także po 13:00 w dni skrócone) jest odrzucany.
3. Brakujące świece **nie są uzupełniane**, tylko oznaczane i wypisywane do `spy_brakujace_swiece.csv`.

Kontrola jakości (stan na 2026-10-13):
- 2 701 dni sesyjnych = 2 701 dni w kalendarzu NYSE; 2 674 dni po 78 świec, 21 skróconych sesji po 42 świece.
- 14 brakujących świec w 6 dniach: **marzec 2020 – wstrzymania handlu** (po 2 świece 9, 12, 16 i 18 marca);
  2019-08-12 – brak 5 ostatnich świec (15:35–15:55, luka po stronie dostawcy); 2016-02-02 – brak 1 świecy (11:20).
- Spójność OHLC poprawna, brak zerowego wolumenu.
- Log-zwroty 5-min: odch. std ≈ 0,00125; największe ruchy w dzień w kwietniu 2025 (−2,9% / +3,8%),
  największe luki overnight w marcu 2020 (do −11,8%). Cena: ~170 USD (2016) → ~762 USD (09.2026).

### 3.2 BTCUSDT

Pobieranie (`src/pobieranie/btcusdt_binance.py`):
- miesięczne ZIP-y `BTCUSDT-5m-RRRR-MM.zip` z `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/`,
  każdy weryfikowany sumą SHA-256 (`.CHECKSUM`), zapisywane w `dane/cache/binance/` (ponowne uruchomienie pobiera tylko brakujące),
- **od 2025-01-01 znaczniki czasu są w mikrosekundach** (wcześniej ms) – skrypt rozpoznaje to automatycznie,
- obsługa plików z nagłówkiem i bez, usuwanie duplikatów, `vw = qv / v`.

Czyszczenie (`src/czyszczenie/btcusdt.py`): siatka co 5 min od pierwszej do ostatniej świecy (UTC, bez kalendarza), kolumny:

| Kolumna | Typ | Znaczenie |
|---|---|---|
| `luka_przed` | int | ile świec brakuje bezpośrednio przed tą świecą (szereg ciągły – także przez północ) |
| `dzien_niepelny` | bool | w tym dniu (UTC) brakuje co najmniej jednej świecy |
| `zerowy_wolumen` | bool | świeca bez transakcji (`v == 0`) – cena nie jest rynkowa |
| `czas_skorygowany` | bool | znacznik czasu wyrównany do siatki 5-min (241 świec, luty 2018) |

Skrypt wypisuje też listę najdłuższych przerw (ciągłych bloków braków) – przerwy techniczne Binance.

Kontrola jakości (stan na 2026-10-13):
- Surowe = clean: 957 865 świec, 2017-08-17 04:00 – 2026-09-30 23:55 UTC (siatka: 959 568), brak duplikatów i NaN.
- 3 332 dni, 3 298 pełnych (288 świec); pierwszy dzień 240 świec (start notowań o 04:00 UTC).
- **1 703 brakujące świece w 33 przerwach technicznych**; najdłuższa 2018-02-08 00:30 – 2018-02-09 09:55
  (402 świece, ~33,5 h), dalej 2018-06-26 i 2019-05-15 (po 10 h).
- **Przesunięte znaczniki czasu**: po przerwie z lutego 2018 Binance zapisał 241 świec przesuniętych o 194,789 s
  względem siatki – wyrównane do najbliższej granicy 5 min (flaga `czas_skorygowany`).
- 932 świece z zerowym wolumenem (wszystkie płaskie: `o = h = l = c`), z czego 874 w 2017 r.
- Spójność OHLC poprawna, VWAP zawsze w `[l, h]`.
- Log-zwroty 5-min (bez świec po luce): odch. std ≈ 0,00237 (ok. 1,9× SPY); skrajne ruchy w marcu 2020
  (−10,8% 12.03, +16,9% 13.03). Cena: ~4 300 USDT (08.2017) → ~83 600 USDT (09.2026), szczyt ~126 000 USDT (2025-10-06).

### 3.3 NYC Taxi (Numenta Anomaly Benchmark)

Zbiór z **oznaczonymi anomaliami** – jedyny z trzech, na którym na danych rzeczywistych można policzyć czułość,
precyzję i fałszywe alarmy. Silna okresowość dobowa i tygodniowa, więc dobrze testuje **zaburzenia okresowości**.

- źródło: Numenta Anomaly Benchmark (NAB), `data/realKnownCause/nyc_taxi.csv` – liczba pasażerów taksówek w Nowym Jorku
  w przedziałach 30 min (agregacja Numenta z danych NYC Taxi and Limousine Commission); licencja MIT,
- wersja przypięta do commita `ea702d7` (2024-12-03) w `src/config.py` – wyniki są powtarzalne,
- 10 320 punktów, 2014-07-01 00:00 – 2015-01-31 23:30, 48 na dobę, bez braków i duplikatów,
- czas lokalny Nowego Jorku **bez strefy czasowej** (jak w NAB).

Etykiety NAB (`labels/combined_windows.json`, `labels/combined_labels.json`) – 5 okien po 207 punktów (~4,3 doby),
łącznie 10% danych; w każdym oknie jeden punkt anomalii:

| Okno anomalii | Punkt NAB | Zdarzenie |
|---|---|---|
| 2014-10-30 15:30 – 2014-11-03 22:30 | 2014-11-01 19:00 | maraton nowojorski (2.11) |
| 2014-11-25 12:00 – 2014-11-29 19:00 | 2014-11-27 15:30 | Święto Dziękczynienia |
| 2014-12-23 11:30 – 2014-12-27 18:30 | 2014-12-25 15:00 | Boże Narodzenie |
| 2014-12-29 21:30 – 2015-01-03 04:30 | 2015-01-01 01:00 | Nowy Rok |
| 2015-01-24 20:30 – 2015-01-29 03:30 | 2015-01-27 00:00 | śnieżyca – zakaz ruchu (26–27.01, minimum: 8 pasażerów o 03:00) |

Kolumny: `pasazerowie`, `anomalia_nab` (w oknie), `punkt_nab`, `zdarzenie`, `zmiana_czasu`, `luka_przed`, `dzien_niepelny`.

Uwagi (ważne przy ocenie metod):
- **Artefakt zmiany czasu**: 2014-11-02 01:00 i 01:30 mają 39 197 i 35 212 pasażerów – to **maksimum całego szeregu**,
  ale nie prawdziwy ruch: godzina 1:00–2:00 wystąpiła tej nocy dwa razy i została zsumowana w tych samych przedziałach.
  Leży w oknie „maraton”, więc metoda, która „wykrywa maraton”, może w rzeczywistości wykrywać ten artefakt.
  Oznaczone kolumną `zmiana_czasu`, wartości bez zmian.
- **Etykiety NAB są niepełne**: 4–6 lipca 2014 (Dzień Niepodległości) dzienny ruch wynosi 0,65–0,77 typowego dla danego
  dnia tygodnia – tyle co Święto Dziękczynienia (0,69) – a nie jest oznaczony. Prawdopodobnie dlatego, że NAB nie
  ocenia pierwszych 15% danych (okres „rozbiegowy”). Alarm metody 4 lipca to więc nie zawsze fałszywy alarm.
- Okna NAB są szerokie (~4,3 doby) i umowne – lepsza ocena „na poziomie zdarzeń” niż punkt po punkcie.

### 3.4 Uwagi do dalszej analizy

1. **SPY – luki overnight**: zwrot między ostatnią świecą dnia a pierwszą następnego nie jest zwrotem 5-minutowym.
   Liczyć zwroty tylko wewnątrz dnia albo traktować overnight osobno.
2. Ceny nie są stacjonarne – do detekcji anomalii używać **log-zwrotów** lub reszt po usunięciu
   sezonowości dziennej (SPY: kształt „U” zmienności w sesji; BTC: cykl dobowy i tygodniowy aktywności).
3. Świece z `luka_przed > 0` – zwrot do takiej świecy obejmuje więcej niż 5 minut.
4. Rzeczywiste okresy zaburzeń do części empirycznej: marzec 2020 (oba zbiory), kwiecień 2025 (SPY),
   dla BTC m.in. maj 2021 i listopad 2022.

## 4. Stan prac

- [x] Repozytorium na GitHubie, `.env` z kluczami poza Gitem
- [x] SPY 5-min 2016–2026: pobranie, czyszczenie wg kalendarza NYSE, oznaczenie luk i wstrzymań handlu
- [x] Uporządkowana struktura projektu (`src/`, `dane/`, `wyniki/`, `config.py`)
- [x] BTCUSDT 5-min: skrypty pobierania i czyszczenia (przetestowane)
- [x] BTCUSDT: pobranie danych (2026-10-13), czyszczenie i kontrola jakości (p. 3.2)
- [x] Przegląd anomalii w danych i plan doboru metod – [docs/plan_metod.md](docs/plan_metod.md)
- [x] NYC Taxi (NAB): trzeci zbiór, z etykietami anomalii – pobranie, czyszczenie, weryfikacja
- [x] Konsultacja 2026-10-13 – zawężony plan etapu 1: [docs/plan_etap_1.md](docs/plan_etap_1.md)
- [x] Pilot etapu 1: sprawdzenie doboru metod na S1–S3 (40/54 par z AUC ≥ 0,90) – [docs/plan_etap_1.md](docs/plan_etap_1.md), rozdz. 8
- [ ] **Etap 1, scenariusz S1 (biały szum)**: generator + 3 własne anomalie zbiorowe (w każdym szeregu wszystkie 3), ocena, 6 metod, eksperyment
- [ ] Etap 1: scenariusze S2 (AR(1)) i S3 (biały szum + sinus), porównanie scenariuszy
- [ ] **Generator szeregów syntetycznych** (AR/ARMA, trend, sezonowość, szum, GARCH-podobna zmienność) – `src/symulacje/`
- [ ] **Wstrzykiwanie anomalii** z etykietami: punktowe, zmiana poziomu, zmiana wariancji, zaburzenie okresowości
      – sterowane parametry: amplituda, długość, poziom szumu
- [ ] Implementacja metod (statystyczne → fragmentowe → ML) we wspólnym interfejsie `fit / score / predict` – `src/metody/`
- [ ] Moduł ewaluacji (metryki z p. 1) i pętla eksperymentów – `src/ewaluacja/`
- [ ] Analiza rzeczywista: SPY i BTCUSDT (zwroty, wolumen)
- [ ] Wyniki, wykresy, opis w pracy

## 5. Uwagi porządkowe

- Pliki `.parquet` w repozytorium: SPY ~24 MB, BTCUSDT ~103 MB (surowy 51,6 MB + clean 51,6 MB) – razem ok. 127 MB.
  Każdy plik mieści się w limicie GitHuba (100 MB na plik), ale repo rośnie z każdą nową wersją danych –
  przy kolejnych aktualizacjach rozważyć Git LFS albo trzymanie danych poza repo.
- `luka_przed` nie widzi braków na końcu sesji (SPY 2019-08-12 15:35–15:55) – te są w `dzien_niepelny` i w CSV.
- Brakuje w `requirements.txt` pakietów do dalszych etapów: `numpy`, `scipy`, `statsmodels`, `scikit-learn`
  (+ ewentualnie `stumpy` do matrix profile).

### Historia porządków

- 2026-10-13: usunięto puste repo `.git` z folderu nadrzędnego `Projekt Inż/`; usunięto duplikat `fetch_alpaca_data.py`
  (zostaje `download_data.py`); `README.md` przekonwertowano z UTF-16 na UTF-8; czyszczenie oparto o kalendarz NYSE;
  `download_data.py` wczytuje klucze z `.env` i zapisuje ścieżki względem katalogu skryptu.
- 2026-10-13: reorganizacja folderów (`src/`, `dane/<zbiór>/`, `wyniki/`), wspólny `src/config.py`;
  dodano drugi zbiór: BTCUSDT 5-min z Binance (pobieranie + czyszczenie).
- 2026-10-13: dodano trzeci zbiór: NYC Taxi z Numenta Anomaly Benchmark (z etykietami anomalii).

