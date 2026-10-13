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
- Lokalnie: `Projekt Inż/Projekt-Inz/` – **to jest właściwe repo** (remote `origin` → GitHub).

```
Projekt-Inz/
├── .env.example            # wzór pliku z kluczami Alpaca (prawdziwy .env jest w .gitignore)
├── requirements.txt        # requests, pandas, pyarrow, python-dotenv, matplotlib, pandas_market_calendars
├── download_data.py        # pobieranie świec SPY z Alpaca (klucze z pliku .env)
├── prepare_clean_data.py   # czas NY + filtr sesji wg kalendarza NYSE + oznaczanie luk → dane/clean
├── preview_data.py         # statystyki + wykres ceny zamknięcia → plots/spy_5min_close.png
├── dane/
│   ├── surowe/spy_5min_sip_2016_2026_pobrano_2026-10-13.parquet
│   └── clean/
│       ├── spy_5min_session_2016_2026.parquet
│       └── brakujace_swiece.csv
├── plots/spy_5min_close.png
└── docs/data_pipeline.md   # szczegółowy opis pipeline'u danych
```

### Uruchomienie

```bash
pip install -r requirements.txt
copy .env.example .env          # i wpisać własne klucze Alpaca
python download_data.py         # → dane/surowe/...
python prepare_clean_data.py    # → dane/clean/...
python preview_data.py          # → plots/spy_5min_close.png
```

## 3. Dane (zweryfikowane 2026-10-13)

Źródło: Alpaca Market Data API, `/v2/stocks/bars`, symbol **SPY**, interwał **5 min**, feed `sip`, `adjustment=all`.

| | Surowe | Clean (sesja) |
|---|---|---|
| Plik | `dane/surowe/spy_5min_sip_2016_2026_pobrano_2026-10-13.parquet` | `dane/clean/spy_5min_session_2016_2026.parquet` |
| Wierszy | 496 241 | 209 908 |
| Zakres | 2016-01-01 00:00 UTC – 2026-09-30 23:55 UTC | 2016-01-04 09:30 – 2026-09-30 15:55 (America/New_York) |
| Strefa czasowa indeksu | UTC | America/New_York |
| Braki (NaN) / duplikaty indeksu | 0 / 0 | 0 / 0 |

Kolumny z API: `o, h, l, c` – OHLC, `v` – wolumen, `n` – liczba transakcji, `vw` – VWAP. Indeks `t` = **początek** świecy.

Kolumny dodane przy czyszczeniu:

| Kolumna | Typ | Znaczenie |
|---|---|---|
| `skrocona_sesja` | bool | dzień ze skróconą sesją wg kalendarza NYSE (zamknięcie 13:00) |
| `luka_przed` | int | ile oczekiwanych świec brakuje bezpośrednio przed tą świecą |
| `dzien_niepelny` | bool | w tym dniu brakuje co najmniej jednej świecy |
| `wstrzymanie_handlu` | bool | dzień z circuit breakerem: 9, 12, 16, 18 marca 2020 |

### Czyszczenie (`prepare_clean_data.py`)

1. Konwersja UTC → America/New_York.
2. Filtr do rzeczywistych godzin sesji z kalendarza NYSE (`pandas_market_calendars`): świeca jest w sesji, gdy
   `market_open <= t < market_close`. Zwykły dzień → 78 świec (09:30–15:55), skrócona sesja → 42 świece (09:30–12:55).
   Handel przed- i posesyjny (także po 13:00 w dni skrócone) jest odrzucany.
3. Brakujące świece **nie są uzupełniane**, tylko oznaczane (kolumny wyżej) i wypisywane do `dane/clean/brakujace_swiece.csv`.

### Kontrola jakości danych clean (stan na 2026-10-13)

- 2 701 dni sesyjnych = 2 701 dni w kalendarzu NYSE (brak brakujących dni).
- 2 674 dni po 78 świec, 21 skróconych sesji po 42 świece.
- 14 brakujących świec w 6 dniach:
  - **marzec 2020 – wstrzymania handlu (circuit breaker)**: po 2 świece 9, 12, 16 i 18 marca (flaga `wstrzymanie_handlu`),
  - 2019-08-12 – brak 5 ostatnich świec (15:35–15:55), najpewniej luka po stronie dostawcy danych,
  - 2016-02-02 – brak 1 świecy (11:20).
- Spójność OHLC poprawna (brak `h < l`, `c` zawsze w `[l, h]`), brak zerowego wolumenu.
- Log-zwroty 5-min: odch. std ≈ 0,00125; największe ruchy wewnątrz dnia w kwietniu 2025 (−2,9% / +3,8%),
  największe luki overnight w marcu 2020 (do −11,8%).
- Cena zamknięcia: ~170 USD (2016) → ~762 USD (09.2026).

### Uwagi do dalszej analizy

1. **Luki overnight** – zwrot między ostatnią świecą dnia a pierwszą następnego dnia nie jest zwrotem 5-minutowym.
   Liczyć zwroty tylko wewnątrz dnia albo traktować overnight osobno.
2. Cena nie jest stacjonarna – do detekcji anomalii lepiej używać **log-zwrotów** lub reszt po usunięciu
   sezonowości dziennej (kształt „U” zmienności i wolumenu w ciągu sesji).
3. Dni z `luka_przed > 0` – zwrot do świecy po luce obejmuje więcej niż 5 minut.
4. Marzec 2020 i kwiecień 2025 to dobre **rzeczywiste okresy zaburzeń** do części empirycznej.

## 4. Stan prac

- [x] Repozytorium na GitHubie, `.env` z kluczami poza Gitem
- [x] Pobranie danych SPY 5-min 2016–2026 (surowe)
- [x] Wstępne czyszczenie (strefa czasowa, godziny sesji) + kontrola liczby świec
- [x] Podgląd danych i wykres ceny
- [x] Skrócone sesje wg kalendarza NYSE, oznaczenie luk i wstrzymań handlu (marzec 2020)
- [ ] **Generator szeregów syntetycznych** (AR/ARMA, trend, sezonowość, szum, GARCH-podobna zmienność)
- [ ] **Wstrzykiwanie anomalii** z etykietami: punktowe, zmiana poziomu, zmiana wariancji, zaburzenie okresowości
      – sterowane parametry: amplituda, długość, poziom szumu
- [ ] Implementacja metod (statystyczne → fragmentowe → ML) we wspólnym interfejsie `fit / score / predict`
- [ ] Moduł ewaluacji (metryki z p. 1) i pętla eksperymentów (siatka parametrów × wiele powtórzeń)
- [ ] Analiza rzeczywista: SPY (zwroty, wolumen) – np. marzec 2020, kwiecień 2025 jako znane okresy zaburzeń
- [ ] Wyniki, wykresy, opis w pracy

## 5. Uwagi porządkowe

- Zmiany lokalne **nie są jeszcze zacommitowane** – na GitHubie jest tylko `README.md` z pierwszego commita.
- Nazwa pliku surowego jest wpisana na sztywno z datą `2026-10-13` w kilku skryptach – lepiej trzymać ścieżki w jednym miejscu (np. `config.py`).
- Pliki `.parquet` (~24 MB łącznie) – zdecydować, czy trafiają do repo, czy dodać `dane/` do `.gitignore`
  i opisać, jak je odtworzyć skryptami.
- Brakuje w `requirements.txt` pakietów do dalszych etapów: `numpy`, `scipy`, `statsmodels`, `scikit-learn`
  (+ ewentualnie `stumpy` do matrix profile).

### Historia porządków

- 2026-10-13: usunięto puste repo `.git` z folderu nadrzędnego `Projekt Inż/`; usunięto duplikat `fetch_alpaca_data.py`
  (zostaje `download_data.py`); `README.md` przekonwertowano z UTF-16 na UTF-8; czyszczenie oparto o kalendarz NYSE;
  `download_data.py` wczytuje klucze z `.env` i zapisuje ścieżki względem katalogu skryptu.
