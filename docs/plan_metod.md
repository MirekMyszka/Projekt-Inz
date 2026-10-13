# Plan działania: anomalie w danych i dobór metod detekcji

> **Aktualny etap pracy:** [plan_etap_1.md](plan_etap_1.md) – ustalenia z konsultacji 2026-10-13
> (3 scenariusze, w każdym 3 własne anomalie zbiorowe, 6 metod). Ten dokument to szersze tło i plan na dalsze etapy.

> Dokument roboczy. Liczby i daty policzone na danych clean z 2026-10-13 skryptem
> `python -m src.eksploracja.przeglad_anomalii` (można odtworzyć w każdej chwili).
> Zwroty = 5-min log-zwroty między sąsiednimi świecami (bez luk, w SPY bez przejść przez noc).

## Spis treści

1. [Jak dobierać metody – logika](#1-jak-dobierać-metody--logika)
2. [Właściwości szeregów, które ograniczają wybór metod](#2-właściwości-szeregów-które-ograniczają-wybór-metod)
3. [Rodzaje anomalii w danych – katalog z okresami](#3-rodzaje-anomalii-w-danych--katalog-z-okresami)
4. [Dobór metod do rodzajów anomalii](#4-dobór-metod-do-rodzajów-anomalii)
5. [Jak oceniamy metody](#5-jak-oceniamy-metody)
6. [Plan działania krok po kroku](#6-plan-działania-krok-po-kroku)
7. [Ryzyka i decyzje do podjęcia](#7-ryzyka-i-decyzje-do-podjęcia)
8. [Zgodność z założeniami projektu](#8-zgodność-z-założeniami-projektu)

---

## 1. Jak dobierać metody – logika

Nie ma jednej metody, która wykryje wszystko – każda metoda zakłada, jak wygląda „normalne” zachowanie,
i wykrywa odstępstwa tylko od tego założenia. Dlatego dobór idzie w tej kolejności:

1. **Jakie anomalie występują?** – nazwać typy i znaleźć przykłady w danych (rozdział 3).
2. **Jakie właściwości ma szereg?** – ciężkie ogony, skupianie zmienności, sezonowość dobowa, rozmiar (rozdział 2).
   Właściwości decydują, jak dane trzeba przygotować i które założenia metod są złamane.
3. **Dopasowanie typ ↔ metoda** – do każdego typu anomalii co najmniej jedna metoda z każdej z trzech grup
   z tematu pracy: **statystyczne**, **oparte na fragmentach szeregu**, **uczenie maszynowe** (rozdział 4).
4. **Weryfikacja na symulacjach** – tam, gdzie znamy położenie anomalii, mierzymy skuteczność i fałszywe alarmy
   w zależności od typu, amplitudy, długości anomalii i poziomu szumu (rozdział 5).
5. **Sprawdzenie na danych rzeczywistych** – czy metody wskazują znane zdarzenia z katalogu i ile dają alarmów.

Kluczowa zasada: metody **nie dostają ceny**, tylko przekształcony szereg (zwroty, zmienność, wolumen po usunięciu
sezonowości). Cena jest niestacjonarna (trend), więc każda prosta metoda „wykryłaby” po prostu hossę lub bessę.

---

## 2. Właściwości szeregów, które ograniczają wybór metod

| Właściwość | SPY | BTCUSDT | Konsekwencja dla metod |
|---|---|---|---|
| Liczba świec | 209 908 | 957 865 | metody O(n²) (np. Matrix Profile, kNN, OC-SVM) tylko na fragmentach lub danych zagregowanych |
| Typowa dzienna zmienność (mediana RV) | 0,54% | 2,58% | BTC ~5× bardziej zmienny – progi nie mogą być wspólne |
| Kurtoza zwrotów 5-min | 48 | 108 | bardzo ciężkie ogony – **nie zakładać rozkładu normalnego**; średnia/odch. std → mediana/MAD |
| ACF \|r\| (opóźnienie 1) | 0,39 | 0,42 | silne **skupianie zmienności** – potrzebna normalizacja lokalną zmiennością (krocząca, EWMA lub GARCH) |
| ACF r (opóźnienie 1) | −0,01 | −0,05 | same zwroty prawie nieskorelowane – modele średniej (ARIMA) niewiele wyjaśnią, ważniejsza wariancja |
| Sezonowość dobowa | profil „U”: początek sesji 1,75×, koniec 1,66× zmienności środka dnia | szczyt 13–16 UTC (otwarcie USA, do 1,25×), minimum 3–7 UTC (0,84×) | **usunąć sezonowość** przed detekcją, inaczej każde otwarcie sesji to „anomalia” |
| Sezonowość tygodniowa | – (tylko dni robocze) | weekend 0,80–0,86× zmienności dni roboczych | w BTC profil zależny od dnia tygodnia |
| Przerwy | noc, weekendy, święta; 14 brakujących świec | 33 przerwy techniczne (1 703 świece) | metody okienkowe nie mogą „przeskakiwać” przez luki – okna tylko z ciągłych fragmentów |
| Zmiany reżimu | 2017 bardzo spokojny, 2020-03 i 2022 burzliwe | 2017–2018 bańka, 2023 bardzo spokojny | progi globalne vs lokalne dają różne wyniki – porównać oba |

Wspólne przygotowanie danych (moduł `src/przygotowanie` lub w `src/metody/wspolne.py`):

1. log-zwroty `r_t` (bez przejść przez luki i przez noc),
2. **usunięcie sezonowości**: `r_t / s(pora dnia)`, gdzie `s` = mediana |r| dla danej pory dnia (BTC: pora dnia × dzień tygodnia),
3. **normalizacja zmiennością lokalną**: `z_t = r_t / σ_t`, gdzie `σ_t` z kroczącej MAD, EWMA lub GARCH(1,1),
4. szeregi pomocnicze: |r| lub r² (zmienność), log-wolumen po usunięciu sezonowości, zakres świecy `log(h/l)`.

---

## 3. Rodzaje anomalii w danych – katalog z okresami

Typy pokrywają się z tymi z tematu pracy: pojedyncze obserwacje, zmiana poziomu, zmiana zmienności,
zaburzenie struktury czasowej (okresowości). Dodatkowo wyróżniamy anomalie kontekstowe (wolumen, pora dnia)
i anomalie techniczne (wady danych).

### A. Anomalie punktowe (pojedyncze skrajne obserwacje)

Pojedyncza świeca ze zwrotem wielokrotnie większym niż typowy. Często „flash crash”: gwałtowny ruch i częściowy powrót.
Skala: w SPY 576 zwrotów ma |robust z| > 10, w BTC 5 118.

| Data (czas lokalny zbioru) | Zbiór | Zwrot 5-min | Co to było |
|---|---|---|---|
| 2025-04-07 10:10 i 10:20 ET | SPY | +3,81%, potem −2,93% | fałszywa informacja o wstrzymaniu ceł – skok i powrót w 10 min (największy zwrot w całym SPY, z = 72) |
| 2025-04-09 13:15–13:20 ET | SPY | +2,05%, +2,07% | ogłoszenie 90-dniowego zawieszenia ceł |
| 2020-03-12 09:55 ET | SPY | +2,75% | COVID, dzień ze wstrzymaniem handlu |
| 2019-08-01 13:25 ET | SPY | −0,75% (z lokalny −19,5) | zapowiedź nowych ceł na Chiny – mały ruch, ale skrajny jak na spokojny dzień |
| 2020-03-13 02:35 UTC | BTC | +16,91% | odbicie po krachu COVID (największy zwrot w BTC, z = 149) |
| 2023-08-17 21:40 UTC | BTC | −9,07% (z lokalny −115) | flash crash w najspokojniejszym okresie – **najbardziej „czysty” przykład anomalii punktowej** |
| 2021-09-07 15:05 UTC | BTC | −10,41% | dzień wprowadzenia BTC jako waluty w Salwadorze |
| 2023-10-16 13:25 UTC | BTC | +4,61% | fałszywa wiadomość o zatwierdzeniu ETF na BTC |
| 2025-10-10 21:15 UTC | BTC | −7,62% | kaskada likwidacji po zapowiedzi ceł (SPY tego dnia o 10:55 ET: −0,55%, z lokalny −16,5) |

**Wniosek:** ten sam zwrot jest anomalią albo nie, zależnie od otoczenia. Globalny próg znajdzie głównie marzec 2020
i kwiecień 2025, lokalny – nagłe ruchy w spokojnych okresach. W pracy porównać oba podejścia.

### B. Anomalie kolektywne (krótka sekwencja nietypowych świec)

Kilka–kilkanaście świec, z których pojedyncze mogą nie być skrajne, ale razem tworzą nietypowy fragment.

| Okres | Zbiór | Opis |
|---|---|---|
| 2021-05-19 12:50–13:15 UTC | BTC | −7,63%, −10,20%, +8,99% w 25 min – krach i odbicie |
| 2025-04-07 10:05–10:25 ET | SPY | seria: skok, utrzymanie, spadek; wolumen do 5× wyższy niż w świecach tuż przed |
| 2019-04-02 04:30–05:15 UTC | BTC | seria wzrostów +3,45%, +3,16%, +3,11% w 45 min (cały dzień +17,2%) |
| 2018-10-15 06:00–06:50 UTC | BTC | utrata parytetu USDT do USD: BTC/USDT +4,4%, potem +6,7% w 5 min, wolumen dzienny 5× – anomalia **notowania**, nie samego BTC |

### C. Zmiana poziomu (skok/przesunięcie ceny, które się utrzymuje)

W cenie – nagłe przesunięcie poziomu; w zwrotach – pojedynczy duży zwrot bez powrotu.
Wykrywamy na log-cenie (lokalnie) albo na skumulowanych zwrotach.

| Okres | Zbiór | Opis |
|---|---|---|
| luki overnight 2020-03-09, 12, 16, 18 | SPY | otwarcie niżej o 7–11,6% (2020-03-16: −11,6%) |
| 2024-08-05 | SPY | luka overnight −4,1% (rozpad carry trade na jenie) |
| 2020-03-12 | BTC | −39,5% w ciągu doby (UTC), 7 960 → 4 800 USDT |
| 2017-09-14, 2018-01-16 | BTC | −19,2%, −19,5% w dobę (zakaz giełd w Chinach; krach po bańce) |
| 2019-04-02, 2019-10-25, 2021-02-08 | BTC | +17,2%, +16,8%, +19,5% w dobę |
| 2022-06-13 | BTC | −15,4% w dobę (upadek Celsius) |
| 2025-04-09 13:15–13:30 | SPY | trwałe przesunięcie poziomu o ok. +6% w 15 min |

### D. Zmiana zmienności (reżimy wariancji)

Okresy, w których zmienność jest kilkukrotnie większa (lub mniejsza) niż zwykle – trwają od dni do miesięcy.

| Okres | Zbiór | Mediana dziennej RV | Względem typowej |
|---|---|---|---|
| **2020-02-24 – 2020-04** (COVID) | SPY | 3,48% w marcu 2020 (dzienna RV: 0,25% 19.02 → 1,03% 24.02 → 3,86% 28.02) | **6,4×** (SPY −33,8% od 19.02 do 23.03) |
| 2025-04 (cła) | SPY | 1,12%; 2025-04-07 RV 6,42% (rekord) | 2× miesiąc, 12× dzień |
| 2018-12, 2022 (05, 06, 09, 10) | SPY | 1,14–1,50% | 2–2,8× |
| **2017 – reżim nienaturalnego spokoju** | SPY | 0,21–0,26% (najspokojniejsze miesiące: 05, 07, 09–11.2017) | 0,4× – anomalia „w dół” |
| **2017-08 – 2018-03** (bańka i krach) | BTC | 5,2–9,5% | 2–3,7× |
| 2020-03-12/13 | BTC | RV dzienna 22–33% | 9–13× |
| 2021-01, 2021-05 | BTC | 5,9–6,5% | 2,3–2,5× |
| **2023-07/08, 2022-12, 2018-10** | BTC | 1,2–1,3% | 0,5× – spokój, na którego tle flash crash z 2023-08-17 jest skrajny |

### E. Zaburzenia struktury czasowej i okresowości

Szereg ma regularny wzór dobowy (SPY: „U”, BTC: szczyt przy otwarciu USA, słabsze weekendy).
Anomalią jest dzień lub fragment, który ten wzór łamie, albo regularne zdarzenie, które wzór zmienia.

| Zdarzenie | Zbiór | Opis |
|---|---|---|
| posiedzenia Fed (FOMC), 14:00 ET | SPY | skok zmienności w środku dnia zamiast w „dołku” profilu U: 2021-06-16 14:00 (z lok. −20), 2022-09-21 14:00 (−1,64%), 2023-12-13 14:00, 2024-12-18 15:55, 2025-10-29 14:35 |
| wygasanie opcji (3. piątek miesiąca, szczególnie marzec/czerwiec/wrzesień/grudzień) | SPY | skoki wolumenu w środku sesji: 2018-06-15, 2018-09-21, 2019-12-20, 2024-12-20, 2026-09-18 (kwartalne), 2021-01-15, 2024-11-15 (miesięczne) – **anomalie przewidywalne (kontekstowe)** |
| skrócone sesje | SPY | 21 dni po 42 świece, wolumen np. 0,33–0,44× normy (2018-11-23, 2019-12-24) – oznaczone w danych (`skrocona_sesja`) |
| wstrzymania handlu 2020-03 | SPY | przerwa w środku sesji (`wstrzymanie_handlu`) |
| weekendy | BTC | zmienność 0,80–0,86× dni roboczych; dni o najniższym wolumenie (0,19–0,28× normy) to soboty z lat 2024–2026 |
| zmiana struktury rynku | BTC | cykl dobowy coraz silniejszy: stosunek najaktywniejszej do najspokojniejszej godziny 1,3–1,5× (2017–2021) → 2,0–2,1× (2023–2026), szczyt o 14–16 UTC – profil „normalności” zmienia się w czasie |

### F. Anomalie wolumenu (kontekstowe)

Wolumen wielokrotnie większy niż zwykle **o tej porze dnia** – często bez skrajnego ruchu ceny.

| Data | Zbiór | Wolumen dzienny względem 60-dniowej mediany |
|---|---|---|
| 2018-02-05/06 („Volmageddon” – krach produktów na zmienność) | SPY | 3,4× i **4,2×** (rekord) |
| 2016-06-24 (Brexit) | SPY | 3,7× |
| 2024-08-05 | SPY / BTC | 3,1× / **5,9×** (rekord BTC) – to samo zdarzenie w obu zbiorach |
| 2025-04-04, 04-07, 04-09 | SPY | 3,0×, 3,8×, 3,4× |
| 2020-03-13, 2021-05-19 | BTC | 5,2×, 4,1× |
| 2026-02-05/06 | BTC | 5,0×, 4,3× |

### G. Anomalie techniczne (wady danych) – nie są celem, ale pomagają sprawdzić metody

Już oznaczone w danych: luki (`luka_przed`, `dzien_niepelny`), świece bez transakcji (`zerowy_wolumen`),
przesunięte znaczniki czasu (`czas_skorygowany`), wstrzymania handlu. Metody **nie powinny** alarmować na samych
lukach – jeśli alarmują, to znak źle przygotowanych danych.

### Zdarzenia wspólne dla obu zbiorów

Dobre do porównania „ten sam szok, inny szereg”: **marzec 2020**, **2024-08-05**, **kwiecień 2025**, **2025-10-10**.

---

## 4. Dobór metod do rodzajów anomalii

### 4.1 Macierz: typ anomalii × metoda

✔ = metoda do tego stworzona, ◐ = działa częściowo / po przystosowaniu, – = nie nadaje się

| Metoda | Grupa | A. punktowe | B. kolektywne | C. zmiana poziomu | D. zmiana zmienności | E. struktura / okresowość | F. wolumen |
|---|---|---|---|---|---|---|---|
| **S1. Robust z-score / filtr Hampela** (mediana, MAD, okno kroczące) | statystyczna | ✔ | ◐ | ◐ | – | – | ✔ |
| **S2. Karty kontrolne CUSUM i EWMA** | statystyczna | – | ◐ | ✔ | ✔ (na r² lub \|r\|) | – | ◐ |
| **S3. Reszty modelu: STL + GARCH(1,1)** | statystyczna | ✔ | ◐ | ◐ | ✔ | ✔ (STL usuwa okresowość) | ◐ |
| **S4. Wykrywanie punktów zmiany (PELT, `ruptures`)** | statystyczna | – | – | ✔ | ✔ | ◐ | ◐ |
| **F1. Matrix Profile – „discords”** (`stumpy`) | fragmenty | ◐ | ✔ | ◐ | ◐ | ✔ | ◐ |
| **F2. Cechy okien + odległość kNN** | fragmenty | ◐ | ✔ | ◐ | ✔ | ✔ | ✔ |
| **M1. Isolation Forest** (na cechach okien) | ML | ✔ | ✔ | ◐ | ✔ | ◐ | ✔ |
| **M2. Local Outlier Factor / One-Class SVM** | ML | ✔ | ✔ | ◐ | ◐ | ◐ | ✔ |
| **M3. Autoenkoder (np. LSTM / sieć splotowa)** – opcjonalnie | ML (głębokie) | ◐ | ✔ | ◐ | ◐ | ✔ | ◐ |

### 4.2 Krótki opis i uzasadnienie każdej metody

**S1. Robust z-score / filtr Hampela** – dla każdej świecy `z = (x − mediana) / (1,4826 · MAD)` w oknie kroczącym;
alarm, gdy |z| > próg. Mediana i MAD zamiast średniej i odchylenia, bo ogony są bardzo ciężkie (kurtoza 48 / 108).
Najprostszy punkt odniesienia (baseline) – każda inna metoda musi go pokonać.
Na danych: zwroty po usunięciu sezonowości (A), log-wolumen po usunięciu sezonowości (F).

**S2. CUSUM / EWMA** – sumują małe odchylenia w czasie, więc wykrywają **trwałe** zmiany (poziomu lub wariancji),
których pojedyncza świeca nie zdradza. CUSUM na zwrotach → zmiana poziomu/dryfu; CUSUM/EWMA na r² → początek reżimu
wysokiej zmienności (np. 2020-02-24). Daje też **opóźnienie wykrycia** – dobra metryka do porównań.

**S3. STL + GARCH** – STL rozkłada szereg na trend, sezonowość (okres = 78 lub 288 świec) i resztę;
GARCH(1,1) modeluje skupianie zmienności. Anomalia = reszta standaryzowana zmiennością warunkową GARCH > próg.
Klasyczne podejście „model + reszty”, dobrze dopasowane do szeregów finansowych (`statsmodels`, `arch`).

**S4. PELT (wykrywanie punktów zmiany)** – dzieli szereg na odcinki o stałych parametrach (średnia i/lub wariancja),
wskazując momenty zmian. Najlepsze do D (reżimy: start i koniec marca 2020, spokój 2017) i C.
Na pełnych danych 5-min wolne – stosować na danych dziennych/godzinowych (RV) albo na wybranych okresach.

**F1. Matrix Profile (discords)** – dla każdego okna długości m liczy odległość do najbardziej podobnego okna
w całym szeregu; okno, które nie ma „bliźniaka”, jest anomalią (discord). Nie wymaga modelu ani rozkładu – to wprost
metoda „oparta na podobieństwie fragmentów” z tematu pracy. Długości okien: 12 (1 h), 78 (sesja SPY), 288 (doba BTC).
Uwaga: koszt ~O(n²) – stosować na |r| zagregowanych lub na wybranych latach; `stumpy` liczy to wydajnie.

**F2. Cechy okien + kNN** – każde okno opisujemy wektorem cech (średnia, odch. std, skośność, kurtoza, maks. |r|,
energia widmowa w okresie dobowym, wolumen względny, zakres), anomalia = duża średnia odległość do k najbliższych okien.
Metoda „oparta na charakterystykach fragmentów” – interpretowalna (widać, która cecha odstaje).

**M1. Isolation Forest** – losowe drzewa izolują punkty; anomalie izolują się szybciej. Działa na wielu cechach naraz
(zwrot + wolumen + zakres), szybki na dużych danych, mało założeń. Główny kandydat ML.

**M2. LOF / One-Class SVM** – LOF porównuje gęstość punktu z gęstością sąsiadów (anomalie lokalne);
OC-SVM uczy granicy „normalnych” danych. Oba wolne dla setek tysięcy punktów – uczyć na próbie lub na oknach.

**M3. Autoenkoder** – sieć uczy się odtwarzać typowe okna (np. dzienny profil 78/288 świec); duży błąd odtworzenia
= nietypowy fragment (B, E). Opcjonalny – większy nakład pracy; dodać, jeśli zostanie czas.

### 4.3 Proponowany zestaw do pracy

Minimum, które pokrywa wszystkie typy i wszystkie trzy grupy z tematu:

| Grupa | Metody w zestawie podstawowym | Opcjonalnie |
|---|---|---|
| statystyczne | S1 (Hampel / robust z), S2 (CUSUM + EWMA), S3 (STL + GARCH) | S4 (PELT) |
| fragmenty szeregu | F1 (Matrix Profile), F2 (cechy okien + kNN) | – |
| uczenie maszynowe | M1 (Isolation Forest), M2 (LOF) | M3 (autoenkoder), OC-SVM |

Wszystkie metody we **wspólnym interfejsie**: `fit(train)`, `score(x) → wynik anomalii dla każdej świecy`,
`predict(x, prog) → 0/1`. Dzięki temu ewaluacja jest jedna dla wszystkich.

Oczekiwania do sprawdzenia (hipotezy pracy):
- A (punktowe): S1 i S3 najskuteczniejsze i najtańsze; ML nie daje dużej przewagi.
- B, E (kolektywne, struktura): przewaga F1, F2, M1 – metody punktowe ich nie widzą.
- C, D (poziom, wariancja): przewaga S2 i S4 (szybsze wykrycie, mniej alarmów); metody punktowe dają serie alarmów.
- Bez usunięcia sezonowości i normalizacji zmiennością wszystkie metody dają dużo fałszywych alarmów
  (otwarcie sesji SPY, godziny 13–16 UTC w BTC, marzec 2020) – warto to pokazać w pracy jako eksperyment.

---

## 5. Jak oceniamy metody

### 5.1 Symulacje (główna część – znamy położenie anomalii)

**Szeregi tła o różnych właściwościach** (`src/symulacje/`). Temat pracy wymaga zbadania, jak skuteczność zależy
od **właściwości szeregu**, dlatego rodzaj szeregu tła jest osobnym czynnikiem eksperymentu, a nie tylko tłem:

| Szereg tła | Właściwość, którą testuje | Parametry |
|---|---|---|
| T1. biały szum gaussowski | przypadek wzorcowy (wszystkie założenia spełnione) | σ |
| T2. AR(1) | autokorelacja (pamięć szeregu) | φ ∈ {0,3; 0,7; 0,95} |
| T3. trend + szum | niestacjonarność średniej | nachylenie trendu |
| T4. sezonowość (sinusoida lub profil dobowy) + szum | okresowość | okres P, amplituda sezonowości |
| T5. GARCH(1,1) z szumem t-Studenta | ciężkie ogony + skupianie zmienności (jak rynki finansowe) | parametry dopasowane do SPY i do BTC |
| T6. trend + sezonowość + AR + GARCH | wszystko naraz (najbliżej danych rzeczywistych) | jak wyżej |

**Poziom szumu** – osobny czynnik: stosunek sygnału do szumu (amplituda anomalii / σ szumu) oraz, dla T3–T4,
stosunek σ szumu do amplitudy trendu/sezonowości.

**Wstrzykiwane anomalie z etykietami** (każda z parametrami: amplituda, długość, położenie):

| Typ | Jak wstrzykujemy | Parametry | Wzorzec z danych |
|---|---|---|---|
| punktowa | +k·σ w jednej świecy | k ∈ {3, 5, 8, 12, 20} | 2023-08-17 BTC |
| flash crash | skok −k·σ i powrót w 1–3 świecach | k, czas powrotu | 2025-04-07 SPY |
| krótkotrwała zmiana poziomu | przesunięcie poziomu o Δ przez L świec, potem powrót (wariant dodatkowy: trwałe przesunięcie) | Δ ∈ {1, 2, 3, 5}·σ, L ∈ {6, 12, 78, 390} | 2025-04-07 SPY (skok i powrót), 2020-03-16 SPY (luka), 2019-04-02 BTC |
| krótkotrwała zmiana zmienności | σ × c przez L świec | c ∈ {0,3; 0,5; 1,5; 2; 3; 5}, L ∈ {12, 78, 390, 1560} | marzec 2020, 2017 (c < 1) |
| zaburzenie okresowości | w fragmencie długości L: zanik sezonowości, przesunięcie fazy, zmiana okresu lub amplitudy | L, przesunięcie fazy, nowy okres/amplituda | dni FOMC, dni skrócone, weekend BTC |
| zmiana struktury zależności | w fragmencie długości L zmiana współczynnika AR (np. 0,7 → 0 lub 0,7 → 0,95) przy tej samej wariancji | L, nowe φ | – (test czysto symulacyjny; tylko dla T2 i T6) |
| kolektywna | sekwencja L świec o nietypowym kształcie (np. trend + wysoka σ) | L, amplituda | 2021-05-19 BTC |
| wolumen | wolumen × c bez zmiany ceny | c ∈ {2, 3, 5} | dni wygasania opcji, 2018-02-06 |

**Plan eksperymentu:** siatka parametrów (amplituda, długość, poziom szumu) × typ anomalii × szereg tła T1–T6 × metoda, **co najmniej 50–100 powtórzeń**
(różne ziarna losowe) → średnia i przedział ufności metryk.
Dodatkowo **szeregi bez anomalii** – do pomiaru odsetka fałszywych alarmów.

**Zakres:** pełna siatka (8 typów × 6 szeregów × ~5 amplitud × ~4 długości × 3 poziomy szumu × 7 metod × 100 powtórzeń)
to setki tysięcy przebiegów. Dlatego: krótkie szeregi syntetyczne (np. 5 000 punktów), pełna siatka tylko dla
metod szybkich, a dla wolnych (Matrix Profile, LOF, OC-SVM) siatka zredukowana.

**Dane półsyntetyczne:** wstrzykiwanie tych samych anomalii w **prawdziwe spokojne okresy**
(SPY 2017, BTC 2023-07/08) – realistyczne tło, a położenie anomalii nadal znane.

### 5.2 Metryki

- **precyzja, czułość, F1** – na poziomie punktów oraz **na poziomie zdarzeń** (zdarzenie wykryte, jeśli choć jeden
  alarm w jego oknie ± tolerancja),
- **odsetek fałszywych alarmów** (FPR) i **liczba alarmów na dzień** na szeregu bez anomalii,
- **opóźnienie wykrycia** (w świecach) – dla zmian poziomu i wariancji,
- **PR-AUC** (lepsza niż ROC-AUC przy bardzo rzadkich anomaliach) – ocena niezależna od progu,
- **czas obliczeń** – istotny przy 210 tys. / 958 tys. świec.

**Próg alarmu:** ustalany tak samo dla wszystkich metod – np. na stały poziom fałszywych alarmów
(1 alarm na dzień na danych bez anomalii) albo jako kwantyl wyniku na okresie uczącym. Inaczej porównanie jest nieuczciwe.

**Wyniki:** wykresy skuteczności (F1, czułość, FPR) w funkcji amplitudy, długości anomalii, poziomu szumu
**i rodzaju szeregu tła (T1–T6)** – osobno dla każdego typu anomalii. To bezpośrednio odpowiada na pytanie z tematu pracy.

### 5.3 Dane rzeczywiste – część uzupełniająca

Zgodnie z tematem pracy weryfikacja opiera się **przede wszystkim na symulacjach**; dane rzeczywiste służą do
pokazania, jak metody zachowują się w praktyce (temat: „w miarę dostępności”). Nie należy na nich budować głównych wniosków.


**NYC Taxi (NAB) – dane rzeczywiste z etykietami.** 5 oznaczonych okien (maraton, Święto Dziękczynienia, Boże Narodzenie,
Nowy Rok, śnieżyca) – to głównie **zaburzenia okresowości** (typ E) i krótkotrwałe zmiany poziomu (typ C).
Tu liczymy te same metryki co w symulacjach (na poziomie zdarzeń: czy w oknie był alarm; fałszywe alarmy poza oknami),
z dwoma zastrzeżeniami (szczegóły: `docs/data_pipeline.md`, rozdz. 3): artefakt zmiany czasu w oknie „maraton”
oraz nieoznaczone 4–6 lipca 2014.

Na SPY i BTC nie wiemy, gdzie są „wszystkie” anomalie. Dlatego:
- **lista znanych zdarzeń z rozdziału 3** jako etykiety częściowe – miara: ile zdarzeń metoda wskazuje wśród
  N najwyższych wyników (*recall@N*),
- **liczba alarmów na miesiąc** – czy metoda nie „wybucha” alarmami w burzliwych okresach,
- **zgodność metod** – które zdarzenia wykrywa większość metod, a które tylko jedna,
- podział czasowy: kalibracja na wcześniejszych latach, test na późniejszych (np. SPY: 2016–2021 / 2022–2026),
- **analiza jakościowa** wybranych przypadków na wykresach (marzec 2020, 2024-08-05, kwiecień 2025, 2025-10-10).

---

## 6. Plan działania krok po kroku

| # | Krok | Wynik | Folder |
|---|---|---|---|
| 1 | ✅ Dane: pobranie, czyszczenie, przegląd anomalii | `dane/clean/`, ten dokument | `src/pobieranie`, `src/czyszczenie`, `src/eksploracja` |
| 2 | Przygotowanie szeregów: log-zwroty bez luk, usunięcie sezonowości, normalizacja zmiennością, cechy okien | funkcje + zapisane szeregi pochodne | `src/metody/przygotowanie.py` |
| 3 | Generator szeregów tła T1–T6 (szum, AR, trend, sezonowość, GARCH + t-Student) + dopasowanie parametrów do SPY/BTC | `src/symulacje/generator.py` + test wizualny | `src/symulacje` |
| 4 | Wstrzykiwanie anomalii 8 typów z etykietami | `src/symulacje/anomalie.py` | `src/symulacje` |
| 5 | Wspólny interfejs metod + S1 (baseline) | `fit/score/predict`, pierwszy wynik | `src/metody` |
| 6 | Moduł ewaluacji: metryki punktowe i zdarzeniowe, FPR, opóźnienie, PR-AUC, dobór progu | `src/ewaluacja/metryki.py` | `src/ewaluacja` |
| 7 | Pierwszy pełny eksperyment na małej siatce (S1 × 2 typy anomalii) – sprawdzenie całego łańcucha | pierwszy wykres F1 vs amplituda | `wyniki/` |
| 8 | Pozostałe metody: S2, S3, F1, F2, M1, M2 (+ opcjonalnie S4, M3) | kolejne klasy w `src/metody` | `src/metody` |
| 9 | Pełna siatka eksperymentów symulacyjnych + półsyntetycznych | tabele i wykresy wyników | `src/ewaluacja`, `wyniki/` |
| 10 | Dane rzeczywiste: recall@N na katalogu zdarzeń, alarmy na miesiąc, studia przypadków | wykresy z alarmami na tle ceny | `wyniki/` |
| 11 | Wnioski: która metoda do którego typu anomalii, zależność od intensywności, czasu trwania, szumu i szeregu | rozdział pracy | – |

Kolejność ma znaczenie: kroki 3–7 budują **łańcuch end-to-end z jedną prostą metodą**, zanim powstaną kolejne metody.
Każdą nową metodę od razu widać w tych samych wykresach.

Pakiety do dopisania w `requirements.txt` (w miarę postępu):
`numpy`, `scipy`, `statsmodels` (STL, ARIMA), `arch` (GARCH), `scikit-learn` (Isolation Forest, LOF, OC-SVM),
`ruptures` (PELT), `stumpy` (Matrix Profile); opcjonalnie `torch` (autoenkoder).

---

## 7. Ryzyka i decyzje do podjęcia

- **Rozmiar danych** – Matrix Profile, kNN, LOF, OC-SVM na 958 tys. punktów są wolne. Decyzja: agregacja
  (np. 15 min / 1 h), próbkowanie albo analiza wybranych lat.
- **Częstotliwość danych** – 5 min to dużo szumu mikrostruktury. Rozważyć równoległe wersje 15 min / 1 h / 1 dzień
  (zmiany reżimu D lepiej widać na danych dziennych).
- **Co jest „prawdziwą” anomalią w danych rzeczywistych** – katalog z rozdziału 3 jest subiektywny; w pracy
  traktować go jako etykiety częściowe i jasno to opisać.
- **Anomalie przewidywalne** (FOMC, wygasanie opcji, skrócone sesje) – decyzja: wykrywać je (test metody)
  czy traktować jako normalne (kontekst znany z kalendarza). Proponuję raportować oba warianty.
- **BTCUSDT 2018-10-15** – anomalia kursu USDT, nie BTC; opisać jako przykład anomalii wynikającej z konstrukcji szeregu.
- **Uczciwe porównanie** – wszystkie metody na tych samych danych wejściowych, z tym samym sposobem doboru progu.

---

## 8. Zgodność z założeniami projektu

| Założenie z tematu pracy | Gdzie w planie |
|---|---|
| opracowanie i porównanie metod automatycznego wykrywania anomalii | rozdz. 4 (dobór, wspólny interfejs), rozdz. 5 (wspólna ewaluacja) |
| anomalie: pojedyncze nietypowe obserwacje | typ A; symulacje: punktowa, flash crash |
| anomalie: krótkotrwałe zmiany poziomu | typ C; symulacje: przesunięcie przez L świec z powrotem |
| anomalie: krótkotrwałe zmiany zmienności | typ D; symulacje: σ × c przez L świec |
| anomalie: zaburzenia struktury czasowej, np. okresowości | typ E; symulacje: zaburzenie okresowości + zmiana struktury zależności (AR) |
| szeregi czasowe o różnych właściwościach | szeregi tła T1–T6 (rozdz. 5.1) + dwa szeregi rzeczywiste o różnych właściwościach (rozdz. 2) |
| klasyczne podejścia statystyczne | S1–S4 |
| metody wykorzystujące podobieństwo lub charakterystyki fragmentów | F1 (podobieństwo – Matrix Profile), F2 (charakterystyki okien) |
| wybrane algorytmy uczenia maszynowego | M1–M3 |
| zależność skuteczności od rodzaju, intensywności i czasu trwania anomalii | czynniki eksperymentu: typ, amplituda, długość L |
| zależność od właściwości szeregu i poziomu szumu | czynniki eksperymentu: szereg tła T1–T6, poziom szumu |
| weryfikacja przede wszystkim symulacyjna, znane położenie i rodzaj anomalii | rozdz. 5.1 (+ dane półsyntetyczne) |
| ocena skuteczności detekcji i częstości fałszywych alarmów | rozdz. 5.2 (czułość, precyzja, F1, FPR, alarmy na dzień, opóźnienie) |
| w miarę dostępności – analiza rzeczywistych szeregów | rozdz. 3 (katalog anomalii SPY i BTCUSDT), rozdz. 5.3 – NYC Taxi z etykietami NAB (ocena ilościowa) + SPY/BTC (ocena jakościowa) |

Elementy **ponad** założenia (do ewentualnego ograniczenia, jeśli zabraknie czasu): typ B (kolektywne) i F (wolumen)
jako osobne typy, metoda S4 (PELT), M3 (autoenkoder), dane półsyntetyczne.

