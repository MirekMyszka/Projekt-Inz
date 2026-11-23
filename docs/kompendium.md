# Kompendium wiedzy projektu

> Ściąga dla członków grupy: **wszystkie pojęcia i metody, których użyliśmy, w kolejności, w jakiej pojawiały się
> w projekcie** – od danych, przez symulacje, po dane rzeczywiste. Każde hasło: *co to jest*, *jak działa*, *po co nam
> było*. Na końcu: najważniejsze liczby i pytania, które mogą paść na prezentacji, z odpowiedziami.
> Szczegóły i wzory: [metody.md](metody.md); wyniki: raporty PDF w `docs/`. Stan na **2026-11-23**.

## Spis treści

0. [Projekt w jednym akapicie](#0-projekt-w-jednym-akapicie)
1. [Podstawowe pojęcia: szereg czasowy i anomalia](#1-podstawowe-pojęcia-szereg-czasowy-i-anomalia)
2. [Krok 1 – dane](#2-krok-1--dane)
3. [Krok 2 – właściwości danych i przegląd anomalii](#3-krok-2--właściwości-danych-i-przegląd-anomalii)
4. [Krok 3 – symulacje: scenariusze S1–S3](#4-krok-3--symulacje-scenariusze-s1s3)
5. [Krok 4 – metody](#5-krok-4--metody)
6. [Krok 5 – jak oceniamy metody](#6-krok-5--jak-oceniamy-metody)
7. [Krok 6 – wyniki symulacji i porównanie S1–S3](#7-krok-6--wyniki-symulacji-i-porównanie-s1s3)
8. [Krok 7 – dane rzeczywiste](#8-krok-7--dane-rzeczywiste)
9. [Narzędzia i repozytorium](#9-narzędzia-i-repozytorium)
10. [Najważniejsze liczby – ściąga](#10-najważniejsze-liczby--ściąga)
11. [Pytania z dyskusji – z odpowiedziami](#11-pytania-z-dyskusji--z-odpowiedziami)
12. [Słowniczek polsko-angielski](#12-słowniczek-polsko-angielski)

---

## 0. Projekt w jednym akapicie

Temat: **wykrywanie anomalii w szeregach czasowych metodami statystycznymi i uczenia maszynowego**. Pytanie: *jak
właściwości szeregu (zależność w czasie, okresowość, szum) wpływają na skuteczność metod?* Odpowiedź budujemy na
**symulacjach** (sami generujemy szeregi i wstawiamy anomalie, więc wiemy, gdzie są), a potem sprawdzamy ją na
**danych rzeczywistych** (SPY, BTCUSDT, NYC Taxi). Główny wniosek: **nie ma jednej najlepszej metody – o wyniku
decyduje to, czy model „normy” w metodzie pasuje do struktury szeregu; zły model jest gorszy niż brak modelu.**

Kolejność prac: dane → przegląd danych → scenariusze S1–S3 → metody → sposób oceny → wyniki i porównanie S1–S3 →
dane rzeczywiste. Ten dokument idzie w tej samej kolejności.

---

## 1. Podstawowe pojęcia: szereg czasowy i anomalia

| Pojęcie | Co to jest |
|---|---|
| **Szereg czasowy** | ciąg pomiarów w równych odstępach czasu: x₁, x₂, …, xₙ (np. cena co 5 minut, pasażerowie co 30 minut). |
| **Anomalia** | fragment szeregu, który nie pasuje do jego „normalnego” zachowania. U nas zwykle **anomalia zbiorowa** – trwa kilkadziesiąt punktów, a nie jeden. |
| **Anomalia punktowa** | pojedyncza wartość odstająca (np. jedna skrajna świeca). |
| **Anomalia kolektywna / zbiorowa** | seria punktów, z których każdy osobno może wyglądać normalnie, ale razem są nietypowe (np. zmiana zmienności przez 50 punktów). |
| **Intensywność anomalii** | jak silna jest zmiana względem szumu (np. przesunięcie poziomu o 1σ, 2σ). |
| **Długość anomalii (L)** | ile punktów trwa (u nas 10–200). |
| **Wynik (score)** | liczba, którą metoda przypisuje każdemu punktowi – im większa, tym bardziej nietypowy. |
| **Alarm** | punkt, którego wynik przekracza **próg**. Fałszywy alarm = alarm poza anomalią. |

Trzy grupy anomalii, które rozróżniamy w porównaniu:
- **anomalie wartości** – zmienia się poziom albo rozrzut wartości (zmiana poziomu, zmiana zmienności, szok);
- **anomalie struktury** – zmienia się kształt cyklu (faza, amplituda, okres);
- **anomalie dynamiki** – zmienia się zależność między kolejnymi punktami (pojawienie się zależności, zmiana φ,
  przesunięcie równowagi w AR). Te są najtrudniejsze.

---

## 2. Krok 1 – dane

### 2.1 Zbiory

| Zbiór | Źródło | Co mierzy | Okres | Wielkość |
|---|---|---|---|---|
| **SPY** | Alpaca Market Data API (feed SIP – wszystkie giełdy USA) | ETF na indeks S&P 500, świece 5-min, tylko sesja NYSE (9:30–16:00) | 2016-01 – 2026-09 | 209 908 świec, 2 701 sesji |
| **BTCUSDT** | publiczne archiwum Binance (data.binance.vision) | kurs bitcoina w USDT, świece 5-min, 24/7 | 2017-08 – 2026-09 | 957 865 świec |
| **NYC Taxi** | Numenta Anomaly Benchmark (NAB) | liczba pasażerów taksówek w Nowym Jorku co 30 min | 2014-07 – 2015-01 | 10 320 punktów, **5 oznaczonych okien anomalii** |

NYC Taxi to jedyny zbiór z **etykietami** (wiemy, gdzie są anomalie) – dlatego tylko na nim liczymy skuteczność
ilościowo. Dla SPY i BTC zbudowaliśmy **katalog znanych zdarzeń** (rozdz. 3.3).

### 2.2 Pojęcia z danych rynkowych

- **Świeca (OHLCV)** – podsumowanie 5 minut handlu: **O**pen (cena otwarcia), **H**igh (maksimum), **L**ow (minimum),
  **C**lose (zamknięcie), **V**olume (wolumen – ile jednostek przehandlowano). **VWAP** – średnia cena ważona wolumenem.
- **Sesja NYSE** – godziny handlu na giełdzie nowojorskiej; zwykła sesja = 78 świec 5-min, skrócona (np. 24 grudnia) = 42.
- **Kalendarz giełdowy** (`pandas_market_calendars`) – lista dni i godzin sesji; z niego wiemy, ilu świec oczekiwać.
- **Ceny skorygowane** – uwzględniają dywidendy i splity, żeby nie było sztucznych skoków.
- **Wstrzymanie handlu (circuit breaker)** – automatyczna przerwa po spadku o 7% (w marcu 2020 – 4 razy).

### 2.3 Czyszczenie – zasady

1. **Dane surowe zapisujemy bez zmian** (`dane/surowe/`, z datą pobrania w nazwie); czyszczenie tworzy osobne pliki
   (`dane/clean/`).
2. **Braków nie uzupełniamy** (żadnej interpolacji) – tylko je **oznaczamy**: kolumna `luka_przed` (ile świec brakuje
   przed daną), `dzien_niepelny`, lista brakujących świec w CSV. Metody nie liczą zwrotów ani okien przez luki.
3. Problemy znalezione i rozwiązane:
   - Binance od 2025 r. zapisuje czas w **mikrosekundach** (16 cyfr) zamiast milisekund – skrypt rozpoznaje oba;
   - po przerwie technicznej 8–9.02.2018 część świec BTC była **przesunięta o 194,789 s** względem siatki 5-min –
     wyrównujemy je (kolumna `czas_skorygowany`);
   - NYC Taxi: przy zmianie czasu 2.11.2014 dwa półgodzinne punkty sumują dwie godziny (maksimum szeregu – artefakt) –
     oznaczone `zmiana_czasu`, w etapie 2 dzielone przez 2.
4. **Kontrola jakości** (`src/eksploracja/raport_danych.py`) – niezależna od czyszczenia: liczy od nowa siatkę
   oczekiwanych świec i porównuje. Wynik: SPY 14 brakujących świec (8 to wstrzymania handlu w marcu 2020), BTC 1 703
   w 33 przerwach, NYC 0; wszędzie 0 duplikatów i 0 niespójnych świec (np. Low > Close).

---

## 3. Krok 2 – właściwości danych i przegląd anomalii

### 3.1 Przekształcenia szeregu

| Pojęcie | Wzór / opis | Po co |
|---|---|---|
| **Zwrot logarytmiczny** | rₜ = ln(Cₜ) − ln(Cₜ₋₁) | cena ma trend, zwroty są „stacjonarne” – z nimi da się pracować |
| **Zmienność zrealizowana (RV)** | RV dnia = √(Σ rₜ²) ze zwrotów 5-min w tym dniu | miara „nerwowości” rynku w danym dniu |
| **log RV** | ln(RV) | rozkład RV jest skośny; po logarytmie bliższy normalnemu |
| **Wolumen względny** | ln(wolumen dnia) − mediana z 250 dni wokół | usuwa wieloletni trend, zostawia nietypowe dni |
| **Sezonowość dobowa / profil** | średnie |r| o danej porze dnia | bez jej usunięcia każde otwarcie sesji wygląda na anomalię |

### 3.2 Statystyki opisowe, które się przewijają

- **Średnia, odchylenie standardowe (σ)** – klasyczne, ale wrażliwe na wartości skrajne.
- **Mediana i MAD** (mediana odchyleń bezwzględnych od mediany) – **odporne** odpowiedniki. Odporny z-score:
  `z = (x − mediana) / (1,4826 · MAD)`; stała 1,4826 sprawia, że dla rozkładu normalnego MAD·1,4826 = σ.
- **Kurtoza** – „grubość ogonów”; dla rozkładu normalnego 0 (nadwyżkowa). U nas: zwroty SPY **48**, BTC **108** –
  bardzo ciężkie ogony, czyli skrajne ruchy są dużo częstsze niż w rozkładzie normalnym. Dlatego wszędzie mediana i MAD.
- **Autokorelacja z opóźnieniem k** – korelacja szeregu z nim samym przesuniętym o k punktów. Mówi, jak bardzo punkt
  zależy od poprzednich. Zwroty: ≈ 0 (SPY −0,01). |zwroty|: 0,39 SPY, 0,42 BTC → **skupianie zmienności** (po dużym
  ruchu przychodzi kolejny duży). NYC: po tygodniu 0,92 → silny **cykl**.
- **Stacjonarność** – rozkład szeregu nie zmienia się w czasie. Większość metod ją zakłada; anomalia ją łamie.

### 3.3 Właściwości → scenariusze, katalog zdarzeń

Trzy właściwości z danych stały się trzema scenariuszami symulacji:

| Właściwość w danych | Scenariusz |
|---|---|
| zwroty prawie nieskorelowane | **S1 – biały szum** |
| zmienność skupia się w czasie (zależność) | **S2 – AR(1)** |
| cykl dobowy/tygodniowy (NYC, profil „U” sesji SPY) | **S3 – szum + sinus** |

**Katalog zdarzeń SPY/BTC** (`src/rzeczywiste/katalog.py`) – 49 znanych zdarzeń w sześciu typach: **A** punktowe
(skrajna świeca), **B** kolektywne (krótka seria), **C** zmiana poziomu (np. luka overnight), **D** reżim zmienności
(np. marzec 2020 albo „nienaturalny spokój”), **E** struktura dobowa (posiedzenia Fed – FOMC), **F** wolumen.
Zastępuje etykiety, ale jest *częściowy* (powstał z przeglądu tych samych danych).

---

## 4. Krok 3 – symulacje: scenariusze S1–S3

**Po co symulacje?** Na danych rzeczywistych nie wiemy, gdzie dokładnie są anomalie. W symulacji generujemy szereg
o znanej strukturze, wstawiamy anomalie w znane miejsca i możemy policzyć, ile metoda wykryła i ile dała fałszywych
alarmów. **Ziarno losowania** (seed) sprawia, że każdy szereg da się odtworzyć.

| | Tło | Anomalie A / B / C |
|---|---|---|
| **S1** | **biały szum**: xₜ = εₜ, εₜ ~ N(0, 1) – punkty niezależne | zmiana poziomu / zmiana zmienności / pojawienie się zależności |
| **S2** | **AR(1)**: xₜ = φ·xₜ₋₁ + εₜ, φ = 0,7 – punkt zależy od poprzedniego | szok (impuls, który wygasa) / zmiana siły zależności φ / przesunięcie poziomu równowagi |
| **S3** | **sinus + szum**: xₜ = 3·sin(2πt/50) + εₜ – okres P = 50 | przesunięcie fazy / zmiana amplitudy / zmiana okresu |

- **AR(1) (model autoregresyjny rzędu 1)** – każdy punkt to φ razy poprzedni plus szum. φ mówi, jak silna jest „pamięć”:
  φ = 0 to biały szum, φ bliskie 1 – szereg długo „wędruje”. Autokorelacja z opóźnieniem k w AR(1) = φᵏ (maleje
  wykładniczo) – to ważne w etapie 2.
- Każdy scenariusz: **1 200 szeregów** po **3 000 punktów** (4 poziomy intensywności × 3 długości L × 100 szeregów),
  w każdym szeregu wszystkie trzy anomalie. Do tego części dodatkowe: długość okna, poziom szumu (S1), siła tła (S2: φ,
  S3: amplituda cyklu).

---

## 5. Krok 4 – metody

**Wspólna zasada:** metoda nie decyduje, co jest anomalią – zwraca **wynik dla każdego punktu**. Alarm daje dopiero
**wspólny próg** (rozdz. 6). Dzięki temu porównujemy metody o zupełnie różnych skalach wyników.

Sześć **rodzin** metod, w każdym scenariuszu ta sama rola, ale **konfiguracja dopasowana do szeregu**:

| Grupa | Rodzina | S1 | S2 | S3 |
|---|---|---|---|---|
| statystyczne | karta kontrolna | CUSUM (3 karty) | reszty AR + CUSUM | STL + EWMA |
| statystyczne | podział na odcinki | PELT | PELT, koszt AR | PELT na resztach STL |
| fragmenty szeregu | Matrix Profile | na szeregu | na resztach AR | z normalizacją okna |
| fragmenty szeregu | kNN | cechy okien | cechy okien | cechy cykli |
| uczenie maszynowe | Isolation Forest | cechy okien | cechy okien | cechy cykli |
| uczenie maszynowe | druga metoda ML | LOF | One-Class SVM | autoenkoder |

### 5.1 Elementy wspólne

- **Okno przesuwne** – fragment długości w (u nas 50 punktów; w S3 = jeden cykl) przesuwany co 5 punktów. Metody
  okienkowe oceniają okna, a wynik przypisujemy punktom (interpolacja między środkami okien).
- **Cechy okna** – liczby opisujące fragment. **3 cechy** = po jednej na rodzaj anomalii; **5 cech** = te 3 + 2
  pomocnicze. S1: średnia, odchylenie std, autokorelacja (+ kurtoza, rozstęp). S2: średnia, autokorelacja,
  maks. |reszta AR| (+ odch. std reszt, autokorelacja z opóźnieniem 2). S3 („cechy cykli”): faza, amplituda, dominująca
  częstotliwość dopasowanego sinusa (+ odch. std reszt, korelacja z wzorcowym cyklem).
- **Odporna standaryzacja cech** – (cecha − mediana) / (1,4826·MAD), żeby żadna cecha nie dominowała.
- **Reszty modelu** – to, co zostaje po odjęciu znanej struktury: S2 – **reszty AR** eₜ = xₜ − c − φ̂·xₜ₋₁ (φ̂ z MNK –
  metody najmniejszych kwadratów); S3 – **reszty STL**. W resztach anomalia jest widoczna jak w białym szumie.
- **STL** (Seasonal-Trend decomposition using Loess) – rozkład szeregu na **trend + sezonowość + resztę**; Loess to
  lokalna regresja wygładzająca. Wersja odporna (robust) nie daje się „ściągnąć” wartościom skrajnym.

### 5.2 Metody statystyczne

**Karta kontrolna** (wywodzi się ze statystycznej kontroli jakości) – śledzi statystykę w czasie i alarmuje, gdy
wyjdzie poza normę. Najprostsza – **Shewhart** (alarm, gdy |z| > 3).

- **CUSUM** (cumulative sum, Page 1954) – sumuje nadwyżki ponad wartość odniesienia k:
  `Sₜ = max(0, Sₜ₋₁ + zₜ − k)`. W normie trzyma się przy zerze; przy **trwałej** zmianie rośnie liniowo. Dobry na
  małe, długie zmiany. U nas trzy karty w S1: poziom (z = standaryzowana wartość), zmienność i zależność (z kwadratów
  różnic sąsiednich punktów: E[(xₜ − xₜ₋₁)²] = 2σ² – rośnie przy większej zmienności, maleje przy zależności).
  Liczona w przód i w tył (lokalizacja), wynik = maksimum kart.
- **Reszty AR + CUSUM** (S2) – najpierw model AR(1) usuwa znaną zależność, potem karty CUSUM na resztach + test
  szoku |reszta|. Najlepsza metoda S2.
- **EWMA** (exponentially weighted moving average, Roberts 1959) – średnia z wagami malejącymi wykładniczo:
  `Eₜ = λ·zₜ + (1 − λ)·Eₜ₋₁`; λ = 2/(P + 1) – pamięć ≈ jeden cykl. **STL + EWMA** (S3): na resztach STL liczymy
  zₜ = rₜ² − 1 („energia” reszt rośnie przy każdej anomalii cyklu).
- **PELT** (Pruned Exact Linear Time, Killick i in. 2012) – **wykrywanie punktów zmiany**: dzieli szereg na odcinki
  tak, żeby zminimalizować `Σ koszt(odcinek) + β·liczba odcinków`. **Funkcja kosztu** mierzy, jak dobrze odcinek
  opisuje jeden zestaw parametrów (u nas n·ln σ̂² – przy stałej wariancji); **kara β** chroni przed dzieleniem na zbyt
  wiele odcinków. „Pruned” – przycinanie kandydatów daje czas ≈ liniowy. Odcinek różny od tła = anomalia.
  Własna implementacja (ok. 140× szybsza od biblioteki `ruptures`, te same granice). Kara dobrana na szeregach
  bez anomalii.

### 5.3 Metody oparte na fragmentach szeregu

- **Matrix Profile** (Yeh i in. 2016, biblioteka `stumpy`) – dla każdego fragmentu długości m liczy odległość do
  **najbardziej podobnego innego fragmentu**. Fragment bez „bliźniaka” = **discord** = anomalia. Pomija **dopasowania
  trywialne** (fragmenty nakładające się w czasie). **Z normalizacją** (każde okno sprowadzone do średniej 0
  i odchylenia 1) porównuje tylko **kształt**; **bez normalizacji** – także poziom. Słabość: nie widzi fragmentów
  „gładszych” niż reszta (mają bliższych sąsiadów).
- **kNN na cechach okien** (Ramaswamy i in. 2000) – każde okno to punkt w przestrzeni cech; wynik = średnia odległość
  do **k = 10 najbliższych** okien (bez nakładających się). Okno daleko od innych = anomalia. Najbardziej uniwersalna
  metoda projektu, ale tak dobra, jak jej cechy.

### 5.4 Uczenie maszynowe

- **Uczenie nienadzorowane** – model uczy się na tym samym szeregu, który ocenia (bez etykiet). **Półnadzorowane /
  novelty detection** – model uczy się na danych „czystych” (u nas 20 osobnych szeregów bez anomalii), a potem ocenia
  nowy szereg. Sprawdzaliśmy oba warianty.
- **Maskowanie** – długa anomalia daje wiele podobnych okien, więc model uczony na tym samym szeregu uznaje je za normę.
- **Isolation Forest** (Liu i in. 2008) – 200 losowych drzew; każde drzewo dzieli dane losowymi cięciami. Punkt
  nietypowy **izoluje się po niewielu cięciach** (krótka ścieżka w drzewie) → wysoki wynik. Szybki, dobry ranking,
  ale podatny na maskowanie.
- **LOF** (Local Outlier Factor, Breunig i in. 2000) – porównuje **gęstość** wokół punktu z gęstością wokół jego
  k sąsiadów; punkt w rzadszym obszarze niż sąsiedzi = anomalia.
- **One-Class SVM** (Schölkopf i in. 2001) – wyznacza granicę obszaru „normy” w przestrzeni cech; **jądro RBF**
  K = exp(−γ·d²) mierzy podobieństwo. **ν** (u nas 0,05) – jaki ułamek danych może leżeć poza granicą; **γ** (u nas
  0,01 zamiast domyślnego) – szerokość jądra; za duże γ powoduje **nasycenie** (wszystkie dalekie okna dostają ten
  sam wynik).
- **Autoenkoder** – sieć neuronowa 50 → 32 → 2 → 32 → 50, która uczy się **odtwarzać** okno przez wąskie „gardło”
  (2 neurony). Normalne okna odtwarza dobrze; wynik = **błąd odtworzenia**. U nas słaby: wszystkie sinusy o danym
  okresie mieszczą się w 2 wymiarach, więc sieć odtwarza też cykl przesunięty czy osłabiony.

---

## 6. Krok 5 – jak oceniamy metody

| Pojęcie | Znaczenie u nas |
|---|---|
| **Wspólny próg (kalibracja)** | na **200 szeregach bez anomalii** liczymy maksimum wyniku metody w każdym; próg = **kwantyl 90%** tych maksimów → fałszywy alarm w ok. **10% szeregów bez anomalii**, dla każdej metody tak samo. Porównujemy więc czułość przy tym samym poziomie fałszywych alarmów. |
| **Epizod** | ciągła seria punktów z alarmem (jeden „sygnał”). |
| **Strefa tolerancji** | anomalia ± 25 punktów – metody okienkowe i kumulacyjne rozmywają alarm, więc alarm tuż obok nie jest błędem. |
| **Anomalia wykryta** | choć jeden alarm w jej strefie. **Fałszywy alarm** – epizod, który nie dotyka żadnej strefy. |
| **Precyzja** | ile z alarmów to trafienia: TP / (TP + FP). |
| **Czułość (recall)** | ile anomalii znaleziono: TP / (TP + FN). |
| **F1** | średnia harmoniczna precyzji i czułości. |
| **ROC i AUC** | krzywa ROC: odsetek trafień w funkcji odsetka fałszywych alarmów przy wszystkich możliwych progach. **AUC** = pole pod nią = prawdopodobieństwo, że losowy punkt anomalii ma wyższy wynik niż losowy punkt normy. 0,5 = zgadywanie, 1 = idealnie. **Nie zależy od progu.** |
| **AP (PR-AUC)** | pole pod krzywą precyzja–czułość; czulsze na rzadkie anomalie. |
| **Walidacja progu** | progi sprawdzone na **1 000 nowych** szeregach bez anomalii: fałszywe alarmy 4,0–15,7% (cel 10%). |

Dlaczego AUC *i* odsetek wykrytych? AUC mówi, czy metoda dobrze **porządkuje** punkty; odsetek wykrytych – czy
przy rozsądnym progu **naprawdę alarmuje**. Isolation Forest ma często AUC 0,98, a wykrywa mało (maskowanie
podnosi maksimum wyniku na szeregach bez anomalii, więc próg jest wysoko).

---

## 7. Krok 6 – wyniki symulacji i porównanie S1–S3

| Scenariusz | Najlepsza metoda | Wykryte (średnio A, B, C) | Uwagi |
|---|---|---|---|
| S1 | kNN na 3 cechach | 69% | CUSUM 66% – prawie równo |
| S2 | reszty AR + CUSUM | 47% | najwyższe średnie AUC metody tylko 0,85 |
| S3 | kNN na cechach cykli | 91% | AUC 0,99 |

**Eksperyment przeniesienia** – metody dopasowane do jednego scenariusza uruchamiamy na szeregach innego (te same
anomalie). Mierzy, ile daje dopasowanie metody do szeregu:
- na szeregu okresowym metody **bez modelu** wykrywają średnio **55%**, metody **z modelem AR – 34%**;
- na szeregu AR metody bez modelu **18%**, z modelem cyklu – **5%**;
- model ogólniejszy nic nie kosztuje: reszty AR + CUSUM na białym szumie 88 / 95 / 32% wobec 92 / 95 / 38% CUSUM
  (AR(1) z φ̂ ≈ 0 to po prostu biały szum).

**Wnioski z porównania:**
1. Nie ma jednej najlepszej metody; kolejność zmienia się z szeregiem.
2. Najbardziej uniwersalny – kNN na cechach okien (o ile cechy opisują zaburzaną właściwość).
3. Decyduje dopasowanie modelu normy do struktury szeregu; **zły model jest gorszy niż brak modelu**.
4. Zależność w czasie utrudnia najbardziej (AUC 0,85 w S2 wobec 0,93 w S1 i 0,99 w S3) – szereg AR sam „wędruje”.
5. Okresowość pomaga, gdy cykl jest wyraźny; przy amplitudzie równej szumowi cechy z jednego okna zawodzą, a model
   cyklu z całego szeregu (STL) działa dalej.
6. Szum liczy się względem wielkości anomalii (przesunięcie Δ = 1: przy σ = 0,5 karta wykrywa 100%, przy σ = 2 – 20%).
7. Anomalie wartości najłatwiejsze (84%), struktury 72%, dynamiki 41%.
8. Ustawienia też zależą od szeregu: 3 cechy lepsze w S1, 5 w S2; okno w S3 – co najmniej jeden cykl. Isolation
   Forest uczony bez anomalii zamiast na tym samym szeregu: S3 31 / 25 / 78% → 76 / 82 / 94%.

---

## 8. Krok 7 – dane rzeczywiste

### 8.1 Jak oceniamy bez etykiet / bez szeregów „czystych”

- **Okres kalibracyjny i okres oceny** – próg, karę PELT i uczenie ML ustawiamy na „historii”, oceniamy na późniejszym
  okresie (jak w praktyce). NYC: pierwsze 105 dni / 110 dni z 5 oknami NAB. SPY, BTC: do 2019 / od 2020.
- **Jednostka oceny – dzień.** Próg = kwantyl (1 − α) dziennych maksimów wyniku w dniach kalibracji, czyli alarm w α
  dni „historii” (NYC: α = 2, 5, 10%; SPY/BTC: 1, 2, 5%).
- **recall@N** (SPY, BTC) – odsetek zdarzeń z katalogu, w które (± 1 dzień) trafia któryś z N% dni o najwyższym wyniku.
  Precyzji nie liczymy, bo katalog nie jest kompletny.
- **Indeks Jaccarda** – |A ∩ B| / |A ∪ B|; mierzy, czy dwie metody wskazują te same dni (mediana ok. 0,3 – metody
  w dużej części wskazują różne dni).

### 8.2 NYC Taxi (szereg okresowy → metody S3)

- Zmiany względem S3: **log** liczby pasażerów (cykl multiplikatywny → addytywny), **okres STL = tydzień** (336
  punktów), **okno = doba** (48), cechy liczone **względem profilu tygodnia** (mediana każdego punktu tygodnia).
- Wynik przy α = 5%: **9 z 13 wariantów metod wykrywa wszystkie 5 okien** przy 3–5% dni z fałszywym alarmem.
  AUC: Matrix Profile bez normalizacji **0,93**, kNN na cechach cykli **0,90**; **CUSUM z S1 (bez modelu cyklu): AUC
  0,50, 0/5** – potwierdzenie eksperymentu przeniesienia.
- Najtrudniejsze okno – maraton (zaburzenie kilku godzin jednego dnia). Metody wskazują też **święta nieoznaczone
  w NAB** (Labor Day) – w ocenie liczą się jako fałszywe alarmy, choć są prawdziwe.
- **PELT wymagał poprawki**: reszty STL mają autokorelację 0,98, a test istotności odcinka zakłada niezależność.
  **Efektywna liczba obserwacji** n_eff = n·(1 − ρ)/(1 + ρ) – przy silnej zależności n punktów niesie informację jak
  znacznie mniej niezależnych.

### 8.3 SPY i BTC (zależność → metody S2)

- Szereg dzienny: **log RV** (zmienność) i wolumen względny. Autokorelacja log RV po 1 dniu 0,80 (SPY) – jak w S2.
- Ale **długa pamięć**: po 10 dniach autokorelacja wciąż 0,53, a w AR(1) byłoby 0,8¹⁰ ≈ 0,11. Zmienność po kryzysie
  wraca do normy miesiącami.
- Skutek: **reszty AR + CUSUM** (najlepsza w S2) – recall@2% **0% (SPY), 10% (BTC)**; reszty są dodatnie przez miesiące,
  karta sumuje je i wskazuje okres po zdarzeniu (SPY: wynik na maksimum przez 212 sesji, 10.2020–08.2021). **PELT z kosztem
  AR** – „wszystko albo nic”. **Metody okienkowe** (kNN, Isolation Forest, One-Class SVM, Matrix Profile) –
  **43–67%** zdarzeń w 2% najwyższych dni, przy 8–15 (SPY) i 2–5 (BTC) dniach z alarmem na rok.
- To mocniejsza wersja wniosku z symulacji: zbyt prosty model tła szkodzi.

### 8.4 Skala czasu – studia przypadków 5-min

- Ruch 5-min dzielimy przez medianę |r| o tej samej porze dnia (usunięcie sezonowości dobowej).
- **Flash crash BTC 10.10.2025**: o 21:15 UTC ruch 97× większy od typowego – a w szeregu dziennym **żadna** metoda nie
  ma tego dnia w 2% najwyższych. **17.08.2023** – ten sam log RV (−2,29), a wskazuje go 11 z 13 metod. Różnica: **tło**
  (mediana log RV ± 30 dni: −4,43 w 2023 wobec −4,01 w 2025). Ten sam ruch jest anomalią albo nie – zależnie od otoczenia.
- **Dni FOMC** (komunikat Fed o 14:00): ruchy o 14:00–14:55 kilkukrotnie większe niż zwykle, ale w szeregu dziennym
  metody wskazują 1 z 5 takich dni. **Anomalie krótsze niż jednostka szeregu trzeba szukać na danych o wyższej
  częstotliwości.**

---

## 9. Narzędzia i repozytorium

| Narzędzie | Do czego |
|---|---|
| Python 3 | wszystko |
| `pandas`, `numpy`, `pyarrow` | tabele, obliczenia, pliki parquet |
| `requests`, `python-dotenv` | pobieranie z API; klucze w `.env` (nigdy w Git!) |
| `pandas_market_calendars` | kalendarz sesji NYSE |
| `scipy`, `statsmodels` | statystyki, rozkład STL |
| `scikit-learn` | Isolation Forest, LOF, One-Class SVM, autoenkoder (MLPRegressor), AUC/AP |
| `stumpy` | Matrix Profile |
| `ruptures` | punkt odniesienia dla naszej implementacji PELT |
| `matplotlib` | wykresy |
| `pytest` | 50 testów jednostkowych (`tests/`) |

Struktura repo (szczegóły: [PROJEKT.md](../PROJEKT.md)): `src/pobieranie`, `src/czyszczenie`, `src/eksploracja` (dane),
`src/symulacje` (generator szeregów i anomalii), `src/metody` (metody), `src/ewaluacja` (ocena, eksperymenty S1–S3,
porównanie), `src/rzeczywiste` (etap 2); wyniki w `wyniki/`, dokumentacja i raporty PDF w `docs/`.

Uruchomienie: `pip install -r requirements.txt`, testy `pytest`, eksperymenty np. `python -m src.ewaluacja.eksperyment_s1`,
etap 2 `python -m src.rzeczywiste.nyc_taxi` (pełna lista – [README.md](../README.md)).

---

## 10. Najważniejsze liczby – ściąga

| Co | Liczba |
|---|---|
| dane | SPY 209 908 świec, BTC 957 865, NYC 10 320 punktów; braki 14 / 1 703 / 0 |
| kurtoza zwrotów | SPY 48, BTC 108 (rozkład normalny: 0) |
| symulacje | 3 scenariusze × 1 200 szeregów × 3 000 punktów, 3 anomalie w szeregu |
| próg | fałszywy alarm w 10% szeregów bez anomalii; walidacja na 1 000 szeregach: 4,0–15,7% |
| najlepsze w symulacjach | S1 kNN 69%, S2 reszty AR + CUSUM 47%, S3 kNN na cechach cykli 91% |
| przeniesienie | szereg okresowy: bez modelu 55% vs model AR 34%; szereg AR: bez modelu 18% vs model cyklu 5% |
| NYC Taxi | 5/5 okien przy 3–5% dni z fałszywym alarmem; AUC 0,93 (Matrix Profile), 0,90 (kNN), 0,50 (CUSUM bez cyklu) |
| SPY / BTC | autokorelacja log RV po 10 dniach 0,53 (AR(1): 0,11); recall@2%: metody okienkowe 43–67%, reszty AR + CUSUM 0–10% |
| flash crash BTC 10.10.2025 | ruch 5-min 97× typowy, 0 metod w szeregu dziennym |
| repo | 6 rodzin metod, 50 testów, 7 raportów PDF |

---

## 11. Pytania z dyskusji – z odpowiedziami

**Dlaczego symulacje, a nie od razu dane rzeczywiste?**
Na danych rzeczywistych nie znamy dokładnego położenia anomalii, więc nie da się policzyć skuteczności ani fałszywych
alarmów. W symulacji kontrolujemy też jedną właściwość naraz (zależność, okresowość, szum) – a o to pyta temat.

**Skąd 10% fałszywych alarmów?**
To umowny poziom; ważne, że **ten sam** dla wszystkich metod, bo wtedy ich czułość jest porównywalna. Na danych
rzeczywistych podajemy wyniki dla kilku poziomów (NYC Taxi: alarm w 2, 5 i 10% dni kalibracji).

**Dlaczego próg ustalany na szeregach bez anomalii, a nie np. 3σ?**
Wyniki metod mają różne skale (odległość, statystyka testowa, błąd odtworzenia) i różne rozkłady; „3σ” znaczyłoby
co innego dla każdej. Kwantyl maksimów na szeregach bez anomalii daje każdej metodzie ten sam odsetek fałszywych alarmów.

**Czym różni się AUC od odsetka wykrytych?**
AUC – czy metoda dobrze porządkuje punkty, przy wszystkich progach naraz. Odsetek wykrytych – przy jednym, konkretnym
progu. Metoda może mieć wysokie AUC, a mało wykrywać (Isolation Forest – maskowanie).

**Dlaczego Isolation Forest wypada słabo, skoro jest popularny?**
Uczony na tym samym szeregu „uczy się” długiej anomalii (wiele podobnych okien) – maskowanie. Uczony na szeregach bez
anomalii poprawia się wyraźnie (S3: 31/25/78% → 76/82/94%). Na danych rzeczywistych „czystej” historii nie ma, więc
zysk znika.

**Co to jest strefa tolerancji i czy nie zawyża wyników?**
± 25 punktów wokół anomalii – metody okienkowe i CUSUM sygnalizują z lekkim przesunięciem. Może sprzyjać metodom
„rozmywającym” alarm, dlatego obok podajemy AUC i miary punktowe (precyzja, F1).

**Dlaczego kNN jest najlepszy, skoro to prosta metoda?**
Bo dostaje dobre cechy – opisujące dokładnie te właściwości, które anomalia zaburza. Zaleta i wada: wymaga wiedzy
o szeregu przy doborze cech. Przy zbędnych cechach słabnie (S1: 5 cech gorzej niż 3).

**Dlaczego metody z modelem AR zawiodły na SPY/BTC, skoro w S2 były najlepsze?**
W S2 tło było **dokładnie** AR(1). Zmienność rynkowa ma **długą pamięć** – autokorelacja maleje dużo wolniej niż φᵏ.
Reszty AR(1) zostają skorelowane, CUSUM sumuje je miesiącami i wskazuje okres po kryzysie zamiast samego zdarzenia.

**Jak oceniacie SPY i BTC bez etykiet?**
Katalogiem 49 znanych zdarzeń (krachy, decyzje Fed, rekordowy wolumen) i miarą recall@N. Precyzji nie liczymy, bo
katalog jest niekompletny – metoda może wskazać prawdziwą anomalię, której w nim nie ma (np. BTC 14–16.03.2020).

**Co to jest PELT i jak dobieracie karę?**
Algorytm dzielący szereg na odcinki o stałych parametrach (minimalizuje koszt + kara za każdy odcinek). Karę dobieramy
na szeregach bez anomalii: najmniejsza, przy której fałszywy odcinek pojawia się w ≤ 10% szeregów.

**Dlaczego STL, a nie np. średnia z cykli?**
STL jest odporny na wartości skrajne i nie „wchłania” długich anomalii (okno sezonowości dłuższe niż szereg). Średnia
z cykli przesunęłaby się razem z anomalią.

**Co to jest Matrix Profile i kiedy normalizować okno?**
Odległość każdego fragmentu do najbliższego innego fragmentu; fragment bez podobnego = anomalia. Normalizacja usuwa
poziom i skalę – dobra, gdy anomalia zmienia kształt (S3), zła, gdy zmienia poziom (NYC Taxi: bez normalizacji AUC 0,93).

**Dlaczego dane dzienne, a nie 5-minutowe dla SPY/BTC?**
Metody O(n²) (Matrix Profile, kNN) na ~1 mln punktów są za wolne, a katalog zdarzeń jest dzienny. Studia przypadków
5-min pokazały, że część anomalii (minutowe, FOMC) wymaga danych śróddziennych – to propozycja rozszerzenia.

**Czy wyniki są powtarzalne?**
Tak – każdy szereg ma własne ziarno losowania, szeregi kalibracji, walidacji i uczenia mają osobne zakresy ziaren;
`pytest` (50 testów) sprawdza metody i ocenę; każdy wynik odtwarza jedno polecenie (README).

**Co zostało do zrobienia?**
Raport końcowy. Opcjonalnie: metody na danych 5-min SPY/BTC (anomalie punktowe, dni FOMC) i ranking dni względem ich
otoczenia, a nie całej historii.

---

## 12. Słowniczek polsko-angielski

| Polski | Angielski |
|---|---|
| anomalia, obserwacja odstająca | anomaly, outlier |
| anomalia zbiorowa / kolektywna | collective anomaly |
| punkt zmiany | change point |
| szum biały | white noise |
| autokorelacja | autocorrelation |
| model autoregresyjny AR(1) | autoregressive model AR(1) |
| zmienność zrealizowana | realized volatility (RV) |
| skupianie zmienności | volatility clustering |
| długa pamięć | long memory |
| okno przesuwne | sliding window |
| cechy okna | window features |
| karta kontrolna | control chart |
| karta sum skumulowanych | CUSUM |
| średnia ruchoma ważona wykładniczo | EWMA |
| rozkład STL | STL decomposition |
| las izolacyjny | Isolation Forest |
| lokalny współczynnik odstawania | Local Outlier Factor (LOF) |
| maszyna wektorów nośnych jednej klasy | One-Class SVM |
| autoenkoder, błąd odtworzenia | autoencoder, reconstruction error |
| maskowanie | masking |
| próg, fałszywy alarm | threshold, false alarm / false positive |
| czułość, precyzja | recall (sensitivity), precision |
| pole pod krzywą ROC | ROC-AUC |
| średnia precyzja | average precision (PR-AUC) |
| uczenie nienadzorowane / półnadzorowane | unsupervised / semi-supervised (novelty detection) |
| efektywna liczba obserwacji | effective sample size |
