# Metody wykrywania anomalii – szczegółowy opis działania

> Stan na **2026-10-26** (po scenariuszach S1, S2 i S3). Dokument opisuje każdą metodę użytą w projekcie: intuicję,
> działanie krok po kroku (ze wzorami), parametry przyjęte w projekcie, sposób kalibracji, mocne i słabe strony
> potwierdzone w eksperymentach oraz miejsce w kodzie. Wyniki liczbowe: [wyniki_s1.md](wyniki_s1.md),
> [wyniki_s2.md](wyniki_s2.md), [wyniki_s3.md](wyniki_s3.md). Dobór metod do scenariuszy:
> [plan_etap_1.md](plan_etap_1.md), rozdz. 4.

## Spis treści

0. [Elementy wspólne wszystkich metod](#0-elementy-wspólne-wszystkich-metod)
1. [CUSUM – karta sum skumulowanych](#1-cusum--karta-sum-skumulowanych) (statystyczna; S1)
2. [Reszty modelu AR(1) + CUSUM](#2-reszty-modelu-ar1--cusum) (statystyczna; S2)
3. [PELT – optymalny podział na odcinki](#3-pelt--optymalny-podział-na-odcinki) (statystyczna; S1, S2, S3)
4. [Matrix Profile – fragmenty bez „bliźniaka”](#4-matrix-profile--fragmenty-bez-bliźniaka) (fragmenty szeregu; S1, S2, S3)
5. [Cechy okien + kNN](#5-cechy-okien--knn) (fragmenty szeregu; S1, S2, S3)
6. [Isolation Forest – las izolacyjny](#6-isolation-forest--las-izolacyjny) (uczenie maszynowe; S1, S2, S3)
7. [LOF – Local Outlier Factor](#7-lof--local-outlier-factor) (uczenie maszynowe; S1)
8. [One-Class SVM](#8-one-class-svm) (uczenie maszynowe; S2)
9. [STL + karta EWMA](#9-stl--karta-ewma) (statystyczna; S3)
10. [Autoenkoder](#10-autoenkoder) (uczenie maszynowe; S3)
11. [Zestawienie metod](#11-zestawienie-metod)
12. [Literatura](#12-literatura)

---

## 0. Elementy wspólne wszystkich metod

### 0.1 Wspólny interfejs: wynik dla każdego punktu

Każda metoda (`src/metody/baza.py`, klasa `Detektor`) robi jedno: dla szeregu `x_1, …, x_n` zwraca **wynik anomalii
dla każdego punktu** – liczbę, która jest tym większa, im bardziej punkt (albo fragment, do którego należy) odbiega od
normy. Metoda **nie decyduje**, co jest anomalią. Decyzję podejmuje wspólny dla wszystkich metod **próg**:
punkt jest alarmem, gdy jego wynik przekracza próg (rozdz. 0.7). Dzięki temu metody o zupełnie różnych skalach
wyników (odległość, prawdopodobieństwo, statystyka testowa) ocenia się według jednej zasady.

Wszystkie metody działają **offline** – dostają cały szereg naraz (ustalenie z konsultacji).

### 0.2 Okna i przypisanie wyniku punktom

Anomalie w projekcie są **zbiorowe** (fragmenty szeregu), więc większość metod ocenia **okna**: fragmenty długości
`w` (domyślnie 50 punktów), przesuwane co `krok` = 5 punktów (`okna()` w `baza.py`). Wynik okna przypisujemy punktom
przez **interpolację liniową między środkami okien** (`do_punktow()`): punkt dostaje wynik okna, którego środek leży
najbliżej, i płynne przejście między sąsiednimi oknami. Wcześniejsza wersja („maksimum ze wszystkich okien
pokrywających punkt”) rozlewała alarm o całą długość okna poza anomalię.

### 0.3 Cechy okien i odporna standaryzacja

Metody kNN, Isolation Forest, LOF i One-Class SVM nie patrzą na surowe wartości okna, tylko na jego **cechy** –
kilka liczb opisujących okno. Cechy dobieramy **pod anomalie danego scenariusza**, w dwóch wariantach (ustalenie
z konsultacji 2026-10-27 – w każdym scenariuszu oba, bo nie wiadomo z góry, który działa lepiej):

- **3 cechy** – po jednej cesze na każdy rodzaj anomalii scenariusza,
- **5 cech** – te 3 + 2 cechy pomocnicze z planu ([plan_etap_1.md](plan_etap_1.md), rozdz. 4.3–4.5).

| Scenariusz | Cecha okna | Na którą anomalię | 3 cechy | 5 cech |
|---|---|---|---|---|
| S1 | średnia | zmiana poziomu | ✔ | ✔ |
| | odchylenie standardowe | zmiana zmienności | ✔ | ✔ |
| | autokorelacja z opóźnieniem 1 | pojawienie się zależności | ✔ | ✔ |
| | kurtoza, rozstęp | pomocnicze | | ✔ |
| S2 | średnia x | przesunięcie równowagi | ✔ | ✔ |
| | autokorelacja x z opóźnieniem 1 | zmiana siły zależności | ✔ | ✔ |
| | maks. \|reszta modelu AR\| | szok | ✔ | ✔ |
| | odch. std reszt modelu AR | pomocnicza – rośnie, gdy model przestaje pasować (szok, zmiana φ) | | ✔ |
| | autokorelacja x z opóźnieniem 2 | pomocnicza (plan: autokorelacja z opóźnieniem 1–3) | | ✔ |
| S3 („cechy cykli”) | przesunięcie fazy okna względem fazy typowej dla szeregu | przesunięcie fazy | ✔ | ✔ |
| | amplituda sinusa dopasowanego w oknie | zmiana amplitudy | ✔ | ✔ |
| | dominująca częstotliwość okna | zmiana okresu | ✔ | ✔ |
| | odch. std reszt dopasowania sinusa | pomocnicza – rośnie przy każdej anomalii | | ✔ |
| | korelacja okna z wzorcowym cyklem (sinus o fazie typowej) | pomocnicza (plan) | | ✔ |

Do 2026-10-27 S2 i S3 używały po 4 cechy (zestaw 3-cechowy + odch. std reszt); wyniki tamtej wersji – w historii
repozytorium. Metody na cechach mają w tabelach dopisek „(5 cech)” albo „(3 cechy)”.

Autokorelacja okna: `ρ̂ = Σ (x_t − x̄)(x_{t−1} − x̄) / Σ (x_t − x̄)²`.

**Cechy cykli (S3, `cechy_okien_s3()`).** Okres cyklu P jest znany. W każdym oknie dopasowujemy metodą najmniejszych
kwadratów `x_t ≈ c + a·sin(2πt/P) + b·cos(2πt/P) = c + R·sin(2πt/P + φ)`, gdzie t to **indeks w całym szeregu** (dzięki
temu fazy różnych okien są porównywalne). Amplituda `R = √(a² + b²)`, faza `φ = atan2(b, a)`. Cecha fazy to
przesunięcie `Δφ = φ − φ_ref` zawinięte do (−π, π], gdzie φ_ref – kierunek mediany wektorów (cos φ, sin φ) wszystkich
okien (faza typowa dla szeregu; fazy nie można uśredniać zwykłą średnią, bo π i −π to ten sam kąt). Częstotliwość –
maksimum periodogramu okna (FFT z dopełnieniem zerami do 1 024 punktów), szukane w paśmie 0,25–4 cykli na P: bez
tego ograniczenia przy słabym cyklu lub krótkim oknie maksimum trafiało czasem w szum o wysokiej częstotliwości.
Korelacja z wzorcem – współczynnik korelacji Pearsona okna z `sin(2πt/P + φ_ref)`: ≈ 1 dla typowego cyklu, maleje
przy przesunięciu fazy, zmianie okresu i przy słabym cyklu (więcej szumu względem sygnału).

Cechy mają różne jednostki, więc przed liczeniem odległości każdą kolumnę **standaryzujemy odpornie**:
`(cecha − mediana) / (1,4826 · MAD)`, gdzie MAD to mediana odchyleń bezwzględnych od mediany. Mediana i MAD
(w przeciwieństwie do średniej i odchylenia standardowego) nie dają się „przesunąć” samym anomaliom – czynnik 1,4826
sprawia, że dla rozkładu normalnego MAD·1,4826 = σ.

**Skala szumu szeregu (`odporna_skala()`).** Metody statystyczne (CUSUM, STL + EWMA, metody na resztach AR, PELT przy
podziale na odcinki) i autoenkoder dzielą szereg (albo reszty) przez odchylenie standardowe szumu σ̂. Liczymy je jako **pierwiastek
z mediany wariancji kolejnych, nienakładających się okien po 50 punktów** (z poprawką na to, że mediana rozkładu χ²
jest nieco mniejsza od średniej). Pierwsza wersja używała 1,4826·MAD całego szeregu – okazało się, że długie anomalie
zbiorowe (do 20% punktów) wprawdzie nie „psują” MAD, ale go **zawyżają**: mediana |x − μ| przesuwa się w stronę
wyższego kwantyla. Przy L = 100 w S1 wariancja normy po standaryzacji wynosiła ok. 0,86 zamiast 1, przy L = 200
w S3 – 0,73. Skutki: karta zależności CUSUM dryfowała w górę na zwykłych fragmentach (dwa razy więcej fałszywych
alarmów w szeregach z długimi anomaliami), a PELT dawał podwyższony wynik długim odcinkom normy (niskie AUC; PELT
porównuje teraz odcinki z tłem – rozdz. 3).
Okno z anomalią to tylko jedno z ok. 60 okien, więc mediana wariancji okien praktycznie się nie zmienia.
Poprawka wprowadzona 2026-10-27, wszystkie scenariusze przeliczone ponownie (porównanie przed / po:
[wyniki_s1.md](wyniki_s1.md), rozdz. 7).

### 0.4 Dopasowania trywialne

Dwa okna przesunięte o kilka punktów mają prawie te same wartości, więc są do siebie bardzo podobne. Gdyby metoda
porównywała okno z jego „sąsiadami w czasie”, każde okno – także anomalne – miałoby bliźniaka i nic by nie odstawało.
Dlatego w kNN i LOF przy szukaniu sąsiadów **pomijamy okna, które nakładają się w czasie** (|Δstart| < w)
– funkcja `odleglosci_bez_trywialnych()`. Matrix Profile ma własną, standardową strefę wykluczenia (rozdz. 4).

### 0.5 Model AR(1) i reszty (S2)

W S2 szereg ma zależność: każdy punkt zależy od poprzedniego. Wiele metod S2 nie patrzy na sam szereg, tylko na
**reszty** – to, czego model zależności nie przewidział (`dopasuj_ar1()` w `baza.py`):

1. dopasowanie regresji metodą najmniejszych kwadratów: `x_t = c + φ̂·x_{t−1} + e_t`,
2. reszty `e_t = x_t − c − φ̂·x_{t−1}`, poziom równowagi `μ̂ = c / (1 − φ̂)`,
3. odporne oszacowanie skali reszt `σ̂_e` – mediana wariancji okien reszt (rozdz. 0.3) – szok czy anomalia go nie zawyżą.

Dla szeregu AR(1) bez anomalii reszty są (prawie) białym szumem – więc działają na nich narzędzia z S1. Model
dopasowujemy do **całego** szeregu z anomaliami (nie mamy osobnych danych uczących); anomalie zajmują kilka procent
punktów, więc model jest prawie taki sam jak dla szeregu czystego.

### 0.6 Usunięcie cyklu: STL i reszty (S3)

W S3 szereg to cykl (sinus o okresie P = 50) plus szum. Metody statystyczne S3 – jak w S2 – patrzą na **reszty**:
to, co zostaje po usunięciu cyklu (`reszty_stl()` w `baza.py`). **STL** (*Seasonal-Trend decomposition using Loess*,
Cleveland i in., 1990) rozkłada szereg na `x_t = trend_t + sezonowość_t + reszta_t` w pętli:

1. **wygładzanie podszeregów cyklu** – dla każdej pozycji w cyklu (np. wszystkie punkty t, t + P, t + 2P, …)
   wartości wygładzamy regresją lokalną (loess) wzdłuż kolejnych cykli; długość tego wygładzania (`seasonal`,
   w cyklach) decyduje, jak szybko sezonowość może się zmieniać w czasie,
2. filtr dolnoprzepustowy usuwa z sezonowości to, co należy do trendu,
3. **trend** – loess szeregu po odjęciu sezonowości,
4. kroki 1–3 powtarzamy; wersja **odporna** (`robust=True`) w zewnętrznej pętli nadaje punktom o dużych resztach małe
   wagi, żeby odstające fragmenty nie przesuwały trendu i sezonowości.

**Długość wygładzania sezonowości:** domyślnie w `statsmodels` 7 cykli – sezonowość może wtedy „podążać” za
anomalią trwającą kilka cykli i ją wchłonąć (w reszcie nic nie zostaje). Cykl tła S3 jest stały, więc w projekcie
przyjęto okno **dłuższe niż cały szereg**: `2·(n/P) + 1 = 121` cykli – sezonowość jest wtedy prawie stała w czasie.
Porównanie obu ustawień i globalnego dopasowania sinusa: analiza dodatkowa S3 (a) ([wyniki_s3.md](wyniki_s3.md), rozdz. 5).

Reszty standaryzujemy odpornie (mediana i mediana wariancji okien, rozdz. 0.3). Dla szeregu bez anomalii są one prawie białym szumem
(odch. std ≈ 1), a każda anomalia S3 – zmiana fazy, amplitudy albo okresu – zostawia w nich fragment o **większej
energii** (rozdz. 9).

### 0.7 Kalibracja i próg (wspólne dla wszystkich metod)

Dwa etapy, oba wyłącznie na **szeregach bez anomalii** (200 szeregów; `src/ewaluacja/metryki.py`):

1. **Stałe metody** – np. skale kart CUSUM, kara PELT (opisane przy metodach).
2. **Próg alarmu** – kwantyl 90% maksimów wyniku metody na szeregach bez anomalii. Wtedy fałszywy alarm pojawia się
   w ok. 10% szeregów bez anomalii – **dla każdej metody tak samo**, więc porównanie czułości jest uczciwe.
   Próg sprawdzany jest na 200 nowych szeregach (S1: 4–16,5%, S2: 6,5–13,5%, S3: 6,5–16,5% – wszystkie warianty
   metod), a dla części głównej każdego scenariusza dodatkowo na 1 000 nowych szeregach (`src/ewaluacja/walidacja_progow.py`,
   `wyniki/walidacja_progow.md`): S1 5,7–15,7%, S2 4,0–12,3%, S3 7,2–13,1%. Najdalej od celu jest PELT w S2 (4,0% – kara całkowita, a odsetek fałszywych alarmów stromo zależy od kary); szczegóły w rozdz. 8 wyników każdego scenariusza.

Ocena (te same zasady we wszystkich scenariuszach): anomalia jest **wykryta**, gdy choć jeden alarm leży w jej strefie
(fragment ± 25 punktów); epizod alarmu poza wszystkimi strefami to **fałszywy alarm**; **AUC** mierzy, czy punkty
anomalii mają wyższy wynik niż punkty normy, niezależnie od progu.

### 0.8 Na czym uczą się metody ML: ten sam szereg albo szeregi bez anomalii

Metody uczenia maszynowego (Isolation Forest, LOF, One-Class SVM, autoenkoder) występują w dwóch wariantach
(ustalenie z konsultacji 2026-10-27 – w każdym scenariuszu oba):

- **uczenie na tym samym szeregu** (wersja główna, nienadzorowana): model uczy się na oknach szeregu, w którym szuka
  anomalii. Nie wymaga żadnych danych „wzorcowych”, ale anomalie są w danych uczących – długa anomalia daje wiele
  podobnych okien, które model uznaje za normę (**maskowanie**, najsilniejsze w Isolation Forest);
- **uczenie na szeregach bez anomalii** (półnadzorowane, *novelty detection*): model uczy się raz, na oknach
  **20 osobnych szeregów bez anomalii** tego samego rodzaju, a potem ocenia okna badanego szeregu. To standardowy
  sposób użycia One-Class SVM, LOF w trybie *novelty* i autoenkoderów (Schölkopf i in., 2001; Breunig i in., 2000;
  Chandola i in., 2009). W praktyce wymaga fragmentu danych, o którym wiadomo, że jest „czysty”.

Szczegóły wariantu uczonego na szeregach bez anomalii:
- szeregi uczące mają **własne ziarna** (9 500 000 + i), inne niż 200 szeregów kalibracji progu – próg nie jest
  ustalany na danych, na których uczył się model (inaczej wynik szeregów kalibracyjnych byłby zaniżony, a próg za niski);
- okna uczące **bez nakładania się** (krok = długość okna; 20 × 60 = 1 200 okien) – sąsiednie, prawie identyczne okna
  tego samego szeregu zaniżałyby odległości w zbiorze uczącym (ważne dla LOF);
- cechy każdego szeregu (uczącego i badanego) standaryzowane tak samo jak w wersji głównej – odpornie, w obrębie szeregu;
- autoenkoder uczony bez odrzucania najgorzej odtwarzanych okien (dane uczące nie mają anomalii);
- próg alarmu – tak samo jak dla każdej metody (fałszywy alarm w 10% szeregów bez anomalii).

---

## 1. CUSUM – karta sum skumulowanych

*Grupa: statystyczne. Scenariusz: S1. Kod: `CUSUM` w `src/metody/statystyczne.py`.*

### Intuicja

Pojedynczy punkt zmienionego szeregu zwykle niczym się nie wyróżnia – przesunięcie poziomu o 1σ ginie w szumie.
Ale jeśli przez 50 punktów z rzędu wartości są średnio o 1σ za wysoko, to **suma** tych nadwyżek rośnie systematycznie,
podczas gdy dla szumu sumuje się do zera. CUSUM (Page, 1954) to właśnie suma nadwyżek z „hamulcem”, który nie pozwala
jej spaść poniżej zera.

### Jak działa

Dla statystyki `z_t` o średniej 0 w normie (np. wystandaryzowanego szeregu):

```
S⁺_t = max(0, S⁺_{t−1} + z_t − k)      karta wzrostu
S⁻_t = max(0, S⁻_{t−1} − z_t − k)      karta spadku
```

- Gdy `z` ma średnią 0, każdy krok obniża sumę średnio o `k`, więc karta trzyma się przy zerze.
- Gdy `z` ma średnią δ > k, karta rośnie średnio o (δ − k) na punkt – po L punktach ≈ (δ − k)·L.
- **Wartość odniesienia k** to „próg czułości”: zmiany mniejsze niż k karta ignoruje. Klasyczna reguła: k = połowa
  najmniejszej zmiany, którą chcemy szybko wykryć.

**Wersja offline z lokalizacją** (`_lokalizuj()`): karta tworzy „wycieczki” – ciągłe odcinki S > 0. W każdej wycieczce
zmiana zaczyna się w ostatnim zerze przed nią, a kończy w szczycie karty (klasyczne oszacowanie punktu zmiany).
Punkty od początku do szczytu dostają wysokość szczytu; punkty **po** szczycie (karta tylko opada) – zero. Bez tego
karta po silnej anomalii opadałaby tylko o k na punkt i „ciągnęła” alarm setki punktów za anomalią (wykryte w S1 –
dawało to 0,84 fałszywego alarmu na szereg z anomaliami).

### Trzy karty dla trzech anomalii S1

| Karta | Statystyka z_t (średnia 0, odch. 1 w normie) | Strona | k | Anomalia |
|---|---|---|---|---|
| poziom | `(x_t − μ̂) / σ̂` | obie | 0,5 | S1-A zmiana poziomu |
| zmienność | `zd_t = ((x_t − x_{t−1})² / (2σ̂²) − 1) / √2` | wzrost | 0,5 | S1-B zmiana zmienności |
| zależność | `−zd_t` | wzrost | 0,15 | S1-C pojawienie się zależności |

Dlaczego zmienność i zależność liczymy na **różnicach sąsiednich punktów**: dla białego szumu
`E[(x_t − x_{t−1})²] = 2σ²`; wzrost zmienności (×c) podnosi to do 2c²σ², a zależność AR(1) obniża do 2σ²(1 − φ),
czyli `E[−zd] = φ/√2`. Jedna statystyka, dwie strony. A zmiana poziomu różnic **nie zmienia** – więc karty
zmienności i zależności nie reagują na anomalię A. (Pierwsza wersja liczyła zależność z iloczynu x_t·x_{t−1}, który
przy przesunięciu o Δ rośnie o ≈ Δ² – karta zależności po silnej anomalii A opadała przez ~1 500 punktów i zasłaniała
anomalię C; po zmianie wykrywanie C przy φ = 0,9 wzrosło z 51% do 95%.)

μ̂ i σ̂ szacowane odpornie (mediana; σ̂ – mediana wariancji okien, rozdz. 0.3). **Łączenie kart:** karty mają różne rozkłady (karta zmienności
jest skośna), więc każdą dzielimy przez medianę jej maksimum na 200 szeregach bez anomalii (`kalibruj`), a wynik
punktu to maksimum z trzech przeskalowanych kart.

### Mocne i słabe strony (wyniki)

- ✔ **Najlepsza metoda w S1** (wykrywa 76% / 84% / 39% anomalii A / B / C) – bo każda karta patrzy dokładnie na to,
  co zmienia dana anomalia; najlepsza dokładność lokalizacji (F1 0,65).
- ✔ Bardzo szybka (≈ 0,02 s na szereg 3 000 punktów), przejrzysta – wiadomo, która karta zaalarmowała.
- ✘ Wymaga wiedzy, **czego** szukamy (każda karta = jedna hipoteza o anomalii) i statystyk, które nie „przeciekają”
  między rodzajami anomalii.
- ✘ Alarm bywa przesunięty przed początek anomalii (karta rośnie od ostatniego zera) – precyzja 0,89.

---

## 2. Reszty modelu AR(1) + CUSUM

*Grupa: statystyczne. Scenariusz: S2. Kod: `ResztyARCUSUM` w `src/metody/statystyczne.py`.*

### Intuicja

W szeregu AR(1) naturalne, długie wahania wyglądają jak zmiany poziomu – CUSUM na samym szeregu myliłby je
z anomaliami. Najpierw więc „odejmujemy” znaną zależność: dopasowujemy model AR(1) i patrzymy na reszty (rozdz. 0.5).
Reszty są białym szumem, a anomalie S2 zostawiają w nich ślady, które już umiemy wykrywać narzędziami z S1.

### Jak działa

1. Dopasowanie AR(1), reszty `r_t = e_t / σ̂_e`.
2. **Karty CUSUM z S1 liczone na resztach** (poziom, zmienność, zależność):
   - przesunięcie równowagi o Δ (S2-C) daje w resztach serię o średniej `(1 − φ)·Δ` – dla φ = 0,7 tylko 30% Δ,
     dlatego karta poziomu ma tu mniejsze k = 0,25;
   - zmiana φ (S2-B) zmienia wariancję i autokorelację reszt (np. φ → 0 przy stałej wariancji szeregu: wariancja
     reszt rośnie 2,9 raza);
3. **Test szoku** (*innovational outlier*): `|r_t|` – szok to w resztach jedna ogromna wartość.
4. Każda składowa dzielona przez medianę jej maksimum na szeregach bez anomalii; wynik = maksimum składowych.

### Mocne i słabe strony (wyniki S2)

- ✘ **Uczony na tym samym szeregu – najsłabszy w S2** (4 / 17 / 0%, 5 cech; AUC 0,75–0,81); przy
  najsilniejszej zmianie φ (φ' = −0,5) wykrywa rzadziej niż przy słabszej (31% vs
  64%, L = 50). Wrażliwy na inne anomalie w szeregu (szok: 11% /
  0% / 0% przy L = 20 / 50 / 100).
- ✔ **Uczony na szeregach bez anomalii – jedna z lepszych metod S2** (41 / 60 / 38%, F1 0,04 →
  0,44); wynik rośnie monotonicznie z intensywnością (φ' = 0 / −0,5: 96% /
  100%). Słabość wersji głównej brała się więc głównie z anomalii w danych uczących (granica
  modelu obejmowała także ich okna), a nie tylko z nasycenia jądra. To standardowy sposób użycia One-Class SVM.
- Zestaw cech: 5 cech lepsze dla zmiany φ (uczony na szeregach bez anomalii: 48% → 60%).

---

## 3. PELT – optymalny podział na odcinki

*Grupa: statystyczne. Scenariusze: S1 (`PELT`), S2 (`PELTAR`), S3 (`PELTSTL`). Kod: `pelt_ogolny`, `PELT`, `PELTAR`,
`PELTSTL` w `src/metody/statystyczne.py`.*

### Intuicja

Zamiast oceniać punkty, PELT pyta: **jak najlepiej podzielić cały szereg na odcinki**, z których każdy ma stałe
parametry (np. średnią i wariancję)? Każdy dodatkowy odcinek kosztuje karę – więc szereg zostanie podzielony tylko
tam, gdzie zmiana jest wyraźna. Anomalia zbiorowa ma początek i koniec, więc idealnie pasuje do tego modelu.
To metoda typowo offline (potrzebuje całego szeregu).

### Jak działa

Szukamy podziału na odcinki minimalizującego

```
Σ_odcinki C(odcinek)  +  β · (liczba odcinków)
```

gdzie `C` – koszt dopasowania modelu do odcinka (im gorzej pasuje, tym większy), `β` – kara za odcinek.
Programowanie dynamiczne: `F(t) = min_{s<t} [ F(s) + C(x_{s:t}) + β ]` – najlepszy koszt podziału pierwszych t
punktów. **Przycinanie** (Killick i in., 2012): kandydat s, dla którego `F(s) + C(s, t) > F(t)`, nigdy już nie da
lepszego podziału i jest usuwany – dzięki temu algorytm działa w czasie ≈ liniowym zamiast kwadratowego.
Granice rozważane co 5 punktów, minimalna długość odcinka 10.

**Koszty:**

| Wersja | Koszt odcinka długości n | Co wykrywa |
|---|---|---|
| S1 – normalny | `n · ln σ̂²_odcinka` dla x oraz dla x_t·x_{t−1} (suma) | zmiana średniej lub wariancji x (A, B); zmiana średniej iloczynu sąsiadów = zależność (C) |
| S2 – AR(1) | `n · ln σ̂²_reszt` regresji `x_t = c + φ·x_{t−1}` dopasowanej **osobno na odcinku** | odcinki z innym φ (B), innym poziomem (C), ogromną wariancją reszt (szok A) |
| S3 – normalny na resztach STL | `n · ln σ̂²_odcinka` standaryzowanych reszt STL (rozdz. 0.6) | odcinki reszt o większej energii – każda anomalia S3 (faza, amplituda, okres) |

`n · ln σ̂²` to (z dokładnością do stałej) −2·log-wiarygodność modelu normalnego. Sumy skumulowanych wartości,
kwadratów i iloczynów pozwalają policzyć koszt dowolnego odcinka w stałym czasie. Własna, wektorowa implementacja
daje identyczne granice jak biblioteka `ruptures` (sprawdzone), ok. 140× szybciej.

**Z granic do wyniku punktu.** Wynik punktu = jak bardzo **jego odcinek** odbiega od **tła** – maksimum statystyk typu z
(≈ N(0,1) dla odcinka bez anomalii). Tło (`maska_tla()`) to najdłuższe odcinki, które razem obejmują co najmniej
połowę szeregu: anomalie są krótkie (≤ 200 punktów), a odcinki normy między nimi długie, więc tło to norma. Na tle
dopasowujemy parametry odniesienia (μ_t, σ_t, ρ_t albo model AR(1) tła: c_t, φ_t, σ_t):

- S1: `|średnia − μ_t|·√n/σ_t`, `|ln(wariancja/σ_t²)|·√(n/2)`, `|autokorelacja − ρ_t|·√n`,
- S2 (reszty e modelu AR tła): `|średnia e|·√n/σ_t`, `|ln(war. e/σ_t²)|·√(n/2)`, `|ac₁ − φ_t|·√n/√(1−φ_t²)`,
- S3 (standaryzowane reszty STL r): `|średnia r − μ_t|·√n/σ_t`, `|ln(war. r/σ_t²)|·√(n/2)`.

Odcinek dostaje wynik tylko wtedy, gdy jest **istotnie** inny od tła: z > 3 (reguła „trzech sigm”), w przeciwnym razie 0.

**Kalibracja kary β:** najmniejsza kara całkowita od 3 do 300 (szukana bisekcją), przy której odcinek istotnie inny od tła pojawia się
w ≤ 10% szeregów bez anomalii – ta sama zasada fałszywych alarmów co dla innych metod, realizowana przez naturalny
„suwak” czułości PELT.

**Dlaczego tak (poprawka 2026-10-27).** Pierwsza wersja porównywała odcinek z parametrami **całego** szeregu (σ̂ z MAD,
model AR z całego szeregu), a karę dobierała tak, żeby średnio było < 0,5 fałszywej granicy na szereg. Parametry
całego szeregu są zaburzone przez anomalie, a statystyki z rosną jak √n – mały błąd odniesienia dawał wysoki wynik
**długim odcinkom normy**: w S1 przy L = 100 nad progiem leżało 44% punktów normy, a część „wykrytych” anomalii brała
się stąd, że taki odcinek obejmował ich strefę. Porównanie z tłem i próg istotności to usuwają (punkty normy nad progiem:
1–3%, precyzja 0,90–0,95 w sprawdzeniu na szeregach spoza eksperymentu).

### Mocne i słabe strony (wyniki)

- ✔ Wskazuje początek i koniec anomalii; mało fałszywych alarmów w szeregach z anomaliami (0,04 na szereg w S1 i S2).
- ✔ W S2 **najlepsza metoda dla wzrostu zależności** (φ' = 0,95: 98%) – osobny model AR na odcinku dokładnie
  odpowiada tej anomalii.
- ✘ Krótkie anomalie przegrywają z karą: szoku w S2 prawie nie wykrywa (23%, AUC 0,56) – odcinek 10 punktów rzadko
  „opłaca się” wydzielić. Precyzja 0,83–0,88 – odcinki bywają szersze niż anomalia.
- ✘ Słabszy od CUSUM dla zależności w S1 (19% vs 39%).
- ✔ S3 (na resztach STL): wykrywa 84% / 72% / 93% anomalii fazy / amplitudy / okresu przy bardzo małej liczbie
  fałszywych alarmów (0,05 na szereg) i precyzji 0,93 – drugie F1 w S3 (0,73). Najodporniejszy na słaby cykl
  (A = 1: faza 84%, okres 96%). AUC (0,79–0,86) niższe, niż sugeruje odsetek wykrytych – odcinki nieistotnie różne
  od tła dostają wynik 0, tak jak norma.

---

## 4. Matrix Profile – fragmenty bez „bliźniaka”

*Grupa: fragmenty szeregu (podobieństwo). Scenariusze: S1 (na szeregu), S2 (na resztach AR), S3 (z normalizacją,
okno = okres cyklu). Kod: `MatrixProfile` w `src/metody/fragmenty.py`, biblioteka `stumpy`.*

### Intuicja

Normalny fragment szeregu ma gdzieś w szeregu „bliźniaka” – inny, podobny fragment. Fragment anomalny jest jedyny
w swoim rodzaju, więc nawet jego najbliższy odpowiednik jest daleko. **Matrix Profile** (Yeh i in., 2016) dla
każdego fragmentu podaje odległość do najbardziej podobnego innego fragmentu; fragment z największą wartością to
**discord** – najbardziej nietypowy kawałek szeregu. Metoda nie potrzebuje modelu ani założeń o rozkładzie.

### Jak działa

Dla każdego fragmentu `T_i` długości m (= w = 50):

```
P[i] = min_{j : |i − j| ≥ m/4}  d(T_i, T_j)
```

- `d` – odległość euklidesowa. **Bez normalizacji** (`stumpy.aamp`) porównujemy surowe wartości; **z normalizacją**
  (`stumpy.stump`) każdy fragment jest najpierw standaryzowany (średnia 0, odch. 1), więc porównywany jest tylko
  **kształt** (wtedy `d = √(2m(1 − ρ))`, ρ – korelacja fragmentów).
- **Strefa wykluczenia** |i − j| < m/4 – pomijamy fragmenty, które prawie pokrywają się z T_i (dopasowania trywialne);
  m/4 to standard biblioteki i literatury (sprawdzono, że strefa = m zmienia wyniki nieznacznie).
- Obliczenie wszystkich P[i] to O(n²) porównań – biblioteka `stumpy` liczy je wydajnie (≈ 0,1–0,7 s na szereg).
- W S2 Matrix Profile liczony jest na **resztach AR** (rozdz. 0.5): naturalne wahania AR znikają, szok to jedna duża
  reszta, zmiana φ – fragment reszt o innej wariancji.
- W S3 okno m = P = 50 (jeden cykl) i wersja **z normalizacją** – porównywany jest kształt cyklu; wersja bez
  normalizacji liczona porównawczo.
- Wynik P[i] przypisywany punktom przez środki fragmentów (rozdz. 0.2).

### Mocne i słabe strony (wyniki)

- ✔ **Najlepszy do zmiany zmienności w S1** (91%, AUC 0,99; najsłabsza zmiana c = 1,5: 70%, inne metody 25–60%)
  i do spadku zależności w S2 (97% przy φ' = 0).
- ✘ **Wersja z normalizacją nie widzi żadnej anomalii S1** (≤ 1%, AUC ≈ 0,5) – standaryzacja usuwa dokładnie to,
  co zmieniają anomalie A i B (średnią i skalę). Dlatego w S1–S2 używana jest wersja bez normalizacji.
- ✘ **Ślepy na fragmenty „gładsze / mniej zmienne”**: S1-C (zależność; AUC 0,46) i wzrost φ w S2 (AUC 0,03 –
  ranking odwrócony). Fragment o mniejszej zmienności lub mniejszej liczbie „stopni swobody” ma **bliższych**
  sąsiadów, a discord szuka fragmentów **dalekich** od wszystkich – z zasady przeciwnego zjawiska.
- ✘ Słabo wykrywa przesunięcie poziomu na resztach AR (24% w S2) – w resztach zostaje tylko (1 − φ)·Δ.
- S3 (z normalizacją, okno = cykl): **najlepszy do zmiany amplitudy** (87%, AUC 0,96 – na równi z kNN na cechach
  cykli, 86%; przy słabym cyklu A = 1 – jedyna metoda wykrywająca zanik cyklu, 91%). Hipoteza z planu, że normalizacja
  ukryje zmianę amplitudy, **nie potwierdziła się** – słabszy cykl to większy udział szumu w oknie, więc po
  standaryzacji kształt jest „poszarpany” i nie ma bliźniaka (bez normalizacji: 70%).
- ✘ S3: **wzmocnienia cyklu nie widzi wcale** (×1,5 i ×2: 1%, AUC 0,05–0,09 – ranking odwrócony): silniejszy cykl
  po standaryzacji jest „czystszym” sinusem, bliżej wszystkich innych okien. Ten sam mechanizm co S1-C i wzrost φ
  w S2 – trzeci przykład ślepoty na fragmenty „gładsze” niż reszta. Bez normalizacji wzmocnienie jest widoczne (92–100%).
- ✘ S3: **przesunięcie fazy widzi tylko na brzegach** (61%, AUC 0,68). Wewnątrz przesuniętego fragmentu każde okno
  to zwykły sinus – a sinus o tej samej fazie występuje w każdym innym cyklu szeregu, więc okno ma bliźniaka.
  Wynik wewnątrz fragmentu = 0,98 × wynik normy, na brzegach (skok fazy) – 1,4 ×.

---

## 5. Cechy okien + kNN

*Grupa: fragmenty szeregu (charakterystyki). Scenariusze: S1, S2, S3 („cechy cykli + kNN”). Kod: `CechyKNN`
w `src/metody/fragmenty.py`.*

### Intuicja

Każde okno opisujemy kilkoma liczbami (cechami, rozdz. 0.3) – staje się punktem w kilkuwymiarowej przestrzeni.
Normalne okna tworzą gęstą chmurę; okno anomalne leży z boku. **Miarą nietypowości jest odległość do najbliższych
sąsiadów** (Ramaswamy i in., 2000; Angiulli i Pizzuti, 2002): okno, którego k najbliższych sąsiadów jest daleko,
jest nietypowe.

### Jak działa

1. Okna długości w = 50 co 5 punktów; dla każdego okna wektor cech, standaryzowany odpornie (rozdz. 0.3).
2. Odległości euklidesowe między wszystkimi oknami; okna nakładające się w czasie wykluczone (rozdz. 0.4).
3. Wynik okna = **średnia odległość do k = 10 najbliższych** (dozwolonych) okien.
4. Wynik punktów – interpolacja między środkami okien.

### Mocne i słabe strony (wyniki)

- ✔ **Interpretowalna**: wiadomo, która cecha odstaje; działa na wszystkie anomalie, o ile mają „swoją” cechę.
- ✔ Odporna na maskowanie (dzięki wykluczeniu okien nakładających się) – w przeciwieństwie do Isolation Forest.
- ✔ **S1, 3 cechy – na równi z CUSUM** (77 / 87 / 42% wobec 76 / 84 / 39%; AUC
  0,85–0,98).
- ✔ **Najlepsza do szoku w S2** (42%; impuls 5σ_ε: 60%) – łączy maksimum
  reszty z innymi cechami okna.
- ✔ **Najlepsza metoda w S3** (cechy cykli, 3 cechy: 92 / 87 / 96% anomalii fazy / amplitudy / okresu, AUC
  ≥ 0,98, F1 0,79). Cechy opisują dokładnie parametry cyklu,
  a faza okna jest liczona względem **globalnego** czasu – więc przesunięty fragment odstaje na całej długości.
- ✘ **Bardzo wrażliwa na dobór cech – i to w różną stronę w różnych scenariuszach.** S1: 5 cech → 3 cechy – zależność
  11% → 42% (kurtoza i rozstęp dokładają tylko losowy rozrzut). S2: odwrotnie –
  zmiana φ 48% (3 cechy) → 60% (5 cech), wzrost φ 8% →
  87% (odch. std reszt AR niesie informację o zmianie dynamiki). S3: bez różnicy
  (91 / 86 / 96% na 5 cechach). Cechy z S1 zastosowane w S2 dają 16% wykrytych zmian φ.
- ✘ S3: **wymaga dobrze określonych cech.** Przy oknie krótszym niż cykl (w = 25) albo słabym cyklu (A = 1) faza
  i częstotliwość pojedynczego okna są niepewne, w szeregach bez anomalii zdarzają się okna o skrajnych cechach i próg
  wypada bardzo wysoko (w = 25: 1 / 0 / 2%; A = 1, faza:
  0%) przy wysokim AUC.
- ✘ Złożoność O(liczba okien²) – przy oknach co 5 punktów (≈ 600 okien) to ≈ 0,02 s na szereg.

---

## 6. Isolation Forest – las izolacyjny

*Grupa: uczenie maszynowe. Scenariusze: S1, S2, S3. Kod: `IsolationForest` w `src/metody/ml.py` (scikit-learn).*

### Intuicja

Zamiast opisywać, jak wygląda „normalność”, Isolation Forest (Liu, Ting, Zhou, 2008) sprawdza, **jak łatwo dany punkt
oddzielić od reszty**. Losowo wybieramy cechę i losowy próg cięcia, dzielimy dane na dwie części i powtarzamy.
Punkt nietypowy – leżący z dala od chmury – zostaje odizolowany (sam w swojej części) po kilku cięciach;
punkt w gęstym środku wymaga wielu cięć.

### Jak działa

1. Dane = wektory cech okien (te same cechy co w kNN – porównujemy sposób wykrywania, a nie dobór cech).
2. Budujemy 200 drzew izolacyjnych; każde na losowej podpróbie ψ = min(256, liczba okien) okien, dzieląc losowymi
   cięciami, aż każde okno jest samo.
3. `h(x)` – liczba cięć potrzebna do odizolowania okna x w drzewie; `E[h(x)]` – średnia po drzewach.
4. Wynik: `s(x) = 2^{−E[h(x)] / c(ψ)}`, gdzie `c(ψ)` – średnia długość ścieżki dla ψ losowych punktów (normalizacja).
   s ≈ 0,5 – okno typowe, s → 1 – okno łatwe do odizolowania (nietypowe). W kodzie: `−score_samples()` = s(x).
5. Ziarno losowania stałe – wynik powtarzalny.

### Mocne i słabe strony (wyniki)

- ✔ Bardzo dobry **ranking** (AUC 0,84–0,98 w S1, 0,80–0,90 w S2, 0,98–0,99 w S3; 5 cech),
  szybki, bez założeń o rozkładzie.
- ✘ **Maskowanie** (wersja uczona na tym samym szeregu): gdy anomalia daje wiele podobnych nietypowych okien, tworzą one
  własną „gromadę” i przestają być łatwe do odizolowania. S1, Δ = 3σ: 99% wykrytych dla L = 20,
  24% dla L = 100. S2 – maskowanie między anomaliami: szok (zawsze 10 punktów) wykrywany w
  17% / 5% / 1% przy pozostałych anomaliach długości
  20 / 50 / 100. S3 – najsilniejsze: przesunięcie fazy 60% / 26% /
  8% przy L = 50 / 100 / 200.
- ✔ **Uczenie na szeregach bez anomalii usuwa większość maskowania** (rozdz. 0.8): S1 26 / 75 / 2% → 53 / 77 / 19%,
  S2 8 / 48 / 1% → 11 / 54 / 16%, S3 31 / 25 / 78% → 76 / 82 / 94% (A / B / C, 5 cech). Największy zysk
  przy długich anomaliach (S1, zmiana poziomu przy L = 100: 8% → 53%).
- ✘ Wynik ma mały zakres (≈ 0,4–0,8), a próg wypada blisko maksimum – mało wykrytych przy dobrym AUC.
- Zestaw cech: w S1 3 cechy lepsze dla zmiany poziomu (26% → 40%), gorsze dla zmiany
  zmienności (75% → 45%) – rozstęp jest dla drzew silnym sygnałem zmienności.

---

## 7. LOF – Local Outlier Factor

*Grupa: uczenie maszynowe. Scenariusz: S1. Kod: `LOF` w `src/metody/ml.py` (scikit-learn).*

### Intuicja

kNN mierzy odległość do sąsiadów **bezwzględnie**. LOF (Breunig i in., 2000) porównuje **gęstość** wokół punktu
z gęstością wokół jego sąsiadów: punkt jest nietypowy, jeśli leży w miejscu rzadszym niż jego sąsiedzi. Dzięki temu
radzi sobie z danymi, w których gęstość chmury jest różna w różnych miejscach.

### Jak działa

Dla k = 10 sąsiadów (te same cechy i to samo wykluczenie okien nakładających się co w kNN):

1. `d_k(b)` – odległość punktu b do jego k-tego sąsiada.
2. Odległość osiągalności: `reach_k(a, b) = max(d_k(b), d(a, b))` – wygładza odległości w gęstych obszarach.
3. Lokalna gęstość osiągalności: `lrd(a) = 1 / średnia_{b ∈ N_k(a)} reach_k(a, b)`.
4. `LOF(a) = średnia_{b ∈ N_k(a)} lrd(b) / lrd(a)` – stosunek gęstości sąsiadów do gęstości punktu.
   LOF ≈ 1 – punkt typowy, LOF ≫ 1 – punkt w rzadszym obszarze niż sąsiedzi.

### Mocne i słabe strony (wyniki S1)

- Wariant uczony na szeregach bez anomalii: LOF w trybie *novelty* (`novelty=True`) – gęstość okna badanego szeregu
  porównywana z gęstością okien szeregów uczących (bez wykluczania sąsiadów, bo to inne szeregi).
- ✔ Na 3 cechach dobry dla zmiany poziomu i silnej zależności (L = 50: Δ = 1σ 94%, φ = 0,9
  97%).
- ✘ Tło białego szumu jest jednorodne, więc lokalna gęstość nie daje przewagi nad prostym kNN – średnio podobnie lub
  gorzej (68 / 78 / 36% wobec kNN 77 / 87 / 42%, 3 cechy).
- ✘ Lekkie maskowanie przy długich anomaliach (zmiana poziomu, 5 cech: 72% przy L = 50,
  65% przy L = 100); po uczeniu na szeregach bez anomalii – 84% i
  89%, F1 0,40 → 0,63.

---

## 8. One-Class SVM

*Grupa: uczenie maszynowe. Scenariusz: S2. Kod: `OneClassSVM` w `src/metody/ml.py` (scikit-learn).*

### Intuicja

One-Class SVM (Schölkopf i in., 2001) uczy się **granicy** obszaru, w którym leży większość danych. Wszystko poza
granicą jest nietypowe, a im dalej od niej, tym bardziej. W przeciwieństwie do drzew (Isolation Forest) granica jest
gładka.

### Jak działa

1. Dane = wektory cech okien S2 (5 albo 3 cechy).
2. Funkcja jądra RBF `K(a, b) = exp(−γ·‖a − b‖²)` przenosi dane do przestrzeni, w której szukamy hiperpłaszczyzny
   oddzielającej dane od początku układu z maksymalnym marginesem:
   `min ½‖w‖² + 1/(ν·n)·Σ ξ_i − ρ` przy `⟨w, φ(x_i)⟩ ≥ ρ − ξ_i`.
3. **ν = 0,05** – górna granica odsetka okien, które model może zostawić poza granicą (≈ spodziewany odsetek
   nietypowych okien).
4. Funkcja decyzyjna `f(x) = Σ α_i K(x_i, x) − ρ` (> 0 wewnątrz, < 0 na zewnątrz); wynik okna = `−f(x)`.

**Nasycenie wyniku i wybór γ.** Dla okna daleko od danych wszystkie `K(x_i, x) ≈ 0`, więc `f(x) → −ρ` – wynik
**przestaje rosnąć**. Okno bardzo nietypowe dostaje prawie ten sam wynik co lekko nietypowe i nie przekracza progu.
Przy domyślnym γ = „scale” (≈ 1/liczba cech) nasycenie zaczyna się już przy odległości ≈ 2–3, a nietypowe okna odstają
o 5–10 jednostek. Dlatego przyjęto **γ = 0,01** (γ·d² rzędu 1 dla d ≈ 10). Porównanie (S2, analiza (c), 5 cech):
odsetek wykrytych zmian φ – 4% (scale), 51% (0,01), 71% (0,002), przy
prawie identycznym AUC.

### Mocne i słabe strony (wyniki S2)

- ✔ Dobry ranking (AUC 0,76–0,82); dobry dla wzrostu zależności (φ' = 0,95: 91%).
- ✘ **Najsłabsze wykrywanie w S2** (6% / 28% / 3%), a przy najsilniejszej zmianie φ (φ' = −0,5) wykrywa rzadziej
  (53%) niż przy słabszej (79%) – nawet przy γ = 0,01 najsilniejsze anomalie wchodzą w strefę nasycenia.
- ✘ Wrażliwy na parametry (γ, ν) i na inne anomalie w szeregu (szok: 18% / 1% / 0% przy L = 20 / 50 / 100).

---

## 9. STL + karta EWMA

*Grupa: statystyczne. Scenariusz: S3. Kod: `STLEWMA`, `ewma_dwustronna` w `src/metody/statystyczne.py`,
`reszty_stl` w `src/metody/baza.py`.*

### Intuicja

Szereg S3 to cykl plus szum. Jeśli odejmiemy od niego cykl, zostanie sam szum – o znanej, stałej „energii”
(wariancji). Każda anomalia S3 sprawia, że na fragmencie odjęty cykl **nie pasuje** do tego, co się naprawdę dzieje,
więc w resztach pojawia się dodatkowa energia. Karta kontrolna EWMA (Roberts, 1959) śledzi tę energię, wygładzając ją
wykładniczo – pojedyncze duże reszty szumu się uśredniają, a trwały wzrost energii przez kilkadziesiąt punktów –
nie.

### Jak działa

1. **Usunięcie cyklu** – reszty STL standaryzowane odpornie: `r_t` (rozdz. 0.6). W normie `r_t ≈ N(0, 1)`.
2. **Statystyka energii:** `z_t = r_t² − 1` – w normie średnio 0, na anomalii > 0.
3. **Karta EWMA w przód i w tył:** `E_t = λ·z_t + (1 − λ)·E_{t−1}` liczona od początku do końca szeregu i od końca
   do początku; wynik = **minimum** obu. Karta liczona w przód jest wysoka na anomalii i jeszcze przez kilkadziesiąt
   punktów za nią (opada wykładniczo), karta liczona w tył – na anomalii i przed nią; minimum jest wysokie tylko tam,
   gdzie obie – na samej anomalii (lokalizacja, jak w CUSUM, rozdz. 1). Pierwsza wersja brała średnią obu kart:
   po silnej anomalii „ogon” przekraczał próg poza strefą tolerancji (0,46 fałszywego alarmu na szereg z anomaliami;
   z minimum – 0,14), za cenę nieco niższego AUC (o 0,02–0,06) – poprawka 2026-10-27.
4. **λ = 2/(P + 1) ≈ 0,039** – „pamięć” karty ≈ jeden cykl. Reszta przesunięcia fazy jest sama okresowa
   (rozdz. niżej), więc krótsza pamięć dawałaby wynik falujący w rytm cyklu; dłuższa rozmywałaby alarm poza anomalię.

Ile energii zostawia w resztach każda anomalia (A = 3, szum σ = 1; średnia nadwyżka `z` na fragmencie, jeśli model
cyklu nie wchłonie anomalii):

| Anomalia | Reszta na fragmencie | Nadwyżka energii | Poziomy 1 / 2 / 3 / 4 |
|---|---|---|---|
| faza +Δθ | `A·[sin(ψ + Δθ) − sin ψ] = 2A·sin(Δθ/2)·cos(ψ + Δθ/2)` | `2A²·sin²(Δθ/2)` | 0,69 / 2,6 / 9 / 18 |
| amplituda ×k | `(k − 1)·A·sin ψ` | `(1 − k)²·A²/2` | 0,28 / 1,1 / 2,5 / 4,5 |
| okres ×r | różnica dwóch sinusów o różnych częstotliwościach – rośnie w miarę rozjeżdżania się faz | do `A² = 9` | zależy od L |

Stąd słabość dla **słabnięcia cyklu**: ×0,75 daje nadwyżkę 0,28 wariancji szumu – mniej niż najsłabsze przesunięcie
fazy (0,69).

### Mocne i słabe strony (wyniki S3)

- ✔ Dobry ranking (AUC 0,95 / 0,89 / 0,93 dla fazy / amplitudy / okresu), F1 0,65; szybka poza samym
  STL (≈ 1 s na szereg dla wersji odpornej).
- ✔ Odporna na słaby cykl (A = 1: faza 63%, okres 92%, gdy metody okienkowe 0–53%) – model cyklu dopasowany do całego
  szeregu jest dokładny nawet wtedy, gdy w pojedynczym oknie cyklu prawie nie widać.
- ✘ Słabnięcie cyklu wykrywa najsłabiej z metod statystycznych (67%; ×0,75: 5%, ×0,5: 65%) – patrz tabela energii.
- ✘ Nadal więcej fałszywych alarmów w szeregach z anomaliami niż kNN czy PELT (0,14 na szereg, 0,20 przy najsilniejszych
  anomaliach) – mimo lokalizacji karta bywa nad progiem tuż poza strefą tolerancji.
- ✘ Działa tylko tak dobrze, jak model cyklu: STL z domyślnym oknem sezonowości (7 cykli) wchłania długie anomalie –
  wykrywanie spada do 84% / 77% / 53% przy L = 50 / 100 / 200 (z oknem 121 cykli: 89% / 96% / 96%; analiza dodatkowa
  S3 (a), rozdz. 0.6).

---

## 10. Autoenkoder

*Grupa: uczenie maszynowe. Scenariusz: S3. Kod: `Autoenkoder` w `src/metody/ml.py` (scikit-learn `MLPRegressor`).*

### Intuicja

Autoenkoder to sieć neuronowa, która uczy się **odtwarzać swoje wejście**, ale musi przepuścić je przez wąskie
„gardło” – warstwę z kilkoma neuronami. Nie da się więc zapamiętać wszystkiego; sieć uczy się tylko tego, co powtarza
się w większości danych. Okno typowe sieć odtwarza dobrze, okno nietypowe – źle, więc **błąd odtworzenia** jest miarą
nietypowości (np. Sakurada i Yairi, 2014). W S3 „typowe” to jeden cykl sinusa – najbardziej naturalne miejsce na
autoenkoder spośród scenariuszy (plan, rozdz. 4.5).

### Jak działa

1. Okna długości w = P = 50 (jeden cykl) co 5 punktów; od szeregu odejmujemy medianę i dzielimy przez jego odporne
   odchylenie (mediana wariancji okien, rozdz. 0.3). **Okien nie normalizujemy osobno** – inaczej zmiana amplitudy by znikała (poprawka po
   pilocie).
2. Sieć `50 → 32 → 2 → 32 → 50` (warstwy ukryte z aktywacją ReLU, wyjście liniowe), uczona metodą Adam
   (`MLPRegressor`, maks. 300 epok, stałe ziarno) tak, żeby wyjście = wejście (błąd średniokwadratowy).
3. **Nauka na tym samym szeregu** (nienadzorowo, jak pozostałe metody), w dwóch krokach: po pierwszym uczeniu odrzucamy
   10% najgorzej odtwarzanych okien i uczymy ponownie – żeby sieć nie nauczyła się odtwarzać także anomalii.
4. Wynik okna = średni kwadrat błędu odtworzenia; punkty – interpolacja (rozdz. 0.2).

Gardło 2 i aktywację ReLU wybrano na 16 szeregach testowych (ziarna spoza eksperymentu): gardło 2, 4 i 8 dawało
prawie to samo AUC, gardło 1 – wyraźnie gorsze.

**Ograniczenie wynikające z geometrii sinusa.** Okno sinusa o okresie P, **dowolnej fazie i dowolnej amplitudzie**,
to `a·sin(2πt/P) + b·cos(2πt/P)` – kombinacja dwóch tych samych wektorów. Wszystkie takie okna leżą w jednej
**dwuwymiarowej podprzestrzeni**. Sieć z gardłem 2 uczy się dokładnie tej podprzestrzeni – i odtwarza w niej także
cykl przesunięty w fazie czy osłabiony. Błąd odtworzenia rośnie więc tylko dla okien, które z niej wychodzą: ze
skokiem fazy na brzegu anomalii albo z inną częstotliwością.

### Mocne i słabe strony (wyniki S3)

- ✔ Zmianę okresu wykrywa dobrze (93%, AUC 0,93) – cykl o innej częstotliwości nie leży
  w nauczonej podprzestrzeni.
- ✘ **Przesunięcie fazy widzi tylko na skokach** (68%, AUC 0,71): wynik wewnątrz przesuniętego
  fragmentu = 0,98 × wynik normy, na brzegach – 3,2 ×. AUC spada z długością anomalii, bo brzegi to coraz mniejsza część
  fragmentu.
- ✘ Zmianę amplitudy – słabo (62%, AUC 0,72); osłabiony cykl jest tylko „krótszym wektorem”
  w tej samej podprzestrzeni.
- **Sposób uczenia**: odrzucanie 10% okien pomaga przy amplitudzie (analiza dodatkowa S3 (c)); uczenie na 20 szeregach
  bez anomalii – **szkodzi** (amplituda 62% → 46%, faza i okres bez zmian). Problemem nie jest
  „zanieczyszczenie” danych uczących, tylko to, że osłabiony cykl leży w tej samej podprzestrzeni co normalny – sieć
  uczona na wielu szeregach bez anomalii odtwarza go jeszcze lepiej.
- ✘ Najwolniejsza metoda S3 (≈ 1,3 s na szereg – dwa uczenia sieci; wariant uczony na szeregach bez anomalii – jedno
  uczenie na konfigurację, ocena < 0,01 s), wyniki zależą od architektury i ziarna.

---

## 11. Zestawienie metod

Średni odsetek wykrytych anomalii (wszystkie poziomy intensywności i długości; fałszywy alarm w ~10% szeregów
bez anomalii):

| Metoda | Grupa | S1: A / B / C | S2: A / B / C | S3: A / B / C | Czas na szereg | Kluczowa właściwość |
|---|---|---|---|---|---|---|
| CUSUM (3 karty) | statystyczne | 76 / 84 / 39% | – | – | 0,02 s | jedna karta na każdą hipotezę o anomalii |
| reszty AR + CUSUM | statystyczne | – | 31 / 66 / 45% | – | 0,01 s | najpierw usunąć znaną zależność |
| STL + EWMA | statystyczne | – | – | 82 / 67 / 93% | ≈ 1 s (STL) | energia reszt po usunięciu cyklu |
| PELT | statystyczne | 66 / 88 / 19% | 23 / 55 / 36% | 84 / 72 / 93% | 0,04–0,08 s (+ STL w S3) | granice odcinków; najlepszy do wzrostu φ |
| Matrix Profile | fragmenty | 64 / 91 / 6% | 31 / 66 / 24% | 61 / 87 / 90% | 0,02–0,7 s | ślepy na fragmenty „gładsze”; widzi kształt, nie położenie |
| kNN – 5 cech | fragmenty | 57 / 76 / 11% | 42 / 60 / 37% | 91 / 86 / 96% | 0,02–0,1 s | zależy od doboru cech |
| kNN – 3 cechy | fragmenty | 77 / 87 / 42% | 42 / 48 / 38% | 92 / 87 / 96% | 0,02–0,1 s | S1: najlepsza obok CUSUM |
| Isolation Forest – 5 cech | ML | 26 / 75 / 2% | 8 / 48 / 1% | 31 / 25 / 78% | 0,3 s | dobry ranking, maskowanie |
| Isolation Forest – 5 cech, uczony bez anomalii | ML | 53 / 77 / 19% | 11 / 54 / 16% | 76 / 82 / 94% | 0,03 s (+ uczenie raz) | maskowanie usunięte |
| LOF – 5 / 3 cechy | ML | 63 / 72 / 26% / 68 / 78 / 36% | – | – | 0,02 s | gęstość lokalna |
| One-Class SVM – 5 cech | ML | – | 4 / 17 / 0% | – | < 0,01 s | uczony na tym samym szeregu – słaby |
| One-Class SVM – 5 cech, uczony bez anomalii | ML | – | 41 / 60 / 38% | – | < 0,01 s | standardowy sposób użycia – dobry |
| autoenkoder | ML | – | – | 68 / 62 / 93% | ≈ 1,3 s | uczy się podprzestrzeni normalnych okien |
| autoenkoder – uczony bez anomalii | ML | – | – | 67 / 46 / 92% | < 0,01 s (+ uczenie raz) | gorzej dla amplitudy |

S1: A – zmiana poziomu, B – zmiana zmienności, C – pojawienie się zależności.
S2: A – szok wygasający, B – zmiana siły zależności, C – przesunięcie poziomu równowagi.
S3: A – przesunięcie fazy, B – zmiana amplitudy (słabnięcie cyklu), C – zmiana okresu (przyspieszenie cyklu).
Matrix Profile: S1 i S3 – na szeregu (S1 bez normalizacji, S3 z normalizacją, okno = cykl), S2 – na resztach AR.

**Ogólne obserwacje z S1–S3:**
1. Metody statystyczne zbudowane pod konkretne hipotezy (CUSUM) albo pod znaną strukturę szeregu (reszty AR, reszty
   STL) są mocne, gdy wiadomo, czego szukać – i najodporniejsze, gdy sygnał jest słaby (S3, słaby cykl).
2. Metody okienkowe są tak dobre, jak ich cechy; cechy nieistotne szkodzą, a cechy niepewne (faza okna krótszego
   niż cykl) podnoszą próg.
3. Metody ML uczone na tym samym szeregu (Isolation Forest, One-Class SVM, autoenkoder) mają ograniczenia, które nie
   wynikają z „siły” modelu: maskowanie, nasycenie wyniku, uczenie się zbyt ogólnej normy (podprzestrzeń sinusów).
   Uczenie na szeregach bez anomalii usuwa dwa pierwsze (Isolation Forest, One-Class SVM), ale nie trzecie
   (autoenkoder) – pomaga tam, gdzie słabością było zanieczyszczenie danych uczących.
6. Zestaw cech trzeba dobierać do scenariusza: w S1 lepsze 3 cechy, w S2 – 5 cech, w S3 – bez różnicy (kNN).
4. Okno powinno mieć długość porównywalną z długością anomalii – a w szeregu okresowym co najmniej jeden cykl.
5. Model struktury szeregu (AR, STL) trzeba dobrać tak, żeby **nie wchłaniał anomalii** – w S2 model AR „zjadał”
   część przesunięcia poziomu, w S3 STL z krótkim oknem sezonowości – długie anomalie cyklu.

---

## 12. Literatura

- E. S. Page (1954). *Continuous inspection schemes*. Biometrika 41 – karta CUSUM.
- A. J. Fox (1972). *Outliers in time series*. JRSS B 34 – szoki addytywne i innowacyjne w szeregach ARMA.
- R. Killick, P. Fearnhead, I. A. Eckley (2012). *Optimal detection of changepoints with a linear computational cost*.
  JASA 107 – algorytm PELT.
- C.-C. M. Yeh i in. (2016). *Matrix Profile I: All pairs similarity joins for time series*. IEEE ICDM – Matrix Profile.
- S. Ramaswamy, R. Rastogi, K. Shim (2000). *Efficient algorithms for mining outliers from large data sets*. SIGMOD –
  odległość do k-tego sąsiada jako miara nietypowości.
- F. Angiulli, C. Pizzuti (2002). *Fast outlier detection in high dimensional spaces*. PKDD – suma/średnia odległości
  do k sąsiadów.
- F. T. Liu, K. M. Ting, Z.-H. Zhou (2008). *Isolation Forest*. IEEE ICDM.
- M. M. Breunig, H.-P. Kriegel, R. T. Ng, J. Sander (2000). *LOF: Identifying density-based local outliers*. SIGMOD.
- B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, R. C. Williamson (2001). *Estimating the support of
  a high-dimensional distribution*. Neural Computation 13 – One-Class SVM.
- R. B. Cleveland, W. S. Cleveland, J. E. McRae, I. Terpenning (1990). *STL: A seasonal-trend decomposition procedure
  based on Loess*. Journal of Official Statistics 6 – rozkład STL.
- S. W. Roberts (1959). *Control chart tests based on geometric moving averages*. Technometrics 1 – karta EWMA.
- G. E. Hinton, R. R. Salakhutdinov (2006). *Reducing the dimensionality of data with neural networks*. Science 313 –
  autoenkodery.
- M. Sakurada, T. Yairi (2014). *Anomaly detection using autoencoders with nonlinear dimensionality reduction*.
  MLSDA – błąd odtworzenia autoenkodera jako miara nietypowości.
- V. Chandola, A. Banerjee, V. Kumar (2009). *Anomaly detection: A survey*. ACM Computing Surveys 41 – podział na
  wykrywanie nienadzorowane i półnadzorowane (model uczony na danych bez anomalii).
- R. Wu, E. J. Keogh (2021). *Current time series anomaly detection benchmarks are flawed and are creating the illusion
  of progress*. IEEE TKDE – kontekst oceny metod.
