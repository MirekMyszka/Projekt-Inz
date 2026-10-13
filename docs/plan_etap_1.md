# Etap 1 – symulacje: 3 scenariusze, w każdym 3 własne anomalie zbiorowe, 6 metod

> Ustalenia z konsultacji z promotorką **2026-10-13** + plan pracy na najbliższy etap.
> Ten dokument ma **pierwszeństwo** przed szerszym planem w [plan_metod.md](plan_metod.md) – tamten zostaje
> jako tło i plan na dalsze etapy (więcej szeregów, typów anomalii, dane rzeczywiste).

## 1. Ustalenia z konsultacji (notatka uporządkowana)

1. **Zawężyć zakres na początek** – zamiast 6 rodzajów szeregów i 8 typów anomalii: **3 scenariusze** (rodzaje szeregów).
2. Scenariusze mają różnić się **zależnością w czasie**:
   - szereg **bez zależności** – biały szum,
   - szereg **z zależnością** – AR(1),
   - szereg **okresowy** – biały szum + sinus.
3. **Każdy scenariusz ma własne 3 rodzaje anomalii zbiorowych** (fragment szeregu, nie pojedynczy punkt),
   **adekwatne do tego szeregu** – scenariusze nie dzielą tych samych anomalii (np. w sinusie: anomalie okresowości).
4. **W każdym wygenerowanym szeregu są 3 anomalie – każda innego rodzaju** (po jednej z trzech rodzajów danego scenariusza).
5. Do każdego scenariusza **po 2 metody z każdej grupy** (razem 6) – statystyczne, oparte na fragmentach szeregu,
   uczenie maszynowe – dobrane tak, żeby **pasowały do szeregu i do anomalii**. ✅ potwierdzone
6. Wykrywanie **na gotowych szeregach** (offline) – metoda dostaje cały szereg naraz, nie musi alarmować na bieżąco.
7. **Kolejność pracy:** zaczynamy od **jednego scenariusza – białego szumu (S1)** ✅ potwierdzone; na nim budujemy
   i dopracowujemy ocenę metod, dopiero potem S2 i S3.

## 2. Trzy scenariusze (szeregi tła)

| | Scenariusz | Model | Zależność w czasie | Parametry startowe |
|---|---|---|---|---|
| **S1** | biały szum | `x_t = ε_t`, ε ~ N(0, σ²) | brak | σ = 1 |
| **S2** | AR(1) | `x_t = μ + φ·(x_{t−1} − μ) + ε_t` | tak – każdy punkt zależy od poprzedniego | μ = 0, φ = 0,7 (później 0,3 i 0,9) |
| **S3** | biały szum + sinus | `x_t = A·sin(2πt/P + θ) + ε_t` | okresowa (deterministyczny cykl) | A = 3, P = 50, θ = 0 |

Wspólne: długość szeregu **n = 3 000**, poziom szumu σ = 1 (później kilka poziomów), ziarno losowe zapisywane
z każdym szeregiem – każdy szereg da się odtworzyć.

## 3. Anomalie – osobny zestaw 3 rodzajów dla każdego scenariusza

Każda anomalia to **fragment długości L ∈ {20, 50, 100}** (w S3: L ∈ {50, 100, 200}, czyli 1, 2 i 4 cykle – zmiana po pilocie). Etykiety: 0 = norma, A/B/C = rodzaj anomalii –
dzięki temu skuteczność liczymy też **osobno dla każdego rodzaju**.

**Rozmieszczenie w szeregu:** szereg dzielimy na 3 równe części (po 1 000 punktów); w każdej części jedna anomalia
w losowym miejscu, kolejność rodzajów losowa, co najmniej 100 punktów odstępu od brzegów i od sąsiedniej anomalii.
Około 2–10% punktów szeregu to anomalie (zależnie od L).

Rodzaje dobrane tak, żeby każdy łamał to, **co definiuje dany szereg**:

### S1 – biały szum (szereg opisany tylko średnią, wariancją i brakiem zależności)

| | Anomalia | Jak generujemy | Intensywność |
|---|---|---|---|
| **S1-A** | krótkotrwała **zmiana poziomu** | do fragmentu dodajemy stałą Δ; potem powrót do normy | Δ ∈ {0,5; 1; 2; 3}·σ |
| **S1-B** | krótkotrwała **zmiana zmienności** | szum we fragmencie mnożymy przez c | c ∈ {1,5; 2; 3; 4} |
| **S1-C** | **pojawienie się zależności** | fragment zastępujemy procesem AR(1) o tej samej wariancji – średnia i zmienność bez zmian, zmienia się tylko struktura | φ ∈ {0,5; 0,7; 0,9} |

### S2 – AR(1) (szereg opisany poziomem równowagi μ, siłą zależności φ i innowacjami ε)

| | Anomalia | Jak generujemy | Intensywność |
|---|---|---|---|
| **S2-A** | **szok innowacyjny** (*innovational outlier*) | jeden duży impuls w innowacji ε, który **wygasa zgodnie z dynamiką AR** – w szeregu widać ogon o długości ~L | impuls ∈ {3; 5; 8}·σ_ε |
| **S2-B** | **zmiana siły zależności** | we fragmencie φ zmienia się (np. 0,7 → 0 lub 0,7 → 0,95), wariancja szeregu zachowana | φ' ∈ {0; 0,3; 0,95} |
| **S2-C** | **przesunięcie poziomu równowagi** | we fragmencie μ → μ + Δ – szereg **stopniowo** dochodzi do nowego poziomu i stopniowo wraca (inaczej niż skok w S1) | Δ ∈ {1; 2; 3}·σ_x |

### S3 – biały szum + sinus (szereg opisany okresem P, amplitudą A i fazą θ)

| | Anomalia | Jak generujemy | Intensywność |
|---|---|---|---|
| **S3-A** | **przesunięcie fazy** | we fragmencie cykl przesunięty: θ → θ + Δθ („spóźniony” cykl) | Δθ ∈ {π/4; π/2; π} |
| **S3-B** | **zmiana amplitudy** | we fragmencie A → k·A (zanik, osłabienie albo wzmocnienie cyklu) | k ∈ {0; 0,5; 1,5; 2} |
| **S3-C** | **zmiana okresu** | we fragmencie cykl szybszy lub wolniejszy: P → r·P | r ∈ {0,5; 0,75; 1,5} |

W S3 wszystkie trzy anomalie to **zaburzenia okresowości** – zgodnie z przykładem z konsultacji.

Dlaczego te anomalie są trudne w różny sposób:
- S1-A, S1-B – zmienia się **wartość** lub **rozrzut** – widoczne dla prostych statystyk,
- S1-C, S2-B, S3-A, S3-C – średnia i wariancja fragmentu są **takie jak w normie**, zmienia się tylko struktura w czasie –
  metody patrzące wyłącznie na wartości ich nie zobaczą,
- S2-A, S2-C – mają **kształt wynikający z dynamiki szeregu** (wygaszanie, stopniowe dojście) – nie ma ostrej granicy.

## 4. Metody – po 2 z każdej grupy, dobrane do scenariusza

### 4.1 Zasady doboru

1. **Tylko metody z listy w [plan_metod.md](plan_metod.md), rozdz. 4**: karty kontrolne CUSUM / EWMA, reszty modelu
   (AR, STL), wykrywanie punktów zmiany PELT, Matrix Profile, cechy okien + kNN, Isolation Forest, LOF / One-Class SVM,
   autoenkoder. **Odpada robust z-score / filtr Hampela** – wykrywa pojedyncze punkty, a nasze anomalie są zbiorowe.
2. **Wspólny rdzeń we wszystkich scenariuszach** – Isolation Forest, cechy okien + kNN, Matrix Profile i PELT występują
   w S1, S2 i S3. Dzięki temu można porównać, **jak ta sama metoda radzi sobie w szeregu bez zależności, z zależnością
   i okresowym** – to bezpośrednio odpowiada na pytanie z tematu pracy o wpływ właściwości szeregu.
3. **Konfiguracja dopasowana do szeregu i jego anomalii** – co metoda „ogląda” (surowy szereg czy reszty modelu),
   jakie cechy okna, jaka normalizacja, jaka długość okna.
4. **Druga metoda statystyczna i druga ML są różne w każdym scenariuszu** – wybrane tak, żeby najlepiej pasowały
   do jego natury. Przy okazji w całym etapie 1 sprawdzimy wszystkie metody z listy.
5. Wszystkie metody **offline** (na gotowym szeregu) i we wspólnym interfejsie: `score(x) → wynik anomalii dla każdego
   punktu`; metody okienkowe przypisują wynik okna jego punktom. Okno domyślne w = 50 (metoda nie zna prawdziwego L).

### 4.2 Zestawienie

| Grupa | S1 – biały szum | S2 – AR(1) | S3 – szum + sinus |
|---|---|---|---|
| **statystyczne 1** | **CUSUM** na trzech statystykach: x (poziom), x² (zmienność), x_t·x_{t−1} (zależność) | **reszty modelu AR(1) + CUSUM** | **STL + karta EWMA** na resztach |
| **statystyczne 2** | **PELT** – koszt „średnia + wariancja” oraz koszt AR | **PELT z kosztem AR** (segmenty o innym φ i μ) | **PELT** na resztach STL |
| **fragmenty 1** | **Matrix Profile bez normalizacji** (okno 50) | **Matrix Profile bez normalizacji na resztach AR** (zmiana po pilocie, rozdz. 8) | **Matrix Profile z normalizacją**, okno = okres P |
| **fragmenty 2** | **cechy okien + kNN** | **cechy okien + kNN** (cechy dynamiki AR) | **cechy cykli + kNN** (cechy okresowości) |
| **ML 1** | **Isolation Forest** na cechach okien | **Isolation Forest** na cechach okien | **Isolation Forest** na cechach cykli |
| **ML 2** | **LOF** (Local Outlier Factor) | **One-Class SVM** | **autoenkoder** (odtwarzanie kształtu cyklu) |

### 4.3 Uzasadnienie – S1, biały szum (anomalie: S1-A poziom, S1-B zmienność, S1-C pojawienie się zależności)

- **CUSUM** sumuje małe odchylenia od normy, więc wykrywa krótkotrwałe przesunięcia, których pojedynczy punkt nie zdradza.
  Ta sama karta liczona na trzech statystykach odpowiada trzem anomaliom: na x → zmiana średniej (S1-A),
  na x² → zmiana wariancji (S1-B), na iloczynie sąsiednich punktów x_t·x_{t−1} → pojawienie się zależności (S1-C;
  w białym szumie jego średnia = 0, w fragmencie AR(1) jest dodatnia). Wynik = maksimum z trzech kart.
- **PELT** dzieli gotowy szereg na odcinki o stałych parametrach i wskazuje granice – naturalna metoda dla anomalii,
  które mają początek i koniec. Koszt „średnia + wariancja” → S1-A, S1-B; koszt AR → S1-C. Typowo offline.
- **Matrix Profile bez normalizacji** (`stumpy.aamp`) – **ważny szczegół:** zwykły Matrix Profile normalizuje każdy
  fragment (odejmuje średnią, dzieli przez odch. std), więc **nie widzi zmian poziomu i zmienności** – a to dwie
  z trzech anomalii S1. Wersja bez normalizacji je widzi; S1-C (gładszy fragment) też daje inny kształt.
  Porównanie obu wersji to dobry, krótki eksperyment do pracy.
- **Cechy okien + kNN** – każde okno opisane cechami dokładnie odpowiadającymi anomaliom: średnia (S1-A),
  odch. std (S1-B), autokorelacja z opóźnieniem 1 (S1-C), plus kurtoza i zakres. Wynik = średnia odległość do
  k najbliższych okien – interpretowalny (widać, która cecha odstaje).
- **Isolation Forest** na tych samych cechach – drzewa losowo dzielą przestrzeń cech; nietypowe okna izolują się
  po kilku podziałach. Szybki, bez założeń o rozkładzie – główny kandydat ML.
- **LOF** na tych samych cechach – porównuje gęstość okna z gęstością sąsiadów. Tło jest jednorodne (biały szum),
  więc LOF i kNN powinny dać podobne wyniki – ciekawe sprawdzenie, czy „ML” wnosi coś ponad prostą odległość.

### 4.4 Uzasadnienie – S2, AR(1) (anomalie: S2-A szok wygasający, S2-B zmiana φ, S2-C przesunięcie równowagi)

- **Reszty modelu AR(1) + CUSUM** – w AR(1) naturalne długie wahania łatwo pomylić z anomalią. Najpierw dopasowujemy
  model AR(1), potem patrzymy na **reszty** (to, czego model nie przewidział) – są prawie białym szumem.
  Szok wygasający (S2-A) to w resztach **jeden ogromny punkt** (klasyczny test *innovational outlier*), przesunięcie
  równowagi (S2-C) – seria reszt o niezerowej średniej (CUSUM), zmiana φ (S2-B) – zmiana autokorelacji reszt.
- **PELT z kosztem AR** – dzieli szereg na odcinki z osobnym modelem AR; wskazuje fragmenty o innym φ (S2-B) i μ (S2-C).
- **Matrix Profile z normalizacją** – tu normalizacja pomaga: usuwa naturalne wahania poziomu AR, a zostawia
  **kształt** – wygasanie szoku (S2-A) ma charakterystyczny kształt, a fragment o innym φ jest „bardziej poszarpany”
  albo „gładszy” niż reszta.
- **Cechy okien + kNN** – cechy dynamiki: autokorelacja z opóźnieniem 1–3 (S2-B), średnia (S2-C), maksimum |reszty|
  (S2-A), odch. std.
- **Isolation Forest** – te same cechy (rdzeń wspólny).
- **One-Class SVM** – uczy się granicy obszaru „normalnych” okien w przestrzeni cech. Zbiór okien jest mały
  (n/w ≈ 60–3 000), więc metoda jest szybka; sprawdzamy, czy gładka granica radzi sobie lepiej niż drzewa (IF).

### 4.5 Uzasadnienie – S3, szum + sinus (anomalie: S3-A faza, S3-B amplituda, S3-C okres)

- **STL + karta EWMA** – STL rozkłada szereg na trend, sezonowość (okres P) i reszty. Każda z trzech anomalii
  psuje dopasowanie cyklu, więc w resztach pojawia się fragment o większej energii. Karta EWMA na kwadratach reszt
  wygładza je i alarmuje, gdy energia trwale rośnie.
- **PELT na resztach STL** – wskazuje początek i koniec fragmentu, w którym cykl „nie pasuje”.
- **Matrix Profile z normalizacją, okno = P** – okno równe jednemu cyklowi; przesunięcie fazy (S3-A) i zmiana okresu
  (S3-C) dają fragmenty bez „bliźniaka”. **Hipoteza do sprawdzenia:** zmiana amplitudy (S3-B) będzie dla niego
  niewidoczna (normalizacja usuwa skalę) – wtedy dokładamy wersję bez normalizacji.
- **Cechy cykli + kNN** – okno = jeden cykl, cechy dokładnie odpowiadające anomaliom: faza (przesunięcie maksimum
  korelacji z wzorcowym sinusem; S3-A), amplituda (odch. std / zakres; S3-B), dominująca częstotliwość z FFT (S3-C),
  korelacja z wzorcowym cyklem.
- **Isolation Forest** – te same cechy cykli (rdzeń wspólny).
- **Autoenkoder** – sieć uczy się odtwarzać typowy cykl; cykl z inną fazą, amplitudą lub okresem odtworzy źle →
  duży błąd odtworzenia = anomalia. Najbardziej naturalne miejsce na autoenkoder z całej listy, bo „normalny kształt”
  jest tu wyraźnie określony. Wystarczy prosta sieć (`MLPRegressor` ze `scikit-learn`, bez ciężkich bibliotek).

### 4.6 Oczekiwania do sprawdzenia (hipotezy dla S1)

✔ = powinna wykryć dobrze, ◐ = częściowo, – = nie powinna

| Metoda | S1-A poziom | S1-B zmienność | S1-C zależność |
|---|---|---|---|
| CUSUM (x, x², x_t·x_{t−1}) | ✔ | ✔ | ✔ (na iloczynie) |
| PELT | ✔ | ✔ | ◐ (koszt AR) |
| Matrix Profile bez normalizacji | ✔ | ✔ | ◐ |
| Matrix Profile z normalizacją (porównawczo) | – | – | ◐ |
| cechy okien + kNN | ✔ | ✔ | ✔ |
| Isolation Forest | ✔ | ✔ | ◐ |
| LOF | ✔ | ✔ | ◐ |

Spodziewany wniosek: S1-C (zmienia się tylko struktura) jest najtrudniejsza – wykrywają ją tylko metody, które
**jawnie patrzą na zależność** (CUSUM na iloczynie, cecha autokorelacji). To dobry punkt do dyskusji w pracy.

## 5. Kolejność pracy

Najpierw **jeden scenariusz – S1 (biały szum)** (ustalone na konsultacji): najprostszy, każdy błąd w kodzie od razu widać.

| # | Krok | Wynik | Gdzie |
|---|---|---|---|
| 1 | Generator S1 + wstawianie anomalii S1-A, S1-B, S1-C (3 w każdym szeregu) z etykietami | funkcje + wykres kontrolny: szereg z trzema zaznaczonymi anomaliami | `src/symulacje/generator.py`, `src/symulacje/anomalie.py` |
| 2 | Moduł oceny: metryki, dobór progu | funkcje + testy na prostych przykładach | `src/ewaluacja/metryki.py` |
| 3 | Pierwsza metoda (cechy okien + kNN – najprostsza i najbardziej interpretowalna) i pełny łańcuch: generuj → wstaw → wykryj → oceń | pierwszy wynik liczbowy | `src/metody/` |
| 4 | Pozostałe 5 metod dla S1 (zestaw z rozdz. 4.2) | 6 metod we wspólnym interfejsie | `src/metody/` |
| 5 | Eksperyment S1: intensywność × L × 6 metod × 100 szeregów (każdy z 3 anomaliami) + szeregi bez anomalii | tabela wyników + wykresy | `src/ewaluacja/eksperyment.py`, `wyniki/` |
| 6 | **Konsultacja** – pokazać wyniki S1, potwierdzić kierunek | – | – |
| 7 | S2 (AR(1)) – generator, anomalie S2-A/B/C, metody z rozdz. 4.4, eksperyment | wyniki S2 | jw. |
| 8 | S3 (sinus) – generator, anomalie okresowości S3-A/B/C, metody z rozdz. 4.5, eksperyment | wyniki S3 | jw. |
| 9 | Porównanie scenariuszy: jak zależność w szeregu zmienia skuteczność metod | wnioski do pracy | `wyniki/` |
| 10 | (dalej) dane rzeczywiste: NYC Taxi (z etykietami), SPY, BTC; ewentualnie więcej scenariuszy | – | – |

## 6. Jak oceniamy (dla wszystkich scenariuszy tak samo)

- **Wykrycie zdarzenia** – anomalia wykryta, jeśli choć jeden alarm wewnątrz fragmentu (± tolerancja w/2);
  → **czułość** (odsetek wykrytych anomalii) – **osobno dla każdego z 3 rodzajów** anomalii danego scenariusza.
- **Fałszywe alarmy** – alarmy poza fragmentami anomalii, liczone też na szeregach **bez anomalii**
  → **odsetek fałszywych alarmów** (na 1 000 punktów).
- **Precyzja, F1** na poziomie punktów – jak dokładnie metoda wskazuje zasięg fragmentu.
- **PR-AUC** – ocena niezależna od progu.
- **Próg** ustalany jednakowo dla wszystkich metod: kwantyl wyniku na szeregach bez anomalii, tak żeby odsetek
  fałszywych alarmów był z góry ustalony (np. 1%). Dzięki temu porównanie metod jest uczciwe.
- **Powtórzenia:** 100 szeregów na każdą kombinację parametrów, wynik = średnia ± przedział ufności.

Główne wykresy wyników (odpowiadają tematowi pracy):
1. czułość vs **intensywność** anomalii – osobno dla każdego rodzaju anomalii, linia = metoda,
2. czułość vs **długość** anomalii L,
3. (w etapie 9) porównanie **S1, S2, S3** – które metody radzą sobie z anomaliami wartości, a które z anomaliami
   struktury, i jak zależność w szeregu zmienia ich skuteczność.

## 7. Pakiety

`numpy`, `scipy`, `statsmodels` (model AR, STL, autokorelacja), `scikit-learn` (Isolation Forest, LOF, One-Class SVM,
kNN, autoenkoder `MLPRegressor`), `ruptures` (PELT), `stumpy` (Matrix Profile).

## 8. Pilot – czy ten dobór metod ma sens? (2026-10-13)

Szybki prototyp (`src/symulacje/pilot_etap1.py`, wyniki: `wyniki/pilot_etap1_wyniki.txt`,
wykres: `wyniki/wykresy/scenariusze_z_anomaliami.png`): po 30 szeregów z anomaliami + 30 bez, n = 3 000,
**średnie intensywności** – S1: Δ = 1σ, c = 2, φ = 0,7, L = 50; S2: szok 5σ, φ 0,7 → 0, równowaga +2σ_x, L = 50;
S3: faza +π/2, amplituda ×0,5, okres ×0,5, L = 100. Implementacje uproszczone (prototyp, nie wersja docelowa).

**Miara: AUC** – czy wynik metody jest wyższy na fragmencie z anomalią niż w normie, niezależnie od progu
(1,0 = idealnie, 0,5 = zgadywanie). ✔ ≥ 0,90, ◐ 0,75–0,90, ✘ < 0,75.

| Scenariusz | Metoda | anomalia A | anomalia B | anomalia C |
|---|---|---|---|---|
| **S1** (A poziom, B zmienność, C zależność) | CUSUM (x, x², x·x₋₁) | ✔ 0,94 | ✔ 0,98 | ◐ 0,83 |
| | PELT | ✔ 0,92 | ✔ 0,93 | ◐ 0,79 |
| | Matrix Profile bez normalizacji | ✔ 0,96 | ✔ 1,00 | ✘ 0,57 |
| | *Matrix Profile z normalizacją (porównawczo)* | *✘ 0,52* | *✘ 0,40* | *✘ 0,53* |
| | cechy okien + kNN | ✔ 0,98 | ✔ 1,00 | ✔ 0,96 |
| | Isolation Forest | ✔ 0,95 | ✔ 1,00 | ✔ 0,97 |
| | LOF | ✔ 0,99 | ✔ 0,99 | ✔ 0,97 |
| **S2** (A szok, B zmiana φ, C równowaga) | reszty AR + CUSUM | ✔ 0,92 | ✔ 0,98 | ✔ 0,94 |
| | PELT (koszt AR) | ✘ 0,45 | ◐ 0,87 | ◐ 0,77 |
| | Matrix Profile z normalizacją | ✘ 0,41 | ✔ 0,99 | ✘ 0,36 |
| | cechy okien + kNN | ✔ 0,95 | ✔ 0,98 | ✔ 0,98 |
| | Isolation Forest | ◐ 0,89 | ✔ 0,98 | ✔ 0,95 |
| | One-Class SVM | ✔ 0,91 | ✔ 0,93 | ✔ 0,97 |
| **S3** (A faza, B amplituda, C okres) | STL + EWMA | ✔ 1,00 | ◐ 0,75 | ✔ 1,00 |
| | PELT (reszty STL) | ✘ 0,30 | ✘ 0,41 | ✘ 0,33 |
| | Matrix Profile z normalizacją, okno = P | ✔ 0,90 | ✔ 0,96 | ✔ 0,97 |
| | cechy cykli + kNN | ✔ 0,99 | ✔ 0,93 | ✔ 1,00 |
| | Isolation Forest | ✔ 0,98 | ✔ 0,94 | ✔ 1,00 |
| | autoenkoder | ✔ 0,90 | ✘ 0,73 | ✔ 1,00 |

### Wnioski

1. **Dobór ma sens:** 40 z 54 par (metoda × anomalia) ma AUC ≥ 0,90. Metody oparte na cechach okien
   (kNN, Isolation Forest, LOF, One-Class SVM) działają na wszystkich anomaliach – bo cechy dobrano pod anomalie.
2. **Potwierdzone hipotezy:** Matrix Profile z normalizacją **nie widzi zmian poziomu i zmienności** (S1: AUC ≈ 0,5),
   wersja bez normalizacji je widzi. Anomalia „pojawienie się zależności” (S1-C) jest najtrudniejsza w S1.
3. **Hipotezy obalone (ciekawe do pracy):**
   - Matrix Profile z normalizacją **wykrywa zmianę amplitudy** w S3 (0,96), choć zakładaliśmy, że nie –
     przy mniejszej amplitudzie rośnie udział szumu, więc znormalizowany kształt cyklu staje się „poszarpany”;
   - w S2 Matrix Profile z normalizacją **nie widzi szoku ani przesunięcia równowagi** – szok trwa ~9 punktów
     (krótko wobec okna 50), a przesunięcie poziomu znika po normalizacji (jak w S1).
4. **Najtrudniejsza anomalia S3:** zmiana amplitudy (S3-B) – słabe STL + EWMA (0,75) i autoenkoder (0,73).
5. **PELT w obecnej formie nie działa w S2 i S3** – to problem implementacji pilota (PELT zwraca granice odcinków,
   a nie wynik dla każdego punktu; zamiana granic na wynik była prowizoryczna i wrażliwa na karę). W S1 działa dobrze.
6. **Największe wyzwanie to ocena, nie metody:** liczba „wykrytych” anomalii przy stałym progu bardzo zależy
   od sposobu ustawienia progu i od tego, jak długo alarm trwa po końcu anomalii (część alarmów wliczała się
   jako fałszywe). Moduł oceny (krok 2) wymaga starannego zaprojektowania – patrz zmiany niżej.

### Zmiany w planie wynikające z pilota

| Co | Zmiana |
|---|---|
| S2 – fragmenty 1 | **Matrix Profile bez normalizacji na resztach modelu AR** zamiast z normalizacją na surowym szeregu (szok = duża reszta; przesunięcie równowagi = reszty o niezerowej średniej) |
| PELT (wszystkie scenariusze) | oceniać jako **wykrywanie granic**: czy znaleziona granica odcinka leży blisko początku/końca anomalii; karę dobierać na szeregach bez anomalii (np. tak, żeby średnio < 1 fałszywa granica na szereg) |
| S2-A (szok) | szok wygasa w ~9 punktach – dla metod okienkowych okno w = 20 zamiast 50 (sprawdzić oba) |
| S3 – długość anomalii | L w **cyklach**: L ∈ {1, 2, 4}·P = {50, 100, 200} (fragment krótszy niż cykl nie ma sensu dla anomalii okresowości) |
| S3 – autoenkoder | uczyć na szeregu bez anomalii (albo z odrzuceniem najgorzej odtwarzanych okien) i nie normalizować amplitudy okien |
| Ocena (krok 2) | (1) AUC jako miara niezależna od progu; (2) próg ustalany na szeregach bez anomalii tak, żeby **każda metoda miała ten sam odsetek fałszywych alarmów**; (3) strefa tolerancji po końcu anomalii (alarm tuż po anomalii to nie fałszywy alarm); (4) fałszywe alarmy liczone jako osobne epizody |
