# Wyniki – scenariusz S1: biały szum

> Stan na **2026-10-27**: po przeliczeniu z poprawioną skalą szumu i po uwagach z konsultacji 27.10 (warianty 5 i 3 cech,
> metody ML uczone także na szeregach bez anomalii, wyniki osobno dla każdej długości L – rozdz. 7). Realizacja kroków 1–5
> z [plan_etap_1.md](plan_etap_1.md), rozdz. 5.
> Wszystkie liczby w tym dokumencie pochodzą z `wyniki/s1/podsumowanie.md` (tabele generowane automatycznie)
> i można je odtworzyć poleceniami z rozdz. 9.

## Spis treści

1. [Co zbadano](#1-co-zbadano)
2. [Najważniejsze wnioski](#2-najważniejsze-wnioski)
3. [Wyniki dla każdego rodzaju anomalii](#3-wyniki-dla-każdego-rodzaju-anomalii) (w tym zestaw cech i uczenie ML)
4. [Wpływ długości anomalii, długości okna i poziomu szumu](#4-wpływ-długości-anomalii-długości-okna-i-poziomu-szumu)
5. [Analiza dodatkowa: dlaczego zależność jest trudna](#5-analiza-dodatkowa-dlaczego-zależność-jest-trudna)
6. [Weryfikacja hipotez z planu](#6-weryfikacja-hipotez-z-planu)
7. [Zmiany wprowadzone w trakcie prac](#7-zmiany-wprowadzone-w-trakcie-prac)
8. [Ograniczenia i wnioski na S2 i S3](#8-ograniczenia-i-wnioski-na-s2-i-s3)
9. [Jak odtworzyć wyniki](#9-jak-odtworzyć-wyniki)

---

## 1. Co zbadano

**Szereg tła:** biały szum `x_t ~ N(0, σ²)`, σ = 1, n = 3 000 punktów.

**Anomalie** – w każdym szeregu wszystkie trzy, po jednej w każdej trzeciej części szeregu, w losowym miejscu
i losowej kolejności, z odstępem ≥ 100 punktów od brzegów:

| | Anomalia | Jak wstawiona | Poziomy intensywności 1 / 2 / 3 / 4 |
|---|---|---|---|
| **S1-A** | krótkotrwała zmiana poziomu | `x += Δ·σ` na fragmencie | Δ = 0,5 / 1 / 2 / 3 |
| **S1-B** | krótkotrwała zmiana zmienności | `x *= c` na fragmencie | c = 1,5 / 2 / 3 / 4 |
| **S1-C** | pojawienie się zależności | fragment zastąpiony AR(1) o **tej samej wariancji** | φ = 0,3 / 0,5 / 0,7 / 0,9 |

Długość każdej anomalii L ∈ {20, 50, 100}. (W planie dla φ były 3 wartości; dodano φ = 0,3, żeby każdy rodzaj miał
4 poziomy.)

**Metody** – po 2 z każdej grupy (plan, rozdz. 4.2) + jeden wariant porównawczy:

| Grupa | Metoda | Co „ogląda” |
|---|---|---|
| statystyczne | **CUSUM** (3 karty: poziom, zmienność, zależność) | x oraz różnice sąsiednich punktów |
| statystyczne | **PELT** (podział na odcinki) | x oraz iloczyn sąsiednich punktów |
| fragmenty szeregu | **Matrix Profile** bez normalizacji, okno 50 | surowe fragmenty |
| fragmenty szeregu | **cechy okien + kNN**, okno 50 | cechy okien: 5 albo 3 (niżej) |
| uczenie maszynowe | **Isolation Forest** | te same cechy okien |
| uczenie maszynowe | **LOF** | te same cechy okien |
| *porównawczo* | *Matrix Profile z normalizacją* | *fragmenty po standaryzacji (sam kształt)* |

**Warianty (ustalenia z konsultacji 27.10.2026):**
- **zestaw cech** kNN, Isolation Forest i LOF: **5 cech** (plan: średnia, odch. std, autokorelacja lag 1, kurtoza,
  rozstęp) albo **3 cechy** (po jednej na rodzaj anomalii: średnia → A, odch. std → B, autokorelacja → C);
- **uczenie** Isolation Forest i LOF: na oknach **tego samego szeregu** (wersja nienadzorowana, jak w planie) albo na oknach
  **20 osobnych szeregów bez anomalii** (wersja półnadzorowana – standardowy sposób użycia metod wykrywania nowości;
  szczegóły: [metody.md](metody.md), rozdz. 0.8).
Wszystkie warianty liczone na tych samych 1 200 szeregach części głównej.

**Ocena** (`src/ewaluacja/metryki.py`):
- **próg alarmu** ustalany jednakowo dla każdej metody na 200 szeregach bez anomalii: fałszywy alarm w 10% takich
  szeregów; sprawdzony na 200 **nowych** szeregach bez anomalii – rzeczywisty odsetek 4–16,5%
  (na 1 000 szeregach – rozdz. 8),
- **anomalia wykryta**, jeśli choć jeden alarm leży w jej strefie (fragment ± 25 punktów),
- **fałszywy alarm** = epizod alarmu poza wszystkimi strefami,
- **AUC** – miara niezależna od progu: czy punkty anomalii mają wyższy wynik niż punkty normy (0,5 = losowo, 1 = idealnie),
- precyzja, czułość punktowa i F1 – jak dokładnie metoda wskazuje zasięg anomalii.

**Skala:** część główna – 4 poziomy × 3 długości × 100 szeregów = **1 200 szeregów (3 600 anomalii na metodę)**;
dodatkowo części: długość okna (900 szeregów), poziom szumu (300), analiza dodatkowa (400).

---

## 2. Najważniejsze wnioski

1. **Zmiana zmienności (S1-B) jest najłatwiejsza, pojawienie się zależności (S1-C) – najtrudniejsza.**
   Średnio (wszystkie poziomy i L) metody główne wykrywają 45–91% anomalii B,
   26–77% anomalii A i tylko 2–42% anomalii C. Przy φ = 0,3
   żadna metoda (także uczona na szeregach bez anomalii) nie wykrywa C częściej niż w 7%
   przypadków.

2. **Najlepsze metody: CUSUM i kNN na 3 cechach – porównywalne.** CUSUM: A 76%, B 84%,
   C 39%, AUC 0,88–0,92, F1 0,65; kNN (3 cechy): A 77%,
   B 87%, C 42%, AUC 0,85–0,98, F1 0,65. Obie metody patrzą dokładnie
   na to, co zmienia dana anomalia. Próg kNN jest nieco łagodniejszy (fałszywe alarmy na walidacji 12,5% wobec
   10%; na 1 000 szeregach 11,1% wobec 9,8%), więc jego przewaga w odsetku wykrytych jest
   częściowo efektem progu; w AUC kNN jest lepszy dla A i B, CUSUM – dla C.

3. **Do zmiany zmienności najlepszy jest Matrix Profile** (91%, AUC 0,99; przy najsłabszej zmianie
   c = 1,5 – 70%). Do zależności się **nie nadaje** (6%, AUC 0,46 – poniżej losowego).

4. **Matrix Profile z normalizacją nie widzi żadnej anomalii S1** (wykrywa ≤ 1%, AUC ≈ 0,5) – potwierdza hipotezę:
   standaryzacja każdego fragmentu usuwa właśnie to, co zmieniają anomalie A i B (średnią i skalę).

5. **Dobór cech jest ważniejszy niż wybór algorytmu.** kNN na 3 cechach wykrywa zależność φ = 0,7 przy L = 50 w
   76% przypadków, na 5 cechach – w 2%; średnio C: 11% → 42%.
   Kurtoza i rozstęp nie niosą informacji o żadnej anomalii S1, a dokładają losowy rozrzut do odległości między oknami.
   Dla LOF 3 cechy też są lepsze; dla Isolation Forest – lepsze dla zmiany poziomu (26% → 40%),
   gorsze dla zmiany zmienności (75% → 45%; rozstęp jest dla lasu silnym sygnałem zmienności) –
   zestaw cech trzeba sprawdzać dla każdej metody (rozdz. 3).

6. **Isolation Forest uczony na tym samym szeregu: świetny ranking, słabe wykrywanie – efekt maskowania.** AUC
   0,84–0,98, ale przy Δ = 3σ wykrywa 99% anomalii długości 20 i tylko
   24% długości 100: długa anomalia daje wiele podobnych nietypowych okien, które tworzą „gromadę”
   i przestają się wyróżniać. **Uczenie na szeregach bez anomalii usuwa większość tego efektu** (Δ = 3σ, L = 100:
   100%; zmiana poziomu przy L = 100 średnio 8% → 53%). LOF
   zyskuje mniej, głównie przy długich anomaliach i w dokładności lokalizacji (F1 0,40 → 0,63).

7. **Dłuższa anomalia jest łatwiejsza** – dla wszystkich metod poza ML uczonymi na tym samym szeregu. Najsilniej dla
   zależności: CUSUM wykrywa C w 9% (L = 20), 37% (L = 50) i 72%
   (L = 100) przypadków. Dlatego wyniki według poziomu podajemy osobno dla każdego L (rozdz. 3).

8. **Okno powinno mieć długość rzędu długości anomalii.** Okno dużo krótsze od anomalii prowadzi do maskowania: kNN
   (5 cech) przy L = 100 wykrywa zmianę poziomu w 9% przypadków z oknem 20
   i w 100% z oknem 100; na 3 cechach – 80% i
   100%.

9. **Poziom szumu ma znaczenie tylko dla zmiany poziomu.** Przy stałym Δ = 1 i szumie σ = 0,5 / 1 / 2 CUSUM wykrywa
   A w 100% / 98% / 20%.
   Anomalie B i C są niezależne od σ (zmiana zmienności jest względna, a zależność nie zależy od skali). Dla białego
   szumu o trudności decyduje stosunek Δ/σ.

---

## 3. Wyniki dla każdego rodzaju anomalii

Wykresy: `wyniki/wykresy/s1_czulosc_intensywnosc.png` (odsetek wykrytych – wiersz = długość L),
`wyniki/wykresy/s1_auc_intensywnosc.png` (AUC), `wyniki/wykresy/s1_uczenie.png` (uczenie ML),
`wyniki/wykresy/s1_przyklad.png` (przykładowy szereg i wynik każdej metody z progiem). Wszystkie tabele (także AUC
dla każdego L i średnie po L) – `wyniki/s1/podsumowanie.md`.

**Odsetek wykrytych anomalii – rodzaj × poziom intensywności, osobno dla każdej długości L** (ustalenie z konsultacji:
średnia po L zaciera silną zależność od długości anomalii). A: Δ = 0,5 / 1 / 2 / 3σ; B: c = 1,5 / 2 / 3 / 4;
C: φ = 0,3 / 0,5 / 0,7 / 0,9; pogrubienie – najlepszy wynik w kolumnie.

**L = 20** (100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM | 1% | **54%** | 100% | 100% | 20% | 66% | 97% | 100% | 1% | 4% | **10%** | 22% |
| PELT | 2% | 12% | 100% | 100% | 22% | 74% | 98% | 100% | 1% | 3% | 5% | 11% |
| Matrix Profile | 1% | 13% | 99% | 100% | **35%** | **78%** | 100% | 100% | 0% | 1% | 2% | 6% |
| kNN (5 cech) | 0% | 0% | 79% | 100% | 9% | 41% | 96% | 99% | 0% | 0% | 2% | 0% |
| kNN (3 cechy) | 2% | 34% | 100% | 100% | 25% | 71% | 98% | 100% | 1% | 1% | 9% | **24%** |
| Isolation Forest (5 cech) | 1% | 2% | 54% | 99% | 15% | 54% | 95% | 98% | 1% | 0% | 0% | 0% |
| Isolation Forest (3 cechy) | 0% | 7% | 96% | 100% | 3% | 21% | 44% | 69% | 1% | 0% | 0% | 0% |
| LOF (5 cech) | 1% | 10% | 94% | 97% | 7% | 41% | 89% | 97% | 1% | 0% | 1% | 13% |
| LOF (3 cechy) | 2% | 21% | 91% | 92% | 19% | 56% | 90% | 93% | 1% | 2% | 7% | 20% |
| *Matrix Profile z normalizacją* | *1%* | *0%* | *0%* | *0%* | *0%* | *0%* | *1%* | *0%* | *0%* | *0%* | *0%* | *1%* |

**L = 50** (100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM | 17% | **98%** | 100% | 100% | 41% | **99%** | 100% | 100% | 1% | 12% | 43% | 93% |
| PELT | 6% | 60% | 100% | 100% | 60% | **99%** | 100% | 100% | 4% | 1% | 14% | 42% |
| Matrix Profile | 5% | 70% | 100% | 100% | **77%** | **99%** | 100% | 100% | 1% | 1% | 6% | 14% |
| kNN (5 cech) | 0% | 49% | 100% | 100% | 27% | 96% | 100% | 100% | 0% | 0% | 2% | 54% |
| kNN (3 cechy) | **32%** | **98%** | 100% | 100% | 60% | **99%** | 100% | 100% | 7% | **23%** | **76%** | **99%** |
| Isolation Forest (5 cech) | 1% | 5% | 35% | 79% | 37% | 90% | 100% | 100% | 1% | 0% | 0% | 14% |
| Isolation Forest (3 cechy) | 3% | 20% | 89% | 100% | 22% | 49% | 82% | 86% | 2% | 1% | 4% | 27% |
| LOF (5 cech) | 9% | 82% | 99% | 100% | 39% | 97% | 100% | 97% | 1% | 5% | 30% | 94% |
| LOF (3 cechy) | 25% | 94% | 99% | 100% | 54% | 98% | 99% | 99% | 7% | 15% | 57% | 97% |
| *Matrix Profile z normalizacją* | *0%* | *0%* | *1%* | *0%* | *1%* | *1%* | *1%* | *0%* | *0%* | *0%* | *0%* | *0%* |

**L = 100** (100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUSUM | 38% | 100% | 100% | 100% | 88% | 100% | 100% | 100% | **19%** | **67%** | 100% | 100% |
| PELT | 12% | 99% | 100% | 100% | 97% | 100% | 100% | 100% | 8% | 18% | 41% | 82% |
| Matrix Profile | 12% | 68% | 99% | 100% | **99%** | 100% | 100% | 100% | 2% | 6% | 13% | 17% |
| kNN (5 cech) | 1% | 56% | 100% | 100% | 40% | 98% | 100% | 100% | 1% | 0% | 5% | 67% |
| kNN (3 cechy) | **53%** | 100% | 100% | 100% | 90% | 100% | 100% | 100% | 12% | 55% | 98% | 100% |
| Isolation Forest (5 cech) | 1% | 0% | 5% | 24% | 43% | 73% | 100% | 100% | 1% | 0% | 0% | 1% |
| Isolation Forest (3 cechy) | 0% | 3% | 19% | 39% | 14% | 34% | 56% | 57% | 0% | 0% | 0% | 5% |
| LOF (5 cech) | 13% | 73% | 86% | 87% | 51% | 72% | 84% | 91% | 4% | 13% | 59% | 95% |
| LOF (3 cechy) | 30% | 89% | 93% | 81% | 71% | 86% | 90% | 83% | 10% | 33% | 78% | 99% |
| *Matrix Profile z normalizacją* | *1%* | *1%* | *1%* | *1%* | *0%* | *0%* | *0%* | *1%* | *2%* | *0%* | *1%* | *0%* |

Przy L = 20 słabe anomalie (Δ = 0,5, c = 1,5, φ ≤ 0,5) są praktycznie niewykrywalne dla wszystkich metod, a nawet
najsilniejsza zależność (φ = 0,9) – tylko w 24% (kNN, 3 cechy) i 22% (CUSUM).
Przy L = 100 ta sama zależność jest wykrywana w 100% / 100%, a φ = 0,5 – w
67% / 55% (CUSUM / kNN 3 cechy).

**Średnio dla rodzaju** (wszystkie poziomy i L; odsetek wykrytych / AUC):

| metoda | A – poziom | B – zmienność | C – zależność |
|---|---|---|---|
| CUSUM | 76% / 0,90 | 84% / 0,92 | 39% / 0,88 |
| PELT | 66% / 0,81 | 88% / 0,90 | 19% / 0,55 |
| Matrix Profile | 64% / 0,90 | **91% / 0,99** | 6% / 0,46 |
| kNN (5 cech) | 57% / 0,94 | 76% / 0,97 | 11% / 0,83 |
| kNN (3 cechy) | **77% / 0,95** | 87% / 0,98 | **42% / 0,85** |
| Isolation Forest (5 cech) | 26% / 0,94 | 75% / 0,98 | 2% / 0,84 |
| Isolation Forest (3 cechy) | 40% / 0,95 | 45% / 0,98 | 3% / 0,86 |
| LOF (5 cech) | 63% / 0,92 | 72% / 0,91 | 26% / 0,81 |
| LOF (3 cechy) | 68% / 0,92 | 78% / 0,93 | 36% / 0,82 |
| *Matrix Profile z normalizacją* | *0% / 0,45* | *0% / 0,50* | *0% / 0,42* |

**Zestaw cech: 5 czy 3?** (odsetek wykrytych, w nawiasie AUC; średnio po poziomach i L)

| metoda | A: 5 cech → 3 cechy | B: 5 cech → 3 cechy | C: 5 cech → 3 cechy |
|---|---|---|---|
| kNN | 57% → 77% (0,94 → 0,95) | 76% → 87% (0,97 → 0,98) | 11% → 42% (0,83 → 0,85) |
| Isolation Forest | 26% → 40% (0,94 → 0,95) | 75% → 45% (0,98 → 0,98) | 2% → 3% (0,84 → 0,86) |
| Isolation Forest – uczony bez anomalii | 53% → 59% (0,93 → 0,96) | 77% → 56% (0,98 → 0,97) | 19% → 32% (0,83 → 0,86) |
| LOF | 63% → 68% (0,92 → 0,92) | 72% → 78% (0,91 → 0,93) | 26% → 36% (0,81 → 0,82) |
| LOF – uczony bez anomalii | 77% → 79% (0,95 → 0,95) | 82% → 84% (0,96 → 0,97) | 36% → 35% (0,82 → 0,83) |

- **kNN i LOF**: 3 cechy lepsze dla wszystkich rodzajów, najwyraźniej dla zależności (kNN C 11% → 42%).
- **Isolation Forest**: 3 cechy lepsze dla zmiany poziomu i zależności, gorsze dla zmiany zmienności – drzewa dobrze
  wykorzystują rozstęp jako dodatkowy sygnał zmienności (B 75% → 45%).
- AUC zmienia się mniej niż odsetek wykrytych: cechy nieistotne przede wszystkim podnoszą maksima wyniku na szeregach
  bez anomalii, czyli próg.

**Uczenie metod ML: na tym samym szeregu czy na szeregach bez anomalii?** (odsetek wykrytych dla każdego L, średnio po
poziomach; ostatnia kolumna – średnie AUC)

| metoda | uczenie na | A L=20 | A L=50 | A L=100 | B L=20 | B L=50 | B L=100 | C L=20 | C L=50 | C L=100 | AUC (śr.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest (5 cech) | tym samym szeregu | 39% | 30% | 8% | 66% | 82% | 79% | 0% | 4% | 0% | 0,92 |
|  | szeregach bez anomalii | 51% | 55% | 53% | 66% | 80% | 86% | 2% | 23% | 33% | 0,91 |
| LOF (5 cech) | tym samym szeregu | 50% | 72% | 65% | 58% | 83% | 74% | 4% | 32% | 43% | 0,88 |
|  | szeregach bez anomalii | 59% | 84% | 89% | 67% | 86% | 95% | 6% | 44% | 57% | 0,91 |
| Isolation Forest (3 cechy) | tym samym szeregu | 51% | 53% | 15% | 34% | 60% | 40% | 0% | 8% | 1% | 0,93 |
|  | szeregach bez anomalii | 56% | 62% | 60% | 34% | 61% | 73% | 8% | 35% | 52% | 0,93 |
| LOF (3 cechy) | tym samym szeregu | 52% | 80% | 73% | 64% | 88% | 82% | 8% | 44% | 55% | 0,89 |
|  | szeregach bez anomalii | 61% | 84% | 90% | 68% | 88% | 96% | 6% | 43% | 57% | 0,92 |

- **Uczenie na szeregach bez anomalii usuwa większość maskowania w Isolation Forest**: przy L = 100 zmiana poziomu
  8% → 53% (5 cech), zależność 1% → 52% (3 cechy).
  Odsetek wykrytych prawie nie zależy już od długości anomalii (zmiana poziomu, 5 cech: 51% /
  55% / 53% przy L = 20 / 50 / 100).
- **LOF** zyskuje głównie przy długich anomaliach (zmiana poziomu, L = 100: 65% → 89%)
  i w lokalizacji (F1 0,40 → 0,63); LOF porównuje okno tylko z oknami nienakładającymi się w czasie,
  więc maskowanie było u niego słabsze. Próg LOF uczonego na szeregach bez anomalii jest jednak łagodniejszy od celu
  (fałszywe alarmy na 1 000 szeregach: 15,7% / 13,7% wobec 11,6% / 13,2% – 5 / 3 cechy), więc część
  tego zysku to efekt progu; średnie AUC też rośnie (0,88 → 0,91).
- AUC prawie się nie zmienia – ranking punktów był dobry już wcześniej; poprawia się położenie progu względem wyniku
  anomalii (model uczony na danych bez anomalii nie „przyzwyczaja się” do nich).
- Wariant uczony na szeregach bez anomalii zakłada, że takie dane są dostępne – w danych rzeczywistych oznacza to
  fragment uznany za normalny.

**Fałszywe alarmy i dokładność lokalizacji:**

| metoda | szeregi bez anomalii z ≥ 1 fałszywym alarmem (walidacja) | fałszywe alarmy na szereg z anomaliami | precyzja | czułość punktowa | F1 |
|---|---|---|---|---|---|
| CUSUM | 10,0% | 0,07 | 0,89 | 0,62 | **0,65** |
| PELT | 9,5% | 0,04 | 0,88 | 0,52 | 0,58 |
| Matrix Profile | 13,0% | 0,21 | 0,98 | 0,43 | 0,56 |
| kNN (5 cech) | 8,5% | 0,09 | 0,96 | 0,39 | 0,48 |
| kNN (3 cechy) | 12,5% | 0,05 | 0,99 | 0,56 | **0,65** |
| Isolation Forest (5 cech) | 10,5% | 0,01 | 0,98 | 0,19 | 0,27 |
| Isolation Forest (3 cechy) | 4,0% | 0,01 | 0,99 | 0,13 | 0,19 |
| LOF (5 cech) | 10,0% | 0,11 | 0,95 | 0,30 | 0,40 |
| LOF (3 cechy) | 16,5% | 0,09 | 0,96 | 0,33 | 0,44 |
| Isolation Forest (5 cech, uczony bez anomalii) | 10,5% | 0,05 | 0,97 | 0,26 | 0,36 |
| Isolation Forest (3 cechy, uczony bez anomalii) | 7,5% | 0,02 | 0,98 | 0,24 | 0,33 |
| LOF (5 cech, uczony bez anomalii) | 14,0% | 0,07 | 0,98 | 0,53 | 0,63 |
| LOF (3 cechy, uczony bez anomalii) | 10,5% | 0,06 | 0,98 | 0,54 | 0,64 |
| *Matrix Profile z normalizacją* | *8,5%* | *0,08* | *0,08* | *0,00* | *0,00* |

Jak czytać: metody okienkowe mają wysoką precyzję (alarmy prawie zawsze trafiają w anomalię), ale niską czułość
punktową (oznaczają tylko część fragmentu). CUSUM, kNN na 3 cechach i LOF uczony na szeregach bez anomalii najlepiej
oddają zasięg anomalii (F1 0,63–0,65). Alarm CUSUM bywa przesunięty przed początek anomalii (karta
zaczyna rosnąć od ostatniego zera), a odcinek PELT bywa szerszy niż anomalia.

---

## 4. Wpływ długości anomalii, długości okna i poziomu szumu

**Długość anomalii L** (`wyniki/wykresy/s1_czulosc_dlugosc.png`; odsetek wykrytych, średnio po poziomach):

| metoda | A: L = 20 / 50 / 100 | B: L = 20 / 50 / 100 | C: L = 20 / 50 / 100 |
|---|---|---|---|
| CUSUM | 64 / 79 / 84% | 71 / 85 / 97% | 9 / 37 / 72% |
| PELT | 54 / 66 / 78% | 74 / 90 / 99% | 5 / 15 / 37% |
| Matrix Profile | 53 / 69 / 70% | 78 / 94 / 100% | 2 / 6 / 10% |
| kNN (5 cech) | 45 / 62 / 64% | 61 / 81 / 84% | 0 / 14 / 18% |
| kNN (3 cechy) | 59 / 82 / 88% | 74 / 90 / 98% | 9 / 51 / 66% |
| Isolation Forest (5 cech) | 39 / 30 / 8% | 66 / 82 / 79% | 0 / 4 / 0% |
| Isolation Forest (3 cechy) | 51 / 53 / 15% | 34 / 60 / 40% | 0 / 8 / 1% |
| LOF (5 cech) | 50 / 72 / 65% | 58 / 83 / 74% | 4 / 32 / 43% |
| LOF (3 cechy) | 52 / 80 / 73% | 64 / 88 / 82% | 8 / 44 / 55% |
| Isolation Forest (5 cech, uczony bez anomalii) | 51 / 55 / 53% | 66 / 80 / 86% | 2 / 23 / 33% |
| Isolation Forest (3 cechy, uczony bez anomalii) | 56 / 62 / 60% | 34 / 61 / 73% | 8 / 35 / 52% |
| LOF (5 cech, uczony bez anomalii) | 59 / 84 / 89% | 67 / 86 / 95% | 6 / 44 / 57% |
| LOF (3 cechy, uczony bez anomalii) | 61 / 84 / 90% | 68 / 88 / 96% | 6 / 43 / 57% |

Dłuższa anomalia to więcej danych, na których widać zmianę, więc zwykle łatwiej ją wykryć. Wyjątki – Isolation Forest
(maskowanie: im dłuższa anomalia, tym gorzej) i lekki spadek LOF przy L = 100 – dotyczą metod ML uczonych na tym samym
szeregu; po uczeniu na szeregach bez anomalii wyjątki znikają.

**Długość okna w** (`wyniki/wykresy/s1_okno.png`; poziomy 2–3; metody okienkowe uczone na tym samym szeregu):

| metoda | A: w = 20 / 50 / 100 | B: w = 20 / 50 / 100 | C: w = 20 / 50 / 100 |
|---|---|---|---|
| Matrix Profile | 56 / 72 / 75% | 98 / 98 / 95% | 3 / 4 / 3% |
| kNN (5 cech) | 32 / 62 / 74% | 80 / 90 / 91% | 0 / 4 / 17% |
| kNN (3 cechy) | 83 / 88 / 82% | 97 / 95 / 90% | 28 / 46 / 39% |
| Isolation Forest (5 cech) | 9 / 17 / 12% | 93 / 88 / 82% | 0 / 0 / 0% |
| Isolation Forest (3 cechy) | 35 / 39 / 48% | 80 / 46 / 24% | 6 / 2 / 1% |
| LOF (5 cech) | 48 / 76 / 75% | 69 / 85 / 86% | 10 / 18 / 24% |
| LOF (3 cechy) | 56 / 82 / 73% | 76 / 87 / 77% | 17 / 36 / 26% |

Najlepiej, gdy okno jest **porównywalne z długością anomalii lub dłuższe**. Przykład (zmiana poziomu, L = 100):
kNN (5 cech) wykrywa 9% / 71% /
100% przy w = 20 / 50 / 100, LOF (5 cech) – 14% /
80% / 100%. Krótkie okno widzi długą anomalię jako
wiele podobnych okien, które maskują się nawzajem. Na 3 cechach kNN jest prawie niezależny od okna dla zmiany poziomu
i zmienności. Dla zależności dłuższe okno pomaga też dlatego, że autokorelację da się oszacować dokładniej na większej
liczbie punktów. W praktyce długość anomalii nie jest znana; w = 50 okazało się rozsądnym wyborem domyślnym.
Isolation Forest na 3 cechach traci na zmianie zmienności przy dłuższym oknie (B 80% →
24%) – mniej okien i silniejsze maskowanie.

**Poziom szumu σ** (`wyniki/wykresy/s1_szum.png`; stała wielkość anomalii: Δ = 1, c = 2, φ = 0,5; L = 50):

| metoda | A: σ = 0,5 / 1 / 2 | B: σ = 0,5 / 1 / 2 | C: σ = 0,5 / 1 / 2 |
|---|---|---|---|
| CUSUM | 100 / 98 / 20% | 99 / 99 / 99% | 17 / 17 / 17% |
| PELT | 100 / 57 / 6% | 100 / 100 / 100% | 5 / 5 / 5% |
| Matrix Profile | 100 / 64 / 11% | 100 / 100 / 100% | 2 / 2 / 2% |
| kNN (5 cech) | 100 / 42 / 0% | 98 / 98 / 98% | 0 / 0 / 0% |
| kNN (3 cechy) | 100 / 98 / 24% | 100 / 100 / 100% | 31 / 33 / 32% |
| Isolation Forest (5 cech) | 68 / 4 / 0% | 80 / 88 / 94% | 1 / 1 / 2% |
| Isolation Forest (3 cechy) | 96 / 9 / 0% | 28 / 43 / 54% | 0 / 3 / 9% |
| LOF (5 cech) | 100 / 84 / 7% | 91 / 99 / 100% | 10 / 9 / 10% |
| LOF (3 cechy) | 100 / 94 / 17% | 99 / 100 / 100% | 27 / 28 / 27% |
| *Matrix Profile z normalizacją* | *0 / 0 / 1%* | *0 / 0 / 0%* | *0 / 0 / 0%* |

Szum „zjada” tylko zmianę poziomu – liczy się stosunek Δ/σ (σ = 2 przy Δ = 1 to to samo co Δ = 0,5σ). Zmiana
zmienności (mnożnik c) i zależność (φ) są niezależne od skali, więc wyniki B i C prawie się nie zmieniają.
Odsetek fałszywych alarmów jest dla każdego σ identyczny (zależnie od metody) – wszystkie metody są niezależne od skali
szeregu, więc kalibracja progu działa przy każdym poziomie szumu.

---

## 5. Analiza dodatkowa: dlaczego zależność jest trudna

Pytanie: czy metody wykrywają S1-C lepiej, gdy patrzą **tylko** na zależność? Porównano (L = 50, 100 szeregów na poziom;
`wyniki/wykresy/s1_ablacja_zaleznosc.png`):

| wariant | C: φ = 0,3 / 0,5 / 0,7 / 0,9 | AUC dla C | czy wykrywa A i B? |
|---|---|---|---|
| CUSUM – 3 karty (wersja z eksperymentu) | 2 / 13 / 48 / 87% | 0,74–0,99 | tak (A 14–100%, B 46–100%) |
| CUSUM – tylko karta zależności | 7 / 22 / 61 / **100%** | 0,72–0,99 | nie (0–2%) |
| kNN – 5 cech | 0 / 0 / 8 / 51% | 0,70–0,98 | tak |
| kNN – 3 cechy: średnia, odch. std, autokorelacja | 5 / 24 / **76 / 99%** | 0,74–0,98 | **tak, lepiej niż na 5 cechach** (A 29–100%, B 70–100%) |
| kNN – tylko autokorelacja | 7 / 28 / 45 / 33% | 0,75–0,95 | nie |

Wnioski:
- **Karta zależności CUSUM jest specyficzna** – reaguje tylko na zależność (A i B wykrywa w 0–2% przypadków).
  Połączenie trzech kart kosztuje niewiele przy silnej zależności (φ = 0,9: 87% zamiast 100%), więcej przy słabszej
  (φ = 0,7: 48% zamiast 61%) – próg połączonej karty musi „pomieścić” też maksima dwóch pozostałych kart.
- **Kurtoza i rozstęp szkodzą kNN.** Usunięcie tych dwóch cech poprawia wykrywanie WSZYSTKICH trzech rodzajów:
  zależności przy φ = 0,7 z 8% do 76%, a najsłabszej zmiany poziomu z 2% do 29%. Każda zbędna cecha dokłada losowy
  rozrzut do odległości między oknami, a sygnał anomalii jest w jednej, dwóch cechach.
- Sama autokorelacja to za mało dla kNN – przy φ = 0,9 wykrywa mniej niż przy 0,7, choć AUC pozostaje wysokie
  (0,95): rozkład tej jednej cechy na szeregach bez anomalii ma ciężki ogon, więc próg wychodzi wysoki.

**Wniosek metodyczny:** dla anomalii „strukturalnych”, w których średnia i wariancja się nie zmieniają, kluczowe jest,
żeby metoda **jawnie** mierzyła strukturę (autokorelację, różnice sąsiednich punktów) i żeby ten sygnał nie ginął
wśród cech nieistotnych. Ta analiza (wykonana przed konsultacją 27.10) była podstawą pytania o zestaw cech; od tej pory
oba zestawy (5 i 3 cechy) są liczone w części głównej każdego scenariusza (rozdz. 3).

---

## 6. Weryfikacja hipotez z planu

Hipotezy z [plan_etap_1.md](plan_etap_1.md), rozdz. 4.6 (✔ dobrze, ◐ częściowo, – nie) vs wynik
(średni odsetek wykrytych / AUC):

| Metoda | S1-A: hipoteza → wynik | S1-B: hipoteza → wynik | S1-C: hipoteza → wynik |
|---|---|---|---|
| CUSUM | ✔ → ✔ 76% / 0,90 | ✔ → ✔ 84% / 0,92 | ✔ → ◐ 39% / 0,88 (✔ dla φ = 0,9 przy L ≥ 50 i φ ≥ 0,7 przy L = 100) |
| PELT | ✔ → ✔ 66% / 0,81 | ✔ → ✔ 88% / 0,90 | ◐ → ◐ 19% / 0,55 |
| Matrix Profile bez normalizacji | ✔ → ✔ 64% / 0,90 | ✔ → ✔ 91% / 0,99 | ◐ → **–** 6% / 0,46 |
| Matrix Profile z normalizacją | – → – 0% | – → – 0% | ◐ → **–** 0% |
| cechy okien + kNN (5 / 3 cechy) | ✔ → ✔ 57% / 77% (AUC 0,94 / 0,95) | ✔ → ✔ 76% / 87% (AUC 0,97 / 0,98) | ✔ → **◐** 11% / 42% (AUC 0,83 / 0,85) |
| Isolation Forest (5 / 3 cechy) | ✔ → **◐** 26% / 40% (AUC 0,94 / 0,95) | ✔ → ✔ / ◐ 75% / 45% (AUC 0,98 / 0,98) | ◐ → **–** 2% / 3% (AUC 0,84 / 0,86) |
| LOF (5 / 3 cechy) | ✔ → ✔ 63% / 68% (AUC 0,92 / 0,92) | ✔ → ✔ 72% / 78% (AUC 0,91 / 0,93) | ◐ → ◐ 26% / 36% (AUC 0,81 / 0,82) |

Metody ML – w wersji uczonej na tym samym szeregu (jak w planie); wersja uczona na szeregach bez anomalii – rozdz. 3.

**Potwierdzone:** S1-C jest najtrudniejsza i wykrywają ją tylko metody, które jawnie patrzą na zależność (CUSUM,
cecha autokorelacji); Matrix Profile z normalizacją nie widzi zmian poziomu i zmienności.

**Nie potwierdzone (ciekawe do pracy):**
- **Isolation Forest słabszy, niż zakładano** – nie z powodu słabego rankingu (AUC 0,84–0,98), lecz przez
  maskowanie długich anomalii i to, że jego wynik ma mały zakres (próg blisko maksimum). Uczenie na szeregach bez
  anomalii usuwa większość tego efektu.
- **Matrix Profile nie wykrywa zależności** – fragment AR(1) jest gładszy, ma mniej „stopni swobody”, więc jego
  odległości do innych fragmentów są bardziej rozrzucone i najbliższy sąsiad bywa bliżej niż dla szumu. Discord
  (fragment najdalszy od najbliższego sąsiada) z zasady szuka czegoś odwrotnego. Sprawdzono, że to nie wynik zbyt
  małej strefy wykluczenia (strefa = całe okno zamiast ¼ okna daje AUC 0,41 zamiast 0,38 przy φ = 0,9).
- **kNN na 5 cechach słabo wykrywa zależność** – przyczyną są cechy nieistotne (rozdz. 5), a nie sama metoda.

---

## 7. Zmiany wprowadzone w trakcie prac

Dla przejrzystości – co zmieniono względem planu i pilota, i dlaczego:

| Co | Zmiana | Powód |
|---|---|---|
| **Zestawy cech (konsultacja 27.10)** | kNN, Isolation Forest i LOF w dwóch wariantach: 5 cech (plan) i 3 cechy (po jednej na rodzaj anomalii) | w S1 3 cechy były lepsze dla kNN, ale w innych scenariuszach może być odwrotnie – w każdym scenariuszu oba warianty |
| **Uczenie ML (konsultacja 27.10)** | Isolation Forest i LOF także uczone na oknach 20 osobnych szeregów bez anomalii (własne ziarna, okna bez nakładania się) | standardowy sposób użycia metod wykrywania nowości; maskowanie w Isolation Forest |
| **Wyniki per L (konsultacja 27.10)** | odsetek wykrytych według poziomu intensywności – osobna tabela dla każdej długości L (zamiast średniej po L) | średnia po L zaciera silną zależność wyników od długości anomalii |
| CUSUM – lokalizacja | wynik = wysokość szczytu karty na odcinku od ostatniego zera do szczytu; po szczycie 0 | zwykła karta po silnej anomalii opada tylko o k na punkt i „ciągnęła” alarm setki punktów za anomalię; w szeregach z anomaliami dawało to 0,84 fałszywego alarmu na szereg (przy 0,1 bez anomalii) |
| CUSUM – karty zmienności i zależności | liczone na **różnicach** sąsiednich punktów zamiast na x² i iloczynie x_t·x_{t−1} | zmiana poziomu o Δ podnosiła iloczyn o ~Δ² – karta zależności po silnej anomalii A opadała przez ~1 500 punktów i zasłaniała anomalię C; po zmianie wykrywanie C przy φ = 0,9 (L = 50) wzrosło z 51% do 95%; koszt: najsłabsza zmiana zmienności 63% → 50% i więcej alarmów przesuniętych przed anomalię |
| PELT | własna implementacja (wektorowa) zamiast biblioteki `ruptures` | ten sam wynik (sprawdzono: identyczne granice w 12/12 przypadków testowych), ok. 140× szybciej (0,05 s zamiast 6,5 s na szereg) |
| PELT – koszt | koszt normalny na (x, x·x₋₁) zamiast osobnego kosztu AR | ten sam efekt (zmiana średniej iloczynu = zależność), dużo szybciej; kara dobierana na szeregach bez anomalii (pierwotnie 60, po poprawce z 2026-10-27 – 62) |
| Ocena | epizody alarmów, strefa tolerancji ± 25 pkt, próg na stały odsetek fałszywych alarmów, walidacja na nowych szeregach | poprawki po pilocie (plan, rozdz. 8) |
| Poziomy φ | dodano φ = 0,3 | żeby każdy rodzaj anomalii miał 4 poziomy intensywności |
| **Skala szumu (2026-10-27)** | σ̂ = pierwiastek z mediany wariancji okien po 50 punktów zamiast 1,4826·MAD (`odporna_skala` w `src/metody/baza.py`) | przegląd projektu wykazał, że długie anomalie zawyżają MAD (szczegóły niżej) |
| **PELT – wynik i kara (2026-10-27)** | odcinek porównywany z **tłem** (najdłuższe odcinki obejmujące ≥ 50% szeregu), wynik tylko dla odcinków istotnie innych od tła (z > 3), kara ustalana tak, żeby fałszywy alarm był w ≤ 10% szeregów bez anomalii (wyszło 62) | j.w. – długie odcinki normy dostawały wysoki wynik |

**Przeliczenie po przeglądzie projektu (2026-10-27).** Przegląd wykazał dwa powiązane błędy; cały eksperyment S1
(i S2, S3) przeliczono ponownie z tymi samymi ziarnami.

1. **Skala szumu.** CUSUM dzieli szereg przez σ̂. Wcześniej σ̂ = 1,4826·MAD całego szeregu; anomalie zbiorowe
   (do 10% punktów w S1) wprawdzie nie „psują” MAD, ale go zawyżają – przy L = 100 wariancja normy po standaryzacji
   wynosiła ok. 0,86 zamiast 1. Karta zależności CUSUM (statystyka −zd, k = 0,15) dryfowała przez to w górę na zwykłych
   fragmentach szeregu: w szeregach z długimi anomaliami fałszywych alarmów było dwa razy więcej (0,21 na szereg przy
   L = 100 wobec 0,11 przy L = 20; po poprawce: 0,06 i 0,07), a część „wykrytych” zależności była skutkiem tego dryfu.
   Teraz σ̂ to pierwiastek z mediany wariancji kolejnych okien po 50 punktów – okno z anomalią to jedno z 60 okien.
2. **PELT.** Wynik odcinka był statystyką z względem parametrów całego szeregu, zaburzonych przez anomalie, a statystyka
   rośnie jak √n – długie odcinki normy dostawały wysoki wynik. W sprawdzeniu na 40 szeregach (poziom 3, L = 100) nad
   progiem leżało **44% punktów normy** (poza strefami anomalii). Fałszywych alarmów to nie dawało (takie odcinki
   stykają się ze strefą anomalii), ale „wykrywało” anomalie przypadkiem, gdy odcinek normy obejmował ich strefę.
   Teraz odcinek porównujemy z tłem, liczy się tylko odcinek istotnie inny od tła, a karę ustala ta sama zasada
   10% fałszywych alarmów co próg innych metod.

Pozostałe metody nie używają σ̂ ani PELT – ich wyniki są identyczne co do bitu (sprawdzono). Zmiany:

| | CUSUM przed → po | PELT przed → po |
|---|---|---|
| odsetek wykrytych A / B / C | 77 / 84 / 42% → 76 / 84 / 39% | 67 / 88 / 30% → 66 / 88 / 19% |
| AUC A / B / C | 0,90 / 0,91 / 0,87 → 0,90 / 0,92 / 0,88 | 0,82 / 0,90 / 0,59 → 0,81 / 0,90 / 0,55 |
| fałszywe alarmy: walidacja / na szereg z anomaliami | 14,5% / 0,14 → 10,0% / 0,07 | 9,5% / 0,04 → 9,5% / 0,04 |
| precyzja / F1 | 0,79 / 0,60 → 0,89 / 0,65 | 0,71 / 0,48 → 0,88 / 0,58 |

Wnioski jakościowe się nie zmieniają (CUSUM nadal wśród najlepszych, kolejność trudności anomalii ta sama), ale obie metody
są teraz **uczciwiej** ocenione: mniej wykrytych zależności, za to dużo mniej alarmów w złym miejscu (precyzja
0,71–0,79 → 0,88–0,89). Spadek wykrywania zależności przez PELT (30% → 19%) pokazuje, ile z poprzedniego wyniku
było przypadkowe.

**Uzupełnienie po konsultacji (2026-10-27).** Warianty 3-cechowe i uczone na szeregach bez anomalii policzono na tych
samych szeregach i konfiguracjach (części: główna, okno, szum); warianty 5-cechowe kNN, Isolation Forest i LOF
policzone ponownie dały wyniki identyczne co do bitu z poprzednią wersją (sprawdzone skryptem). Nazwy metod na cechach
mają teraz dopisek „(5 cech)” / „(3 cechy)”.

---

## 8. Ograniczenia i wnioski na S2 i S3

**Ograniczenia:**
- Próg kalibrowany na 200 szeregach – na nowych szeregach odsetek fałszywych alarmów wynosi 4–16,5%
  zamiast dokładnie 10%. Metoda o progu łagodniejszym ma przez to lekko zawyżoną czułość (najbardziej LOF uczony na
  szeregach bez anomalii), o ostrzejszym – zaniżoną (Isolation Forest na 3 cechach); dlatego obok odsetka wykrytych
  podawane jest AUC, które od progu nie zależy. Dokładniejsze sprawdzenie na 1 000 szeregach: odsetek 5,7–15,7% (CUSUM 9,8%, PELT 8,6%, Matrix Profile 12,4%, kNN 9,7% / 11,1% (5 / 3 cechy), Isolation Forest 9,6% / 5,7%, LOF 11,6% / 13,2%; uczone na szeregach bez anomalii: Isolation Forest 11,1% / 7,8%, LOF 15,7% / 13,7%; `wyniki/walidacja_progow.md`).
- Niepewność wyników: odsetek wykryty na 100 anomaliach (jedna komórka tabeli poziom × L) ma 95% przedział ufności
  do ±10 punktów procentowych, średnia dla rodzaju (1 200 anomalii) – do ±3 punktów. Różnice mniejsze niż kilka punktów
  między metodami nie są rozstrzygające.
- Wszystkie trzy anomalie w szeregu mają ten sam poziom intensywności – silna anomalia może wpływać na wykrywanie
  słabszej w tym samym szeregu (to właśnie ujawniło problem karty zależności CUSUM).
- Wersja główna metod ML i kNN uczy się na tym samym szeregu, w którym szuka anomalii (zgodnie z planem); wersja ML
  uczona na szeregach bez anomalii usuwa większość maskowania, ale zakłada dostęp do danych bez anomalii.

**Rekomendacje na S2 (AR(1)) i S3 (szum + sinus):**
1. **Cechy okien dobierać pod anomalie danego scenariusza** i porównywać zestaw 3-cechowy (po jednej cesze na anomalię)
   z 5-cechowym – w S1 cechy pomocnicze szkodzą kNN i LOF, ale pomagają Isolation Forest dla zmiany zmienności.
2. **Statystyki odporne na przeciekanie** – każdą kartę / cechę sprawdzać, czy nie reaguje na INNE rodzaje anomalii
   (test `test_karty_cusum_nie_przeciekaja` w `tests/test_s1.py` jako wzór).
3. **Metody ML** – liczyć także w wersji uczonej na szeregach bez anomalii (Isolation Forest uczony na tym samym
   szeregu nie nadaje się do długich anomalii).
4. **Długość okna** dobierać do spodziewanej długości anomalii (w S3 – do długości cyklu, zgodnie z planem).
5. Zachować te same zasady oceny (wspólny próg, walidacja, AUC + odsetek wykrytych), żeby wyniki S1, S2 i S3 dało się
   porównać.

---

## 9. Jak odtworzyć wyniki

Z katalogu `Projekt-Inz`:

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # testy S1–S3: generatory, ocena, metody
python -m src.ewaluacja.eksperyment_s1 --szybko # szybkie sprawdzenie całego łańcucha (ok. 6 min)
python -m src.ewaluacja.eksperyment_s1          # pełny eksperyment (ok. 1 h na 2 rdzeniach) → wyniki/s1/*.csv
python -m src.ewaluacja.wykresy_s1              # wykresy → wyniki/wykresy/s1_*.png, tabele → wyniki/s1/podsumowanie.md
python -m src.ewaluacja.walidacja_progow s1     # progi na 1 000 nowych szeregach → wyniki/walidacja_progow.md
```

Ziarna losowe są stałe – ponowne uruchomienie daje identyczne liczby.

| Plik | Zawartość |
|---|---|
| `src/symulacje/generator.py`, `anomalie.py` | szereg tła, anomalie S1-A/B/C, rozmieszczenie, etykiety |
| `src/metody/` | wspólny interfejs (`baza.py`, zestawy cech `KOLUMNY_3`) i 6 metod (`statystyczne.py`, `fragmenty.py`, `ml.py` – także uczenie na szeregach bez anomalii) |
| `src/ewaluacja/metryki.py` | epizody alarmów, strefa tolerancji, kalibracja progu, AUC, AP |
| `src/ewaluacja/eksperyment_s1.py` | eksperyment (części: główna, okno, szum, analiza dodatkowa) |
| `src/ewaluacja/wykresy_s1.py`, `warianty_metod.py` | wykresy i tabele (wspólne funkcje dla S1–S3), nazwy i styl wariantów metod |
| `wyniki/s1/*_anomalie.csv` | wiersz = metoda × szereg × anomalia (wykryta, AUC, AP) |
| `wyniki/s1/*_szeregi.csv` | wiersz = metoda × szereg (fałszywe alarmy, precyzja, czułość punktowa, F1) |
| `wyniki/s1/*_czyste.csv`, `*_kalibracja.csv` | walidacja fałszywych alarmów, progi i stałe metod |
| `wyniki/s1/podsumowanie.md` | wszystkie tabele liczbowe |
| `wyniki/wykresy/s1_*.png` | 8 wykresów (w tym `s1_uczenie.png`) |
