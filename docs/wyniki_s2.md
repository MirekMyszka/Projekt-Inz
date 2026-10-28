# Wyniki – scenariusz S2: AR(1) (szereg z zależnością)

> Stan na **2026-10-27** (po przeliczeniu z poprawioną skalą szumu i PELT oraz po uwagach z konsultacji 27.10: warianty
> 5 i 3 cech, metody ML uczone także na szeregach bez anomalii, wyniki osobno dla każdego L – rozdz. 7). Realizacja kroku 7 z [plan_etap_1.md](plan_etap_1.md), rozdz. 5 – według tych samych zasad co S1
> ([wyniki_s1.md](wyniki_s1.md)). Wszystkie liczby pochodzą z `wyniki/s2/podsumowanie.md` (tabele generowane
> automatycznie); polecenia do odtworzenia – rozdz. 9. Szczegółowy opis każdej metody: [metody.md](metody.md).

## Spis treści

1. [Co zbadano](#1-co-zbadano)
2. [Najważniejsze wnioski](#2-najważniejsze-wnioski)
3. [Wyniki dla każdego rodzaju anomalii](#3-wyniki-dla-każdego-rodzaju-anomalii) (w tym zestaw cech i uczenie ML)
4. [Wpływ długości anomalii, długości okna i siły zależności tła](#4-wpływ-długości-anomalii-długości-okna-i-siły-zależności-tła)
5. [Analizy dodatkowe](#5-analizy-dodatkowe)
6. [Weryfikacja hipotez z planu i porównanie z S1](#6-weryfikacja-hipotez-z-planu-i-porównanie-z-s1)
7. [Zmiany wprowadzone w trakcie prac](#7-zmiany-wprowadzone-w-trakcie-prac)
8. [Ograniczenia i wnioski na S3](#8-ograniczenia-i-wnioski-na-s3)
9. [Jak odtworzyć wyniki](#9-jak-odtworzyć-wyniki)

---

## 1. Co zbadano

**Szereg tła:** AR(1) `x_t = μ + φ·(x_{t−1} − μ) + ε_t`, φ = 0,7, μ = 0, wariancja szeregu σ_x² = 1
(czyli σ_ε = √(1 − φ²) ≈ 0,71), n = 3 000. Każdy punkt zależy od poprzedniego – szereg „błądzi” i powoli wraca do
średniej (autokorelacja 0,7).

**Anomalie** – w każdym szeregu wszystkie trzy, rozmieszczone jak w S1 (po jednej w każdej trzeciej części szeregu):

| | Anomalia | Jak wstawiona | Poziomy intensywności 1 / 2 / 3 / 4 |
|---|---|---|---|
| **S2-A** | szok wygasający (*innovational outlier*) | do jednej innowacji ε dodany impuls; ślad maleje jak φ^k | impuls = 2 / 3 / 5 / 8 σ_ε (≈ 1,4 / 2,1 / 3,6 / 5,7 σ_x) |
| **S2-B** | zmiana siły zależności | na fragmencie φ → φ', wariancja szeregu **bez zmian** | φ' = 0,5 / 0,3 / 0 / −0,5 |
| **S2-C** | przesunięcie poziomu równowagi | na fragmencie μ → Δ·σ_x; szereg dochodzi do nowego poziomu **stopniowo** i stopniowo wraca | Δ = 0,5 / 1 / 2 / 3 |

Długość B i C: L ∈ {20, 50, 100}. **Szok ma zawsze długość 10** – tyle punktów trwa, zanim jego ślad spadnie poniżej
5% (0,7^9 ≈ 0,04); wynika to z dynamiki szeregu, a nie z L. Względem planu dodano po jednym słabszym poziomie każdej
anomalii, a zmianę φ uporządkowano jako słabnięcie zależności (od 0,5 do odwrócenia −0,5), żeby intensywność rosła
monotonicznie; wzrost zależności (φ' = 0,95) zbadano osobno (rozdz. 5).

**Metody** – po 2 z każdej grupy, dobrane do AR(1) (plan, rozdz. 4.2 i 4.4, z poprawkami po pilocie):

| Grupa | Metoda | Co „ogląda” |
|---|---|---|
| statystyczne | **reszty AR + CUSUM** | reszty modelu AR(1): karty poziomu, zmienności i zależności z S1 + test szoku (\|reszta\|) |
| statystyczne | **PELT (koszt AR)** | podział szeregu na odcinki z własnym modelem AR(1) |
| fragmenty szeregu | **Matrix Profile (reszty AR)** | fragmenty reszt modelu AR(1), bez normalizacji, okno 50 |
| fragmenty szeregu | **cechy okien + kNN** | cechy okien: 5 albo 3 (niżej) |
| uczenie maszynowe | **Isolation Forest** | te same cechy |
| uczenie maszynowe | **One-Class SVM** | te same cechy (jądro RBF, gamma = 0,01 – rozdz. 7) |

**Warianty (ustalenia z konsultacji 27.10.2026):**
- **zestaw cech** kNN, Isolation Forest i One-Class SVM: **3 cechy** – po jednej na rodzaj anomalii: maks. \|reszta AR\|
  (szok), autokorelacja x z opóźnieniem 1 (zmiana φ), średnia x (przesunięcie równowagi); **5 cech** – te 3 + odch. std
  reszt AR i autokorelacja x z opóźnieniem 2 (cechy pomocnicze z planu). Do 27.10 używany był zestaw 4 cech (3 + odch.
  std reszt);
- **uczenie** Isolation Forest i One-Class SVM: na oknach **tego samego szeregu** (wersja nienadzorowana) albo na oknach
  **20 osobnych szeregów AR(1) bez anomalii** (wersja półnadzorowana – [metody.md](metody.md), rozdz. 0.8).

**Ocena** – identyczna jak w S1: próg każdej metody ustalony na 200 szeregach AR(1) bez anomalii (fałszywy alarm w 10%
z nich), sprawdzony na 200 nowych szeregach – **rzeczywisty odsetek 6,5–13,5%**; wykrycie = alarm
w strefie anomalia ± 25 punktów; AUC jako miara niezależna od progu.

**Skala:** część główna 4 poziomy × 3 długości × 100 = **1 200 szeregów (3 600 anomalii na metodę)**; dodatkowo:
długość okna (900 szeregów), siła zależności tła (300), analizy dodatkowe (a)–(d) (1 000).

---

## 2. Najważniejsze wnioski

1. **Zależność w szeregu utrudnia wykrywanie.** Średnio metody (wersje uczone na tym samym szeregu) wykrywają
   4–42% szoków, 14–66% zmian zależności i 0–45%
   przesunięć równowagi. Naturalne, długie wahania AR(1) łatwo pomylić z anomalią, więc próg (przy tym samym odsetku
   fałszywych alarmów) musi być wyżej.

2. **Najlepsza metoda ogólnie: reszty AR + CUSUM** – najwyższe wykrywanie zmian zależności (66%, na równi
   z Matrix Profile na resztach – 66%) i przesunięć równowagi (45%), najwyższe F1 (0,52).
   Przy Δ = 2σ_x i L = 50 wykrywa 93% przesunięć równowagi; przy φ' = 0 – 98% zmian φ.
   Tak jak w S1, działa, bo każda karta patrzy na to, co zmienia dana anomalia.

3. **Szok najlepiej wykrywają kNN (oba zestawy cech: 42% / 42%) i One-Class SVM uczony na
   szeregach bez anomalii (41% / 42%)**; impuls 5σ_ε: kNN 60%, impuls 8σ_ε:
   100%. Pojedynczy szok wielkości 5σ_ε to w resztach jedna wartość ≈ 5 – niewiele ponad typowe maksimum
   3 000 punktów szumu (≈ 3,7) – więc metody patrzące na pojedynczą resztę wykrywają go rzadko (CUSUM 23%).
   kNN łączy maksimum reszty z innymi cechami okna i to wystarcza.

4. **W S2 lepsze jest 5 cech – odwrotnie niż w S1.** kNN wykrywa zmianę φ w 60% przypadków na 5 cechach
   i 48% na 3 cechach; wzrost zależności (φ' = 0,95, analiza b) – 87% wobec 8%.
   Cechy pomocnicze (odch. std reszt AR, autokorelacja z opóźnieniem 2) niosą tu informację o zmianie dynamiki, której
   jedna autokorelacja z opóźnieniem 1 nie oddaje. Dla szoku i przesunięcia równowagi zestaw ma mały wpływ. Potwierdza to
   ustalenie z konsultacji: zestaw cech trzeba sprawdzać w każdym scenariuszu osobno.

5. **Uczenie na szeregach bez anomalii najbardziej pomaga One-Class SVM**: 4% / 17% /
   0% → 41% / 60% / 38% (A / B / C, 5 cech), F1 0,04 →
   0,44. W wersji uczonej na tym samym szeregu najsilniejsza zmiana φ (φ' = −0,5) była wykrywana rzadziej niż
   słabsza (31% vs 64% przy L = 50); po uczeniu na szeregach bez anomalii wynik rośnie
   z intensywnością (96% / 100%). To wskazuje, że kłopot brał się głównie z anomalii
   obecnych w danych uczących (granica modelu obejmowała także ich okna), a nie tylko z nasycenia jądra (rozdz. 5c).
   Isolation Forest zyskuje głównie dla przesunięcia równowagi (1% → 16%).

6. **Im silniejsza zależność tła, tym trudniej wykryć przesunięcie poziomu – i tym łatwiej zmianę zależności.**
   Przy tym samym przesunięciu (Δ = 2σ_x) CUSUM wykrywa je w 100% (φ = 0,3), 90%
   (φ = 0,7) i 23% (φ = 0,9) przypadków. Silnie zależny szereg sam „wędruje”, więc przesunięcie
   ginie w naturalnych wahaniach, a do nowego poziomu dochodzi powoli (stała czasowa 1/(1 − φ) = 10 punktów przy
   φ = 0,9). Zmiana φ → 0 jest za to tym wyraźniejsza, im większe było φ.

7. **Model AR nie zawsze pomaga** (analiza a). CUSUM z S1 zastosowany wprost do szeregu (bez modelu) wykrywa szoki
   lepiej (34% vs 14% przy poziomach 2–3), przesunięcia równowagi
   podobnie (56% vs 52%), a zmiany zależności gorzej
   (57% vs 82%). Model „tłumaczy” część zmiany poziomu jako
   zwykłą dynamikę – w resztach zostaje tylko (1 − φ)·Δ. Model jest **niezbędny** do wykrywania zmian zależności
   (Matrix Profile: 0% bez modelu vs 87% z modelem).

8. **Isolation Forest – maskowanie także MIĘDZY anomaliami.** Szok ma zawsze 10 punktów, a mimo to Isolation Forest
   (5 cech, uczony na tym samym szeregu) wykrywa go tym rzadziej, im dłuższe są POZOSTAŁE anomalie w szeregu:
   17% / 5% / 1% przy L = 20 / 50 / 100. Przesunięcia równowagi
   praktycznie nie wykrywa (1%), choć AUC = 0,82. kNN jest na to odporny (45% /
   42% / 38%), One-Class SVM uczony na szeregach bez anomalii – też
   (44% / 40% / 38%).

9. **Matrix Profile nie widzi WZROSTU zależności** (φ' = 0,95: 0% wykrytych, AUC = 0,03 –
   odwrócony ranking), choć spadek zależności wykrywa w 97%. To ten sam mechanizm co dla S1-C: fragment
   bardziej zależny / gładszy ma mniejsze reszty i bliższych sąsiadów, a Matrix Profile szuka fragmentów odległych od
   wszystkich innych.

10. **Wnioski z S1 potwierdzają się w S2:** okno powinno pasować do długości anomalii (krótki szok – okno 20 lepsze
    niż 100 dla Matrix Profile, Isolation Forest i One-Class SVM), dłuższa anomalia jest łatwiejsza (dlatego wyniki
    podajemy osobno dla każdego L), a dobór cech ma ogromne znaczenie (kNN z cechami S1 wykrywa
    16% zmian zależności, z cechami S2 – 80% / 58%
    przy 5 / 3 cechach).

---

## 3. Wyniki dla każdego rodzaju anomalii

Wykresy: `wyniki/wykresy/s2_czulosc_intensywnosc.png` (wiersz = długość L), `s2_auc_intensywnosc.png`,
`s2_uczenie.png`, `s2_przyklad.png`. Wszystkie tabele (także AUC dla każdego L) – `wyniki/s2/podsumowanie.md`.

**Odsetek wykrytych anomalii – rodzaj × poziom intensywności, osobno dla każdej długości L** (A: impuls 2 / 3 / 5 /
8 σ_ε; B: φ' = 0,5 / 0,3 / 0 / −0,5; C: Δ = 0,5 / 1 / 2 / 3 σ_x; pogrubienie – najlepszy wynik w kolumnie):

**L = 20** (długość B i C; szok zawsze 10 punktów; 100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 2% | 1% | 26% | 98% | 8% | **27%** | **56%** | **85%** | 0% | 0% | **12%** | **58%** |
| PELT (koszt AR) | 3% | 0% | 4% | 85% | 4% | 6% | 32% | 76% | 0% | 0% | 6% | 45% |
| Matrix Profile (reszty AR) | 1% | 4% | 24% | 94% | 7% | 24% | 45% | 74% | 0% | 0% | 7% | 40% |
| kNN (5 cech) | 0% | 10% | **69%** | 100% | 3% | 12% | 43% | 78% | 0% | 0% | 7% | 54% |
| kNN (3 cechy) | 0% | **12%** | **69%** | 100% | 2% | 8% | 30% | 68% | 1% | 0% | 5% | 48% |
| Isolation Forest (5 cech) | 0% | 3% | 12% | 54% | 3% | 4% | 32% | 51% | 0% | 0% | 1% | 4% |
| Isolation Forest (3 cechy) | 1% | 7% | 29% | 66% | 1% | 2% | 15% | 39% | 1% | 0% | 1% | 8% |
| One-Class SVM (5 cech) | 0% | 3% | 19% | 23% | 2% | 3% | 9% | 5% | 0% | 0% | 0% | 0% |
| One-Class SVM (3 cechy) | 0% | 4% | 38% | 50% | 1% | 4% | 11% | 14% | 0% | 0% | 0% | 0% |

**L = 50** (długość B i C; szok zawsze 10 punktów; 100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 1% | 2% | 26% | 97% | 16% | 67% | **98%** | 100% | 1% | **10%** | **93%** | 100% |
| PELT (koszt AR) | 0% | 1% | 6% | 87% | 1% | 40% | 90% | 100% | 0% | 1% | 66% | 100% |
| Matrix Profile (reszty AR) | 1% | 3% | 27% | 98% | **19%** | **77%** | 97% | 99% | 0% | 3% | 25% | 86% |
| kNN (5 cech) | 1% | 8% | **61%** | 100% | 11% | 66% | 94% | 100% | 0% | 2% | 75% | 100% |
| kNN (3 cechy) | 1% | 8% | **61%** | 100% | 5% | 32% | 84% | 100% | 0% | 2% | 84% | 100% |
| Isolation Forest (5 cech) | 1% | 1% | 1% | 18% | 7% | 54% | 90% | 99% | 0% | 0% | 2% | 4% |
| Isolation Forest (3 cechy) | 1% | 3% | 9% | 38% | 2% | 13% | 36% | 57% | 1% | 1% | 11% | 34% |
| One-Class SVM (5 cech) | 1% | 0% | 0% | 0% | 6% | 44% | 64% | 31% | 0% | 0% | 3% | 1% |
| One-Class SVM (3 cechy) | 1% | 1% | 3% | 1% | 3% | 15% | 42% | 43% | 0% | 1% | 29% | 22% |

**L = 100** (długość B i C; szok zawsze 10 punktów; 100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reszty AR + CUSUM | 1% | 2% | 17% | 95% | 43% | **95%** | 100% | 100% | 4% | **62%** | 100% | 100% |
| PELT (koszt AR) | 3% | 1% | 4% | 80% | 17% | 88% | 100% | 100% | 2% | 18% | 100% | 100% |
| Matrix Profile (reszty AR) | 1% | 5% | 20% | 91% | **58%** | 93% | 100% | 100% | 1% | 3% | 35% | 86% |
| kNN (5 cech) | 0% | 5% | 50% | **99%** | 34% | 77% | 100% | 100% | 2% | 12% | 90% | 100% |
| kNN (3 cechy) | 0% | 5% | **54%** | **99%** | 9% | 45% | 96% | 100% | 3% | 19% | 97% | 100% |
| Isolation Forest (5 cech) | 0% | 0% | 0% | 3% | 15% | 62% | 84% | 78% | 0% | 0% | 3% | 2% |
| Isolation Forest (3 cechy) | 0% | 0% | 3% | 17% | 5% | 10% | 27% | 10% | 4% | 2% | 5% | 6% |
| One-Class SVM (5 cech) | 0% | 0% | 0% | 0% | 7% | 27% | 11% | 0% | 1% | 0% | 0% | 0% |
| One-Class SVM (3 cechy) | 1% | 1% | 0% | 0% | 5% | 16% | 19% | 0% | 1% | 0% | 2% | 0% |

Przy L = 20 zmiana φ i przesunięcie równowagi są trudne dla wszystkich metod (najsilniejsze przesunięcie Δ = 3σ_x:
CUSUM 58%, kNN 5 cech 54%); przy L = 100 – CUSUM 100% / 100%.
Szok nie zależy od L (zawsze 10 punktów) – poza metodami, które maskują go długimi pozostałymi anomaliami (rozdz. 2, wniosek 8).

**Średnio dla rodzaju** (wszystkie poziomy i L; odsetek wykrytych / AUC):

| metoda | A – szok | B – zmiana φ | C – równowaga |
|---|---|---|---|
| reszty AR + CUSUM | 31% / 0,61 | **66% / 0,79** | **45% / 0,82** |
| PELT (koszt AR) | 23% / 0,56 | 55% / 0,75 | 36% / 0,66 |
| Matrix Profile (reszty AR) | 31% / 0,80 | **66% / 0,92** | 24% / 0,75 |
| kNN (5 cech) | **42% / 0,81** | 60% / 0,91 | 37% / 0,82 |
| kNN (3 cechy) | **42% / 0,81** | 48% / 0,87 | 38% / 0,83 |
| Isolation Forest (5 cech) | 8% / 0,80 | 48% / 0,90 | 1% / 0,82 |
| Isolation Forest (3 cechy) | 14% / 0,81 | 18% / 0,86 | 6% / 0,84 |
| One-Class SVM (5 cech) | 4% / 0,75 | 17% / 0,81 | 0% / 0,80 |
| One-Class SVM (3 cechy) | 8% / 0,74 | 14% / 0,71 | 5% / 0,80 |

AUC szoku jest niskie dla metod patrzących na pojedyncze punkty (CUSUM 0,61, PELT 0,56): etykieta obejmuje
10 punktów wygasania, a duża jest tylko pierwsza reszta – pozostałe 9 punktów model AR przewiduje poprawnie.

**Zestaw cech: 5 czy 3?** (odsetek wykrytych, w nawiasie AUC; średnio po poziomach i L)

| metoda | A: 5 cech → 3 cechy | B: 5 cech → 3 cechy | C: 5 cech → 3 cechy |
|---|---|---|---|
| kNN | 42% → 42% (0,81 → 0,81) | 60% → 48% (0,91 → 0,87) | 37% → 38% (0,82 → 0,83) |
| Isolation Forest | 8% → 14% (0,80 → 0,81) | 48% → 18% (0,90 → 0,86) | 1% → 6% (0,82 → 0,84) |
| Isolation Forest – uczony bez anomalii | 11% → 15% (0,80 → 0,80) | 54% → 38% (0,90 → 0,86) | 16% → 18% (0,82 → 0,84) |
| One-Class SVM | 4% → 8% (0,75 → 0,74) | 17% → 14% (0,81 → 0,71) | 0% → 5% (0,80 → 0,80) |
| One-Class SVM – uczony bez anomalii | 41% → 42% (0,78 → 0,77) | 60% → 48% (0,88 → 0,80) | 38% → 39% (0,85 → 0,85) |

- **Zmiana φ: 5 cech wyraźnie lepsze** dla kNN, Isolation Forest i wariantów uczonych na szeregach bez anomalii (kNN 48% → 60% przy przejściu z 3 na 5
  cech; wzrost φ w analizie (b): 8% → 87%). Prawdopodobna przyczyna: autokorelacja
  z opóźnieniem 1 szacowana na 50 punktach ma duży rozrzut, a odch. std reszt modelu dopasowanego do całego szeregu
  zmienia się wyraźnie, gdy na fragmencie model przestaje pasować: wariancja reszt wynosi 1 − φ² = 0,51 w normie,
  (0,7 − φ')² + 1 − φ'² w anomalii – 1,49 przy φ' = 0 i 0,16 przy φ' = 0,95.
- **Szok i przesunięcie równowagi**: różnice małe (kNN A 42% / 42%, C 38% / 37%);
  Isolation Forest wykrywa szok lepiej na 3 cechach (8% → 14%).
- W S1 było odwrotnie (3 cechy lepsze) – cechy pomocnicze S1 (kurtoza, rozstęp) nie niosły informacji o anomaliach.

**Uczenie metod ML: na tym samym szeregu czy na szeregach bez anomalii?** (odsetek wykrytych dla każdego L, średnio po
poziomach; ostatnia kolumna – średnie AUC)

| metoda | uczenie na | A L=20 | A L=50 | A L=100 | B L=20 | B L=50 | B L=100 | C L=20 | C L=50 | C L=100 | AUC (śr.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest (5 cech) | tym samym szeregu | 17% | 5% | 1% | 22% | 62% | 60% | 1% | 2% | 1% | 0,84 |
|  | szeregach bez anomalii | 14% | 12% | 7% | 28% | 60% | 75% | 13% | 17% | 17% | 0,84 |
| One-Class SVM (5 cech) | tym samym szeregu | 11% | 0% | 0% | 5% | 36% | 11% | 0% | 1% | 0% | 0,79 |
|  | szeregach bez anomalii | 44% | 40% | 38% | 33% | 68% | 79% | 17% | 44% | 51% | 0,84 |
| Isolation Forest (3 cechy) | tym samym szeregu | 26% | 13% | 5% | 14% | 27% | 13% | 2% | 12% | 4% | 0,83 |
|  | szeregach bez anomalii | 20% | 14% | 10% | 21% | 40% | 51% | 14% | 20% | 21% | 0,84 |
| One-Class SVM (3 cechy) | tym samym szeregu | 23% | 2% | 0% | 8% | 26% | 10% | 0% | 13% | 1% | 0,75 |
|  | szeregach bez anomalii | 45% | 42% | 39% | 27% | 54% | 62% | 15% | 47% | 56% | 0,81 |

- **One-Class SVM**: uczenie na szeregach bez anomalii zmienia metodę z najsłabszej w jedną z lepszych (5 cech:
  A 4% → 41%, B 17% → 60%, C 0% → 38%;
  F1 0,04 → 0,44) i usuwa spadek wykrywania przy długich anomaliach (B, L = 100: 11% →
  79%).
- **Isolation Forest** zyskuje mniej: przesunięcie równowagi 1% → 16%, zmiana φ przy L = 100
  60% → 75%; szok nadal słabo (11%).
- AUC rośnie mniej niż odsetek wykrytych – poprawia się głównie położenie wyniku anomalii względem progu.

**Fałszywe alarmy i dokładność lokalizacji:**

| metoda | szeregi bez anomalii z ≥ 1 fałszywym alarmem (walidacja) | fałszywe alarmy na szereg z anomaliami | precyzja | czułość punktowa | F1 |
|---|---|---|---|---|---|
| reszty AR + CUSUM | 10,5% | 0,05 | 0,93 | 0,47 | **0,52** |
| PELT (koszt AR) | 7,0% | 0,04 | 0,83 | 0,40 | 0,41 |
| Matrix Profile (reszty AR) | 12,0% | 0,17 | 0,92 | 0,27 | 0,36 |
| kNN (5 cech) | 11,5% | 0,06 | 0,95 | 0,35 | 0,42 |
| kNN (3 cechy) | 10,5% | 0,06 | 0,94 | 0,31 | 0,37 |
| Isolation Forest (5 cech) | 10,5% | 0,03 | 0,95 | 0,08 | 0,14 |
| Isolation Forest (3 cechy) | 9,0% | 0,03 | 0,90 | 0,03 | 0,04 |
| One-Class SVM (5 cech) | 12,5% | 0,03 | 0,86 | 0,02 | 0,04 |
| One-Class SVM (3 cechy) | 11,0% | 0,03 | 0,89 | 0,03 | 0,05 |
| Isolation Forest (5 cech, uczony bez anomalii) | 9,5% | 0,06 | 0,94 | 0,18 | 0,26 |
| Isolation Forest (3 cechy, uczony bez anomalii) | 6,5% | 0,04 | 0,94 | 0,15 | 0,20 |
| One-Class SVM (5 cech, uczony bez anomalii) | 13,5% | 0,09 | 0,93 | 0,38 | 0,44 |
| One-Class SVM (3 cechy, uczony bez anomalii) | 9,5% | 0,08 | 0,92 | 0,34 | 0,39 |

Kara PELT dobrana na szeregach bez anomalii wyniosła 18. PELT ma nieco niższą precyzję (0,83) niż metody okienkowe –
odcinki z innym modelem AR bywają szersze niż anomalia (np. łączą przesunięcie równowagi z jego stopniowym powrotem);
odsetek fałszywych alarmów na walidacji (7%) jest poniżej celu, bo zmienia się skokowo z karą (rozdz. 8).

---

## 4. Wpływ długości anomalii, długości okna i siły zależności tła

**Długość anomalii L** (`wyniki/wykresy/s2_czulosc_dlugosc.png`; średnio po poziomach):

| metoda | A (zawsze 10 pkt): L = 20 / 50 / 100 | B: L = 20 / 50 / 100 | C: L = 20 / 50 / 100 |
|---|---|---|---|
| reszty AR + CUSUM | 32 / 32 / 29% | 44 / 70 / 84% | 18 / 51 / 66% |
| PELT (koszt AR) | 23 / 24 / 22% | 30 / 58 / 76% | 13 / 42 / 55% |
| Matrix Profile (reszty AR) | 31 / 32 / 29% | 38 / 73 / 88% | 12 / 28 / 31% |
| kNN (5 cech) | 45 / 42 / 38% | 34 / 68 / 78% | 15 / 44 / 51% |
| kNN (3 cechy) | 45 / 42 / 40% | 27 / 55 / 62% | 14 / 46 / 55% |
| Isolation Forest (5 cech) | 17 / 5 / 1% | 22 / 62 / 60% | 1 / 2 / 1% |
| Isolation Forest (3 cechy) | 26 / 13 / 5% | 14 / 27 / 13% | 2 / 12 / 4% |
| One-Class SVM (5 cech) | 11 / 0 / 0% | 5 / 36 / 11% | 0 / 1 / 0% |
| One-Class SVM (3 cechy) | 23 / 2 / 0% | 8 / 26 / 10% | 0 / 13 / 1% |
| Isolation Forest (5 cech, uczony bez anomalii) | 14 / 12 / 7% | 28 / 60 / 75% | 13 / 17 / 17% |
| Isolation Forest (3 cechy, uczony bez anomalii) | 20 / 14 / 10% | 21 / 40 / 51% | 14 / 20 / 21% |
| One-Class SVM (5 cech, uczony bez anomalii) | 44 / 40 / 38% | 33 / 68 / 79% | 17 / 44 / 51% |
| One-Class SVM (3 cechy, uczony bez anomalii) | 45 / 42 / 39% | 27 / 54 / 62% | 15 / 47 / 56% |

Dla B i C – jak w S1 – dłuższa anomalia jest łatwiejsza. Kolumna A pokazuje maskowanie MIĘDZY anomaliami: szok ma
zawsze 10 punktów, a Isolation Forest i One-Class SVM uczone na tym samym szeregu wykrywają go tym rzadziej, im dłuższe
są pozostałe anomalie w szeregu (więcej nietypowych okien w danych uczących = „rozmyta” granica normy). Uczenie na
szeregach bez anomalii usuwa ten efekt dla One-Class SVM, dla Isolation Forest – tylko częściowo.

**Długość okna w** (`wyniki/wykresy/s2_okno.png`; poziomy 2–3, wszystkie L; metody uczone na tym samym szeregu):

| metoda | A: w = 20 / 50 / 100 | B: w = 20 / 50 / 100 | C: w = 20 / 50 / 100 |
|---|---|---|---|
| Matrix Profile (reszty AR) | 20 / 13 / 6% | 70 / 73 / 65% | 3 / 11 / 15% |
| kNN (5 cech) | 34 / 32 / 26% | 53 / 67 / 60% | 12 / 32 / 33% |
| kNN (3 cechy) | 34 / 33 / 29% | 25 / 46 / 49% | 14 / 36 / 35% |
| Isolation Forest (5 cech) | 14 / 4 / 1% | 49 / 59 / 52% | 1 / 1 / 1% |
| Isolation Forest (3 cechy) | 9 / 6 / 2% | 11 / 16 / 11% | 2 / 3 / 3% |
| One-Class SVM (5 cech) | 22 / 2 / 0% | 42 / 27 / 10% | 3 / 0 / 0% |
| One-Class SVM (3 cechy) | 26 / 6 / 0% | 15 / 16 / 15% | 7 / 6 / 2% |

Reguła z S1 się potwierdza: **okno porównywalne z długością anomalii**. Krótki szok (10 pkt) – najlepiej okno 20;
długie zmiany φ (L = 100) – okno 100 (kNN 5 cech 100%, Matrix Profile
99%); przesunięcie równowagi przy L = 100 – kNN (5 cech) 63%
z oknem 100 wobec 51% z oknem 50.

**Siła zależności tła φ** – właściwości szeregu (`wyniki/wykresy/s2_tlo.png`; stałe anomalie: szok 5σ_ε,
φ' = 0, Δ = 2σ_x; L = 50):

| metoda | A: φ = 0,3 / 0,7 / 0,9 | B (φ → 0): φ = 0,3 / 0,7 / 0,9 | C: φ = 0,3 / 0,7 / 0,9 |
|---|---|---|---|
| reszty AR + CUSUM | 33 / 29 / 24% | 12 / 97 / 100% | 100 / 90 / 23% |
| PELT (koszt AR) | 12 / 8 / 7% | 3 / 92 / 100% | 100 / 59 / 3% |
| Matrix Profile (reszty AR) | 29 / 26 / 29% | 5 / 95 / 100% | 99 / 33 / 5% |
| kNN (5 cech) | 68 / 63 / 39% | 5 / 97 / 100% | 100 / 73 / 3% |
| kNN (3 cechy) | 68 / 62 / 38% | 4 / 86 / 100% | 100 / 81 / 4% |
| Isolation Forest (5 cech) | 14 / 4 / 0% | 1 / 88 / 100% | 98 / 1 / 0% |
| Isolation Forest (3 cechy) | 8 / 9 / 1% | 1 / 35 / 100% | 92 / 16 / 1% |
| One-Class SVM (5 cech) | 1 / 1 / 0% | 0 / 62 / 8% | 96 / 2 / 0% |
| One-Class SVM (3 cechy) | 4 / 2 / 0% | 0 / 41 / 26% | 100 / 22 / 0% |

- **B:** zmiana φ → 0 to przy φ = 0,3 zmiana tylko o 0,3 – prawie niewidoczna; przy φ = 0,9 – o 0,9 – oczywista.
- **C:** to samo przesunięcie (2σ_x) jest łatwe w słabo zależnym szeregu i bardzo trudne w silnie zależnym
  (naturalne wędrówki szeregu + powolne dochodzenie do nowego poziomu + w resztach zostaje tylko (1 − φ)·Δ = 0,2σ_x).
- Odsetek fałszywych alarmów pozostaje 6–15% dla każdego φ – kalibracja na szeregach bez anomalii działa.

To bezpośrednia odpowiedź na pytanie z tematu pracy o wpływ **właściwości szeregu**: ta sama anomalia może być
łatwa albo prawie niewykrywalna zależnie od siły zależności tła.

---

## 5. Analizy dodatkowe

Wykres: `wyniki/wykresy/s2_dodatkowa.png`. Wszystkie przy L = 50, 100 szeregów na konfigurację, ten sam sposób
ustalania progu (odsetek fałszywych alarmów na walidacji 7–14,5%).

**(a) Czy model AR jest potrzebny?** Metody z S1 zastosowane wprost do szeregu AR(1) (bez modelu), poziomy 2–3,
średni odsetek wykrytych:

| metoda | A – szok | B – zmiana φ | C – równowaga |
|---|---|---|---|
| CUSUM z S1 (bez modelu) | 34% | 57% | 56% |
| reszty AR + CUSUM (S2) | 14% | 82% | 52% |
| Matrix Profile (bez modelu) | 2% | 0% | 44% |
| Matrix Profile (reszty AR) | 15% | 87% | 14% |
| kNN z cechami S1 (bez modelu) | 14% | 16% | 15% |
| kNN (S2, 5 cech) | 34% | 80% | 38% |
| kNN (S2, 3 cechy) | 34% | 58% | 43% |

Model AR jest niezbędny do wykrywania **zmian zależności**, ale „zjada” część sygnału **przesunięcia poziomu**
(w resztach zostaje (1 − φ)·Δ) i **szoku** (wygasający „garb” zamienia się w jedną resztę). CUSUM z S1 wykrywa
szok dzięki karcie zmienności liczonej na różnicach surowego szeregu – szok to jedna ogromna różnica.

**(b) Wzrost zależności φ' = 0,95** zamiast spadku (S2-B; poziomy 2–3):

| metoda | spadek: φ' = 0 (część główna, L = 50) | wzrost: φ' = 0,95 |
|---|---|---|
| reszty AR + CUSUM | 98% | 66% |
| PELT (koszt AR) | 90% | 98% |
| Matrix Profile (reszty AR) | 97% | 0% (AUC 0,03) |
| kNN (5 cech) | 94% | 87% |
| kNN (3 cechy) | 84% | 8% |
| Isolation Forest (5 cech) | 90% | 59% |
| Isolation Forest (3 cechy) | 36% | 16% |
| One-Class SVM (5 cech) | 64% | 85% |
| One-Class SVM (3 cechy) | 42% | 24% |

Wzrost zależności jest dla większości metod porównywalnie trudny jak spadek; **PELT jest tu najlepszy** – osobny model
AR na odcinku dokładnie odpowiada tej anomalii. **Metody na 3 cechach prawie go nie widzą** (kNN 8%,
Isolation Forest 16%) – bez cech pomocniczych pozostaje tylko autokorelacja z opóźnieniem 1, która
przy φ = 0,7 → 0,95 zmienia się niewiele w porównaniu z jej rozrzutem w oknie 50 punktów. **Matrix Profile całkowicie zawodzi** (ranking odwrócony):
bardziej zależny fragment ma mniejsze reszty, więc jest „bliżej” innych fragmentów – to przeciwieństwo tego, czego
szuka Matrix Profile.

**(c) One-Class SVM – szerokość jądra gamma** (5 cech, uczony na tym samym szeregu; poziomy 2–3; odsetek wykrytych,
przy B w nawiasie AUC):

| gamma | A | B | C |
|---|---|---|---|
| „scale” (domyślne) | 2% | 4% (0,91) | 1% |
| 0,01 (przyjęte w eksperymencie) | 0% | 51% (0,91) | 2% |
| 0,002 | 6% | 71% (0,90) | 6% |

AUC prawie się nie zmienia, a wykrywanie – dramatycznie. Przy wąskim jądrze wynik okien bardzo nietypowych przestaje
rosnąć i nie przekracza progu („nasycenie”). Drugą, silniejszą przyczyną słabości One-Class SVM uczonego na tym samym
szeregu są same anomalie w danych uczących: po uczeniu na szeregach bez anomalii (przy tym samym gamma = 0,01) zmiana φ
jest wykrywana w 81% przypadków (poziomy 2–3, L = 50) wobec 54%.

**(d) CUSUM mieszany** – wariant zaproponowany PO wynikach (a): karta poziomu na surowym szeregu, karty zmienności
i zależności oraz test szoku na resztach AR (poziomy 1–4, średnio):

| metoda | A | B | C |
|---|---|---|---|
| reszty AR + CUSUM | 31% | 75% | 52% |
| CUSUM z S1 (bez modelu) | **42%** | 58% | 53% |
| CUSUM mieszany | 30% | 74% | 52% |

**Hipoteza się nie potwierdziła** – wersja mieszana daje to samo co CUSUM na resztach. Przewaga CUSUM z S1 przy
szoku nie wynikała z karty poziomu, tylko z karty zmienności na różnicach surowego szeregu. Wariant nie wchodzi
do zestawu metod (zostaje w kodzie: `CUSUMHybryda`, jako udokumentowany wynik negatywny).

---

## 6. Weryfikacja hipotez z planu i porównanie z S1

**Hipotezy z planu ([plan_etap_1.md](plan_etap_1.md), rozdz. 4.4):**

| Hipoteza | Wynik |
|---|---|
| W resztach AR szok to „jeden ogromny punkt” – łatwy dla testu reszt | ◐ tylko dla silnego szoku (8σ_ε: 97%); szok 5σ_ε ledwie wystaje ponad maksimum szumu (26%) |
| Przesunięcie równowagi = seria reszt o niezerowej średniej – CUSUM | ✔ najlepsza metoda dla C (45%; Δ = 2σ_x: 93%), ale sygnał w resztach jest osłabiony do (1 − φ)·Δ |
| Zmiana φ = zmiana autokorelacji / wariancji reszt | ✔ (CUSUM 66%, Δφ = 0,7: 98%) |
| PELT z kosztem AR wskaże fragmenty o innym φ i μ | ◐ dobry dla φ (55%; wzrost φ: 98% – najlepszy), słabszy dla μ (36%); szoku prawie nie wykrywa (23%, impuls 5σ_ε: 6%) – krótki odcinek nie „opłaca się” przy karze |
| Matrix Profile na resztach: szok = duża reszta, zmiana φ = inna wariancja reszt | ◐ spadek φ ✔ (66%), szok ◐ (31%), wzrost φ ✘ (0%), przesunięcie równowagi ✘ (24%) |
| One-Class SVM – gładka granica lepsza od drzew? | ✘ w wersji uczonej na tym samym szeregu gorszy od Isolation Forest (4% / 17% / 0% vs 8% / 48% / 1%, 5 cech); ✔ uczony na szeregach bez anomalii – lepszy (41% / 60% / 38% vs 11% / 54% / 16%) |

**Porównanie z S1** (co się powtarza, co jest nowe):

| Zjawisko | S1 | S2 |
|---|---|---|
| Najlepsza metoda ogólnie | CUSUM | reszty AR + CUSUM |
| Najtrudniejsza anomalia | zależność (S1-C) | przesunięcie równowagi (S2-C) przy silnej zależności tła |
| Maskowanie w Isolation Forest | długie anomalie maskują same siebie; uczenie na szeregach bez anomalii usuwa większość efektu | + anomalie maskują się nawzajem (szok przy długich B, C); uczenie na szeregach bez anomalii pomaga częściowo |
| Matrix Profile ślepy na „gładsze / bardziej zależne” fragmenty | S1-C (AUC 0,46) | wzrost φ (AUC 0,03) |
| Okno ≈ długość anomalii | ✔ | ✔ (krótki szok → krótkie okno) |
| Dobór cech kluczowy | kNN 5 vs 3 cechy – lepsze 3 (zależność: 11% vs 42%) | kNN 5 vs 3 cechy – lepsze 5 (zmiana φ: 60% vs 48%; wzrost φ: 87% vs 8%); cechy S1 vs S2: 16% vs 80% |
| Nowe w S2 | – | One-Class SVM wymaga uczenia na danych bez anomalii; model AR pomaga dla zmian zależności, szkodzi dla zmian poziomu |

---

## 7. Zmiany wprowadzone w trakcie prac

| Co | Zmiana | Powód |
|---|---|---|
| **Zestawy cech (konsultacja 27.10)** | kNN, Isolation Forest i One-Class SVM w dwóch wariantach: 3 cechy (po jednej na rodzaj anomalii) i 5 cech (+ odch. std reszt AR, autokorelacja lag 2) zamiast dotychczasowych 4 | ustalenie z konsultacji – w każdym scenariuszu oba warianty; w S2 lepsze okazało się 5 cech |
| **Uczenie ML (konsultacja 27.10)** | Isolation Forest i One-Class SVM także uczone na oknach 20 osobnych szeregów bez anomalii | standardowy sposób użycia metod wykrywania nowości |
| **Wyniki per L (konsultacja 27.10)** | odsetek wykrytych według poziomu intensywności – osobna tabela dla każdej długości L (wcześniej tylko L = 50) | średnia albo jedna długość nie pokazują zależności wyników od L |
| Poziomy intensywności | po jednym słabszym poziomie każdej anomalii; φ' uporządkowane jako słabnięcie zależności; wzrost φ osobno | 4 poziomy jak w S1, monotoniczna intensywność |
| Matrix Profile | na resztach AR, bez normalizacji | poprawka po pilocie (plan, rozdz. 8) |
| PELT | własny koszt AR(1) z wyrazem wolnym, liczony z sum skumulowanych (wspólny silnik `pelt_ogolny` z S1) | ten sam algorytm co w S1, szybki; S1 daje identyczne wyniki po refaktoryzacji |
| Reszty AR + CUSUM | k karty poziomu = 0,25 zamiast 0,5 | przesunięcie równowagi o Δσ_x to w resztach tylko ≈ 0,42·Δ·σ_ε |
| One-Class SVM | gamma = 0,01 zamiast domyślnego „scale” | wybrane po teście na 30 szeregach, który pokazał nasycenie wyniku przy domyślnej wartości; uzasadnienie niezależne od wyników (skala odchyleń cech) i pełne porównanie w analizie (c) |
| Silnik eksperymentu | uniwersalny `src/ewaluacja/eksperyment.py` | ten sam kod dla S2 i S3; S1 zostawiony bez zmian (wyniki odtwarzalne bit w bit) |
| Wariant CUSUM mieszany | dodany jako analiza (d), nie wszedł do zestawu | hipoteza z analizy (a) się nie potwierdziła |
| **Skala szumu (2026-10-27)** | σ̂_e i σ̂ reszt = pierwiastek z mediany wariancji okien po 50 punktów zamiast 1,4826·MAD | długie anomalie zawyżają MAD (opis: [wyniki_s1.md](wyniki_s1.md), rozdz. 7) |
| **PELT (2026-10-27)** | odcinek porównywany z modelem AR(1) **tła** (najdłuższe odcinki, ≥ 50% szeregu), wynik tylko dla odcinków istotnie innych od tła (z > 3), kara ustalana na 10% fałszywych alarmów (siatka co 1) | model z całego szeregu jest zaburzony przez anomalie, a statystyka rośnie jak √n – długie odcinki normy dostawały wysoki wynik |

**Skutki przeliczenia po przeglądzie (2026-10-27)** – metody na cechach okien (wtedy w wersji 4-cechowej) dały identyczne wyniki:

| metoda | wykryte A / B / C przed → po | F1 przed → po | fałszywe alarmy (walidacja) przed → po |
|---|---|---|---|
| reszty AR + CUSUM | 31 / 68 / 45% → 31 / 66 / 45% | 0,51 → 0,52 | 11,5% → 10,5% |
| PELT (koszt AR) | 28 / 55 / 39% → 23 / 55 / 36% | 0,27 → 0,41 | 11,0% → 7,0% |
| Matrix Profile (reszty AR) | 30 / 64 / 21% → 31 / 66 / 24% | 0,35 → 0,36 | 11,5% → 12,0% |

Wnioski jakościowe bez zmian. PELT wykrywa nieco mniej (głównie szok: 28% → 23%), ale dużo dokładniej (precyzja
0,56 → 0,83) – część wcześniejszych „wykryć” brała się prawdopodobnie z wysokiego wyniku długich odcinków normy
(sprawdzenie na 30–40 szeregach spoza eksperymentu: ok. 6,5% punktów normy nad progiem w starej wersji, ok. 1% w nowej).

**Uzupełnienie po konsultacji (2026-10-27).** Metody na cechach okien przeliczono w wariantach 5 i 3 cech we wszystkich
częściach eksperymentu (główna, okno, tło, analiza dodatkowa (b) i (c)), a warianty ML uczone na szeregach bez anomalii –
w części głównej; pozostałe metody bez zmian. Wersja 5-cechowa kNN daje prawie to samo co dawna 4-cechowa (A / B / C:
42% / 60% / 37% wobec 42 / 59 / 37%) – dodana autokorelacja z opóźnieniem 2 niewiele zmienia;
różnice między zestawami wynikają z odch. std reszt AR, którego nie ma w zestawie 3-cechowym.

---

## 8. Ograniczenia i wnioski na S3

**Ograniczenia:**
- Odsetek fałszywych alarmów na walidacji 6,5–13,5% (cel 10%) – metody porównywalne; obok
  podawane AUC. Sprawdzenie na 1 000 nowych szeregach (`wyniki/walidacja_progow.md`): reszty AR + CUSUM 8,6%, PELT 4%, Matrix Profile 12%, kNN 9,3% / 10,8% (5 / 3 cechy), Isolation Forest 10% / 9,4%, One-Class SVM 11,8% / 12,3%; uczone na szeregach bez anomalii: Isolation Forest 9,9% / 7,8%, One-Class SVM 12,2% / 12,2%. Próg PELT jest
  ostrzejszy niż cel (walidacja 7%, na 1 000 szeregach 4%), a czułość PELT – raczej zaniżona względem pozostałych metod. Przyczyna: kara jest
  liczbą całkowitą, a odsetek fałszywych alarmów spada z karą bardzo stromo – na 200 szeregach kalibracyjnych / 200
  walidacyjnych / 400 nowych wynosi 13 / 8,5 / 8,5% przy karze 17 i 8 / 7 / 4,8% przy karze 18 (wybranej). Kara ułamkowa
  zbliżyłaby odsetek do 10%; nie zmieniłoby to wniosków – PELT ustępuje resztom AR + CUSUM także w AUC, które od progu nie
  zależy (szok / zmiana φ / przesunięcie równowagi: 0,56 / 0,75 / 0,66 wobec 0,61 /
  0,79 / 0,82).
- Niepewność: komórka tabeli (100 anomalii) ma 95% przedział ufności do ±10 punktów procentowych, średnia dla rodzaju
  (1 200 anomalii) – do ±3 punktów.
- Szok ma etykietę 10 punktów, a widoczny w resztach jest tylko pierwszy – AUC szoku dla metod resztowych jest przez to
  zaniżone (miarą właściwą dla szoku jest odsetek wykrytych).
- Wszystkie anomalie w szeregu mają ten sam poziom intensywności; Isolation Forest i One-Class SVM uczone na tym samym
  szeregu są wrażliwe na pozostałe anomalie w szeregu (rozdz. 4). Wariant uczony na szeregach bez anomalii zakłada,
  że takie dane są dostępne.

**Rekomendacje na S3 (szum + sinus):**
1. **Najpierw usunąć znaną strukturę szeregu (tu: cykl), potem szukać anomalii w resztach** – ale sprawdzić, które
   anomalie model „zjada” (jak przesunięcie równowagi w S2). W S3 wszystkie anomalie są zaburzeniami samego cyklu,
   więc reszty po dopasowaniu cyklu powinny je zachować.
2. **Metody ML uczyć także na szeregach bez anomalii** – w S2 to warunek, żeby One-Class SVM w ogóle działał;
   skalę jądra dobierać do skali odchyleń cech (w S3 – planowany autoenkoder).
3. **Oba zestawy cech** – w S2 cechy pomocnicze pomagają (zmiana dynamiki), w S1 szkodziły.
4. **Matrix Profile** – uwaga na anomalie, które czynią fragment „gładszym / mniejszym” (np. zanik amplitudy w S3-B):
   dla nich discord może zawodzić.
5. Zachować te same zasady oceny (wspólny próg, walidacja, AUC + odsetek wykrytych), żeby wyniki S1–S3 dało się
   zestawić w kroku 9.

---

## 9. Jak odtworzyć wyniki

Z katalogu `Projekt-Inz`:

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # testy S1–S3: generatory, ocena, metody
python -m src.ewaluacja.eksperyment_s2 --szybko # szybkie sprawdzenie całego łańcucha (ok. 4 min)
python -m src.ewaluacja.eksperyment_s2          # pełny eksperyment (ok. 35 min na 2 rdzeniach) → wyniki/s2/*.csv
python -m src.ewaluacja.wykresy_s2              # wykresy → wyniki/wykresy/s2_*.png, tabele → wyniki/s2/podsumowanie.md
python -m src.ewaluacja.walidacja_progow s2     # progi na 1 000 nowych szeregach → wyniki/walidacja_progow.md
```

| Plik | Zawartość |
|---|---|
| `src/symulacje/generator.py` (`ar1_zmienne`), `anomalie.py` (`szereg_s2`) | szereg AR(1) z parametrami zmiennymi w czasie, anomalie S2-A/B/C |
| `src/metody/baza.py` (`dopasuj_ar1`, `cechy_okien_s2`) | model AR(1) i cechy okien S2 |
| `src/metody/statystyczne.py` (`ResztyARCUSUM`, `PELTAR`, `CUSUMHybryda`) | metody statystyczne S2 |
| `src/metody/fragmenty.py`, `ml.py` | Matrix Profile na resztach, kNN, Isolation Forest, One-Class SVM |
| `src/ewaluacja/eksperyment.py`, `eksperyment_s2.py` | silnik eksperymentu i części S2 |
| `src/ewaluacja/wykresy_s2.py` | wykresy i tabele |
| `wyniki/s2/*.csv`, `wyniki/s2/podsumowanie.md` | surowe wyniki i wszystkie tabele |
| `wyniki/wykresy/s2_*.png` | 8 wykresów |
