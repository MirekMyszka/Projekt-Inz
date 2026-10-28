# Wyniki – scenariusz S3: biały szum + sinus (szereg okresowy)

> Stan na **2026-10-27** (po przeliczeniu z poprawioną skalą szumu, PELT i kartą EWMA oraz po uwagach z konsultacji
> 27.10: warianty 5 i 3 cech, metody ML uczone także na szeregach bez anomalii, wyniki osobno dla każdego L – rozdz. 7). Realizacja kroku 8 z [plan_etap_1.md](plan_etap_1.md), rozdz. 5 – według tych samych zasad co S1
> i S2 ([wyniki_s1.md](wyniki_s1.md), [wyniki_s2.md](wyniki_s2.md)). Wszystkie liczby pochodzą z
> `wyniki/s3/podsumowanie.md` (tabele generowane automatycznie); polecenia do odtworzenia – rozdz. 9.
> Szczegółowy opis każdej metody: [metody.md](metody.md). Porównanie scenariuszy S1–S3 – osobny krok (krok 9 planu),
> po dopracowaniu wszystkich trzech scenariuszy.

## Spis treści

1. [Co zbadano](#1-co-zbadano)
2. [Najważniejsze wnioski](#2-najważniejsze-wnioski)
3. [Wyniki dla każdego rodzaju anomalii](#3-wyniki-dla-każdego-rodzaju-anomalii) (w tym zestaw cech i uczenie ML)
4. [Wpływ długości anomalii, długości okna i siły cyklu](#4-wpływ-długości-anomalii-długości-okna-i-siły-cyklu)
5. [Analizy dodatkowe](#5-analizy-dodatkowe)
6. [Weryfikacja hipotez z planu i porównanie z S1, S2](#6-weryfikacja-hipotez-z-planu-i-porównanie-z-s1-s2)
7. [Zmiany wprowadzone w trakcie prac](#7-zmiany-wprowadzone-w-trakcie-prac)
8. [Ograniczenia](#8-ograniczenia)
9. [Jak odtworzyć wyniki](#9-jak-odtworzyć-wyniki)

---

## 1. Co zbadano

**Szereg tła:** `x_t = A·sin(2πt/P + θ) + ε_t`, amplituda A = 3, okres P = 50, ε ~ N(0, 1), n = 3 000 (60 cykli).
Faza początkowa θ losowana dla każdego szeregu (metody nie mogą zakładać θ = 0). Szereg ma **deterministyczny cykl**:
wartość zależy od położenia w cyklu, a cykl powtarza się co 50 punktów.

**Anomalie** – w każdym szeregu wszystkie trzy, rozmieszczone jak w S1 i S2 (po jednej w każdej trzeciej części
szeregu, w losowym miejscu, w losowej kolejności). Wszystkie trzy to **zaburzenia okresowości** – zgodnie
z przykładem z konsultacji:

| | Anomalia | Jak wstawiona | Poziomy intensywności 1 / 2 / 3 / 4 |
|---|---|---|---|
| **S3-A** | przesunięcie fazy („spóźniony” cykl) | na fragmencie ψ_t → ψ_t + Δθ; po fragmencie cykl wraca na swoje miejsce | Δθ = π/8 / π/4 / π/2 / π |
| **S3-B** | zmiana amplitudy (słabnięcie / zanik cyklu) | na fragmencie amplituda k·A | k = 0,75 / 0,5 / 0,25 / 0 |
| **S3-C** | zmiana okresu (przyspieszenie cyklu) | od początku fragmentu cykl biegnie z okresem r·P (faza ciągła), po fragmencie wraca do normalnej fazy | r = 0,9 / 0,75 / 0,6 / 0,5 |

Długość anomalii **w cyklach**: L ∈ {50, 100, 200} = 1, 2 i 4 cykle (poprawka po pilocie: fragment krótszy niż
cykl nie ma sensu dla anomalii okresowości). Względem planu dodano najsłabszy poziom fazy (π/8), amplitudę
uporządkowano jako słabnięcie cyklu, a okres jako przyspieszenie – żeby intensywność rosła monotonicznie;
odwrotne kierunki (faza −Δθ, wzmocnienie ×1,5 / ×2, spowolnienie ×1,25 / ×1,5) zbadano osobno (rozdz. 5b).

Każda anomalia ma średnią i wariancję fragmentu **prawie jak w normie** (wyjątek: zmiana amplitudy zmienia wariancję
z A²/2 + 1 = 5,5 do k²·4,5 + 1) – zmienia się położenie, siła albo tempo cyklu, czyli struktura w czasie.

**Metody** – po 2 z każdej grupy, dobrane do szeregu okresowego (plan, rozdz. 4.2 i 4.5):

| Grupa | Metoda | Co „ogląda” |
|---|---|---|
| statystyczne | **STL + EWMA** | reszty rozkładu STL (szereg bez cyklu); karta EWMA energii reszt r_t² − 1 w przód i w tył (minimum obu) |
| statystyczne | **PELT (reszty STL)** | podział reszt STL na odcinki o stałej średniej i wariancji; wynik = różnica odcinka od tła |
| fragmenty szeregu | **Matrix Profile z normalizacją** | fragmenty długości jednego cyklu (okno w = P = 50), porównywany kształt |
| fragmenty szeregu | **cechy cykli + kNN** | cechy okna = 1 cyklu: 5 albo 3 (niżej) |
| uczenie maszynowe | **Isolation Forest** | te same cechy cykli |
| uczenie maszynowe | **autoenkoder** | surowe okna (1 cykl), sieć 50 → 32 → 2 → 32 → 50; wynik = błąd odtworzenia |
| *porównawczo* | *Matrix Profile bez normalizacji* | *jak wyżej, bez standaryzacji fragmentów (hipoteza z planu o amplitudzie)* |

**Warianty (ustalenia z konsultacji 27.10.2026):**
- **zestaw cech** kNN i Isolation Forest: **3 cechy** – po jednej na rodzaj anomalii: przesunięcie fazy (A), amplituda (B),
  częstotliwość (C); **5 cech** – te 3 + odch. std reszt dopasowania sinusa i korelacja okna z wzorcowym cyklem (cechy
  pomocnicze z planu). Do 27.10 używany był zestaw 4 cech (3 + odch. std reszt);
- **uczenie** Isolation Forest i autoenkodera: na oknach **tego samego szeregu** (wersja nienadzorowana; autoenkoder
  z odrzuceniem 10% najgorzej odtwarzanych okien) albo na oknach **20 osobnych szeregów bez anomalii** (wersja
  półnadzorowana – [metody.md](metody.md), rozdz. 0.8).

Szczegóły (wzory, parametry, sposób uczenia): [metody.md](metody.md), rozdz. 0.3, 0.6, 0.8, 3–6, 9, 10.

**Ocena** – identyczna jak w S1 i S2: próg każdej metody ustalony na 200 szeregach bez anomalii (fałszywy alarm
w 10% z nich), sprawdzony na 200 nowych szeregach – **rzeczywisty odsetek 6,5–16,5%** (na 1 000
dodatkowych szeregach – rozdz. 8); wykrycie = alarm w strefie anomalia ± 25 punktów; AUC jako miara niezależna od progu.

**Skala:** część główna 4 poziomy × 3 długości × 100 = **1 200 szeregów (3 600 anomalii na metodę)**; dodatkowo:
długość okna (900 szeregów), amplituda cyklu tła (300), analizy dodatkowe (a)–(c) (700).

---

## 2. Najważniejsze wnioski

1. **Przy wyraźnym cyklu (A = 3σ) anomalie okresowości są stosunkowo łatwe – różnice widać przy najsłabszych.**
   Od poziomu 3 prawie każda metoda (poza Isolation Forest uczonym na tym samym szeregu) wykrywa 87–100% anomalii.
   Średnio po wszystkich poziomach: przesunięcie fazy 26–92%, zmiana amplitudy
   25–87%, zmiana okresu 77–96% (zależnie od metody). Najtrudniejsze jest
   **słabnięcie cyklu** – zgodnie z pilotem.

2. **Najlepsza metoda: cechy cykli + kNN** – 92% / 87% / 96% wykrytych anomalii fazy /
   amplitudy / okresu na 3 cechach (91% / 86% / 96% na 5), AUC ≥ 0,98,
   najwyższe F1 (0,79 / 0,80). Cechy opisują dokładnie parametry cyklu (amplituda, faza, częstotliwość),
   a fazę okna liczymy względem czasu całego szeregu – więc przesunięty fragment odstaje na całej długości. Najlepiej
   wykrywa najsłabsze przesunięcie fazy (π/8: 67%, pozostałe metody bez wariantów uczonych ≤
   35%) i najsłabszą zmianę okresu (×0,9:
   82%). Warunek: cechy muszą być dobrze określone (wniosek 10).

3. **Zestaw cech ma w S3 mały wpływ na kNN** (3 cechy / 5 cech: 92% / 91%, 87% /
   86%, 96% / 96%) – trzy cechy cyklu już opisują anomalie, a cechy pomocnicze są z nimi
   silnie powiązane. Dla Isolation Forest 5 cech jest lepsze dla fazy (26% → 31%), gorsze dla
   amplitudy (39% → 25%). Razem z S1 (lepsze 3 cechy) i S2 (lepsze 5 cech): **zestawu cech nie da
   się ustalić raz dla wszystkich szeregów.**

4. **Uczenie na szeregach bez anomalii usuwa maskowanie Isolation Forest – największy zysk ze wszystkich scenariuszy.**
   Isolation Forest (5 cech): 31% / 25% / 78% → 76% / 82% /
   94%, F1 0,27 → 0,61; przesunięcie fazy przy L = 200: 8% →
   74%. **Autoenkoderowi to nie pomaga**: zmiana amplitudy 62% → 46% –
   osłabiony cykl leży w tej samej podprzestrzeni co normalny (wniosek 9), a sieć uczona na wielu szeregach bez anomalii
   uczy się tej podprzestrzeni jeszcze lepiej.

5. **Przesunięcie fazy jest „niewidoczne lokalnie”.** Wewnątrz przesuniętego fragmentu każde okno to zwykły cykl –
   taki sam kształt występuje w każdym innym miejscu szeregu. Metody porównujące lokalny kształt widzą tylko skoki
   na brzegach: wynik Matrix Profile wewnątrz fragmentu = 0,98 × wynik normy, na brzegach 1,4 ×; autoenkodera –
   0,98 × i 3,2 × (sprawdzenie na 15 szeregach, Δθ = π/2, L = 200). Stąd ich niskie AUC dla fazy
   (0,68–0,71). Metody odnoszące szereg do **jednego cyklu dla całego szeregu** (reszty STL, faza
   względem czasu globalnego) widzą cały fragment: STL + EWMA AUC 0,95, kNN 0,99.

6. **Kierunek zmiany amplitudy ma ogromne znaczenie dla Matrix Profile z normalizacją.** Słabnięcie cyklu wykrywa
   najlepiej z metod porównujących kształt (87%; hipoteza z planu, że normalizacja je ukryje, **nie
   potwierdziła się** – jak w pilocie), a **wzmocnienie cyklu – w 1% (AUC 0,07,
   odwrócony ranking)**. Słabszy cykl to więcej szumu w oknie, więc po standaryzacji kształt jest „poszarpany” i nie ma
   bliźniaka; silniejszy cykl to kształt **czystszy** niż wszystkie inne okna – bliżej każdego z nich. To trzeci raz ten
   sam mechanizm: Matrix Profile jest ślepy na fragmenty „gładsze” niż reszta szeregu (S1-C, wzrost φ w S2, wzmocnienie
   cyklu w S3).

7. **Zmiana okresu jest tym łatwiejsza, im dłużej trwa** – przyspieszony cykl coraz bardziej „rozjeżdża się”
   z normalnym (różnica faz 2π·(L/P)·(1/r − 1)). Najsłabsza zmiana (r = 0,9) przy jednym cyklu: STL + EWMA
   13%, kNN 47%; przy dwóch cyklach – 100% i 100%.
   Spowolnienie cyklu wykrywane tak samo dobrze jak przyspieszenie.

8. **Model struktury szeregu nie może „wchłaniać” anomalii** (analiza a). STL z domyślnym oknem sezonowości (7 cykli)
   pozwala sezonowości dopasować się do anomalii trwającej kilka cykli (STL + EWMA przy L = 200: 53% zamiast 96%; PELT –
   50–59% zamiast 92–99%). To ta sama lekcja co w S2 (model AR „zjadał” część przesunięcia poziomu): **model tła trzeba
   dobrać tak, żeby opisywał normę, a nie anomalię**.

9. **Właściwości szeregu – siła cyklu względem szumu – zmieniają kolejność metod.** Przy A = 1 (amplituda równa
   odchyleniu szumu) najlepsze są metody na resztach STL (faza 63–84%,
   okres 92–96%): model cyklu dopasowany do 60 cykli jest dokładny, nawet gdy
   w pojedynczym oknie cyklu prawie nie widać. kNN na cechach cykli spada do 0% / 0%
   (faza, 3 / 5 cech), a zanik cyklu wykrywa już tylko Matrix Profile z normalizacją (91%).
   **Autoenkoder** (uczony na tym samym szeregu) – cykle o dowolnej fazie i amplitudzie leżą w jednej dwuwymiarowej
   podprzestrzeni (kombinacje sin i cos), a sieć z wąskim gardłem uczy się właśnie jej – i odtwarza także cykl
   przesunięty czy osłabiony: zmianę okresu wykrywa (93%), fazę i amplitudę słabo (68%,
   62%; AUC 0,71–0,72).

10. **Okno co najmniej jednego cyklu.** Przy oknie pół cyklu (w = 25) kNN na cechach cykli wykrywa
    0–2% anomalii (3 cechy;
    AUC nadal wysokie): faza i częstotliwość pół cyklu są niepewne, w szeregach bez anomalii
    zdarzają się skrajne okna i próg (ustalany na maksimach) wypada bardzo wysoko. Matrix Profile i autoenkoder zyskują
    na oknie dwóch cykli (przesunięcie fazy: 98% i 100% wobec
    74% i 87%). W S1 i S2 reguła brzmiała „okno ≈ długość
    anomalii”; w szeregu okresowym dochodzi dolna granica – długość cyklu.

---

## 3. Wyniki dla każdego rodzaju anomalii

Wykresy: `wyniki/wykresy/s3_czulosc_intensywnosc.png` (wiersz = długość L), `s3_auc_intensywnosc.png`, `s3_uczenie.png`,
`s3_przyklad.png`. Wszystkie tabele (także AUC dla każdego L) – `wyniki/s3/podsumowanie.md`.

**Odsetek wykrytych anomalii – rodzaj × poziom intensywności, osobno dla każdej długości L** (A: Δθ = π/8 / π/4 / π/2 / π; B: amplituda ×0,75 / ×0,5 / ×0,25 / ×0; C: okres ×0,9 / ×0,75 / ×0,6 / ×0,5);
pogrubienie – najlepszy wynik w kolumnie:

**L = 50** (1 cykl; 100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 12% | 97% | 100% | 100% | 2% | 42% | 94% | 100% | 13% | 100% | 100% | 100% |
| PELT (reszty STL) | 19% | 100% | 100% | 100% | 3% | 57% | 100% | 100% | 18% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 2% | 46% | 100% | 100% | 24% | 96% | 100% | 100% | 3% | 100% | 100% | 100% |
| kNN cykli (5 cech) | 64% | 100% | 100% | 100% | 33% | 100% | 100% | 100% | **50%** | 100% | 100% | 100% |
| kNN cykli (3 cechy) | **69%** | 100% | 100% | 100% | **42%** | 100% | 100% | 100% | 47% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 25% | 19% | 97% | 100% | 17% | 17% | 24% | 92% | 27% | 100% | 100% | 98% |
| Isolation Forest (3 cechy) | 23% | 15% | 78% | 100% | 19% | 48% | 51% | 96% | 23% | 100% | 100% | 95% |
| autoenkoder | 7% | 65% | 100% | 100% | 4% | 58% | 97% | 100% | 12% | 100% | 100% | 100% |
| *Matrix Profile bez normalizacji* | *3%* | *64%* | *100%* | *100%* | *2%* | *73%* | *100%* | *100%* | *9%* | *100%* | *100%* | *100%* |

**L = 100** (2 cykle; 100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 31% | 100% | 100% | 100% | 6% | 75% | 100% | 100% | 100% | 100% | 100% | 100% |
| PELT (reszty STL) | 35% | 100% | 100% | 100% | 4% | 92% | 100% | 100% | 100% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 2% | 44% | 100% | 100% | 46% | 100% | 100% | 100% | 82% | 100% | 100% | 100% |
| kNN cykli (5 cech) | 85% | 100% | 100% | 100% | 56% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| kNN cykli (3 cechy) | **91%** | 100% | 100% | 100% | **64%** | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 0% | 0% | 4% | 99% | 0% | 0% | 23% | 85% | 100% | 100% | 100% | 16% |
| Isolation Forest (3 cechy) | 0% | 0% | 12% | 71% | 6% | 0% | 33% | 95% | 99% | 100% | 100% | 22% |
| autoenkoder | 5% | 66% | 100% | 100% | 7% | 65% | 95% | 99% | 99% | 100% | 100% | 100% |
| *Matrix Profile bez normalizacji* | *2%* | *52%* | *100%* | *100%* | *10%* | *78%* | *100%* | *100%* | *96%* | *100%* | *100%* | *100%* |

**L = 200** (4 cykle; 100 anomalii w komórce):

| metoda | A1 | A2 | A3 | A4 | B1 | B2 | B3 | B4 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STL + EWMA | 40% | 100% | 100% | 100% | 6% | 77% | 100% | 100% | 100% | 100% | 100% | 100% |
| PELT (reszty STL) | **51%** | 100% | 100% | 100% | 9% | 98% | 100% | 100% | 100% | 100% | 100% | 100% |
| Matrix Profile z normalizacją | 2% | 39% | 100% | 100% | **81%** | 100% | 100% | 100% | 100% | 99% | 100% | 100% |
| kNN cykli (5 cech) | 42% | 98% | 100% | 100% | 49% | 94% | 100% | 100% | 100% | 100% | 100% | 100% |
| kNN cykli (3 cechy) | 41% | 99% | 100% | 100% | 46% | 94% | 100% | 100% | 100% | 100% | 100% | 100% |
| Isolation Forest (5 cech) | 0% | 0% | 0% | 33% | 0% | 0% | 17% | 23% | 100% | 100% | 100% | 0% |
| Isolation Forest (3 cechy) | 0% | 0% | 2% | 6% | 0% | 2% | 36% | 83% | 100% | 99% | 82% | 0% |
| autoenkoder | 4% | 73% | 100% | 100% | 12% | 45% | 70% | 98% | 100% | 100% | 100% | 100% |
| *Matrix Profile bez normalizacji* | *3%* | *54%* | *100%* | *100%* | *16%* | *57%* | *99%* | *100%* | *100%* | *100%* | *100%* | *100%* |

Przy jednym cyklu (L = 50) najsłabsze anomalie są trudne dla wszystkich metod poza kNN (faza π/8: kNN 3 cechy
69%, STL + EWMA 12%; okres ×0,9: 47% / 13%). Przy czterech
cyklach (L = 200) Isolation Forest uczony na tym samym szeregu traci nawet silne anomalie (faza π: 33%,
okres ×0,5: 0%) – maskowanie.

**Średnio dla rodzaju** (wszystkie poziomy i L; odsetek wykrytych / AUC / AP):

| metoda | A – faza | B – amplituda | C – okres |
|---|---|---|---|
| STL + EWMA | 82% / 0,95 / 0,80 | 67% / 0,89 / 0,64 | 93% / 0,93 / 0,79 |
| PELT (reszty STL) | 84% / 0,86 / 0,73 | 72% / 0,79 / 0,60 | 93% / 0,85 / 0,71 |
| Matrix Profile z normalizacją | 61% / 0,68 / 0,34 | **87% / 0,96 / 0,85** | 90% / 0,88 / 0,66 |
| kNN cykli (5 cech) | 91% / 0,99 / 0,95 | 86% / 0,98 / 0,90 | **96% / 0,99 / 0,95** |
| kNN cykli (3 cechy) | **92% / 0,99 / 0,95** | **87% / 0,98 / 0,91** | **96% / 0,99 / 0,95** |
| Isolation Forest (5 cech) | 31% / 0,99 / 0,90 | 25% / 0,98 / 0,87 | 78% / 0,99 / 0,94 |
| Isolation Forest (3 cechy) | 26% / 0,99 / 0,88 | 39% / 0,98 / 0,90 | 77% / 0,99 / 0,93 |
| autoenkoder | 68% / 0,71 / 0,39 | 62% / 0,72 / 0,32 | 93% / 0,93 / 0,81 |
| *Matrix Profile bez normalizacji* | *65% / 0,69 / 0,36* | *70% / 0,83 / 0,54* | *92% / 0,88 / 0,67* |

**Zestaw cech: 5 czy 3?** (odsetek wykrytych, w nawiasie AUC; średnio po poziomach i L)

| metoda | A: 5 cech → 3 cechy | B: 5 cech → 3 cechy | C: 5 cech → 3 cechy |
|---|---|---|---|
| kNN (cechy cykli) | 91% → 92% (0,99 → 0,99) | 86% → 87% (0,98 → 0,98) | 96% → 96% (0,99 → 0,99) |
| Isolation Forest | 31% → 26% (0,99 → 0,99) | 25% → 39% (0,98 → 0,98) | 78% → 77% (0,99 → 0,99) |
| Isolation Forest – uczony bez anomalii | 76% → 74% (0,99 → 0,99) | 82% → 83% (0,98 → 0,98) | 94% → 93% (0,99 → 0,99) |

- **kNN**: zestaw ma mały wpływ (różnice ≤ 3 punkty) – cechy pomocnicze (odch. std reszt, korelacja z wzorcem) są silnie
  powiązane z fazą i amplitudą, więc nie dokładają ani informacji, ani dużo szumu.
- **Isolation Forest**: 5 cech lepsze dla fazy, gorsze dla amplitudy (uczony na tym samym szeregu); po uczeniu na
  szeregach bez anomalii różnice znikają.

**Uczenie metod ML: na tym samym szeregu czy na szeregach bez anomalii?** (odsetek wykrytych dla każdego L, średnio po
poziomach; ostatnia kolumna – średnie AUC)

| metoda | uczenie na | A L=50 | A L=100 | A L=200 | B L=50 | B L=100 | B L=200 | C L=50 | C L=100 | C L=200 | AUC (śr.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest (5 cech) | tym samym szeregu | 60% | 26% | 8% | 38% | 27% | 10% | 81% | 79% | 75% | 0,98 |
|  | szeregach bez anomalii | 77% | 76% | 74% | 78% | 81% | 86% | 82% | 100% | 100% | 0,99 |
| Isolation Forest (3 cechy) | tym samym szeregu | 54% | 21% | 2% | 54% | 34% | 30% | 80% | 80% | 70% | 0,99 |
|  | szeregach bez anomalii | 74% | 74% | 74% | 77% | 82% | 88% | 81% | 98% | 100% | 0,99 |
| autoenkoder | tym samym szeregu | 68% | 68% | 69% | 65% | 66% | 56% | 78% | 100% | 100% | 0,79 |
|  | szeregach bez anomalii | 65% | 67% | 69% | 43% | 48% | 48% | 78% | 99% | 100% | 0,77 |

- **Isolation Forest**: uczenie na szeregach bez anomalii usuwa maskowanie – odsetek wykrytych przestaje spadać z długością
  anomalii (faza, 5 cech: 60% / 26% / 8% → 77% /
  76% / 74%). AUC było prawie idealne już wcześniej (0,99) – zmienia się
  położenie progu względem wyniku anomalii.
- **Autoenkoder** uczony na szeregach bez anomalii wykrywa **gorzej** zmianę amplitudy (62% → 46%),
  fazę i okres – tak samo. Problemem autoenkodera nie jest „zanieczyszczenie” danych uczących, tylko to, że osłabiony
  cykl leży w tej samej podprzestrzeni co normalny; sieć uczona na 20 szeregach bez anomalii odtwarza go jeszcze lepiej.
  W wersji głównej dla amplitudy pomaga za to odrzucanie najgorzej odtwarzanych okien (analiza c).

**Fałszywe alarmy i dokładność lokalizacji:**

| metoda | szeregi bez anomalii z ≥ 1 fałszywym alarmem (walidacja) | fałszywe alarmy na szereg z anomaliami | precyzja | czułość punktowa | F1 |
|---|---|---|---|---|---|
| STL + EWMA | 13,5% | 0,14 | 0,99 | 0,55 | 0,65 |
| PELT (reszty STL) | 6,5% | 0,05 | 0,93 | 0,67 | 0,73 |
| Matrix Profile z normalizacją | 13,5% | 0,18 | 0,99 | 0,44 | 0,55 |
| kNN cykli (5 cech) | 15,5% | 0,03 | 1,00 | 0,73 | **0,80** |
| kNN cykli (3 cechy) | 13,0% | 0,02 | 1,00 | 0,71 | 0,79 |
| Isolation Forest (5 cech) | 13,0% | 0,00 | 1,00 | 0,18 | 0,27 |
| Isolation Forest (3 cechy) | 12,0% | 0,00 | 1,00 | 0,14 | 0,21 |
| autoenkoder | 16,5% | 0,14 | 0,97 | 0,39 | 0,50 |
| Isolation Forest (5 cech, uczony bez anomalii) | 11,0% | 0,01 | 1,00 | 0,51 | 0,61 |
| Isolation Forest (3 cechy, uczony bez anomalii) | 13,0% | 0,02 | 0,99 | 0,44 | 0,55 |
| autoenkoder (uczony bez anomalii) | 13,0% | 0,12 | 0,97 | 0,36 | 0,48 |
| *Matrix Profile bez normalizacji* | *13,5%* | *0,15* | *0,98* | *0,31* | *0,40* |

Kara PELT dobrana na szeregach bez anomalii: 13.

Obserwacje:
- **Najsłabsze poziomy rozdzielają metody.** Od poziomu 3 prawie każda metoda wykrywa 87–100% anomalii, więc różnice
  widać przy Δθ = π/8 i amplitudzie ×0,75. Tu wyraźnie wygrywa **kNN na cechach cykli** (średnio A1 67%,
  B1 51%).
- **PELT na resztach STL** wykrywa dużo (84% / 72% / 93%) przy bardzo małej liczbie
  fałszywych alarmów (0,05 na szereg) i dobrej precyzji (0,93) – F1 0,73. AUC
  (0,79–0,86) jest niższe, niż sugeruje odsetek wykrytych, bo PELT daje wynik tylko odcinkom istotnie
  innym od tła – słabe anomalie dostają wynik 0, jak norma.
- **STL + EWMA** ma drugi najlepszy ranking (AUC 0,89–0,95) i F1 0,65.

---

## 4. Wpływ długości anomalii, długości okna i siły cyklu

**Długość anomalii L** (`wyniki/wykresy/s3_czulosc_dlugosc.png`; średnio po poziomach):

| metoda | A: L = 50 / 100 / 200 | B: L = 50 / 100 / 200 | C: L = 50 / 100 / 200 |
|---|---|---|---|
| STL + EWMA | 77 / 83 / 85% | 60 / 70 / 71% | 78 / 100 / 100% |
| PELT (reszty STL) | 80 / 84 / 88% | 65 / 74 / 77% | 80 / 100 / 100% |
| Matrix Profile z normalizacją | 62 / 62 / 60% | 80 / 86 / 95% | 76 / 96 / 100% |
| kNN cykli (5 cech) | 91 / 96 / 85% | 83 / 89 / 86% | 88 / 100 / 100% |
| kNN cykli (3 cechy) | 92 / 98 / 85% | 86 / 91 / 85% | 87 / 100 / 100% |
| Isolation Forest (5 cech) | 60 / 26 / 8% | 38 / 27 / 10% | 81 / 79 / 75% |
| Isolation Forest (3 cechy) | 54 / 21 / 2% | 54 / 34 / 30% | 80 / 80 / 70% |
| autoenkoder | 68 / 68 / 69% | 65 / 66 / 56% | 78 / 100 / 100% |
| Isolation Forest (5 cech, uczony bez anomalii) | 77 / 76 / 74% | 78 / 81 / 86% | 82 / 100 / 100% |
| Isolation Forest (3 cechy, uczony bez anomalii) | 74 / 74 / 74% | 77 / 82 / 88% | 81 / 98 / 100% |
| autoenkoder (uczony bez anomalii) | 65 / 67 / 69% | 43 / 48 / 48% | 78 / 99 / 100% |
| *Matrix Profile bez normalizacji* | *67 / 64 / 64%* | *69 / 72 / 68%* | *77 / 99 / 100%* |

- **Zmiana okresu jest tym łatwiejsza, im dłużej trwa – bo narasta różnica faz.** Przyspieszony cykl „ucieka” normalnemu:
  do końca fragmentu różnica faz wynosi 2π·(L/P)·(1/r − 1), dla r = 0,9 – 0,70 rad przy L = 50, 1,40 rad przy L = 100
  i 2,79 rad przy L = 200. Najsłabsza zmiana okresu (r = 0,9) przy L = 50 jest trudna (STL + EWMA 13%,
  kNN 47%), a przy L = 100 wykrywa ją większość metod.
- **Przesunięcie fazy zależy od L słabo** dla metod statystycznych (77–88%) – różnica między przesuniętym a normalnym
  cyklem jest taka sama na całym fragmencie, dłuższy fragment daje tylko więcej punktów do uśrednienia.
- **Maskowanie przy długich anomaliach:** Isolation Forest (5 cech, uczony na tym samym szeregu) wykrywa przesunięcie
  fazy w 60% / 26% / 8% przy L = 50 / 100 / 200. Długa anomalia daje
  wiele podobnych nietypowych okien, które tworzą własną gromadę i przestają być „łatwe do odizolowania” (jak w S1 i S2);
  uczenie na szeregach bez anomalii usuwa ten efekt (77% / 76% / 74%).
  Słabiej dotyczy to też kNN (A: 98% → 85% przy L = 200): okna długiej anomalii mają
  „sąsiadów” w tej samej anomalii – w sprawdzeniu na 15 szeregach (Δθ = π/4) maksimum wyniku kNN na anomalii wynosiło
  2,25 × 99. percentyl normy przy L = 100 i tylko 1,19 × przy L = 200.

**Długość okna w** (`wyniki/wykresy/s3_okno.png`; poziomy 2–3, wszystkie L; metody statystyczne nie mają okna):

| metoda | A: w = 25 / 50 / 100 | B: w = 25 / 50 / 100 | C: w = 25 / 50 / 100 |
|---|---|---|---|
| Matrix Profile z normalizacją | 41 / 74 / 98% | 78 / 100 / 100% | 39 / 100 / 100% |
| kNN cykli (5 cech) | 2 / 100 / 100% | 1 / 99 / 99% | 47 / 100 / 100% |
| kNN cykli (3 cechy) | 1 / 99 / 100% | 0 / 99 / 99% | 2 / 100 / 100% |
| Isolation Forest (5 cech) | 11 / 19 / 31% | 6 / 12 / 1% | 37 / 100 / 100% |
| Isolation Forest (3 cechy) | 18 / 14 / 11% | 22 / 28 / 1% | 28 / 97 / 99% |
| autoenkoder | 53 / 87 / 100% | 10 / 69 / 81% | 97 / 100 / 100% |
| *Matrix Profile bez normalizacji* | *56 / 80 / 98%* | *60 / 86 / 94%* | *97 / 100 / 100%* |

- **Okno krótsze niż cykl (w = 25) psuje metody, które opisują cykl.** kNN na cechach cykli wykrywa wtedy
  0–2% anomalii (3 cechy), choć jego AUC
  pozostaje wysokie (na 5 cechach fazę i amplitudę – w 2% / 1%, zmianę okresu – w
  47%; prawdopodobnie dzięki odch. std reszt dopasowania, która rośnie przy przyspieszonym cyklu
  i nie zależy od niepewnej częstotliwości pół cyklu):
  na pół cyklu częstotliwości i fazy nie da się wiarygodnie oszacować, więc w szeregach bez anomalii zdarzają się okna
  o skrajnych wartościach cech, a próg (ustalany na maksimach) wypada bardzo wysoko. Matrix Profile z normalizacją przy
  w = 25 słabo widzi zmianę okresu (39%) – krótki fragment przyspieszonego cyklu po standaryzacji
  jest podobny do fragmentu normalnego cyklu.
- **Okno dłuższe (2 cykle) pomaga metodom porównującym kształt:** Matrix Profile i autoenkoder wykrywają przesunięcie
  fazy w 98% / 100% przy w = 100 wobec 74% /
  87% przy w = 50 – w dłuższym oknie skok fazy na brzegu anomalii jest „widoczny” dla większej
  liczby okien. Isolation Forest traci przy dłuższym oknie (amplituda, 5 cech: 12% →
  1%; prawdopodobnie maskowanie – mniej okien, każde dłuższej anomalii podobne do siebie).
- Wniosek z S1 i S2 („okno porównywalne z długością anomalii”) w S3 przybiera postać: **okno co najmniej jednego
  cyklu**; najlepszy wybór zależy od metody (kNN – 1 cykl, Matrix Profile i autoenkoder – 2 cykle).

**Amplituda cyklu tła A** – właściwości szeregu: stosunek sygnału do szumu (`wyniki/wykresy/s3_tlo.png`;
stałe anomalie poziomu 3: faza +π/2, amplituda ×0,25, okres ×0,6; L = 100; szum σ = 1):

| metoda | A – faza: A = 1 / 3 / 5 | B – amplituda: A = 1 / 3 / 5 | C – okres: A = 1 / 3 / 5 |
|---|---|---|---|
| STL + EWMA | 63 / 100 / 100% | 3 / 100 / 100% | 92 / 100 / 100% |
| PELT (reszty STL) | 84 / 100 / 100% | 1 / 100 / 100% | 96 / 100 / 100% |
| Matrix Profile z normalizacją | 25 / 100 / 100% | 91 / 100 / 100% | 53 / 100 / 100% |
| kNN cykli (5 cech) | 0 / 100 / 100% | 11 / 100 / 100% | 3 / 100 / 100% |
| kNN cykli (3 cechy) | 0 / 100 / 100% | 7 / 100 / 100% | 0 / 100 / 100% |
| Isolation Forest (5 cech) | 0 / 8 / 34% | 2 / 19 / 2% | 1 / 100 / 100% |
| Isolation Forest (3 cechy) | 0 / 10 / 20% | 6 / 35 / 16% | 0 / 100 / 100% |
| autoenkoder | 13 / 100 / 100% | 2 / 92 / 100% | 45 / 100 / 100% |
| *Matrix Profile bez normalizacji* | *9 / 100 / 100%* | *2 / 100 / 100%* | *27 / 100 / 100%* |

- **Słaby cykl (A = 1, czyli amplituda równa odchyleniu szumu) zmienia ranking metod.** Najodporniejsze są metody
  na resztach STL (faza 63–84%, okres 92–96%): model cyklu dopasowany do całego szeregu (60 cykli) jest dokładny nawet
  przy słabym cyklu, a reszty uśredniają różnicę na całym fragmencie. Metody oceniające pojedyncze okna tracą,
  bo w jednym oknie słaby cykl ledwo widać.
- **kNN na cechach cykli przy A = 1: 0% / 0% wykrytych przesunięć fazy / zmian
  okresu (3 cechy) przy AUC 1,00 / 0,97.** Ranking jest dobry, ale przy A = 1 faza
  i częstotliwość pojedynczego okna są źle określone, więc w szeregach bez anomalii zdarzają się okna o skrajnych cechach
  i próg wypada wysoko. Ten sam mechanizm co przy oknie w = 25.
- **Zanik cyklu przy A = 1 wykrywa tylko Matrix Profile z normalizacją (91%)** – fragment czystego szumu ma po
  standaryzacji zupełnie inny kształt niż okno z cyklem. Pozostałe metody: różnica energii (0,75·A)²/2 ≈ 0,28
  przy szumie 1 – prawie niewidoczna.
- Próg STL + EWMA (0,719) i kara PELT (13) są **identyczne dla każdego A** – po usunięciu cyklu reszty to ten sam szum.
  Odsetek fałszywych alarmów na walidacji: 6–17,5%.

To odpowiedź na pytanie z tematu pracy o wpływ **właściwości szeregu**: ta sama (względna) anomalia jest oczywista
przy wyraźnym cyklu i prawie niewykrywalna przy słabym, a kolejność metod zależy od stosunku sygnału do szumu.

---

## 5. Analizy dodatkowe

Wykres: `wyniki/wykresy/s3_dodatkowa.png`. Ten sam sposób ustalania progu (odsetek fałszywych alarmów na
walidacji 6–16,5%).

**(a) Czy model cyklu „wchłania” anomalię?** STL + EWMA i PELT na resztach trzech modeli cyklu (poziomy 2–3,
50 szeregów na kombinację; odsetek wykrytych, A–C łącznie):

| metoda | L = 50 (1 cykl) | L = 100 (2 cykle) | L = 200 (4 cykle) |
|---|---|---|---|
| STL + EWMA – STL, okno sezonowości 7 cykli (domyślne) | 84% | 77% | **53%** |
| STL + EWMA – STL, okno 121 cykli (wersja główna) | 89% | **96%** | **96%** |
| STL + EWMA – sinus dopasowany globalnie (MNK) | **94%** | 95% | 93% |
| PELT – STL, okno 7 cykli | **59%** | **55%** | **50%** |
| PELT – STL, okno 121 cykli (wersja główna) | 92% | **99%** | **99%** |
| PELT – sinus dopasowany globalnie | **95%** | **99%** | 98% |

- **Domyślne STL wchłania długie anomalie.** Sezonowość wygładzana tylko na 7 cyklach może się zmienić w ciągu kilku
  cykli – i dopasowuje się do anomalii. STL + EWMA wykrywa osłabienie cyklu (×0,5 i ×0,25) w 54% / 41% / 0% przy L = 50 /
  100 / 200 (wersja główna: 67% / 87% / 87%), przesunięcie fazy – 99% / 90% / 59% (główna: 99–100%). Zmianę okresu
  wykrywa zawsze (100%) – przyspieszonego cyklu sezonowość nie zdąży „dogonić”.
- **PELT z domyślnym STL traci połowę anomalii** (50–59%): osłabienia cyklu prawie nie wykrywa (0–10%), przesunięcie
  fazy – coraz gorzej z długością anomalii (66% / 56% / 49%). Na szeregach bez anomalii reszty takiego STL dawały
  więcej fałszywych odcinków – kalibracja wymusiła karę 47 zamiast 13 – a anomalia, częściowo wchłonięta przez
  sezonowość, zostawia w resztach mniej energii. (Sezonowość z krótkim oknem dopasowuje się też do szumu: na
  przykładowym szeregu odporne odch. std reszt wyniosło 0,73 zamiast ok. 1.)
- **Sinus dopasowany globalnie** (model „idealny” – znamy kształt tła) daje prawie to samo co STL z długim oknem
  (STL + EWMA 93–95% wobec 89–96%, PELT 95–99% wobec 92–99%); jedyna wyraźna różnica – osłabienie cyklu przy L = 200
  w STL + EWMA (78% vs 87%), prawdopodobnie dlatego, że dopasowanie metodą najmniejszych kwadratów nie jest odporne
  i długa anomalia zmienia dopasowany cykl całego szeregu.

**(b) Odwrotny kierunek anomalii** (L = 100, 100 szeregów na poziom). Faza −Δθ i spowolnienie cyklu wykrywane jak
w części głównej (spowolnienie ×1,25 / ×1,5: 100% wszystkie metody; faza: różnice tylko dla metod okienkowych
i Isolation Forest). Duże różnice – dla **amplitudy**:

| metoda | osłabienie ×0,5 / ×0,25 (część główna) | wzmocnienie ×1,5 / ×2 | AUC wzmocnienia |
|---|---|---|---|
| STL + EWMA | 75% / 100% | 81% / 100% | 0,93 / 0,99 |
| PELT (reszty STL) | 92% / 100% | 94% / 100% | 0,83 / 0,96 |
| Matrix Profile z normalizacją | 100% / 100% | 1% / 1% | 0,09 / 0,05 |
| kNN cykli (5 cech) | 100% / 100% | 100% / 100% | 1,00 / 1,00 |
| kNN cykli (3 cechy) | 100% / 100% | 100% / 100% | 1,00 / 1,00 |
| Isolation Forest (5 cech) | 0% / 23% | 0% / 0% | 0,99 / 1,00 |
| Isolation Forest (3 cechy) | 0% / 33% | 0% / 0% | 1,00 / 1,00 |
| autoenkoder | 65% / 95% | 91% / 100% | 0,78 / 0,88 |
| *Matrix Profile bez normalizacji* | *78% / 100%* | *92% / 100%* | *0,97 / 1,00* |

- **Matrix Profile z normalizacją nie widzi wzmocnienia cyklu** – ranking odwrócony (AUC 0,05–0,09): okno silniejszego
  cyklu po standaryzacji jest „czystszym” sinusem, bliżej wszystkich innych okien. Bez normalizacji wzmocnienie jest
  widoczne (92–100%) – tu normalizacja rzeczywiście ukrywa zmianę amplitudy (hipoteza z planu potwierdza się dla
  tego kierunku).
- Wzmocnienie ×1,5 jest wykrywane częściej niż osłabienie ×0,5 (STL + EWMA 81% vs 75%, PELT 94% vs 92%, autoenkoder
  91% vs 65%), choć teoretyczna nadwyżka energii reszt jest w obu przypadkach taka sama ((0,5·A)²/2 = 1,125) –
  przyczyny nie badano.
- Isolation Forest (uczony na tym samym szeregu) prawie nie wykrywa wzmocnienia (0% / 0%,
  5 cech), mimo AUC bliskiego 1 – maskowanie.

**(c) Autoenkoder – na czym się uczy** (L = 100, 100 szeregów na poziom; odsetek wykrytych, w nawiasie AUC):

| wariant | faza π/4 / π/2 | amplituda ×0,5 / ×0,25 | okres ×0,75 / ×0,6 |
|---|---|---|---|
| ten sam szereg, bez odrzucania okien | 70% / 100% (0,65 / 0,73) | 30% / 72% (0,72 / 0,75) | 100% / 100% |
| ten sam szereg, odrzucanie 10% okien (wersja główna) | 70% / 100% (0,65 / 0,73) | 57% / 90% (0,75 / 0,77) | 100% / 100% |
| 20 szeregów bez anomalii (półnadzorowo, osobne szeregi uczące) | 66% / 100% (0,65 / 0,73) | 17% / 61% (0,62 / 0,72) | 100% / 100% |

- Odrzucanie najgorzej odtwarzanych okien (poprawka po pilocie) pomaga dla amplitudy (30% →
  57% przy ×0,5); odsetek fałszywych alarmów na walidacji:
  16,5% wobec 13% bez odrzucania.
- **Uczenie na szeregach bez anomalii nie pomaga** (amplituda ×0,5: 17%) – choć sieć nie widzi wtedy żadnej
  anomalii. Problemem nie jest „zanieczyszczenie” danych uczących, tylko to, że osłabiony cykl leży w tej samej
  podprzestrzeni co normalny; więcej danych bez anomalii (20 szeregów zamiast jednego) – lepiej wyuczona podprzestrzeń
  i jeszcze lepsze odtwarzanie osłabionego cyklu. Ten sam wynik daje część główna (rozdz. 3). Szeregi uczące mają
  własne ziarna (od 27.10; wcześniej sieć uczyła się na 20 szeregach kalibracyjnych – wynik był podobny: ×0,5 – 16%).
- Przesunięcie fazy i zmiana okresu – prawie bez różnic między wariantami.

---

## 6. Weryfikacja hipotez z planu i porównanie z S1, S2

**Hipotezy z planu ([plan_etap_1.md](plan_etap_1.md), rozdz. 4.5 i rozdz. 8):**

| Hipoteza | Wynik |
|---|---|
| STL + EWMA: każda anomalia psuje dopasowanie cyklu, w resztach rośnie energia | ✔ faza 82%, okres 93% (AUC 0,95 / 0,93); ◐ amplituda 67% – osłabienie cyklu zostawia mało energii; tylko jeśli STL nie wchłonie anomalii (analiza a) |
| PELT na resztach STL wskaże początek i koniec fragmentu | ✔ wykrywa 84 / 72 / 93% przy bardzo małej liczbie fałszywych alarmów (0,05 na szereg), precyzja 0,93, F1 0,73 (po poprawce PELT – rozdz. 7) |
| Matrix Profile z normalizacją, okno = P: faza i okres to fragmenty bez bliźniaka | ◐ okres ✔ (90%); faza – tylko skoki na brzegach (61%, AUC 0,68) |
| Matrix Profile z normalizacją nie zobaczy zmiany amplitudy | ✘ dla osłabienia (87% – najlepiej ze wszystkich metod, jak w pilocie); ✔ dla wzmocnienia (1%, AUC 0,05–0,09) |
| Cechy cykli + kNN – cechy odpowiadające anomaliom | ✔ najlepsza metoda S3 (92% / 87% / 96% na 3 cechach, 91% / 86% / 96% na 5) |
| Isolation Forest na cechach cykli | ◐ uczony na tym samym szeregu: AUC 0,99, ale wykrywanie 31% / 25% / 78% (5 cech) – maskowanie; ✔ uczony na szeregach bez anomalii: 76% / 82% / 94% |
| Autoenkoder – „najbardziej naturalne miejsce”, bo normalny kształt jest wyraźny | ◐ okres ✔ (93%); faza i amplituda ✘ (AUC 0,71–0,72) – sieć uczy się podprzestrzeni **wszystkich** sinusów o okresie P |
| Pilot: zmiana amplitudy najtrudniejsza w S3 | ✔ najniżej wykrywana średnio (25–87% zależnie od metody) |

**Co powtarza się w S1, S2 i S3** (pełne porównanie scenariuszy – osobny krok, po dopracowaniu wszystkich trzech):

| Zjawisko | S1 | S2 | S3 |
|---|---|---|---|
| Najlepsza metoda ogólnie | CUSUM, kNN na 3 cechach | reszty AR + CUSUM | cechy cykli + kNN |
| Lepszy zestaw cech | 3 cechy | 5 cech | bez różnicy (kNN) |
| Uczenie ML na szeregach bez anomalii | usuwa większość maskowania IF | konieczne dla One-Class SVM | usuwa maskowanie IF; autoenkoderowi szkodzi |
| Najtrudniejsza anomalia | pojawienie się zależności | przesunięcie równowagi (przy silnej zależności tła) | słabnięcie cyklu |
| Matrix Profile ślepy na fragmenty „gładsze” niż reszta | S1-C (AUC 0,46) | wzrost φ (AUC 0,03) | wzmocnienie cyklu, z normalizacją (AUC 0,05–0,09) |
| Model tła może wchłonąć anomalię | – | model AR: z przesunięcia zostaje (1 − φ)·Δ | STL z krótkim oknem sezonowości: długie anomalie |
| Maskowanie w Isolation Forest (uczony na tym samym szeregu) | długie anomalie | + anomalie maskują się nawzajem | najsilniejsze (faza przy L = 200: 8%) |
| Właściwość szeregu, która najbardziej zmienia wyniki | poziom szumu | siła zależności φ | siła cyklu względem szumu A/σ |
| Długość okna | ≈ długość anomalii | krótki szok → krótkie okno | co najmniej jeden cykl |
| Wysokie AUC przy niskim wykrywaniu (próg na maksimach) | Isolation Forest | Isolation Forest, One-Class SVM | Isolation Forest; kNN przy w = 25 i A = 1 |

---

## 7. Zmiany wprowadzone w trakcie prac

| Co | Zmiana | Powód |
|---|---|---|
| **Zestawy cech (konsultacja 27.10)** | kNN i Isolation Forest w dwóch wariantach: 3 cechy (faza, amplituda, częstotliwość) i 5 cech (+ odch. std reszt, korelacja z wzorcowym cyklem) zamiast dotychczasowych 4 | ustalenie z konsultacji – w każdym scenariuszu oba warianty |
| **Uczenie ML (konsultacja 27.10)** | Isolation Forest i autoenkoder także uczone na oknach 20 osobnych szeregów bez anomalii (własne ziarna; autoenkoder bez odrzucania okien) | standardowy sposób użycia metod wykrywania nowości; wcześniej tylko autoenkoder w analizie (c), uczony na szeregach kalibracyjnych |
| **Wyniki per L (konsultacja 27.10)** | odsetek wykrytych według poziomu intensywności – osobna tabela dla każdej długości L (wcześniej tylko L = 100) | zależność wyników od L |
| Poziomy intensywności | dodany poziom fazy π/8; amplituda jako słabnięcie (×0,75 … ×0), okres jako przyspieszenie (×0,9 … ×0,5); odwrotne kierunki osobno | 4 poziomy jak w S1 i S2, monotoniczna intensywność |
| Faza początkowa θ | losowa dla każdego szeregu zamiast θ = 0 | metody nie mogą korzystać z tego, że cykl zaczyna się w zerze |
| Zmiana okresu | faza ciągła na początku fragmentu, po fragmencie powrót do normalnej fazy | bez powrotu cały dalszy szereg byłby przesunięty – to już punkt zmiany, nie anomalia zbiorowa |
| STL | okno wygładzania sezonowości 121 cykli (dłuższe niż szereg) zamiast domyślnych 7 | domyślne okno pozwala sezonowości wchłonąć anomalię trwającą kilka cykli (próbny przebieg, potem analiza (a)) |
| **Skala szumu (2026-10-27)** | reszty STL i okna autoenkodera standaryzowane medianą wariancji okien zamiast 1,4826·MAD | długie anomalie zawyżają MAD (opis: [wyniki_s1.md](wyniki_s1.md), rozdz. 7) – przy L = 200 wariancja normy po standaryzacji ≈ 0,73 |
| **PELT (2026-10-27)** | odcinek porównywany z tłem (najdłuższe odcinki, ≥ 50% szeregu), wynik tylko dla odcinków istotnie innych od tła (z > 3), kara ustalana bisekcją na 10% fałszywych alarmów (wyszło 13) | wcześniej (odniesienie = cały szereg) długie odcinki normy dostawały wysoki wynik: AUC 0,69–0,81, precyzja 0,73 |
| **Karta EWMA (2026-10-27)** | wynik = minimum kart liczonych w przód i w tył zamiast ich średniej | „ogon” karty po silnej anomalii przekraczał próg poza strefą tolerancji: po poprawce skali – 0,46 fałszywego alarmu na szereg z anomaliami, z minimum – 0,14 (AUC niższe o 0,02–0,06) |
| Cechy cykli | częstotliwość szukana w paśmie 0,25–4 cykli na P; periodogram bez okna Hanninga | bez pasma przy słabym cyklu maksimum trafiało w szum (odchylenia cechy rzędu 150 odch. std); okno Hanninga zawyżało błąd oszacowania na oknie długości 1 cyklu |
| Autoenkoder | gardło 2 neurony, aktywacja ReLU, nauka z odrzuceniem 10% najgorzej odtwarzanych okien | gardło i aktywację wybrano na 16 szeregach testowych (8 ziaren spoza eksperymentu × poziomy 2 i 3, L = 100): AUC fazy / amplitudy / okresu prawie takie same dla gardła 2, 4 i 8 (0,64–0,69 / 0,60–0,78 / 0,92–1,00), gardło 1 wyraźnie gorsze; odrzucanie okien – poprawka po pilocie |
| Matrix Profile bez normalizacji | dodany jako metoda porównawcza | sprawdzenie hipotezy z planu (normalizacja ukrywa zmianę amplitudy) |

**Skutki przeliczenia po przeglądzie (2026-10-27)** – Matrix Profile (obie wersje), kNN i Isolation Forest (wtedy w wersji 4-cechowej) dały identyczne wyniki:

| metoda | wykryte A / B / C przed → po | AUC A / B / C przed → po | F1 przed → po | fałszywe alarmy na szereg z anomaliami |
|---|---|---|---|---|
| STL + EWMA | 79 / 60 / 93% → 82 / 67 / 93% | 0,97 / 0,91 / 0,99 → 0,95 / 0,89 / 0,93 | 0,70 → 0,65 | 0,24 → 0,14 |
| PELT (reszty STL) | 81 / 69 / 94% → 84 / 72 / 93% | 0,81 / 0,69 / 0,76 → 0,86 / 0,79 / 0,85 | 0,56 → 0,73 | 0,02 → 0,05 |
| autoenkoder | 69 / 63 / 92% → 68 / 62 / 93% | 0,71 / 0,71 / 0,93 → 0,71 / 0,72 / 0,93 | 0,51 → 0,50 | 0,26 → 0,14 |

Wnioski jakościowe bez zmian; PELT zyskał najwięcej (precyzja 0,73 → 0,93), a STL + EWMA wymienił nieco AUC
na dwa razy mniej alarmów w złym miejscu.

**Uzupełnienie po konsultacji (2026-10-27).** Metody na cechach cykli przeliczono w wariantach 5 i 3 cech we wszystkich
częściach (główna, okno, tło, analiza dodatkowa (b)), warianty uczone na szeregach bez anomalii – w części głównej, a
autoenkoder uczony na szeregach bez anomalii w analizie (c) – na nowych, osobnych szeregach uczących; pozostałe metody bez
zmian. kNN na 5 cechach daje prawie to samo co dawna wersja 4-cechowa (91% / 86% / 96%
wobec 93 / 86 / 96%); Isolation Forest na 5 cechach – nieco mniej (31% / 25% / 78% wobec
39 / 36 / 85%), bo dodana korelacja z wzorcem jest silnie powiązana z fazą i amplitudą (drzewa częściej dzielą po
cechach powtarzających tę samą informację).

---

## 8. Ograniczenia

- **Fałszywe alarmy na walidacji: 6,5–16,5%** (200 szeregów; cel 10%). Rozkład maksimum wyniku
  kNN ma ciężki ogon (cechy fazy i częstotliwości pojedynczego okna bywają skrajne), więc 200 szeregów walidacyjnych
  daje tu większy rozrzut niż dla innych metod. Sprawdzenie na 1 000 nowych szeregach (`wyniki/walidacja_progow.md`):
  STL + EWMA 12,1%, PELT 7,2%, Matrix Profile 10,4%, kNN 13,1% / 11,6% (5 / 3 cechy), Isolation Forest 10,5% / 9,7%, autoenkoder 10,7%; uczone na szeregach bez anomalii: Isolation Forest 9,3% / 9,7%, autoenkoder 9,2%. Metoda o progu łagodniejszym od celu ma lekko zawyżoną czułość – dlatego obok odsetka wykrytych podawane
  jest AUC.
- Niepewność: komórka tabeli (100 anomalii) ma 95% przedział ufności do ±10 punktów procentowych, średnia dla rodzaju
  (1 200 anomalii) – do ±3 punktów.
- **Okres P jest znany** (STL, okno = P, cechy cykli). W danych rzeczywistych okres trzeba najpierw oszacować
  (np. z periodogramu całego szeregu) – błąd okresu przełoży się na wszystkie metody S3.
- **Anomalie S3 mają skoki na brzegach** (faza – na obu końcach, okres – na końcu): to część definicji anomalii
  (cykl przesunięty na fragmencie), ale metody porównujące lokalny kształt (Matrix Profile, autoenkoder) wykrywają
  przesunięcie fazy **tylko dzięki tym skokom** (rozdz. 2, wniosek 4). Anomalia z płynnym przejściem fazy byłaby dla
  nich trudniejsza.
- Jak w S1 i S2: w szeregu wszystkie trzy anomalie mają ten sam poziom intensywności; Isolation Forest uczony na tym
  samym szeregu jest wrażliwy na pozostałe anomalie w szeregu. Wariant uczony na szeregach bez anomalii zakłada, że
  takie dane są dostępne.

---

## 9. Jak odtworzyć wyniki

Z katalogu `Projekt-Inz`:

```bash
pip install -r requirements.txt
python -m pytest tests -q                       # testy S1, S2, S3
python -m src.ewaluacja.eksperyment_s3 --szybko # szybkie sprawdzenie całego łańcucha (ok. 25 min)
python -m src.ewaluacja.eksperyment_s3          # pełny eksperyment (ok. 4 h na 2 rdzeniach) → wyniki/s3/*.csv
python -m src.ewaluacja.eksperyment_s3 glowny   # albo po jednej części: glowny, okno, tlo, dodatkowa
python -m src.ewaluacja.wykresy_s3              # wykresy → wyniki/wykresy/s3_*.png, tabele → wyniki/s3/podsumowanie.md
python -m src.ewaluacja.walidacja_progow s3     # progi na 1 000 nowych szeregach → wyniki/walidacja_progow.md
```

| Plik | Zawartość |
|---|---|
| `src/symulacje/generator.py` (`sinus_zmienny`), `anomalie.py` (`szereg_s3`) | szereg szum + sinus, anomalie S3-A/B/C |
| `src/metody/baza.py` (`reszty_stl`, `reszty_harmoniczne`, `cechy_okien_s3`) | usunięcie cyklu i cechy cykli |
| `src/metody/statystyczne.py` (`STLEWMA`, `PELTSTL`, `ewma_dwustronna`) | metody statystyczne S3 |
| `src/metody/fragmenty.py`, `ml.py` (`Autoenkoder`) | Matrix Profile, kNN, Isolation Forest, autoenkoder |
| `src/ewaluacja/eksperyment_s3.py` | części eksperymentu S3 (silnik: `eksperyment.py`) |
| `src/ewaluacja/wykresy_s3.py` | wykresy i tabele |
| `tests/test_s3.py`, `tests/test_wspolne.py` | testy generatora, cech cykli, reszt cyklu i metod; skala szumu, PELT |
| `src/ewaluacja/walidacja_progow.py`, `wyniki/walidacja_progow.md` | sprawdzenie progów na 1 000 nowych szeregach |
| `wyniki/s3/*.csv`, `wyniki/s3/podsumowanie.md` | surowe wyniki i wszystkie tabele |
| `wyniki/wykresy/s3_*.png` | 8 wykresów |
