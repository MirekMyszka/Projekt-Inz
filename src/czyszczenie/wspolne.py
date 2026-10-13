"""Funkcje wspólne dla czyszczenia różnych zbiorów danych."""
import numpy as np
import pandas as pd


def oznacz_luki(df: pd.DataFrame, siatka: pd.DatetimeIndex, segment=None) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Porównuje dane z oczekiwaną siatką czasu i oznacza brakujące świece (bez ich uzupełniania).

    df       - dane z indeksem czasu (każdy znacznik musi należeć do siatki),
    siatka   - wszystkie oczekiwane znaczniki czasu (posortowane),
    segment  - funkcja DatetimeIndex -> etykiety segmentów (np. dzień sesji) albo None (szereg ciągły).
               Luka jest liczona tylko w obrębie segmentu (np. noc między sesjami nie jest luką).

    Dodaje kolumny:
      luka_przed     - ile oczekiwanych świec brakuje bezpośrednio przed daną świecą,
      dzien_niepelny - w danym dniu kalendarzowym (w strefie indeksu) brakuje co najmniej jednej świecy.
    Zwraca (df, brakujące znaczniki czasu).
    """
    pos = siatka.get_indexer(df.index)
    if (pos < 0).any():
        raise ValueError(f"{(pos < 0).sum()} świec poza oczekiwaną siatką, np. {df.index[pos < 0][:3].tolist()}")

    seg_siatki = np.zeros(len(siatka), dtype=int) if segment is None else segment(siatka)
    seg_df = np.zeros(len(df), dtype=int) if segment is None else segment(df.index)
    start_segmentu = pd.Series(np.arange(len(siatka))).groupby(np.asarray(seg_siatki)).transform("min").to_numpy()

    pos_s = pd.Series(pos, index=df.index)
    poprzednia = pos_s.groupby(np.asarray(seg_df)).shift()
    luka = (pos_s - poprzednia - 1).fillna(pos_s - start_segmentu[pos]).astype(int)

    brakujace = siatka.difference(df.index)
    df = df.copy()
    df["luka_przed"] = luka
    df["dzien_niepelny"] = pd.Index(df.index.date).isin(set(brakujace.date))
    return df, brakujace


def raport_luk(brakujace: pd.DatetimeIndex, maks_dni: int = 30) -> None:
    dni = pd.Series(brakujace.date).value_counts().sort_index()
    print(f"Brakujące świece: {len(brakujace)} w {len(dni)} dniach")
    if len(dni):
        print(dni.head(maks_dni).to_string())
        if len(dni) > maks_dni:
            print(f"... i {len(dni) - maks_dni} kolejnych dni (pełna lista w pliku CSV)")
