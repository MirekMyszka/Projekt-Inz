"""Podgląd danych clean: kształt, statystyki, wykres ceny zamknięcia.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.eksploracja.podglad            # wszystkie zbiory
    python -m src.eksploracja.podglad btcusdt    # jeden zbiór (spy / btcusdt)
Wykresy: wyniki/wykresy/<zbior>_5min_close.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import BTCUSDT, SPY, WYKRESY

ZBIORY = {"spy": SPY, "btcusdt": BTCUSDT}
KOLUMNY_CEN = ["o", "h", "l", "c", "v", "n", "vw"]


def podglad(nazwa: str):
    cfg = ZBIORY[nazwa]
    path = cfg["clean"]
    if not path.exists():
        print(f"[{nazwa}] Brak pliku {path} - najpierw uruchom pobieranie i czyszczenie.")
        return
    print(f"\n=== {nazwa.upper()}: {path} ===")
    df = pd.read_parquet(path)
    print("Kształt danych:", df.shape)
    print("Zakres:", df.index.min(), "-", df.index.max())
    print("Kolumny:", list(df.columns))
    print("\nPierwsze 5 wierszy:")
    print(df.head().to_string())
    print("\nPodsumowanie:")
    print(df[[k for k in KOLUMNY_CEN if k in df.columns]].describe().to_string())

    WYKRESY.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    df["c"].plot(ax=ax, title=f"{cfg['symbol']} 5Min Close — szereg czasowy", ylabel="Close", xlabel="Time")
    if nazwa == "btcusdt":
        ax.set_yscale("log")
        ax.set_ylabel("Close (USDT, skala log)")
    fig.tight_layout()
    out_path = WYKRESY / f"{nazwa}_5min_close.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWykres zapisano do: {out_path}")


if __name__ == "__main__":
    for nazwa in (sys.argv[1:] or ZBIORY):
        podglad(nazwa.lower())
