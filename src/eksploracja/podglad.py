"""Podgląd danych clean: kształt, statystyki, wykres ceny zamknięcia.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.eksploracja.podglad            # wszystkie zbiory
    python -m src.eksploracja.podglad btcusdt    # jeden zbiór (spy / btcusdt / nyc_taxi)
Wykresy: wyniki/wykresy/<zbior>_5min_close.png (SPY, BTC), wyniki/wykresy/nyc_taxi_30min.png (z oknami anomalii NAB)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import BTCUSDT, NYC_TAXI, SPY, WYKRESY

ZBIORY = {"spy": SPY, "btcusdt": BTCUSDT, "nyc_taxi": NYC_TAXI}
KOLUMNY_CEN = ["o", "h", "l", "c", "v", "n", "vw", "pasazerowie"]


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
    if nazwa == "nyc_taxi":
        wykres_nyc_taxi(df, cfg)
        return
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


def wykres_nyc_taxi(df: pd.DataFrame, cfg: dict):
    kol = cfg["kolumna"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df[kol], lw=0.6, color="#1f77b4", label="pasażerowie / 30 min")
    for i, (zd, grupa) in enumerate(df[df["anomalia_nab"]].groupby("zdarzenie", sort=False)):
        ax.axvspan(grupa.index.min(), grupa.index.max(), color="#d62728", alpha=0.18,
                   label="okno anomalii NAB" if i == 0 else None)
        ax.annotate(zd, (grupa.index.min(), ax.get_ylim()[1]), xytext=(2, -12), textcoords="offset points",
                    fontsize=8, color="#a01010", rotation=90, va="top")
    zc = df[df["zmiana_czasu"]]
    ax.scatter(zc.index, zc[kol], color="black", s=18, zorder=3, label="zmiana czasu (artefakt)")
    ax.set_title("NYC Taxi (NAB) — liczba pasażerów co 30 min, z oknami anomalii")
    ax.set_xlabel("Czas (lokalny, Nowy Jork)")
    ax.set_ylabel("Pasażerowie / 30 min")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path = WYKRESY / "nyc_taxi_30min.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWykres zapisano do: {out_path}")


if __name__ == "__main__":
    for nazwa in (sys.argv[1:] or ZBIORY):
        podglad(nazwa.lower())
