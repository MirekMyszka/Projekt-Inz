from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "dane" / "surowe" / "spy_5min_sip_2016_2026_pobrano_2026-10-13.parquet"
CLEAN_PATH = BASE_DIR / "dane" / "clean" / "spy_5min_session_2016_2026.parquet"


def preview_file(path: Path):
    print(f"\n=== Podgląd pliku: {path} ===")
    df = pd.read_parquet(path)
    print("Kształt danych:", df.shape)
    print("Kolumny:", list(df.columns))
    print("\nPierwsze 5 wierszy:")
    print(df.head().to_string())
    print("\nPodsumowanie:")
    print(df.describe(include="all").to_string())

    if "c" in df.columns:
        plot_dir = BASE_DIR / "plots"
        plot_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(14, 6))
        df["c"].plot(ax=ax, title="SPY 5Min Close — szereg czasowy", ylabel="Close", xlabel="Time")
        fig.tight_layout()
        out_path = plot_dir / "spy_5min_close.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWykres zapisano do: {out_path}")
    else:
        print("Brak kolumny 'c' do wykresu ceny zamknięcia.")


if __name__ == "__main__":
    if CLEAN_PATH.exists():
        preview_file(CLEAN_PATH)
    elif RAW_PATH.exists():
        preview_file(RAW_PATH)
    else:
        print("Brak pliku Parquet. Najpierw pobierz dane albo wygeneruj dataset clean.")
