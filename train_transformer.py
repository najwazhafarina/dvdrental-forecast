"""
train_transformer.py — Membuat transformer_model.pkl menggunakan NHITS
Jalankan sekali sebelum api.py:
    python train_transformer.py

Requires:
    pip install neuralforecast

NHITS adalah neural forecasting model berbasis transformer architecture
yang ringan — tidak butuh download model besar dari internet.
"""

import pickle
import pandas as pd
import psycopg2

# ─────────────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "dvdrental",
    "user":     "postgres",
    "password": "postgres",
}
# ─────────────────────────────────────────────────────────────────────────────

HORIZON    = 90
OUTPUT_PKL = "transformer_model.pkl"


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def fetch_daily_rentals():
    conn = get_conn()
    df = pd.read_sql("""
        SELECT DATE(rental_date) AS ds, COUNT(*) AS y
        FROM rental
        GROUP BY 1
        ORDER BY 1
    """, conn)
    conn.close()
    df['ds'] = pd.to_datetime(df['ds'])
    df['y']  = df['y'].astype(float)
    print(f"  Data fetched: {len(df)} hari ({df['ds'].min().date()} → {df['ds'].max().date()})")
    return df


def train_nhits(df):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    print("\n  Training NHITS model...")

    df_input = df.copy()
    df_input['unique_id'] = 'rental'
    df_input = df_input[['unique_id', 'ds', 'y']]

    # input_size harus <= panjang data - horizon
    # data = 42 hari, horizon = 90, jadi pakai input_size kecil
    input_size = max(7, len(df) - HORIZON) if len(df) > HORIZON else max(7, len(df) // 2)

    model = NeuralForecast(
        models=[NHITS(
            h=HORIZON,
            input_size=input_size,
            max_steps=100,
            start_padding_enabled=True,   # ← izinkan padding untuk data pendek
        )],
        freq='D'
    )
    model.fit(df_input)
    print("  ✅ Training selesai\n")

    forecast_df = model.predict()
    forecast_df = forecast_df.reset_index()
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    print("  Contoh hasil forecast:")
    print(forecast_df.head(5).to_string(index=False))

    records = [
        {
            "ds":   row['ds'].strftime('%Y-%m-%d'),
            "NHITS": round(float(row['NHITS']), 1),
        }
        for _, row in forecast_df.iterrows()
    ]

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump({
            "forecast":   records,
            "last_date":  df['ds'].max(),
            "train_rows": len(df),
            "model_name": "NHITS (Neural Basis Expansion)",
            "horizon":    HORIZON,
        }, f)

    print(f"\n  ✅ {OUTPUT_PKL} tersimpan")
    return forecast_df


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  train_transformer.py — NHITS Transformer Forecast")
    print("=" * 55)

    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
    except ImportError:
        print("\n❌ neuralforecast belum terinstall!")
        print("   Jalankan: pip install neuralforecast")
        exit(1)

    df = fetch_daily_rentals()
    forecast_df = train_nhits(df)

    print("\n✅ Training selesai!")
    print(f"   → {OUTPUT_PKL}")
    print(f"   Total baris forecast: {len(forecast_df)}")
    print("\nSekarang jalankan: python api.py")