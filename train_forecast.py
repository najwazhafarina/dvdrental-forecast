"""
train_forecast.py — Membuat prophet_model.pkl dan ml_forecast_models.pkl
Jalankan sekali sebelum api.py:
    python train_forecast.py
"""

import pickle
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "dvdrental",
    "user":     "postgres",
    "password": "postgres",
}
# ─────────────────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def fetch_daily_rentals():
    """Ambil data rental harian dari database."""
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


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROPHET
# ══════════════════════════════════════════════════════════════════════════════

def train_prophet(df):
    print("\n[1/2] Training Prophet...")
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,
        n_changepoints=10,
        changepoint_range=0.8,
        seasonality_mode='additive',
    )
    model.fit(df)

    # Hitung residual std untuk confidence interval manual
    fitted    = model.predict(model.history)
    resid_std = float((df['y'].values - fitted['yhat'].values).std())

    with open("prophet_model.pkl", "wb") as f:
        pickle.dump({
            "model":      model,
            "last_date":  df['ds'].max(),
            "train_rows": len(df),
            "resid_std":  resid_std,
        }, f)

    print(f"  ✅ prophet_model.pkl tersimpan  (resid_std={resid_std:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RANDOM FOREST + DECISION TREE (dengan lag features)
# ══════════════════════════════════════════════════════════════════════════════

def make_features(df):
    """Buat fitur time series dari lag dan rolling window."""
    d = df.copy().set_index('ds').sort_index()
    d['lag_1']      = d['y'].shift(1)
    d['lag_2']      = d['y'].shift(2)
    d['lag_3']      = d['y'].shift(3)
    d['lag_7']      = d['y'].shift(7)
    d['lag_14']     = d['y'].shift(14)
    d['rolling_7']  = d['y'].shift(1).rolling(7).mean()
    d['rolling_14'] = d['y'].shift(1).rolling(14).mean()
    d['dayofweek']  = d.index.dayofweek
    d['dayofmonth'] = d.index.day
    d['weekofyear'] = d.index.isocalendar().week.astype(int)
    d['is_weekend'] = (d.index.dayofweek >= 5).astype(int)
    return d.dropna().reset_index()


def forecast_future_ml(model, scaler, feature_cols, last_known, last_date, horizon=90):
    """
    Buat forecast iteratif untuk horizon hari ke depan.
    Setiap prediksi dipakai sebagai lag untuk prediksi berikutnya.
    """
    history = list(last_known)  # rolling buffer nilai aktual/prediksi
    results = []

    for i in range(horizon):
        future_date = last_date + pd.Timedelta(days=i+1)
        h = len(history)

        row = {
            'lag_1':      history[-1]  if h >= 1  else 0,
            'lag_2':      history[-2]  if h >= 2  else 0,
            'lag_3':      history[-3]  if h >= 3  else 0,
            'lag_7':      history[-7]  if h >= 7  else 0,
            'lag_14':     history[-14] if h >= 14 else 0,
            'rolling_7':  np.mean(history[-7:])  if h >= 7  else np.mean(history),
            'rolling_14': np.mean(history[-14:]) if h >= 14 else np.mean(history),
            'dayofweek':  future_date.dayofweek,
            'dayofmonth': future_date.day,
            'weekofyear': future_date.isocalendar()[1],
            'is_weekend': int(future_date.dayofweek >= 5),
        }

        X = np.array([[row[c] for c in feature_cols]])
        X_scaled = scaler.transform(X)
        yhat = float(model.predict(X_scaled)[0])
        yhat = max(0, round(yhat, 1))

        results.append({"ds": future_date.strftime('%Y-%m-%d'), "yhat": yhat})
        history.append(yhat)

    return results


def train_ml(df):
    print("\n[2/2] Training Random Forest & Decision Tree...")

    df_feat = make_features(df)
    feature_cols = ['lag_1','lag_2','lag_3','lag_7','lag_14',
                    'dayofweek','dayofmonth','weekofyear','is_weekend',
                    'rolling_7','rolling_14']

    X = df_feat[feature_cols].values
    y = df_feat['y'].values

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Random Forest ──
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    rf_mae  = round(float(mean_absolute_error(y_test, rf_pred)), 2)
    rf_r2   = round(float(r2_score(y_test, rf_pred)), 3)
    print(f"  Random Forest → MAE={rf_mae}, R²={rf_r2}")

    # ── Decision Tree ──
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train_s, y_train)
    dt_pred = dt.predict(X_test_s)
    dt_mae  = round(float(mean_absolute_error(y_test, dt_pred)), 2)
    dt_r2   = round(float(r2_score(y_test, dt_pred)), 3)
    print(f"  Decision Tree  → MAE={dt_mae}, R²={dt_r2}")

    # ── Forecast 90 hari ke depan ──
    last_date   = df['ds'].max()
    last_known  = df.sort_values('ds')['y'].values[-14:].tolist()

    rf_forecast = forecast_future_ml(rf, scaler, feature_cols, last_known, last_date)
    dt_forecast = forecast_future_ml(dt, scaler, feature_cols, last_known, last_date)

    with open("ml_forecast_models.pkl", "wb") as f:
        pickle.dump({
            "rf": {
                "model":        rf,
                "scaler":       scaler,
                "feature_cols": feature_cols,
                "mae":          rf_mae,
                "r2":           rf_r2,
                "last_date":    last_date,
            },
            "dt": {
                "model":        dt,
                "scaler":       scaler,
                "feature_cols": feature_cols,
                "mae":          dt_mae,
                "r2":           dt_r2,
                "last_date":    last_date,
            },
            "rf_forecast":  rf_forecast,
            "dt_forecast":  dt_forecast,
            "train_rows":   len(df),
            "last_date":    last_date,
        }, f)

    print(f"  ✅ ml_forecast_models.pkl tersimpan")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  train_forecast.py — DVDRental Time Series Training")
    print("=" * 55)

    df = fetch_daily_rentals()
    train_prophet(df)
    train_ml(df)

    print("\n✅ Semua model selesai dilatih!")
    print("   → prophet_model.pkl")
    print("   → ml_forecast_models.pkl")
    print("\nSekarang jalankan: python api.py")
