"""
api.py  —  Flask backend for home_dashboard.html (all 4 pages merged into one)
           One file, endpoints:
             GET  /api/data                →  ABIGAIL  (Rental Activity & Customer Behaviour)
             GET  /api/abel                →  ABEL     (Customer Geographic Dashboard)
             GET  /api/tita                →  TITA     (Customer Lifecycle & CLV)
             GET  /api/enja                →  ENJA     (RFM Customer Segmentation)
             POST /api/enja/customer       →  Add new customer to database
             GET  /api/query               →  Complex customer filter
             GET  /api/forecast            →  Prophet forecast (90 hari)
             GET  /api/forecast/multi      →  Perbandingan Prophet + RF + DT
             GET  /api/forecast/transformer→  TimesFM / NHITS forecast
             POST /api/upload_forecast     →  Re-train Prophet dengan CSV baru

HOW TO RUN:
  pip install flask flask-cors psycopg2-binary pandas scikit-learn joblib numpy
  pip install prophet timesfm[torch]   ← untuk time series & transformer
  python train_forecast.py             ← buat prophet_model.pkl & ml_forecast_models.pkl
  python train_transformer.py          ← buat transformer_model.pkl
  python api.py

Then open home_dashboard.html in your browser — it connects to localhost:5050 automatically.

  Health check          : http://localhost:5050/api/ping
  ABIGAIL data          : http://localhost:5050/api/data
  ABEL data             : http://localhost:5050/api/abel
  TITA data             : http://localhost:5050/api/tita
  ENJA data             : http://localhost:5050/api/enja
  Add customer          : POST http://localhost:5050/api/enja/customer
  Complex query         : http://localhost:5050/api/query
  Forecast (Prophet)    : http://localhost:5050/api/forecast
  Forecast (Multi)      : http://localhost:5050/api/forecast/multi
  Forecast (Transformer): http://localhost:5050/api/forecast/transformer
"""

import os
import json
import numpy as np
import pandas as pd
import psycopg2
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS


# ── optional ML ───────────────────────────────────────────────────────────────
try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

# ─────────────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "dvdrental",
    "user":     "postgres",
    "password": "postgres",
}
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH       = "rental_model.pkl"
TREND_MODEL_PATH = "rental_trend_model.pkl"

app = Flask(__name__)
CORS(app)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_conn():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        raise RuntimeError(
            f"Cannot connect to PostgreSQL.\n"
            f"Check DB_CONFIG at the top of api.py.\n"
            f"Details: {e}"
        )


def safe(val):
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def clean_df(df):
    return df.where(pd.notnull(df), other=None)


def _json_default(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def json_response(payload, status=200):
    return app.response_class(
        response=json.dumps(payload, default=_json_default),
        status=status,
        mimetype='application/json'
    )


# ══════════════════════════════════════════════════════════════════════════════
# PROPHET HELPERS — config terpusat agar konsisten di semua endpoint
# ══════════════════════════════════════════════════════════════════════════════

def _build_prophet():
    """
    Buat instance Prophet yang sudah dioptimasi:
      - uncertainty_samples=0  → skip MCMC, hemat ~65% waktu training
      - yearly_seasonality=False → data DVDRental <2 tahun, tidak signifikan
      - n_changepoints=10       → default 25, lebih sedikit = lebih cepat
    CI (yhat_lower / yhat_upper) dihitung manual dari residual std setelah fit.
    """
    from prophet import Prophet
    return Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,      # ← penghematan terbesar
        n_changepoints=10,
        changepoint_range=0.8,
        seasonality_mode='additive',
    )


def _fit_and_save(model, df_combined, filepath="prophet_model.pkl"):
    """Fit model, hitung resid_std, simpan ke pickle."""
    import pickle
    model.fit(df_combined)
    fitted    = model.predict(model.history)
    resid_std = float((df_combined['y'].values - fitted['yhat'].values).std())
    with open(filepath, "wb") as f:
        pickle.dump({
            "model":      model,
            "last_date":  df_combined['ds'].max(),
            "train_rows": len(df_combined),
            "resid_std":  resid_std,
        }, f)
    return resid_std


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — UPLOAD & RE-TRAIN FORECAST  POST /api/upload_forecast
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/upload_forecast", methods=["POST"])
def upload_forecast():
    import pickle

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload"}), 400

    file = request.files['file']

    try:
        df_new = pd.read_csv(file)

        if 'date' not in df_new.columns or 'rentals' not in df_new.columns:
            return jsonify({
                "error": "CSV harus punya kolom 'date' dan 'rentals'. "
                         "Contoh format: date,rentals\n2007-01-01,42"
            }), 400

        df_new = df_new.rename(columns={'date': 'ds', 'rentals': 'y'})
        df_new['ds'] = pd.to_datetime(df_new['ds'])
        df_new['y']  = pd.to_numeric(df_new['y'], errors='coerce').fillna(0)

        conn = get_conn()
        df_hist = pd.read_sql("""
            SELECT DATE(rental_date) AS ds, COUNT(*) AS y
            FROM rental GROUP BY 1 ORDER BY 1
        """, conn)
        conn.close()
        df_hist['ds'] = pd.to_datetime(df_hist['ds'])

        df_combined = (
            pd.concat([df_hist, df_new])
            .drop_duplicates('ds', keep='last')
            .sort_values('ds')
            .reset_index(drop=True)
        )

        model     = _build_prophet()
        resid_std = _fit_and_save(model, df_combined)

        return jsonify({
            "status":    "ok",
            "new_rows":  len(df_new),
            "total_rows": len(df_combined),
            "resid_std": round(resid_std, 2),
            "date_range": {
                "from": str(df_combined['ds'].min().date()),
                "to":   str(df_combined['ds'].max().date())
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — FORECAST  GET /api/forecast
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/forecast")
def get_forecast():
    """
    Mengembalikan prediksi rental 90 hari ke depan menggunakan Prophet.
    Membutuhkan prophet_model.pkl yang dibuat oleh train_forecast.py
    """
    import pickle

    MODEL_FILE = "prophet_model.pkl"

    if not os.path.exists(MODEL_FILE):
        return jsonify({
            "error": "Model belum dilatih. Jalankan: python train_forecast.py"
        }), 404

    try:
        with open(MODEL_FILE, "rb") as f:
            saved = pickle.load(f)

        model     = saved["model"]
        last_date = saved["last_date"]

        Z         = 1.96
        resid_std = float(saved.get("resid_std", 3.0))

        future   = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        hist = forecast[forecast['ds'] <= last_date].tail(60)
        pred = forecast[forecast['ds'] >  last_date]

        payload = {
            "meta": {
                "last_actual_date": str(last_date.date()),
                "forecast_from":    str((last_date + pd.Timedelta(days=1)).date()),
                "forecast_to":      str(forecast['ds'].max().date()),
                "train_rows":       saved.get("train_rows", 0),
            },
            "historical": [
                {
                    "ds":         row['ds'].strftime('%Y-%m-%d'),
                    "yhat":       round(float(row['yhat']), 1),
                    "yhat_lower": round(max(0, float(row['yhat']) - Z * resid_std), 1),
                    "yhat_upper": round(float(row['yhat']) + Z * resid_std, 1),
                }
                for _, row in hist.iterrows()
            ],
            "forecast": [
                {
                    "ds":         row['ds'].strftime('%Y-%m-%d'),
                    "yhat":       round(float(row['yhat']), 1),
                    "yhat_lower": round(max(0, float(row['yhat']) - Z * resid_std), 1),
                    "yhat_upper": round(float(row['yhat']) + Z * resid_std, 1),
                }
                for _, row in pred.iterrows()
            ],
            "monthly_summary": (
                pred.set_index('ds')
                    .resample('ME')['yhat']
                    .agg(['sum','mean','min','max'])
                    .round(1)
                    .reset_index()
                    .rename(columns={'ds':'month','sum':'total','mean':'avg','min':'min_day','max':'max_day'})
                    .assign(month=lambda x: x['month'].dt.strftime('%B %Y'))
                    .to_dict('records')
            ),
        }

        return json_response(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — MULTI-MODEL FORECAST  GET /api/forecast/multi
# Returns Prophet + Random Forest + Decision Tree comparison
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/forecast/multi")
def get_forecast_multi():
    """
    Returns comparison forecast dari 3 model:
      - Prophet (jika prophet_model.pkl ada)
      - Random Forest + Decision Tree (jika ml_forecast_models.pkl ada)
    """
    import pickle

    PROPHET_FILE = "prophet_model.pkl"
    ML_FILE      = "ml_forecast_models.pkl"

    result = {
        "models_available": [],
        "prophet": None,
        "random_forest": None,
        "decision_tree": None,
        "transformer": None,
        "meta": {},
        "comparison": [],
    }

    # ── Prophet ──────────────────────────────────────────────────────
    if os.path.exists(PROPHET_FILE):
        try:
            with open(PROPHET_FILE, "rb") as f:
                saved = pickle.load(f)
            model     = saved["model"]
            last_date = saved["last_date"]
            resid_std = float(saved.get("resid_std", 3.0))
            Z         = 1.96

            future   = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)
            pred     = forecast[forecast['ds'] > last_date]

            result["prophet"] = [
                {
                    "ds":   row['ds'].strftime('%Y-%m-%d'),
                    "yhat": round(float(row['yhat']), 1),
                    "yhat_lower": round(max(0, float(row['yhat']) - Z * resid_std), 1),
                    "yhat_upper": round(float(row['yhat']) + Z * resid_std, 1),
                }
                for _, row in pred.iterrows()
            ]
            result["meta"]["last_date"]    = str(last_date.date())
            result["meta"]["forecast_from"] = str((last_date + pd.Timedelta(days=1)).date())
            result["meta"]["forecast_to"]   = str(pred['ds'].max().date())
            result["meta"]["prophet_resid_std"] = round(resid_std, 2)
            result["models_available"].append("prophet")
        except Exception as e:
            result["prophet_error"] = str(e)

    # ── ML models (RF + DT) ──────────────────────────────────────────
    if os.path.exists(ML_FILE):
        try:
            with open(ML_FILE, "rb") as f:
                ml = pickle.load(f)

            result["random_forest"] = ml.get("rf_forecast", [])
            result["decision_tree"] = ml.get("dt_forecast", [])

            rf_meta = ml.get("rf", {})
            dt_meta = ml.get("dt", {})

            result["meta"]["rf_mae"] = rf_meta.get("mae")
            result["meta"]["rf_r2"]  = rf_meta.get("r2")
            result["meta"]["dt_mae"] = dt_meta.get("mae")
            result["meta"]["dt_r2"]  = dt_meta.get("r2")
            result["meta"]["train_rows"] = ml.get("train_rows", 0)

            if result["random_forest"]:
                result["models_available"].append("random_forest")
            if result["decision_tree"]:
                result["models_available"].append("decision_tree")
        except Exception as e:
            result["ml_error"] = str(e)

    # ── Transformer (TimesFM / NHITS) ────────────────────────────────
    TRANSFORMER_FILE = "transformer_model.pkl"
    if os.path.exists(TRANSFORMER_FILE):
        try:
            with open(TRANSFORMER_FILE, "rb") as f:
                tr = pickle.load(f)
            fc_list = tr.get("forecast", [])
            col_name = None
            if fc_list:
                sample = fc_list[0]
                for c in ["timesfm", "NHITS", "yhat"]:
                    if c in sample:
                        col_name = c
                        break
            if col_name:
                result["transformer"] = [
                    {
                        "ds": r["ds"] if isinstance(r["ds"], str)
                              else pd.Timestamp(r["ds"]).strftime('%Y-%m-%d'),
                        "yhat": round(max(0.0, float(r.get(col_name, 0))), 1),
                    }
                    for r in fc_list
                ]
                result["models_available"].append("transformer")
                result["meta"]["transformer_model"]      = tr.get("model_name", "Transformer")
                result["meta"]["transformer_train_rows"] = tr.get("train_rows", 0)
                result["meta"]["transformer_last_date"]  = str(tr.get("last_date", ""))
        except Exception as e:
            result["transformer_error"] = str(e)

    if not result["models_available"]:
        return jsonify({
            "error": "Tidak ada model yang tersedia. Jalankan: python train_forecast.py atau train_transformer.py"
        }), 404

    # ── Comparison table: avg per bulan per model ────────────────────
    try:
        from collections import defaultdict

        def monthly_avg(forecast_list):
            by_month = defaultdict(list)
            for r in forecast_list:
                month_key = r['ds'][:7]  # YYYY-MM
                by_month[month_key].append(r['yhat'])
            return {k: round(sum(v)/len(v), 1) for k, v in sorted(by_month.items())}

        comparison = {}
        if result["prophet"]:
            comparison["prophet"] = monthly_avg(result["prophet"])
        if result["random_forest"]:
            comparison["random_forest"] = monthly_avg(result["random_forest"])
        if result["decision_tree"]:
            comparison["decision_tree"] = monthly_avg(result["decision_tree"])
        if result["transformer"]:
            comparison["transformer"] = monthly_avg(result["transformer"])

        all_months = sorted(set(
            list(comparison.get("prophet", {}).keys()) +
            list(comparison.get("random_forest", {}).keys()) +
            list(comparison.get("decision_tree", {}).keys()) +
            list(comparison.get("transformer", {}).keys())
        ))
        result["comparison"] = [
            {
                "month": m,
                "prophet":       comparison.get("prophet",       {}).get(m),
                "random_forest": comparison.get("random_forest", {}).get(m),
                "decision_tree": comparison.get("decision_tree", {}).get(m),
                "transformer":   comparison.get("transformer",   {}).get(m),
            }
            for m in all_months
        ]
    except Exception as e:
        result["comparison_error"] = str(e)

    return json_response(result)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — TRANSFORMER FORECAST  GET /api/forecast/transformer
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/forecast/transformer")
def get_forecast_transformer():
    """
    Mengembalikan prediksi dari TimesFM atau NHITS (transformer-based).
    Membutuhkan transformer_model.pkl dari train_transformer.py
    """
    import pickle

    FILE = "transformer_model.pkl"
    if not os.path.exists(FILE):
        return jsonify({"error": "Model belum dilatih. Jalankan: python train_transformer.py"}), 404

    try:
        with open(FILE, "rb") as f:
            saved = pickle.load(f)

        fc_list    = saved.get("forecast", [])
        last_date  = saved.get("last_date")
        model_name = saved.get("model_name", "Transformer")

        col_name = "yhat"
        if fc_list:
            sample = fc_list[0]
            for c in ["timesfm", "NHITS", "yhat"]:
                if c in sample:
                    col_name = c
                    break

        forecast_out = [
            {
                "ds":   r["ds"] if isinstance(r["ds"], str)
                        else pd.Timestamp(r["ds"]).strftime('%Y-%m-%d'),
                "yhat": round(max(0.0, float(r.get(col_name, 0))), 1),
            }
            for r in fc_list
        ]

        return json_response({
            "model": model_name,
            "meta": {
                "last_date":  str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
                "train_rows": saved.get("train_rows", 0),
                "total_days": len(forecast_out),
                "avg_yhat":   round(sum(r["yhat"] for r in forecast_out) / len(forecast_out), 1) if forecast_out else 0,
            },
            "forecast": forecast_out,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — COMPLEX QUERY  GET /api/query
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/query")
def complex_query():
    """
    Query pelanggan dengan filter kompleks:
      ?min_spending=100     → total spending >= 100
      ?max_spending=500     → total spending <= 500
      ?city=Jakarta         → kota (case-insensitive, partial match)
      ?country=Indonesia    → negara (case-insensitive, partial match)
      ?segment=Champions    → segmen (Champions/Regular/At Risk)
      ?min_rentals=10       → total rentals >= 10
      ?sort_by=total_spent  → sort kolom
      ?sort_dir=desc        → asc / desc
      ?limit=50             → max rows (default 50, max 200)
    """
    try:
        conn = get_conn()

        df = pd.read_sql("""
            SELECT
                cu.customer_id,
                cu.first_name || ' ' || cu.last_name   AS customer_name,
                cu.email,
                ci.city,
                co.country,
                MIN(r.rental_date)                     AS acquisition_date,
                MAX(r.rental_date)                     AS last_rental_date,
                COUNT(r.rental_id)                     AS total_rentals,
                COALESCE(SUM(p.amount), 0)             AS total_spent
            FROM customer cu
            JOIN rental  r  ON cu.customer_id = r.customer_id
            LEFT JOIN payment p ON r.rental_id = p.rental_id
            JOIN address a  ON cu.address_id  = a.address_id
            JOIN city    ci ON a.city_id      = ci.city_id
            JOIN country co ON ci.country_id  = co.country_id
            GROUP BY cu.customer_id, cu.first_name, cu.last_name,
                     cu.email, ci.city, co.country
        """, conn)
        conn.close()

        df['acquisition_date'] = pd.to_datetime(df['acquisition_date'])
        df['last_rental_date'] = pd.to_datetime(df['last_rental_date'])
        ref_date = df['last_rental_date'].max()
        df['days_since_last_rental'] = ((ref_date - df['last_rental_date']).dt.days).astype(int)
        df['active_months'] = ((df['last_rental_date'] - df['acquisition_date']).dt.days / 30.0).clip(lower=1)
        df['rental_rate']   = (df['total_rentals'] / df['active_months']).round(2)

        spend_p75   = df['total_spent'].quantile(0.75)
        recency_p33 = df['days_since_last_rental'].quantile(0.33)
        recency_p75 = df['days_since_last_rental'].quantile(0.75)
        def segment(row):
            if row['total_spent'] >= spend_p75 and row['days_since_last_rental'] <= recency_p33:
                return 'Champions'
            elif row['days_since_last_rental'] >= recency_p75:
                return 'At Risk'
            return 'Regular'
        df['segment'] = df.apply(segment, axis=1)

        filters_applied = []

        min_spending = request.args.get('min_spending', type=float)
        if min_spending is not None:
            df = df[df['total_spent'] >= min_spending]
            filters_applied.append(f"spending ≥ ${min_spending:,.0f}")

        max_spending = request.args.get('max_spending', type=float)
        if max_spending is not None:
            df = df[df['total_spent'] <= max_spending]
            filters_applied.append(f"spending ≤ ${max_spending:,.0f}")

        city = request.args.get('city', '')
        if city:
            df = df[df['city'].str.lower().str.contains(city.lower(), na=False)]
            filters_applied.append(f"city contains '{city}'")

        country = request.args.get('country', '')
        if country:
            df = df[df['country'].str.lower().str.contains(country.lower(), na=False)]
            filters_applied.append(f"country contains '{country}'")

        segment_filter = request.args.get('segment', '')
        if segment_filter:
            df = df[df['segment'].str.lower() == segment_filter.lower()]
            filters_applied.append(f"segment = '{segment_filter}'")

        min_rentals = request.args.get('min_rentals', type=int)
        if min_rentals is not None:
            df = df[df['total_rentals'] >= min_rentals]
            filters_applied.append(f"rentals ≥ {min_rentals}")

        max_rentals = request.args.get('max_rentals', type=int)
        if max_rentals is not None:
            df = df[df['total_rentals'] <= max_rentals]
            filters_applied.append(f"rentals ≤ {max_rentals}")

        VALID_SORT = ['total_spent', 'total_rentals', 'days_since_last_rental', 'rental_rate']
        sort_by  = request.args.get('sort_by', 'total_spent')
        sort_dir = request.args.get('sort_dir', 'desc')
        if sort_by not in VALID_SORT:
            sort_by = 'total_spent'
        ascending = (sort_dir.lower() == 'asc')
        df = df.sort_values(sort_by, ascending=ascending)

        limit = min(int(request.args.get('limit', 50)), 200)
        df = df.head(limit)

        df['acquisition_date'] = df['acquisition_date'].dt.strftime('%Y-%m-%d')
        df['last_rental_date'] = df['last_rental_date'].dt.strftime('%Y-%m-%d')
        df = df.drop(columns=['active_months'], errors='ignore')

        return json_response({
            "count":           len(df),
            "filters_applied": filters_applied,
            "sort_by":         sort_by,
            "sort_dir":        sort_dir,
            "customers":       json.loads(df.to_json(orient='records')),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — ABIGAIL  GET /api/data
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/data")
def get_data():
    try:
        conn = get_conn()

        df_top = pd.read_sql("""
            SELECT c.customer_id,
                   CONCAT(c.first_name, ' ', c.last_name) AS full_name,
                   COUNT(r.rental_id)        AS total_rentals,
                   COALESCE(SUM(p.amount),0) AS total_spent
            FROM customer c
            JOIN rental r  ON c.customer_id = r.customer_id
            LEFT JOIN payment p ON r.rental_id = p.rental_id
            GROUP BY c.customer_id, c.first_name, c.last_name
            ORDER BY total_rentals DESC
        """, conn)

        df_wkd = pd.read_sql("""
            SELECT
                CASE WHEN EXTRACT(DOW FROM rental_date) IN (0,6)
                     THEN 'Weekend' ELSE 'Weekday' END AS day_type,
                COUNT(*) AS total
            FROM rental GROUP BY 1
        """, conn)

        df_trend = pd.read_sql("""
            SELECT DATE(rental_date) AS date, COUNT(*) AS rentals
            FROM rental GROUP BY 1 ORDER BY 1
        """, conn)

        df_raw = pd.read_sql("""
            SELECT r.customer_id, r.rental_date,
                   CONCAT(c.first_name,' ',c.last_name) AS full_name
            FROM rental r JOIN customer c ON r.customer_id = c.customer_id
        """, conn)

        df_gaps_raw = pd.read_sql("""
            SELECT customer_id, rental_date::date AS rental_date
            FROM rental
            ORDER BY customer_id, rental_date
        """, conn)

        conn.close()

        df_raw['rental_date'] = pd.to_datetime(df_raw['rental_date'])
        df_bhv = df_raw.groupby(['customer_id','full_name']).agg(
            first_rental  = ('rental_date','min'),
            last_rental   = ('rental_date','max'),
            total_rentals = ('rental_date','count')
        ).reset_index()
        df_bhv['duration_days'] = (df_bhv['last_rental'] - df_bhv['first_rental']).dt.days
        df_bhv['frequency']     = df_bhv['total_rentals'] / (df_bhv['duration_days'] + 1)

        model        = None
        model_active = False
        if JOBLIB_OK and os.path.exists(MODEL_PATH):
            model        = joblib.load(MODEL_PATH)
            model_active = True

        X = df_bhv[['frequency','duration_days']]

        if model_active:
            label_map = {'Loyal':'High Engagement','At Risk':'Low Engagement'}
            raw_preds  = model.predict(X)
            df_bhv['predicted_segment'] = [label_map.get(p, p) for p in raw_preds]
            classes    = list(model.classes_)
            loyal_idx  = classes.index('Loyal') if 'Loyal' in classes else 0
            df_bhv['loyal_prob'] = model.predict_proba(X)[:, loyal_idx]
        else:
            threshold = df_bhv['frequency'].median()
            df_bhv['predicted_segment'] = df_bhv['frequency'].apply(
                lambda x: 'High Engagement' if x > threshold else 'Low Engagement'
            )
            df_bhv['loyal_prob'] = df_bhv['frequency'] / df_bhv['frequency'].max()

        total_rentals = int(df_top['total_rentals'].sum())

        # Count ALL customers including newly added with 0 rentals
        _conn_c = get_conn(); _cur_c = _conn_c.cursor()
        _cur_c.execute("SELECT COUNT(*) FROM customer")
        total_customers = int(_cur_c.fetchone()[0])
        _cur_c.close(); _conn_c.close()

        top10_contrib   = round(df_top.head(10)['total_rentals'].sum() / total_rentals * 100, 1)
        weekend_row     = df_wkd[df_wkd['day_type'] == 'Weekend']['total'].values
        weekend_total   = int(weekend_row[0]) if len(weekend_row) else 0
        total_wkd       = int(df_wkd['total'].sum())
        weekend_ratio   = round(weekend_total / total_wkd * 100, 1) if total_wkd else 0
        weekday_ratio   = round(100 - weekend_ratio, 1)
        loyal_pct       = round((df_bhv['predicted_segment'] == 'High Engagement').mean() * 100, 1)
        at_risk_n       = int((df_bhv['predicted_segment'] == 'Low Engagement').sum())

        df_trend['date']      = df_trend['date'].astype(str)
        df_trend['rolling_7'] = df_trend['rentals'].rolling(7).mean().round(2)

        forecast_payload = None
        if JOBLIB_OK and os.path.exists(TREND_MODEL_PATH):
            try:
                td         = joblib.load(TREND_MODEL_PATH)
                lr_model   = td["model"]
                origin     = pd.Timestamp(td["origin"])
                last_date  = pd.Timestamp(td["last_date"])
                resid_std  = float(td["resid_std"])
                slope      = float(td["slope"])
                r2         = round(float(td["r2"]), 3)

                df_th = df_trend.copy()
                df_th['date'] = pd.to_datetime(df_th['date'])
                df_th = df_th[df_th['date'] >= df_th['date'].max() - pd.Timedelta(days=90)].copy()
                df_th['day_num'] = (df_th['date'] - origin).dt.days
                df_th['fitted']  = lr_model.predict(df_th[['day_num']])

                fc_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
                fc_nums  = (fc_dates - origin).days.values.reshape(-1, 1)
                fc_vals  = lr_model.predict(fc_nums)
                fc_upper = fc_vals + 1.96 * resid_std
                fc_lower = np.maximum(0, fc_vals - 1.96 * resid_std)

                forecast_payload = {
                    "hist_dates":   df_th['date'].dt.strftime('%Y-%m-%d').tolist(),
                    "hist_rentals": df_th['rentals'].tolist(),
                    "hist_fitted":  [round(float(v), 2) for v in df_th['fitted']],
                    "fc_dates":     fc_dates.strftime('%Y-%m-%d').tolist(),
                    "fc_vals":      [round(float(v), 2) for v in fc_vals],
                    "fc_upper":     [round(float(v), 2) for v in fc_upper],
                    "fc_lower":     [round(float(v), 2) for v in fc_lower],
                    "is_declining": bool(slope < 0),
                    "slope":        slope,
                    "r2":           r2,
                    "pred_avg":     round(float(np.maximum(0, fc_vals[-30:]).mean()), 1),
                    "origin":       origin.isoformat(),
                    "last_date":    last_date.isoformat(),
                }
            except Exception:
                forecast_payload = None

        avg_dur_loyal  = safe(df_bhv[df_bhv['predicted_segment'] == 'High Engagement']['duration_days'].mean())
        avg_dur_atrisk = safe(df_bhv[df_bhv['predicted_segment'] == 'Low Engagement']['duration_days'].mean())

        df_bhv_out = df_bhv[['customer_id','full_name','total_rentals',
                              'duration_days','frequency','loyal_prob','predicted_segment']].copy()
        df_bhv_out['loyal_prob'] = (df_bhv_out['loyal_prob'] * 100).round(1)
        df_bhv_out = df_bhv_out.sort_values('total_rentals', ascending=False)

        seg_counts = df_bhv['predicted_segment'].value_counts().to_dict()

        gap_payload = None
        try:
            df_gaps_raw['rental_date'] = pd.to_datetime(df_gaps_raw['rental_date'])
            df_gaps_raw = df_gaps_raw.sort_values(['customer_id','rental_date'])
            df_gaps_raw['gap_days'] = df_gaps_raw.groupby('customer_id')['rental_date'].diff().dt.days
            gaps = df_gaps_raw['gap_days'].dropna()
            bins     = [0, 7, 14, 30, 60, 90, 180, 9999]
            labels_b = ['1-7d','8-14d','15-30d','31-60d','61-90d','91-180d','181d+']
            counts   = pd.cut(gaps, bins=bins, labels=labels_b).value_counts().reindex(labels_b, fill_value=0)
            p75 = int(gaps.quantile(0.75))
            p90 = int(gaps.quantile(0.90))
            gap_payload = {
                "labels":          labels_b,
                "counts":          counts.tolist(),
                "p75_days":        p75,
                "p90_days":        p90,
                "median_gap":      int(gaps.median()),
                "mean_gap":        round(float(gaps.mean()), 1),
                "churn_threshold": p90,
            }
        except Exception as ge:
            gap_payload = {"error": str(ge)}

        payload = {
            "kpi": {
                "total_rentals":   total_rentals,
                "total_customers": total_customers,
                "top10_contrib":   top10_contrib,
                "weekend_ratio":   weekend_ratio,
                "weekday_ratio":   weekday_ratio,
                "loyal_pct":       loyal_pct,
                "at_risk_n":       at_risk_n,
            },
            "top_renters": json.loads(df_top.to_json(orient='records')),
            "weekend":     json.loads(df_wkd.to_json(orient='records')),
            "daily_trend": json.loads(df_trend.to_json(orient='records')),
            "customers":   json.loads(df_bhv_out.to_json(orient='records')),
            "seg_counts":  seg_counts,
            "forecast":    forecast_payload,
            "insight": {
                "top10_contrib":  top10_contrib,
                "weekend_ratio":  weekend_ratio,
                "at_risk_n":      at_risk_n,
                "avg_dur_loyal":  avg_dur_loyal  or 0,
                "avg_dur_atrisk": avg_dur_atrisk or 0,
            },
            "model_active": model_active,
            "rental_gaps":  gap_payload,
        }

        return json_response(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — ABEL  GET /api/abel
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/abel")
def get_abel_data():
    try:
        conn = get_conn()

        df_raw = pd.read_sql("""
            SELECT
                p.payment_date,
                co.country,
                ci.city,
                cu.customer_id,
                cu.first_name || ' ' || cu.last_name AS customer_name,
                p.amount,
                p.payment_id
            FROM customer cu
            JOIN payment  p  ON cu.customer_id  = p.customer_id
            JOIN address  a  ON cu.address_id   = a.address_id
            JOIN city     ci ON a.city_id        = ci.city_id
            JOIN country  co ON ci.country_id    = co.country_id
        """, conn)

        conn.close()

        df_raw['payment_date'] = pd.to_datetime(df_raw['payment_date'])

        total_customers    = int(df_raw['customer_id'].nunique())
        total_revenue      = float(df_raw['amount'].sum())
        total_transactions = int(df_raw['payment_id'].count())
        total_cities       = int(df_raw['city'].nunique())
        total_countries    = int(df_raw['country'].nunique())

        avg_trans_per_city       = total_transactions / total_cities      if total_cities      else 0
        avg_spending_per_city    = total_revenue      / total_cities      if total_cities      else 0
        avg_trans_per_country    = total_transactions / total_countries   if total_countries   else 0
        avg_spending_per_country = total_revenue      / total_countries   if total_countries   else 0

        df_grouped = df_raw.groupby('country').agg(
            total_customers    = ('customer_id',  'nunique'),
            total_transactions = ('payment_id',   'count'),
            total_revenue      = ('amount',        'sum')
        ).reset_index()

        df_grouped['avg_payment']        = df_grouped['total_revenue']      / df_grouped['total_transactions']
        df_grouped['avg_trans_per_cust'] = df_grouped['total_transactions'] / df_grouped['total_customers']

        total_cust_all = df_grouped['total_customers'].sum()
        df_grouped['cust_pct'] = (df_grouped['total_customers'] / total_cust_all * 100).round(2)

        avg_customer       = float(df_grouped['total_customers'].mean())
        avg_transaction    = float(df_grouped['total_transactions'].mean())
        avg_avgtransaction = float(df_grouped['avg_trans_per_cust'].mean())

        df_raw_out = df_raw[['payment_date','country','city','customer_id','customer_name','amount','payment_id']].copy()
        df_raw_out['payment_date'] = df_raw_out['payment_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        payload = {
            "kpi": {
                "total_customers":          total_customers,
                "total_revenue":            round(total_revenue, 2),
                "total_transactions":       total_transactions,
                "total_cities":             total_cities,
                "total_countries":          total_countries,
                "avg_trans_per_city":       round(avg_trans_per_city, 1),
                "avg_spending_per_city":    round(avg_spending_per_city, 2),
                "avg_trans_per_country":    round(avg_trans_per_country, 1),
                "avg_spending_per_country": round(avg_spending_per_country, 2),
                "avg_customer":             round(avg_customer, 1),
                "avg_transaction":          round(avg_transaction, 1),
                "avg_avgtransaction":       round(avg_avgtransaction, 2),
            },
            "grouped": json.loads(df_grouped.to_json(orient='records')),
            "raw":     json.loads(df_raw_out.to_json(orient='records')),
        }

        return json_response(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — TITA  GET /api/tita
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/tita")
def get_tita_data():
    try:
        conn = get_conn()

        df_cust = pd.read_sql("""
            SELECT
                cu.customer_id,
                cu.first_name || ' ' || cu.last_name   AS customer_name,
                cu.email,
                ci.city,
                co.country,
                MIN(r.rental_date)                     AS acquisition_date,
                MAX(r.rental_date)                     AS last_rental_date,
                COUNT(r.rental_id)                     AS total_rentals,
                COALESCE(SUM(p.amount), 0)             AS total_spent
            FROM customer cu
            LEFT JOIN rental  r  ON cu.customer_id = r.customer_id
            LEFT JOIN payment p  ON r.rental_id    = p.rental_id
            JOIN address a  ON cu.address_id  = a.address_id
            JOIN city    ci ON a.city_id      = ci.city_id
            JOIN country co ON ci.country_id  = co.country_id
            GROUP BY cu.customer_id, cu.first_name, cu.last_name,
                     cu.email, ci.city, co.country
        """, conn)

        df_early = pd.read_sql("""
            SELECT cu.customer_id,
                   COALESCE(SUM(p.amount), 0) AS early_spending
            FROM customer cu
            JOIN rental r   ON cu.customer_id = r.customer_id
            LEFT JOIN payment p ON r.rental_id = p.rental_id
            WHERE r.rental_date <= (
                SELECT MIN(r2.rental_date) + INTERVAL '60 days'
                FROM rental r2 WHERE r2.customer_id = cu.customer_id
            )
            GROUP BY cu.customer_id
        """, conn)

        df_hist = pd.read_sql("""
            SELECT r.customer_id, r.rental_date, cat.name AS genre
            FROM rental r
            JOIN inventory   inv ON r.inventory_id  = inv.inventory_id
            JOIN film          f ON inv.film_id      = f.film_id
            JOIN film_category fc ON f.film_id       = fc.film_id
            JOIN category    cat ON fc.category_id  = cat.category_id
            ORDER BY r.rental_date
        """, conn)

        conn.close()

        df_cust = df_cust.merge(df_early, on='customer_id', how='left')
        df_cust['early_spending']   = df_cust['early_spending'].fillna(0)
        df_cust['acquisition_date'] = pd.to_datetime(df_cust['acquisition_date'])
        df_cust['last_rental_date'] = pd.to_datetime(df_cust['last_rental_date'])

        ref_date = df_cust['last_rental_date'].max()
        # New customers with no rentals get days_since = 9999
        df_cust['days_since_last_rental'] = df_cust['last_rental_date'].apply(
            lambda x: 9999 if pd.isnull(x) else int((ref_date - x).days)
        )

        df_cust['active_months'] = df_cust.apply(
            lambda row: 1.0 if pd.isnull(row['last_rental_date']) or pd.isnull(row['acquisition_date'])
                        else max(1.0, (row['last_rental_date'] - row['acquisition_date']).days / 30.0),
            axis=1
        )
        df_cust['rental_rate'] = (df_cust['total_rentals'] / df_cust['active_months']).round(2)

        spend_p75   = df_cust['total_spent'].quantile(0.75)
        recency_p33 = df_cust['days_since_last_rental'].quantile(0.33)
        recency_p75 = df_cust['days_since_last_rental'].quantile(0.75)

        def segment(row):
            if row['total_spent'] >= spend_p75 and row['days_since_last_rental'] <= recency_p33:
                return 'Champions'
            elif row['days_since_last_rental'] >= recency_p75:
                return 'At Risk'
            return 'Regular'

        df_cust['segment'] = df_cust.apply(segment, axis=1)

        df_cust['acquisition_date'] = df_cust['acquisition_date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None
        )
        df_cust['last_rental_date'] = df_cust['last_rental_date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None
        )
        df_hist['rental_date'] = pd.to_datetime(df_hist['rental_date']).dt.strftime('%Y-%m-%d')
        df_cust = df_cust.drop(columns=['active_months'])

        countries = sorted(df_cust['country'].dropna().unique().tolist())

        payload = {
            "countries":   countries,
            "customers":   json.loads(df_cust.to_json(orient='records')),
            "rental_hist": json.loads(df_hist.to_json(orient='records')),
        }

        return json_response(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — ENJA  GET /api/enja
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/enja")
def get_enja_data():
    try:
        conn = get_conn()

        df = pd.read_sql("""
            SELECT
                c.customer_id,
                CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
                c.email,
                MAX(r.rental_date)          AS last_rental,
                COUNT(DISTINCT r.rental_id) AS frequency,
                COALESCE(SUM(p.amount), 0)  AS monetary
            FROM customer c
            LEFT JOIN rental  r ON c.customer_id = r.customer_id
            LEFT JOIN payment p ON c.customer_id = p.customer_id
            GROUP BY c.customer_id, c.first_name, c.last_name, c.email
        """, conn)
        conn.close()

        df['last_rental'] = pd.to_datetime(df['last_rental'])
        ref = df['last_rental'].max()
        # New customers with no rentals get recency = 9999 (worst possible score)
        df['recency'] = df['last_rental'].apply(
            lambda x: 9999 if pd.isnull(x) else (ref - x).days
        )

        df['r_score'] = np.where(df['recency'] == 0, 3,
                        np.where(df['recency'] <= 175, 2, 1))

        df['f_score'] = np.where(df['frequency'] >= 30, 3,
                        np.where(df['frequency'] >= 23, 2, 1))

        df['m_score'] = np.where(df['monetary'] >= 3439, 3,
                        np.where(df['monetary'] >= 2041, 2, 1))

        df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']

        df['segment'] = np.where(df['rfm_score'] >= 8, 'Champions',
                        np.where(df['rfm_score'] >= 6, 'Regular',
                        np.where(df['rfm_score'] >= 4, 'At Risk', 'Low Value')))

        total   = int(len(df))
        champ_n = int((df['segment'] == 'Champions').sum())
        reg_n   = int((df['segment'] == 'Regular').sum())
        risk_n  = int((df['segment'] == 'At Risk').sum())
        low_n   = int((df['segment'] == 'Low Value').sum())

        desc = df[['recency', 'frequency', 'monetary']].describe().round(2)
        desc.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        desc_dict = desc.to_dict()

        rev_by_seg  = df.groupby('segment')['monetary'].sum().reset_index()
        rev_by_seg.columns = ['segment', 'revenue']
        avg_rev_seg = float(rev_by_seg['revenue'].mean())
        top20_rev   = float(rev_by_seg['revenue'].quantile(0.80))

        seg_counts_dict = df['segment'].value_counts().to_dict()

        champ_avg = float(df[df['segment'] == 'Champions']['monetary'].mean()) if champ_n > 0 else 0.0

        df_out = df[['customer_id','customer_name','email','recency','frequency',
                     'monetary','r_score','f_score','m_score','rfm_score','segment']].copy()
        df_out['last_rental'] = df['last_rental'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None
        )
        df_out = df_out.sort_values('monetary', ascending=False)

        payload = {
            "kpi": {
                "total":      total,
                "champ_n":    champ_n,
                "reg_n":      reg_n,
                "risk_n":     risk_n,
                "low_n":      low_n,
                "champ_avg":  round(champ_avg, 2),
                "avg_rev_seg": round(avg_rev_seg, 2),
                "top20_rev":   round(top20_rev, 2),
            },
            "desc":       desc_dict,
            "rev_by_seg": json.loads(rev_by_seg.to_json(orient='records')),
            "customers":  json.loads(df_out.to_json(orient='records')),
        }

        return json_response(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT — ADD CUSTOMER  POST /api/enja/customer
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/enja/customer", methods=["POST"])
def add_customer():
    """
    Expected JSON body:
    {
      "first_name": "John",
      "last_name":  "Doe",
      "email":      "john.doe@example.com",
      "address":    "123 Main St",   // optional
      "city":       "Jakarta",       // optional – matched case-insensitively
      "phone":      "081234567890"   // optional
    }
    """
    try:
        body = request.get_json(force=True)
        first_name = (body.get("first_name") or "").strip()
        last_name  = (body.get("last_name")  or "").strip()
        email      = (body.get("email")      or "").strip()

        if not first_name or not last_name or not email:
            return jsonify({"error": "first_name, last_name, and email are required"}), 400

        conn = get_conn()
        cur  = conn.cursor()

        address_line = (body.get("address") or "").strip() or "N/A"
        city_name    = (body.get("city")    or "").strip()
        phone        = (body.get("phone")   or "").strip() or ""

        if city_name:
            cur.execute(
                "SELECT city_id FROM city WHERE LOWER(city) LIKE LOWER(%s) LIMIT 1",
                (f"%{city_name}%",)
            )
            row = cur.fetchone()
            city_id = row[0] if row else None
        else:
            city_id = None

        if not city_id:
            cur.execute("SELECT city_id FROM city ORDER BY city_id LIMIT 1")
            city_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO address (address, address2, district, city_id, postal_code, phone, last_update)
            VALUES (%s, '', 'N/A', %s, '', %s, NOW())
            RETURNING address_id
            """,
            (address_line, city_id, phone)
        )
        address_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO customer
              (store_id, first_name, last_name, email, address_id, activebool, create_date, last_update, active)
            VALUES (1, %s, %s, %s, %s, TRUE, CURRENT_DATE, NOW(), 1)
            RETURNING customer_id
            """,
            (first_name, last_name, email, address_id)
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        new_customer = {
            "customer_id":   new_id,
            "customer_name": f"{first_name} {last_name}",
            "email":         email,
            "recency":       9999,
            "frequency":     0,
            "monetary":      0.0,
            "r_score":       1,
            "f_score":       1,
            "m_score":       1,
            "rfm_score":     3,
            "segment":       "Low Value",
            "last_rental":   None,
        }
        return json_response({"status": "ok", "customer_id": new_id, "customer": new_customer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ROOT — serve dashboard HTML
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_file("home_dashboard.html")


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/ping")
def ping():
    try:
        conn = get_conn()
        conn.close()
        return jsonify({"status": "ok", "db": "connected"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("  DVD Rental API  —  running on http://localhost:5050")
    print("  Serving: home_dashboard.html (all 4 pages merged)")
    print()
    print("  Health check          : http://localhost:5050/api/ping")
    print("  ABIGAIL data          : http://localhost:5050/api/data")
    print("  ABEL data             : http://localhost:5050/api/abel")
    print("  TITA data             : http://localhost:5050/api/tita")
    print("  ENJA data             : http://localhost:5050/api/enja")
    print("  Add customer          : POST /api/enja/customer")
    print("  Complex query         : http://localhost:5050/api/query")
    print("  Forecast (Prophet)    : http://localhost:5050/api/forecast")
    print("  Forecast (Multi)      : http://localhost:5050/api/forecast/multi")
    print("  Forecast (Transformer): http://localhost:5050/api/forecast/transformer")
    print()
    import glob
    pkls = glob.glob("*.pkl")
    if pkls:
        print("  Model files ditemukan:", ", ".join(pkls))
    else:
        print("  ⚠  Tidak ada .pkl — jalankan train_forecast.py & train_transformer.py")
    print("=" * 62)
    app.run(host="0.0.0.0", port=5050, debug=True)