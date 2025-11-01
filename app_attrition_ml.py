# app_attrition_ml.py
# ---------------------------------
# CSV or Oracle DB → Train XGBoost → Predict → Download CSV and/or Write back to DB (with buttons)
#
# Usage:
#   streamlit run app_attrition_ml.py --server.address=0.0.0.0 --server.port=8501
#
# Requirements:
#   pip install streamlit xgboost scikit-learn pandas numpy
#   # For DB mode:
#   pip install sqlalchemy cx_Oracle

from __future__ import annotations
import io
import os
import json
import uuid
from datetime import datetime, date
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Optional DB libs (only used if Database selected)
try:
    import cx_Oracle
    from sqlalchemy import create_engine
except Exception:
    cx_Oracle = None
    create_engine = None

DEFAULT_PATH = "/mnt/data/flight_risk_input_sample.csv"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

st.set_page_config(page_title="Flight Risk XGBoost", page_icon="✈️", layout="wide")
st.title("✈️ Employee Flight Risk — XGBoost (CSV / Database)")

# -----------------------------
# Helpers
# -----------------------------
def bytes_from_df_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def load_default_df() -> pd.DataFrame | None:
    if os.path.exists(DEFAULT_PATH):
        try:
            return pd.read_csv(DEFAULT_PATH)
        except Exception as e:
            st.warning(f"Found default file but failed to read: {e}")
            return None
    return None

def detect_target_candidates(df: pd.DataFrame) -> list[str]:
    candidates = ["flight_risk", "attrition", "left", "active_flag", "target", "churn", "label"]
    return [c for c in candidates if c in df.columns]

def synthesize_target_if_missing(df: pd.DataFrame, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, str]:
    tmp = df.copy()
    z = 0
    if "engagement_score" in tmp.columns:
        z += -0.03 * tmp["engagement_score"].fillna(tmp["engagement_score"].median())
    if "work_life_balance" in tmp.columns:
        z += -0.35 * pd.to_numeric(tmp["work_life_balance"], errors="coerce").fillna(3)
    if "performance_rating" in tmp.columns:
        z += -0.25 * pd.to_numeric(tmp["performance_rating"], errors="coerce").fillna(3)
    if "overtime_hours" in tmp.columns:
        z += 0.06 * pd.to_numeric(tmp["overtime_hours"], errors="coerce").fillna(0)
    p = 1 / (1 + np.exp(-pd.to_numeric(z)))
    rnd = np.random.default_rng(seed)
    df["__synthetic_target__"] = (rnd.random(len(tmp)) < p).astype(int)
    return df, "__synthetic_target__"

def get_oracle_engine(user: str, password: str, host: str, port: str, sid: str):
    if cx_Oracle is None or create_engine is None:
        raise RuntimeError("cx_Oracle / SQLAlchemy not installed. Run: pip install sqlalchemy cx_Oracle")
    dsn = cx_Oracle.makedsn(host, int(port), sid=sid)
    engine = create_engine(f"oracle+cx_oracle://{user}:{password}@{dsn}")
    return engine

# -----------------------------
# Sidebar — Data source
# -----------------------------
with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Choose source", ["CSV", "Database"], horizontal=True)

    uploaded = None
    db_params = {}
    connect_clicked = False

    if data_source == "CSV":
        st.caption("Default: /mnt/data/flight_risk_input_sample.csv")
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    else:
        st.markdown('<h2 class="section-header">🔌 Database Connection</h2>', unsafe_allow_html=True)
        with st.form("db_connection_form"):
            st.write("Enter your Oracle database connection details:")
            user = st.text_input("Username", value="csv_one_hd100")
            password = st.text_input("Password", value="csv_one_hd100", type="password")
            host = st.text_input("Host", value="192.168.4.23")
            port = st.text_input("Port", value="1521")
            sid = st.text_input("SID", value="19cdev")
            source_table = st.text_input("Source table (read)", value="HR_EMPLOYEES")
            output_table = st.text_input("Output table (write predictions to)", value="HR_EMP_FLIGHT_RISK_PRED")
            write_mode = st.selectbox("Write mode", ["replace", "append"], index=0,
                                      help="replace = drop/create table; append = insert rows")
            connect_button = st.form_submit_button("🔄 Connect & Load Data")
        db_params = dict(user=user, password=password, host=host, port=port, sid=sid,
                         source_table=source_table, output_table=output_table, write_mode=write_mode)
        connect_clicked = connect_button

# -----------------------------
# Load data
# -----------------------------
engine = None
if data_source == "CSV":
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("Using uploaded CSV.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    else:
        df = load_default_df()
        if df is None:
            st.error("No uploaded file and default file not found at /mnt/data/flight_risk_input_sample.csv.")
            st.stop()
        else:
            st.info(f"Loaded default file: {DEFAULT_PATH}")
else:
    # Database flow: only load when user clicks button
    if not connect_clicked:
        st.info("Fill DB details and click **Connect & Load Data** to load from the database.")
        st.stop()
    try:
        engine = get_oracle_engine(db_params["user"], db_params["password"], db_params["host"], db_params["port"], db_params["sid"])
        with engine.connect() as conn:
            df = pd.read_sql(f'SELECT * FROM {db_params["source_table"]}', conn)
        st.success(f"Connected and loaded data from {db_params['source_table']}.")
    except Exception as e:
        st.error(f"DB connection/read failed: {e}")
        st.stop()

# ---------------------------------
# Column selection
# ---------------------------------
st.subheader("① Preview & Select Columns")
st.dataframe(df.head(50), use_container_width=True)

all_cols = list(df.columns)
left, right = st.columns([2, 1])
with right:
    candidates = detect_target_candidates(df)
    default_target = candidates[0] if candidates else None
    target_col = st.selectbox(
        "Target column (binary)",
        options=[None] + all_cols,
        index=(all_cols.index(default_target) + 1) if default_target in all_cols else 0,
        help="Binary column: 1=risk/left, 0=not at risk/stayed"
    )
    synth_ok = st.checkbox(
        "No target? Synthesize a temporary one for demo training",
        value=(target_col is None),
        help="Check only if you don't have a real target column."
    )
with left:
    default_features = [c for c in all_cols if c != target_col]
    feature_cols = st.multiselect(
        "Feature columns",
        options=[c for c in all_cols if c != target_col],
        default=default_features,
        help="Pick columns to use as model input features."
    )

st.markdown("**Target column:** " + (str(target_col) if target_col else "(none)"))

if not feature_cols:
    st.error("Please select at least one feature column.")
    st.stop()
if target_col is None and not synth_ok:
    st.error("Select a target or check 'Synthesize a temporary target' to continue.")
    st.stop()

# Synthesize target if missing
if target_col is None:
    df, target_col = synthesize_target_if_missing(df, seed=RANDOM_STATE)

# ---------------------------------
# Train & Predict
# ---------------------------------
st.subheader("② Train model and generate predictions")
start = st.button("🚀 Start Model (Train & Predict)")

# Will store predictions in session so the DB write button can use them
if "out_df" not in st.session_state:
    st.session_state.out_df = None
if "feature_space" not in st.session_state:
    st.session_state.feature_space = None
if "engine" not in st.session_state:
    st.session_state.engine = None
if engine is not None:
    st.session_state.engine = engine
st.session_state.db_params = db_params if data_source == "Database" else {}

if start:
    X_all = df[feature_cols].copy()
    categorical_cols = [c for c in X_all.columns if X_all[c].dtype == "object"]
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found.")
        st.stop()
    y_all = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.25, stratify=y_all, random_state=RANDOM_STATE
        )
    except ValueError as e:
        st.error(f"Train/test split failed: {e}")
        st.stop()

    X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    feature_space = list(X_train_enc.columns)

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_enc, y_train)

    proba_test = model.predict_proba(X_test_enc)[:, 1]
    preds_test = (proba_test >= 0.5).astype(int)
    roc = roc_auc_score(y_test, proba_test)
    acc = accuracy_score(y_test, preds_test)
    f1 = f1_score(y_test, preds_test)
    cm = confusion_matrix(y_test, preds_test)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("ROC AUC", f"{roc:.3f}")
    with c2: st.metric("Accuracy", f"{acc:.3f}")
    with c3: st.metric("F1", f"{f1:.3f}")

    st.write("**Confusion Matrix**")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))
    st.expander("Classification report").text(classification_report(y_test, preds_test))

    # Full predictions
    full_enc = pd.get_dummies(X_all, columns=categorical_cols, drop_first=False)
    full_enc = full_enc.reindex(columns=feature_space, fill_value=0)
    full_proba = model.predict_proba(full_enc)[:, 1]
    full_pred = (full_proba >= 0.5).astype(int)

    out_df = df.copy()
    out_df["flight_risk_prob"] = full_proba
    out_df["flight_risk_pred"] = full_pred

    # ---- Risk bands summary (Low <80, Medium 80–90, High >90) ----
    out_df["flight_risk_pct"] = out_df["flight_risk_prob"] * 100
    out_df["risk_band"] = pd.cut(
        out_df["flight_risk_pct"],
        bins=[-np.inf, 80, 90, np.inf],
        labels=["Low (<80%)", "Medium (80–90%)", "High (>90%)"],
        right=True,
    )
    band_counts = out_df["risk_band"].value_counts().reindex(
        ["Low (<80%)", "Medium (80–90%)", "High (>90%)"]
    ).fillna(0).astype(int)

    st.subheader("③ Risk bands — counts")
    st.write({
        "Low (<80%)": int(band_counts.get("Low (<80%)", 0)),
        "Medium (80–90%)": int(band_counts.get("Medium (80–90%)", 0)),
        "High (>90%)": int(band_counts.get("High (>90%)", 0)),
    })
    st.bar_chart(pd.DataFrame({"count": band_counts}).rename_axis("risk_band"))

    # ---- Downloads ----
    st.subheader("④ Download output CSV")
    st.dataframe(out_df.head(50), use_container_width=True)
    st.download_button(
        "⬇️ Download predictions CSV",
        data=bytes_from_df_csv(out_df),
        file_name="flight_risk_predictions.csv",
        mime="text/csv",
    )

    # Save to session for DB write button
    st.session_state.out_df = out_df
    st.session_state.feature_space = feature_space

    # ---- Download model & features ----
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.download_button(
            "⬇️ Download feature columns (JSON)",
            data=json.dumps(feature_space, indent=2).encode("utf-8"),
            file_name="feature_columns.json",
            mime="application/json",
        )
    with col_m2:
        booster = model.get_booster()
        raw = booster.save_raw()
        if isinstance(raw, (memoryview, bytearray)):
            raw = bytes(raw)
        buf = io.BytesIO()
        buf.write(bytes(raw))
        buf.seek(0)
        st.download_button(
            "⬇️ Download XGBoost model (binary)",
            data=buf,
            file_name="flight_risk_model.xgb",
            mime="application/octet-stream",
        )

# ---------------------------------
# DB Write Button (runs only in DB mode and after predictions exist)
# ---------------------------------
if data_source == "Database":
    st.subheader("⑤ Database Actions")
    if st.session_state.out_df is None:
        st.info("Run the model first to generate predictions, then you can write them to the database.")
    else:
        write_db = st.button("💾 Write predictions to database")
        if write_db:
            if st.session_state.engine is None:
                st.error("No database connection available. Please connect & load data from the sidebar first.")
            else:
                try:
                    process_id = str(uuid.uuid4())
                    run_date = date.today()
                    run_time = datetime.now().strftime("%H:%M:%S")

                    out_db = st.session_state.out_df.copy()
                    out_db["process_id"] = process_id
                    out_db["prediction_run_date"] = run_date
                    out_db["prediction_run_time"] = run_time

                    with st.session_state.engine.begin() as conn:
                        out_db.to_sql(
                            st.session_state.db_params.get("output_table", "HR_EMP_FLIGHT_RISK_PRED"),
                            con=conn,
                            if_exists=st.session_state.db_params.get("write_mode", "replace"),
                            index=False,
                            chunksize=1000,
                            method="multi",
                        )
                    st.success(
                        f"Predictions written to {st.session_state.db_params.get('output_table')} "
                        f"(process_id={process_id}, date={run_date}, time={run_time})."
                    )
                except Exception as e:
                    st.error(f"Failed to write predictions to DB: {e}")

st.caption("Tip: Choose CSV or Database in the sidebar. In DB mode, click “Connect & Load Data”, run the model, then use “💾 Write predictions to database”.")
