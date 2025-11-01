# app_attrition_ml.py
# ---------------------------------
# Streamlit app that:
#  1) Reads a default CSV (/mnt/data/flight_risk_input_sample.csv)
#  2) Lets you upload a new CSV
#  3) Select target and feature columns
#  4) Trains XGBoost model, shows metrics, and provides downloads
#  5) Displays Low/Medium/High risk bands bar chart
#
# Usage:
#   streamlit run app_attrition_ml.py --server.address=0.0.0.0 --server.port=8501
#
# Requirements:
#   pip install streamlit xgboost scikit-learn pandas numpy

from __future__ import annotations
import io
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

DEFAULT_PATH = "/mnt/data/flight_risk_input_sample.csv"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

st.set_page_config(page_title="Flight Risk XGBoost", page_icon="✈️", layout="wide")
st.title("✈️ Employee Flight Risk — XGBoost (CSV → CSV)")

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

# -----------------------------
# Data input
# -----------------------------
with st.sidebar:
    st.header("Data Source")
    st.caption("By default the app reads /mnt/data/flight_risk_input_sample.csv if available.")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

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
    rnd = np.random.default_rng(RANDOM_STATE)
    df["__synthetic_target__"] = (rnd.random(len(tmp)) < p).astype(int)
    target_col = "__synthetic_target__"

# ---------------------------------
# Train & Predict
# ---------------------------------
st.subheader("② Train model and generate predictions")
start = st.button("🚀 Start Model (Train & Predict)")

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

    # ---- Download predictions ----
    st.subheader("④ Download output CSV")
    st.dataframe(out_df.head(50), use_container_width=True)
    st.download_button(
        "⬇️ Download predictions CSV",
        data=bytes_from_df_csv(out_df),
        file_name="flight_risk_predictions.csv",
        mime="text/csv",
    )

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

st.caption("Tip: Upload a new file from the sidebar or rely on the default CSV in /mnt/data.")
