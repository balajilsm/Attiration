# streamlit_flight_risk_xgb.py
# ---------------------------------
# Streamlit app to:
#  1) Generate a synthetic CSV input dataset (2,000 employees by default) and let you download it
#  2) Train an XGBoost classifier on that dataset
#  3) Predict flight risk and let you download an output CSV with probabilities & labels
#
# Usage:
#   streamlit run streamlit_flight_risk_xgb.py --server.address=0.0.0.0 --server.port=8501
#
# Requirements (suggested versions):
#   pip install streamlit xgboost>=2.0 scikit-learn>=1.3 pandas>=2.0 numpy>=1.24

from __future__ import annotations
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

st.set_page_config(page_title="Flight Risk (XGBoost)", page_icon="✈️", layout="wide")
st.title("✈️ Employee Flight Risk — XGBoost Demo")

# ---------------------------------
# Synthetic data generator
# ---------------------------------

def _random_choice(options, size, p=None):
    return np.random.choice(options, size=size, p=p)


def generate_synthetic_hr(n: int = 2000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    departments = ["Engineering", "Sales", "HR", "Finance", "Operations", "Marketing"]
    states = ["GA", "TX", "CA", "NY", "NC", "FL", "WA", "OH"]
    job_levels = [1, 2, 3, 4, 5]
    genders = ["Male", "Female", "Other"]
    job_roles = ["Analyst", "Engineer", "Manager", "Director", "Associate"]

    emp_id = np.arange(1, n + 1)
    age = rng.integers(20, 65, size=n)
    tenure_years = np.clip(rng.normal(5.0, 3.0, size=n), 0, 40)
    job_level = _random_choice(job_levels, size=n, p=[0.35, 0.3, 0.2, 0.1, 0.05])
    department = _random_choice(departments, size=n)
    gender = _random_choice(genders, size=n, p=[0.5, 0.49, 0.01])
    job_role = _random_choice(job_roles, size=n)
    location_state = _random_choice(states, size=n)

    dept_base = {
        "Engineering": 95000,
        "Sales": 80000,
        "HR": 70000,
        "Finance": 90000,
        "Operations": 75000,
        "Marketing": 78000,
    }
    level_multiplier = np.array([0.7, 1.0, 1.3, 1.6, 2.0])
    base = np.array([dept_base[d] for d in department])
    salary = base * level_multiplier[np.array(job_level) - 1] * (1 + rng.normal(0, 0.15, size=n))
    salary = np.clip(salary, 35000, 300000)

    engagement_score = np.clip(rng.normal(70, 15, size=n), 0, 100)
    performance_rating = np.clip(np.round(rng.normal(3.2, 0.8, size=n)), 1, 5)
    wlb = np.clip(np.round(rng.normal(3.2, 0.9, size=n)), 1, 5)
    overtime_hours = np.clip(rng.normal(5, 7, size=n), 0, 40)
    commute_km = np.clip(rng.normal(18, 12, size=n), 0, 80)
    promotion_count = np.clip((tenure_years / 3.5 + rng.normal(0, 0.6, size=n)), 0, None)
    manager_changes = np.clip(rng.poisson(0.6, size=n), 0, 10)
    last_raise_pct = np.clip(rng.normal(4, 3, size=n), -10, 25)
    remote_ratio = _random_choice([0.0, 0.5, 1.0], size=n, p=[0.45, 0.25, 0.30])
    is_contract = _random_choice([0, 1], size=n, p=[0.85, 0.15])

    df = pd.DataFrame({
        "employee_id": emp_id,
        "age": age,
        "tenure_years": tenure_years,
        "job_level": job_level,
        "department": department,
        "gender": gender,
        "job_role": job_role,
        "location_state": location_state,
        "salary": salary,
        "engagement_score": engagement_score,
        "performance_rating": performance_rating,
        "work_life_balance": wlb,
        "overtime_hours": overtime_hours,
        "commute_distance_km": commute_km,
        "promotion_count": promotion_count,
        "manager_changes": manager_changes,
        "last_raise_pct": last_raise_pct,
        "remote_ratio": remote_ratio,
        "is_contract": is_contract,
    })

    # Latent propensity
    z = (
        -0.03 * df["engagement_score"]
        -0.35 * df["work_life_balance"]
        -0.25 * df["performance_rating"]
        +0.06 * df["overtime_hours"]
        +0.02 * df["commute_distance_km"]
        -0.08 * df["last_raise_pct"]
        +0.18 * (df["remote_ratio"] == 0.0).astype(float)
        +0.15 * (df["tenure_years"] < 1.0).astype(float)
        +0.10 * (df["tenure_years"] > 10).astype(float)
        +0.22 * df["is_contract"]
        +0.07 * df["manager_changes"]
    )
    dept_bias = {"Engineering": -0.10, "Sales": 0.25, "HR": -0.05, "Finance": 0.05, "Operations": 0.10, "Marketing": 0.05}
    z += df["department"].map(dept_bias).astype(float)
    p = 1 / (1 + np.exp(-z))
    df["flight_risk"] = (np.random.random(len(df)) < p).astype(int)

    return df


def bytes_from_df_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------
# Sidebar — data source & actions
# ---------------------------------
with st.sidebar:
    st.header("Data")
    n_rows = st.number_input("# of employees (for synthetic)", 2000, 100000, 2000, step=500)
    seed = st.number_input("Random seed", 0, 10_000, RANDOM_STATE, step=1)

    generate_btn = st.button("Generate sample dataset")
    st.markdown("Upload your own CSV (optional). If omitted, the generated sample will be used.")
    uploaded_file = st.file_uploader("CSV with columns like the sample (target 'flight_risk' optional)", type=["csv"])

# Maintain state of the base dataframe
if "base_df" not in st.session_state:
    st.session_state.base_df = generate_synthetic_hr(n_rows, seed)

if generate_btn:
    st.session_state.base_df = generate_synthetic_hr(n_rows, seed)

base_df = st.session_state.base_df.copy()

st.subheader("① Download / review input dataset")
st.write("This is the input data we'll use to train & predict. You can download it and/or upload your own.")
st.dataframe(base_df.head(50), use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "⬇️ Download sample input CSV",
        data=bytes_from_df_csv(base_df),
        file_name="flight_risk_input_sample.csv",
        mime="text/csv",
    )

# ---------------------------------
# Prep data for training/prediction
# ---------------------------------

st.subheader("② Train model & generate predictions")

use_uploaded = False
if uploaded_file is not None:
    try:
        df_in = pd.read_csv(uploaded_file)
        use_uploaded = True
        st.success("Using uploaded CSV for training/prediction.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        df_in = base_df.copy()
else:
    df_in = base_df.copy()

# Ensure required input columns exist
required_base_cols = [
    "employee_id","age","tenure_years","job_level","department","gender","job_role","location_state",
    "salary","engagement_score","performance_rating","work_life_balance","overtime_hours","commute_distance_km",
    "promotion_count","manager_changes","last_raise_pct","remote_ratio","is_contract",
]
missing = [c for c in required_base_cols if c not in df_in.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# If target is present, we'll train with it; otherwise we can still fit a model using the synthetic target from sample
has_target = "flight_risk" in df_in.columns
if not has_target:
    st.info("No 'flight_risk' column found. For demo training, we'll synthesize a temporary target (won't be saved back to your file).")
    tmp = generate_synthetic_hr(len(df_in))
    df_in["flight_risk"] = tmp["flight_risk"].values

# Train/test split
X_all = df_in[required_base_cols].copy()
y_all = df_in["flight_risk"].astype(int)

# Detect categorical columns as object dtype + remote_ratio/is_contract kept numeric
categorical_cols = [c for c in X_all.columns if X_all[c].dtype == "object"]

# One-hot encode
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, stratify=y_all, random_state=RANDOM_STATE)
X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
X_test_enc = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
feature_cols = list(X_train_enc.columns)

# Train XGBoost
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

# Evaluate
proba_test = model.predict_proba(X_test_enc)[:, 1]
preds_test = (proba_test >= 0.5).astype(int)
roc = roc_auc_score(y_test, proba_test)
acc = accuracy_score(y_test, preds_test)
f1 = f1_score(y_test, preds_test)
cm = confusion_matrix(y_test, preds_test)

m1, m2, m3 = st.columns(3)
with m1: st.metric("ROC AUC", f"{roc:.3f}")
with m2: st.metric("Accuracy", f"{acc:.3f}")
with m3: st.metric("F1", f"{f1:.3f}")

st.write("**Confusion Matrix**  ")
st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

st.expander("Classification report").text(classification_report(y_test, preds_test))

# Predict for full input and offer download
full_enc = pd.get_dummies(X_all, columns=categorical_cols, drop_first=False)
full_enc = full_enc.reindex(columns=feature_cols, fill_value=0)
full_proba = model.predict_proba(full_enc)[:, 1]
full_pred = (full_proba >= 0.5).astype(int)

out_df = df_in.copy()
out_df["flight_risk_prob"] = full_proba
out_df["flight_risk_pred"] = full_pred

st.subheader("③ Download predictions")
st.dataframe(out_df.head(50), use_container_width=True)

csv_bytes = bytes_from_df_csv(out_df)
st.download_button(
    "⬇️ Download predictions CSV",
    data=csv_bytes,
    file_name="flight_risk_predictions.csv",
    mime="text/csv",
)

# Optional: download model + feature columns
art_col1, art_col2 = st.columns(2)
with art_col1:
    st.download_button(
        "⬇️ Download feature columns (JSON)",
        data=json.dumps(feature_cols, indent=2).encode("utf-8"),
        file_name="feature_columns.json",
        mime="application/json",
    )
with art_col2:
    # Save XGBoost model to JSON text (in-memory)
    buf = io.BytesIO()
    # xgboost Python API doesn't export to bytes directly, so we save to string via Booster.save_raw
    booster = model.get_booster()
    raw_bytes = booster.save_raw()
    st.download_button(
        "⬇️ Download XGBoost model (binary)",
        data=raw_bytes,
        file_name="flight_risk_model.xgb",
        mime="application/octet-stream",
    )

st.caption("Tip: If you upload your own CSV without a 'flight_risk' column, the app will synthesize a temporary target just for demo training. For production, include your real target column to train on historical outcomes.")
