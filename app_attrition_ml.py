
import io
import sys
import math
import json
import warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


from pandas.api.types import is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CSV → ML Predictor (Streamlit)", layout="wide")

st.title("📈 CSV → Feature Selection → Model Comparison → Best Model → Predictions")
st.caption("Upload a CSV, select target & features, compare models, pick the best, and export predictions.")

# -----------------------------
# Utilities
# -----------------------------
def _detect_encoding_from_bytes(b: bytes) -> str:
    # quick heuristic; utf-8 most common
    sample = b[:4096]
    try:
        sample.decode("utf-8")
        return "utf-8"
    except Exception:
        pass
    # fallback attempts
    for enc in ("cp1252", "latin1"):
        try:
            sample.decode(enc)
            return enc
        except Exception:
            pass
    return "latin1"  # last resort

@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    data = uploaded_file.read()
    enc = _detect_encoding_from_bytes(data)
    try:
        return pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
    except UnicodeDecodeError:
        # try permissive
        return pd.read_csv(io.BytesIO(data.decode(enc, errors="ignore").encode()), low_memory=False)

def make_ohe() -> OneHotEncoder:
    """
    Create a OneHotEncoder that works across sklearn versions.
    Newer sklearn uses sparse_output; older uses sparse.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df[feature_cols].copy()
    numeric_cols = [c for c in feature_cols if is_numeric_dtype(X[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, categorical_cols

def build_models(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    models["LogisticRegression"] = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None if hasattr(LogisticRegression(), "n_jobs") else None)
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample"
    )
    models["GradientBoosting"] = GradientBoostingClassifier(random_state=random_state)
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            objective="binary:logistic",
            n_jobs=0,
            eval_metric="logloss",
            tree_method="hist"
        )
    return models

def evaluate_model(clf, X_test, y_test, proba_threshold: float = 0.5) -> Dict[str, float]:
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        # scale to (0,1) via logistic for comparison
        df_scores = clf.decision_function(X_test)
        y_proba = 1.0 / (1.0 + np.exp(-df_scores))
    else:
        # fallback to predictions
        y_pred = clf.predict(X_test)
        y_proba = y_pred.astype(float)

    y_pred_thresh = (y_proba >= proba_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_thresh)),
        "precision": float(precision_score(y_test, y_pred_thresh, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_thresh, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_thresh, zero_division=0)),
    }
    # ROC AUC requires both classes
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics

def rank_models(results_df: pd.DataFrame, primary_metric: str = "roc_auc", secondary_metric: str = "f1") -> str:
    df = results_df.copy()
    # Replace nan with -inf to avoid ranking issues
    df[primary_metric] = df[primary_metric].fillna(-np.inf)
    df[secondary_metric] = df[secondary_metric].fillna(-np.inf)
    df = df.sort_values(by=[primary_metric, secondary_metric], ascending=[False, False])
    return df.index[0] if len(df) else ""

# -----------------------------
# Sidebar: Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    test_size = st.slider("Test size (validation split)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state (reproducibility)", value=42, step=1)
    threshold = st.slider("Prediction threshold (for classification)", 0.05, 0.95, 0.5, 0.05)
    rank_metric = st.selectbox("Primary ranking metric", ["roc_auc", "f1", "accuracy", "precision", "recall"], index=0)

# -----------------------------
# Main: Upload & Configure
# -----------------------------
uploaded = st.file_uploader("📤 Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = load_csv_from_upload(uploaded)
st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20))

# Target selection
target_col = st.selectbox("🎯 Select target column (binary classification; e.g., 1=leave / 0=stay)", options=list(df.columns))
if target_col is None:
    st.warning("Please pick a target column.")
    st.stop()

# Convert target to 0/1 int robustly
y_raw = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
# Normalize any 'Active' or 'Yes/No' variants to 0/1 if needed
if sorted(pd.unique(y_raw)) == [0, 1]:
    y = y_raw.copy()
else:
    # If non-binary, try mapping commonly seen variants
    mapping = {"yes": 1, "y": 1, "true": 1, "t": 1, "left": 1, "terminated": 1,
               "no": 0, "n": 0, "false": 0, "f": 0, "stay": 0, "active": 0}
    y = df[target_col].astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)

# Feature selection
default_features = [c for c in df.columns if c != target_col]
feature_cols = st.multiselect("🧩 Select feature columns", options=[c for c in df.columns if c != target_col], default=default_features)

if len(feature_cols) == 0:
    st.error("Please select at least one feature.")
    st.stop()

# -----------------------------
# Preprocessing & Split
# -----------------------------
preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col, feature_cols)

X = df[feature_cols].copy()

# Train/validation split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
except Exception as e:
    st.error(f"❌ Train/test split failed: {e}")
    st.stop()

# -----------------------------
# Model Selection
# -----------------------------
st.subheader("🤖 Choose models to train and compare")
available_models = build_models(random_state=random_state)
selected_model_names = []

cols = st.columns(4)
i = 0
for name in available_models.keys():
    with cols[i % 4]:
        on = st.checkbox(name, value=True)
    if on:
        selected_model_names.append(name)
    i += 1

if len(selected_model_names) == 0:
    st.warning("Select at least one model to train.")
    st.stop()

# -----------------------------
# Train & Evaluate
# -----------------------------
results = {}
trained_models = {}

with st.spinner("Training models..."):
    for name in selected_model_names:
        model = available_models[name]
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            metrics = evaluate_model(pipe, X_test, y_test, proba_threshold=threshold)
            results[name] = metrics
            trained_models[name] = pipe
        except Exception as e:
            results[name] = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan}
            st.error(f"❌ {name} failed: {e}")

# Show results
if not results:
    st.error("No models were successfully trained.")
    st.stop()

results_df = pd.DataFrame(results).T
st.subheader("📊 Validation metrics (higher is better)")
st.dataframe(results_df.style.format("{:.4f}"))

# Matplotlib bar chart per metric (no specific colors)
import matplotlib.pyplot as plt

for metric in ["roc_auc", "f1", "accuracy", "precision", "recall"]:
    fig = plt.figure()
    vals = results_df[metric].fillna(0.0)
    ax = vals.plot(kind="bar")
    ax.set_title(f"{metric} by model")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# -----------------------------
# Pick best model
# -----------------------------
best_name = rank_models(results_df, primary_metric=rank_metric, secondary_metric="f1")
if best_name == "":
    st.error("Could not determine a best model.")
    st.stop()

st.success(f"🏆 Best model by **{rank_metric}** (tie-breaker: F1): **{best_name}**")

best_model = trained_models[best_name]

# -----------------------------
# Predict full dataset & Download
# -----------------------------
st.subheader("🧾 Generate predictions with the best model")
proba_col = f"{best_name}_probability"
pred_col = f"{best_name}_prediction"

try:
    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X)[:, 1]
    elif hasattr(best_model, "decision_function"):
        scores = best_model.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-scores))
    else:
        proba = best_model.predict(X).astype(float)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

pred = (proba >= threshold).astype(int)

out_df = df.copy()
out_df[proba_col] = proba
out_df[pred_col] = pred

st.write("Preview of output with predictions:")
st.dataframe(out_df.head(20))

csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download predictions CSV",
    data=csv_bytes,
    file_name="predictions_with_best_model.csv",
    mime="text/csv"
)

# -----------------------------
# Optional: Show confusion matrix for the validation split
# -----------------------------
with st.expander("Validation Confusion Matrix & Classification Report"):
    # Use validation set predictions
    if hasattr(best_model, "predict_proba"):
        val_proba = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        val_scores = best_model.decision_function(X_test)
        val_proba = 1.0 / (1.0 + np.exp(-val_scores))
    else:
        val_proba = best_model.predict(X_test).astype(float)
    val_pred = (val_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, val_pred)
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
    st.text("Classification Report:")
    st.text(classification_report(y_test, val_pred, zero_division=0))
