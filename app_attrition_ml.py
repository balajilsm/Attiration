
import io
import warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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

st.set_page_config(page_title="CSV → ML Predictor (Robust)", layout="wide")

st.title("📈 CSV → Feature Selection → Model Comparison → Best Model → Predictions")
st.caption("Robust handling for single-class targets, stratified split, and simple undersampling.")

# -----------------------------
# Utilities
# -----------------------------
def _detect_encoding_from_bytes(b: bytes) -> str:
    sample = b[:4096]
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            sample.decode(enc)
            return enc
        except Exception:
            continue
    return "latin1"

@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    data = uploaded_file.read()
    enc = _detect_encoding_from_bytes(data)
    try:
        return pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data.decode(enc, errors="ignore").encode()), low_memory=False)

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    from pandas.api.types import is_numeric_dtype
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
    models["LogisticRegression"] = LogisticRegression(max_iter=1000, class_weight="balanced")
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=random_state, class_weight="balanced_subsample"
    )
    models["GradientBoosting"] = GradientBoostingClassifier(random_state=random_state)
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=random_state, objective="binary:logistic",
            n_jobs=0, eval_metric="logloss", tree_method="hist"
        )
    return models

def evaluate_model(clf, X_test, y_test, proba_threshold: float = 0.5):
    # If model ended up single-class, bail gracefully
    try:
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            df_scores = clf.decision_function(X_test)
            y_proba = 1.0 / (1.0 + np.exp(-df_scores))
        else:
            y_pred = clf.predict(X_test)
            y_proba = y_pred.astype(float)
    except Exception:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan}

    y_pred_thresh = (y_proba >= proba_threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_thresh)),
        "precision": float(precision_score(y_test, y_pred_thresh, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_thresh, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_thresh, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics

def rank_models(results_df: pd.DataFrame, primary_metric: str = "roc_auc", secondary_metric: str = "f1") -> str:
    df = results_df.copy()
    for m in [primary_metric, secondary_metric]:
        if m not in df.columns:
            df[m] = np.nan
        df[m] = df[m].fillna(-np.inf)
    df = df.sort_values(by=[primary_metric, secondary_metric], ascending=[False, False])
    return df.index[0] if len(df) else ""

def simple_undersample(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    vc = y.value_counts()
    if len(vc) < 2:
        return X, y
    n_min = vc.min()
    idxs = []
    for cls, n in vc.items():
        cls_idx = y[y == cls].index
        take = min(n_min, len(cls_idx))
        idxs.append(cls_idx.to_series().sample(n=take, random_state=random_state).index)
    keep = idxs[0].union_many(idxs[1:]) if hasattr(idxs[0], "union_many") else idxs[0].union(idxs[1])
    return X.loc[keep], y.loc[keep]

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    test_size = st.slider("Test size (validation split)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)
    threshold = st.slider("Prediction threshold", 0.05, 0.95, 0.5, 0.05)
    rank_metric = st.selectbox("Primary ranking metric", ["roc_auc", "f1", "accuracy", "precision", "recall"], index=0)
    do_undersample = st.checkbox("Balance training set by undersampling majority class", value=False)

# -----------------------------
# Main
# -----------------------------
uploaded = st.file_uploader("📤 Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = load_csv_from_upload(uploaded)
st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20))

# Target & features
target_col = st.selectbox("🎯 Target column (binary: 0/1)", options=list(df.columns))

# Try robust mapping to 0/1
y_raw_numeric = pd.to_numeric(df[target_col], errors="coerce")
if y_raw_numeric.notna().all():
    y = y_raw_numeric.fillna(0).astype(int)
else:
    mapping = {"yes": 1, "y": 1, "true": 1, "t": 1, "left": 1, "terminated": 1, "1": 1,
               "no": 0, "n": 0, "false": 0, "f": 0, "stay": 0, "active": 0, "0": 0}
    y = df[target_col].astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)

st.write("Target class counts:", y.value_counts(dropna=False).to_dict())

if y.nunique() < 2:
    st.error("❌ Your target only has one class. Please choose a different target or remap values so both classes (0 and 1) are present.")
    st.stop()

feature_cols = st.multiselect("🧩 Select feature columns", options=[c for c in df.columns if c != target_col],
                              default=[c for c in df.columns if c != target_col])

if len(feature_cols) == 0:
    st.error("Please select at least one feature.")
    st.stop()

preprocessor, num_cols, cat_cols = build_preprocessor(df, feature_cols)
X = df[feature_cols].copy()

# Split with stratification; if it fails, try fallback strategies
split_ok = False
err_msg = None
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Ensure both sets contain both classes
    if y_train.nunique() == 2 and y_test.nunique() == 2:
        split_ok = True
except Exception as e:
    err_msg = str(e)

if not split_ok:
    # Try StratifiedShuffleSplit explicitly
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_idx, test_idx in sss.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() == 2 and y_test.nunique() == 2:
            split_ok = True
    except Exception as e:
        err_msg = str(e)

if not split_ok:
    st.error("❌ Could not create a split that contains both classes in train and test.\n"
             "Tips: reduce test size, enable undersampling, or verify the minority class has enough samples (≥2).")
    st.stop()

# Optional undersampling on training set only
if do_undersample:
    X_train, y_train = simple_undersample(X_train, y_train, random_state=random_state)
    st.write("After undersampling class counts (train):", y_train.value_counts().to_dict())
    if y_train.nunique() < 2:
        st.error("❌ After undersampling, only one class remained. Disable undersampling or adjust parameters.")
        st.stop()

# Model training & evaluation
st.subheader("🤖 Choose models")
available = build_models(random_state=random_state)
selected_names = []
cols = st.columns(4)
for i, name in enumerate(available.keys()):
    with cols[i % 4]:
        if st.checkbox(name, value=True):
            selected_names.append(name)
if not selected_names:
    st.error("Select at least one model.")
    st.stop()

results, trained = {}, {}
with st.spinner("Training models..."):
    for name in selected_names:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", available[name])])
        try:
            pipe.fit(X_train, y_train)
            # Guard: if model accidentally trained as single-class, skip
            if hasattr(pipe.named_steps["model"], "classes_") and len(getattr(pipe.named_steps["model"], "classes_", [])) < 2:
                st.warning(f"⚠️ {name} trained on a single class — skipping metrics.")
                results[name] = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan}
                continue
            metrics = evaluate_model(pipe, X_test, y_test, proba_threshold=threshold)
            results[name] = metrics
            trained[name] = pipe
        except Exception as e:
            st.error(f"❌ {name} failed: {e}")
            results[name] = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan}

if not trained:
    st.error("❌ No models finished training with two classes. Verify your target and split.")
    st.stop()

results_df = pd.DataFrame(results).T
st.subheader("📊 Validation metrics")
st.dataframe(results_df.style.format("{:.4f}"))

# Charts
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

# Best model
best_name = rank_models(results_df, primary_metric=rank_metric, secondary_metric="f1")
if best_name == "" or best_name not in trained:
    st.error("Could not determine a best model with valid metrics.")
    st.stop()

st.success(f"🏆 Best model by **{rank_metric}** (tie-breaker F1): **{best_name}**")
best_model = trained[best_name]

# Predict entire dataset
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
    pred = (proba >= threshold).astype(int)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

out_df = df.copy()
out_df[proba_col] = proba
out_df[pred_col] = pred

st.write("Preview of output with predictions:")
st.dataframe(out_df.head(20))

csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download predictions CSV", data=csv_bytes, file_name="predictions_with_best_model.csv", mime="text/csv")

# Validation confusion matrix/report
with st.expander("Validation Confusion Matrix & Classification Report"):
    try:
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
    except Exception as e:
        st.warning(f"Could not compute confusion matrix/report: {e}")
