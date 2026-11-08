"""Breast Cancer ML Lab: evaluate saved classifiers on labelled test data."""
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_curve,
)

st.set_page_config(page_title="Breast Cancer ML Lab", page_icon="🧪", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "Diagnosis"

# UCI Wisconsin Diagnostic dataset predictors. ID is intentionally excluded.
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine (SVM)": "svm.joblib",
}


@st.cache_resource
def load_models():
    loaded, missing, errors = {}, [], {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            missing.append(filename)
            continue
        try:
            loaded[name] = joblib.load(path)
        except Exception as exc:
            errors[filename] = str(exc)
    return loaded, missing, errors


def encode_target(values):
    mapping = {
        "B": 0, "BENIGN": 0, "0": 0,
        "M": 1, "MALIGNANT": 1, "1": 1,
    }
    normalized = values.astype(str).str.strip().str.upper()
    encoded = normalized.map(mapping)
    if encoded.isna().any():
        bad = sorted(normalized[encoded.isna()].unique().tolist())
        raise ValueError(
            "Diagnosis must contain B/M, Benign/Malignant, or 0/1. "
            f"Unrecognized values: {bad}"
        )
    return encoded.astype(int)


def positive_scores(model, X):
    """Return scores for class 1, for AUC/ROC when the model supports it."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(getattr(model, "classes_", []))
        index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
        return probabilities[:, index]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        classes = list(getattr(model, "classes_", []))
        if len(classes) == 2 and classes[1] == 0:
            scores = -scores
        return scores
    return None


def evaluate(model, X, y):
    predicted = model.predict(X)
    scores = positive_scores(model, X)
    auc_value = float("nan")
    if scores is not None and y.nunique() == 2:
        fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
        auc_value = auc(fpr, tpr)
    metrics = {
        "Accuracy": accuracy_score(y, predicted),
        "AUC": auc_value,
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "F1 Score": f1_score(y, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y, predicted),
    }
    return predicted, scores, metrics


st.title("🧪 Breast Cancer ML Lab")
st.write(
    "Evaluate five saved classification models on a labelled test CSV. "
    "Models are loaded from the `model/` folder; the app does not retrain them."
)
st.info(
    "Educational project only. This dashboard is not a medical diagnostic tool "
    "and must not be used to make healthcare decisions."
)

with st.expander("How to use this dashboard"):
    st.markdown(
        "1. Upload the held-out `test_data.csv` exported by your notebook.\n"
        "2. The file must contain all 30 predictor columns and a `Diagnosis` column.\n"
        "3. Choose a model and inspect its metrics, confusion matrix, report, and ROC curve.\n"
        "4. Compare all available models in the table below.\n\n"
        "`B` = benign and `M` = malignant. The app maps these to 0 and 1 internally. "
        "An `ID` column may be present, but is not used as a feature."
    )

models, missing, load_errors = load_models()
if missing:
    st.error("Missing model files. Place the files below inside `model/`:")
    st.code("\n".join(missing))
if load_errors:
    st.error("Some models could not be loaded. Check package-version compatibility.")
    for filename, message in load_errors.items():
        st.write(f"**{filename}:** `{message}`")
if not models:
    st.stop()

st.subheader("1. Upload labelled test data")
uploaded = st.file_uploader(
    "Choose your test CSV", type=["csv"],
    help="Upload the held-out test set, not the full training dataset.",
)
if uploaded is None:
    st.caption("Upload the test CSV to begin.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
if TARGET not in data.columns:
    st.error(f"The CSV must contain a target column named `{TARGET}`.")
    st.stop()

missing_features = [col for col in FEATURES if col not in data.columns]
if missing_features:
    st.error("Required predictor columns are missing. Check the uploaded CSV.")
    st.code(", ".join(missing_features))
    st.stop()

X_test = data[FEATURES].copy()
for col in FEATURES:
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
if X_test.isna().any().any():
    bad_columns = X_test.columns[X_test.isna().any()].tolist()
    st.error("Some predictor values are missing or non-numeric:")
    st.code(", ".join(bad_columns))
    st.stop()

try:
    y_test = encode_target(data[TARGET])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if y_test.nunique() < 2:
    st.warning("The test target contains only one class; AUC is not meaningful.")

st.success("Test data loaded and validated.")
c1, c2, c3 = st.columns(3)
c1.metric("Test rows", f"{len(data):,}")
c2.metric("Input features", len(FEATURES))
c3.metric("Models available", len(models))

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10), use_container_width=True)
    st.write("Target distribution:")
    st.dataframe(data[TARGET].astype(str).value_counts().rename("Count"))

st.subheader("2. Select a model")
chosen_name = st.selectbox("Model", list(models.keys()))
chosen_model = models[chosen_name]
try:
    predictions, scores, metrics = evaluate(chosen_model, X_test, y_test)
except Exception as exc:
    st.error(
        "The selected model could not evaluate this CSV. Check that the model was "
        "trained with these same 30 feature columns and compatible package versions."
    )
    st.exception(exc)
    st.stop()

metric_cols = st.columns(6)
for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, "N/A" if pd.isna(value) else f"{value:.3f}")
st.caption("Metrics are calculated against the Diagnosis labels in the uploaded CSV.")

st.subheader("3. Confusion matrix")
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4))
image = ax_cm.imshow(cm, interpolation="nearest")
ax_cm.set_title(f"{chosen_name} — Confusion Matrix")
ax_cm.set_xlabel("Predicted label")
ax_cm.set_ylabel("Actual label")
ax_cm.set_xticks([0, 1], labels=["Benign (B)", "Malignant (M)"])
ax_cm.set_yticks([0, 1], labels=["Benign (B)", "Malignant (M)"])
for row in range(2):
    for col in range(2):
        ax_cm.text(col, row, str(cm[row, col]), ha="center", va="center")
fig_cm.colorbar(image, ax=ax_cm, fraction=0.046, pad=0.04)
fig_cm.tight_layout()
st.pyplot(fig_cm)
plt.close(fig_cm)

st.subheader("4. Classification report")
report = classification_report(
    y_test, predictions, labels=[0, 1],
    target_names=["Benign (B)", "Malignant (M)"],
    output_dict=True, zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.subheader("5. ROC curve")
if scores is not None and y_test.nunique() == 2:
    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Chance baseline")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{chosen_name} — ROC Curve")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    st.pyplot(fig_roc)
    plt.close(fig_roc)
else:
    st.info("ROC curve unavailable for this model or because the test data has one class.")

st.subheader("6. Compare all available models")
rows = []
for name, model in models.items():
    try:
        _, _, result = evaluate(model, X_test, y_test)
        rows.append({"Model": name, **result})
    except Exception as exc:
        warnings.warn(f"Could not evaluate {name}: {exc}")
        rows.append({
            "Model": name, "Accuracy": float("nan"), "AUC": float("nan"),
            "Precision": float("nan"), "Recall": float("nan"),
            "F1 Score": float("nan"), "MCC": float("nan"),
        })

comparison = pd.DataFrame(rows).set_index("Model")
st.dataframe(comparison.round(3), use_container_width=True)
st.download_button(
    "Download comparison table as CSV",
    data=comparison.reset_index().to_csv(index=False).encode("utf-8"),
    file_name="model_comparison_on_uploaded_test_data.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Use your own final notebook results and explanations in the assignment README. "
    "This dashboard reports performance on the uploaded test set and does not establish "
    "clinical reliability."
)
