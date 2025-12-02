# app.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

from model import (
    train_model,
    predict_dataframe,
    save_model
)

st.set_page_config(page_title="Stockout Prediction", layout="wide")

st.title("📦 Stockout Prediction App (LR / SVM)")

# ================================
# FILE UPLOAD
# ================================
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.write(df.head())

# ================================
# CHOOSE TARGET + FEATURES
# ================================
all_cols = df.columns.tolist()

possible_targets = ["stockout_indicator", "stockout", "target"]
target_detected = None
for c in all_cols:
    if c.lower() in possible_targets:
        target_detected = c
        break

target = st.selectbox("Pilih Kolom Target", all_cols, index=all_cols.index(target_detected) if target_detected else 0)

numerics = df.select_dtypes(include=[np.number]).columns.tolist()
if target in numerics:
    numerics.remove(target)

features = st.multiselect("Pilih fitur numerik", numerics, default=numerics[:2])

if not features:
    st.error("Minimal 1 fitur diperlukan.")
    st.stop()

# ================================
# MODEL OPTIONS
# ================================
st.sidebar.header("Pengaturan model")
algo = st.sidebar.selectbox("Algoritma", ["Logistic Regression", "SVM (RBF kernel)"])
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
random_state = st.sidebar.number_input("Random State", 0, 999, 42)
tuning = st.sidebar.checkbox("Hyperparameter Tuning (GridSearchCV)", value=True)

# ================================
# TRAINING
# ================================
st.header("2. Training Model")

if st.button("Train Model"):
    with st.spinner("Training model..."):
        results = train_model(
            df,
            features=features,
            target=target,
            algorithm="Logistic Regression" if algo.startswith("Logistic") else "SVM",
            test_size=test_size,
            random_state=random_state,
            tuning=tuning
        )

    st.success("Training selesai!")

    # Metrics display
    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {results['accuracy']:.4f}")
    st.write(f"Precision: {results['precision']:.4f}")
    st.write(f"Recall: {results['recall']:.4f}")
    st.write(f"F1: {results['f1']:.4f}")
    st.write(f"AUC: {results['auc']:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(results["confusion_matrix"], annot=True, fmt="d",
                cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Best params
    if results["best_params"] is not None:
        st.subheader("Best Parameters (GridSearch)")
        st.json(results["best_params"])

    # Save model
    file_path = save_model(results["model"])
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="model.joblib">Download Model</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.session_state["model"] = results["model"]
    st.session_state["features"] = features

# ================================
# PREDICTION
# ================================
st.header("3. Prediksi dengan Model")

if "model" in st.session_state:
    pred_file = st.file_uploader("Upload CSV untuk prediksi", type=["csv"], key="pred")

    if pred_file:
        df_pred = pd.read_csv(pred_file)

        preds, proba = predict_dataframe(st.session_state["model"], df_pred, st.session_state["features"])

        df_pred["pred_stockout"] = preds
        if proba is not None:
            df_pred["pred_proba"] = proba

        st.write(df_pred.head())

        csv_buffer = BytesIO()
        df_pred.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        b64 = base64.b64encode(csv_buffer.read()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Hasil Prediksi</a>',
                    unsafe_allow_html=True)
else:
    st.info("Train model dulu untuk melakukan prediksi.")
