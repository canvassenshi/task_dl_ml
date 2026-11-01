import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import os

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="AI Project: Defect Detection & RUL Prediction", layout="wide")

st.title("🎓 Deep Learning Project: Defect Detection & RUL Prediction")
st.write("This project combines Computer Vision (Casting Defect Classification) and Time-Series Regression (Turbofan Remaining Useful Life).")

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("Navigation")
task = st.sidebar.radio("Select Task", ["Image Classification (Casting Defect)", "Regression (Turbofan Engine RUL)"])

models_dir = "Models"

# ---------------------------------------------------------
# TASK 1: Image Classification
# ---------------------------------------------------------
if task == "Image Classification (Casting Defect)":
    st.header("📸 Task 1: Casting Defect Detection")
    st.markdown("""
    **How it works:**
    1. Select a model from the dropdown (`mobilenet_v2.keras` recommended).
    2. Upload a casting part image (`.jpg` or `.png`).
    3. Click **Classify Image** to see if the part is **Normal (Pass)** or **Defective (Reject)**.
    """)

    # Model selector
    model_choice = st.selectbox(
        "Choose Model",
        ["mobilenet_v2.keras", "resnet50.keras"]
    )
    model_path = os.path.join(models_dir, model_choice)

    # File uploader
    uploaded_file = st.file_uploader("Upload an Image of Casting Product", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview", width=300)

        # Predict button
        if st.button("Classify Image", type="primary"):
            if not os.path.exists(model_path):
                st.error(f"Model file not found at: `{model_path}`. Please ensure your models are placed in the `Models/` folder.")
            else:
                with st.spinner("Classifying image..."):
                    # 1. Load model
                    model = load_model(model_path)

                    # 2. Preprocess: resize to 224x224 and scale to [0, 1]
                    img_rgb = image.convert("RGB")
                    img_resized = img_rgb.resize((224, 224))
                    img_array = np.array(img_resized) / 255.0
                    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

                    # 3. Model Inference
                    predictions = model.predict(img_batch)
                    prob_normal = float(predictions[0][0])
                    prob_defective = float(predictions[0][1])

                    # 4. Show Result
                    st.subheader("Inspection Result:")
                    if prob_defective > prob_normal:
                        st.error(f"❌ **Defective Part Detected** (Confidence: {prob_defective * 100:.2f}%)")
                    else:
                        st.success(f"✅ **Normal Part (Pass)** (Confidence: {prob_normal * 100:.2f}%)")

                    # Display bar chart of probabilities
                    chart_data = pd.DataFrame({
                        "Class": ["Normal (Pass)", "Defective (Reject)"],
                        "Confidence Score": [prob_normal, prob_defective]
                    })
                    st.bar_chart(chart_data.set_index("Class"))

# ---------------------------------------------------------
# TASK 2: Turbofan Engine RUL Regression
# ---------------------------------------------------------
else:
    st.header("⏱️ Task 2: Turbofan Engine Remaining Useful Life (RUL)")
    st.markdown("""
    **How it works:**
    1. Select a regression model (`Seq2Seq LSTM`, `SimpleRNN`, or `Static ANN`).
    2. Provide sensor telemetry (either upload a `.csv` file or use pre-loaded sample data).
    3. The model processes 25 operational parameters across 30 flight cycles and estimates remaining engine life.
    """)

    # Model selector
    reg_model_choice = st.selectbox(
        "Choose Regression Model",
        ["seq2seq_lstm_best.keras", "seq2seq_rnn_best.keras", "static_ann_model.pkl"]
    )
    reg_model_path = os.path.join(models_dir, reg_model_choice)
    scaler_path = os.path.join(models_dir, "seq2seq_scaler.pkl")

    # 25 features expected by the models
    feature_cols = [
        'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
        'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
        'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
        'sensor_21'
    ]

    # Input method selector
    input_method = st.radio("Choose Input Method:", ["Use Sample Demo Engine Data", "Upload CSV File"], horizontal=True)

    df_to_predict = None

    if input_method == "Use Sample Demo Engine Data":
        sample_choice = st.selectbox(
            "Select Engine Condition:",
            ["Early Life Engine (Healthy - Cycle 25)", "Mid-Life Engine (Cycle 120)", "High-Wear Engine (Near Failure - Cycle 195)"]
        )

        # Generate realistic 30-cycle sample data matching dataset statistics
        np.random.seed(42 if "Early" in sample_choice else (99 if "Mid" in sample_choice else 7))
        base_c = 25 if "Early" in sample_choice else (120 if "Mid" in sample_choice else 195)
        wear = 0.05 if "Early" in sample_choice else (0.35 if "Mid" in sample_choice else 0.85)

        demo_data = {
            'cycle': np.arange(base_c - 29, base_c + 1),
            'op_setting_1': np.random.normal(0.000, 0.001, 30),
            'op_setting_2': np.random.normal(0.000, 0.0002, 30),
            'op_setting_3': [100.0] * 30
        }
        for i in range(1, 22):
            demo_data[f'sensor_{i}'] = 500.0 + (i * 10.0) + (wear * 5.0) + np.random.normal(0, 0.5, 30)

        df_to_predict = pd.DataFrame(demo_data)[feature_cols]
        st.write("Loaded Sample Sensor Data (Last 5 Cycles):")
        st.dataframe(df_to_predict.tail())

    else:
        uploaded_csv = st.file_uploader("Upload Engine Sensor Data (.csv)", type=["csv"])
        if uploaded_csv is not None:
            raw_df = pd.read_csv(uploaded_csv)
            missing = [c for c in feature_cols if c not in raw_df.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing[:5]}...")
            else:
                df_to_predict = raw_df[feature_cols]
                st.write("Uploaded Sensor Data Preview:")
                st.dataframe(df_to_predict.tail())

    # Predict button
    if df_to_predict is not None:
        if st.button("Predict RUL", type="primary"):
            if not os.path.exists(scaler_path):
                st.error(f"Scaler not found at `{scaler_path}`. Please place `seq2seq_scaler.pkl` in the `Models/` folder.")
            elif not os.path.exists(reg_model_path):
                st.error(f"Model not found at `{reg_model_path}`.")
            else:
                with st.spinner("Standardizing sensor signals and calculating RUL..."):
                    # 1. Load fitted StandardScaler and scale features
                    scaler = joblib.load(scaler_path)
                    scaled_values = scaler.transform(df_to_predict[feature_cols].values)

                    # 2. Sequence (LSTM/RNN) vs Static ANN
                    if reg_model_choice.endswith(".keras"):
                        # Sequence models need the latest 30-cycle window: shape (1, 30, 25)
                        if len(scaled_values) < 30:
                            st.warning("Data has fewer than 30 cycles; padding earlier cycles.")
                            pad = np.tile(scaled_values, (int(np.ceil(30 / len(scaled_values))), 1))
                            seq_window = pad[-30:]
                        else:
                            seq_window = scaled_values[-30:]

                        seq_input = np.expand_dims(seq_window, axis=0)  # Shape: (1, 30, 25)

                        # Load model & predict
                        model = load_model(reg_model_path)
                        pred_seq = model.predict(seq_input)  # Shape: (1, 30, 1)

                        final_rul = float(pred_seq[0, -1, 0])
                        trajectory = pred_seq[0, :, 0]

                        # 3. Display Results
                        st.subheader("RUL Prediction Result:")
                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Remaining Life", f"{final_rul:.1f} Cycles")
                        col2.metric("Current Cycle", f"{int(df_to_predict['cycle'].iloc[-1])}")

                        # Maintenance Status Alert
                        if final_rul < 30:
                            st.error("🚨 **Critical Warning**: Engine is near failure! Immediate overhaul required.")
                        elif final_rul < 75:
                            st.warning("⚠️ **Maintenance Alert**: Plan component replacement within next 20-30 cycles.")
                        else:
                            st.success("🟢 **Normal Condition**: Engine operating within safe parameters.")

                        # Degradation line chart
                        st.write("Estimated Degradation Trajectory (Past 30 Cycles):")
                        chart_df = pd.DataFrame({
                            "Cycle": df_to_predict['cycle'].values[-30:],
                            "Estimated RUL": trajectory
                        }).set_index("Cycle")
                        st.line_chart(chart_df)

                    else:
                        # Static ANN model (uses latest single cycle snapshot: 1 x 25)
                        ann_model = joblib.load(reg_model_path)
                        latest_snapshot = scaled_values[-1:]
                        final_rul = float(ann_model.predict(latest_snapshot)[0])

                        st.subheader("RUL Prediction Result:")
                        st.metric("Predicted Remaining Life", f"{final_rul:.1f} Cycles")

                        if final_rul < 30:
                            st.error("🚨 Critical Alert: High risk of engine failure!")
                        elif final_rul < 75:
                            st.warning("⚠️ Warning: Schedule maintenance soon.")
                        else:
                            st.success("🟢 Healthy: Engine in good condition.")