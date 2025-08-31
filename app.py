# app.py
# Streamlit Application - Section 4 from Wiki

import streamlit as st
import tensorflow as tf
import numpy as np
import joblib


# Load the trained model once when the app starts
@st.cache_resource  # This decorator prevents the model from reloading every time Streamlit reruns
def load_drowsiness_model():
    """
    4.2 Build the Streamlit Application
    Load the saved model for predictions
    """
    try:
        model = tf.keras.models.load_model('drowsiness_detector.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'drowsiness_detector.h5' is in the same directory.")
        return None


@st.cache_resource
def load_scaler():
    """Load the saved scaler"""
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}. Make sure 'scaler.pkl' is in the same directory.")
        return None


def main():
    """Main Streamlit application"""

    # Load model and scaler
    model = load_drowsiness_model()
    scaler = load_scaler()

    st.title("Drowsiness Detection Prototype 😴")
    st.write("Enter sensor feature values to get a drowsiness prediction (0 = Alert, 1 = Asleep).")

    if model and scaler:
        st.success("✅ Model and scaler loaded successfully!")

        # Input widgets for your simplified features
        st.subheader("Sensor Feature Input")

        col1, col2, col3 = st.columns(3)

        with col1:
            blink_rate = st.slider(
                "Blink Rate (blinks/min)",
                min_value=0,
                max_value=30,
                value=15,
                help="Normal range: 12-20 blinks/min"
            )

        with col2:
            avg_blink_duration = st.slider(
                "Average Blink Duration (seconds)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Normal range: 0.1-0.4 seconds"
            )

        with col3:
            nodding_frequency = st.slider(
                "Nodding Frequency (nods/min)",
                min_value=0,
                max_value=5,
                value=0,
                help="Any nodding indicates drowsiness"
            )

        # Display current feature values
        st.subheader("Current Feature Values")
        feature_data = {
            "Feature": ["Blink Rate", "Avg Blink Duration", "Nodding Frequency"],
            "Value": [f"{blink_rate} /min", f"{avg_blink_duration:.2f} sec", f"{nodding_frequency} /min"],
            "Status": [
                "⚠️ Low" if blink_rate < 10 else "✅ Normal",
                "⚠️ Long" if avg_blink_duration > 0.4 else "✅ Normal",
                "⚠️ Detected" if nodding_frequency > 0 else "✅ None"
            ]
        }
        st.table(feature_data)

        if st.button("🔍 Predict Drowsiness", type="primary"):
            # Prepare input for the model (must match training input shape)
            input_features = np.array([[blink_rate, avg_blink_duration, nodding_frequency]])

            # Scale the input using the saved scaler
            scaled_input = scaler.transform(input_features)

            # Reshape for LSTM input (samples, timesteps, features)
            # For this prototype, we're using 1 timestep for direct input
            model_input = scaled_input.reshape(1, 1, 3)  # (1 sample, 1 timestep, 3 features)

            # Make prediction
            drowsiness_score = model.predict(model_input, verbose=0)[0][0]  # Get the single score

            # Display results
            st.subheader(f"🎯 Predicted Drowsiness Score: {drowsiness_score:.3f}")

            # Progress bar for visual representation
            # Convert to percentage (0–100 integer)
            progress_value = int(float(drowsiness_score) * 100)

            # Progress bar for visual representation
            st.progress(progress_value)

            # Status classification
            if drowsiness_score < 0.3:
                st.success("🟢 **Status: Alert** 😊")
                st.balloons()
            elif drowsiness_score < 0.7:
                st.warning("🟡 **Status: Drowsy** 😴")
                st.write("⚠️ Consider taking a break or getting some fresh air.")
            else:
                st.error("🔴 **Status: Severely Drowsy (Asleep)** 💤")
                st.write("🚨 **ALERT**: Immediate attention required! Stop driving if applicable.")

            # Status classification
            if drowsiness_score < 0.3:
                st.success("🟢 **Status: Alert** 😊")
                st.balloons()
            elif drowsiness_score < 0.7:
                st.warning("🟡 **Status: Drowsy** 😴")
                st.write("⚠️ Consider taking a break or getting some fresh air.")
            else:
                st.error("🔴 **Status: Severely Drowsy (Asleep)** 💤")
                st.write("🚨 **ALERT**: Immediate attention required! Stop driving if applicable.")

            # Detailed analysis
            with st.expander("📊 Detailed Analysis"):
                st.write("**Contributing Factors:**")
                factors = []

                if blink_rate < 10:
                    factors.append(f"• Low blink rate ({blink_rate}/min) - contributes +0.3 to drowsiness score")
                if avg_blink_duration > 0.4:
                    factors.append(
                        f"• Long blink duration ({avg_blink_duration:.2f}s) - contributes +0.4 to drowsiness score")
                if nodding_frequency > 0:
                    factors.append(
                        f"• Head nodding detected ({nodding_frequency}/min) - contributes +0.3 to drowsiness score")

                if factors:
                    for factor in factors:
                        st.write(factor)
                else:
                    st.write("• All features within normal ranges")

                st.write(f"\n**Model Confidence:** {drowsiness_score:.1%}")

    else:
        st.error("❌ Model not loaded. Please check the console for errors.")
        st.write("**Troubleshooting:**")
        st.write("1. Make sure `drowsiness_detector.h5` is in the same directory as this app")
        st.write("2. Make sure `scaler.pkl` is in the same directory as this app")
        st.write("3. Train a model first using `train_model.py`")

    # Sidebar with information
    st.sidebar.title("ℹ️ About")
    st.sidebar.write("This drowsiness detection system uses:")
    st.sidebar.write("• **Photodiode data** for blink detection")
    st.sidebar.write("• **MPU6050 Y-axis** for head nod detection")
    st.sidebar.write("• **LSTM neural network** for prediction")

    st.sidebar.subheader("🎯 Drowsiness Indicators")
    st.sidebar.write("**Alert (0.0 - 0.3):** Normal behavior")
    st.sidebar.write("**Drowsy (0.3 - 0.7):** Warning signs")
    st.sidebar.write("**Severely Drowsy (0.7 - 1.0):** Critical")

    st.sidebar.subheader("📈 Feature Ranges")
    st.sidebar.write("**Normal Blink Rate:** 12-20 /min")
    st.sidebar.write("**Normal Blink Duration:** 0.1-0.4 sec")
    st.sidebar.write("**Head Nodding:** Should be 0 /min")


if __name__ == "__main__":
    main()