#MPU6050 Drowsiness Prediction Model

This branch contains a LogisticRegression model built for MPU6050 sensor data. The model has been trained and tested using mock data, achieving 100% cross-validated accuracy.

⚠️ Note: This perfect score is expected due to the synthetic nature of the mock dataset. Real-world sensor data will include noise and variability, so similar performance should not be expected.

Preparing for Real Data

The mpu6050_builder.py script is fully implemented and ready to run on actual MPU6050 sensor data once it becomes available.

Akhona’s preprocessed data has been integrated into the IRIS_ML repository and stored in the appropriate folders to ensure seamless access for training and testing.

The data pipeline is organized so that when real sensor data arrives, it can be processed immediately, allowing the model to be evaluated and tuned accordingly.

Branch Purpose

This branch serves as a proof-of-concept for the MPU6050 drowsiness prediction model. It includes all necessary scripts, data organization, and documentation for future testing with real sensor inputs.

# IRIS Year-End Project

This project detects drowsiness using sensor data acquired via Bluetooth, cleans the data, analyzes it with machine learning models, and stores results in CSV and SQLite formats.

## Project Setup & Configuration

1. **Clone the repository and navigate to the project folder.**
2. **Create a Python virtual environment:**
	```powershell
	python -m venv .venv
	.\.venv\Scripts\activate
	```
3. **Install required packages:**
	```powershell
	pip install -r requirements.txt
	```
4. **Run the Streamlit GUI:**
	```powershell
	streamlit run <your_app_file.py>
	```

## Building & Running the Project

- Place your sensor data CSV files in the appropriate folders.
- Follow the workflow described in the wiki for data cleansing, feature extraction, modeling, and deployment.
- Use the provided code snippets and scripts to process data and train models.
- Visualize results using the Streamlit GUI.

For more details, see `wiki_structured.html`.
