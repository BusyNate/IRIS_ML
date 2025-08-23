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
