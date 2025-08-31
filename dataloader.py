import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import MinMaxScaler
import joblib


class DataLoader:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.df = None
        self.scaler = MinMaxScaler()
        self.features_raw = None
        self.y_labels = None
        self.scaled_features = None

    def load_data(self):
        """1.1 Data Loading and Initial Structure"""
        print("Loading dataset...")

        # Load the dataset - Replace 'your_drowsiness_dataset.csv' with actual file path
        self.df = pd.read_csv(self.csv_file_path)

        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")

        # Check if timestamp column exists
        if 'timestamp' not in self.df.columns:
            print("Creating timestamps...")
            self.df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(self.df), freq='10ms')
        else:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Set timestamp as index
        self.df = self.df.set_index('timestamp')
        return self.df

    def apply_filters(self, fs_pd=100, fs_mpu=50):
        """1.2 Signal Filtering"""
        print("Applying filters...")

        # Detect photodiode and MPU columns
        photodiode_col = self._detect_photodiode_column()
        mpu_y_col = self._detect_mpu_column()

        # --- Photodiode Filtering ---
        # High-pass filter (cutoff at 0.5 Hz to remove slow ambient changes)
        b_high, a_high = butter(N=2, Wn=0.5, fs=fs_pd, btype='high')
        self.df['photodiode_filtered_high'] = filtfilt(b_high, a_high, self.df[photodiode_col])

        # Low-pass filter (cutoff at 10 Hz to remove high-frequency noise)
        b_low, a_low = butter(N=2, Wn=10, fs=fs_pd, btype='low')
        self.df['photodiode_final_filtered'] = filtfilt(b_low, a_low, self.df['photodiode_filtered_high'])

        # --- MPU6050 Filtering ---
        # Low-pass filter (cutoff at 5 Hz to smooth jitters)
        b_mpu, a_mpu = butter(N=2, Wn=5, fs=fs_mpu, btype='low')
        self.df['mpu_y_filtered'] = filtfilt(b_mpu, a_mpu, self.df[mpu_y_col])

        print("Filtering completed!")
        return self.df

    def extract_features(self, window_size_seconds=60, fs_pd=100):
        """1.4 Feature Extraction & Drowsiness Score Labeling"""
        print("Extracting features...")

        window_size_samples = int(window_size_seconds * fs_pd / 100)
        features_list = []
        labels_list = []

        for i in range(0, len(self.df) - window_size_samples, window_size_samples // 2):
            window_df = self.df.iloc[i:i + window_size_samples]

            if len(window_df) < window_size_samples:
                continue

            # Extract features for this window
            blink_rate = self._detect_blink_rate(window_df['photodiode_final_filtered'], window_size_seconds)
            avg_blink_duration = self._detect_avg_blink_duration(window_df['photodiode_final_filtered'], fs_pd)
            nodding_frequency = self._detect_nodding_frequency(window_df['mpu_y_filtered'], window_size_seconds)

            features_list.append([blink_rate, avg_blink_duration, nodding_frequency])
            labels_list.append(self.calculate_drowsiness_score(blink_rate, avg_blink_duration, nodding_frequency))

        self.features_raw = np.array(features_list)
        self.y_labels = np.array(labels_list)

        print(f"Extracted {len(features_list)} feature windows")
        return self.features_raw, self.y_labels

    def normalize_features(self):
        """1.3 Normalization (Scaling)"""
        if self.features_raw is None:
            raise ValueError("Extract features first!")

        # Fit the scaler to training data and transform it
        self.scaled_features = self.scaler.fit_transform(self.features_raw)

        # Save scaler for deployment
        joblib.dump(self.scaler, 'scaler.pkl')
        print("Features normalized and scaler saved!")
        return self.scaled_features

    def prepare_for_lstm(self, time_steps_per_window=1):
        """1.5 Prepare for LSTM Input"""
        if self.scaled_features is None:
            raise ValueError("Normalize features first!")

        num_features = self.scaled_features.shape[1]
        X = self.scaled_features.reshape(-1, time_steps_per_window, num_features)
        y = self.y_labels

        print(f"LSTM input shape: {X.shape}")
        return X, y

    def calculate_drowsiness_score(self, blink_rate, avg_blink_duration, nodding_frequency):
        """Calculate drowsiness score based on wiki rules"""
        score = 0.0

        # Lower blink rate indicates drowsiness
        if blink_rate < 10:
            score += 0.3
        # Longer blinks indicate drowsiness
        if avg_blink_duration > 0.4:
            score += 0.4
        # Any nodding indicates drowsiness
        if nodding_frequency > 0:
            score += 0.3

        return max(0.0, min(1.0, score))

    # Helper methods
    def _detect_photodiode_column(self):
        """Auto-detect photodiode column"""
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['photodiode', 'photo', 'blink']):
                print(f"Using photodiode column: {col}")
                return col

        # Default to first numeric column
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"Using first numeric column as photodiode: {numeric_cols[0]}")
            return numeric_cols[0]

        raise ValueError("No suitable photodiode column found!")

    def _detect_mpu_column(self):
        """Auto-detect MPU Y-axis column"""
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['mpu', 'accel', 'gyro']) and 'y' in col.lower():
                print(f"Using MPU column: {col}")
                return col

        # Default to second numeric column or create dummy
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            print(f"Using second numeric column as MPU: {numeric_cols[1]}")
            return numeric_cols[1]
        else:
            print("Creating dummy MPU data")
            self.df['mpu_y_raw'] = np.random.normal(0, 10, len(self.df))
            return 'mpu_y_raw'

    def _detect_blink_rate(self, signal, window_seconds):
        """Detect blinks in photodiode signal"""
        # Find peaks in inverted signal (blinks are dips)
        inverted = -signal.values
        threshold = np.mean(inverted) + 2 * np.std(inverted)
        peaks, _ = find_peaks(inverted, height=threshold, distance=10)
        return (len(peaks) / window_seconds) * 60  # per minute

    def _detect_avg_blink_duration(self, signal, fs):
        """Calculate average blink duration"""
        inverted = -signal.values
        threshold = np.mean(inverted) + 2 * np.std(inverted)
        peaks, properties = find_peaks(inverted, height=threshold, distance=10, width=(2, 50))

        if len(peaks) > 0 and 'widths' in properties:
            return np.mean(properties['widths']) / fs
        return 0.2  # default

    def _detect_nodding_frequency(self, signal, window_seconds):
        """Detect nodding in MPU signal"""
        abs_signal = np.abs(signal.values)
        threshold = np.mean(abs_signal) + 2 * np.std(abs_signal)
        peaks, _ = find_peaks(abs_signal, height=threshold, distance=25)
        return (len(peaks) / window_seconds) * 60  # per minute


# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    loader = DataLoader("your_drowsiness_dataset.csv")

    # Load and process data
    df = loader.load_data()
    df_filtered = loader.apply_filters()
    features, labels = loader.extract_features()
    scaled_features = loader.normalize_features()
    X, y = loader.prepare_for_lstm()

    print("Data preparation completed!")
    print(f"Final shapes - X: {X.shape}, y: {y.shape}")