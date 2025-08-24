import numpy as np
import tensorflow as tf
from keras import layers, models



num_windows = 10           # Simulate 10 windows
window_size = 30 * 50      # 30 seconds * 50 Hz = 1500 samples per window

# Simulate head movement data: shape (10, 1500, 1)
X = np.random.normal(loc=0, scale=1, size=(num_windows, window_size, 1))

# Simulate drowsiness scores: shape (10, 1)
y = np.random.uniform(low=0.1, high=0.9, size=(num_windows, 1))

def build_cnn_lstm(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(1, activation='sigmoid')  # Output between 0 and 1
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build the model
input_shape = (window_size, 1)
model = build_cnn_lstm(input_shape)
model.summary()

# Train the model
history = model.fit(X, y, epochs=10, batch_size=22, verbose=1)

# Evaluate on the training data
loss, mae = model.evaluate(X, y)
print(f"Loss: {loss:.4f}, MAE: {mae:.4f}")

# Predict drowsiness score for a new window
sample_window = np.random.normal(loc=0, scale=1, size=(1, window_size, 1))
pred_score = model.predict(sample_window)[0][0]
print(f"Predicted drowsiness score: {pred_score:.2f}")



