from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def build_model(input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
    """
    2.2 Build the Model Architecture
    Build LSTM model with specified parameters
    """
    print("Building model architecture...")

    # Model definition
    model = Sequential([
        # LSTM layer: 50 units, 'relu' activation, input shape matches data (timesteps, features)
        LSTM(units=lstm_units, activation='relu', input_shape=input_shape),
        # Dropout layer: 20% of neurons are randomly deactivated during training to prevent overfitting
        Dropout(dropout_rate),
        # Output layer: 1 unit for regression, 'sigmoid' activation for 0-1 score
        Dense(units=1, activation='sigmoid')
    ])

    return model


def compile_model(model, learning_rate=0.001):
    """
    2.3 Compile the Model
    Configure the learning process
    """
    print("Compiling model...")

    # Model compilation
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),  # Adam optimizer with initial learning rate
        loss='mean_squared_error',  # MSE for regression
        metrics=['mean_absolute_error']  # MAE for easy interpretation
    )

    # Display model summary
    model.summary()
    return model


def prepare_data_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and validation sets
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    return X_train, X_val, y_train, y_val


# Example usage
if __name__ == "__main__":
    # This would be run after data preparation
    # Assuming X and y are available from data_loader.py

    from dataloader import DataLoader

    # Load data (replace with your file)
    loader = DataLoader("dataset.csv")
    df = loader.load_data()
    df_filtered = loader.apply_filters()
    features, labels = loader.extract_features()
    scaled_features = loader.normalize_features()
    X, y = loader.prepare_for_lstm()

    # Split data
    X_train, X_val, y_train, y_val = prepare_data_split(X, y)

    # Build and compile model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model = compile_model(model)

    print("Model ready for training!")