# train_model.py
# Model Training Module - Section 3 from Wiki

import matplotlib.pyplot as plt
from dataloader import DataLoader
from model_builder import build_model, compile_model, prepare_data_split


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    3.1 Train the Model
    Train the model with specified parameters
    """
    print("Starting model training...")

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,  # Initial number of training epochs
        batch_size=batch_size,  # Batch size for training
        validation_data=(X_val, y_val),  # Data for performance evaluation during training
        verbose=1
    )

    print("Training completed!")
    return history


def plot_training_history(history):
    """
    3.2 Monitor Performance
    Plot training history to visualize performance
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()

    # Plot training & validation MAE
    ax2.plot(history.history['mean_absolute_error'], label='Training MAE')
    ax2.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    ax2.set_title('Model Mean Absolute Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print final performance
    final_val_mae = history.history['val_mean_absolute_error'][-1]
    print(f"\nFinal Validation MAE: {final_val_mae:.4f}")
    print("Monitor this value - lower is better!")


def save_trained_model(model, filepath='drowsiness_detector.h5'):
    """
    4.1 Save the Tuned Model
    Save the trained model for deployment
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")


def main():
    """
    Main training pipeline
    """
    print("=== Drowsiness Detection Model Training ===")

    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    loader = DataLoader("dataset.csv")  # Replace with your CSV file

    df = loader.load_data()
    df_filtered = loader.apply_filters()
    features, labels = loader.extract_features()
    scaled_features = loader.normalize_features()
    X, y = loader.prepare_for_lstm()

    # 2. Split data
    print("\n2. Splitting data...")
    X_train, X_val, y_train, y_val = prepare_data_split(X, y)

    # 3. Build and compile model
    print("\n3. Building model...")
    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=50,  # Adjust as needed for tuning
        dropout_rate=0.2,  # Adjust as needed for tuning
        learning_rate=0.001  # Adjust as needed for tuning
    )
    model = compile_model(model)

    # 4. Train model
    print("\n4. Training model...")
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=50,  # Adjust as needed
        batch_size=32  # Adjust as needed
    )

    # 5. Plot results
    print("\n5. Plotting training results...")
    plot_training_history(history)

    # 6. Save model
    print("\n6. Saving trained model...")
    save_trained_model(model)

    print("\n=== Training Complete! ===")
    print("Your model is ready for deployment!")


if __name__ == "__main__":
    main()