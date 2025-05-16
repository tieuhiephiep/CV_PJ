import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_emotion_model, save_model
from utils import load_dataset, plot_training_history

def train_model(data_dir, model_save_path, batch_size=32, epochs=50):
    """
    Train the emotion recognition model
    """
    # Load and preprocess dataset
    print("Loading training dataset...")
    X_train, y_train = load_dataset(data_dir, mode='train')
    
    print("Loading test dataset...")
    X_test, y_test = load_dataset(data_dir, mode='test')
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Reshape data for CNN input
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # Create model
    print("Creating model...")
    model = create_emotion_model()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    save_model(model, model_save_path)
    
    return model, history

if __name__ == "__main__":
    # Define paths
    DATA_DIR = "data"  # Path to directory containing train and test folders
    MODEL_SAVE_PATH = "models/emotion_model.h5"
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    
    # Train model
    model, history = train_model(DATA_DIR, MODEL_SAVE_PATH) 