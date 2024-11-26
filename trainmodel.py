import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
DATA_PATH = 'MP_Data'
actions = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
max_sequence_length = 30

def load_data():
    """
    Load sign language data with detailed preprocessing
    """
    sequences, labels = [], []
    
    for action_idx, action in enumerate(actions):
        action_path = os.path.join(DATA_PATH, action)
        
        for sequence in range(30):  # Assuming 30 sequences per action
            sequence_path = os.path.join(action_path, str(sequence))
            
            # Read all frames for this sequence
            window = []
            for frame_num in range(max_sequence_length):
                file_path = os.path.join(sequence_path, f"{frame_num}.npy")
                
                try:
                    frame = np.load(file_path)
                    window.append(frame)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    break
            
            # Ensure consistent sequence length
            if len(window) == max_sequence_length:
                sequences.append(window)
                labels.append(action_idx)
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = to_categorical(labels)
    
    return X, y

def create_advanced_model(input_shape, num_classes):
    """
    Create a more sophisticated LSTM model
    """
    model = Sequential([
        # First LSTM layer with batch normalization
        LSTM(128, return_sequences=True, 
             input_shape=input_shape,
             dropout=0.2, 
             recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(256, return_sequences=True, 
             dropout=0.3, 
             recurrent_dropout=0.3),
        BatchNormalization(),
        
        # Third LSTM layer
        LSTM(128, return_sequences=False),
        
        # Dense layers with dropout and batch normalization
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with custom learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_model():
    """
    Comprehensive model training and evaluation
    """
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Create model
    num_classes = y.shape[1]
    model = create_advanced_model(
        input_shape=(X.shape[1], X.shape[2]), 
        num_classes=num_classes
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=10, 
        min_lr=0.00001
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model.h5', 
        save_best_only=True, 
        monitor='val_accuracy'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train, 
        epochs=300, 
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        batch_size=32
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    
    # Predictions for detailed analysis
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=actions))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Training History Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Save final model
    model.save('final_model.h5')
    
    return model, history

# Run the training
if __name__ == "__main__":
    model, history = train_and_evaluate_model()