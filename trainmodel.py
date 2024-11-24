from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os
from collections import Counter

# Path where the data is stored
DATA_PATH = os.path.join('MP_Data')

# Actions and number of sequences
actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
no_sequences = 30  # Number of sequences for each action

# Label map to convert action labels to numeric values
label_map = {label: num for num, label in enumerate(actions)}

# Store sequences and labels
sequences, labels = [], []

# Maximum sequence length (based on the longest sequence you want to handle)
max_sequence_length = 30  # Can be adjusted depending on your data

# Loop through all actions (A to Z)
for action in actions:
    action_sequences = 0  # Keep track of valid sequences for this action
    
    # Loop through sequence folders (0 to 14)
    for sequence in range(no_sequences):
        window = []
        
        # Loop through frames (0 to n.npy) in each sequence folder
        for frame_num in range(max_sequence_length):  # Attempt to read max_sequence_length frames
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(file_path):  # Check if file exists
                res = np.load(file_path)
                window.append(res)
            else:
                break  # Stop if a frame is missing in this sequence
        
        # If the sequence has at least one frame, process it
        if window:
            # Pad sequences that have fewer frames to max_sequence_length
            if len(window) < max_sequence_length:
                padding = [np.zeros_like(window[0])] * (max_sequence_length - len(window))  # Create zero padding
                window.extend(padding)
            sequences.append(window)
            labels.append(label_map[action])
            action_sequences += 1  # Increment count of sequences for this action
    
    # Print the number of valid sequences for each action
    print(f"Action {action} has {action_sequences} valid sequences.")

# Check the label distribution after filtering out incomplete sequences
print("Label distribution in training data:", Counter(labels))

# Convert sequences and labels to numpy arrays
X = np.array(sequences)  # Shape: (num_samples, max_sequence_length, feature_dim)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Model definition
num_classes = 26  # Get the number of classes based on one-hot encoding

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(max_sequence_length, X.shape[2])),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer for classification
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Save the model in Keras format
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')  # Save model in native Keras format

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Testing predictions on the first 5 samples
print("Testing model predictions:")
for i in range(5):  # Testing on first 5 samples
    prediction = model.predict(np.expand_dims(X_test[i], axis=0))
    predicted_class = np.argmax(prediction)
    print(f"Actual: {actions[np.argmax(y_test[i])]}, Predicted: {actions[predicted_class]}")
