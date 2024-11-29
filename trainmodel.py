from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os
from collections import Counter
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['A','B','C'])

no_sequences = 30 

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []
max_sequence_length = 30 
for action in actions:
    action_sequences = 0
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(max_sequence_length):  # Attempt to read max_sequence_length frames
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(file_path):  # Check if file exists
                res = np.load(file_path)
                window.append(res)
            else:
                break
        if window:
            if len(window) < max_sequence_length:
                padding = [np.zeros_like(window[0])] * (max_sequence_length - len(window))  # Create zero padding
                window.extend(padding)
            sequences.append(window)
            labels.append(label_map[action])
            action_sequences += 1
    print(f"Action {action} has {action_sequences} valid sequences.")
print("Label distribution in training data:", Counter(labels))



X = np.array(sequences)  # Shape: (num_samples, max_sequence_length, feature_dim)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
num_classes = 3  # Get the number of classes based on one-hot encoding

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', 
         input_shape=(max_sequence_length, X.shape[2])),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax') 
])


model.compile(optimizer='Adam', 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
print("Testing model predictions:")
for i in range(5):
    prediction = model.predict(np.expand_dims(X_test[i], axis=0))
    predicted_class = np.argmax(prediction)
    print(f"Actual: {actions[np.argmax(y_test[i])]}, Predicted: {actions[predicted_class]}")
