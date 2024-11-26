import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('model.h5')

# Define the actions (letters)
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                    'Y', 'Z'])

def preprocess_image(image_path, input_shape=(30, 63)):
    """
    Preprocess image for LSTM input
    
    Args:
    - image_path (str): Path to the input image
    - input_shape (tuple): Expected input shape (sequence_length, feature_dim)
    
    Returns:
    - numpy array: Preprocessed image ready for model prediction
    """
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize image to match expected input shape
    img = img.resize((input_shape[1], input_shape[0]))
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Reshape to match model's expected input
    # If model expects (batch_size, sequence_length, feature_dim)
    img_array = img_array.reshape((1, *input_shape))
    
    return img_array

# Preprocess the input image
try:
    image_path = 'image.png'
    preprocessed_image = preprocess_image(image_path)
    
    # Print the shape of the preprocessed image for debugging
    print("Shape of the preprocessed image:", preprocessed_image.shape)
    
    # Predict the class
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    
    # Output the prediction
    print(f"Predicted Class: {actions[predicted_class]} ({predicted_class})")

except Exception as e:
    print("An error occurred:", e)
    print("\nTroubleshooting tips:")
    print("1. Verify the model's expected input shape")
    print("2. Check if the image preprocessing matches the training data preprocessing")
    print("3. Confirm the model architecture supports the current input shape")