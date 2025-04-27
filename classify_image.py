import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('intel_image_classification_model.h5')

# Define class names (must match your training class order)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizing like training
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    
    # Print results
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    # Example usage:
    img_path = input("Enter path to the image you want to classify: ")
    predict_image(img_path)