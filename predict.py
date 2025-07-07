
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


def predict_image(model_path, labels_path, image_path):
    # Load the model
    model = load_model(model_path, compile=False)
    
    # Load the labels
    class_names = open(labels_path, "r").readlines()
    
    # Create the array for the image
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

if __name__ == "__main__":
    # File paths
    MODEL_PATH = "model/keras_model.h5"
    LABELS_PATH = "model/labels.txt"
    IMAGE_PATH = "test_images/cat.jpg"  # Replace with your image
    
    # Get prediction
    class_name, confidence = predict_image(MODEL_PATH, LABELS_PATH, IMAGE_PATH)
    
    # Print results
    print("--- Prediction Results ---")
    print(f"Class: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print("--------------------------")
