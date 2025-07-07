import pytest
from predict import predict_image

def test_prediction():
    # Paths to test files
    MODEL_PATH = "model/keras_model.h5"
    LABELS_PATH = "model/labels.txt"
    TEST_IMAGE = "test_images/cat.jpg"  # Replace with your test image
    
    # Ensure files exist
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(LABELS_PATH)
    assert os.path.exists(TEST_IMAGE)
    
    # Run prediction
    class_name, confidence = predict_image(MODEL_PATH, LABELS_PATH, TEST_IMAGE)
    
    # Validate output
    assert isinstance(class_name, str)
    assert 0 <= confidence <= 1
