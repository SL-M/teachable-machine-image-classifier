
from predict import predict_image

def test_prediction():
    result = predict_image(
        "model/keras_model.h5",
        "model/labels.txt",
        "test_images/cat.jpg"
    )
    assert isinstance(result[0], str)  # Class name is string
    assert 0 <= result[1] <= 1        # Confidence is between 0-1