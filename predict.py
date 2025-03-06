import tensorflow as tf
import numpy as np
from PIL import Image

# Class names for CIFAR-10
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_image(image_path):
    """
    LoadS and preprocesses the input image (need to enter the image path).
    Args:
        image_path: Path to the input image file.
    Returns:
        Preprocessed image as a Numpy array.
    """
    # Loading the image using PIL
    image = Image.open(image_path)
    # Converting the image to RGB (remove alpha channel if present to prevent errors on PNGs)
    image = image.convert('RGB')
    # Resizing the image to the CIFAR-10 input size (32x32)
    image = image.resize((32, 32))
    # Converting the image to a Numpy array
    image = np.array(image)
    # Normalizing pixel values to [0, 1]
    image = image / 255.0

    return image

def load_model():
    """
    Loads the pre-trained and saved model.
    Returns:
        Loaded model.
    """
    model = tf.keras.models.load_model('cifar10_model.h5')
    return model

def predict(image):
    """
    Makes a prediction for an input image using the trained/loaded model.
    Args:
        image: Input image as a numpy array.
    Returns:
        Predicted class label and confidence score.
    """
    # Add batch dimension (model expects input shape [batch_size, 32, 32, 3])
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    predictions = model.predict(image)

    # Get the predicted class label and confidence score
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions)
    return predicted_class, confidence_score

# Execute the prediction script by changing the image path

model = load_model()
image_path = 'example_image.png' 
image = load_image(image_path)  

predicted_class, confidence_score = predict(image)

#For displaying the results
print(f'Predicted class: {CLASS_NAMES[predicted_class]}')
print(f'Confidence score: {confidence_score:.4f}')