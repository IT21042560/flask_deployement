# import joblib
# from matplotlib.pyplot import imread
# from matplotlib.pyplot import imshow
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
# import numpy as np

# def predict_disease(image_path):
#     # Define the file path to your .pkl file
#     file_path = 'model_efficientOrginal.pkl'

#     # Load the model
#     model = joblib.load(file_path)

#     img_path = image_path

#     img = image.load_img(img_path, target_size=(224, 224))  # Load image with color channels
#     x = image.img_to_array(img)  # Convert to numpy array
#     x = np.expand_dims(x, axis=0)  # Add batch dimension
#     x = preprocess_input(x)  # Preprocess the input

#     print('Input image shape:', x.shape)

#     preds=model.predict(x)
#     return preds

import joblib
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2

def color_change(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the binary image (if needed)
    inverted_image = cv2.bitwise_not(otsu_thresh)

    # Apply morphological operations to enhance the patterns
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)
    # return processed_image
    # imshow(processed_image, cmap='gray')
    # Save the processed image
    print(image_path)
    cv2.imwrite(image_path, cv2.resize(processed_image, (224, 224)))

def predict_disease(image_path):
    # Define the file path to your .pkl file
    file_path = 'model_efficientOrginal.pkl'

    # Load the model
    model = joblib.load(file_path)
    color_change(image_path)
    img_path = image_path

    img = image.load_img(img_path, target_size=(224, 224))  # Load image with color channels
    x = image.img_to_array(img)  # Convert to numpy array
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)  # Preprocess the input

    print('Input image shape:', x.shape)

    preds=model.predict(x)
    return preds