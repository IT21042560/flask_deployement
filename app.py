from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_pymongo import PyMongo
from bson import json_util
from datetime import date
import uuid
from ultralytics import YOLO
import os
import cv2
import io
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import data_preprocessing as dp
import joblib
import pandas as pd
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
import leaf_disease
from recommender import recommender  # Import your recommender class
from pest_recommender import pest_recommender  # Import your recommender class
import pickle
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb+srv://Amanda:amanda1234@mernapp.juehqhn.mongodb.net/Ino_Agri_Mobile_App"
mongo = PyMongo(app)

# Load the models
regression_model_path = 'models/regression_model.pkl'
classification_model_path = 'models/classification_model.pkl'
regression_model = joblib.load(regression_model_path)
classification_model = joblib.load(classification_model_path)

harvest_model_path = './Actual_harvest_predictor.pickle'

with open(harvest_model_path, 'rb') as model_file:
    harvest_model = pickle.load(model_file)

# Load and preprocess the dataset (you might want to adjust this based on your use case)
def load_dataset(path):
    return pd.read_csv(path)


def pestPredict(mdl):
    model = YOLO('pest_model.pt')
    pred_value = model.predict(mdl, save=True)
    return pred_value

def translate_text(text):
    translator = GoogleTranslator(source='en', target='si')
    return translator.translate(text)

@app.route("/user/signup", methods=['POST'])
def detect_object():
    if request.method == 'POST':
        users = mongo.db.users 
        data = request.json
       # user_data = dict(request.form)
        uID = str(date.today()) + uuid.uuid4().hex
        uName = data['uName']
        uEmail = data['uEmail']
        uLocation = data['uLocation']
        uContactNo = data['uContactNo']
        uPassword = data['uPassword']
        
        user = {
            "uID": uID,
            "uName": uName,
            "uEmail": uEmail,
            "uLocation": uLocation,
            "uContactNo": uContactNo,
            "uPassword": uPassword
        }
        #insert_users_json = json_util.dumps(user)
        insert_user = users.insert_one(user)
        return jsonify({"message":"User Created", "user_id": str(insert_user.inserted_id)})
    
@app.route("/user/login", methods=['POST'])
def userLogin():
    if request.method == 'POST':
        users = mongo.db.users  # Reference to the MongoDB users collection
        data = request.json  # Get JSON data from the request

        username = data.get('uEmail')
        password = data.get('uPassword')

        # Fetch the user by username and password
        user = users.find_one({"uEmail": username, "uPassword": password})

        if user:
            # If login is successful
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"message": "Invalid username or password"}), 401


@app.route("/user", methods=['GET'])
def welcome():
    users = mongo.db.users
    all_users = list(users.find())
    all_users_json = json_util.dumps(all_users)
    return jsonify(all_users_json)

# Define function for preprocessing images
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Assuming your model requires input size (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize pixel values to [0, 1]
    return img_array      
        
# @app.route("/pest/predict", methods=['POST'])
# def mytest():
#     # Fetch the image
#     file = request.files['image']
#     # Validate the image
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     # Save user entered image
#     save_path = './Pest_Image_Uploads'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     upload_path = os.path.join(save_path, file.filename)
#     file.save(upload_path)

#     # Save user entered image 2
#     save_path_2 = '../frontend/pest_uploaded_images'
#     if not os.path.exists(save_path_2):
#         os.makedirs(save_path_2)

#     upload_path_2 = os.path.join(save_path_2, file.filename)
#     file.save(upload_path_2)

#     # ----Yolov8 object detection-----
#     # Load yolo model
#     model = YOLO("pest_model.pt")
#     # Load input image
#     results = model.predict(upload_path)
#     # Get the result
#     result = results[0]
#     len(result.boxes)
#     box = result.boxes[0]
#     cords = box.xyxy[0].tolist()
#     class_id = result.names[box.cls[0].item()]
#     conf = round(box.conf[0].item(), 2)

#     if class_id == "Thirps":
#         class_id = "Thrips"

#     yolo_prediction = {
#         'object_type': class_id,
#         'probability': conf
#     }

#     # -------Inception V8 computer vision---------
#     # Load your custom Keras model
#     model_path = './pest_classify_model.h5'
#     model = load_model(model_path)
#     # Create the label names
#     class_names = ['Apids', 'Catterpillar', 'Leaf miner', 'Mites', 'Thrips', 'Whiteflies']
#     # Preprocess the image
#     preprocessed_image = preprocess_image(upload_path)
#     # Make prediction
#     prediction = model.predict(preprocessed_image)
#     # Get probabilities
#     predicted_class_index = np.argmax(prediction)
#     predicted_class = class_names[predicted_class_index]
#     probability = round(prediction[0][predicted_class_index] * 100, 2)  # Converting to percentage for consistency with YOLO

#     if predicted_class == "Apids":
#         predicted_class = "Aphids"

#      # Create an instance of your recommender class
#     recommender_instance = pest_recommender()
#     # print(yolo_prediction)
#     # Get recommendations
#     recommendations = recommender_instance.getRecommendations(yolo_prediction)
#     trans = translate_text(recommendations)


#     inception_prediction = {
#         'predicted_class': predicted_class,
#         'probability': probability
#     }

#     response = {
#         'yolo_prediction': yolo_prediction,
#         'inception_prediction': inception_prediction,
#         'image_name': file.filename,
#         'recommendations':recommendations,
#         'trans':trans
#     }

#     if predicted_class == class_id:
#         response['message'] = True
#     else:
#         response['message'] = False
#     # print(response)
#     return jsonify(response)

@app.route("/pest/predict", methods=['POST'])
def mytest():
    # Fetch the image
    file = request.files['image']
    # Validate the image
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save user-entered image
    save_path = './Pest_Image_Uploads'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    upload_path = os.path.join(save_path, file.filename)
    file.save(upload_path)

    # Save user-entered image 2
    save_path_2 = '../frontend/pest_uploaded_images'
    if not os.path.exists(save_path_2):
        os.makedirs(save_path_2)

    upload_path_2 = os.path.join(save_path_2, file.filename)
    file.save(upload_path_2)

    # ----YOLOv8 Object Detection-----
    # Load YOLO model
    yolo_model = YOLO("pest_model.pt")
    # Load input image
    results = yolo_model.predict(upload_path)
    # Get the result
    result = results[0]
    box = result.boxes[0]
    cords = box.xyxy[0].tolist()
    yolo_class_id = result.names[box.cls[0].item()]
    yolo_conf = round(box.conf[0].item() * 100, 2)  # Confidence in percentage

    if yolo_class_id == "Thirps":
        yolo_class_id = "Thrips"

    yolo_prediction = {
        'object_type': yolo_class_id,
        'probability': yolo_conf
    }

    # -------Inception V3 Image Classification---------
    # Load your custom Keras model
    inception_model_path = './pest_classify_model.h5'
    inception_model = load_model(inception_model_path)
    # Create the label names
    class_names = ['Apids', 'Catterpillar', 'Leaf miner', 'Mites', 'Thrips', 'Whiteflies']
    # Preprocess the image
    preprocessed_image = preprocess_image(upload_path)
    # Make prediction
    inception_prediction = inception_model.predict(preprocessed_image)
    # Get probabilities
    inception_class_index = np.argmax(inception_prediction)
    inception_class = class_names[inception_class_index]
    inception_conf = round(inception_prediction[0][inception_class_index] * 100, 2)  # Confidence in percentage

    if inception_class == "Apids":
        inception_class = "Aphids"

    inception_prediction_result = {
        'predicted_class': inception_class,
        'probability': inception_conf
    }

    # ----Ensemble Learning: Weighted Average----
    # Define the weights for YOLOv8 and Inception V3
    yolo_weight = 0.6  # You can adjust the weight based on model performance
    inception_weight = 0.4  # You can adjust the weight based on model performance

    # Normalize probabilities (if necessary) and calculate weighted average
    combined_confidence = (yolo_conf * yolo_weight + inception_conf * inception_weight) / (yolo_weight + inception_weight)

    # Final decision: Choose the class with higher combined confidence
    if yolo_class_id == inception_class:
        final_class = yolo_class_id
    else:
        # Choose the class with the higher probability
        final_class = yolo_class_id if yolo_conf >= inception_conf else inception_class

    # Create an instance of your recommender class
    recommender_instance = pest_recommender()
    # Get recommendations
    recommendations = recommender_instance.getRecommendations(yolo_prediction)
    trans = translate_text(recommendations)

    # Construct the final response
    response = {
        'yolo_prediction': yolo_prediction,
        'inception_prediction': inception_prediction_result,
        'final_class': final_class,
        'combined_confidence': round(combined_confidence, 2),
        'image_name': file.filename,
        'recommendations': recommendations,
        'trans': trans
    }
    print(response)

    return jsonify(response)


@app.route("/pest/try", methods=['POST'])
def mytest2():
    print(request.files.keys())  # Print keys to see what keys are present
    if 'image' in request.files:
        image = request.files['image']
        print(image) 
        return "Done"
    else:
        return "No image file found in the request"
    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    # Change this path to the folder where you want to save the images
    save_path = './Pest_Image_Uploads'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file.save(os.path.join(save_path, file.filename))
    return 'File uploaded successfully'

@app.route("/pest/chemical", methods=['POST'])
def add_chemicals():
    if request.method == 'POST':
        pest_chemicals = mongo.db.chemicals
        data = request.json

        chemical_name = data['chemical_name']
        print(chemical_name)
        pest_name = [pest_name.strip() for pest_name in data['pest_name']]
        print(pest_name)
        mrl_level = data['mrl_level']
        phi_days = data['phi_days']
        chemical_image = data['chemical_image']

        chemical = {
            "chemical_name": chemical_name,
            "pest_name":pest_name,
            "mrl_level":mrl_level,
            "phi_days":phi_days,
            "chemical_image":chemical_image
        }

        insert_pest_detection = pest_chemicals.insert_one(chemical)
        return jsonify({"Message":"Chemical Added"})
    

@app.route("/upload_chemical_img", methods=['POST'])
def uploadChemicalImage():
    if 'image' not in request.files:
        return jsonify({"message":"No image part in the request"}),400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"message":"No selected File"}),400
    
    save_path = './chemical_images'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    file_path = os.path.join(save_path,file.filename)
    file.save(file_path)

    return jsonify({"message":"Filed Saved"})

@app.route("/pest/chemical/<string:pest_name>", methods=['GET'])
def get_chemicals_by_pest_name(pest_name):
    pest_chemicals = mongo.db.chemicals
    trimmed_pest_name = pest_name.strip()
    print(trimmed_pest_name)
    print(pest_chemicals)
    chemicals = list(pest_chemicals.find({"pest_name":trimmed_pest_name}))
    if chemicals:
        return json_util.dumps(chemicals)
    else:
        return jsonify({"message":"No chemical Found"}),404
    

@app.route('/chemical_images/<filename>', methods=['GET'])
def serve_chemical_image(filename):
    try:
        # Check if the directory exists
        if not os.path.exists('chemical_images'):
            return jsonify({"error": "Directory not found"}), 404
        
        # Check if the file exists in the directory
        print(filename)
        file_path = os.path.join('chemical_images', filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        # Serve the file
        return send_from_directory('chemical_images', filename)
    except Exception as e:
        # Print the full stack trace for better debugging
        print(f"Error serving file '{filename}': {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500



def preprocess_data(df, investment_value, price_per_kg, additional_cost=0, area=None, acres=0):
    """
    Preprocess the input data, calculate total expenses, yield, and profitability.
    :param df: pd.DataFrame, dataset
    :param investment_value: float, amount invested
    :param price_per_kg: float, price of Gherkins per kg
    :param additional_cost: float, optional additional cost
    :param area: str, area filter (optional)
    :param acres: int, number of acres filter (optional)
    :return: pd.DataFrame, preprocessed dataset
    """
    df.drop('Total', axis=1, inplace=True)
    
    # Remove unwanted spaces from column names
    df = df.rename(columns=lambda x: x.strip())

    # Calculate total expenses
    df['Total_Expenses'] = df[['Land_Planting', 'Strate_Fertilizer', 'Liquid_Fertilizer', 'Fungicide', 'Insecticide', 'Others']].sum(axis=1)

    # Add investment and calculate yield, profit
    df['Investment'] = investment_value - additional_cost
    df['Estimated_Yield'] = (df['Investment'] / df['Total_Expenses']) * df['KG']
    df['Revenue'] = df['Estimated_Yield'] * price_per_kg
    df['Profit'] = df['Revenue'] - df['Investment']
    df['Is_Profitable'] = df['Profit'] > 0

    # Filter by area and acres if specified
    filtered_df = df.copy()
    if area is not None:
        filtered_df = filtered_df[filtered_df['Area'] == area]
    if acres > 0:
        filtered_df = filtered_df[filtered_df['Acre'] == acres]
        
    # If dataframe is empty, create new data frame with the specified area and calculate values based on acres
    if filtered_df.empty:
        filter_area = area
        filter_acres = 1
        
        # Filter by area and one acre
        refilter = df[(df['Area'] == filter_area) & (df['Acre'] == filter_acres)]
        
        if not refilter.empty:
            # Multiply all relevant columns by the given acres except investment value
            refilter.loc[:, 'Total_Expenses'] *= acres
            refilter.loc[:, 'Acre'] *= acres
            refilter.loc[:, 'KG'] *= acres
            refilter.loc[:, 'Estimated_Yield'] *= acres
            refilter.loc[:, 'Revenue'] *= acres
            refilter.loc[:, 'Profit'] *= acres
            refilter.loc[:, 'Land_Planting'] *= acres
            refilter.loc[:, 'Strate_Fertilizer'] *= acres
            refilter.loc[:, 'Liquid_Fertilizer'] *= acres
            refilter.loc[:, 'Fungicide'] *= acres
            refilter.loc[:, 'Insecticide'] *= acres
            refilter.loc[:, 'Others'] *= acres
            refilter.loc[:, 'Is_Profitable'] = refilter['Profit'] > 0
            
            filtered_df = refilter

    return filtered_df


# Update predict function
@app.route('/predict/cost', methods=['POST'])
def predict_profitability():
    """
    Predict the profitability of Gherkin farming based on the input parameters.
    :param investment_value: float, amount invested
    :param price_per_kg: float, price of Gherkins per kg
    :param initial_cost: float, optional initial cost
    :param area: str, area filter (optional)
    :param acres: int, number of acres filter (optional)
    :param regression_model_path: str, path to the trained regression model
    :param classification_model_path: str, path to the trained classification model
    :param expenses_model_path: str, path to the trained expenses regression model
    :return: dict, predicted profitability
    """

    data = request.json
    investment_value = data.get('investment_value')
    price_per_kg = data.get('price_per_kg')
    #initial_cost = data.get('initial_cost')
    additional_cost = data.get('additional_cost')
    area = data.get('area')
    acres = data.get('acres')

    if None in (investment_value, price_per_kg, additional_cost, area, acres):
        return jsonify({'error': 'Missing data'}), 400
    
    # Preprocess the input data
    dataset_path = 'data/Raw/Dataset.csv'
    df = load_dataset(dataset_path)
    filtered_df = load_dataset(dataset_path)
    processed_df = preprocess_data(filtered_df, investment_value, price_per_kg, additional_cost, area, acres)

    # Strip column names of any leading or trailing spaces
    df.columns = df.columns.str.strip()
    processed_df.columns = processed_df.columns.str.strip()

    # Debug prints to check the DataFrame columns
    print("Initial columns in df:", df.columns)
    print("Initial columns in processed_df:", processed_df.columns)

    # Ensure 'Area' column exists before dropping missing values
    if 'Area' not in df.columns or 'Area' not in processed_df.columns:
        raise KeyError("The 'Area' column is missing from the DataFrame.")

    # Encode the 'Area' column
    df.dropna(subset=['Area'], inplace=True)
    processed_df.dropna(subset=['Area'], inplace=True)
    df['Area'] = df['Area'].str.strip().str.lower()
    processed_df['Area'] = processed_df['Area'].str.strip().str.lower()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[['Area']])
    encoded_area = encoder.transform(processed_df[['Area']])
    encoded_area_df = pd.DataFrame(encoded_area, columns=encoder.get_feature_names_out(['Area']))
    df_encoded = pd.concat([processed_df.reset_index(drop=True), encoded_area_df], axis=1)
    df_encoded.drop(columns=['Area'], inplace=True)

    # Debug prints to check the DataFrame columns after encoding
    print("Columns in df_encoded after encoding 'Area':", df_encoded.columns)

    # Predict profitability and expenses
    X = df_encoded.drop(columns=['Profit', 'Is_Profitable', 'Total_Expenses'])
    predicted_profit = float(regression_model.predict(X)[0])
    predicted_profitability = bool(classification_model.predict(X)[0]) 
    print(predicted_profitability)
    # Calculate adjusted profitability
    adjusted_profitability = predicted_profit - additional_cost
    is_profitable_adjusted = adjusted_profitability > investment_value

    # Return the predicted profitability
    response =  {
        'Predicted_Profit': predicted_profit,
        'Is_Profitable': predicted_profitability,
        'Adjusted_Profitability': adjusted_profitability,
        'Is_Profitable_Adjusted': is_profitable_adjusted,
        'Total_Cost':investment_value+additional_cost,
    }
    print(response)
    return jsonify(response)


@app.route("/predict/leaf", methods=['POST'])
def mydisease():
    # Fetch the image
    file = request.files['image']
    
    # Validate the image
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save user-entered image
    save_path = './Leaf_Image_Uploads'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    upload_path = os.path.join(save_path, file.filename)
    file.save(upload_path)

    # ----YOLOv8 object detection-----
    # Load YOLO model
    model = YOLO("leaf_model.pt")
    
    # Load input image and perform prediction
    results = model.predict(upload_path)
    
    # Get the first result
    result = results[0]
    
    # Process the first bounding box and class name
    if len(result.boxes) > 0:
        box = result.boxes[0]
        cords = box.xyxy[0].tolist()
        class_id = result.names[box.cls[0].item()]
        conf = round(box.conf[0].item(), 2)
        
        # YOLO prediction dictionary
        yolo_prediction = {
            'object_type': class_id,
            'probability': conf
        }
         # Predict disease using the PyTorch model
        disease_prediction_array = leaf_disease.predict_disease(upload_path)
        print(disease_prediction_array)

        # Disease names
        disease_names = ['Fresh leaf', 'downy mildew', 'gummy blight']
        
        # Extract the predicted disease names based on the array values
        predicted_diseases = [
            disease_names[i] for i in range(len(disease_prediction_array[0])) if disease_prediction_array[0][i] == 1
        ]
        
        # Prepare the response
        response = {
            'yolo_prediction': yolo_prediction,
            'image_name': file.filename,
            'disease_prediction': predicted_diseases  # Return the disease names instead of the array
        }

        print(response)

        return jsonify(response)
    else:
        
        print("Error")
        return jsonify({"error": "Internal Server Error"}), 404

@app.route("/predict/leaf_disease", methods=['POST'])
def predict_leaf_disease():
    # Fetch the image and additional data
    file = request.files.get('image')
    disease_name = request.form.get('disease_name')
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))

    if not file or not disease_name or temperature is None or humidity is None:
        return jsonify({'error': 'Missing required data'}), 400

    # Save the uploaded image
    save_path = './Leaf_Image_Uploads'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    upload_path = os.path.join(save_path, file.filename)
    file.save(upload_path)

    # Create an instance of your recommender class
    recommender_instance = recommender()

    # Get recommendations
    recommendations = recommender_instance.getRecommendations(disease_name, temperature, humidity)

    trans = translate_text(recommendations)

    # Get severity
    severity = recommender_instance.getSeverity(disease_name, upload_path)

    # Return the results
    response = {
        'recommendations': recommendations,
        'severity': severity,
        'trans':trans
    }

    return jsonify(response)


@app.route("/pest/ai", methods=['POST'])
def pest_ai_recommender():
    # Fetch the image and additional data
    data = request.json
    pest_name = data.get('pest_name')
    # Create an instance of your recommender class
    recommender_instance = pest_recommender()

    # Get recommendations
    recommendations = recommender_instance.getRecommendations(pest_name)
    trans = translate_text(recommendations)


    # Return the results
    response = {
        'recommendations': recommendations,
        'trans':trans
    }

    return jsonify(response)

@app.route('/submitData', methods=['POST'])
def submit_form():
    try:
        # Get form data from request
        data = request.json
        print("Received Data:", data)
        harvest = mongo.db.harvest
        
        # Insert data into MongoDB
        harvest.insert_one(data)
        
        return jsonify({"message": "Data submitted successfully"}), 201
    except Exception as e:
        print(e)
        return jsonify({"message": "Failed to submit data"}), 500



@app.route('/harvest/predict', methods=['POST'])
def predict_harvest():
    try:
        data = request.json
        print(data)
        # Validate the required fields in the input data
        required_fields = ['pH', 'Acerage', 'Ca', 'Mg', 'K', 'N', 'P', 'Zn', 'Urea', 'TSP', 'MOP', 'CaNO3', 'Rainfall', 'Temperature', 'Expected Harvest']
        if not all(field in data for field in required_fields):
            return jsonify({"message": "Missing required fields"}), 400
        
        harvest = mongo.db.harvest_predictions

        # Extract and validate features from the input data
        try:
            features = [
                float(data['pH']),
                float(data['Acerage']),
                float(data['Ca']),
                float(data['Mg']),
                float(data['K']),
                float(data['N']),
                float(data['P']),
                float(data['Zn']),
                float(data['Urea']),
                float(data['TSP']),
                float(data['MOP']),
                float(data['CaNO3']),
                float(data['Rainfall']),
                float(data['Temperature']),
                float(data['Expected Harvest']),
            ]
        except ValueError as e:
            return jsonify({"message": "Invalid data types for features", "error": str(e)}), 400

        # Convert features to the required format 
        features_array = np.array([features])

        # Make a prediction using the Random Forest model
        predicted_harvest = harvest_model.predict(features_array)

        # Save the data and the predicted results to MongoDB
        harvest.insert_one({
            'input_data': data,
            'predicted_harvest': float(predicted_harvest[0]),
            'prediction_date': str(date.today())
        })

        # Return the predicted value as a response
        return jsonify({
            'predicted_harvest': float(predicted_harvest[0])
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "Failed to predict harvest", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)