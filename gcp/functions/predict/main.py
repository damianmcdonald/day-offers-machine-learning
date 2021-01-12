from flask import jsonify
from google.cloud import storage
import firebase_admin
from firebase_admin import auth
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from joblib import load
import json
import os


def env_vars(var_name):
    return os.environ.get(var_name, f"Specified environment variable {var_name} is not set.")

FIREBASE_APP = firebase_admin.initialize_app()

PREFIX = 'Bearer'
ENV_BUCKET_NAME = "BUCKET_NAME"
ENV_BUCKET_PREFIX = "BUCKET_PREFIX"
ENV_SCALER_PHASE_1 = "SCALER_PHASE_1"
ENV_SCALER_PHASE_2 = "SCALER_PHASE_2"
ENV_SCALER_PHASE_3 = "SCALER_PHASE_3"
ENV_SCALER_PHASE_4 = "SCALER_PHASE_4"
ENV_MIN_VAL_PHASE_1 = "ENV_MIN_VAL_PHASE_1"
ENV_MIN_VAL_PHASE_2 = "ENV_MIN_VAL_PHASE_2"
ENV_MIN_VAL_PHASE_3 = "ENV_MIN_VAL_PHASE_3"
ENV_MIN_VAL_PHASE_4 = "ENV_MIN_VAL_PHASE_4"
ENV_CORS_ALLOWED_ORIGINS = "ENV_CORS_ALLOWED_ORIGINS"

SCALER_PHASE_1 = float(env_vars(ENV_SCALER_PHASE_1))
SCALER_PHASE_2 = float(env_vars(ENV_SCALER_PHASE_2))
SCALER_PHASE_3 = float(env_vars(ENV_SCALER_PHASE_3))
SCALER_PHASE_4 = float(env_vars(ENV_SCALER_PHASE_4))

MIN_VAL_PHASE_1 = float(env_vars(ENV_MIN_VAL_PHASE_1))
MIN_VAL_PHASE_2 = float(env_vars(ENV_MIN_VAL_PHASE_2))
MIN_VAL_PHASE_3 = float(env_vars(ENV_MIN_VAL_PHASE_3))
MIN_VAL_PHASE_4 = float(env_vars(ENV_MIN_VAL_PHASE_4))

CORS_ALLOWED_ORIGINS = env_vars(ENV_CORS_ALLOWED_ORIGINS)

predicition_result = []

# create google storage client
storage_client = storage.Client()
# get bucket name
bucket_name = env_vars(ENV_BUCKET_NAME)
# get bucket prefix
bucket_prefix = env_vars(ENV_BUCKET_PREFIX)
# grab bucket handle
bucket = storage_client.get_bucket(bucket_name)

print(f"bucket_name == {bucket_name}")
print(f"bucket_prefix == {bucket_prefix}")


def get_standard_value(scaler_max, scaled_value, phase_number):
    
    
    def min_phase_val(i):
        switcher = {
                1: MIN_VAL_PHASE_1,
                2: MIN_VAL_PHASE_2,
                3: MIN_VAL_PHASE_3,
                4: MIN_VAL_PHASE_4
             }
        return switcher.get(i,f"Invalid phase:{i}")


    val_to_round = scaled_value * scaler_max
    rounded_val = round((float(val_to_round)*4))/4
    if rounded_val > 0 and rounded_val > min_phase_val(phase_number):
        return rounded_val
             
    return min_phase_val(phase_number)


def get_phase_model(phase_number):
    phase_model = bucket.get_blob(f"{bucket_prefix}/phase_{phase_number}_model.model")
    phase_model_file = f"/tmp/phase_{phase_number}_model.model"
    with open(phase_model_file, "wb+") as model_file:
        model_file.write(phase_model.download_as_string())
    
    return phase_model_file


def make_phase_predictions(phase_number, phase_name, offer_details, scaler_value):
    print(f"Generating prediction for {phase_number}, {phase_name}.")
    regression_ad_hoc = load(get_phase_model(phase_number))
    print(regression_ad_hoc)
    phase_prediction = regression_ad_hoc.predict(offer_details)
    prediction_scaled = get_standard_value(scaler_value, phase_prediction[0], phase_number)
    predict_result = [phase_number, phase_name, prediction_scaled]
    predicition_result.append(predict_result)
    print(f"Predciction for phase {phase_number}, {phase_name}: real: {phase_prediction}, scaled: {prediction_scaled}")
    return prediction_scaled


def generate_predictions(predict_json):

    # phase 1 predicition (Recopilacion)
    phase1_predicition = make_phase_predictions(1, "Recopilacion", pd.json_normalize(predict_json), SCALER_PHASE_1)
    phase1_predicition_scaled = phase1_predicition / SCALER_PHASE_1
    # add the phase prediction to the predict_json so it is included in the next prediction
    predict_json["offer"]["phase1prediction"] = phase1_predicition_scaled

    # phase 2 predicition (Diseno)
    phase2_predicition = make_phase_predictions(2, "Diseno", pd.json_normalize(predict_json), SCALER_PHASE_2)
    phase2_predicition_scaled = phase2_predicition / SCALER_PHASE_2
    # add the phase prediction to the predict_json so it is included in the next prediction
    predict_json["offer"]["phase2prediction"] = phase2_predicition_scaled

    # phase 3 predicition (Implantacion)
    phase3_predicition = make_phase_predictions(3, "Implantacion", pd.json_normalize(predict_json), SCALER_PHASE_3)
    phase3_predicition_scaled = phase3_predicition / SCALER_PHASE_3
    # add the phase prediction to the predict_json so it is included in the next prediction
    predict_json["offer"]["phase3prediction"] = phase3_predicition_scaled

    # phase 4 predicition (Soporte)
    phase4_predicition = make_phase_predictions(4, "Soporte", pd.json_normalize(predict_json), SCALER_PHASE_4)
    phase4_predicition_scaled = phase4_predicition / SCALER_PHASE_4

    # create the prediction results
    predicted_offer = {}
    predicted_offer['phase1prediction'] = phase1_predicition
    predicted_offer['phase2prediction'] = phase2_predicition
    predicted_offer['phase3prediction'] = phase3_predicition
    predicted_offer['phase4prediction'] = phase4_predicition
    predicted_offer['totalprediction'] = phase1_predicition + phase2_predicition + phase3_predicition + phase4_predicition
    
    print("Predicted result:")
    print(json.dumps(predicted_offer, indent=4, sort_keys=True))

    return jsonify(predicted_offer)


def clean_up():
    for x in range(1, 5):
        model_file = f"/tmp/phase_{x}_model.model"
        if os.path.exists(model_file):
            os.remove(model_file)
        else:
            print(f"The file {model_file} does not exist")


def predict_phases(request):

    print(f"Request method: {request.method}")

    # Set CORS headers for preflight requests
    if request.method == 'OPTIONS':
        # Allows GET requests from origin https://mydomain.com with
        # Authorization header
        headers = {
            'Access-Control-Allow-Origin': CORS_ALLOWED_ORIGINS,
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Authorization, Origin, X-Requested-With, Content-Type, Accept',
            'Access-Control-Max-Age': '3600',
            'Access-Control-Allow-Credentials': 'true'
        }
        print(f"Preflight CORS request, return the following headers: {headers}")
        return ('', 204, headers)

    # Set CORS headers for main requests
    headers = {
        'Access-Control-Allow-Origin': CORS_ALLOWED_ORIGINS,
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Authorization, Origin, X-Requested-With, Content-Type, Accept',
        'Access-Control-Max-Age': '3600',
        'Access-Control-Allow-Credentials': 'true'
    }

    # validate POST method
    if request.method != 'POST':
        message = 'Method ' + request.method + ' not supported. Only POST method is supported.'
        return (message, 405, headers)

    # validate MIME TYPE
    if request.headers['Content-Type'] != 'application/json':
        message = 'Mime type ' + request.headers['Content-Type'] + ' is an unsupported Media Type. Only application/json method is supported.'
        return (message, 415, headers)

    # validate Firebase authentication
    if not 'Authorization' in request.headers:
        message = 'Forbidden. Authorization header missing.'
        return (message, 403, headers)

    if 'Authorization' in request.headers:
        auth_header = request.headers['Authorization']
        print(f"Authorization header: {auth_header}")
        bearer, _, token = auth_header.partition(' ')
        if bearer != PREFIX:
            message = 'Forbidden. Invalid authorization token.'
            return (message, 403, headers)

        try:
            print(f"Decoding user token: {token}")
            decoded_token = auth.verify_id_token(token)
            uid = decoded_token['uid']
            print(f"Verified user id as {uid}")
        except:
            print(f"Unable to verify token: {token}")
            message = 'Invalid authorization token.'
            return (message, 403, headers)
       

    predict_json = request.get_json(silent=True)

    print("Prediction request:")
    print(json.dumps(predict_json, indent=4, sort_keys=True))

    predicted_result = generate_predictions(predict_json)
    
    clean_up()

    return (predicted_result, 200, headers)