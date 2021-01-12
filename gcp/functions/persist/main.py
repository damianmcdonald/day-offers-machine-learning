import firebase_admin
from firebase_admin import auth
import pygsheets
import json
import os


def env_vars(var_name):
    return os.environ.get(var_name, f"Specified environment variable {var_name} is not set.")


FIREBASE_APP = firebase_admin.initialize_app()

PREFIX = 'Bearer'
ENV_CORS_ALLOWED_ORIGINS = "ENV_CORS_ALLOWED_ORIGINS"
ENV_SA_CREDENTIALS = "ENV_SA_CREDENTIALS"
ENV_GSHEET_WBOOK_ID = "ENV_GSHEET_WBOOK_ID"
ENV_GSHEET_SHEET_NAME = "ENV_GSHEET_SHEET_NAME"

CORS_ALLOWED_ORIGINS = env_vars(ENV_CORS_ALLOWED_ORIGINS)
SA_CREDENTIALS = env_vars(ENV_SA_CREDENTIALS)
GSHEET_WBOOK_ID = env_vars(ENV_GSHEET_WBOOK_ID)
GSHEET_SHEET_NAME = env_vars(ENV_GSHEET_SHEET_NAME)


def write_secret_to_file():
    print(f"Writing secret to file")
    sa_credentials_file = f"/tmp/service_account_credentials.json"
    with open(sa_credentials_file, "w+") as credentials_file:
        credentials_file.write(SA_CREDENTIALS)
    
    print(f"Secret file written to: {sa_credentials_file}")
    return sa_credentials_file


def authenticate_api():
    print(f"Authenticating to GCP Sheets API")
    credentials_file_path = write_secret_to_file()
    return pygsheets.authorize(service_account_file=credentials_file_path)


def get_next_worksheet_row(workbook, sheet_name):
    print(f"Get the next available writable row")
    cells = workbook.worksheet_by_title(sheet_name).get_all_values(include_tailing_empty_rows=False,
                                                                  include_tailing_empty=False,
                                                                  returnas='matrix')
    end_row = len(cells)
    return end_row + 1


def update_worksheet(predictions):
    print(f"Updating the worksheet")
    print(predictions)
    # cell ranges for the DATASET worksheet
    RANGES_CELL_FIRST = "A"
    RANGES_CELL_LAST = "F"
    # grab the handle to the gcp sheets api
    api_handle = authenticate_api()
    # grab the workbook
    workbook = api_handle.open_by_key(GSHEET_WBOOK_ID)
    worksheet_data = workbook.worksheet_by_title(GSHEET_SHEET_NAME)
    row_start = get_next_worksheet_row(workbook, GSHEET_SHEET_NAME)
    range_update = f"{RANGES_CELL_FIRST}{row_start}:{RANGES_CELL_LAST}{row_start}"
    print(f"Writing data for client {predictions['client']} to sheet {GSHEET_SHEET_NAME}, range {range_update}")
    predictions_values = [
                             predictions['client'],
                             predictions['datetime'],
                             predictions['phase1prediction'],
                             predictions['phase2prediction'],
                             predictions['phase3prediction'],
                             predictions['phase4prediction'],
                             predictions['totalprediction']
                         ]
    worksheet_data.update_row(row_start, predictions_values, col_offset=0)

    print(f"Worksheet updated")


def persist_predictions(request):
    
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
    print(predict_json)
    print(f"Prediction DICT request: {predict_json}")

    # update the worksheet
    print(f"Updating worksheet: {GSHEET_WBOOK_ID}")
    predict_dict = predict_json['phasePredictions']
    print(f"Predict dict: {predict_dict}")
    update_worksheet(predict_dict)

    return (json.dumps(predict_dict, indent=4, sort_keys=True), 200, headers)