# functions

Cloud functions that provide ML prediction and persistence capabilities.

Function | Description
------------ | -------------
[predict](predict) | Cloud Function that uses a persisted ML model (*LinearRegression* or *DecisionTree*) stored in Cloud Storage to provide ML predictions for the phases of a JSON provided offer. Requires Firebase authentication.
[predict-noauth](predict-noauth) | Cloud Function that uses a persisted ML model (*LinearRegression* or *DecisionTree*) stored in Cloud Storage to provide ML predictions for the phases of a JSON provided offer. No authentication required.
[persist](persist) | Cloud Function that persists data to a Google Sheet.

# Testing a cloud function

The [predict-noauth](predict-noauth) Cloud Function can be tested using the following `curl` command.

Firstly, we need a JSON payload that represents the offer that we want to predict on.

```json
{
  "offer": {
    "greenfield": 1,
    "vpc": 0.5,
    "subnets": 0.75,
    "connectivity": 1,
    "peerings": 0,
    "directoryservice": 0,
    "otherservices": 0.2,
    "advsecurity": 0,
    "advlogging": 0,
    "advmonitoring": 0,
    "advbackup": 0,
    "vms": 0.5,
    "buckets": 0.5,
    "databases": 1,
    "elb": 0,
    "autoscripts": 0,
    "administered": 0
  }
}
```

The function can be invoked with the command line below.

```bash
curl --silent -H "Content-Type: application/json" --data-binary @offer.json "https://europe-west1-day-offers-ml-poc.cloudfunctions.net/predict-noauth" | jq
```

The function will return a prediction similar to the response shown below:

```json 
{
  "phase1prediction": 1,
  "phase2prediction": 1.75,
  "phase3prediction": 2,
  "phase4prediction": 1,
  "totalprediction": 5.75
}
```
