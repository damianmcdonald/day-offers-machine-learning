#!/usr/bin/env python
# coding: utf-8

# # Predicciones de jornadas de fases con AA (Aprendizaje Automático)

# ## Familia de algoritmos: <code>XGBoost</code>

# ## Importación de las bibliotecas de AA en Python

# In[ ]:


import pandas as pd
import numpy as np 
from scipy.stats import uniform, randint
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import datetime
import uuid
import math
import sys
import csv
import sqlite3
import json
get_ipython().system('conda install --yes --prefix {sys.prefix} prettytable xgboost graphviz python-graphviz mscorefonts')
import xgboost as xgb
from prettytable import PrettyTable
import matplotlib.font_manager as font_manager
font_manager._rebuild()
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Helper functions

# In[ ]:


PATIENCE_THRESHOLD = 10
EPOCHS = 1000

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
    

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters")
            print(json.dumps(results['params'][candidate], indent=4, sort_keys=True))
            print("")
            
            
def get_min_value(predicted_value, phase_number):
    
    
    def min_phase_val(i):
        switcher = {
                1: 0.75,
                2: 1.5,
                3: 2,
                4: 0.25
             }
        return switcher.get(i,f"Invalid phase:{i}")


    rounded_val = round((float(predicted_value)*4))/4
    if rounded_val > 0 and rounded_val > min_phase_val(phase_number):
        return rounded_val
             
    return min_phase_val(phase_number)


# ## Cargar los datos para entreñar y testar el modelo de AA

# In[ ]:


# load the csv data sets
csv_dataset = pd.read_csv("train/train-offers-dataset.csv")

# print the first few rows to make sure that data has been loaded as expected
csv_dataset.head()


# ## Preparar los datos de entreño-test; quitar funciones, normalizar y asignar escaladores

# In[ ]:


# drop features with low occurance/correlation
df_train_test_base = pd.DataFrame(csv_dataset)

# shuffle the data
df_train_test_base_shuffled = shuffle(df_train_test_base, random_state=0)

# phaseprediction test train split
df_train_test_phase1 = df_train_test_base_shuffled.drop(['phase2prediction', 'phase3prediction', 'phase4prediction'], axis = 1)
phase1_x_train = df_train_test_phase1[['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered']]
phase1_y_train = df_train_test_phase1['phase1prediction']

df_train_test_phase2 = df_train_test_base_shuffled.drop(['phase3prediction', 'phase4prediction'], axis = 1) 
phase2_x_train = df_train_test_phase2[['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered', 'phase1prediction']]
phase2_y_train = df_train_test_phase2['phase2prediction']

df_train_test_phase3 = df_train_test_base_shuffled.drop(['phase4prediction'], axis = 1)
phase3_x_train = df_train_test_phase3[['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered', 'phase1prediction', 'phase2prediction']]
phase3_y_train = df_train_test_phase3['phase3prediction']

df_train_test_phase4 = df_train_test_base_shuffled.copy()
phase4_x_train = df_train_test_phase4[['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered', 'phase1prediction', 'phase2prediction', 'phase3prediction']]
phase4_y_train = df_train_test_phase4['phase4prediction']

df_train_test_base_shuffled.head()


# ## Cargar los datos para validar el modelo de AA

# In[ ]:


# load the csv data sets
csv_datavalidation = pd.read_csv("validation/validation-offers-dataset.csv")


# ## Preparar los datos de validación; quitar funciones, normalizar y asignar escaladores

# In[ ]:


# drop features with low occurance/correlation
df_validation_base = pd.DataFrame(csv_datavalidation)
          
# shuffle the data
df_validation_base_shuffled = shuffle(df_validation_base, random_state=0)

# phaseprediction test train split
df_validation_phase1 = df_validation_base_shuffled.drop(['phase2prediction', 'phase3prediction', 'phase4prediction'], axis = 1)
phase1_x_validation = df_validation_phase1[['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered']]
phase1_y_validation = df_validation_phase1['phase1prediction']

df_validation_phase2 = df_validation_base_shuffled.drop(['phase3prediction', 'phase4prediction'], axis = 1) 
phase2_x_validation = df_validation_phase2[['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered', 'phase1prediction']]
phase2_y_validation = df_validation_phase2['phase2prediction']

df_validation_phase3 = df_validation_base_shuffled.drop(['phase4prediction'], axis = 1)
phase3_x_validation = df_validation_phase3[['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered', 'phase1prediction', 'phase2prediction']]
phase3_y_validation = df_validation_phase3['phase3prediction']

df_validation_phase4 = df_validation_base_shuffled.copy()
phase4_x_validation = df_validation_phase4[['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered', 'phase1prediction', 'phase2prediction', 'phase3prediction']]
phase4_y_validation = df_validation_phase4['phase4prediction']

df_validation_phase1.head()


# ## Evaluacion del modelo

# In[ ]:


# uuid for the evaluation run
eval_id = uuid.uuid4()

# datetime of the evaluation run
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# results list
results_list = []

def k_fold_score_model(phase_number, phase_name, X_train, Y_train):
    print(f"Evaluating model for phase: {phase_number}: {phase_name}")
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)

    model = xgb.XGBRegressor(
        booster="gbtree", 
        n_jobs=6, 
        verbosity=0, 
        n_estimators=EPOCHS, 
        importance_type="weight", 
        random_state=42, 
        eval_metric="mae"
    )
    
    print(f"Scoring neg_mean_absolute_error for phase: {phase_number}: {phase_name}")
    scoring_mae = 'neg_mean_absolute_error'
    result_mae = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring_mae)
    
    print(f"Scoring neg_mean_squared_error for phase: {phase_number}: {phase_name}")
    scoring_mse = 'neg_mean_squared_error'
    result_mse = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring_mse)

    print(f"Scoring r2 (r squared) for phase: {phase_number}: {phase_name}")
    scoring_r2 = 'r2'
    result_r2 = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring_r2)
    
    score_results = [eval_id,now,phase_number,phase_name,result_mse.mean(),result_mse.std(),result_mae.mean(),result_mae.std(),result_r2.mean(),result_r2.std()]
    results_list.append(score_results)


# execute ML
k_fold_score_model("1", "Recopilacion", phase1_x_train, phase1_y_train)
k_fold_score_model("2", "Diseno", phase2_x_train, phase2_y_train)
k_fold_score_model("3", "Implantacion", phase3_x_train, phase3_y_train)
k_fold_score_model("4", "Soporte", phase4_x_train, phase4_y_train)

# write the text and csv results
df_score_results = pd.DataFrame(results_list, columns = ['evaluation_id','datetime','phasenumber','phasename','mse_mean','mse_std','mae_mean','mae_std','r2_mean','r2_std'])

with open("analysis/model/xgboost/model-evaluation.csv", "a+") as csv_file:
    df_score_results.to_csv(csv_file, header=False, index=False)
    
df_score_results.head()


# ## Buscar los hyperparametros optimales

# In[ ]:


def search_optimal_hyperparameters(phase_number, phase_name, x_train, y_train):
    print(f"Searching optimal hyperparameters for phase: {phase_number}: {phase_name}")
    
    xgb_model = xgb.XGBRegressor()

    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3), # default 0.1 
        "max_depth": randint(2, 6), # default 3
        "subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=3, n_jobs=4, return_train_score=True)

    print(f"Fitting the data for phase: {phase_number}: {phase_name}")
    search.fit(x_train, y_train)

    print(f"Reporting best scores for phase: {phase_number}: {phase_name}")
    report_best_scores(search.cv_results_, 1)

  
# execute ML
search_optimal_hyperparameters("1", "Recopilacion", phase1_x_train, phase1_y_train)
search_optimal_hyperparameters("2", "Diseno", phase2_x_train, phase2_y_train)
search_optimal_hyperparameters("3", "Implantacion", phase3_x_train, phase3_y_train)
search_optimal_hyperparameters("4", "Soporte", phase4_x_train, phase4_y_train)


# ## Determinar el mejor model utilizando early stopping

# In[ ]:


best_model_iterations = {}


def determine_early_stopping(phase_number, phase_name, x_train, y_train):
    print(f"Determine best early stopping run for phase: {phase_number}: {phase_name}")
    
    xgb_model = xgb.XGBRegressor(
        booster="gbtree", 
        n_jobs=4, 
        verbosity=0, 
        n_estimators=EPOCHS, 
        importance_type="weight", 
        random_state=42, 
        eval_metric="mae"
    )

    X_axis_train, X_axis_test, y_axis_train, y_axis_test = train_test_split(x_train, y_train, random_state=42)

    xgb_model.fit(X_axis_train, y_axis_train, early_stopping_rounds=PATIENCE_THRESHOLD, eval_set=[(X_axis_test, y_axis_test)])

    y_pred = xgb_model.predict(X_axis_test)

    print("best score: {0}, best iteration: {1}, best ntree limit {2}".format(xgb_model.best_score, xgb_model.best_iteration, xgb_model.best_ntree_limit))
    best_model_iterations[phase_number] = xgb_model.best_iteration + 1
    print("**************************************")

  
# execute ML
determine_early_stopping("1", "Recopilacion", phase1_x_train, phase1_y_train)
determine_early_stopping("2", "Diseno", phase2_x_train, phase2_y_train)
determine_early_stopping("3", "Implantacion", phase3_x_train, phase3_y_train)
determine_early_stopping("4", "Soporte", phase4_x_train, phase4_y_train)


# ## Generar el modelo y guardarlo al disco

# In[ ]:


def save_best_model(phase_number, phase_name, phase_params, phase_predictor, x_train, y_train):
    print(f"Save best model for phase: {phase_number}: {phase_name}")
    print(f"Column number: {x_train.num_col()}")
    print(f"Best number of rounds: {best_model_iterations[phase_number]} for phase: {phase_number}: {phase_name}")
    best_model = xgb.train(
        phase_params, 
        x_train, 
        num_boost_round=best_model_iterations[phase_number], 
        evals=[(y_train, phase_predictor)]
    )
    
    xgb.plot_importance(best_model)

    # converts the target tree to a graphviz instance
    xgb.to_graphviz(best_model, num_trees=best_model.best_iteration)
    
    print(f"Saving model to phase_{phase_number}_model.model")
    best_model.save_model(f"models-saved/xgboost/phase_{phase_number}_model.model")
    

# model 1
phase1_params = {
    "booster": "gbtree",
    "importance_type": "weight",
    "n_jobs": 6,
    "random_state": 42, 
    "eval_metric": "mae",
    "colsample_bytree": 0.7992694074557947,
    "gamma": 0.03177917514301182,
    "learning_rate": 0.12329469651469865,
    "max_depth": 5,
    "subsample": 0.8918424713352255
}

# model 2
phase2_params = {
    "booster": "gbtree",
    "importance_type": "weight",
    "n_jobs": 6, 
    "random_state": 42, 
    "eval_metric": "mae",
    "colsample_bytree": 0.796805916483275,
    "gamma": 0.02170039164908638,
    "learning_rate": 0.30739299906707884,
    "max_depth": 5,
    "subsample": 0.7011960671956965
}

# model 3
phase3_params = {
    "booster": "gbtree",
    "importance_type": "weight",
    "n_jobs": 6,
    "random_state": 42, 
    "eval_metric": "mae",
    "colsample_bytree": 0.796805916483275,
    "gamma": 0.02170039164908638,
    "learning_rate": 0.30739299906707884,
    "max_depth": 5,
    "subsample": 0.7011960671956965
}

# model 4
phase4_params = {
    "booster": "gbtree",
    "importance_type": "weight",
    "n_jobs": 6,
    "random_state": 42, 
    "eval_metric": "mae",
    "colsample_bytree": 0.9248848696025762,
    "gamma": 0.01523607636679708,
    "learning_rate": 0.290164494322417,
    "max_depth": 3,
    "subsample": 0.758865533001633
}

# execute ML
X_dmatrix_train_phase1, X_dmatrix_test_phase1, y_dmatrix_train_phase1, y_dmatrix_test_phase1 = train_test_split(phase1_x_train,phase1_y_train,test_size=.3, random_state=42)
dtrain_phase1 = xgb.DMatrix(X_dmatrix_train_phase1, label=y_dmatrix_train_phase1)
dtest_phase1 = xgb.DMatrix(X_dmatrix_test_phase1, label=y_dmatrix_test_phase1)
save_best_model("1", "Recopilacion", phase1_params, "phase1prediction", dtrain_phase1, dtest_phase1)

X_dmatrix_train_phase2, X_dmatrix_test_phase2, y_dmatrix_train_phase2, y_dmatrix_test_phase2 = train_test_split(phase2_x_train,phase2_y_train,test_size=.3, random_state=42)
dtrain_phase2 = xgb.DMatrix(X_dmatrix_train_phase2, label=y_dmatrix_train_phase2)
dtest_phase2 = xgb.DMatrix(X_dmatrix_test_phase2, label=y_dmatrix_test_phase2)
save_best_model("2", "Diseno", phase2_params, "phase2prediction", dtrain_phase2, dtest_phase2)

X_dmatrix_train_phase3, X_dmatrix_test_phase3, y_dmatrix_train_phase3, y_dmatrix_test_phase3 = train_test_split(phase3_x_train,phase3_y_train,test_size=.3, random_state=42)
dtrain_phase3 = xgb.DMatrix(X_dmatrix_train_phase3, label=y_dmatrix_train_phase3)
dtest_phase3 = xgb.DMatrix(X_dmatrix_test_phase3, label=y_dmatrix_test_phase3)
save_best_model("3", "Implantacion", phase3_params, "phase3prediction", dtrain_phase3, dtest_phase3)

X_dmatrix_train_phase4, X_dmatrix_test_phase4, y_dmatrix_train_phase4, y_dmatrix_test_phase4 = train_test_split(phase4_x_train,phase4_y_train,test_size=.3, random_state=42)
dtrain_phase4 = xgb.DMatrix(X_dmatrix_train_phase4, label=y_dmatrix_train_phase4)
dtest_phase4 = xgb.DMatrix(X_dmatrix_test_phase4, label=y_dmatrix_test_phase4)
save_best_model("4", "Soporte", phase4_params, "phase4prediction", dtrain_phase4, dtest_phase4)


# ## Predecir las jornadas por fase con los datos de validación

# In[ ]:


# uuid for the evaluation run
eval_id = uuid.uuid4()

# datetime of the evaluation run
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# results list
results_list = []


def predict_phases(phase_number, phase_name, phase_predictions, X_validation):
    print(f"Phase predictions for phase: {phase_number}: {phase_name}")
    RESULT_TYPE_OVER_ESTIMATE = "SOBRE_ESTIMACION"
    RESULT_TYPE_UNDER_ESTIMATE = "SUB_ESTIMACION"
    RESULT_TYPE_EQUAL_ESTIMATE = "EQUAL_ESTIMACION"
    
    loaded_model = xgb.Booster()
    loaded_model.load_model(f"models-saved/xgboost/phase_{phase_number}_model.model")
    
    # get predictions based on validation data set
    predictors = loaded_model.predict(X_validation)
    
    # enumerate the predictions
    for idx, j in enumerate(predictors):
        predicted = get_min_value(j, phase_number)
        actual = phase_predictions[idx]
        loss = 0
        percentage = 0
        result_type = RESULT_TYPE_EQUAL_ESTIMATE
        
        # OVERestimate
        if predicted > actual:
            loss = predicted - actual
            if loss > 0 and actual > 0:
                percentage = (loss/predicted)*100
            result_type = RESULT_TYPE_OVER_ESTIMATE
        # UNDERestimate
        elif actual > predicted:
            loss = actual - predicted
            if loss > 0 and actual > 0:
                percentage = ((loss/actual)*100) * -1
            
            result_type = RESULT_TYPE_UNDER_ESTIMATE

        predict_result = [eval_id, idx, now,phase_number, phase_name, actual, predicted, loss, percentage, result_type]
        results_list.append(predict_result)
            

# execute predictions using the validation dataset
dvalidation_phase1 = xgb.DMatrix(phase1_x_validation, label=phase1_y_validation)
predict_phases(1, "Recopilacion", phase1_y_validation, dvalidation_phase1)

dvalidation_phase2 = xgb.DMatrix(phase2_x_validation, label=phase2_y_validation)
predict_phases(2, "Diseno", phase2_y_validation, dvalidation_phase2)

dvalidation_phase3 = xgb.DMatrix(phase3_x_validation, label=phase3_y_validation)
predict_phases(3, "Implantacion", phase3_y_validation, dvalidation_phase3)

dvalidation_phase4 = xgb.DMatrix(phase4_x_validation, label=phase4_y_validation)
predict_phases(4, "Soporte", phase4_y_validation, dvalidation_phase4)

# write the text and csv result
df_prediction_results = pd.DataFrame(results_list, columns = ['evaluation_id','correlation_id','datetime','phasenumber','phasename','actual','predicted','loss','percentage', 'result_type']).sort_values('correlation_id', ascending=True)

model_predictions_csv = f"analysis/predictions/xgboost/model-predictions_{eval_id}.csv"
with open(model_predictions_csv, "w+") as csv_file:
    df_prediction_results.to_csv(csv_file, header=True, index=False)
print("\n")
print(f"Evaluatiuon id: {eval_id}")
print("\n")

df_prediction_results.head()


# ## Analisis de los resultados

# In[ ]:


conn = sqlite3.connect(":memory:")
df_model_predictions = pd.read_csv(model_predictions_csv)
df_model_predictions.to_sql("predictions", conn, if_exists='append', index=False)

query_details = [ {"phasenumber": 1, "phasename": "Recopilacion", "min_val": -0.5, "max_val": 0.5}, 
                  {"phasenumber": 2, "phasename": "Diseno", "min_val": -0.75, "max_val": 0.75},
                  {"phasenumber": 3, "phasename": "Implantacion", "min_val": -0.75, "max_val": 0.75},
                  {"phasenumber": 4, "phasename": "Soporte", "min_val": -0.50, "max_val": 0.50}
                ]

accum_acceptable_offers = 0
accum_precise_offers = 0
accum_non_acceptable_offers = 0
accum_non_acceptable_under = 0
accum_non_acceptable_over = 0
accum_total_offers = 0

# uuid for the evaluation run
eval_id = uuid.uuid4()

# datetime of the evaluation run
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

phase_analysis_results = []
    
for detail in query_details:
    print("####################################################################################")
    print("Estadisticas de fase {}: {}".format(detail['phasenumber'], detail['phasename']))
    print("\n")
    
    phase_query_ofertas_precise = "SELECT COUNT(correlation_id) AS ofertas_precise FROM predictions WHERE phasenumber = {} AND ((actual_std-predicted_std) = 0)".format(detail['phasenumber'])
    pd.set_option('display.max_rows', df_model_predictions.shape[0]+1)
    df_precise_offers = pd.read_sql_query(phase_query_ofertas_precise, conn)
    
    phase_query_ofertas_aceptable = "SELECT COUNT(correlation_id) AS ofertas_aceptable FROM predictions WHERE phasenumber = {} AND ((actual_std-predicted_std) >= {} AND (actual_std-predicted_std) <= {})".format(detail['phasenumber'], detail['min_val'], detail['max_val'])
    pd.set_option('display.max_rows', df_model_predictions.shape[0]+1)
    df_acceptable_offers = pd.read_sql_query(phase_query_ofertas_aceptable, conn)

    phase_query_ofertas_noaceptable = "SELECT COUNT(correlation_id) AS ofertas_no_aceptable FROM predictions WHERE phasenumber = {} AND ((actual_std-predicted_std) < {} OR (actual_std-predicted_std) > {})".format(detail['phasenumber'], detail['min_val'], detail['max_val'])
    pd.set_option('display.max_rows', df_model_predictions.shape[0]+1)
    df_non_acceptable_offers = pd.read_sql_query(phase_query_ofertas_noaceptable, conn)
    
    phase_query_ofertas_under = "SELECT COUNT(correlation_id) AS sub_estimaciones FROM predictions WHERE phasenumber = {} AND ((actual_std-predicted_std) > {})".format(detail['phasenumber'], detail['max_val'])
    pd.set_option('display.max_rows', df_model_predictions.shape[0]+1)
    df_non_acceptable_under = pd.read_sql_query(phase_query_ofertas_under, conn)
    
    phase_query_ofertas_over = "SELECT COUNT(correlation_id) AS sobre_estimaciones FROM predictions WHERE phasenumber = {} AND ((actual_std-predicted_std) < {})".format(detail['phasenumber'], detail['min_val'])
    pd.set_option('display.max_rows', df_model_predictions.shape[0]+1)
    df_non_acceptable_over = pd.read_sql_query(phase_query_ofertas_over, conn)
    
    precise_offers = df_precise_offers['ofertas_precise'].iloc[0]
    acceptable_offers = (df_acceptable_offers['ofertas_aceptable'].iloc[0]) - precise_offers
    non_acceptable_offers = df_non_acceptable_offers['ofertas_no_aceptable'].iloc[0]
    total_offers = precise_offers + acceptable_offers + non_acceptable_offers
    acceptable_percentage = "{:.2f}".format(((precise_offers + acceptable_offers)/total_offers)*100)
    margin_error = detail['max_val'] * 8

    under_estimations = df_non_acceptable_under['sub_estimaciones'].iloc[0]
    over_estimations = df_non_acceptable_over['sobre_estimaciones'].iloc[0]
    
    accum_acceptable_offers = accum_acceptable_offers + acceptable_offers
    accum_non_acceptable_offers = accum_non_acceptable_offers + non_acceptable_offers
    accum_precise_offers = accum_precise_offers + precise_offers
    accum_non_acceptable_under = accum_non_acceptable_under + under_estimations
    accum_non_acceptable_over = accum_non_acceptable_over + over_estimations
    accum_total_offers = accum_total_offers + total_offers
    
    table = PrettyTable(['ofertas total','ofertas precisas','ofertas aceptable','ofertas no aceptable','aceptable %','margen error horas +/-','sub estimaciones','sobre estimaciones'])
    table.add_row([
        total_offers,
        precise_offers,
        acceptable_offers, 
        non_acceptable_offers,
        acceptable_percentage+"%",
        margin_error,
        under_estimations,
        over_estimations
    ])

    print(table)
    print("\n")
    
    phase_query_ofertas = "SELECT correlation_id AS oferta, phasenumber, phasename, actual_std AS actual, predicted_std As predicted, actual_std-predicted_std As diff, printf('%.2f', CASE WHEN predicted_std > actual_std THEN ((predicted_std-actual_std)/predicted_std)*100 WHEN actual_std > predicted_std THEN ((actual_std-predicted_std)/actual_std)*100*-1 ELSE 0 END) AS 'diff %' FROM predictions WHERE phasenumber = {} ORDER BY DIFF ASC".format(detail['phasenumber'])
    print("POR OFERTA: fase {} - {}".format(detail['phasenumber'], detail['phasename']))
    pd.set_option('display.max_rows', df_model_predictions.shape[0]+1)
    df_prediction_by_offer = pd.read_sql_query(phase_query_ofertas, conn)
    print(pd.DataFrame(df_prediction_by_offer))
    print("\n")
    
    validation_offer_results_csv = "analysis/predictions/linearlearner/validation-metrics-offers-{}.csv".format(eval_id)
    with open(validation_offer_results_csv, "a+") as csv_file:
        pd.DataFrame(df_prediction_by_offer).to_csv(csv_file, header=False, index=False)
    
    phase_data = [eval_id, now, detail['phasenumber'], detail['phasename'], total_offers, precise_offers, acceptable_offers, non_acceptable_offers, acceptable_percentage, margin_error, under_estimations, over_estimations]
    phase_analysis_results.append(phase_data)
  

print("####################################################################################")
print("Totales de todas las fases")
accum_table = PrettyTable(['fases total','fases precisa','fases aceptable','fases no aceptable','aceptable %','sub estimaciones','sobre estimaciones'])

accum_acceptable_percentage = "{:.2f}".format(((accum_precise_offers + accum_acceptable_offers)/accum_total_offers)*100)
accum_table.add_row([
    accum_total_offers, 
    accum_precise_offers,
    accum_acceptable_offers, 
    accum_non_acceptable_offers,
    accum_acceptable_percentage,
    accum_non_acceptable_under,
    accum_non_acceptable_over
])

print(accum_table)
print("\n")
    
phase_query_ofertas = "SELECT correlation_id AS oferta, phasenumber, phasename, actual_std AS actual, predicted_std As predicted, actual_std-predicted_std As diff, printf('%.2f', CASE WHEN predicted_std > actual_std THEN ((predicted_std-actual_std)/predicted_std)*100 WHEN actual_std > predicted_std THEN ((actual_std-predicted_std)/actual_std)*100*-1 ELSE 0 END) AS 'diff %' FROM predictions ORDER BY correlation_id, phasenumber ASC"
print("####################################################################################")
print("CADA FASE AGRUPADO POR OFERTA")
print("\n")
pd.set_option('display.max_rows', df_model_predictions.shape[0]+1)
print(pd.read_sql_query(phase_query_ofertas, conn))
print("\n")

total_data = [eval_id, now, 5, 'Total', accum_total_offers, accum_precise_offers, accum_acceptable_offers, accum_non_acceptable_offers, accum_acceptable_percentage, 0, accum_non_acceptable_under, accum_non_acceptable_over]
phase_analysis_results.append(total_data)
df_analysis_results = pd.DataFrame(phase_analysis_results, columns = ['evaluation_id', 'datetime', 'phasenumber', 'phasename', 'ofertas_total','estimaciones_precisa','ofertas_aceptable','ofertas_no_aceptable','percentage_aceptable','margen error horas +/-','sub_estimaciones','sobre_estimaciones'])

validation_results_csv = "analysis/predictions/linearlearner/validation-metrics.csv"
with open(validation_results_csv, "a+") as csv_file:
    df_analysis_results.to_csv(csv_file, header=False, index=False)
    
df_analysis_results.head()


# ## Visualización de los resultados

# In[ ]:


phases = ['Recopilación', 'Disneo', 'Implantación', 'Soporte', 'Totales']
precisas = df_analysis_results['estimaciones_precisa'].values
aceptable = df_analysis_results['ofertas_aceptable'].values
errors = df_analysis_results['ofertas_no_aceptable'].values

pos = np.arange(len(phases))

plt.bar(pos, precisas, width=0.8, label='precisas', color='#00b871', bottom=aceptable+errors)
plt.bar(pos, aceptable, width=0.8, label='aceptable', color='#68fc1e', bottom=errors)
plt.bar(pos, errors, width=0.8, label='errors', color='red')

plt.xticks(pos, phases)
plt.ylabel("Precisión")
plt.xlabel("Phases")
plt.legend(loc="upper left")
plt.title("Precisión de las estimaciones")

# set the figure size
plt.figure(figsize=(15,15))
plt.rcParams["figure.figsize"] = [12,10]
plt.show()


# ## Prediccion EndPoint / Invocación Ad-Hoc

# ### Definición de oferta y invocación de predicción ad-hoc

# In[ ]:


##################################
#        OFFER DETAILS           #
##################################

offer = {
  "greenfield": 1,          # 1 == TRUE, 0 == FALSE
  "vpc": 0.5,               # 0 == 0 vpc, 0.5 == 1 vpc, 1 == 2 vpc
  "subnets": 0.75,          # 0 == 0 subnets, 0.25 == 1 subnets, 0.5 == 2 subnets, 0.75 == 3 subnets, 1 == 4 subnets
  "connectivity": 1,        # 1 == TRUE, 0 == FALSE 
  "peerings": 0,            # 1 == TRUE, 0 == FALSE 
  "directoryservice": 0,    # 1 == TRUE, 0 == FALSE
  "otherservices": 1,     # 0 == 0 otherservices, 0.2 == 1 otherservices, 0.4 == 2 otherservices, 0.6 == 3 otherservices, 0.8 == 4 otherservices, 1 == 5 otherservices
  "advsecurity": 0,         # 1 == TRUE, 0 == FALSE
  "advlogging": 0,          # 1 == TRUE, 0 == FALSE
  "advmonitoring": 0,       # 1 == TRUE, 0 == FALSE
  "advbackup": 0,           # 1 == TRUE, 0 == FALSE
  "vms": 0.3,               # 0 == 0 vms, 0.1 == 1 vms, 0.2 == 2 vms, 0.3 == 3 vms .... 0.8 == 8 vms, 0.9 == 9 vms, 1 == 10 vms
  "buckets": 0.5,           # 0 == 0 buckets, 0.5 == 1 buckets, 1 == 2 buckets
  "databases": 1,           # 0 == 0 BBDD, 0.5 == 1 BBDD, 1 == 2 BBDD
  "elb": 0,                 # 1 == TRUE, 0 == FALSE 
  "autoscripts": 0,         # 1 == TRUE, 0 == FALSE 
  "administered": 0         # 1 == TRUE, 0 == FALSE 
}

##################################
#      OFFER DETAILS END         #
##################################


##################################
# HERE BE DRAGONS!!! 
# Modify the code below 
# at your own risk
##################################

predicition_result = []

def make_phase_predictions(phase_number, phase_name, offer_details):
    loaded_model = xgb.XGBRegressor()
    loaded_model.load_model(f"models-saved/xgboost/phase_{phase_number}_model.model")
    # dvalidation_phase1 = xgb.DMatrix(phase1_x_validation, label=phase1_y_validation)
    phase_prediction = loaded_model.predict(offer_details)
    prediction_scaled = get_min_value(phase_prediction[0], phase_number)
    predict_result = [phase_number, phase_name, prediction_scaled]
    predicition_result.append(predict_result)
    return prediction_scaled


# phase 1 predicition
offer_rows = {'offer': offer}
offer_frame = pd.DataFrame.from_dict(offer_rows, orient='index')
phase1_predicition = make_phase_predictions(1, "Recopilacion", offer_frame)

# phase 2 predicition
offer['phase1prediction'] = phase1_predicition
offer_rows = {'offer': offer}
offer_frame = pd.DataFrame.from_dict(offer_rows, orient='index')
phase2_predicition = make_phase_predictions(2, "Diseno", offer_frame)

# phase 3 predicition
offer['phase2prediction'] = phase2_predicition
offer_rows = {'offer': offer}
offer_frame = pd.DataFrame.from_dict(offer_rows, orient='index')
phase3_predicition = make_phase_predictions(3, "Implantacion", offer_frame)

# phase 4 predicition
offer['phase3prediction'] = phase3_predicition
offer_rows = {'offer': offer}
offer_frame = pd.DataFrame.from_dict(offer_rows, orient='index')
phase4_predicition = make_phase_predictions(4, "Soporte", offer_frame)

print("Detalles de la oferta:\n")
offer['phase1prediction'] = predicition_result[0][2]
offer['phase2prediction'] = predicition_result[1][2]
offer['phase3prediction'] = predicition_result[2][2]
offer['phase4prediction'] = predicition_result[3][2]
print(json.dumps(offer, indent=4, sort_keys=True))

print("\nPrediccion:\n")
table = PrettyTable(['# Fase','Fase','Jornadas'])
table.align["# Fase"] = "c"
table.align["Fase"] = "l"
table.align["Jornadas"] = "c"
for row in predicition_result:
    table.add_row(row)

table.add_row([5, 'Total', (phase1_predicition + phase2_predicition + phase3_predicition + phase4_predicition)])

print(table)

