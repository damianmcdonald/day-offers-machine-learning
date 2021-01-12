# notebook-data

`notebook-data` is a child project of the parent project [day-offers-ml](/day-offers-machine-learning).

This module contains the Jupyter notebooks and training data used by the Jupyter Docker image.

The project is based on [Jupyter Notebook](https://jupyter.org/) technology and includes a common stack of machine learning technologies:

* [Python](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)
* [XGBoost](https://en.wikipedia.org/wiki/XGBoost)
* [TensorFlow](https://www.tensorflow.org/)

The project permits the training, testing and validation of day offer models. The project can provide project phase day effort predicitions in batch or indidivudal modes.

## Prequisites

The only pre-requisite for the `day-offers-machine-learning` project is a functioning [Jupyter Notebook](https://jupyter.org/) environment.

### Docker

The prefered way to execute a Jupyter Notebook is via a [docker](https://www.docker.com/) image.

```bash
#!/bin/bash
#
#   ____   __          _                 _                            _                              
#  / __ \ / _|        | |               | |                          | |             /\        /\    
# | |  | | |_ ___ _ __| |_ __ _ ___     | | ___  _ __ _ __   __ _  __| | __ _ ___   /  \      /  \   
# | |  | |  _/ _ \ '__| __/ _` / __|_   | |/ _ \| '__| '_ \ / _` |/ _` |/ _` / __| / /\ \    / /\ \  
# | |__| | ||  __/ |  | || (_| \__ \ |__| | (_) | |  | | | | (_| | (_| | (_| \__ \/ ____ \  / ____ \ 
#  \____/|_| \___|_|   \__\__,_|___/\____/ \___/|_|  |_| |_|\__,_|\__,_|\__,_|___/_/    \_\/_/    \_\
#                                                                                                    
#                                                                                                    

##################################### NOTES #####################################

# Once the notebook is launched, the docker container will output a URL similar to the one shown below:
# http://127.0.0.1:8888/?token=312600380cba7ef641c4a625fd56b10a7015601332d0251c
# This URL refers to the container port NOT the host port.
# Therefore, you should modify the port number (from 8888 to 10000) prior to accessing the URL on the host.

#################################################################################

# define the notebook docker image
SCIPY_NOTEBOOK=jupyter/scipy-notebook:3b1f4f5e6cc1

# pull the image (if the image is already pulled it will skip this step)
docker pull ${SCIPY_NOTEBOOK}


# run the docker container
##  the notebook will listen on port 10000 on the host machine
##  the notebook will persist any data saved in the "work" folder onm the guest to the "notebook-data" folder on the host
docker run --rm -p 10000:8888 -e JUPYTER_ENABLE_LAB=yes -v "${PWD}/notebook-data":/home/jovyan/work ${SCIPY_NOTEBOOK}
```

### Portable Jupyter Workbook

In cases where it is not possible to use the Jupyter Workbook docker image, an alternative approach is to use the [Jupyter Workbook Portable](https://www.portabledevapps.net/jupyter-portable.php) version which can be installed and executed on a Windows workstation without Admin permissions.

## Grab and run the project

Execute the steps below to grab and run the project using docker.

```bash
git clone https://github.com/damianmcdonald/day-offers-machine-learning.git day-offers-machine-learning
cd day-offers-machine-learning
chmod u+x launch-notebook.sh
./launch-notebook.sh
```

## Jupyter Workbooks

The project is composed of 5 workbooks:

| Workbook | Description |
|----------|-------------|
| dataset-analysis | Workbook which is used to perform dataset analysis. This workbook performs PCA (Principal Component Analysis) cumulative variance graphing, PairPlot graphing and covariance matrix generation |
| linear-regression | Workbook which trains, test and validates a day offer model using a [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) algorithm. This Workbook also permits ad-hoc predicition. |
| decisiontree-regression | Workbook which trains, tests and validates a day offer model using a [DecisionTreeRegression](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) algorithm. This Workbook also permits ad-hoc predicition. |
| tensorflow-regression | Workbook which trains, tests and validates a day offer model using a [TensorFlow Dense Layer](https://www.tensorflow.org/js/guide/models_and_layers) algorithm. This Workbook also permits ad-hoc predicition. |
| xgboost-regression | Workbook which trains, tests and validates a day offer model using a [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/index.html) algorithm. This Workbook also permits ad-hoc predicition. |

Static instances of the workbooks are available at the links below:

* [dataset-analysis workboook](/day-offers-machine-learning/src/dataset-analysis.md)
* [linear-regression workboook](/day-offers-machine-learning/src/linear-regression.md)
* [decisiontree-regression workboook](/day-offers-machine-learning/src/decisiontree-regression.md)
* [tensorflow-regression workboook](/day-offers-machine-learning/src/tensorflow-regression.md)
* [xgboost-regression workboook](/day-offers-machine-learning/src/xgboost-regression.md)
