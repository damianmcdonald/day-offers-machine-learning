# datalab

This module contains Jupyter notebooks and training data.

The project is based on [Jupyter Notebook](https://jupyter.org/) technology and includes a common stack of machine learning technologies:

* [Python](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)

The project provides Jupyter notebooks for data exploration, analysis, visualization, and machine learning of day offers.

# Jupyter Workbooks

The project is composed of 3 workbooks:

| Workbook | Description |
|----------|-------------|
| dataset-analysis | Workbook which is used to perform dataset analysis. This workbook performs PCA (Principal Component Analysis) cumulative variance graphing, PairPlot graphing and covariance matrix generation |
| linear-regression | Workbook which trains, test and validates a day offer model using a [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) algorithm. This Workbook also permits ad-hoc predicition. |
| decisiontree-regression | Workbook which trains, tests and validates a day offer model using a [DecisionTreeRegression](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) algorithm. This Workbook also permits ad-hoc predicition. |

# Create Datalab environment

Additional information related to the creation of a Datalab environment can be found on the [GCP Quickstart page](https://cloud.google.com/datalab/docs/quickstart).

```bash
#!/bin/bash

DATALAB_INSTANCE_NAME=aa-datalab-instance
DATALAB_MACHINE_TYPE=n1-highmem-2

# Update your gcloud command-line tool components
sudo apt update -y && sudo apt upgrade google-cloud-sdk -y && sudo apt install google-cloud-sdk-datalab -y

# Create a Cloud Datalab instance
datalab create --machine-type $DATALAB_MACHINE_TYPE $DATALAB_INSTANCE_NAME

# Connect to the Cloud Datalab instance
datalab connect $DATALAB_INSTANCE_NAME
```

# Cleanup the Cloud Datalab instance

```bash
#!/bin/bash

DATALAB_INSTANCE_NAME=aa-datalab-instance

# Delete the Datalab instance
datalab delete --delete-disk $DATALAB_INSTANCE_NAME

# Delete the Datalab firewall rule
gcloud compute firewall-rules delete datalab-network-allow-ssh

# Connect to the Cloud Datalab network
gcloud compute networks delete datalab-network
```