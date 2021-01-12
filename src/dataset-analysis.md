# Analisis del los datos de Ofertas por Jornada en el contexto AA (Aprendizaje Automático)

## Importación de las bibliotecas de AA en Python


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import datetime
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Cargar los datos para evaluar correlaciones


```python
# load the csv data sets
csv_dataset = pd.read_csv("train/train-offers-dataset.csv")

# create a Panda DataFramce
df_dataset = pd.DataFrame(csv_dataset)

# print the first few rows to make sure that data has been loaded as expected
df_dataset.head()
```

## Inspectión de los tipos de Datos


```python
df_dataset.dtypes
```

## Descripción del DataFrame


```python
df_dataset.describe()
```

## Quitar las funciones que no aportan valor al modelo y normalizar los datos


```python
# drop featuresd with low occurance/correlation

# drop the labels from the dataset
df_dataset_pca = pd.DataFrame(csv_dataset).drop(['phase1prediction','phase2prediction','phase3prediction','phase4prediction'], axis = 1)
df_dataset_pca.head()
```

## Visualizar el Variance Ratio entre las functiones 


```python
pca = PCA().fit(df_dataset_pca)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('numero de funciones')
plt.ylabel('cumulative explained variance');
```

## Visualizar un PairPlot del modelo


```python
jet= plt.get_cmap('jet')

def generate_pairplot(df, variables, predictor_name, phase_name, n_rows, n_cols):
    print("\n")
    print(f"Correlacion de funciones en {phase_name}")
    print("\n")
    fig = plt.figure(figsize=(22,20))
    for i, var in enumerate(variables):
        ax = fig.add_subplot(n_rows,n_cols,i+1)
        asset = df.loc[:,var]
        ax.scatter(df[predictor_name], asset)
        ax.set_xlabel(predictor_name)
        ax.set_ylabel("{}".format(var))
        ax.set_title(f"{var} vs {predictor_name}")
    fig.tight_layout() 
    plt.show()

# normalize the magnitude of the dataset
df_dataset_scaler = preprocessing.MinMaxScaler()
df_dataset_scaled = pd.DataFrame(df_dataset_scaler.fit_transform(df_dataset), columns = ['greenfield','vpc', 'subnets', 'connectivity', 'peerings', 'directoryservice', 'otherservices', 'advsecurity', 'advlogging', 'advmonitoring', 'advbackup', 'vms', 'buckets', 'databases', 'elb', 'autoscripts', 'administered', 'phase1prediction', 'phase2prediction', 'phase3prediction', 'phase4prediction'])

df_dataset_phase1 = df_dataset_scaled.drop(['phase2prediction', 'phase3prediction', 'phase4prediction'], axis = 1)
df_dataset_phase2 = df_dataset_scaled.drop(['phase3prediction', 'phase4prediction'], axis = 1)
df_dataset_phase3 = df_dataset_scaled.drop(['phase4prediction'], axis = 1)
df_dataset_phase4 = df_dataset_scaled.copy()

# generate_pairplot(df_dataset_phase1, df_dataset_phase1.columns, "phase1prediction", "Phase 1: Recopilacion", 5, 4)
# generate_pairplot(df_dataset_phase2, df_dataset_phase2.columns, "phase2prediction", "Phase 2: Diseno", 5, 4)
# generate_pairplot(df_dataset_phase3, df_dataset_phase3.columns, "phase3prediction", "Phase 3: Implantacion", 5, 4)
generate_pairplot(df_dataset_phase4, df_dataset_phase4.columns, "phase4prediction", "Todas las fases", 5, 5)
```

## Calcular el Covariance Matrix


```python
def generate_covariance_matrix_graph(df, cmap_color, phase_name):
    # plot the covariance matrix
    corr = df.corr()
    # figsize in inches
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_title(phase_name)
    sns.heatmap(corr, annot = True, square=True, fmt='.2g', vmin=-1, vmax=1, center= 0, cmap= cmap_color, linewidths=3, linecolor='black', ax=ax)

    
# generate_covariance_matrix_graph(df_dataset_phase1, "coolwarm", "Phase 1: Recopilacion")
# generate_covariance_matrix_graph(df_dataset_phase2, "BrBG_r", "Phase 2: Diseno")
# generate_covariance_matrix_graph(df_dataset_phase3, "gist_rainbow_r", "Phase 3: Implantacion")
generate_covariance_matrix_graph(df_dataset_phase4, "twilight_shifted_r", "Phase 4: Soporte")
```

## Covariance matrix en formato tabular


```python
# uuid for the evaluation run
eval_id = uuid.uuid4()

# datetime of the evaluation run
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

corr_data_frames_arr = []


def generate_covariance_matrix_table(df, phase_name, phase_predictor):
    corr_df = df.corr(method='pearson', min_periods=10)[phase_predictor]
    corr_data_frames_arr.append(pd.DataFrame(corr_df).T)
    

generate_covariance_matrix_table(df_dataset_phase1, "Phase 1: Recopilacion", "phase1prediction")
generate_covariance_matrix_table(df_dataset_phase2, "Phase 2: Diseno", "phase2prediction")
generate_covariance_matrix_table(df_dataset_phase3, "Phase 3: Implantacion", "phase3prediction")
generate_covariance_matrix_table(df_dataset_phase4, "Phase 4: Soporte", "phase4prediction")

corr_data_frame = pd.DataFrame(pd.concat(corr_data_frames_arr))
corr_data_frame.insert(0, 'evaluation_id', eval_id)
corr_data_frame.insert(1, 'datetime', now)

with open(f"analysis/dataset/covariance-matrix.csv", "a+") as csv_file:
    corr_data_frame.to_csv(csv_file, header=False, index=False)
    
print("\n")
print(f"Evaluatiuon id: {eval_id}")
print("\n")
corr_data_frame.head()
```
