# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## User Installation

- Create an environment using `deploy/conda/env.yaml` file
```
conda env create -f deploy/conda/env.yaml
```
> Note: The name of the environment is `mle-dev-mlflow` which can be found in the name section of `env.yaml` file
- Activate the environment once environment is created
```
conda activate mle-dev-mlflow
```
- To install `housing_price` package, run the below command:
```
python setup.py install
```
> **Note:** All the above code commands have to be run in the root folder of the project.

## Testing the installation

- After activating `mle-dev-mlflow` environment, run the following command
```
python tests/installation_tests/housing_price_test.py
```
If there is no error thrown up, then the package installation can be deemed successful.
> **Note:** All the above code commands have to be run in the root folder of the project.

## Execute Scripts

- To run the scripts, use the below command:
```
python src/house_price_prediction/<script_name>.py --args
```
- To get the arguments taken by each script, use the below command:
```
python src/house_price_prediction/<script_name>.py --help
```

> **Note:** All the above code commands have to be run in the root folder of the project.
