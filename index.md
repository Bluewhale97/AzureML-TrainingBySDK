## Introduction

Azure machine learning provides the complete SDKs to create scripts for multiple compute environments, log model performance and version the models and logging metrics. It is very helpful to engineer the machine learning project. As a data scientist, not only do we need perform machine learning assignments but also deliver the values to the bussiness objectives, SDKs in Azure helps a lot to engineering the works of machine learning pipelines and manage them.

Now let's us see how to train a machine learning model with Azure machine leanring SDKs.

## 1. Running a training script

When you want to run a training script, you can use a ScriptRunConfig to run a script-based experiment that trains a machine learnig model.

Your script should save the trained model in the outputs folder when you want to use an experiment to train a model. 

The following code is using Scikit-learn to train a model and save it in the outputs folder using the joblib package.

```python
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the experiment run context
run = Run.get_context()

# Prepare the dataset
diabetes = pd.read_csv('./diabetes.csv')
X, y = diabetes[['Pregnancies','PlasmaGlucose','TricepsThickness']].values, diabetes['Diabetic'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model
reg = 0.1
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetic_model.pkl')

run.complete()
```

It shows that the accuracy is 0.68:

![image](https://user-images.githubusercontent.com/71245576/116010027-a7bc5500-a5ea-11eb-9165-2c8fea2833f6.png)

Now you can see this file diabetic_model.pkl is saved in outputs root:

![image](https://user-images.githubusercontent.com/71245576/116010050-d20e1280-a5ea-11eb-8883-267dbf35eefd.png)

To prepare for an experiment that trains a model, a script like this is created and saved in a folder. For example, you can save this script as training_script.py in a folder named training_folder. Since the script includes code to load training data from data.csv, this file should also be saved in this folder.

To run the script as an experiment, create a ScriptRunConfig that references the folder and script file. You need to define a Python environment that includes any packages required by the script. In this example, the script uses Scikit-Learn so you must create an environment that includes that. The script also uses Azure machine learning to log metrics, so you need to remember to include the azureml-defaults package in the environment.

```python
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create a Python environment for the experiment
sklearn_env = Environment("sklearn-env")

# Ensure the required packages are installed
packages = CondaDependencies.create(conda_packages=['scikit-learn','pip'],
                                    pip_packages=['azureml-defaults'])
sklearn_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory='training_folder',
                                script='training.py',
                                environment=sklearn_env) 

# Submit the experiment
experiment = Experiment(workspace=ws, name='training-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion()
```

It runs me 90.4s, a little bit long...

## 2. Using script parameters

We can use arguments to set variables in the script. To use parameters in a script, we must use a library such as argparse to read the arguments passed to the script and assign them to variables.

The following script reads an argument named --reg-rate, which is used to set the regularization rate hyperparameter for the logistic regression algorithm used to train a model:

```python
from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg-rate', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args(args=[])
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()

# Prepare the dataset
diabetes = pd.read_csv('./diabetes.csv')
X, y = diabetes[['Pregnancies','PlasmaGlucose','TricepsThickness']].values, diabetes['Diabetic'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model
reg = 0.1
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetic_model.pkl')

run.complete()
```
There is a bug that is:

![image](https://user-images.githubusercontent.com/71245576/116010604-f8817d00-a5ed-11eb-9c0a-0cff7af24ea0.png)

I did not find a good solution online but I solved it. It is obviously a problem about compatibility by kernels and pips. Only thing you need to do is to exchange the statement: args = parser.parse_args() as args = parser.parse_args(args=[]). It will be solved through this interface.

See the accuracy:

![image](https://user-images.githubusercontent.com/71245576/116010679-72b20180-a5ee-11eb-9f4a-b3612989a932.png)

Now let's pass arguments to an experiment script, we need to provide an arguments value containing a list of comman-separated arguments and their values to the script:

```python
# Create a script config
script_config = ScriptRunConfig(source_directory='training_folder',
                                script='training.py',
                                arguments = ['--reg-rate', 0.1],
                                environment=sklearn_env)
```

## 3. Registering models

After running an experiment that trains a model, you can use a reference to the Run object to retrieve its output, including the trained models.

First, you can use the run objects get_file-names method to list the files generated. Standard practice is for scripts that train models to save them in the run's outputs folder.

You also can use the run object's download_file and download_files methods to download output files to the local file system:

```python
# "run" is a reference to a completed experiment run

# List the files generated by the experiment
for file in run.get_file_names():
    print(file)

# Download a named file
run.download_file(name='outputs/model.pkl', output_file_path='model.pkl')
```
Actually you can save your model to outputs automatically before the run completes.

Now let's registering a model. Model registration enables you to track multiple versions of a model and retrieve models for inferencing(predicting label values from new data). When you register a model, you can specify a name, description, tags, framework (such as Scikit-Learn or PyTorch), framework version, custom properties, and other useful metadata. Registering a model with the same name as an existing model automatically creates a new version of the model, starting with 1 and increasing in units of 1.

See:
```python
from azureml.core import Model

model = Model.register(workspace=ws,
                       model_name='classification_model',
                       model_path='model.pkl', # local path
                       description='A classification model',
                       tags={'data-format': 'CSV'},
                       model_framework=Model.Framework.SCIKITLEARN,
                       model_framework_version='0.20.3')
```
You can go back to your models pane under Assets, you can find that the model is registered:

![image](https://user-images.githubusercontent.com/71245576/116011165-a17da700-a5f1-11eb-82b7-1db8d9c63562.png)

Note that about the model path, you can use either the local path or the run outputs path.

Now view registered models using the Model object to retrieve details of registered models like this:
```python
from azureml.core import Model

for model in Model.list(ws):
    # Get model name and auto-generated version
    print(model.name, 'version:', model.version)
```
Because I have just registered one model, so it showed:

![image](https://user-images.githubusercontent.com/71245576/116011240-f02b4100-a5f1-11eb-969e-2fa5866f0b52.png)

## Reference:

Build AI solutions with Azure Machine Learning, retrieved from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/


