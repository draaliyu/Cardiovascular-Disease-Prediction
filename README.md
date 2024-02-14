# Cardiovascular Disease Prediction

This project aims to predict the presence of cardiovascular disease in individuals using various health parameters. The machine learning model is built using Python and utilizes deep learning techniques to make predictions based on patient data.

## Installation

Before running this project, ensure you have the following Python libraries installed:
- pandas
- numpy
- scikit-learn
- TensorFlow
- Keras
- Matplotlib
- Seaborn

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn
```
## Data
The dataset used in this project is 'cardio_train.csv', which contains various health metrics for individuals, along with a target variable indicating the presence of cardiovascular disease.

## Features
The following features are used in this project to predict cardiovascular disease:

- Age (in years)
- Height (in cm)
- Weight (in kg)
- Blood pressure values (systolic and diastolic)
- Cholesterol levels
- Glucose levels
- Smoking status
- Alcohol intake
- Physical activity

## Model
The predictive model is a deep neural network built using TensorFlow and Keras. The network consists of multiple dense layers with ReLU activation functions and a final sigmoid layer for binary classification.

## Results
The results of the model training, including accuracy and loss metrics, are output to the console. Additionally, confusion matrices and classification reports are provided for evaluating the model's performance.
ons to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.


## Acknowledgments
Dataset provided by [Kaggle](https://www.kaggle.com/code/prabathwijethilaka/cardiovascular-disease/).