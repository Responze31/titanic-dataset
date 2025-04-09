# Titanic Survival Prediction

## Overview
This project analyzes the famous Titanic dataset to predict passenger survival using various machine learning models. The implementation compares the performance of KNN, Decision Trees, Random Forests, Gaussian Naive Bayes, and SVM classifiers.
Dataset
The Titanic dataset contains passenger information including:

## Demographics (age, gender)
Ticket class
Fare paid
Cabin information
Port of embarkation
Survival status (target variable)

## Models Implemented

K-Nearest Neighbors (KNeighborsClassifier)
Decision Tree (DecisionTreeClassifier)
Random Forest (RandomForestClassifier)
Gaussian Naive Bayes (GaussianNB)
Support Vector Machine (SVC)

## Requirements

scikit-learn
matplotlib
seaborn
numpy
pandas

## Usage

Clone this repository
Install the required packages: pip install -r requirements.txt
Run the Jupyter notebook or Python script

## Results
The models were evaluated using K-fold cross-validation to ensure robust performance assessment. Detailed analysis of each model's performance can be found in the notebook.
Future Work

Feature engineering to improve model performance
Hyperparameter tuning
Ensemble methods exploration
Deployment of the best performing model
