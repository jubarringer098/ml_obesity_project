# Introduction
## Overview
This month's Kaggle challenge is centered around a critical health issue: predicting obesity risk levels in individuals. Obesity is directly linked to numerous health complications, particularly cardiovascular diseases.

## Goal
The goal is to employ machine learning techniques to predict different obesity levels based on various factors, thereby contributing to broader health analytics and preventive medicine efforts.

## Evaluation Metrics
The effectiveness of the submitted models will be evaluated using the accuracy score, a common metric for classification problems

## Dataset Description
The dataset provided for this competition has been synthetically generated, mimicking the distribution of a real-world dataset related to obesity and cardiovascular disease risks. While the dataset mirrors real-world scenarios, it has been slightly altered for the competition. The dataset includes demographic information, dietary habits, and physical activity levels, collected from a diverse group of individuals across Mexico, Peru, and Colombia.

The attributes related with eating habits are: Frequent consumption of high caloric food (FAVC), Frequency of consumption of vegetables (FCVC), Number of main meals (NCP), Consumption of food between meals (CAEC), Consumption of water daily (CH20), and Consumption of alcohol (CALC). The attributes related with the physical condition are: Calories consumption monitoring (SCC), Physical activity frequency (FAF), Time using technology devices (TUE), Transportation used (MTRANS).

The target variable, NObeyesdad, represents the obesity level classified into 7 classes: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III.

**train.csv**: the training dataset includes all features and the target variable. This data will be used to train the model.

**test.csv**: the test dataset includes all features and excludes the target variable. This data will be used to evaluate the accuracy of the model for the competition.


### [Data Ingestion](https://github.com/xxgracexx098/ml_obesity_project/blob/main/src/components/data_ingestion.py)
### [Exploratory Data Analysis](https://github.com/xxgracexx098/ml_obesity_project/blob/main/notebook/Exploratory%20Data%20Analysis%20%7C%20Obesity%20Risk.ipynb)
#### [Charts file](https://github.com/xxgracexx098/ml_obesity_project/blob/main/notebook/Charts_for_EDA.ipynb)
### [Data Preprocessing](https://github.com/xxgracexx098/ml_obesity_project/blob/main/src/components/data_preprocessing.py)
### [Model Selection and Training](https://github.com/xxgracexx098/ml_obesity_project/blob/main/src/components/model_trainer.py)
