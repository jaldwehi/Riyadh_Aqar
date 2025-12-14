# Riyadh Villa Price Prediction - ML Project
## Project Overview

This project aims to predict villa prices in Riyadh based on a variety of property-related features.
We chose Riyadh because it is a large and diverse real estate market, which provides rich data and variation across neighborhoods.

The main goal is to build a model that can accurately predict the price for any property entered by the user through a FastAPI interface.


## Dataset
 • Source: https://www.kaggle.com/code/alialmuhaysin/riyadh-villas-aqar-dataset-analysis-eda
 
 • Size: 46,826 rows and 26 columns
 
 • Target variable: Price
 
 • Most influential features identified:
 
 • space
 
 • neighborhood
 
 • location


## Data Cleaning
The dataset contained a high number of:

 • Outliers
 
 • Missing values
 

We also:

 • Removed an unimportant column
 
 • Fixed inconsistent values
 
 • Converted all columns to the correct data types
 
 • Documented all cleaning steps clearly in the EDA notebook


 ## Exploratory Data Analysis (EDA)
 
Our EDA included:

 • Price distribution
 
 • Relationship between numerical features and price
 
 • Outlier detection & visualization
 
 • Log transformation to normalize highly skewed data

These steps helped us better understand the patterns and improve modeling.


## Preprocessing

Each model had its own preprocessing pipeline to match its requirements.

We used:

 • One-Hot Encoding
 
 • Label Encoding
 
 • Train/Test Split before encoding to avoid data leakage
 
 • Normalization / scaling where necessary
 

This allowed every algorithm to train on properly prepared data.


## Machine Learning Models

We selected algorithms that handle high-dimensional data and non-linear relationships, including:

 • SVR (Support Vector Regression)
 
 • Random Forest Regressor
 
 • XGBoost Regressor

These were the most suitable for our dataset due to the large number of features and variety in property characteristics.


## 🏆Best Model

The XGBoost Regressor achieved the best performance among all tested models, giving the most accurate price predictions.


## Deployment (FastAPI)
The final model was deployed using FastAPI, allowing users to input property details and receive a predicted villa price instantly.
