# 🏠 California Housing Price Prediction — Task 2

## Feature Engineering, Model Optimization & Performance Comparison

This project is an extension of the **California Housing Price Prediction — Task 1** project.

The objective of Task 2 is to improve the machine learning workflow by applying **feature engineering, feature scaling, multiple regression algorithms, hyperparameter optimization, and model performance comparison**.

---

## 🎯 Project Objectives

- Perform exploratory data analysis (EDA)
- Create meaningful engineered features
- Apply feature scaling
- Train multiple regression models
- Optimize model hyperparameters
- Evaluate models using MAE, RMSE, and R²
- Compare model performance
- Identify the best-performing model
- Analyze residuals and feature importance
- Save the trained model for future use

---

## 📊 Dataset

### California Housing Dataset

The project uses the **California Housing Dataset** available through `scikit-learn`.

The target variable is:

```text
MedHouseVal

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

California Housing Dataset
          ↓
Exploratory Data Analysis
          ↓
Data Preprocessing
          ↓
Feature Engineering
          ↓
Feature Scaling
          ↓
Train / Test Split
          ↓
 ┌───────────────────────────┐
 │                           │
Linear Regression       Ridge Regression
 │                           │
 └──────────────┬────────────┘
                ↓
       Decision Tree
        Regressor
                ↓
   Hyperparameter Tuning
                ↓
     Model Evaluation
                ↓
      Model Comparison
                ↓
       Best Model

            Feature Engineering

Two additional features are created from the original dataset.

1. RoomsPerPerson
df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]

This represents the relationship between the average number of rooms and average household occupancy.

2. BedroomsPerRoom
df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]

This represents the proportion of bedrooms relative to the average number of rooms.

                Feature Scaling

StandardScaler is used to standardize the numerical features.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Scaling is particularly useful for the Linear Regression and Ridge Regression workflows.
