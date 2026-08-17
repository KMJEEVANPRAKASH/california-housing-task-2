"""
California Housing - Model Training Script

Trains and evaluates multiple regression models:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor

Models are saved in the ../models/ directory.
"""

import os
import joblib
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

print("Loading California Housing dataset...")

data = fetch_california_housing()

X = data.data
y = data.target

print(f"Dataset shape: {X.shape}")


# ---------------------------------------------------------
# 2. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ---------------------------------------------------------
# 3. Define Models
# ---------------------------------------------------------

models = {

    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "ridge_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "lasso_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001))
    ]),

    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ---------------------------------------------------------
# 4. Create Models Directory
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# 5. Train and Evaluate Models
# ---------------------------------------------------------

results = {}

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        f"{name}.pkl"
    )

    joblib.dump(model, model_path)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"Saved: {model_path}")


# ---------------------------------------------------------
# 6. Display Model Comparison
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<25}"
    f"{'MAE':<12}"
    f"{'RMSE':<12}"
    f"{'R²':<12}"
)

print("-" * 61)

for name, metrics in results.items():

    print(
        f"{name:<25}"
        f"{metrics['MAE']:<12.4f}"
        f"{metrics['RMSE']:<12.4f}"
        f"{metrics['R2']:<12.4f}"
    )


# ---------------------------------------------------------
# 7. Find Best Model
# ---------------------------------------------------------

best_model_name = max(
    results,
    key=lambda x: results[x]["R2"]
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best Model: {best_model_name}")
print(f"R² Score  : {results[best_model_name]['R2']:.4f}")
print(f"RMSE      : {results[best_model_name]['RMSE']:.4f}")
print(f"MAE       : {results[best_model_name]['MAE']:.4f}")

print("\nAll models trained and saved successfully!")
