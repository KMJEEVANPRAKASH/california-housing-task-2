import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data():
    """Load and prepare the California Housing dataset."""

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    # Feature engineering
    df["RoomsPerPerson"] = (
        df["AveRooms"] / df["AveOccup"].replace(0, np.nan)
    )

    df["BedroomsPerRoom"] = (
        df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
    )

    # Remove invalid values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a regression model."""

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def main():

    print("=" * 55)
    print("California Housing — Model Testing")
    print("=" * 55)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Define models
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
    }

    results = []

    # Train and evaluate
    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)

    # Create comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="RMSE"
    ).reset_index(drop=True)

    print("\n" + "=" * 55)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 55)

    print(results_df.to_string(index=False))

    # Best model
    best_model_name = results_df.loc[0, "Model"]

    print("\n" + "=" * 55)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Lowest RMSE: {results_df.loc[0, 'RMSE']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
