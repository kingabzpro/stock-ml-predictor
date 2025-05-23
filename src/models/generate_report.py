#!/usr/bin/env python3
"""
Generate evaluation report for trained model
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.fetch_data import StockDataFetcher
from src.data.preprocess import DataPreprocessor
from src.models.evaluate import ModelEvaluator


def main():
    # Get the project root directory (3 levels up from this script)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Set up paths relative to project root
    model_path = os.path.join(
        project_root, "models", "trained", "AAPL_gradient_boosting_latest.pkl"
    )
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Load data
    fetcher = StockDataFetcher()
    test_data = fetcher.fetch_single("AAPL", "2020-01-01", "2023-12-31")

    # Initialize evaluator
    evaluator = ModelEvaluator(model_path)

    # Prepare data
    preprocessor = DataPreprocessor()
    X_test, y_test = preprocessor.prepare_data(test_data)

    # Get predictions
    if evaluator.scaler:
        X_test_scaled = pd.DataFrame(
            evaluator.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
    else:
        X_test_scaled = X_test

    predictions = evaluator.model.predict(X_test_scaled)

    # Calculate metrics
    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "mape": np.mean(np.abs((y_test - predictions) / y_test)) * 100,
    }

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(reports_dir, "model_metrics.csv"), index=False)

    # Generate and save prediction plot
    pred_fig = evaluator.plot_predictions(
        y_test, predictions, "AAPL Price Predictions vs Actual"
    )
    pred_fig.write_html(os.path.join(reports_dir, "predictions_plot.html"))

    # Generate and save residuals plot
    residual_fig = evaluator.plot_residuals(y_test, predictions)
    residual_fig.write_html(os.path.join(reports_dir, "residuals_plot.html"))

    print(f"Report generated and saved to {reports_dir}/")
    print("Metrics:")
    for metric, value in metrics.items():
        print(f"  - {metric.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
