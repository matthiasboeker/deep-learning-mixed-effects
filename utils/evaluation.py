import torch
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
)
import numpy as np


def evaluate_model(predictions, y_true, model_name):
    with torch.no_grad():
        y_pred = predictions.numpy()
        y_true = y_true.numpy()
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)

        # Number of observations and predictors (if available)
        n = len(y_true)
        p = predictions.shape[1] if len(predictions.shape) > 1 else 1

        # Adjusted R²
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Print results
        print(
            f"{model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Adjusted R²: {adjusted_r2:.4f},"
        )
        print(f"Explained Variance: {explained_var:.4f}")
