from typing import Dict
import torch
import matplotlib.pyplot as plt  # type: ignore
from matplotlib import cm  # type: ignore


def visualise_longitudinal_data(y, samples, q):
    timesteps = samples // q
    time = torch.arange(1, timesteps + 1)
    colors = cm.get_cmap("tab20", q)  # Get q colors from the 'tab20' colormap
    # Plot each subject's data over time
    plt.figure(figsize=(10, 6))
    for i in range(q):
        plt.plot(
            time.numpy(),
            y[i * timesteps : (i + 1) * timesteps].numpy(),
            marker="o",
            color=colors(i),
            label=f"Subject {i+1}",
        )

    plt.xlabel("Time (Time Steps)")
    plt.ylabel("Observed Output (y)")
    plt.title("Random Intercepts and Slopes for Each Subject Over Time")
    plt.show()


def visualise_regression_results(
    test_y_full: torch.Tensor, predictions: Dict[str, torch.Tensor]
):
    plt.figure(figsize=(12, 6))
    colors = ["red", "blue", "green"]
    for i, (model_name, prediction) in enumerate(predictions.items()):
        plt.scatter(
            test_y_full, prediction, color=colors[i], alpha=0.5, label=model_name
        )
    plt.plot(
        [test_y_full.min(), test_y_full.max()],
        [test_y_full.min(), test_y_full.max()],
        "k--",
    )
    plt.xlabel("Actual Outcomes")
    plt.ylabel("Predicted Outcomes")
    plt.title("Comparison of Predictions")
    plt.legend()
    plt.show()


def plot_residuals(true_values: torch.Tensor, predictions: Dict[str, torch.Tensor]):
    colors = ["red", "blue", "green"]
    plt.figure(figsize=(12, 6))

    for i, (model_name, prediction) in enumerate(predictions.items()):
        res = prediction - true_values
        plt.scatter(true_values, res, color=colors[i], alpha=0.5, label=model_name)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("True Values")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot for RE and No RE Models")
    plt.legend()
    plt.show()
