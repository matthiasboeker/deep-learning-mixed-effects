from typing import List, Tuple
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import seaborn as sns
import matplotlib.pyplot as plt

from dataloader.ts_dataloader import TSDataset, custom_collate_fn
from models.cnn_models import TSCNNRE
from models.loss_functions import negative_log_likelihood
from utils.evaluation import evaluate_model
from models.explainability import integrated_gradients, compute_gradients


def train_model(optimizer, model, train_loader, epochs):
    for epoch in range(epochs):
        for ts_features, meta_features, y_batch, Z_batch, _, _ in train_loader:
            for i in range(ts_features.size()[1]):
                # Model with random effects

                optimizer.zero_grad()
                predictions_re = model(
                    ts_features[:, i, :], meta_features, Z_batch[:, i, :]
                )
                mask = (y_batch[:, i] != 0).float()
                predictions_re = predictions_re.squeeze()
                covariance_matrix = model.random_effects.get_covariance_matrix()
                nll_re = negative_log_likelihood(
                    y_batch[:, i] * mask,
                    predictions_re,
                    Z_batch[:, i, :],
                    covariance_matrix,
                    model.random_effects.nr_random_effects,
                    model.random_effects.nr_groups,
                )
                nll_re.backward()
                optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")


def test_model(model, test_loader):
    with torch.no_grad():
        test_y_list = []
        test_predictions_re_list = []

        for ts_features, meta_features, y_batch, Z_batch, _, _ in test_loader:
            for i in range(ts_features.size(1)):
                # Model with random effects
                predictions_re = model(
                    ts_features[:, i, :], meta_features, Z_batch[:, i, :]
                )
                test_predictions_re_list.append(predictions_re)
                test_y_list.append(y_batch[:, i])

        # Concatenate all batches
        test_predictions_re_full = torch.cat(test_predictions_re_list, dim=0)
        test_y_full = torch.cat(test_y_list, dim=0)

        # Evaluate the models
        evaluate_model(test_predictions_re_full, test_y_full, "TSNN Model with RE")


def get_all_predictions(model, train_loader, test_loader):
    with torch.no_grad():
        scoring_list = []
        for (
            ts_features,
            meta_features,
            y_batch,
            Z_batch,
            quarters,
            types,
        ) in train_loader:
            for i in range(ts_features.size(1)):
                predictions_re = model(
                    ts_features[:, i, :], meta_features, Z_batch[:, i, :]
                ).squeeze()
                scorings = pd.DataFrame(
                    {
                        "gt": y_batch[:, i].numpy(),
                        "preds": predictions_re.numpy(),
                        "quarters": quarters[:, i].numpy(),
                        "type": types[:, 0].numpy(),
                        "id": types[:, 1].numpy(),
                    }
                )
                scoring_list.append(scorings)
        for (
            ts_features,
            meta_features,
            y_batch,
            Z_batch,
            quarters,
            types,
        ) in test_loader:
            for i in range(ts_features.size(1)):
                predictions_re = model(
                    ts_features[:, i, :], meta_features, Z_batch[:, i, :]
                ).squeeze()
                scorings = pd.DataFrame(
                    {
                        "gt": y_batch[:, i].numpy(),
                        "preds": predictions_re.numpy(),
                        "quarters": quarters[:, i].numpy(),
                        "type": types[:, 0].squeeze().numpy(),
                        "id": types[:, 1].numpy(),
                    }
                )

                scoring_list.append(scorings)
    return pd.concat(scoring_list)


def train_test_split(dataset: TSDataset, split: float) -> Tuple[TSDataset]:
    dataset_size = len(dataset)
    train_size = int(split * dataset_size)
    test_size = dataset_size - train_size
    return random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )


def read_in_meta_data(path_to_meta_data: Path) -> pd.DataFrame:
    return pd.read_csv(path_to_meta_data, engine="python", sep=";")


def preprocess_metadata(
    metadata: pd.DataFrame, selected_meta_features: List[str]
) -> pd.DataFrame:
    """Preprocess the metadata dataframe.
    Male: 0 Female: 1
    TODO: STAI SCORING"""
    metadata = metadata[selected_meta_features].copy()
    metadata["Sex"] = metadata["Sex"].apply(lambda x: 0 if x == "Male" else 1)
    # metadata["Approximate number climbing"] = metadata[
    #   [
    #        "Approximate the number of hours that you spend top-rope climbing per week",
    #        "Approximate the number of hours that you spend lead climbing per week.",
    #    ]
    # ].sum(axis=1)
    # metadata["bmi"] = metadata.apply(
    #    lambda x: x["Body mass (kg)"] / (x["Height (cm)"] / 100) ** 2, axis=1
    # )
    # metadata = metadata.drop(["Body mass (kg)", "Height (cm)"], axis=1)
    metadata.fillna(metadata.mean(), inplace=True)
    return metadata


def main():
    sequence_length = 30
    batch_size = 64
    epochs = 100
    hidden_ts_size = 64
    hidden_meta_size = 32
    hidden_merge_size = 32
    num_channels = 16
    output_size = 1  # For regression

    path_to_ts_folder = Path(__file__).parent / "data" / "climbing_data"
    metadata = read_in_meta_data(
        Path(__file__).parent / "data" / "stai_questionnaire.csv"
    )
    selected_meta_features = [
        "Participant ID",
        "Age",
        "Sex",
        # "Body mass (kg)",
        # "Height (cm)",
        # "Climbing experience (years)",
        # "Bouldering experience (years)",
        # "Approximate the number of hours that you spend lead climbing per week.",
        # "Approximate the number of hours that you spend top-rope climbing per week",
        # "Approximate the number of hours that you spend bouldering per week",
    ]
    metadata = preprocess_metadata(metadata, selected_meta_features)
    selected_ts_features = ["MNF", "HR_mean"]
    selected_targets = [
        "Anxiety level",
        "Fear of falling due to fatigue",
        "Fear of heights",
    ]

    dataset = TSDataset(
        path_to_ts_folder,
        metadata,
        selected_ts_features,
        selected_targets,
        aggregation_fun=np.sum,
        sequence_length=sequence_length,
    )
    train_dataset, test_dataset = train_test_split(dataset, split=0.8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn
    )
    model = TSCNNRE(
        len(selected_ts_features),
        metadata.shape[1],
        hidden_merge_size,
        hidden_ts_size,
        hidden_meta_size,
        output_size,
        19,
        "intercepts",
        num_channels,
    )

    optimizer_re = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    train_model(optimizer_re, model, train_loader, epochs)
    test_model(model, test_loader)

    grads_l = []
    for ts_features, meta_features, y_batch, Z_batch, quarters, types in test_loader:
        for i in range(ts_features.size(1)):
            grads = compute_gradients(
                model, ts_features[:, i, :], meta_features, Z_batch[:, i, :]
            )
            grads_l.append(grads)
    grads = torch.stack(grads_l)
    grads = torch.mean(grads, dim=0)
    grads = pd.DataFrame(grads.numpy(), columns=selected_ts_features)
    sns.heatmap(grads)
    plt.show()
    print(grads.mean(axis=0))
    train_model(optimizer_re, model, test_loader, epochs)
    scores = get_all_predictions(model, train_loader, test_loader)
    scores.to_csv(Path(__file__).parent / "data" / "scores.csv")


if __name__ == "__main__":
    main()
