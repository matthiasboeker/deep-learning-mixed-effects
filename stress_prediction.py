from typing import List, Tuple
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


from dataloader.ts_dataloader import TSDataset, custom_collate_fn
from models.timeseries_networks import TSNN, TSNNRE
from models.loss_functions import negative_log_likelihood
from example_script import evaluate_model, visualise_regression_results


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
    metadata["Approximate number climbing"] = metadata[
        [
            "Approximate the number of hours that you spend top-rope climbing per week",
            "Approximate the number of hours that you spend lead climbing per week.",
        ]
    ].sum(axis=1)
    metadata["bmi"] = metadata.apply(
        lambda x: x["Body mass (kg)"] / (x["Height (cm)"] / 100) ** 2, axis=1
    )
    metadata.fillna(metadata.mean(), inplace=True)
    scaler = StandardScaler()
    metadata.iloc[:, 1:] = scaler.fit_transform(metadata.iloc[:, 1:])
    return metadata


def main():
    sequence_length = 90
    batch_size = 16
    epochs = 100
    hidden_ts_size = 64
    hidden_meta_size = 32
    hidden_merge_size = 64
    num_layers = 1
    output_size = 1  # For regression

    path_to_ts_folder = Path(__file__).parent / "data" / "climbing_data"
    metadata = read_in_meta_data(
        Path(__file__).parent / "data" / "stai_questionnaire.csv"
    )
    selected_meta_features = [
        "Participant ID",
        "Age",
        "Sex",
        "Body mass (kg)",
        "Height (cm)",
        "Climbing experience (years)",
        "Bouldering experience (years)",
        "Approximate the number of hours that you spend lead climbing per week.",
        "Approximate the number of hours that you spend top-rope climbing per week",
        "Approximate the number of hours that you spend bouldering per week",
    ]
    metadata = preprocess_metadata(metadata, selected_meta_features)
    selected_ts_features = ["MNF"]
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

    # Create the model with and without random effects
    model_re = TSNN(
        len(selected_ts_features),
        metadata.shape[1] + 1,
        hidden_merge_size,
        hidden_ts_size,
        hidden_meta_size,
        num_layers,
        output_size,
        19,
        "intercepts",
    )
    model_no_re = TSNNRE(
        len(selected_ts_features),
        metadata.shape[1] + 1,
        hidden_merge_size,
        hidden_ts_size,
        hidden_meta_size,
        num_layers,
        output_size,
    )

    optimizer_re = optim.Adam(model_re.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_no_re = optim.Adam(model_no_re.parameters(), lr=0.001, weight_decay=1e-4)

    loss_iterations_re_nn = []
    loss_iterations_no_re_nn = []

    for epoch in range(epochs):
        for ts_features, meta_features, y_batch, Z_batch in train_loader:
            for i in range(ts_features.size()[1]):
                # Model with random effects
                optimizer_re.zero_grad()
                predictions_re = model_re(
                    ts_features[:, i, :], meta_features, Z_batch[:, i, :]
                )
                covariance_matrix = model_re.random_effects.get_covariance_matrix()
                nll_re = negative_log_likelihood(
                    y_batch[:, i],
                    predictions_re,
                    Z_batch[:, i, :],
                    covariance_matrix,
                    model_re.random_effects.nr_random_effects,
                    model_re.random_effects.nr_groups,
                )
                nll_re.backward()
                optimizer_re.step()

                # Model without random effects
                optimizer_no_re.zero_grad()
                predictions_no_re = model_no_re(ts_features[:, i, :], meta_features)
                loss_no_re = torch.nn.functional.mse_loss(
                    predictions_no_re.squeeze(1), y_batch[:, i]
                )
                loss_no_re.backward()
                optimizer_no_re.step()

        loss_iterations_re_nn.append(nll_re.item())
        loss_iterations_no_re_nn.append(loss_no_re.item())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")

    # Plot the loss curves for both models
    plt.plot(loss_iterations_re_nn, label="Model with RE")
    plt.plot(loss_iterations_no_re_nn, label="Model without RE")
    plt.legend()
    plt.show()

    with torch.no_grad():
        test_y_list = []
        test_predictions_re_list = []
        test_predictions_no_re_list = []

        for ts_features, meta_features, y_batch, Z_batch in test_loader:
            for i in range(ts_features.size(1)):
                # Model with random effects
                predictions_re = model_re(
                    ts_features[:, i, :], meta_features, Z_batch[:, i, :]
                )
                test_predictions_re_list.append(predictions_re)

                # Model without random effects
                predictions_no_re = model_no_re(ts_features[:, i, :], meta_features)
                test_predictions_no_re_list.append(predictions_no_re)

                # Append the true labels
                test_y_list.append(y_batch[:, i])

        # Concatenate all batches
        test_predictions_re_full = torch.cat(test_predictions_re_list, dim=0)
        test_predictions_no_re_full = torch.cat(test_predictions_no_re_list, dim=0)
        test_y_full = torch.cat(test_y_list, dim=0)

        # Evaluate the models
        evaluate_model(test_predictions_re_full, test_y_full, "TSNN Model with RE")
        evaluate_model(
            test_predictions_no_re_full, test_y_full, "TSNN Model without RE"
        )

        visualise_regression_results(
            test_y_full,
            {
                "RE NN Model": test_predictions_re_full,
                "No RE NN Model": test_predictions_no_re_full,
            },
        )


if __name__ == "__main__":
    main()
