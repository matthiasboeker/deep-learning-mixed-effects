from typing import List, Tuple
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataloader.ts_dataloader import TSDataset


def train_test_split(dataset: TSDataset, split: float) -> Tuple[TSDataset]:
    dataset_size = len(dataset)
    train_size = int(split * dataset_size)
    test_size = dataset_size - train_size
    return random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )


def custom_collate_fn(batch):
    """
    Custom collate function to pad windows in the batch to ensure all tensors are of the same size
    and handle metadata at the participant level (not window level).
    """
    # Find the max number of windows across all items in the batch
    max_num_windows = max([item[0].size(0) for item in batch])

    # Pad the windows_tensor and targets_tensor for each item in the batch
    padded_windows = []
    padded_targets = []
    metadata_batch = []
    random_intercepts_batch = []

    for windows_tensor, metadata_repeated, targets_tensor, Z_random_intercept in batch:
        # Pad the windows_tensor to the max number of windows
        pad_size = max_num_windows - windows_tensor.size(0)
        if pad_size > 0:
            windows_tensor = torch.nn.functional.pad(
                windows_tensor,
                (0, 0, 0, 0, 0, pad_size),  # Only pad the first dimension (num_windows)
            )
            targets_tensor = torch.nn.functional.pad(targets_tensor, (0, pad_size))
            Z_random_intercept = torch.nn.functional.pad(
                Z_random_intercept, (0, 0, 0, pad_size)
            )

        padded_windows.append(windows_tensor)
        padded_targets.append(targets_tensor)
        random_intercepts_batch.append(Z_random_intercept)
        # Metadata should only be added once per participant, so we just append it without repeating it
        metadata_batch.append(
            metadata_repeated[0]
        )  # Take the first occurrence (same for all windows)

    # Stack the windows and targets tensors to create a batch
    windows_batch = torch.stack(padded_windows)
    targets_batch = torch.stack(padded_targets)
    random_intercepts_batch = torch.stack(random_intercepts_batch)

    # Stack metadata (one entry per participant)
    metadata_batch = torch.stack(metadata_batch)

    return windows_batch, metadata_batch, targets_batch, random_intercepts_batch


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
    return metadata


def main():
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
    selected_ts_features = ["IEMG"]
    selected_targets = [
        "Anxiety level",
        "Fear of falling due to fatigue",
        "Fear of heights",
    ]
    sequence_length = 30
    batch_size = 16

    dataset = TSDataset(
        path_to_ts_folder,
        metadata,
        selected_ts_features,
        selected_targets,
        aggregation_fun=np.sum,
        sequence_length=sequence_length,
    )
    print(len(dataset.file_names))
    train_dataset, test_dataset = train_test_split(dataset, split=0.8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn
    )
    ts_features, meta_features, labels, Z_random_intercept = next(iter(train_loader))
    print(f"Time Series Features: {ts_features.shape}")
    print(f"Metadata: {meta_features.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Random Intercepts Design Matrix: {Z_random_intercept.shape}")
    print(Z_random_intercept[:, :, 0])


if __name__ == "__main__":
    main()
