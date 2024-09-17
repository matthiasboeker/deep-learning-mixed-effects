from pathlib import Path
import re
from typing import Callable, List, Tuple
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def empty_aggregation(targets: pd.Series):
    if isinstance(targets, pd.DataFrame):
        raise ValueError("Select single target or change aggregration function")
    return targets


def preprocess_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the metadata dataframe.
    Male: 0 Female: 1
    TODO: STAI SCORING"""

    metadata["Sex"] = metadata["Sex"].apply(lambda x: 0 if x == "Male" else 1)
    metadata["Approximate number climbing"] = metadata[
        [
            "Approximate the number of hours that you spend top-rope climbing per week",
            "Approximate the number of hours that you spend lead climbing per week.",
        ]
    ].sum(axis=0)
    metadata["bmi"] = metadata.apply(
        lambda x: x["Body mass (kg)"] / (x["Height (cm)"] / 10) ** 2
    )
    return metadata


def get_ts_file_names(path_to_ts_folder: Path) -> List[str]:
    return list(filter(lambda x: ".csv" in x, os.listdir(path_to_ts_folder)))


def read_in_meta_data(path_to_meta_data: Path) -> pd.DataFrame:
    return pd.read_csv(path_to_meta_data, engine="python")


def get_participant_id(file_name: str):
    participant_id, *rest = re.findall(r"\d+", file_name)
    return participant_id


def create_windows(data: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    Create sliding windows of fixed size from time series data.

    Args:
        data (torch.Tensor): The time series data, shape [num_timesteps, num_features].
        sequence_length (int): The desired window length.

    Returns:
        torch.Tensor: The windows, shape [num_windows, sequence_length, num_features].
    """
    # If data is shorter than sequence_length, pad the data
    if data.size(0) < sequence_length:
        pad_size = sequence_length - data.size(0)
        data = torch.nn.functional.pad(
            data, (0, 0, 0, pad_size)
        )  # Pad along the time axis

    windows = data.unfold(0, sequence_length, sequence_length)
    windows = windows.permute(0, 2, 1).contiguous()
    # Return the windows as a tensor [num_windows, sequence_length, num_features]
    return windows


class TSDataset(Dataset):
    def __init__(
        self,
        path_to_ts_folder: Path,
        path_to_meta_data: Path,
        sequence_length: int,
        selected_features: List[str],
        selected_target: List[str],
        aggregation_fun: Callable = empty_aggregation,
    ):
        self.metadata = read_in_meta_data(path_to_meta_data)
        self.path_to_ts_folder = path_to_ts_folder
        self.sequence_length = sequence_length
        self.file_names = get_ts_file_names(path_to_ts_folder)
        self.selected_features = selected_features
        self.selected_target = selected_target
        self.aggregation_fun = aggregation_fun

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """Anxiety level,Fear of falling due to fatigue,Fear of heights,Pumped,quarters"""
        file_name = self.file_names[idx]
        participant_id = get_participant_id(file_name)
        metadata_row = self.metadata[
            self.metadata["Participant ID"] == participant_id
        ].copy()
        metadata = metadata_row[
            [
                "Age",
                "Sex",
                "bmi",
                "Climbing experience (years)",
                "Bouldering experience (years)",
                "Approximate number climbing",
                "Approximate the number of hours that you spend bouldering per week",
            ]
        ].values
        metadata["type"] = 1 if "toprope" in file_name else 0
        metadata = torch.tensor(metadata, dtype=torch.float32)
        time_series_data = pd.read_csv(self.path_to_ts_folder / file_name)
        time_series_features = torch.tensor(
            time_series_data[self.selected_features].values, dtype=torch.float32
        )
        time_series_targets = torch.tensor(
            self.aggregation_fun(time_series_data[self.selected_target], dim=0),
            dtype=torch.float32,
        )

        windows_tensor = create_windows(
            time_series_features, self.sequence_length
        )  # Shape: [num_windows, sequence_length, num_features]
        targets_tensor = create_windows(time_series_targets, self.sequence_length)
        metadata_repeated = metadata.unsqueeze(0).repeat(
            windows_tensor.size(0), 1
        )  # Repeat metadata for each window

        return windows_tensor, metadata_repeated, targets_tensor
