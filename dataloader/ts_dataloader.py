from pathlib import Path
import re
from typing import Callable, List
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


def create_random_intercepts_design_matrix(n: int, q: int) -> torch.Tensor:
    Z = torch.zeros(n, q)
    Z[:, 0] = 1  # Random intercept for each participant (1 for all windows)
    return Z


def empty_aggregation(targets: pd.Series):
    if isinstance(targets, pd.DataFrame):
        raise ValueError("Select single target or change aggregration function")
    return targets


def get_ts_file_names(path_to_ts_folder: Path) -> List[str]:
    return list(filter(lambda x: ".csv" in x, os.listdir(path_to_ts_folder)))


def get_participant_id(file_name: str):
    participant_id, *rest = re.findall(r"\d+", file_name)
    return participant_id


def get_participant_ids(file_names: List[str]) -> List[str]:
    participant_ids = list(
        set([re.findall(r"\d+", file_name)[0] for file_name in file_names])
    )
    return participant_ids


def create_ts_windows(data: torch.Tensor, sequence_length: int) -> torch.Tensor:
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


def create_target_windows(
    targets: torch.Tensor, sequence_length: int, reduction: str = "majority"
) -> torch.Tensor:
    """
    Create windows of target values where one target is generated per window (e.g., by taking the mean).

    Args:
        targets (torch.Tensor): The target values, shape [num_timesteps].
        sequence_length (int): The length of each window.
        reduction (str): The reduction method to apply to the targets per window ('mean', 'last', 'sum').

    Returns:
        torch.Tensor: A tensor of shape [num_windows], where each element is the reduced target for that window.
    """
    num_timesteps = targets.size(0)

    # If targets are shorter than the sequence length, pad with zeros
    if num_timesteps < sequence_length:
        pad_size = sequence_length - num_timesteps
        targets = torch.nn.functional.pad(targets, (0, pad_size))
    # Reshape the targets into windows (this will reshape to [num_windows, sequence_length])
    target_windows = targets.unfold(0, sequence_length, sequence_length)
    # Apply reduction (e.g., mean, last value, or sum)
    if reduction == "mean":
        target_windows = target_windows.mean(dim=1)
    elif reduction == "sum":
        target_windows = target_windows.sum(dim=1)
    elif reduction == "last":
        target_windows = target_windows[:, -1]
    elif reduction == "majority":
        target_windows, _ = torch.mode(target_windows, dim=1)
    else:
        raise ValueError(f"Unsupported reduction method: {reduction}")

    return target_windows


class TSDataset(Dataset):
    def __init__(
        self,
        path_to_ts_folder: Path,
        meta_data: pd.DataFrame,
        selected_ts_features: List[str],
        selected_target: List[str],
        sequence_length: int,
        aggregation_fun: Callable = empty_aggregation,
    ):
        self.metadata = meta_data
        self.path_to_ts_folder = path_to_ts_folder
        self.sequence_length = sequence_length
        self.file_names = get_ts_file_names(path_to_ts_folder)
        self.participants_ids = get_participant_ids(self.file_names)
        self.selected_ts_features = selected_ts_features
        self.selected_target = selected_target
        self.aggregation_fun = aggregation_fun
        self.participant_id_to_idx = {
            pid: idx for idx, pid in enumerate(self.participants_ids)
        }

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """Anxiety level,Fear of falling due to fatigue,Fear of heights,Pumped,quarters"""
        file_name = self.file_names[idx]
        participant_id = get_participant_id(file_name)
        metadata_row = self.metadata[
            self.metadata["Participant ID"] == int(participant_id)
        ].copy()
        metadata_row["type"] = 1 if "toprope" in file_name else 0
        metadata = torch.tensor(metadata_row.values, dtype=torch.float32)
        time_series_data = pd.read_csv(self.path_to_ts_folder / file_name)
        time_series_features = torch.tensor(
            time_series_data[self.selected_ts_features].values, dtype=torch.float32
        )
        time_series_targets = torch.tensor(
            self.aggregation_fun(time_series_data[self.selected_target], axis=1),
            dtype=torch.float32,
        )

        windows_tensor = create_ts_windows(
            time_series_features, self.sequence_length
        )  # Shape: [num_windows, sequence_length, num_features]
        targets_tensor = create_target_windows(
            time_series_targets, self.sequence_length
        )
        metadata_repeated = metadata.repeat(
            windows_tensor.size(0), 1
        )  # Repeat metadata for each window
        Z_dynamic = self.create_random_intercepts_design_matrix(
            [participant_id], self.participants_ids, windows_tensor.size(0)
        )

        return windows_tensor, metadata_repeated, targets_tensor, Z_dynamic

    def create_random_intercepts_design_matrix(
        self,
        batch_participants: List[int],
        all_participants: List[int],
        num_windows: int,
    ) -> torch.Tensor:
        """
        Create the design matrix for random intercepts.

        Args:
            batch_participants (List[int]): List of participant IDs in the current batch.
            all_participants (List[int]): List of all participant IDs in the dataset.
            num_windows (int): Number of windows (repeated rows for each participant).

        Returns:
            torch.Tensor: Design matrix of size [num_windows, num_participants].
        """
        num_participants = len(all_participants)

        # Create a design matrix of size [num_windows, num_participants]
        Z_batch = torch.zeros(num_windows, num_participants)

        # Use the mapping to find the correct column in Z_batch
        for participant_id in batch_participants:
            participant_idx = self.participant_id_to_idx[str(participant_id)]
            Z_batch[
                :, participant_idx
            ] = 1  # Set the column corresponding to the participant to 1

        return Z_batch
