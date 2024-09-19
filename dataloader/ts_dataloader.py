from pathlib import Path
import re
from typing import Callable, List
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


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
        quarters = torch.tensor(
            time_series_data["quarters"].values, dtype=torch.float32
        )
        ts_scaler = StandardScaler()
        time_series_features = ts_scaler.fit_transform(
            time_series_data[self.selected_ts_features]
        )
        time_series_features = torch.tensor(time_series_features, dtype=torch.float32)
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
        # Update the random intercept and slopes matrix
        Z_dynamic = self.create_random_slopes_design_matrix(
            [participant_id], self.participants_ids, windows_tensor.size(0), quarters
        )
        return windows_tensor, metadata_repeated, targets_tensor, Z_dynamic

    def create_random_slopes_design_matrix(
        self,
        batch_participants: List[str],
        all_participants: List[str],
        num_windows: int,
        quarters: torch.Tensor,
    ) -> torch.Tensor:
        num_participants = len(all_participants)
        Z_batch = torch.zeros(num_windows, num_participants * 2)

        participant_indices = torch.tensor(
            [
                self.participant_id_to_idx[participant_id]
                for participant_id in batch_participants
            ]
        )

        # For each window, set the appropriate column to 1 for the intercept and quarters for the slope
        for i, participant_idx in enumerate(participant_indices):
            # Random intercept column
            Z_batch[:, participant_idx] = 1
            Z_batch[:, num_participants + participant_idx] = quarters[i]
        return Z_batch
