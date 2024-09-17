import torch
from dataloader.ts_dataloader import create_windows


def test_create_windows_no_padding():
    """Test that the function splits the time series into the correct number of windows with no padding."""
    sequence_length = 5
    num_timesteps = 15
    num_features = 3

    # Create a dummy time series with 15 timesteps and 3 features
    time_series_data = torch.randn(num_timesteps, num_features)

    # Call the function
    windows = create_windows(time_series_data, sequence_length)

    # Check the shape of the output (should be [num_windows, sequence_length, num_features])
    print(windows)
    assert windows.shape == (
        3,
        sequence_length,
        num_features,
    )  # 15 timesteps / 5 sequence_length = 3 windows


def test_create_windows_with_padding():
    """Test that the function correctly pads the time series if it is shorter than the sequence length."""
    sequence_length = 10
    num_timesteps = 7  # Less than the sequence length
    num_features = 3

    # Create a dummy time series with 7 timesteps and 3 features
    time_series_data = torch.randn(num_timesteps, num_features)

    # Call the function
    windows = create_windows(time_series_data, sequence_length)

    # Check that the shape of the output is [1, sequence_length, num_features]
    assert windows.shape == (1, sequence_length, num_features)

    # Ensure that padding has been applied (the padded values should be zero)
    assert torch.all(windows[0, num_timesteps:, :] == 0)  # The padding should be zeros


def test_create_windows_exact_sequence_length():
    """Test that the function works correctly when the number of timesteps equals the sequence length."""
    sequence_length = 5
    num_timesteps = 5  # Equal to the sequence length
    num_features = 3

    # Create a dummy time series with exactly 5 timesteps and 3 features
    time_series_data = torch.randn(num_timesteps, num_features)

    # Call the function
    windows = create_windows(time_series_data, sequence_length)

    # Check that the shape of the output is [1, sequence_length, num_features]
    assert windows.shape == (1, sequence_length, num_features)


def test_create_windows_partial_window():
    """Test the case where the data has a partial window that does not fit perfectly into sequence_length."""
    sequence_length = 4
    num_timesteps = 10  # 2 full windows and 1 partial window
    num_features = 3

    # Create a dummy time series with 10 timesteps and 3 features
    time_series_data = torch.randn(num_timesteps, num_features)

    # Call the function
    windows = create_windows(time_series_data, sequence_length)

    # Check the shape of the output (should be [2, sequence_length, num_features])
    assert windows.shape == (
        2,
        sequence_length,
        num_features,
    )  # Only full windows are returned
