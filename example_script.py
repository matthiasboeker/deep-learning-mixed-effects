import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler  # type: ignore

from models.models import (
    RandomEffectLayer,
    initialize_covariances,
    initialize_variances,
)
from models.loss_functions import negative_log_likelihood
from utils.visualisation_func import (
    visualise_regression_results,
    visualise_longitudinal_data,
)
from utils.evaluation import evaluate_model

# Neural Network Model with Random Effects
class NetWithRE(nn.Module):
    def __init__(self, input_size, output_size, group_number, num_random_effects):
        super(NetWithRE, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.5)
        self.random_effects = RandomEffectLayer(group_number, num_random_effects)

    def forward(self, x, Z):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x + self.random_effects(Z)


# Neural Network Model without Random Effects
class NetWithoutRE(nn.Module):
    def __init__(self, input_size, output_size):
        super(NetWithoutRE, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, group_number, num_random_effects):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.random_effects = RandomEffectLayer(group_number, num_random_effects)

    def forward(self, x, Z):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + self.random_effects(Z)


# Utility Functions
def create_random_effects_design_matrix(n, q, K):
    Z = torch.zeros(n, K * q)
    timesteps = n // q
    time = torch.arange(1, timesteps + 1)
    for i in range(q):
        Z[i * timesteps : (i + 1) * timesteps, i * K] = 1  # intercept
        if K > 1:
            Z[i * timesteps : (i + 1) * timesteps, i * K + 1] = time  # slope
    return Z


def initialize_fixed_random_effects(p, q, K):
    true_beta = torch.randn(p)
    variances_intercept_true = initialize_variances(q)
    variances_slope_true = initialize_variances(q)
    cov_intercept_slope_true = initialize_covariances(q)

    cov_matrices = [
        torch.tensor(
            [[variances_intercept_true[i]]]
            if K == 1
            else [
                [variances_intercept_true[i], cov_intercept_slope_true[i]],
                [cov_intercept_slope_true[i], variances_slope_true[i]],
            ]
        )
        for i in range(q)
    ]
    G_truth = torch.block_diag(*cov_matrices)
    b_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(K * q), covariance_matrix=G_truth
    )
    true_b = b_distribution.sample()

    return true_beta, true_b


def prepare_data(X, Z, y):
    dataset = TensorDataset(X, Z, y)
    return train_test_split(dataset, test_size=0.2, random_state=42)


def scale_data(X, Z):
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.numpy())
    scaler_Z = StandardScaler()
    Z_scaled = scaler_Z.fit_transform(Z.numpy())
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(
        Z_scaled, dtype=torch.float32
    )


# Main Function
def main():
    n, q, K, p = 500, 10, 2, 1
    epochs = 250
    batch_size = 128
    X = torch.randn(n, p)
    Z = create_random_effects_design_matrix(n, q, K)

    # Initialize fixed and random effects
    true_beta, true_b = initialize_fixed_random_effects(p, q, K)

    # Scale and prepare data
    X, Z = scale_data(X, Z)

    y = X @ true_beta + Z @ true_b + 0.1 * torch.randn(n)
    visualise_longitudinal_data(y, n, q)
    train_dataset, test_dataset = prepare_data(X, Z, y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    linear_model = LinearModel(p, 1, q, "slopes")
    re_nn_model = NetWithRE(p, 1, q, "slopes")
    nn_model = NetWithoutRE(p, 1)

    optimizer_linear = optim.Adam(linear_model.parameters(), lr=0.01, weight_decay=1e-6)
    optimizer_re_nn = optim.Adam(re_nn_model.parameters(), lr=0.01, weight_decay=1e-6)
    optimizer_nn = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    loss_iterations_linear = []
    for epoch in range(epochs):
        for X_batch, Z_batch, y_batch in train_loader:
            optimizer_linear.zero_grad()
            predictions = linear_model(X_batch, Z_batch)
            covariance_matrix = linear_model.random_effects.get_covariance_matrix()
            nll = negative_log_likelihood(
                y_batch,
                predictions,
                Z_batch,
                covariance_matrix,
                linear_model.random_effects.nr_random_effects,
                linear_model.random_effects.nr_groups,
            )
            nll.backward()
            optimizer_linear.step()
        loss_iterations_linear.append(nll.item())

    loss_iterations_re_nn = []
    for epoch in range(epochs):
        for X_batch, Z_batch, y_batch in train_loader:
            optimizer_re_nn.zero_grad()
            predictions = re_nn_model(X_batch, Z_batch)
            covariance_matrix = re_nn_model.random_effects.get_covariance_matrix()
            nll = negative_log_likelihood(
                y_batch,
                predictions,
                Z_batch,
                covariance_matrix,
                re_nn_model.random_effects.nr_random_effects,
                re_nn_model.random_effects.nr_groups,
            )
            nll.backward()
            optimizer_re_nn.step()
        loss_iterations_re_nn.append(nll.item())

    loss_iterations_nn = []
    for epoch in range(epochs):
        for X_batch, Z_batch, y_batch in train_loader:
            optimizer_nn.zero_grad()
            predictions = nn_model(X_batch)
            loss = loss_fn(y_batch, predictions.squeeze())
            loss.backward()
            optimizer_nn.step()
        loss_iterations_nn.append(nll.item())

    test_X_list = []
    test_Z_list = []
    test_y_list = []

    with torch.no_grad():
        for X_test, Z_test, y_test in test_loader:
            test_X_list.append(X_test)
            test_Z_list.append(Z_test)
            test_y_list.append(y_test)

        # Concatenate all batches
        test_X_full = torch.cat(test_X_list, dim=0)
        test_Z_full = torch.cat(test_Z_list, dim=0)
        test_y_full = torch.cat(test_y_list, dim=0)

        linear_predictions = linear_model(test_X_full, test_Z_full)
        evaluate_model(linear_predictions, test_y_full, "Linear Model")
        re_nn_predictions = re_nn_model(test_X_full, test_Z_full)
        evaluate_model(re_nn_predictions, test_y_full, "RE NN Model")
        nn_predictions = nn_model(test_X_full)
        evaluate_model(nn_predictions, test_y_full, "NN Model")

        visualise_regression_results(
            test_y_full,
            {
                "Linear Model": linear_predictions,
                "RE NN Model": re_nn_predictions,
                "NN Model": nn_predictions,
            },
        )


if __name__ == "__main__":
    main()
