import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from models.models import (
    RandomEffectLayer,
    initialize_covariances,
    initialize_variances,
)
from models.loss_functions import negative_log_likelihood
from utils.visualisation_func import visualise_regression_results

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


def evaluate_model(predictions, y_true, model_name):
    with torch.no_grad():
        y_pred = predictions
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{model_name} - MSE: {mse:.4f}, R^2: {r2:.4f}")


# Main Function
def main():
    # Data Setup
    n = 500
    q = 5
    K = 2
    p = 1
    batch_size = 128
    X = torch.randn(n, p)
    timesteps = n // q

    # Create Z matrix for random intercepts and slopes
    Z = torch.zeros(n, K * q)
    time = torch.arange(1, timesteps + 1)
    for i in range(q):
        Z[i * timesteps : (i + 1) * timesteps, i * K] = 1  # intercept for subject i
        Z[i * timesteps : (i + 1) * timesteps, i * K + 1] = time  # slope for subject i

    # True coefficients for fixed effects and random effects
    true_beta = torch.randn(p)
    variances_true = initialize_variances(q)
    cov_intercept_slope_true = initialize_covariances(q)

    # Verify initialization
    cov_matrices = [
        torch.tensor(
            [
                [variances_true["variances_intercept"][i], cov_intercept_slope_true[i]],
                [cov_intercept_slope_true[i], variances_true["variances_slopes"][i]],
            ]
        )
        for i in range(q)
    ]

    for i, cov in enumerate(cov_matrices):
        eigenvalues = torch.linalg.eigvals(cov)
        if torch.any(eigenvalues.real < 0):
            print(f"Matrix {i} is not positive semi-definite")

    G_truth = torch.block_diag(*cov_matrices)
    b_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(K * q), covariance_matrix=G_truth
    )
    true_b = b_distribution.sample()

    # Convert to numpy for scaling purposes
    X_np = X.numpy()
    Z_np = Z.numpy()

    # Applying standard scaling
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_np)

    scaler_Z = StandardScaler()
    Z_scaled = scaler_Z.fit_transform(Z_np)

    # Convert back to tensors
    X = torch.tensor(X_scaled, dtype=torch.float32)
    Z = torch.tensor(Z_scaled, dtype=torch.float32)

    y = X @ true_beta + Z @ true_b + 0.1 * torch.randn(n)

    dataset = TensorDataset(X, Z, y)
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    linear_model = LinearModel(p, 1, q, K)
    re_nn_model = NetWithRE(p, 1, q, K)
    nn_model = NetWithoutRE(p, 1)

    optimizer_linear = optim.Adam(linear_model.parameters(), lr=0.01, weight_decay=1e-6)
    optimizer_re_nn = optim.Adam(re_nn_model.parameters(), lr=0.01, weight_decay=1e-6)
    optimizer_nn = optim.Adam(nn_model.parameters(), lr=0.01, weight_decay=1e-6)

    loss_iterations_linear = []
    for epoch in range(250):
        for X_batch, Z_batch, y_batch in train_loader:
            optimizer_linear.zero_grad()
            predictions = linear_model(X_batch, Z_batch)
            nll = negative_log_likelihood(
                y_batch, predictions, Z_batch, linear_model.named_parameters()
            )
            nll.backward()
            optimizer_linear.step()
        loss_iterations_linear.append(nll.item())

    loss_iterations_re_nn = []
    for epoch in range(250):
        for X_batch, Z_batch, y_batch in train_loader:
            optimizer_re_nn.zero_grad()
            predictions = re_nn_model(X_batch, Z_batch)
            nll = negative_log_likelihood(
                y_batch, predictions, Z_batch, linear_model.named_parameters()
            )
            nll.backward()
            optimizer_re_nn.step()
        loss_iterations_re_nn.append(nll.item())

    loss_iterations_nn = []
    for epoch in range(250):
        for X_batch, Z_batch, y_batch in train_loader:
            optimizer_nn.zero_grad()
            predictions = nn_model(X_batch)
            nll = negative_log_likelihood(
                y_batch, predictions, Z_batch, linear_model.named_parameters()
            )
            nll.backward()
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
