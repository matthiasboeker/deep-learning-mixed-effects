import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.preprocessing import StandardScaler

warnings.simplefilter('ignore', lineno=19)

# Negative Log-Likelihood Function
def negative_log_likelihood(y, X, Z, beta, log_vars, b):
    residuals = y - X @ beta - Z @ b
    G = torch.diag(torch.exp(log_vars))
    V = Z @ G @ Z.T + torch.eye(len(Z))
    V_inv = torch.diag(1 / torch.diag(V))
    nll = 0.5 * residuals.T @ V_inv @ residuals + 0.5 * torch.logdet(V)
    return nll

# Function to train Neural Network models
def train_model_NN(model, optimizer, X, y, num_epochs, loss_function):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_function(predictions, y)
        loss.backward()
        optimizer.step()

# Function to train Neural Network models with Random Effects
def train_model_RE(model, optimizer, X, Z, y, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred, V = model(X, Z)
        residuals = y - y_pred
        V_inv = torch.diag(1 / torch.diag(V))
        nll = 0.5 * residuals.T @ V_inv @ residuals + 0.5 * torch.logdet(V)
        nll.backward()
        optimizer.step()

# Linear Prediction Function
def linear_predict(X, Z, beta, b):
    return X @ beta + Z @ b

# Function for Comparison Plot
def comparison_plot(y_true, y_pred_re, y_pred_no_re, y_pred_linear):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred_re, color='blue', alpha=0.5, label='NN with RE')
    plt.scatter(y_true, y_pred_no_re, color='green', alpha=0.5, label='NN without RE')
    plt.scatter(y_true, y_pred_linear, color='red', alpha=0.5, label='Linear Model')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel('Actual Outcomes')
    plt.ylabel('Predicted Outcomes')
    plt.title('Comparison of Predictions')
    plt.legend()
    plt.show()

# Model Evaluation Functions
def evaluate_model_NN(model, name, X, y_true):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{name} - MSE: {mse:.4f}, R^2: {r2:.4f}")

def evaluate_model_RE(model, name, X, Z, y_true):
    model.eval()
    with torch.no_grad():
        y_pred, _ = model(X, Z)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{name} - MSE: {mse:.4f}, R^2: {r2:.4f}")

def evaluate_linear_model(X, Z, beta, b, y_true):
    with torch.no_grad():
        y_pred = linear_predict(X, Z, beta, b)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Linear Model - MSE: {mse:.4f}, R^2: {r2:.4f}")

# Neural Network Model with Random Effects
class NetWithRE(nn.Module):
    def __init__(self, input_size, output_size, num_random_effects):
        super(NetWithRE, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.5)
        self.b = nn.Parameter(torch.randn(num_random_effects))
        self.log_vars = nn.Parameter(torch.randn(num_random_effects))

    def forward(self, x, Z):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x).squeeze()
        G = torch.diag(torch.exp(self.log_vars))
        V = Z @ G @ Z.T + torch.eye(len(Z))
        random_effects = Z @ self.b
        return x + random_effects, V

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
        return self.fc2(x).squeeze()

# Main Function
def main():
    # Data Setup
    n, p, q = 500, 1, 2
    X = torch.randn(n, p)
    Z = torch.zeros(n, q)
    groups = torch.randint(0, 2, (n,))

    Z = torch.ones((n, q))
    Z[:, 0] = torch.where(groups == 0, 0, 1)*0.5
    Z[:, 1] = X[:, 0] * torch.where(groups == 0, -1, 1)  # Negative slope for group 0, positive for group 1
    #Z[:, 2] = X[:, p-1] * torch.where(groups == 1, -1.5, 1.5)

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

    true_beta = torch.randn(p)
    G_truth = torch.abs(torch.diag(torch.randn(q)))
    b_distribution = torch.distributions.MultivariateNormal(torch.zeros(q), covariance_matrix=G_truth)
    true_b = b_distribution.sample()
    y = X @ true_beta+ Z @ true_b + 0.1 * torch.randn(n)
    #plt.scatter(X, y)
    #plt.plot()

    # Split the data
    X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X.numpy(), Z.numpy(), y.numpy(), test_size=0.2, random_state=42)
    X_train, X_test, Z_train, Z_test, y_train, y_test = map(torch.tensor, (X_train, X_test, Z_train, Z_test, y_train, y_test))

    # Linear Model Training with Negative Log-Likelihood
    beta = torch.randn(p, requires_grad=True)
    b = torch.randn(q, requires_grad=True)
    log_vars = torch.randn(q, requires_grad=True)
    optimizer_linear = optim.Adam([beta, log_vars, b], lr=0.01, weight_decay=1e-4)

    for iteration in range(500):
        optimizer_linear.zero_grad()
        nll = negative_log_likelihood(y_train, X_train, Z_train, beta, log_vars, b)
        nll.backward()
        optimizer_linear.step()

    beta = beta.detach().numpy()
    G_final = torch.diag(torch.exp(log_vars))
    G_linear = G_final.detach().numpy()
    b = b.detach().numpy()

    # Initialize and Train Neural Network with RE
    net_with_re = NetWithRE(input_size=p, output_size=1, num_random_effects=q)
    optimizer_re = optim.Adam(net_with_re.parameters(), lr=0.01, weight_decay=1e-4)
    train_model_RE(net_with_re, optimizer_re, X_train, Z_train, y_train, 500)

    # Initialize and Train Neural Network without RE
    net_without_re = NetWithoutRE(input_size=p, output_size=1)
    optimizer_no_re = optim.Adam(net_without_re.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    train_model_NN(net_without_re, optimizer_no_re, X_train, y_train, 500, loss_function)

    # Print Random Effects and Covariance Matrix Comparisons
    print("\nComparison of Random Effects and Covariance Matrices:")
    print("Ground Truth b:", true_b.numpy())
    print("Linear Model b:", b)
    print("NN with RE b estimate:", net_with_re.b.detach().numpy())

    print("\nGround Truth Covariance Matrix G:")
    print(G_truth)
    print("Linear Model G: \n", G_linear)
    print("NN with RE G estimate: \n", torch.diag(torch.exp(net_with_re.log_vars)).detach().numpy())

    # Evaluate Models
    evaluate_model_RE(net_with_re, "NN with RE", X_test, Z_test, y_test)
    evaluate_model_NN(net_without_re, "NN without RE", X_test, y_test)
    evaluate_linear_model(X_test, Z_test, beta, b, y_test)

    # Comparison Plot
    predictions, _ = net_with_re(X_test, Z_test)
    comparison_plot(y_test, predictions.detach().numpy(), net_without_re(X_test).detach().numpy(), linear_predict(X_test, Z_test, beta, b).detach().numpy())

if __name__ == "__main__":
    main()
