import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.simplefilter('ignore', lineno=149)

# Neural Network without Random Effects
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

# Neural Network with Random Effects
class NetWithRE(nn.Module):
    def __init__(self, input_size, output_size, num_random_effects):
        super(NetWithRE, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.5)
        self.b = nn.Parameter(torch.randn(num_random_effects))
        self.log_cov = nn.Parameter(torch.randn(num_random_effects))
        self.log_sigma_e2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, Z):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x).squeeze()
        G = torch.diag(torch.exp(self.log_cov))
        sigma_e2 = torch.exp(self.log_sigma_e2)  # Convert log-variance back to variance
        V = Z @ G @ Z.T + sigma_e2 * torch.eye(len(Z))
        random_effects = Z @ self.b
        return x + random_effects, V

def run_experiment(p_values, q_values, iterations):
    results = []
    i = 1
    for p in p_values:
        for q in q_values:
            mse_linear_res = []
            R2_res = []
            var_b_res = []
            Distance_b_res = []
            mse_NN_res = []
            R2_NN_res = []
            mse_RE_res = []
            R2_RE_res = []
            var_b_RE_res = []
            Distance_b_RE_res = []

            for i in range(iterations):
                n = 500  # Number of samples
                X = torch.randn(n, p)
                Z = torch.zeros(n, q)
                groups = torch.randint(0, q, (n,))
                for i in range(q):
                    Z[:, i] = (groups == i).float()

                # True parameters
                true_beta = torch.randn(p)
                G_truth = torch.diag(torch.abs(torch.randn(q)))
                b_distribution = MultivariateNormal(torch.zeros(q), covariance_matrix=G_truth)
                true_b = b_distribution.sample()
                y = np.cos(X @ true_beta)**2 + Z @ true_b + 0.1 * torch.randn(n)

                # Data scaling
                scaler_X = StandardScaler()
                scaler_Z = StandardScaler()
                X = torch.tensor(scaler_X.fit_transform(X), dtype=torch.float32)
                Z = torch.tensor(scaler_Z.fit_transform(Z), dtype=torch.float32)

                # Split the data
                X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X.numpy(), Z.numpy(), y.numpy(), test_size=0.2)
                X_train, X_test, Z_train, Z_test, y_train, y_test = map(torch.tensor, (X_train, X_test, Z_train, Z_test, y_train, y_test))

                # Train Linear Model
                beta = torch.randn(p, requires_grad=True)
                b = torch.randn(q, requires_grad=True)
                var_true_b = np.var(true_b.detach().numpy())
                log_vars = torch.randn(q, requires_grad=True)
                optimizer_linear = optim.Adam([beta, log_vars, b], lr=0.01)
                for _ in range(500):
                    optimizer_linear.zero_grad()
                    residuals = y_train - X_train @ beta - Z_train @ b
                    G = torch.diag(torch.exp(log_vars))
                    V = Z_train @ G @ Z_train.T + torch.eye(len(Z_train))
                    V_inv = torch.diag(1 / torch.diag(V))
                    nll = 0.5 * residuals.T @ V_inv @ residuals + 0.5 * torch.logdet(V)
                    nll.backward()
                    optimizer_linear.step()
                beta, b = beta.detach(), b.detach()
                mse_linear = mean_squared_error(y_test.numpy(), (X_test @ beta + Z_test @ b).numpy())
                r2_linear = r2_score(y_test.numpy(), (X_test @ beta + Z_test @ b).numpy())

                # Initialize and Train Neural Network without RE
                net_without_re = NetWithoutRE(input_size=p, output_size=1)
                optimizer_no_re = optim.Adam(net_without_re.parameters(), lr=0.01)
                loss_function = nn.MSELoss()
                for epoch in range(500):
                    net_without_re.train()
                    optimizer_no_re.zero_grad()
                    predictions = net_without_re(X_train)
                    loss = loss_function(predictions, y_train)
                    loss.backward()
                    optimizer_no_re.step()
                net_without_re.eval()
                with torch.no_grad():
                    predictions_no_re = net_without_re(X_test).numpy()
                mse_no_re = mean_squared_error(y_test.numpy(), predictions_no_re)
                r2_no_re = r2_score(y_test.numpy(), predictions_no_re)

                # Initialize and Train Neural Network with RE
                net_with_re = NetWithRE(input_size=p, output_size=1, num_random_effects=q)
                optimizer_re = optim.Adam(net_with_re.parameters(), lr=0.01)
                for epoch in range(500):
                    net_with_re.train()
                    optimizer_re.zero_grad()
                    y_pred, V = net_with_re(X_train, Z_train)
                    residuals = y_train - y_pred
                    V_inv = torch.diag(1 / torch.diag(V))
                    nll = 0.5 * residuals.T @ V_inv @ residuals + 0.5 * torch.logdet(V)
                    nll.backward()
                    optimizer_re.step()
                net_with_re.eval()
                with torch.no_grad():
                    predictions_re, _ = net_with_re(X_test, Z_test)
                mse_re = mean_squared_error(y_test.numpy(), predictions_re.numpy())
                r2_re = r2_score(y_test.numpy(), predictions_re.numpy())
                i = i+1
                mse_linear_res.append(mse_linear)
                R2_res.append(r2_linear)
                var_b_res.append(np.abs(np.var(b.numpy()) - var_true_b))
                Distance_b_res.append(np.linalg.norm(true_b.numpy() - b.numpy()))

                mse_NN_res.append(mse_no_re)
                R2_NN_res.append(r2_no_re)

                mse_RE_res.append(mse_re)
                R2_RE_res.append(r2_re)
                var_b_RE_res.append(np.abs(np.var(net_with_re.b.detach().numpy()) - var_true_b))
                Distance_b_RE_res.append(np.linalg.norm(true_b.numpy() - net_with_re.b.detach().numpy()))
                
                # Store results
            results.append({
                    'p': p,
                    'q': q,
                    'model': 'Linear',
                    'MSE': np.mean(mse_linear_res),
                    'R2': np.mean(R2_res),
                    'Var_b_error': np.mean(var_b_res),
                    'Distance_b': np.mean(Distance_b_res)
                })
            results.append({
                    'p': p,
                    'q': q,
                    'model': 'NN without RE',
                    'MSE': np.mean(mse_NN_res),
                    'R2': np.mean(mse_RE_res),
                    'Var_b_error': None, 
                    'Distance_b': None
                })
            results.append({
                    'p': p,
                    'q': q,
                    'model': 'NN with RE',
                    'MSE': np.mean(mse_RE_res),
                    'R2': np.mean(R2_RE_res),
                    'Var_b_error': np.mean(var_b_RE_res),
                    'Distance_b': np.mean(Distance_b_RE_res)
                })
            print(f"p: {p}, q: {q}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Define parameter ranges and number of iterations
p_values = [1, 5, 10,  20, 30, 40 ,50]
q_values = [1, 5, 10,  20, 30, 40 ,50]
iterations = 30

# Run the experiment
experiment_results = run_experiment(p_values, q_values, iterations)
experiment_results.to_csv("non_linear_results.csv")