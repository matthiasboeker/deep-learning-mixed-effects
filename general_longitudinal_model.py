import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import warnings
from sklearn.preprocessing import StandardScaler
from matplotlib import cm

warnings.simplefilter('ignore')

def visualise_longitudinal_data(y, time, timesteps, q):
    colors = cm.get_cmap('tab20', q)  # Get q colors from the 'tab20' colormap
    # Plot each subject's data over time
    plt.figure(figsize=(10, 6))
    for i in range(q):
        plt.plot(time.numpy(), y[i*timesteps:(i+1)*timesteps].numpy(), marker='o', color=colors(i), label=f'Subject {i+1}')

    plt.xlabel('Time (Time Steps)')
    plt.ylabel('Observed Output (y)')
    plt.title('Random Intercepts and Slopes for Each Subject Over Time')
    plt.show()

def softplus_param(initial_tensor):
    return nn.Parameter(nn.functional.softplus(initial_tensor))

def initialize_parameters(K, q):
    # Initialize variance for each effect
    variances = [np.abs(torch.randn(q) * 0.5) for _ in range(K)]
    # Initialize covariance terms between every pair of effects
    covariances = [[torch.randn(q) for _ in range(K)] for _ in range(K)]
    return variances, covariances

def assemble_G(variances, covariances, K, q):
    # Initialize the large G matrix
    G = torch.zeros(K * q, K * q)

    # Construct the matrix block by block
    for i in range(K):
        for j in range(K):
            start_i = i * q
            end_i = start_i + q
            start_j = j * q
            end_j = start_j + q
            
            if i == j:  # Diagonal blocks
                G[start_i:end_i, start_j:end_j] = torch.diag_embed(variances[i])
            else:  # Off-diagonal blocks
                # Ensure covariance is symmetric and multiplied by sqrt(var_i * var_j)
                cov_ij = torch.sqrt(variances[i] * variances[j]) * covariances[i][j]
                cov_ji = torch.sqrt(variances[i] * variances[j]) * covariances[j][i]
                # Symmetrize the covariance blocks
                G[start_i:end_i, start_j:end_j] = torch.diag_embed(0.5 * (cov_ij + cov_ji))
    return G

def create_Z_matrix(n, q, K, timesteps):
    Z = torch.zeros(n, q * K)
    time = torch.arange(1, timesteps + 1).float()
    for i in range(q):
        for k in range(K):
            Z[i*timesteps:(i+1)*timesteps, i*K+k] = time.pow(k)
    return Z, time

def negative_log_likelihood(y, X, Z, beta, cov_matrices, b, q, K):
    residuals = y - X @ beta - Z @ b
    residuals = residuals.unsqueeze(1)
    G = cov_matrices
    V = Z @ G @ Z.transpose(0, 1) + torch.eye(len(Z))
    
    try:
        V_inv = torch.inverse(V)
    except RuntimeError:
        U, S, Vh = torch.linalg.svd(V)
        S_inv = torch.diag(1.0 / S)
        V_inv = Vh.T @ S_inv @ U.T
    _, logdet_V = torch.slogdet(V)
    nll = 0.5 * residuals.transpose(0, 1) @ V_inv @ residuals + 0.5 * torch.abs(logdet_V)
    return nll

def linear_predict(X, Z, beta, b):
    return X @ beta + Z @ b

def evaluate_linear_model(X, Z, beta, b, y_true):
    y_pred = linear_predict(X, Z, beta, b)
    mse = mean_squared_error(y_true, y_pred.detach().numpy())
    r2 = r2_score(y_true, y_pred.detach().numpy())
    print(f"Linear Model - MSE: {mse:.4f}, R^2: {r2:.4f}")

def main():
    n = 500
    q = 50
    K = 2
    p = 1
    batch_size = 16
    timesteps = n // q
    X = torch.randn(n, p)
    Z, time = create_Z_matrix(n, q, K, timesteps)
    true_beta = torch.randn(p)
    var_effects, cov_params = initialize_parameters(K, q)
    true_b = torch.randn(q * K)
    # Initialize standard scaler
    scaler_X = StandardScaler()
    scaler_Z = StandardScaler()

    # Scale X and Z
    X_np = X.numpy()
    Z_np = Z.numpy()
    X_scaled_np = scaler_X.fit_transform(X_np)
    Z_scaled_np = scaler_Z.fit_transform(Z_np)

    # Convert back to tensors
    X_scaled = torch.tensor(X_scaled_np, dtype=torch.float32)
    Z_scaled = torch.tensor(Z_scaled_np, dtype=torch.float32)

    y = X_scaled @ true_beta + Z_scaled @ true_b + 0.1 * torch.randn(n)

    visualise_longitudinal_data(y, time, timesteps, q)
    dataset = TensorDataset(X, Z, y)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    beta = torch.randn(p, requires_grad=True)
    b = torch.randn(q * K, requires_grad=True)
    optimizer = optim.Adam([beta, b] +var_effects + [element for sublist in cov_params for element in sublist], lr=0.01, weight_decay=1e-6)

    loss_nl = []
    for epoch in range(500):
        loss_avg = []
        for X_batch, Z_batch, y_batch in train_loader:
            optimizer.zero_grad()
            G = assemble_G(var_effects, cov_params, K, q)
            nll = negative_log_likelihood(y_batch, X_batch, Z_batch, beta, G, b, q, K)
            nll.backward()
            optimizer.step()
            loss_avg.append(nll.item())
        loss_nl.append(np.mean(loss_avg))
        if epoch%10 == 0:
            print(f"Epoch: {epoch}")
    plt.plot(loss_nl)
    plt.show()

    beta = beta.detach().numpy()
    b = b.detach().numpy()
    print(f"True beta: {true_beta} \n Estiamted beta: {beta} \n")
    print(f"True b: {true_b} \n Estiamted b: {b} \n")
    print(f"Var b: {true_b.detach().numpy().var()} \n Estiamted b: {b.var()} \n")

    test_X_list = []
    test_Z_list = []
    test_y_list = []
    with torch.no_grad():
        for X_test, Z_test, y_test in test_loader:
            test_X_list.append(X_test)
            test_Z_list.append(Z_test)
            test_y_list.append(y_test)
        test_X_full = torch.cat(test_X_list)
        test_Z_full = torch.cat(test_Z_list)
        test_y_full = torch.cat(test_y_list)
        evaluate_linear_model(test_X_full, test_Z_full, beta, b, test_y_full)

    plt.figure(figsize=(12, 6))
    plt.scatter(test_y_full, linear_predict(test_X_full, test_Z_full, beta, b).detach().numpy(), color='red', alpha=0.5, label='Linear Model')
    plt.plot([test_y_full.min(), test_y_full.max()], [test_y_full.min(), test_y_full.max()], 'k--')
    plt.xlabel('Actual Outcomes')
    plt.ylabel('Predicted Outcomes')
    plt.title('Comparison of Predictions')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
