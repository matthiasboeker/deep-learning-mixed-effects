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
warnings.simplefilter('ignore', lineno=52)

def safe_exp(x):
    clamped_x = torch.clamp(x, min=-10, max=10)  # adjust bounds as needed
    return torch.exp(clamped_x)

def softplus_param(initial_tensor):
    return nn.functional.softplus(initial_tensor)

# Ensuring positive definite covariance matrices for initialization
def initialize_parameters(q):
    # Initialize variances as positive by using the exponential of normally distributed numbers
    var_intercepts = softplus_param(torch.randn(q) * 0.5)  # Scale down to ensure not too large
    var_slopes = softplus_param(torch.randn(q) * 0.5)
    # Initialize covariance with a tighter bound
    cov_intercept_slope = torch.randn(q) * 0.1  # Smaller initial values
    for i in range(q):
        max_cov = torch.sqrt(var_intercepts[i] * var_slopes[i]) * 0.9  # 90% of the maximum allowable value
        cov_intercept_slope[i] = torch.abs(cov_intercept_slope[i]+1e-5)

    return var_intercepts, var_slopes, cov_intercept_slope

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

def negative_log_likelihood(y, X, Z, beta, log_var_intercepts, log_var_slopes, cov_intercept_slope, b, q):
    residuals = y - X @ beta - Z @ b
    residuals = residuals.unsqueeze(1)
    var_intercept = log_var_intercepts
    var_slope = log_var_slopes
    if torch.any(torch.isnan(var_intercept)) or torch.any(torch.isinf(var_slope)):
        print("Stability issues detected in var_intercepts")
    block_a = torch.diag_embed(var_intercept)
    cov_intercept_slope = cov_intercept_slope * log_var_slopes * log_var_intercepts
    block_b = torch.diag_embed(cov_intercept_slope)
    block_c = torch.diag_embed(var_slope)

    upper = torch.cat([block_a, block_b], dim=1)
    lower = torch.cat([block_b.transpose(0, 1), block_c], dim=1)
    G = torch.cat([upper, lower], dim=0)
    regularization_strength = 1e-4 * torch.max(torch.abs(G)).item()
    G +=  regularization_strength   * torch.eye(G.size(0))
    V = Z @ G @ Z.transpose(0, 1) + torch.eye(len(Z))

    regularization_strength = 1e-4 * torch.max(torch.abs(V)).item()
    V +=  regularization_strength   * torch.eye(V.size(0))
    # Try Cholesky decomposition, fallback to SVD if it fails
    try:
        #chol = torch.linalg.cholesky(V)
        V_inv = torch.inverse(V)
    except RuntimeError as e:
        print("Cholesky failed, falling back to SVD: ", e)
        U, S, Vh = torch.linalg.svd(V)
        S_inv = torch.diag(1.0 / S)
        V_inv = Vh.T @ S_inv @ U.T
    _, logdet_V = torch.slogdet(V)
    nll = 0.5 * residuals.transpose(0, 1) @ V_inv @ residuals + 0.5 * torch.abs(logdet_V)
    return nll

# Linear Prediction Function
def linear_predict(X, Z, beta, b):
    return X @ beta + Z @ b

def evaluate_linear_model(X, Z, beta, b, y_true):
    with torch.no_grad():
        y_pred = linear_predict(X, Z, beta, b)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Linear Model - MSE: {mse:.4f}, R^2: {r2:.4f}")

# Main Function
def main():
    # Data Setup
    n = 500
    q = 3
    K = 2
    p = 1
    batch_size = 128  
    X = torch.randn(n, p)
    timesteps = n // q
    # Create Z matrix for random intercepts and slopes
    Z = torch.zeros(n, K*q)
    time = torch.arange(1, timesteps + 1)
    for i in range(q):
        Z[i*timesteps:(i+1)*timesteps, i*K] = 1  # intercept for subject i
        Z[i*timesteps:(i+1)*timesteps, i*K+1] = time  # slope for subject i

    # True coefficients for fixed effects and random effects
    true_beta = torch.randn(p)
    var_intercepts_true, var_slopes_true, cov_intercept_slope_true = initialize_parameters(q)

    # Verify initialization
    cov_matrices = [torch.tensor([[var_intercepts_true[i], cov_intercept_slope_true[i]], 
                                [cov_intercept_slope_true[i], var_slopes_true[i]]]) for i in range(q)]
    
    for i, cov in enumerate(cov_matrices):
        eigenvalues = torch.linalg.eigvals(cov)
        if torch.any(eigenvalues.real < 0):
            print(f"Matrix {i} is not positive semi-definite")

    G_truth = torch.block_diag(*cov_matrices)
    b_distribution = torch.distributions.MultivariateNormal(torch.zeros(K*q), covariance_matrix=G_truth)
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

    y = X @ true_beta+ Z @ true_b + 0.1 * torch.randn(n)
    # Plot each subject's data over time
    #visualise_longitudinal_data(y, time, timesteps, q)

    dataset = TensorDataset(X, Z, y)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #X_train, X_test, Z_train, Z_test, y_train, y_test = map(torch.tensor, (X_train, X_test, Z_train, Z_test, y_train, y_test))

    # Linear Model Training with Negative Log-Likelihood
    beta = torch.randn(p, requires_grad=True)
    b = torch.randn(K*q, requires_grad=True)
    var_intercepts, var_slopes, cov_intercept_slope = initialize_parameters(q)
    var_intercepts.requires_grad = True
    var_slopes.requires_grad = True
    cov_intercept_slope.requires_grad = True

    optimizer_linear = optim.Adam([beta, var_intercepts, var_slopes, cov_intercept_slope, b], lr=0.01, weight_decay=1e-6)
    
    loss_iterations = []
    for epoch in range(250):
        for X_batch, Z_batch, y_batch in train_loader:
            optimizer_linear.zero_grad()
            nll = negative_log_likelihood(y_batch, X_batch, Z_batch, beta, var_intercepts, var_slopes, cov_intercept_slope, b, q)
            nll.backward()
            optimizer_linear.step()
        loss_iterations.append(nll.item())
    plt.plot(loss_iterations)
    plt.show()


    beta = beta.detach().numpy()
    b = b.detach().numpy()
    print(f"True beta: {true_beta} \n Estiamted beta: {beta} \n")
    print(f"True beta: {true_b} \n Estiamted beta: {b} \n")


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

        # Evaluate the model on the full test dataset
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
