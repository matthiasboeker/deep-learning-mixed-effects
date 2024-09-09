import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from numpy.polynomial.hermite import hermgauss




class BinaryGLMM(nn.Module):
    def __init__(self, num_features):
        super(BinaryGLMM, self).__init__()
        self.beta = nn.Parameter(torch.randn(1, num_features))  # Parameters for each class

    def forward(self, X, sigma_b, x_m):
        # X: input features [n, num_features]
        # sigma_b: standard deviation of the random effects
        # x_m: nodes from Gauss-Hermite quadrature [M, 1]
        eta = X @ self.beta.T  # [n, K] = [n, num_features] x [num_features, K].T: Neural Network output
        eta = eta + torch.sqrt(torch.tensor(2.0)) * sigma_b * x_m.T  # Add random effect for each node [n, K, M]
        return eta

def gauss_hermite_quadrature(order):
    nodes, weights = hermgauss(order)
    return torch.tensor(nodes, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)

def compute_nll(model, X, y, sigma_b, nodes, weights):
    # X: input features [n, num_features]
    # y: labels [n]
    # nodes: [M]
    # weights: [M]
    nodes, weights = nodes.unsqueeze(1), weights / torch.sqrt(torch.tensor(np.pi))  # Adjust weights and reshape nodes

    # Get model predictions for each node
    eta = model(X, sigma_b, nodes)  # [n, M]
    # Compute log-sum-exp for stability in log-domain integration
    log_probs = y[:, None] * eta - torch.log1p(torch.exp(eta))  # [n, M]
    integral_approx = torch.logsumexp(log_probs + torch.log(torch.tensor(weights, dtype=torch.float32)), dim=1)  # [n]

    return -integral_approx.mean()  # Mean negative log-likelihood over all samples



n = 1000  # Number of samples
p = 1
q = 5
M = 1
X = torch.randn(n, p)
Z = torch.zeros(n, q)
groups = torch.randint(0, q, (n,))
for i in range(q):
    Z[:, i] = (groups == i).float()

# True parameters
true_beta = torch.randn(p)
G_truth = torch.diag(torch.abs(torch.randn(q)))
b_distribution = MultivariateNormal(torch.zeros(q), covariance_matrix=G_truth*2)
true_b = b_distribution.sample()
noise_level = torch.randn(n)
y = X @ true_beta + Z @ true_b + 0.1 * noise_level
y =torch.sigmoid(y)
#plt.scatter(X, y)
#plt.show()
# Data scaling
scaler_X = StandardScaler()
scaler_Z = StandardScaler()
X = torch.tensor(scaler_X.fit_transform(X), dtype=torch.float32)
Z = torch.tensor(scaler_Z.fit_transform(Z), dtype=torch.float32)

# Split the data
X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X.numpy(), Z.numpy(), y.numpy(), test_size=0.2)
X_train, X_test, Z_train, Z_test, y_train, y_test = map(torch.tensor, (X_train, X_test, Z_train, Z_test, y_train, y_test))

sigma_b = nn.Parameter(torch.tensor(0.0))
model = BinaryGLMM(p)
nodes, weights = gauss_hermite_quadrature(M)
optimizer = torch.optim.Adam([sigma_b] + list(model.parameters()), lr=0.01, weight_decay=1e-3)  # Adjust the weight decay parameter as needed

def l1_penalty(param, lambda_l1):
    return lambda_l1 * torch.sum(torch.abs(param))

lambda_l1 = 0.01  # Regularization strength
loss_l = []
for epoch in range(500):
    optimizer.zero_grad()
    loss = compute_nll(model, X_train, y_train, sigma_b, nodes, weights)
    l1_loss = sum(l1_penalty(param, lambda_l1) for param in model.parameters())
    total_loss = loss + l1_loss
    loss.backward()
    optimizer.step()
    loss_l.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Fit logistic regression model
log_reg = LogisticRegression(max_iter=500)
log_reg_y_train = (y_train >= 0.5).numpy()
log_reg.fit(X_train, log_reg_y_train)  # Ensure y_train is appropriate format

# Predict with logistic regression
log_reg_pred = log_reg.predict(X_test)
log_reg_pred_proba = log_reg.predict_proba(X_test)[:, 1]
# Test the model
with torch.no_grad():
    p_test = model(X_test, sigma_b, nodes) 
    mean_probabilities = p_test.mean(dim=1)
    pred_probabilities = torch.sigmoid(mean_probabilities)
    predictions_test = (pred_probabilities >= 0.5).float()
    y_test_label = (y_test >= 0.5).float()
    print(f"Balance: {sum(y_test_label)/len(y_test_label)}")
    accuracy = accuracy_score(y_test_label, predictions_test)
    mcc_nn = matthews_corrcoef(y_test_label, predictions_test)
    print(f"RE Model \n Test Accuracy: {accuracy.item() * 100:.2f}% \n MCC: {mcc_nn  * 100:.2f}% \n")
    accuracy_log = accuracy_score(y_test_label, log_reg_pred)
    mcc_log = matthews_corrcoef(y_test_label, log_reg_pred)
    print(f"Log Regression \n Test Accuracy: {accuracy_log.item() * 100:.2f}% \n MCC: {mcc_log  * 100:.2f}%")
    # Compute the confusion matrix
    cm_log = confusion_matrix(y_test_label, log_reg_pred)
    nn_cm = confusion_matrix(y_test_label, predictions_test)
    
    plt.figure(figsize=(12, 6))  # Wider figure to accommodate both subplots

    # Confusion matrix for Neural Network
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    sns.heatmap(nn_cm, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("NN Confusion Matrix")

    # Confusion matrix for Logistic Regression
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Logistic Regression Confusion Matrix")

    plt.tight_layout()  # Adjust layout to make sure there's no overlap
    plt.show()

    # Compute ROC curve and AUC for the neural network
    nn_fpr, nn_tpr, _ = roc_curve(y_test_label, pred_probabilities)
    nn_auc = auc(nn_fpr, nn_tpr)

    # Compute ROC curve and AUC for logistic regression
    log_reg_fpr, log_reg_tpr, _ = roc_curve(y_test_label, log_reg_pred_proba)
    log_reg_auc = auc(log_reg_fpr, log_reg_tpr)

    # Plot ROC curves
    plt.figure(figsize=(12, 6))  # Wider figure to accommodate both plots

    # ROC curve for Neural Network
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    lw = 2  # Line width
    plt.plot(nn_fpr, nn_tpr, color='darkorange', lw=lw, label='NN ROC curve (area = %0.2f)' % nn_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('NN Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # ROC curve for Logistic Regression
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    lw = 2  # Line width
    plt.plot(log_reg_fpr, log_reg_tpr, color='blue', lw=lw, label='Logistic Regression ROC (area = %0.2f)' % log_reg_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()  # Adjust layout to make sure there's no overlap
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, pred_probabilities, color='blue', alpha=0.5, label='NN with RE')
    plt.scatter(y_test, log_reg_pred_proba, color='red', alpha=0.5, label='Linear Model')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Outcomes')
    plt.ylabel('Predicted Outcomes')
    plt.title('Comparison of Predictions')
    plt.legend()
    plt.show()

