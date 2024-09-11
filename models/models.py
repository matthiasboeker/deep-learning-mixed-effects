from typing import Dict
import torch
import torch.nn as nn


def initialize_variances(q: int) -> Dict[str, torch.tensor]:
    variances_intercepts = nn.functional.softplus(torch.randn(q) * 0.5)
    variances_slopes = nn.functional.softplus(torch.randn(q) * 0.5)
    return {
        "intercept_variance": nn.Parameter(variances_intercepts),
        "slopes_variances": nn.Parameter(variances_slopes),
    }


def initialize_covariances(q: int) -> Dict[str, torch.tensor]:
    cov_intercept_slope = torch.randn(q) * 0.1  # Smaller initial values
    for i in range(q):
        cov_intercept_slope[i] = torch.abs(cov_intercept_slope[i] + 1e-5)
    return nn.Parameter(cov_intercept_slope)


class RandomEffectLayer(nn.Module):
    def __init__(self, input_size, output_size, groups):
        super(self).__init__()
        variances = initialize_variances(groups)
        covariances = initialize_covariances(groups)
        self.variances_intercept = variances["intercept_variance"]
        self.variances_slopes = variances["slopes_variances"]
        self.covariances = covariances
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x, Z):
        random_effects = Z @ self.b
        x = self.fc1(x) + random_effects
        return x
