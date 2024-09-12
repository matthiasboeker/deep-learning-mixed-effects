from typing import Dict
import torch
import torch.nn as nn


def initialize_variances(q: int) -> Dict[str, nn.Parameter]:
    variances_intercepts = nn.functional.softplus(torch.randn(q) * 0.5)
    variances_slopes = nn.functional.softplus(torch.randn(q) * 0.5)
    return {
        "variances_intercepts": nn.Parameter(variances_intercepts),
        "variances_slopes": nn.Parameter(variances_slopes),
    }


def initialize_covariances(q: int) -> nn.Parameter:
    cov_intercept_slope = torch.randn(q) * 0.1  # Smaller initial values
    for i in range(q):
        cov_intercept_slope[i] = torch.abs(cov_intercept_slope[i] + 1e-5)
    return nn.Parameter(cov_intercept_slope)


class RandomEffectLayer(nn.Module):
    def __init__(self, groups: int):
        super(RandomEffectLayer, self).__init__()
        variances = initialize_variances(groups)
        covariances = initialize_covariances(groups)
        self.b = nn.Parameter(torch.randn(groups))
        self.variances_intercept = variances["variances_intercepts"]
        self.variances_slopes = variances["variances_slopes"]
        self.covariances = covariances

    def forward(self, Z: torch.tensor) -> torch.tensor:
        random_effects = Z @ self.b
        return random_effects.unsqueeze(dim=1)
