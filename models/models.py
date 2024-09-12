from typing import Dict
import torch
import torch.nn as nn


def initialize_variances(q: int) -> Dict[str, nn.Parameter]:
    variances_intercepts = nn.functional.softplus(torch.randn(q) * 0.5)
    variances_slopes = nn.functional.softplus(torch.randn(q) * 0.5)
    return {
        "variances_intercept": nn.Parameter(variances_intercepts),
        "variances_slopes": nn.Parameter(variances_slopes),
    }


def initialize_covariances(q: int) -> nn.Parameter:
    cov_intercept_slope = torch.randn(q) * 0.1  # Smaller initial values
    for i in range(q):
        cov_intercept_slope[i] = torch.abs(cov_intercept_slope[i] + 1e-5)
    return nn.Parameter(cov_intercept_slope)


class RandomEffectLayer(nn.Module):
    def __init__(self, groups: int, random_effects: int):
        super(RandomEffectLayer, self).__init__()
        # Validate that random_effects is either 0, 1, or 2
        if random_effects not in [1, 2]:
            raise ValueError(
                f"Invalid value for random_effects: {random_effects}. Expected 1 or 2."
            )
        variances = initialize_variances(groups)
        self.nr_random_effects = random_effects
        self.groups = groups
        self.b = nn.Parameter(torch.randn(random_effects * groups))
        self.variances_intercept = variances["variances_intercept"]
        self.variances_slopes = (
            variances.get("variances_slopes") if random_effects > 1 else None
        )
        self.covariances = (
            initialize_covariances(groups) if random_effects > 1 else None
        )

    def forward(self, Z: torch.tensor) -> torch.tensor:
        random_effects = Z @ self.b
        return random_effects.unsqueeze(dim=1)
