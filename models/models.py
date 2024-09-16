from typing import Dict
import torch
import torch.nn as nn


def assemble_covariance_matrix(
    dict_parameters: Dict[str, nn.Parameter]
) -> torch.Tensor:
    variances_intercept_param_name = list(
        filter(lambda key: "variances_intercept" in key, dict_parameters.keys())
    )[0]

    variances_slopes_param_name = list(
        filter(lambda key: "variances_slopes" in key, dict_parameters.keys())
    )[0]
    covariances_param_name = list(
        filter(lambda key: "covariance" in key, dict_parameters.keys())
    )[0]

    # Diagonal blocks for intercept variances and slopes variances
    block_a = torch.diag_embed(dict_parameters[variances_intercept_param_name])
    block_b = torch.diag_embed(
        dict_parameters[covariances_param_name]
        * dict_parameters[variances_intercept_param_name]
        * dict_parameters[variances_slopes_param_name]
    )
    block_c = torch.diag_embed(dict_parameters[variances_slopes_param_name])

    # Concatenate blocks to form full covariance matrix
    upper = torch.cat([block_a, block_b], dim=1)
    lower = torch.cat([block_b.transpose(0, 1), block_c], dim=1)
    covariance_matrix = torch.cat([upper, lower], dim=0)
    return covariance_matrix


def initialize_covariances(q: int) -> nn.Parameter:
    cov_intercept_slope = torch.randn(q) * 0.1
    for i in range(q):
        cov_intercept_slope[i] = torch.abs(cov_intercept_slope[i] + 1e-5)
    return nn.Parameter(cov_intercept_slope)


def initialize_variances(q: int) -> Dict[str, nn.Parameter]:
    variances_slopes = nn.functional.softplus(torch.randn(q) * 0.5)
    return nn.Parameter(variances_slopes)


def get_random_slopes_parameters(nr_groups: int):
    intercepts_variances = initialize_variances(nr_groups)
    slopes_variances = initialize_variances(nr_groups)
    covariances = initialize_covariances(nr_groups)
    return {
        "variances_intercept": intercepts_variances,
        "variances_slopes": slopes_variances,
        "covariances": covariances,
    }


def get_random_intercept_parameters(nr_groups: int):
    return {"variances_intercept": initialize_variances(nr_groups)}


class InitCovarianceMatrix:
    def __init__(self) -> None:
        self.init_commands = {
            "intercepts": {"func": get_random_intercept_parameters, "K": 1},
            "slopes": {"func": get_random_slopes_parameters, "K": 2},
        }

    def init_covariance_matrix(self, random_effect, nr_groups):
        return self.init_commands[random_effect]["func"](nr_groups)

    def get_nr_random_effect(self, random_effect):
        return self.init_commands[random_effect]["K"]


class RandomEffectLayer(nn.Module):
    def __init__(self, nr_groups: int, random_effect: str):
        super(RandomEffectLayer, self).__init__()

        init_matrix = InitCovarianceMatrix()
        self.nr_groups = nr_groups
        if self.nr_groups < 1:
            raise ValueError("Number of groups is not positive!")
        if random_effect not in ["slopes", "intercepts"]:
            raise ValueError(
                f"Random effect type {random_effect} not found! \nChoose between types \n 1.'intercepts' \n 2.'slopes'"
            )
        variances_parameters = init_matrix.init_covariance_matrix(
            random_effect, nr_groups
        )
        for key, value in variances_parameters.items():
            setattr(self, key, value)
        self.random_effect = random_effect
        self.nr_random_effects = init_matrix.get_nr_random_effect(random_effect)
        self.nr_groups = nr_groups
        self.b = nn.Parameter(torch.randn(self.nr_random_effects * nr_groups))

    def forward(self, Z: torch.tensor) -> torch.tensor:
        if Z.size()[1] != self.nr_groups * self.nr_random_effects:
            raise ValueError(
                f"Random effect design matrix does not match random effect type. \n Random effect design matrix has size {Z.size()[1]} but should be {self.nr_groups*self.nr_random_effects}."
            )
        random_effects = Z @ self.b
        return random_effects.unsqueeze(dim=1)

    def get_covariance_matrix(self):
        if self.random_effect == "intercepts":
            covariance_matrix = torch.diag_embed(self.variances_intercept)
            self._params_changed = False  # Reset the flag after update
            return covariance_matrix
        if self.random_effect == "slopes":
            covariance_matrix = assemble_covariance_matrix(self.__dict__["_parameters"])
            return covariance_matrix
