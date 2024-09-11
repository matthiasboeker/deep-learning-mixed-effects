from typing import Dict
import torch.nn as nn
import torch


def get_parameter_dict(model_parameters) -> Dict[str, nn.Parameter]:
    return dict(model_parameters)


def assemble_covariance_matrix(
    dict_parameters: Dict[str, nn.Parameter]
) -> torch.Tensor:
    block_a = torch.diag_embed(dict_parameters["variances_intercept"])
    cov_intercept_slope = (
        dict_parameters["covariances"]
        * dict_parameters["variances_slopes"]
        * dict_parameters["variances_intercept"]
    )
    block_b = torch.diag_embed(cov_intercept_slope)
    block_c = torch.diag_embed(dict_parameters["variances_slopes"])
    upper = torch.cat([block_a, block_b], dim=1)
    lower = torch.cat([block_b.transpose(0, 1), block_c], dim=1)
    covariance_matrix = torch.cat([upper, lower], dim=0)
    return covariance_matrix


def negative_log_likelihood(
    y: torch.Tensor,
    predictions: torch.Tensor,
    random_effects_design_matrix: torch.Tensor,
    model_parameters,
    regularization_terms={"re_cov_matrix_reg": 1e-4, "cov_matrix_reg": 1e-4},
):
    # Here we assume that predictions contain y - X @ beta - Z @ b
    residuals = y - predictions
    residuals = residuals.unsqueeze(1)
    dict_parameters = get_parameter_dict(model_parameters)

    if (
        torch.any(torch.isnan(dict_parameters["variances_intercept"]))
        or torch.any(torch.isinf(dict_parameters["variances_intercept"]))
        or torch.any(torch.isinf(dict_parameters["variances_slopes"]))
        or torch.any(torch.isnan(dict_parameters["variances_slopes"]))
    ):
        raise ValueError(
            "Stability issues detected in var_intercepts: NaNs or Infs found in variances."
        )

    random_effects_covariance_matrix = assemble_covariance_matrix(dict_parameters)
    regularization_strength = (
        regularization_terms["re_cov_matrix_reg"]
        * torch.max(torch.abs(random_effects_covariance_matrix)).item()
    )
    random_effects_covariance_matrix += regularization_strength * torch.eye(
        random_effects_covariance_matrix.size(0)
    )

    covariance_matrix = (
        random_effects_design_matrix
        @ random_effects_covariance_matrix
        @ random_effects_design_matrix.transpose(0, 1)
        + torch.eye(len(random_effects_design_matrix))
    )

    regularization_strength = (
        regularization_terms["cov_matrix_reg"]
        * torch.max(torch.abs(covariance_matrix)).item()
    )
    covariance_matrix += regularization_strength * torch.eye(covariance_matrix.size(0))

    try:
        covariance_matrix_inv = torch.inverse(covariance_matrix)
    except RuntimeError as e:
        print("Cholesky failed, falling back to SVD: ", e)
        U, S, Vh = torch.linalg.svd(covariance_matrix)
        S_inv = torch.diag(1.0 / S)
        covariance_matrix_inv = Vh.T @ S_inv @ U.T
    _, logdet_cov_mat = torch.slogdet(covariance_matrix_inv)

    nll = 0.5 * residuals.transpose(
        0, 1
    ) @ covariance_matrix_inv @ residuals + 0.5 * torch.abs(logdet_cov_mat)
    return nll
