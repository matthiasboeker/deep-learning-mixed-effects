from typing import Dict
import torch.nn as nn
import torch


def get_parameter_dict(model_parameters) -> Dict[str, nn.Parameter]:
    return dict(model_parameters)


def assemble_covariance_matrix(
    dict_parameters: Dict[str, nn.Parameter], K: int
) -> torch.Tensor:
    variances_intercept_param_name = list(
        filter(lambda key: "variances_intercept" in key, dict_parameters.keys())
    )[0]

    # Handle dynamic K (when K=1, we don't need slopes and covariances)
    if K > 1:
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
    else:
        covariance_matrix = torch.diag_embed(
            dict_parameters[variances_intercept_param_name]
        )

    return covariance_matrix


def negative_log_likelihood(
    y: torch.Tensor,
    predictions: torch.Tensor,
    random_effects_design_matrix: torch.Tensor,
    model_parameters,
    K: int,
    q: int,
    regularization_terms={"re_cov_matrix_reg": 1e-4, "cov_matrix_reg": 1e-4},
):
    dict_parameters = get_parameter_dict(model_parameters)
    if random_effects_design_matrix.size(1) != q * K:
        raise ValueError(
            f"random_effects_design_matrix has incompatible size {random_effects_design_matrix.size(1)}, "
            f"expected {q * K} based on q={q} and K={K}."
        )

    residuals = y.squeeze() - predictions.squeeze()
    residuals = residuals.unsqueeze(dim=1)
    variances_intercept_param_name = list(
        filter(lambda key: "variances_intercept" in key, dict_parameters.keys())
    )[0]

    # For dynamic K, check stability of intercept variances, and also slopes if K > 1
    if torch.any(
        torch.isnan(dict_parameters[variances_intercept_param_name])
    ) or torch.any(torch.isinf(dict_parameters[variances_intercept_param_name])):
        raise ValueError("Stability issues detected in intercept variances.")

    if K > 1:
        variances_slopes_param_name = list(
            filter(lambda key: "variances_slopes" in key, dict_parameters.keys())
        )[0]
        if torch.any(
            torch.isnan(dict_parameters[variances_slopes_param_name])
        ) or torch.any(torch.isinf(dict_parameters[variances_slopes_param_name])):
            raise ValueError("Stability issues detected in slope variances.")

    # Assemble the random effects covariance matrix dynamically based on K
    random_effects_covariance_matrix = assemble_covariance_matrix(dict_parameters, K)

    # Apply regularization to random effects covariance matrix
    regularization_strength = (
        regularization_terms["re_cov_matrix_reg"]
        * torch.max(torch.abs(random_effects_covariance_matrix)).item()
    )
    random_effects_covariance_matrix += regularization_strength * torch.eye(
        random_effects_covariance_matrix.size(0)
    )

    # Covariance matrix for the residuals (y - predictions)
    covariance_matrix = (
        random_effects_design_matrix
        @ random_effects_covariance_matrix
        @ random_effects_design_matrix.transpose(0, 1)
        + torch.eye(len(random_effects_design_matrix))
    )

    # Apply regularization to the covariance matrix
    regularization_strength = (
        regularization_terms["cov_matrix_reg"]
        * torch.max(torch.abs(covariance_matrix)).item()
    )
    covariance_matrix += regularization_strength * torch.eye(covariance_matrix.size(0))

    # Compute the inverse of the covariance matrix (with fallback to SVD if Cholesky fails)
    try:
        covariance_matrix_inv = torch.inverse(covariance_matrix)
    except RuntimeError as e:
        print("Cholesky failed, falling back to SVD: ", e)
        U, S, Vh = torch.linalg.svd(covariance_matrix)
        S_inv = torch.diag(1.0 / S)
        covariance_matrix_inv = Vh.T @ S_inv @ U.T

    # Log-determinant of the covariance matrix
    _, logdet_cov_mat = torch.slogdet(covariance_matrix_inv)
    if torch.isnan(logdet_cov_mat) or torch.isinf(logdet_cov_mat):
        raise ValueError("Log-determinant of covariance matrix is NaN or Inf.")

    # Compute the negative log-likelihood
    nll = 0.5 * residuals.transpose(
        0, 1
    ) @ covariance_matrix_inv @ residuals + 0.5 * torch.abs(logdet_cov_mat)

    return nll
