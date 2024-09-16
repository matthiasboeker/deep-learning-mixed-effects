import pytest
import torch
import torch.nn as nn

from models.loss_functions import (
    negative_log_likelihood,
    get_parameter_dict,
)
from models.models import assemble_covariance_matrix

# Dummy model for testing purposes
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.variances_intercept = nn.Parameter(torch.rand(3))
        self.variances_slopes = nn.Parameter(torch.rand(3))
        self.covariances = nn.Parameter(torch.rand(3))

    def forward(self, x):
        return x


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def sample_inputs():
    q = 3
    K = 2
    n = 10
    y = torch.rand(n)
    predictions = torch.rand(n)
    random_effects_design_matrix = torch.ones(n, K * q)
    return y, predictions, random_effects_design_matrix, K, q


@pytest.fixture
def valid_model_parameters(dummy_model):
    return dummy_model.named_parameters()


@pytest.fixture
def invalid_model_parameters_nan(dummy_model):
    dummy_model.variances_slopes.data[0] = float("nan")
    return dummy_model.named_parameters()


@pytest.fixture
def invalid_model_parameters_inf(dummy_model):
    dummy_model.variances_slopes.data[0] = float("inf")
    return dummy_model.named_parameters()


# Test for get_parameter_dict
def test_get_parameter_dict(dummy_model):
    params = get_parameter_dict(dummy_model.named_parameters())
    assert isinstance(params, dict)
    assert "variances_intercept" in params
    assert "variances_slopes" in params
    assert "covariances" in params


# Test for assemble_covariance_matrix with valid parameters
def test_assemble_covariance_matrix_valid(dummy_model, sample_inputs):
    y, predictions, random_effects_design_matrix, K, q = sample_inputs
    dict_parameters = get_parameter_dict(dummy_model.named_parameters())
    cov_matrix = assemble_covariance_matrix(dict_parameters)
    assert cov_matrix.size() == (6, 6)  # Check if dimensions are correct


# Test for negative_log_likelihood with valid input
def test_negative_log_likelihood_valid(valid_model_parameters, sample_inputs):
    y, predictions, random_effects_design_matrix, K, q = sample_inputs
    dict_parameters = get_parameter_dict(valid_model_parameters)
    cov_matrix = assemble_covariance_matrix(dict_parameters)
    nll = negative_log_likelihood(
        y, predictions, random_effects_design_matrix, cov_matrix, K, q
    )
    assert nll >= 0  # NLL should be positive


# Test for NaN in variances_intercept, expecting ValueError
def test_negative_log_likelihood_nan(invalid_model_parameters_nan, sample_inputs):
    y, predictions, random_effects_design_matrix, K, q = sample_inputs
    dict_parameters = get_parameter_dict(invalid_model_parameters_nan)
    cov_matrix = assemble_covariance_matrix(dict_parameters)
    with pytest.raises(
        ValueError, match="Stability issues detected in covariance matrix."
    ):
        negative_log_likelihood(
            y,
            predictions,
            random_effects_design_matrix,
            cov_matrix,
            K,
            q,
        )


# Test for Inf in slopes_variances, expecting ValueError
def test_negative_log_likelihood_inf(invalid_model_parameters_inf, sample_inputs):
    y, predictions, random_effects_design_matrix, K, q = sample_inputs
    dict_parameters = get_parameter_dict(invalid_model_parameters_inf)
    cov_matrix = assemble_covariance_matrix(dict_parameters)
    with pytest.raises(
        ValueError, match="Stability issues detected in covariance matrix."
    ):
        negative_log_likelihood(
            y,
            predictions,
            random_effects_design_matrix,
            cov_matrix,
            K,
            q,
        )


# Test the regularization part by manually checking regularization strength
def test_negative_log_likelihood_regularization_strength(dummy_model, sample_inputs):
    y, predictions, random_effects_design_matrix, K, q = sample_inputs
    regularization_terms = {"re_cov_matrix_reg": 1e-3, "cov_matrix_reg": 1e-3}
    dict_parameters = get_parameter_dict(dummy_model.named_parameters())
    cov_matrix = assemble_covariance_matrix(dict_parameters)
    nll = negative_log_likelihood(
        y,
        predictions,
        random_effects_design_matrix,
        cov_matrix,
        K,
        q,
        regularization_terms,
    )

    assert nll >= 0
