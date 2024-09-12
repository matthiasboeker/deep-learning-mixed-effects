import pytest
import torch
import torch.nn as nn
from models.models import (
    RandomEffectLayer,
    initialize_covariances,
    initialize_variances,
)


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 1)
        self.random_effects = RandomEffectLayer(2, 2)

    def forward(self, x, Z):
        x = self.fc1(x)
        x = self.fc2(x)
        return x + self.random_effects(Z)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def sample_inputs():
    q = 2
    K = 2
    p = 1
    n = 100
    y = torch.rand(n)
    random_effects_design_matrix = torch.ones((n, K * q))
    groups = torch.randint(0, q, (n,))
    random_effects_design_matrix[:, 0] = torch.where(groups == 0, 0, 1) * 0.5
    random_effects_design_matrix[:, 1] = torch.where(groups == 0, -1, 1)
    X = torch.randn(n, p)

    true_beta = torch.randn(p)
    G_truth = torch.abs(torch.diag(torch.randn(K * q)))
    b_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(K * q), covariance_matrix=G_truth
    )
    true_b = b_distribution.sample()
    y = X @ true_beta + random_effects_design_matrix @ true_b + 0.1 * torch.randn(n)
    return y, X, random_effects_design_matrix, q, p


def test_initialize_variances(sample_inputs):
    _, _, _, q, _ = sample_inputs
    variances = initialize_variances(q)
    assert isinstance(variances, dict)
    assert isinstance(variances["variances_intercept"], nn.Parameter)
    assert isinstance(variances["variances_slopes"], nn.Parameter)
    assert all(variances["variances_intercept"] >= 0)
    assert all(variances["variances_slopes"] >= 0)


def test_initialize_covariances(sample_inputs):
    _, _, _, q, _ = sample_inputs
    covariances = initialize_covariances(q)
    assert isinstance(covariances, nn.Parameter)


def test_module_integration(dummy_model):
    model_parameters = dict(dummy_model.named_parameters())
    assert isinstance(
        model_parameters["random_effects.variances_intercept"], nn.Parameter
    )
    assert isinstance(model_parameters["random_effects.variances_slopes"], nn.Parameter)
    assert isinstance(model_parameters["random_effects.covariances"], nn.Parameter)
    assert isinstance(model_parameters["random_effects.b"], nn.Parameter)


def test_run_model(sample_inputs, dummy_model):
    y, X, random_effects_design_matrix, q, p = sample_inputs
    output = dummy_model(X, random_effects_design_matrix)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (X.shape[0], 1)


@pytest.mark.parametrize("q", [1, 2, 10, 100])
def test_initialize_variances_for_different_q(q):
    variances = initialize_variances(q)
    assert variances["variances_intercept"].shape == (q,)
    assert variances["variances_slopes"].shape == (q,)


def test_random_effect_layer_with_zero_groups():
    q = 0
    K = 2
    with pytest.raises(
        ValueError, match=f"Invalid value for random_effects: {0}. Expected 1 or 2."
    ):
        layer = RandomEffectLayer(K, q)
