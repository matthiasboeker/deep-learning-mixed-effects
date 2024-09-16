import pytest
import torch
import torch.nn as nn
from models.models import (
    RandomEffectLayer,
    get_random_slopes_parameters,
    get_random_intercept_parameters,
)


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 1)
        self.random_effects = RandomEffectLayer(2, "slopes")

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


def test_get_random_slopes_parameters(sample_inputs):
    _, _, _, q, _ = sample_inputs
    variances = get_random_slopes_parameters(q)
    assert isinstance(variances, dict)
    assert isinstance(variances["variances_intercept"], nn.Parameter)
    assert isinstance(variances["variances_slopes"], nn.Parameter)
    assert isinstance(variances["covariances"], nn.Parameter)
    assert all(variances["variances_intercept"] >= 0)
    assert all(variances["variances_slopes"] >= 0)


def test_get_random_intercept_parameters(sample_inputs):
    _, _, _, q, _ = sample_inputs
    variances = get_random_intercept_parameters(q)
    assert isinstance(variances, dict)
    assert isinstance(variances["variances_intercept"], nn.Parameter)


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
    variances = get_random_slopes_parameters(q)
    assert variances["variances_intercept"].shape == (q,)
    assert variances["variances_slopes"].shape == (q,)


def test_random_effect_layer_with_zero_groups():
    q = 0
    with pytest.raises(ValueError, match=f"Number of groups is not positive!"):
        layer = RandomEffectLayer(q, "slopes")
    K = ""
    with pytest.raises(
        ValueError,
        match=f"Random effect type {K} not found! \nChoose between types \n 1.'intercepts' \n 2.'slopes'",
    ):
        layer = RandomEffectLayer(2, K)
